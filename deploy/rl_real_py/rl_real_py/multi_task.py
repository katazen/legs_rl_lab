"""三策略的实机切换状态机；只通过既有节点发布，不创建 ROS 节点或驱动。

没有足底接触/机身线速度传感器：任务键代表操作者确认落地，停步后 Enter/再次 Start 确认。
这不是接触估计器；不能把关节静止当成已承重，也不能据此自动断电。
"""

from pathlib import Path
import time

import numpy as np

from .deployment_config import check_multi_pd
from .motion_reference import rotation_matrix
from .rl_real_common import Policy, TermGroupedHistory


class MultiTaskController:
    KEYS = {"1": "walk", "2": "crouch", "3": "rise", "4": "return_stand", "0": "stop", "p": "halt", "r": "reset",
            "\n": "confirm_stop", "\r": "confirm_stop"}

    def __init__(self, node, cfg, root):
        self.n, self.cfg = node, cfg["multi_task"]
        required = {"stable_time", "joint_tolerance", "ankle_roll_tolerance", "joint_speed", "tilt_error",
                    "angular_speed", "max_tilt", "joint_limit_tolerance", "max_policy_gap", "handover_time",
                    "handover_max_delta", "stop_blend_time", "stop_timeout", "end_timeout", "return_max_speed"}
        if set(self.cfg) != required or any(not np.isfinite(v) or v <= 0 for v in self.cfg.values()):
            raise ValueError("multi_task 阈值必须完整且为有限正数")
        if self.cfg["max_policy_gap"] <= node.step_dt:
            raise ValueError("max_policy_gap 必须大于策略周期")
        if cfg["target_ema_alpha"] != 1.0 or np.any(node.cmd_bias):
            raise ValueError("统一切换与 sim2sim 一致：关闭额外 EMA 和速度偏置")
        self.buttons = cfg["gamepad_buttons"]
        if (set(self.buttons) != {"a", "b", "x", "y", "lb", "back", "start"}
                or any(type(i) is not int or i < 0 for i in self.buttons.values())
                or len(set(self.buttons.values())) != len(self.buttons)):
            raise ValueError("gamepad_buttons 必须配置七个不同的非负整数索引")
        check_multi_pd(cfg)
        self.policies = {"walk": node}
        for name in ("crouch", "rise"):
            if name == "crouch" and cfg["tasks"][name] is None:
                print("[multi] 下蹲已禁用：2 / LB+X 不可用；仅启用走路和起身。")
                continue
            self.policies[name] = Policy(cfg, Path(cfg["tasks"][name]), root)
        walk, up = self.policies["walk"], self.policies["rise"]
        down = self.policies.get("crouch")
        if walk.motion is not None or set(walk.deploy["commands"]) != {"base_velocity"}:
            raise ValueError("walk 必须是速度控制策略")
        if up.motion is None or (down is not None and down.motion is None):
            raise ValueError("crouch / rise 必须是参考动作跟踪策略")
        for p in self.policies.values():
            if not np.array_equal(p.default_real, walk.default_real):
                raise ValueError("启用任务的默认站姿必须相同")
            if any(v["history_length"] != p.num_history for v in p.deploy["observations"].values()):
                raise ValueError("当前部署要求各观测项历史长度相同")
        if down is not None:
            for fa, fb in ((0, -1), (-1, 0)):
                if (not np.allclose(down.motion.positions[fa, down.sim2real], up.motion.positions[fb, up.sim2real], atol=1e-5)
                        or not np.allclose(down.motion.rotations[fa], up.motion.rotations[fb], atol=1e-5)):
                    raise ValueError("下蹲、起身参考端点不衔接")
            if not np.allclose(down.motion.limits[down.sim2real], up.motion.limits[up.sim2real], atol=1e-6):
                raise ValueError("下蹲和起身任务限位不一致")
        if not np.allclose(up.motion.positions[-1, up.sim2real], walk.default_real, atol=1e-5):
            raise ValueError("起身末帧必须与走路默认站姿一致")
        # 同 sim2sim 共用 limit 机器人；不改 common，不读取或比较 XML 的限位。
        self.lo = np.maximum(node.lo, up.motion.limits[up.sim2real, 0])
        self.hi = np.minimum(node.hi, up.motion.limits[up.sim2real, 1])
        self.pose_q = {"stand": up.motion.positions[-1, up.sim2real],
                       "crouch": up.motion.positions[0, up.sim2real]}
        self.pose_rot = {"stand": up.motion.rotations[-1], "crouch": up.motion.rotations[0]}
        if cfg.get("crouch_calibration") is not None:
            self._calibrate_crouch(cfg["crouch_calibration"])
        for name, p in self.policies.items():
            p.obs_raw[:] = 0.
            p.obs_raw[3:7] = p.motion.quaternions[0] if p.motion else [1., 0., 0., 0.]
            p.obs_raw[7:19] = p.motion.positions[0, p.sim2real] if p.motion else p.default_real
            p.policy_step = 0
            if p.motion:
                p.motion.reset(p.obs_raw[3:7])
            p._check_policy()
            p.obs_raw[:] = 0.
            print(f"[multi {name}] {p.run_dir}")
        self.active = None
        self.has_target = False
        self.state = node.mode = "wait_current"
        self.state_since = time.monotonic()
        self.stable_since = self.stable_pose = None
        self.return_still_since = None
        self.last_policy = self.clock0 = self.pending_target = None
        self.reason = "等待完整关节/IMU，先保持实测姿态，不自动回站立"
        self.end_announced = False
        self.key_escape = False
        if not getattr(node, "preflight_only", False):
            self.show_status()

    def _calibrate_crouch(self, calibration):
        """实测蹲姿只用于验收和任务边界；不改网络观测、训练 clip 或参考动作。"""
        names = self.n.real_joint_names
        if (not isinstance(calibration, dict)
                or set(calibration) != {"joint_pos", "body_quat_wxyz", "stop_sides"}
                or not isinstance(calibration["joint_pos"], dict)
                or set(calibration["joint_pos"]) != set(names)
                or not isinstance(calibration["stop_sides"], dict)
                or set(calibration["stop_sides"]) != set(names) - {"L6", "R6"}
                or any(side not in ("lower", "upper") for side in calibration["stop_sides"].values())):
            raise ValueError("crouch_calibration 需要完整 12 关节姿态、机身四元数和除 ankle roll 外的 10 个限位方向")
        q = np.asarray([calibration["joint_pos"][name] for name in names], dtype=np.float32)
        if (q.shape != (12,) or not np.isfinite(q).all()
                or np.any(q < self.n.lo) or np.any(q > self.n.hi)):
            raise ValueError("实测蹲姿必须为有限角度且位于 common 硬件限位内")
        rot = rotation_matrix(calibration["body_quat_wxyz"])
        if rot[2, 2] < np.cos(self.cfg["max_tilt"]):
            raise ValueError("实测蹲姿倾角超过 max_tilt")
        lo, hi = self.lo.copy(), self.hi.copy()
        for name, side in calibration["stop_sides"].items():
            i = names.index(name)
            (lo if side == "lower" else hi)[i] = q[i]
        if (np.any(lo >= hi) or np.any(q < lo) or np.any(q > hi)
                or np.any(self.pose_q["stand"] < lo) or np.any(self.pose_q["stand"] > hi)):
            raise ValueError("实测任务边界无效，或不能同时包含蹲姿与站姿")
        self.lo, self.hi = lo, hi
        self.pose_q["crouch"], self.pose_rot["crouch"] = q, rot
        print("[multi] 使用实测蹲姿验收和 10 个单侧任务限位；ankle roll 不设实测限位；"
              "common、训练 clip、参考动作与真实观测不变。")

    def show_status(self):
        label, actions = {
            "wait_current": ("等待关节/IMU 反馈", "反馈有效后自动保持当前姿态 → 等待使能反馈；不自动回站立。"),
            "wait_feedback": ("保持当前姿态，等待使能反馈", "反馈连续恢复后自动 → 姿态验收；暂不能启动任务。"),
            "checking": ("姿态验收中", "站姿/蹲姿连续达标后自动 → 对应就绪状态；不自动调整姿态，暂不能启动策略。稳定保持后 4 / LB+Start → 慢回准备站姿。"),
            "stand_ready": ("站立就绪", "1 / LB+A → 走路；2 / LB+X → 下蹲；稳定保持后 4 / LB+Start → 慢回准备站姿。"),
            "crouch_ready": ("下蹲就绪", "3 / LB+Y → 起身；起身完成后才能走路。稳定保持后 4 / LB+Start → 慢回准备站姿。"),
            "walking": ("行走中", "0 / Start → 停步等待确认；W/S 前后、A/D 左右、Q/E 转向，或手柄摇杆调速；空格清零速度，但不退出行走。"),
            "stopping": ("停步中，等待人工确认接地", "速度归零且亲眼确认双脚落地后：Enter / 再次按 Start → 平滑收脚；不是再次按 0。"),
            "standing_transition": ("插值回准备站姿中", "插值完成且实测站姿稳定后自动 → 站立就绪；暂不能切换任务，4 无效。"),
            "lowering": ("下蹲中", "动作结束且实测姿态到位后自动 → 下蹲就绪，继续策略保持；暂不能切换任务。"),
            "rising": ("起身中", "动作结束且实测姿态到位后自动 → 站立就绪，继续策略保持；暂不能切换任务。"),
            "stopped": ("中断/故障锁存", "排除故障后：R / Back → 重新验收站姿/蹲姿；稳定保持后 4 / LB+Start → 慢回准备站姿。不会自动回站立或续播。"),
        }[self.state]
        if "crouch" not in self.policies:
            actions = actions.replace("2 / LB+X → 下蹲", "2 / LB+X → 下蹲已禁用（待新模型）")
        print(f"\n[操作提示] 当前状态：{label} ({self.state})\n"
              f"  下一步：{actions}\n"
              "  P / B → 中断锁存（不是断电急停）。策略键表示确认落地站稳；4 表示已扶稳/吊起，插值不负责平衡。", flush=True)

    def change(self, state, reason):
        previous = self.state
        self.state = self.n.mode = state
        self.reason, self.state_since = reason, time.monotonic()
        self.stable_since = self.stable_pose = None
        self.return_still_since = None
        print(f"[multi] {previous} -> {state}: {reason}")
        self.show_status()

    def reject(self, reason):
        self.n.get_logger().warn(f"[multi 拒绝] {reason}")
        self.show_status()
        return False

    def halt(self, reason):
        self.n.target_real = self.n.target_pub.copy()
        self.active = self.pending_target = None
        self.n._clear_cmd_sources()
        self.n._close_log()
        self.change("stopped", reason + "；保持最后下发目标，恢复后 R 重新验收或稳定后 4 手动回站姿，不自动续播")

    def state_error(self, task_bounds=False):
        n = self.n
        if not n._fresh():
            return "关节或 IMU 缺失/超时"
        if not np.isfinite(n.obs_raw).all():
            return "反馈含 NaN/Inf"
        if n._motor_faults:
            return f"电机故障尚未清除: {n._motor_faults}"
        try:
            rot = rotation_matrix(n.obs_raw[3:7])
        except ValueError as exc:
            return str(exc)
        q = n.obs_raw[7:19]
        if np.any(q < n.lo) or np.any(q > n.hi):
            return "反馈超出 common 硬件限位"
        if task_bounds:
            margin = self.cfg["joint_limit_tolerance"]
            if np.any(q < self.lo - margin) or np.any(q > self.hi + margin):
                return "反馈超出任务限位与容差"
        if rot[2, 2] < np.cos(self.cfg["max_tilt"]):
            return "机身倾角过大"
        return None

    def pose_error(self, pose, stopping=False):
        if (reason := self.state_error(task_bounds=True)) is not None:
            return reason
        n = self.n
        tolerance = np.full(12, .45 if stopping else self.cfg["joint_tolerance"])
        for name in ("L6", "R6"):
            tolerance[n.real_joint_names.index(name)] = self.cfg["ankle_roll_tolerance"]
        error = np.abs(n.obs_raw[7:19] - self.pose_q[pose])
        if np.any(error > tolerance):
            i = int(np.argmax(error / tolerance))
            return f"{n.real_joint_names[i]} 姿态误差 {error[i]:.3f}rad"
        rot = rotation_matrix(n.obs_raw[3:7])
        angle = np.arccos(np.clip(np.dot(rot[2], self.pose_rot[pose][2]), -1., 1.))
        if angle > (.20 if stopping else self.cfg["tilt_error"]):
            return "机身倾斜与参考姿态不符"
        if np.max(np.abs(n.obs_raw[19:31])) > (5.0 if stopping else self.cfg["joint_speed"]):
            return "关节尚未稳定"
        if np.linalg.norm(n.obs_raw[:3]) > (.8 if stopping else self.cfg["angular_speed"]):
            return "机身角速度过大"
        return None

    def stable(self, pose):
        if (reason := self.pose_error(pose)) is not None:
            self.reason = reason
            self.stable_since = self.stable_pose = None
            return False
        if self.stable_pose != pose:
            self.stable_pose, self.stable_since = pose, time.monotonic()
        return time.monotonic() - self.stable_since >= self.cfg["stable_time"]

    def return_error(self):
        if not self.has_target or self.state not in ("checking", "stand_ready", "crouch_ready", "stopped"):
            return "慢回站姿只允许稳定保持状态；执行动作中无效，不排队"
        if (reason := self.state_error(task_bounds=True)) is not None:
            return reason
        if np.max(np.abs(self.n.obs_raw[19:31])) > self.cfg["joint_speed"]:
            return "关节尚未稳定"
        if np.linalg.norm(self.n.obs_raw[:3]) > self.cfg["angular_speed"]:
            return "机身角速度过大"
        held = self.n.target_pub
        if held.shape != (12,) or not np.isfinite(held).all():
            return "当前保持目标无效"
        if np.any(np.maximum(self.lo - held, held - self.hi) > self.cfg["joint_limit_tolerance"]):
            return "当前保持目标超出任务限位容差"
        return None

    def start_stand_blend(self, duration, reason):
        self.blend_start = np.clip(self.n.target_pub, self.lo, self.hi).astype(np.float32)
        self.stand_blend_time = duration
        self.last_policy = time.monotonic()
        self.active = self.pending_target = None
        self.n._clear_cmd_sources()
        self.n._close_log()
        self.change("standing_transition", reason)

    def request(self, task):
        n, now = self.n, time.monotonic()
        if task in ("walk", "crouch", "rise") and task not in self.policies:
            return self.reject("下蹲任务已禁用，待匹配新起身姿态的下蹲模型训练完成后再启用")
        if task == "halt":
            self.halt("操作员中断")
            return True
        if task == "return_stand":
            if (reason := self.return_error()) is not None:
                self.return_still_since = None
                return self.reject(reason)
            if self.return_still_since is None or now - self.return_still_since < self.cfg["stable_time"]:
                return self.reject(f"需连续稳定 {self.cfg['stable_time']:.2f}s 后重新按 4；不会自动执行")
            distance = np.max(np.abs(np.clip(n.target_pub, self.lo, self.hi) - self.pose_q["stand"]))
            # smoothstep 的峰值导数是 1.5；限制的是目标速度，不是电机实际速度。
            duration = max(self.cfg["stop_blend_time"], 1.5 * distance / self.cfg["return_max_speed"])
            self.start_stand_blend(duration, f"操作者确认已支撑；{duration:.1f}s 同步慢回准备站姿，不运行策略")
            return True
        if task == "reset":
            if self.state != "stopped":
                return self.reject("仅停止锁存后可重新验收；R 不自动回站立")
            if all(self.pose_error(p) is not None for p in self.pose_q):
                return self.reject("当前反馈不满足站姿或蹲姿，需人工恢复；不会强行拉回")
            self.change("checking" if self.has_target else "wait_current", "重新连续验收，仍需手动选择任务")
            return True
        if task == "stop":
            if self.state == "walking":
                command = n.cmd.copy()
                n._clear_cmd_sources()
                n.cmd[:] = command
                self.change("stopping", "缓降速度；看到双脚落地后 Enter/再次 Start 确认收脚，不会自动接管")
                return True
            return self.reject("正常停步只用于走路；收脚确认用 Enter，中断其他动作请按 P/B")
        if task == "confirm_stop":
            if self.state == "stopping":
                reason = self.pose_error("stand", stopping=True)
                if reason or np.linalg.norm(n.cmd) >= 1e-6:
                    return self.reject(reason or "速度命令尚未归零，请稍后再次确认")
                self.start_stand_blend(self.cfg["stop_blend_time"], "操作者确认接地，0.5s 平滑收回站姿")
                return True
            return self.reject("尚未进入正常停步，不接受收脚确认")
        required = {"walk": "stand_ready", "crouch": "stand_ready", "rise": "crouch_ready"}
        if task not in required or self.state != required[task] or self.active == task:
            return self.reject(f"{task} 不能从 {self.state} 启动；不排队、不重播")
        if (reason := self.pose_error("crouch" if task == "rise" else "stand")) is not None:
            return self.reject(reason)
        hold_violation = np.maximum(self.lo - n.target_pub, n.target_pub - self.hi)
        if np.any(hold_violation > self.cfg["joint_limit_tolerance"]):
            return self.reject("当前保持目标超出任务限位容差，需先安全恢复并重新接管")
        p = self.policies[task]
        p.policy_step = 0
        p.last_action[:] = 0.
        p.hist = TermGroupedHistory(p.term_dims, p.num_history)
        if p.motion:
            p.motion.reset(n.obs_raw[3:7])
        # 请求的速度先置零；只有接管成功才清空操作端指令。
        previous_command = n.cmd.copy()
        p.cmd[:] = 0.
        try:
            target, _ = self.policy_target(p)
            if np.max(np.abs(target - n.target_pub)) > self.cfg["handover_max_delta"]:
                raise ValueError("新策略首拍目标跳变过大")
        except Exception as exc:
            n.cmd[:] = previous_command
            return self.reject(f"策略接管预检失败: {exc}")
        n._close_log()
        n._clear_cmd_sources()
        self.active, self.pending_target = task, target
        # 初始保持可能来自边界外少量测量偏差；策略接管起点也必须严格落在指令边界内。
        self.handover_from = np.clip(n.target_pub, self.lo, self.hi).astype(np.float32)
        self.activated_at = self.last_policy = now
        self.clock0 = None
        self.end_announced = False
        p.run_t0 = now
        n._open_log(p)
        self.change({"walk": "walking", "crouch": "lowering", "rise": "rising"}[task],
                    "人工确认落地；0.2s 平滑接管，独立历史/时钟从头开始")
        return True

    def policy_target(self, p):
        p.obs_raw[:] = self.n.obs_raw
        terms = p._build_terms()
        x = p.hist.update(terms)
        if x.shape != (p.num_obs * p.num_history,) or not np.isfinite(x).all():
            raise ValueError("策略观测维度错误或含 NaN/Inf")
        start = time.monotonic()
        target = p._target_from_action(p._infer(x))
        self.policy_ms = (time.monotonic() - start) * 1000
        if time.monotonic() - start > self.cfg["max_policy_gap"] or not self.n._fresh():
            raise ValueError("推理超时或推理后反馈过期")
        return np.clip(target, self.lo, self.hi).astype(np.float32), terms

    def update_state(self):
        now = time.monotonic()
        if self.state in ("checking", "stand_ready", "crouch_ready"):
            poses = [{"crouch": "crouch", "rise": "stand"}[self.active]] if self.active in ("crouch", "rise") else self.pose_q
            pose = next((p for p in poses if self.pose_error(p) is None), None)
            if pose is None:
                if self.state != "checking":
                    self.change("checking", "姿态不再满足就绪条件")
                self.stable_since = self.stable_pose = None
            elif self.state != pose + "_ready" and self.stable(pose):
                self.change(pose + "_ready", "姿态连续达标；操作者确认落地后按任务键")
        elif self.state in ("lowering", "rising"):
            p = self.policies[self.active]
            end = int(np.ceil((len(p.motion.positions) - 1) / p.motion.stride))
            if p.motion.steps >= end:
                pose = "crouch" if self.state == "lowering" else "stand"
                if self.stable(pose):
                    self.change(pose + "_ready", "参考结束且实测姿态到位，继续末帧策略保持")
                elif (p.motion.steps - end) * p.step_dt > self.cfg["end_timeout"]:
                    self.halt("参考结束后仍未到位: " + self.reason)
        elif self.state == "stopping" and now - self.state_since > self.cfg["stop_timeout"]:
            self.halt("停步超时，未收到有效的接地/收脚确认")
        elif self.state == "standing_transition":
            elapsed = now - self.state_since
            if elapsed >= self.stand_blend_time and self.stable("stand"):
                self.change("stand_ready", "插值结束且实测站姿稳定，不自动启动走路")
            elif elapsed > self.stand_blend_time + self.cfg["end_timeout"]:
                self.halt("插值后站姿未稳定: " + self.reason)

    def tick(self):
        n, now = self.n, time.monotonic()
        if self.return_error() is not None:
            self.return_still_since = None
        elif self.return_still_since is None:
            self.return_still_since = now
        if self.state == "wait_current":
            if (reason := self.state_error()) is not None:
                n.get_logger().warn(f"[multi 等待] {reason}", throttle_duration_sec=1.0)
                return
            if not n._capture_current_target():
                return
            self.has_target, self.hold_sent_at = True, now
            n._clear_cmd_sources()
            self.change("wait_feedback", "原样保持实测角度，等待首次使能后的反馈恢复")
        if not self.has_target:
            return
        if self.state == "wait_feedback":
            reason = self.state_error()
            if not n._fresh():
                self.stable_since = None
            elif reason:
                self.halt(reason)
            elif n._last_joint_rx > self.hold_sent_at:
                if self.stable_since is None:
                    self.stable_since = now
                elif now - self.stable_since >= self.cfg["stable_time"]:
                    self.change("checking", "反馈连续恢复，识别站姿/蹲姿；不自动插值")
            n._publish_target()
            return
        if self.state == "stopped":
            n._publish_target()
            return
        if (reason := self.state_error(task_bounds=self.active is not None or self.state == "standing_transition")) is not None:
            self.halt(reason)
            n._publish_target()
            return
        if (self.active or self.state == "standing_transition") and now - self.last_policy > self.cfg["max_policy_gap"]:
            self.halt("策略/插值调度间隔超时")
        self.update_state()
        if self.active:
            p = self.policies[self.active]
            joy = n._joy_cmd if now - n._last_joy_rx < n.ctrl_timeout else np.zeros(3)
            desired = np.clip(n._kb_cmd + joy, n.cmd_min, n.cmd_max) if self.state == "walking" else np.zeros(3)
            n.cmd += np.clip(desired - n.cmd, -n.cmd_accel_limit * n.pub_dt, n.cmd_accel_limit * n.pub_dt)
            p.cmd[:] = n.cmd
            if n.tick % n.decimation == 0:
                try:
                    advancing = now - self.activated_at >= self.cfg["handover_time"] - 1e-9
                    if advancing and self.clock0 is None:
                        self.clock0 = now
                    if advancing and now - self.clock0 - p.policy_step * p.step_dt > self.cfg["max_policy_gap"]:
                        raise ValueError("策略时钟落后，不追赶参考")
                    if self.pending_target is not None:
                        target, self.pending_target = self.pending_target, None
                        terms = p._build_terms()
                    else:
                        target, terms = self.policy_target(p)
                    a = np.clip((now - self.activated_at) / self.cfg["handover_time"], 0., 1.)
                    s = a*a*(3-2*a)
                    n.target_real = target
                    n.target_pub = ((1-s)*self.handover_from + s*target).astype(np.float32)
                    self.last_policy = now
                    n._log_row(terms, p)
                    if p.motion and p.motion.frame == len(p.motion.positions) - 1 and not self.end_announced:
                        print("[multi] 参考末帧：继续策略保持，实测到位后才就绪；不自动断电。")
                        self.end_announced = True
                    if advancing:
                        p.policy_step += 1
                        if p.motion:
                            p.motion.steps += 1
                except Exception as exc:
                    self.halt(f"策略异常: {exc}")
        elif self.state == "standing_transition":
            a = np.clip((now - self.state_since) / self.stand_blend_time, 0., 1.)
            s = a*a*(3-2*a)
            n.target_pub = np.clip((1-s)*self.blend_start + s*self.pose_q["stand"], self.lo, self.hi).astype(np.float32)
            n.target_real = n.target_pub.copy()
            self.last_policy = now
        n._publish_target()

    def keys(self, chars):
        chars = chars.lower()
        if "p" in chars:
            self.request("halt")
            return
        speed_changed = False
        for ch in chars:
            if ch == "\x1b":
                self.key_escape = True
                continue
            if self.key_escape:
                if (ch.isalpha() and ch not in "o") or ch == "~":
                    self.key_escape = False
                continue
            if ch in self.KEYS:
                self.request(self.KEYS[ch])
            elif ch == " ":
                self.n._clear_cmd_sources()
                speed_changed = True
            elif ch in "wsadqe" and self.state == "walking":
                index = "wsadqe".index(ch)
                axis, sign = index // 2, (1 if index % 2 == 0 else -1)
                self.n._kb_cmd[axis] = np.clip(self.n._kb_cmd[axis] + sign*self.n.kb_step,
                                               self.n.cmd_min[axis], self.n.cmd_max[axis])
                speed_changed = True
        if speed_changed:
            vx, vy, wz = self.n._kb_cmd
            print(f"[键盘速度设置] 前后={vx:.2f}m/s，左右={vy:.2f}m/s，转向={wz:.2f}rad/s；未含手柄输入。")
            self.show_status()

    def joy(self, msg):
        n = self.n
        def down(name, buttons):
            i = self.buttons[name]
            return i < len(buttons) and buttons[i] == 1
        def pressed(name):
            return down(name, msg.buttons) and not down(name, n._prev_buttons)
        if pressed("b"):
            self.request("halt")
        elif pressed("back"):
            self.request("reset")
        elif pressed("start"):
            self.request("return_stand" if down("lb", msg.buttons)
                         else "confirm_stop" if self.state == "stopping" else "stop")
        elif down("lb", msg.buttons):
            for button, task in (("a", "walk"), ("x", "crouch"), ("y", "rise")):
                if pressed(button):
                    self.request(task)
                    break
        n._prev_buttons = list(msg.buttons)
        if self.state != "walking":
            n._joy_cmd[:] = 0.
