"""三策略的实机切换状态机；自动异常只报告，手柄 B 中断并保持目标。"""

from pathlib import Path
import time

import numpy as np

from .deployment_config import check_multi_pd
from .motion_reference import rotation_matrix
from .rl_real_common import Policy, TermGroupedHistory


class MultiTaskController:
    KEYS = {"1": "walk", "2": "crouch", "3": "rise", "4": "return_stand", "0": "stop", "r": "reset",
            "\n": "confirm_stop", "\r": "confirm_stop"}

    def __init__(self, node, cfg, root):
        self.n, self.cfg = node, cfg["multi_task"]
        required = {"handover_time", "stop_blend_time", "return_max_speed"}
        if set(self.cfg) != required or any(not np.isfinite(v) or v <= 0 for v in self.cfg.values()):
            raise ValueError("multi_task 阈值必须完整且为有限正数")
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
        # 下蹲/起身保留原任务边界；walk 的膝上限随后按自身训练 clip 与 common 解析。
        self.lo = np.maximum(node.lo, up.motion.limits[up.sim2real, 0])
        self.hi = np.minimum(node.hi, up.motion.limits[up.sim2real, 1])
        self.pose_q = {"stand": up.motion.positions[-1, up.sim2real],
                       "crouch": up.motion.positions[0, up.sim2real]}
        if cfg.get("crouch_calibration") is not None:
            self._calibrate_crouch(cfg["crouch_calibration"])
        self.mimic_limits = (self.lo, self.hi)
        self.walk_limits = (self.lo.copy(), self.hi.copy())
        # 膝止挡已拆除：仅取消 walk 额外套用的下蹲膝上限，其他机械端点保持。
        for joint in ("L4", "R4"):
            i = node.real_joint_names.index(joint)
            trained_hi = (walk.action_term_clip[walk.sim2real[i], 1]
                          if walk.action_term_clip is not None else np.inf)
            self.walk_limits[1][i] = min(node.hi[i], trained_hi)
        lo, hi = self.walk_limits
        if (np.any(lo >= hi) or np.any(walk.default_real < lo) or np.any(walk.default_real > hi)):
            raise ValueError("walk 任务边界无效或排除了默认站姿")
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
        self.pending_target = None
        self.reason = "等待首次完整关节/IMU 反馈，先保持实测姿态"
        self.end_announced = False
        self.key_escape = False
        if not getattr(node, "preflight_only", False):
            print("[注意] 手柄 B 中断并保持最后目标（不会切断电机）；4 须扶稳/吊起。")
            self.show_status()

    def _calibrate_crouch(self, calibration):
        """实测蹲姿用于任务边界和初始姿态分类；不改网络观测或参考动作。"""
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
        rotation_matrix(calibration["body_quat_wxyz"])
        lo, hi = self.lo.copy(), self.hi.copy()
        for name, side in calibration["stop_sides"].items():
            i = names.index(name)
            (lo if side == "lower" else hi)[i] = q[i]
        if (np.any(lo >= hi) or np.any(q < lo) or np.any(q > hi)
                or np.any(self.pose_q["stand"] < lo) or np.any(self.pose_q["stand"] > hi)):
            raise ValueError("实测任务边界无效，或不能同时包含蹲姿与站姿")
        self.lo, self.hi = lo, hi
        self.pose_q["crouch"] = q
        print("[multi] 使用实测蹲姿和 10 个单侧任务目标边界；ankle roll 不设实测限位；"
              "common、训练 clip、参考动作与真实观测不变。")

    def show_status(self):
        crouch_key = "；2/LB+X 下蹲" if "crouch" in self.policies else ""
        label, actions = {
            "wait_current": ("等待关节/IMU", "收到后保持当前姿态"),
            "stand_ready": ("站立就绪", f"1/LB+A 走路{crouch_key}；4/LB+Start 慢回站姿"),
            "crouch_ready": ("下蹲就绪", "3/LB+Y 起身；4/LB+Start 慢回站姿"),
            "walking": ("行走中", "0/Start 停步；W/S 前后 A/D 左右 Q/E 转向（或摇杆）；空格零速"),
            "stopping": ("停步待确认", "速度归零、双脚落地后 Enter/再次 Start 收脚"),
            "standing_transition": ("慢回站姿中", "请等待，不能切换"),
            "lowering": ("下蹲中", "请等待，不能切换"),
            "rising": ("起身中", "请等待，不能切换"),
            "stopped": ("手柄中断", "R/Back 解锁；4/LB+Start 慢回站姿"),
        }[self.state]
        print(f"[操作提示] {label} | {actions} | 手柄 B 中断", flush=True)

    def change(self, state, reason):
        self.state = self.n.mode = state
        self.reason, self.state_since = reason, time.monotonic()
        if state == "stopped":
            self.n.get_logger().warn(f"[停止] {reason}")
        elif state == "standing_transition":
            print(f"[multi] 慢回站姿，预计 {self.stand_blend_time:.1f}s")
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
        self.change("stopped", reason)

    def _policy_limits(self, policy):
        return self.walk_limits if policy is self.policies["walk"] else self.mimic_limits

    def closest_pose(self):
        q = self.n.obs_raw[7:19]
        return min(self.pose_q, key=lambda pose: np.linalg.norm(q - self.pose_q[pose]))

    def return_error(self):
        if not self.has_target or self.state not in ("stand_ready", "crouch_ready", "stopped"):
            return "慢回站姿只允许保持状态；执行动作中无效，不排队"
        return None

    def start_stand_blend(self, duration, reason):
        self.blend_start = np.clip(self.n.target_pub, self.lo, self.hi).astype(np.float32)
        self.stand_blend_time = duration
        self.active = self.pending_target = None
        self.n._clear_cmd_sources()
        self.n._close_log()
        self.change("standing_transition", reason)

    def request(self, task):
        n, now = self.n, time.monotonic()
        if task in ("walk", "crouch", "rise") and task not in self.policies:
            return self.reject("下蹲任务已禁用，等待新模型")
        if task == "halt":
            self.halt("操作员中断")
            return True
        if task == "return_stand":
            if (reason := self.return_error()) is not None:
                return self.reject(reason)
            distance = np.max(np.abs(np.clip(n.target_pub, self.lo, self.hi) - self.pose_q["stand"]))
            # smoothstep 的峰值导数是 1.5；限制的是目标速度，不是电机实际速度。
            duration = max(self.cfg["stop_blend_time"], 1.5 * distance / self.cfg["return_max_speed"])
            self.start_stand_blend(duration, f"操作者确认已支撑；{duration:.1f}s 同步慢回准备站姿，不运行策略")
            return True
        if task == "reset":
            if self.state != "stopped":
                return self.reject("仅停止锁存后可重新验收；R 不自动回站立")
            self.change(self.closest_pose() + "_ready" if self.has_target else "wait_current",
                        "人工解除中断；仍需手动选择任务")
            return True
        if task == "stop":
            if self.state == "walking":
                command = n.cmd.copy()
                n._clear_cmd_sources()
                n.cmd[:] = command
                self.change("stopping", "速度指令归零；看到双脚落地后 Enter/再次 Start 确认收脚")
                return True
            return self.reject("正常停步只用于走路；收脚确认用 Enter，其他动作可按手柄 B 中断")
        if task == "confirm_stop":
            if self.state == "stopping":
                self.start_stand_blend(self.cfg["stop_blend_time"], "操作者确认接地，0.5s 平滑收回站姿")
                return True
            return self.reject("尚未进入正常停步，不接受收脚确认")
        required = {"walk": "stand_ready", "crouch": "stand_ready", "rise": "crouch_ready"}
        if task not in required or self.state != required[task] or self.active == task:
            label = {"walk": "走路", "crouch": "下蹲", "rise": "起身"}.get(task, task)
            return self.reject(f"{label}未启动：当前任务状态不允许")
        p = self.policies[task]
        lo, hi = self._policy_limits(p)
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
        except Exception as exc:
            n.cmd[:] = previous_command
            return self.reject(f"策略接管预检失败: {exc}")
        n._close_log()
        n._clear_cmd_sources()
        # 成功接管才换边界；中断保持、停步及慢回站姿继续沿用它，避免膝目标跳回旧上限。
        self.lo, self.hi = lo, hi
        self.active, self.pending_target = task, target
        # 初始保持可能来自边界外少量测量偏差；策略接管起点也必须严格落在指令边界内。
        self.handover_from = np.clip(n.target_pub, self.lo, self.hi).astype(np.float32)
        self.activated_at = now
        self.end_announced = False
        p.run_t0 = now
        n._open_log(p)
        self.change({"walk": "walking", "crouch": "lowering", "rise": "rising"}[task],
                    "0.2s 平滑接管，独立历史/时钟从头开始")
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
        lo, hi = self._policy_limits(p)
        return np.clip(target, lo, hi).astype(np.float32), terms

    def update_state(self):
        now = time.monotonic()
        if self.state in ("stand_ready", "crouch_ready"):
            pose = ({"crouch": "crouch", "rise": "stand"}[self.active]
                    if self.active in ("crouch", "rise") else self.closest_pose())
            if self.state != pose + "_ready":
                self.change(pose + "_ready", "按当前关节角选择就近任务入口")
        elif self.state in ("lowering", "rising"):
            p = self.policies[self.active]
            end = int(np.ceil((len(p.motion.positions) - 1) / p.motion.stride))
            if p.motion.steps >= end:
                pose = "crouch" if self.state == "lowering" else "stand"
                self.change(pose + "_ready", "参考结束，继续末帧策略保持")
        elif self.state == "standing_transition":
            elapsed = now - self.state_since
            if elapsed >= self.stand_blend_time:
                self.change("stand_ready", "插值结束，不自动启动走路")

    def tick(self):
        n, now = self.n, time.monotonic()
        if self.state == "wait_current":
            if not n._capture_current_target():
                return
            self.has_target = True
            n._clear_cmd_sources()
            self.change(self.closest_pose() + "_ready", "保持实测角度，按关节角选择就近任务入口")
        if not self.has_target:
            return
        if self.state == "stopped":
            n._publish_target()
            return
        self.update_state()
        if self.active:
            p = self.policies[self.active]
            joy = n._joy_cmd if now - n._last_joy_rx < n.ctrl_timeout else np.zeros(3)
            desired = np.clip(n._kb_cmd + joy, n.cmd_min, n.cmd_max) if self.state == "walking" else np.zeros(3)
            n.cmd[:] = desired
            p.cmd[:] = n.cmd
            if n.tick % n.decimation == 0:
                try:
                    advancing = now - self.activated_at >= self.cfg["handover_time"] - 1e-9
                    if self.pending_target is not None:
                        target, self.pending_target = self.pending_target, None
                        terms = p._build_terms()
                    else:
                        target, terms = self.policy_target(p)
                    a = np.clip((now - self.activated_at) / self.cfg["handover_time"], 0., 1.)
                    s = a*a*(3-2*a)
                    n.target_real = target
                    n.target_pub = ((1-s)*self.handover_from + s*target).astype(np.float32)
                    n._log_row(terms, p)
                    if p.motion and p.motion.frame == len(p.motion.positions) - 1 and not self.end_announced:
                        print("[multi] 参考结束，继续策略保持。")
                        self.end_announced = True
                    if advancing:
                        p.policy_step += 1
                        if p.motion:
                            p.motion.steps += 1
                except Exception as exc:
                    n.get_logger().warn(f"[策略异常，保持上次目标并重试] {exc}", throttle_duration_sec=1.0)
        elif self.state == "standing_transition":
            a = np.clip((now - self.state_since) / self.stand_blend_time, 0., 1.)
            s = a*a*(3-2*a)
            n.target_pub = np.clip((1-s)*self.blend_start + s*self.pose_q["stand"], self.lo, self.hi).astype(np.float32)
            n.target_real = n.target_pub.copy()
        n._publish_target()

    def keys(self, chars):
        chars = chars.lower()
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
