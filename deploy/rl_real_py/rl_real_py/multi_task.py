"""实机策略切换状态机；自动异常只报告，手柄 B 中断并保持目标。"""

from pathlib import Path
import time

import numpy as np

from .deployment_config import check_multi_pd
from .rl_real_common import Policy, TermGroupedHistory


class MultiTaskController:
    KEYS = {"1": "walk", "2": "crouch", "3": "rise", "4": "return_stand", "0": "stop", "r": "reset"}
    joy_move_mode = "stick"
    joy_wait_neutral = False

    @staticmethod
    def gamepad_buttons(configured):
        buttons = {"rb": 10, **configured}
        required = {"a", "b", "x", "y", "lb", "rb", "back", "start"}
        if (set(buttons) != required
                or any(type(i) is not int or not 0 <= i <= 20 for i in buttons.values())
                or len(set(buttons.values())) != len(buttons)
                or set(buttons.values()) & {11, 12, 13, 14}):
            raise ValueError("gamepad_buttons 须配置不同的 SDL 整数索引 0..20，不能占用方向键 11..14；rb 默认 10")
        return buttons

    def __init__(self, node, cfg, root):
        self.n, self.cfg = node, cfg["multi_task"]
        required = {"handover_time", "stop_blend_time", "return_max_speed"}
        if set(self.cfg) != required or any(not np.isfinite(v) or v <= 0 for v in self.cfg.values()):
            raise ValueError("multi_task 阈值必须完整且为有限正数")
        self.buttons = self.gamepad_buttons(cfg["gamepad_buttons"])
        self.joy_move_mode = "stick"
        self.joy_wait_neutral = False
        check_multi_pd(cfg)
        self.policies = {"walk": node}
        for name in ("crouch", "rise"):
            if cfg["tasks"][name] is None:
                print(f"[multi] {name} 已禁用")
                continue
            self.policies[name] = Policy(cfg, Path(cfg["tasks"][name]), root)
        walk, up = self.policies["walk"], self.policies.get("rise")
        down = self.policies.get("crouch")
        if walk.motion is not None or set(walk.deploy["commands"]) != {"base_velocity"}:
            raise ValueError("walk 必须是速度控制策略")
        if down is not None and up is None:
            raise ValueError("启用 crouch 时必须同时启用 rise")
        if (up is not None and up.motion is None) or (down is not None and down.motion is None):
            raise ValueError("crouch / rise 必须是参考动作跟踪策略")
        if up is not None:
            if not np.allclose(up.motion.positions[-1, up.sim2real], walk.default_real, atol=1e-5):
                raise ValueError("起身末帧必须与走路默认站姿一致")
            self.pose_q = {"stand": up.motion.positions[-1, up.sim2real],
                           "crouch": up.motion.positions[0, up.sim2real]}
        else:
            self.pose_q = {"stand": walk.default_real.copy()}
        if up is not None and cfg.get("crouch_calibration") is not None:
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
        self.state = "wait_current"
        self.state_since = time.monotonic()
        self.pending_target = None
        self.end_announced = False
        self.key_escape = False
        if not getattr(node, "preflight_only", False):
            print("[注意] 手柄 B 中断并保持最后目标（不会切断电机）；4 须扶稳/吊起。")
            self.show_status()

    def _calibrate_crouch(self, calibration):
        """实测蹲姿只用于入口分类，不修改网络观测。"""
        self.pose_q["crouch"] = np.asarray(
            [calibration["joint_pos"][name] for name in self.n.real_joint_names], np.float32)

    def show_status(self):
        crouch_key = "；2/LB+X 下蹲" if "crouch" in self.policies else ""
        move_input = "左摇杆" if self.joy_move_mode == "stick" else "左方向键"
        label, actions = {
            "wait_current": ("等待关节/IMU", "收到后保持当前姿态"),
            "stand_ready": ("站立就绪", f"1/LB+A 走路{crouch_key}；4/LB+Start 慢回站姿"),
            "crouch_ready": ("下蹲就绪", "3/LB+Y 起身；4/LB+Start 慢回站姿"),
            "walking": ("行走中", f"0/Start 立即停策略并回站姿；W/S 前后 A/D 左右 Q/E 转向；{move_input}移动、右摇杆转向；RB 切换移动输入；空格零速"),
            "standing_transition": ("慢回站姿中", "请等待，不能切换"),
            "lowering": ("下蹲中", "请等待，不能切换"),
            "rising": ("起身中", "请等待，不能切换"),
            "stopped": ("手柄中断", "R/Back 解锁；4/LB+Start 慢回站姿"),
        }[self.state]
        print(f"[操作提示] {label} | {actions} | 手柄 B 中断", flush=True)

    def change(self, state, reason):
        self.state = state
        self.state_since = time.monotonic()
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

    def closest_pose(self):
        q = self.n.obs_raw[7:19]
        return min(self.pose_q, key=lambda pose: np.linalg.norm(q - self.pose_q[pose]))

    def return_error(self):
        if not self.has_target or self.state not in ("stand_ready", "crouch_ready", "stopped"):
            return "慢回站姿只允许保持状态；执行动作中无效，不排队"
        return None

    def start_blend(self, duration, reason, goal, done_state):
        self.blend_start = self.n.target_pub.copy()
        self.blend_goal = goal
        self.blend_done_state = done_state
        self.stand_blend_time = duration
        self.active = self.pending_target = None
        self.n._clear_cmd_sources()
        self.n._close_log()
        self.change("standing_transition", reason)

    def request(self, task):
        n, now = self.n, time.monotonic()
        if task in ("walk", "crouch", "rise") and task not in self.policies:
            return self.reject(f"{task} 任务已禁用")
        if task == "halt":
            self.halt("操作员中断")
            return True
        if task == "return_stand":
            if (reason := self.return_error()) is not None:
                return self.reject(reason)
            goal = self.pose_q["stand"]
            distance = np.max(np.abs(n.target_pub - goal))
            # smoothstep 的峰值导数是 1.5；限制的是目标速度，不是电机实际速度。
            duration = max(self.cfg["stop_blend_time"], 1.5 * distance / self.cfg["return_max_speed"])
            self.start_blend(duration, f"操作者确认已支撑；{duration:.1f}s 同步慢回站姿，不运行策略", goal, "stand_ready")
            return True
        if task == "reset":
            if self.state != "stopped":
                return self.reject("仅停止锁存后可重新验收；R 不自动回站立")
            self.change(self.closest_pose() + "_ready" if self.has_target else "wait_current",
                        "人工解除中断；仍需手动选择任务")
            return True
        if task == "stop":
            if self.state == "walking":
                goal = self.pose_q["stand"]
                self.active = self.pending_target = None
                n._clear_cmd_sources()
                n._close_log()
                n.target_real = n.target_pub = goal.copy()
                self.change("stand_ready", "走路策略已停止，直接下发默认站姿")
                n._publish_target()
                return True
            return self.reject("停步回站姿只用于走路；其他动作可按手柄 B 中断")
        required = {"walk": "stand_ready", "crouch": "stand_ready", "rise": "crouch_ready"}
        if task not in required or self.state != required[task] or self.active == task:
            label = {"walk": "走路", "crouch": "下蹲", "rise": "起身"}.get(task, task)
            return self.reject(f"{label}未启动：当前任务状态不允许")
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
        except Exception as exc:
            n.cmd[:] = previous_command
            return self.reject(f"策略接管预检失败: {exc}")
        n._close_log()
        n._clear_cmd_sources()
        self.active, self.pending_target = task, target
        self.handover_from = n.target_pub.copy()
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
        return target, terms

    def update_state(self):
        now = time.monotonic()
        if self.state in ("stand_ready", "crouch_ready") and self.active in ("crouch", "rise"):
            pose = {"crouch": "crouch", "rise": "stand"}[self.active]
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
                self.n.target_pub = self.blend_goal.copy()
                self.n.target_real = self.blend_goal.copy()
                self.change(self.blend_done_state, "插值结束，不自动启动走路")

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
            n.target_pub = ((1-s)*self.blend_start + s*self.blend_goal).astype(np.float32)
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

    def movement_axes(self, axes, buttons):
        # game_controller_node: 方向键是 SDL buttons 11..14，扳机不参与回中检查。
        stick = np.asarray(axes, np.float32)[[1, 0, 2]]
        stick = np.where(np.abs(stick) > self.n.deadzone, stick, 0.)
        dpad = np.array([int(buttons[11] == 1) - int(buttons[12] == 1),
                         int(buttons[13] == 1) - int(buttons[14] == 1)], np.float32)
        if self.joy_wait_neutral:
            if not np.any(stick) and not any(buttons[11:15]):
                self.joy_wait_neutral = False
            return np.zeros(3, np.float32)
        if self.joy_move_mode == "dpad":
            stick[:2] = dpad
        return stick

    def joy(self, msg):
        n = self.n
        def down(name, buttons):
            i = self.buttons.get("rb", 10) if name == "rb" else self.buttons[name]
            return i < len(buttons) and buttons[i] == 1
        def pressed(name):
            return down(name, msg.buttons) and not down(name, n._prev_buttons)
        if pressed("b"):
            self.request("halt")
        elif pressed("back"):
            self.request("reset")
        elif pressed("start"):
            self.request("return_stand" if down("lb", msg.buttons) else "stop")
        elif down("lb", msg.buttons):
            for button, task in (("a", "walk"), ("x", "crouch"), ("y", "rise")):
                if pressed(button):
                    self.request(task)
                    break
        elif pressed("rb"):
            self.joy_move_mode = "dpad" if self.joy_move_mode == "stick" else "stick"
            self.joy_wait_neutral = True
            n._joy_cmd[:] = 0.
            label = "左方向键" if self.joy_move_mode == "dpad" else "左摇杆"
            print(f"[手柄移动输入] 已切换为{label}；松开方向键、左摇杆和右摇杆横轴后生效。键盘指令保持不变。", flush=True)
        n._prev_buttons = list(msg.buttons)
        if self.state != "walking":
            n._joy_cmd[:] = 0.
