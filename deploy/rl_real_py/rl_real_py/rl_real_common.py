"""走路 / 下蹲 / 起身共用的实机节点：模型推理、ROS 通信与日志。

切换由 multi_task.py 管理；上电只保持当前姿态，不自动回站立。
obs_raw: [0:3]=角速度, [3:7]=WXYZ, [7:19]=关节位置, [19:31]=关节速度（实机序）。
"""
import os
import sys
import time
import csv
import datetime
import queue
import threading
import termios
import tty
import fcntl
from pathlib import Path

import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Float64MultiArray, String
from sensor_msgs.msg import JointState, Imu, Joy
from ament_index_python.packages import get_package_share_directory

from .deployment_config import load_settings
from .motion_reference import MotionReference


def gravity_from_quat(q):
    """投影重力(机体系), q = [w, x, y, z]。"""
    w, x, y, z = q
    return np.array([2 * (-z * x + w * y),
                     -2 * (z * y + w * x),
                     1 - 2 * (w * w + z * z)], dtype=np.float32)


class TermGroupedHistory:
    """IsaacLab term-grouped 历史: 每 term 各拼 N 帧, 再把所有 term 拼接。"""

    def __init__(self, term_dims, hist_len):
        self.buffers = [np.zeros((hist_len, d), np.float32) for d in term_dims]
        self.ready = False

    def update(self, terms):
        for buf, obs in zip(self.buffers, terms):
            if not self.ready:
                buf[:] = obs           # 首帧: 用当前观测填满历史
            else:
                buf[:-1] = buf[1:]
                buf[-1] = obs
        self.ready = True
        return np.concatenate([b.flatten() for b in self.buffers])


class Policy:
    """一份模型及其独立观测历史；无 ROS 输出。"""
    motion = None

    def __init__(self, cfg, run_dir, repo):
        # ---------- 读训练导出的 deploy.yaml (所有模型参数) ----------
        with open(os.path.join(run_dir, "params", "deploy.yaml")) as f:
            dep = yaml.safe_load(f)
        with open(os.path.join(run_dir, "params", "agent.yaml")) as f:
            agent = yaml.safe_load(f)
        if dep.get("task_type") == "reference_motion_tracking":
            self.motion = MotionReference(dep, repo, cfg["joint_index_in_real"],
                                          cfg["joint_lower_limits"], cfg["joint_upper_limits"])
        elif dep.get("real_deployment_supported") is False:
            raise ValueError("此任务未实现实机部署")

        self.default_sim = np.array(dep["default_joint_pos"], np.float32)   # 策略序
        self.num_actions = len(self.default_sim)
        self.action_scale = np.asarray(dep["actions"]["JointPositionAction"]["scale"], np.float32)
        if self.action_scale.size != self.num_actions:
            raise ValueError(f"action scale 数量 {self.action_scale.size} != 动作维度 {self.num_actions}")
        self.action_offset = np.asarray(dep["actions"]["JointPositionAction"].get("offset", self.default_sim), np.float32)
        term_clip = dep["actions"]["JointPositionAction"].get("clip")
        self.action_term_clip = None if term_clip is None else np.asarray(term_clip, np.float32)
        self.action_clip = dep.get("policy_action_clip", agent.get("clip_actions"))
        self.gait_period = 1.0 if self.motion is not None else float(dep["gait_period"])
        step_dt = float(dep["step_dt"])
        self.step_dt = step_dt
        command_names = ("lin_vel_x", "lin_vel_y", "ang_vel_z")
        command_ranges = ({name: [0., 0.] for name in command_names} if self.motion is not None
                          else dep["commands"]["base_velocity"]["ranges"])
        self.cmd_min = np.array([command_ranges[name][0] for name in command_names], np.float32)
        self.cmd_max = np.array([command_ranges[name][1] for name in command_names], np.float32)
        if np.any(self.cmd_min > 0) or np.any(self.cmd_max < 0):
            raise ValueError("训练速度范围必须包含零速度")
        if self.motion is None and "velocity_command_limits" in cfg:
            limits = cfg["velocity_command_limits"]
            if not isinstance(limits, dict) or set(limits) != set(command_names):
                raise ValueError("velocity_command_limits 必须包含 lin_vel_x / lin_vel_y / ang_vel_z")
            configured = np.asarray([limits[name] for name in command_names], np.float32)
            if (configured.shape != (3, 2) or not np.isfinite(configured).all()
                    or np.any(configured[:, 0] > 0) or np.any(configured[:, 1] < 0)
                    or np.any(configured[:, 0] < self.cmd_min) or np.any(configured[:, 1] > self.cmd_max)):
                raise ValueError("速度指令截断范围须为有限的 [最小, 最大]，包含零且不能超出训练范围")
            self.cmd_min, self.cmd_max = configured[:, 0], configured[:, 1]

        # 观测项: 顺序 / scale / history 全来自 deploy.yaml
        obs = dep["observations"]
        self.obs_names = list(obs.keys())                                   # 训练时的 obs 顺序
        self.term_scales = [np.array(obs[n]["scale"], np.float32) for n in self.obs_names]
        self.term_dims = [len(s) for s in self.term_scales]
        self.num_obs = sum(self.term_dims)
        self.num_history = int(obs[self.obs_names[0]]["history_length"])
        self.term_clips = [obs[n].get("clip") for n in self.obs_names]
        self.num_commands = 0 if self.motion is not None else len(obs["velocity_commands"]["scale"])
        # legs_static 等"零速静止"策略: gait 相位在零速命令时归零。
        # 直接从 deploy.yaml 的 gait_phase obs 参数自动读 -> 门控策略自动门控, 非门控策略不动。
        gait_params = obs.get("gait_phase", {}).get("params", {})
        self.gait_gate_by_cmd = bool(gait_params.get("gate_by_cmd", False))
        self.gait_command_threshold = float(gait_params.get("command_threshold", 0.1))

        # ---------- 时序 ----------
        self.pub_dt = 1.0 / float(cfg.get("publish_rate", 200))
        self.decimation = max(1, round(step_dt / self.pub_dt))              # 每几次发布跑一次策略
        if not np.isfinite(step_dt) or step_dt <= 0 or not np.isclose(self.decimation * self.pub_dt, step_dt):
            raise ValueError("策略周期必须是发布周期的正整数倍")

        # ---------- 关节序映射 (按名, 硬件事实) ----------
        real = cfg["joint_index_in_real"]
        self.real_joint_names = real
        sim = ([n.removeprefix("joint_") for n in self.motion.joint_names] if self.motion is not None
               else [dep["joint_names"][i].removeprefix("joint_") for i in dep["joint_ids_map"]])
        self.real2sim = [real.index(n) for n in sim]                        # x_sim = x_real[real2sim]
        self.sim2real = [sim.index(n) for n in real]                        # x_real = x_sim[sim2real]
        self.default_real = self.default_sim[self.sim2real].astype(np.float32)
        # common 始终是硬件边界；策略另外保留训练动作裁剪。
        self.lo = np.array(cfg["joint_lower_limits"], np.float32)
        self.hi = np.array(cfg["joint_upper_limits"], np.float32)
        if (self.lo.shape != (12,) or self.hi.shape != (12,)
                or not np.isfinite([self.lo, self.hi]).all() or np.any(self.lo >= self.hi)):
            raise ValueError("common.yaml 关节限位无效")
        self.deploy, self.run_dir = dep, Path(run_dir)
        self.obs_raw = np.zeros(31, np.float32)
        self.cmd = np.zeros(3, np.float32)
        self.cmd_bias = np.asarray(cfg.get("cmd_bias", [0., 0., 0.]), np.float32)
        self.last_action = np.zeros(self.num_actions, np.float32)
        self.hist = TermGroupedHistory(self.term_dims, self.num_history)
        self.run_t0 = None
        self.policy_step = 0
        self._load_policy(os.path.join(run_dir, "exported"))

    # ------------------------------------------------------------------ 策略
    def _load_policy(self, exported):
        onnx_p = os.path.join(exported, "policy.onnx")
        pt_p = os.path.join(exported, "policy.pt")
        if os.path.exists(onnx_p):
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            self._sess = ort.InferenceSession(onnx_p, sess_options=options, providers=["CPUExecutionProvider"])
            self._in = self._sess.get_inputs()[0].name
            self._out = self._sess.get_outputs()[0].name
            self._infer = self._infer_onnx
            print(f"policy(onnx): {onnx_p}")
        elif os.path.exists(pt_p):
            import torch
            self._torch = torch
            torch.set_num_threads(1)
            self._net = torch.jit.load(pt_p, map_location="cpu")
            self._net.eval()
            self._infer = self._infer_pt
            print(f"policy(pt): {pt_p}")
        else:
            raise FileNotFoundError(f"{exported} 下没有 policy.onnx / policy.pt")

    def _infer_onnx(self, x):
        out = self._sess.run([self._out], {self._in: x[None].astype(np.float32)})[0]
        return np.array(out).squeeze().astype(np.float32)

    def _infer_pt(self, x):
        with self._torch.no_grad():
            out = self._net(self._torch.tensor(x, dtype=self._torch.float32)).cpu().numpy()
        return np.array(out).squeeze().astype(np.float32)

    # ------------------------------------------------------------------ 观测
    def _build_terms(self):
        """按 deploy.yaml 的 obs 顺序逐项构造观测 (仿真序), 每项乘该项 scale。"""
        q_sim = self.obs_raw[7:19][self.real2sim]
        qd_sim = self.obs_raw[19:31][self.real2sim]
        phase = 0.0 if self.motion is not None else (self.policy_step * self.step_dt / self.gait_period) % 1.0
        # 门控策略(legs_static): 零速命令时相位归零, 与训练一致 -> 站立不踏步
        gait_flag = 1.0
        if self.gait_gate_by_cmd and np.linalg.norm((self.cmd + self.cmd_bias)[:self.num_commands]) < np.float32(
                getattr(self, "gait_command_threshold", 0.1)):
            gait_flag = 0.0
        feats = {
            "base_ang_vel": self.obs_raw[0:3],
            "projected_gravity": gravity_from_quat(self.obs_raw[3:7]),
            "velocity_commands": (self.cmd + self.cmd_bias)[:self.num_commands],
            "joint_pos_rel": q_sim - self.default_sim,
            "joint_vel_rel": qd_sim,
            "last_action": self.last_action,
            "gait_phase": np.array([np.sin(2 * np.pi * phase) * gait_flag,
                                    np.cos(2 * np.pi * phase) * gait_flag], np.float32),
        }
        if self.motion is not None:
            feats.update(self.motion.features(self.obs_raw[3:7]))
        return [(np.clip(feats[n], *clip) if clip is not None else feats[n]).astype(np.float32) * scale
                for n, scale, clip in zip(self.obs_names, self.term_scales, self.term_clips)]

    def _joint_range_details(self, q, lo, hi, outside):
        return "; ".join(f"{self.real_joint_names[i]}={q[i]:.4f}，范围[{lo[i]:.4f}, {hi[i]:.4f}]"
                         for i in np.flatnonzero(outside))

    def _target_from_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.num_actions,) or not np.isfinite(action).all():
            raise ValueError("策略输出维度错误或包含 NaN/Inf")
        if self.action_clip is not None:
            action = np.clip(action, -self.action_clip, self.action_clip)
        target = self.action_offset + action * self.action_scale
        if self.action_term_clip is not None:
            target = np.clip(target, self.action_term_clip[:, 0], self.action_term_clip[:, 1])
        if target.shape != (12,) or not np.isfinite(target).all():
            raise ValueError("策略目标维度错误或含 NaN/Inf")
        self.last_action = action
        return np.clip(target[self.sim2real], self.lo, self.hi).astype(np.float32)

    def _check_policy(self):
        x = self.hist.update(self._build_terms())
        for _ in range(3):
            self._target_from_action(self._infer(x))
        durations = []
        for _ in range(5):
            begin = time.monotonic()
            self._target_from_action(self._infer(x))
            durations.append(time.monotonic() - begin)
        self.last_action[:] = 0.
        self.hist = TermGroupedHistory(self.term_dims, self.num_history)
        print(f"[policy-check] {x.size}D → {self.num_actions}D；策略 {1/self.step_dt:g}Hz；"
              f"预热后推理最慢 {max(durations)*1000:.2f}ms")


class RL_real(Node, Policy):
    motion = None
    multi = None

    def __init__(self):
        super().__init__("rl_real")
        self.preflight_only = self.declare_parameter("preflight_only", False).value

        # ---------- 读部署配置 (硬件相关) ----------
        pkg = get_package_share_directory("rl_real_py")
        source_config = Path(__file__).resolve().parents[1] / "configs/common.yaml"
        default_config = source_config if source_config.is_file() else Path(pkg) / "configs/common.yaml"
        config_file = self.declare_parameter("config_file", str(default_config)).value
        cfg, run_dir, repo = load_settings(config_file)

        Policy.__init__(self, cfg, run_dir, repo)

        # ---------- 速度来源 / 交互 ----------
        self.use_derived_vel = bool(cfg.get("use_derived_vel", False))
        self.vel_alpha = float(cfg.get("vel_ema_alpha", 0.5))
        # 手柄消息超时后归零；键盘指令由空格清零。
        self.ctrl_timeout = float(cfg.get("ctrl_timeout", 0.4))
        self.deadzone = float(cfg.get("deadzone", 0.12))
        kb = cfg.get("keyboard", {})
        self.kb_step = float(kb.get("step", 0.1))

        # ---------- 运行状态 ----------
        self.target_real = self.default_real.copy()
        self.target_pub = self.default_real.copy()
        self.tick = 0

        self._qd = np.zeros(12, np.float32)
        self._motor_vel = np.zeros(12, np.float32)   # 电机上报原始速度 msg.velocity (实机序, 始终记录)
        self._motor_tau = np.zeros(12, np.float32)   # 电机上报力矩 msg.effort (实机序, 始终记录)
        self._prev_q = None
        self._prev_t = None
        self._last_joint_rx = None     # 首次接管前要求收到关节和 IMU
        self._last_imu_rx = None
        self._prev_buttons = []
        # 指令源由统一状态机合成。
        self._joy_cmd = np.zeros(3, np.float32)
        self._last_joy_rx = -1e9
        self._kb_cmd = np.zeros(3, np.float32)          # 键盘累加指令(空格清零)
        self._log_queue = self._log_path = None
        self._log_threads = []

        from .multi_task import MultiTaskController
        self.multi = MultiTaskController(self, cfg, repo)
        print(f"[config] {config_file}\n[run] {run_dir}")
        if self.preflight_only:
            print("[preflight] PASS：配置/模型/观测/参考/限位检查完成；未创建控制话题、未使能电机。")
            return

        # ---------- 通信 ----------
        self.create_subscription(JointState, "/left_joint_states", self._on_joint, 5)
        self.create_subscription(Imu, "/imu", self._on_imu, 5)
        # 标准化输入与原始 /joy 隔离，避免 joy_node 的设备相关编号混入控制。
        self.create_subscription(Joy, "/gamepad", self._on_joy, 5)
        # 电机状态仅记录告警，不自动中断策略。
        self.create_subscription(String, "/motor_warn", self._on_motor_warn, 20)
        self.pub = self.create_publisher(Float64MultiArray, "/dog_joint_pos", QoSProfile(depth=1))
        self.create_timer(self.pub_dt, self._tick)

        self.get_logger().info(
            f"walk={run_dir.name}  obs={self.num_obs}x{self.num_history}  "
            f"pub={1/self.pub_dt:.0f}Hz  policy={1/(self.pub_dt*self.decimation):.0f}Hz  "
            f"period={self.gait_period}s")

    def _current_state_ready(self):
        """首次接管需要一组可用反馈，以原样保持当前角度。"""
        reason = None
        quat_norm = np.linalg.norm(self.obs_raw[3:7].astype(np.float64))
        if self._last_joint_rx is None or self._last_imu_rx is None:
            reason = "尚未收到完整关节或 IMU 反馈"
        elif not np.all(np.isfinite(self.obs_raw)):
            reason = "关节/速度/IMU 含 NaN 或 Inf"
        elif not np.isfinite(quat_norm) or abs(quat_norm - 1.0) > 0.1:
            reason = f"IMU 四元数无效（范数 {quat_norm:.5f}，应接近 1）"
        else:
            q = self.obs_raw[7:19]
            outside = (q < self.lo) | (q > self.hi)
            if np.any(outside):
                reason = ("反馈超出 common 硬件限位（rad）: "
                          + self._joint_range_details(q, self.lo, self.hi, outside))
        if reason is not None:
            self.get_logger().warn(f"当前姿态不可接管: {reason}", throttle_duration_sec=1.0)
            return False
        return True

    def _capture_current_target(self):
        if not self._current_state_ready():
            return False
        self.target_real = self.obs_raw[7:19].copy()
        self.target_pub = self.target_real.copy()
        return True

    # ------------------------------------------------------------------ 主循环
    def _tick(self):
        self.tick += 1
        self._read_keys()
        self.multi.tick()

    def _publish_target(self):
        if self.target_pub.shape != (12,) or not np.isfinite(self.target_pub).all():
            self.get_logger().warn("下发目标维度错误或含 NaN/Inf，拒绝发布")
            return
        self.target_pub = np.clip(self.target_pub, self.lo, self.hi).astype(np.float32)
        m = Float64MultiArray()
        m.data = self.target_pub.tolist()
        self.pub.publish(m)

    def _clear_cmd_sources(self):
        """清空手柄/键盘残留 + self.cmd。"""
        self.cmd[:] = 0.0
        self._kb_cmd[:] = 0.0
        self._joy_cmd[:] = 0.0
        self._last_joy_rx = -1e9

    # ------------------------------------------------------------------ 回调
    def _on_motor_warn(self, msg):
        """只报告电机状态，不改变控制状态。"""
        self.get_logger().error(f"/motor_warn: {msg.data}")

    def _on_joint(self, msg):
        if len(msg.position) < 12 or len(msg.velocity) < 12:
            self._last_joint_rx = None
            self.get_logger().warn("当前姿态需要完整的 12 关节位置和速度", throttle_duration_sec=1.0)
            return
        q = np.array(msg.position[:12], np.float32)
        self.obs_raw[7:19] = q
        # 电机上报的原始速度/力矩(始终记录, 用于验证解码与反推电机能力)
        if len(msg.velocity) >= 12:
            self._motor_vel = np.array(msg.velocity[:12], np.float32)
        if len(msg.effort) >= 12:
            self._motor_tau = np.array(msg.effort[:12], np.float32)
        if self.use_derived_vel:
            t = time.monotonic()
            if self._prev_q is not None and (t - self._prev_t) > 1e-4:
                raw = (q - self._prev_q) / (t - self._prev_t)
                self._qd = (self.vel_alpha * raw + (1 - self.vel_alpha) * self._qd).astype(np.float32)
            self._prev_q, self._prev_t = q, t
            self.obs_raw[19:31] = self._qd
        else:
            self.obs_raw[19:31] = self._motor_vel
        self._last_joint_rx = time.monotonic()

    def _on_imu(self, msg):
        # IMU 相对机体绕前进方向 x 轴旋转 180°: x 不变, y/z 反向。
        self.obs_raw[0:3] = [msg.angular_velocity.x, -msg.angular_velocity.y, -msg.angular_velocity.z]
        w, x, y, z = msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z
        self.obs_raw[3:7] = [x, -w, -z, y]  # q_body = q_imu ⊗ q_x(-180°)
        self._last_imu_rx = time.monotonic()

    def _on_joy(self, msg):
        try:
            axes = np.asarray(msg.axes, np.float32)
        except (TypeError, ValueError):
            axes = np.empty(0, np.float32)
        if (axes.shape != (6,) or len(msg.buttons) < 15 or not np.isfinite(axes).all()
                or any(b not in (0, 1) for b in msg.buttons)):
            b = self.multi.buttons["b"]
            if b < len(msg.buttons) and msg.buttons[b] == 1:
                self.multi.request("halt")
            self.get_logger().warn("忽略格式错误的 /gamepad 消息", throttle_duration_sec=1.0)
            return
        # SDL: LEFTY=1、LEFTX=0、RIGHTX=2；扳机 4/5 不参与速度控制。
        axes = axes[[1, 0, 2]]
        axes = np.where(np.abs(axes) > self.deadzone, axes, 0.0)
        # 比例式(回中即0)；手柄停发时由状态机归零。
        self._joy_cmd[:] = axes * np.where(axes >= 0.0, self.cmd_max, -self.cmd_min)
        self._last_joy_rx = time.monotonic()
        if not np.isfinite(self._joy_cmd).all():
            self._joy_cmd[:] = 0.
            self.get_logger().warn("忽略非有限手柄速度", throttle_duration_sec=1.0)
            return
        self.multi.joy(msg)

    def _read_keys(self):
        try:
            # 非阻塞文本流空读会在解码器内抛 TypeError；控制键均为 ASCII。
            chars = os.read(sys.stdin.fileno(), 4096).decode("ascii", errors="ignore")
        except (IOError, OSError):
            chars = ""
        self.multi.keys(chars)

    # ------------------------------------------------------------------ 记录
    @staticmethod
    def _write_log(path, header, rows):
        with open(path, "w", newline="") as log_f:
            writer = csv.writer(log_f)
            writer.writerow(header)
            for i, row in enumerate(iter(rows.get, None), 1):
                writer.writerow(row)
                if i % 50 == 0:
                    log_f.flush()

    def _open_log(self, p):
        directory = p.run_dir / "sim2real"
        os.makedirs(directory, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._log_path = os.path.join(directory, f"{ts}.csv")
        header = (["t", "cmd_vx", "cmd_vy", "cmd_yaw", "wx", "wy", "wz",
                   "qw", "qx", "qy", "qz", "gx", "gy", "gz"]
            + [f"q{i}" for i in range(12)]                     # 关节位置(实机序)
            + [f"qd{i}" for i in range(12)]                    # obs 用的关节速度(实机序; derived 或电机)
            + [f"mvel{i}" for i in range(12)]                  # 电机上报原始速度 msg.velocity(实机序)
            + [f"tau{i}" for i in range(12)]                   # 电机上报力矩 msg.effort(实机序; 与 q/mvel 同关节帧)
            + [f"obs{i}" for i in range(p.num_obs)]         # 单帧观测(仿真序)
            + [f"act{i}" for i in range(p.num_actions)]     # 策略动作(仿真序)
            + [f"cmd{i}" for i in range(p.num_actions)])    # 下发目标角(实机序)
        header += ["motion_frame", "reference_time", "policy_ms"] + [f"pub{i}" for i in range(12)]
        header += ["task", "state"]
        # ponytail: 不设上限，保证控制线程绝不等待磁盘；持续数小时部署时再改成有界丢行队列。
        self._log_queue = queue.SimpleQueue()
        thread = threading.Thread(
            target=self._write_log, args=(self._log_path, header, self._log_queue), daemon=True)
        self._log_threads.append(thread)
        thread.start()
        print(f"\n记录 -> {self._log_path}")

    def _log_row(self, terms, p):
        if self._log_queue is None:
            return
        g = gravity_from_quat(self.obs_raw[3:7])
        row = ([time.monotonic() - p.run_t0, *self.cmd,
                *self.obs_raw[0:3], *self.obs_raw[3:7], *g,
                *self.obs_raw[7:19], *self.obs_raw[19:31],
                *self._motor_vel, *self._motor_tau,
                *np.concatenate(terms), *p.last_action, *self.target_real])
        row += [p.motion.frame if p.motion else -1, p.motion.frame / p.motion.fps if p.motion else 0.,
                self.multi.policy_ms, *self.target_pub]
        row = [round(float(v), 6) for v in row]
        row += [self.multi.active, self.multi.state]
        self._log_queue.put(row)

    def _close_log(self):
        if self._log_queue is not None:
            self._log_queue.put(None)
            print(f"\n已保存 -> {self._log_path}")
            self._log_queue = self._log_path = None

    def _join_logs(self):
        for thread in self._log_threads:
            thread.join(timeout=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = None
    fd = None
    try:
        node = RL_real()
        if node.preflight_only:
            return
        if not sys.stdin.isatty():
            raise RuntimeError("实机控制需要交互终端；仅离线预检可使用 preflight_only:=true")
        fd = sys.stdin.fileno()
        old_term = termios.tcgetattr(fd)
        old_flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        tty.setcbreak(fd)
        fcntl.fcntl(fd, fcntl.F_SETFL, old_flags | os.O_NONBLOCK)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node._close_log()
            node._join_logs()
            node.destroy_node()
        if fd is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
            fcntl.fcntl(fd, fcntl.F_SETFL, old_flags)
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
