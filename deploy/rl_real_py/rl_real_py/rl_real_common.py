"""legs 双足走路模型实机部署节点。

设计要点(对标参考项目 bipedal_ws 的 RL 部署):
- 模型参数全部读训练 run 的 params/deploy.yaml, 部署只在 common.yaml 填一个 run 目录名。
- 步态相位按【真实墙钟时间】推进 (phase = 经过时间 / period), 不再靠计数器堆积,
  循环抖动/掉速也不会让相位漂移 (与 sim2sim.py:454 的 elapsed_time/period 逐位对齐)。
- 数据新鲜度看门狗: run 中关节/IMU 超时没更新则冻结指令, 绝不拿过期观测推理。
- 上电 prepare -> hold -> run 状态机, smoothstep 缓入准备姿态。

get_obs 布局: [0:3]=base_ang_vel  [3:7]=quat(wxyz)  [7:19]=q(实机序)  [19:31]=qd(实机序)
"""
import os
import sys
import time
import csv
import datetime
import termios
import tty
import fcntl

import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState, Imu, Joy
from ament_index_python.packages import get_package_share_directory


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


# 终端设为非阻塞 cbreak, 用于读键盘
_fd = sys.stdin.fileno()
_old_term = termios.tcgetattr(_fd)
tty.setcbreak(_fd)
_old_flags = fcntl.fcntl(_fd, fcntl.F_GETFL)
fcntl.fcntl(_fd, fcntl.F_SETFL, _old_flags | os.O_NONBLOCK)


class RL_real(Node):
    def __init__(self):
        super().__init__("rl_real")

        # ---------- 读部署配置 (硬件相关) ----------
        pkg = get_package_share_directory("rl_real_py")
        src = os.path.normpath(os.path.join(pkg, "..", "..", "..", "..", "src", "rl_real_py"))
        base = src if os.path.isdir(src) else pkg
        with open(os.path.join(base, "configs", "common.yaml")) as f:
            cfg = yaml.safe_load(f)

        run = cfg["run"]
        # logs_root 相对时解析到 legs_rl_lab 根(含 source/legs_rl_lab); 绝对则原样用
        logs_root = cfg["logs_root"]
        if not os.path.isabs(logs_root):
            d = os.path.abspath(os.path.dirname(__file__)); repo = None
            while d != os.path.dirname(d):
                if os.path.isdir(os.path.join(d, "source", "legs_rl_lab")):
                    repo = d; break
                d = os.path.dirname(d)
            if repo is None:
                raise RuntimeError("找不到 legs_rl_lab 根(source/legs_rl_lab), 无法解析相对 logs_root")
            logs_root = os.path.join(repo, logs_root)
        run_dir = os.path.join(logs_root, run)

        # ---------- 读训练导出的 deploy.yaml (所有模型参数) ----------
        with open(os.path.join(run_dir, "params", "deploy.yaml")) as f:
            dep = yaml.safe_load(f)

        self.default_sim = np.array(dep["default_joint_pos"], np.float32)   # 策略序
        self.num_actions = len(self.default_sim)
        self.action_scale = float(dep["actions"]["JointPositionAction"]["scale"][0])
        self.gait_period = float(dep["gait_period"])
        step_dt = float(dep["step_dt"])

        # 观测项: 顺序 / scale / history 全来自 deploy.yaml
        obs = dep["observations"]
        self.obs_names = list(obs.keys())                                   # 训练时的 obs 顺序
        self.term_scales = [np.array(obs[n]["scale"], np.float32) for n in self.obs_names]
        self.term_dims = [len(s) for s in self.term_scales]
        self.num_obs = sum(self.term_dims)
        self.num_history = int(obs[self.obs_names[0]]["history_length"])
        self.num_commands = len(obs["velocity_commands"]["scale"])

        # ---------- 时序 ----------
        self.pub_dt = 1.0 / float(cfg.get("publish_rate", 200))
        self.decimation = max(1, round(step_dt / self.pub_dt))              # 每几次发布跑一次策略

        # ---------- 关节序映射 (按名, 硬件事实) ----------
        real = cfg["joint_index_in_real"]
        sim = cfg["joint_index_in_sim"]
        self.real2sim = [real.index(n) for n in sim]                        # x_sim = x_real[real2sim]
        self.sim2real = [sim.index(n) for n in real]                        # x_real = x_sim[sim2real]
        self.default_real = self.default_sim[self.sim2real].astype(np.float32)
        self.lo = np.array(cfg["joint_lower_limits"], np.float32)
        self.hi = np.array(cfg["joint_upper_limits"], np.float32)

        # ---------- 速度来源 / 交互 / 安全 ----------
        self.use_derived_vel = bool(cfg.get("use_derived_vel", False))
        self.vel_alpha = float(cfg.get("vel_ema_alpha", 0.5))
        self.prepare_time = float(cfg.get("prepare_time", 4.0))
        self.state_timeout = float(cfg.get("state_timeout", 0.2))
        self.cmd_clip = np.array(cfg.get("cmd_clip", [0.5, 0.3, 0.3]), np.float32)
        # 指令零偏修正: 实机零指令下若持续漂移, 给策略一个反向的常量速度指令抵消。
        # [vx, vy, wz], 单位 m/s & rad/s (与 operator 指令同单位, 未经 scale/clip)。
        # 只加到喂给策略的观测上; 日志/clip 仍是 operator 原始指令。全 0 = 不修正。
        self.cmd_bias = np.array(cfg.get("cmd_bias", [0.0, 0.0, 0.0]), np.float32)
        # 下发目标 EMA 平滑: 推理 50Hz, 发布 200Hz, 两拍间用 EMA 插值消除阶梯 -> 电机更平顺。
        # target_pub += alpha*(target - target_pub) 每个 200Hz tick 执行一次。
        # alpha ∈ (0,1]: 1.0=关闭(等同 ZOH), 越小越平滑但滞后越大(T1 约 0.2)。
        self.target_ema_alpha = float(cfg.get("target_ema_alpha", 1.0))
        # 手柄/键盘"松开归零"超时: 超过此时间没有新手柄消息 / 键盘按键重复 -> 对应指令归零。
        # 键盘靠系统按键重复维持"按住"; 松开后重复停止, 超时即归零。手柄靠此防 /joy 停发时 latch。
        self.ctrl_timeout = float(cfg.get("ctrl_timeout", 0.4))
        self.deadzone = float(cfg.get("deadzone", 0.12))
        kb = cfg.get("keyboard", {})
        self.kb_step = float(kb.get("step", 0.1))
        self.kb_max = np.array([kb.get("vx_max", 0.5), kb.get("vy_max", 0.3),
                                kb.get("wz_max", 0.3)], np.float32)

        # ---------- 策略 (onnx 优先, 否则 pt) ----------
        self._load_policy(os.path.join(run_dir, "exported"))

        # ---------- 数据记录: <run_dir>/sim2real/<保存时间>.csv ----------
        self.log_dir = os.path.join(run_dir, "sim2real")

        # ---------- 运行状态 ----------
        self.obs_raw = np.zeros(31, np.float32)
        self.cmd = np.zeros(3, np.float32)                                  # vx, vy, yaw
        self.last_action = np.zeros(self.num_actions, np.float32)
        self.hist = TermGroupedHistory(self.term_dims, self.num_history)
        self.target_real = self.default_real.copy()
        self.target_pub = self.default_real.copy()   # EMA 平滑后的实际下发值
        self.tick = 0

        self.mode = "prepare"          # prepare -> hold -> run
        self.prepare_t0 = None
        self.q_start_real = None
        self.run_t0 = None             # run 起始墙钟时间 (相位基准)

        self._qd = np.zeros(12, np.float32)
        self._motor_vel = np.zeros(12, np.float32)   # 电机上报原始速度 msg.velocity (实机序, 始终记录)
        self._motor_tau = np.zeros(12, np.float32)   # 电机上报力矩 msg.effort (实机序, 始终记录)
        self._prev_q = None
        self._prev_t = None
        self._last_joint_rx = None     # 最近一次收到关节/IMU 的墙钟时间 (看门狗)
        self._last_imu_rx = None
        self._prev_buttons = []
        # 指令源: 手柄比例值(带看门狗) + 键盘累加值(原逻辑), 每 tick 由 _update_cmd 合成 self.cmd。
        self._joy_cmd = np.zeros(3, np.float32)
        self._last_joy_rx = -1e9
        self._kb_cmd = np.zeros(3, np.float32)          # 键盘累加指令(空格清零)
        self._cmd_print = np.full(3, np.nan, np.float32)  # 上次打印的指令(仅变化时打印)
        self._log_f = self._log_w = self._log_path = None

        # ---------- 通信 ----------
        self.create_subscription(JointState, "/left_joint_states", self._on_joint, 5)
        self.create_subscription(Imu, "/imu", self._on_imu, 5)
        self.create_subscription(Joy, "/joy", self._on_joy, 5)
        self.pub = self.create_publisher(Float64MultiArray, "/dog_joint_pos", QoSProfile(depth=1))
        self.create_timer(self.pub_dt, self._tick)

        self.get_logger().info(
            f"run={run}  obs={self.num_obs}x{self.num_history}  "
            f"pub={1/self.pub_dt:.0f}Hz  policy={1/(self.pub_dt*self.decimation):.0f}Hz  "
            f"period={self.gait_period}s")
        print("上电自动缓入准备姿态 -> 站立保持; P/手柄A=行走  B=停  R/X=复位")

    # ------------------------------------------------------------------ 策略
    def _load_policy(self, exported):
        onnx_p = os.path.join(exported, "policy.onnx")
        pt_p = os.path.join(exported, "policy.pt")
        if os.path.exists(onnx_p):
            import onnxruntime as ort
            self._sess = ort.InferenceSession(onnx_p)
            self._in = self._sess.get_inputs()[0].name
            self._out = self._sess.get_outputs()[0].name
            self._infer = self._infer_onnx
            self.get_logger().info(f"policy(onnx): {onnx_p}")
        elif os.path.exists(pt_p):
            import torch
            self._torch = torch
            self._net = torch.jit.load(pt_p)
            self._net.eval()
            self._infer = self._infer_pt
            self.get_logger().info(f"policy(pt): {pt_p}")
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
        phase = ((time.monotonic() - self.run_t0) / self.gait_period) % 1.0   # 墙钟相位
        feats = {
            "base_ang_vel": self.obs_raw[0:3],
            "projected_gravity": gravity_from_quat(self.obs_raw[3:7]),
            "velocity_commands": (self.cmd + self.cmd_bias)[:self.num_commands],
            "joint_pos_rel": q_sim - self.default_sim,
            "joint_vel_rel": qd_sim,
            "last_action": self.last_action,
            "gait_phase": np.array([np.sin(2 * np.pi * phase),
                                    np.cos(2 * np.pi * phase)], np.float32),
        }
        return [(feats[n] * s).astype(np.float32)
                for n, s in zip(self.obs_names, self.term_scales)]

    def _fresh(self):
        """关节与 IMU 是否都在 state_timeout 内更新过。"""
        now = time.monotonic()
        return (self._last_joint_rx is not None and now - self._last_joint_rx < self.state_timeout
                and self._last_imu_rx is not None and now - self._last_imu_rx < self.state_timeout)

    # ------------------------------------------------------------------ 主循环
    def _tick(self):
        self.tick += 1
        self._read_keys()
        self._update_cmd()

        if self.mode == "prepare":
            if self._last_joint_rx is None:          # 没收到关节反馈: 无起点, 不发布
                return
            if self.prepare_t0 is None:
                self.prepare_t0 = time.monotonic()
                self.q_start_real = self.obs_raw[7:19].copy()
                print("\n开始缓入准备姿态 ...")
            a = min(1.0, (time.monotonic() - self.prepare_t0) / max(self.prepare_time, 1e-3))
            s = a * a * (3.0 - 2.0 * a)              # smoothstep: 起停零速
            self.target_real = ((1 - s) * self.q_start_real + s * self.default_real).astype(np.float32)
            self.target_pub = self.target_real.copy()   # prepare 已平滑, 同步滤波器(避免进 run 时跳变)
            if a >= 1.0:
                self.mode = "hold"
                print("\n准备姿态到位, 站立保持 (P/手柄A 开始行走)")

        elif self.mode == "hold":
            self.target_real = self.default_real.copy()
            self.target_pub = self.target_real.copy()   # 保持: 常量, 同步滤波器

        else:  # run
            if self.run_t0 is None:                  # 刚进 run: 起时钟 + 清策略 + 开记录
                self.run_t0 = time.monotonic()
                self.last_action[:] = 0.0
                self.hist = TermGroupedHistory(self.term_dims, self.num_history)
                self.target_pub = self.target_real.copy()   # 从当前保持位起步平滑
                self._open_log()
            if self.tick % self.decimation == 0:
                if not self._fresh():                # 数据陈旧: 冻结指令, 不拿旧观测推理
                    self.get_logger().warn("state stale -> freeze", throttle_duration_sec=1.0)
                else:
                    terms = self._build_terms()
                    x = np.clip(self.hist.update(terms), -100.0, 100.0)
                    self.last_action = np.clip(self._infer(x), -100.0, 100.0)
                    target_sim = self.default_sim + self.last_action * self.action_scale
                    self.target_real = np.clip(target_sim[self.sim2real], self.lo, self.hi)
                    self._log_row(terms)
            # EMA 平滑: 每个 200Hz tick 朝最新推理目标插值一步(alpha=1 时等同 ZOH)
            self.target_pub += self.target_ema_alpha * (self.target_real - self.target_pub)

        m = Float64MultiArray()
        m.data = self.target_pub.tolist()
        self.pub.publish(m)

    # ------------------------------------------------------------------ 模式切换
    def _start_run(self):
        if self.mode == "hold":                      # 只允许站稳后开跑
            self._clear_cmd_sources()                # 进 run 清残留, 防上一轮指令带入
            self.mode = "run"

    def _stop_run(self):
        if self.mode == "run":
            self.mode = "hold"
            self.run_t0 = None
            self._close_log()

    def _reset(self):
        self.mode = "prepare"
        self.prepare_t0 = None
        self.run_t0 = None
        self._clear_cmd_sources()
        self._close_log()
        print("\nreset -> 重新进准备姿态")

    def _clear_cmd_sources(self):
        """清空手柄/键盘残留 + self.cmd。"""
        self.cmd[:] = 0.0
        self._kb_cmd[:] = 0.0
        self._joy_cmd[:] = 0.0
        self._last_joy_rx = -1e9

    # ------------------------------------------------------------------ 回调
    def _on_joint(self, msg):
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
        self.obs_raw[0:3] = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        self.obs_raw[3:7] = [msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z]
        self._last_imu_rx = time.monotonic()

    def _on_joy(self, msg):
        def ax(i):
            v = msg.axes[i] if i < len(msg.axes) else 0.0
            return v if abs(v) > self.deadzone else 0.0
        # 比例式(回中即0); 写入 _joy_cmd + 记时间戳, 由 _update_cmd 合成(超时=/joy停发时归零)
        self._joy_cmd[0] = ax(1) * self.cmd_clip[0]  # 左摇杆纵 = vx
        self._joy_cmd[1] = ax(0) * self.cmd_clip[1]  # 左摇杆横 = vy
        self._joy_cmd[2] = ax(2) * self.cmd_clip[2]  # 右摇杆横 = yaw
        self._last_joy_rx = time.monotonic()

        def pressed(i):
            now = len(msg.buttons) > i and msg.buttons[i] == 1
            was = len(self._prev_buttons) > i and self._prev_buttons[i] == 1
            return now and not was
        if pressed(0):
            self._start_run()                        # A
        elif pressed(1):
            self._stop_run()                         # B
        elif pressed(2):
            self._reset()                            # X
        self._prev_buttons = list(msg.buttons)

    def _update_cmd(self):
        """合成键盘(累加式, 原逻辑) + 手柄(比例式, 带看门狗)指令; 指令变化时实时打印。每 tick 调。"""
        now = time.monotonic()
        joy = self._joy_cmd if (now - self._last_joy_rx < self.ctrl_timeout) else np.zeros(3, np.float32)
        new = np.clip(self._kb_cmd + joy, -self.cmd_clip, self.cmd_clip).astype(np.float32)
        if np.isnan(self._cmd_print[0]) or np.abs(new - self._cmd_print).max() > 0.02:
            print(f"[cmd] vx={new[0]:+.2f}  vy={new[1]:+.2f}  yaw={new[2]:+.2f}")
            self._cmd_print = new.copy()
        self.cmd = new

    def _read_keys(self):
        while True:
            try:
                ch = sys.stdin.read(1)
            except (IOError, OSError):
                ch = ""
            if not ch:
                break
            if ch in "wW":
                self._kb_cmd[0] = min(self.kb_max[0], self._kb_cmd[0] + self.kb_step)
            elif ch in "sS":
                self._kb_cmd[0] = max(-self.kb_max[0], self._kb_cmd[0] - self.kb_step)
            elif ch in "aA":
                self._kb_cmd[1] = min(self.kb_max[1], self._kb_cmd[1] + self.kb_step)
            elif ch in "dD":
                self._kb_cmd[1] = max(-self.kb_max[1], self._kb_cmd[1] - self.kb_step)
            elif ch in "qQ":
                self._kb_cmd[2] = min(self.kb_max[2], self._kb_cmd[2] + self.kb_step)
            elif ch in "eE":
                self._kb_cmd[2] = max(-self.kb_max[2], self._kb_cmd[2] - self.kb_step)
            elif ch == " ":
                self._kb_cmd[:] = 0.0
            elif ch in "pP":
                self._stop_run() if self.mode == "run" else self._start_run()
            elif ch in "rR":
                self._reset()

    # ------------------------------------------------------------------ 记录
    def _open_log(self):
        os.makedirs(self.log_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_path = os.path.join(self.log_dir, f"{ts}.csv")
        self._log_f = open(self._log_path, "w", newline="")
        self._log_w = csv.writer(self._log_f)
        self._log_w.writerow(
            ["t", "cmd_vx", "cmd_vy", "cmd_yaw", "wx", "wy", "wz",
             "qw", "qx", "qy", "qz", "gx", "gy", "gz"]
            + [f"q{i}" for i in range(12)]                     # 关节位置(实机序)
            + [f"qd{i}" for i in range(12)]                    # obs 用的关节速度(实机序; derived 或电机)
            + [f"mvel{i}" for i in range(12)]                  # 电机上报原始速度 msg.velocity(实机序)
            + [f"tau{i}" for i in range(12)]                   # 电机上报力矩 msg.effort(实机序)
            + [f"obs{i}" for i in range(self.num_obs)]         # 单帧观测(仿真序)
            + [f"act{i}" for i in range(self.num_actions)]     # 策略动作(仿真序)
            + [f"cmd{i}" for i in range(self.num_actions)])    # 下发目标角(实机序)
        print(f"\n记录 -> {self._log_path}")

    def _log_row(self, terms):
        if self._log_w is None:
            return
        g = gravity_from_quat(self.obs_raw[3:7])
        row = ([time.monotonic() - self.run_t0, *self.cmd,
                *self.obs_raw[0:3], *self.obs_raw[3:7], *g,
                *self.obs_raw[7:19], *self.obs_raw[19:31],
                *self._motor_vel, *self._motor_tau,
                *np.concatenate(terms), *self.last_action, *self.target_real])
        self._log_w.writerow([round(float(v), 6) for v in row])
        self._log_f.flush()

    def _close_log(self):
        if self._log_f is not None:
            self._log_f.close()
            print(f"\n已保存 -> {self._log_path}")
            self._log_f = self._log_w = None


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RL_real()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node._close_log()
        termios.tcsetattr(_fd, termios.TCSADRAIN, _old_term)
        fcntl.fcntl(_fd, fcntl.F_SETFL, _old_flags)
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
