#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单关节系统辨识 - 实机激励 + 录制 (纯 ROS2 话题, 不改任何现有代码)

发布 /dog_joint_pos (12 维, 实机序 L1..L6,R1..R6) 给 arm_control_node;
订阅 /left_joint_states 记录 q/v/tau。对指定关节叠加 阶跃/正弦/扫频 激励,
其余关节保持起始姿态。同步把 命令 与 状态 写入两份带时间戳的 CSV, 供仿真回放对齐。

安全: 所有命令对被测关节做硬限幅 (base ± max_dev); 起始先缓升到 base。
坏帧检测: 按电机型号的 Q/DQ/TAU 量程判断 railing, 跑完报告坏帧率。

用法示例 (先 source ROS2 与 control_ws):
  # 关节 3, 多级正反阶跃, 每级 hold 1.5s
  python3 excite_record.py --joint 3 --mode step --levels 0.1,0.2,-0.1,-0.2 --hold 1.5
  # 关节 3, 扫频 0.3~5Hz, 幅值 0.15rad, 20s
  python3 excite_record.py --joint 3 --mode chirp --f0 0.3 --f1 5 --dur 20 --amp 0.15
  # 关节 3, 离散多频正弦
  python3 excite_record.py --joint 3 --mode sine --freqs 0.5,1,2,3 --amp 0.15 --cycles 6
"""
import argparse
import math
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

# 按 实机序 (L1..L6, R1..R6) 的电机型号量程: idx0-3,6-9=DM4340; idx4,5,10,11=DM4310
TAU_MAX = np.array([28, 28, 28, 28, 10, 10, 28, 28, 28, 28, 10, 10], dtype=float)
DQ_MAX  = np.array([8,  8,  8,  8,  30, 30, 8,  8,  8,  8,  30, 30], dtype=float)
Q_MAX   = np.full(12, 12.5)


def build_excitation(args):
    """返回 (offset_fn(t_local)->float, total_duration)。t_local 从 0 起。"""
    if args.mode == "step":
        levels = [float(x) for x in args.levels.split(",")]
        seg = args.hold
        total = seg * len(levels)

        def fn(t):
            i = min(int(t / seg), len(levels) - 1)
            return levels[i]
        return fn, total

    if args.mode == "sine":
        freqs = [float(x) for x in args.freqs.split(",")]
        durs = [args.cycles / f for f in freqs]          # 每个频率跑 cycles 个周期
        bounds = np.cumsum(durs)
        total = float(bounds[-1])

        def fn(t):
            i = int(np.searchsorted(bounds, t, side="right"))
            i = min(i, len(freqs) - 1)
            t0 = 0.0 if i == 0 else bounds[i - 1]
            return args.amp * math.sin(2 * math.pi * freqs[i] * (t - t0))
        return fn, total

    if args.mode == "chirp":
        f0, f1, T = args.f0, args.f1, args.dur

        def fn(t):
            t = min(t, T)
            k = (f1 - f0) / T
            phase = 2 * math.pi * (f0 * t + 0.5 * k * t * t)   # 线性扫频瞬时相位
            return args.amp * math.sin(phase)
        return fn, T

    raise ValueError(f"未知 mode: {args.mode}")


def read_joint_limit(j, urdf_dir):
    """返回 (关节名, lower, upper)。idx 0-5=左腿 joint1-6-l, 6-11=右腿 joint1-6-r。"""
    import xml.etree.ElementTree as ET
    side = "l" if j < 6 else "r"
    name = f"joint{j % 6 + 1}-{side}"
    path = os.path.join(urdf_dir, f"damiao_{'left' if side == 'l' else 'right'}.urdf")
    for jt in ET.parse(path).getroot().iter("joint"):
        if jt.get("name") == name:
            lim = jt.find("limit")
            return name, float(lim.get("lower")), float(lim.get("upper"))
    raise ValueError(f"URDF 中未找到关节 {name}")


class ExciteRecorder(Node):
    def __init__(self, args):
        super().__init__("sysid_excite_record")
        self.args = args
        self.j = args.joint
        self.pub = self.create_publisher(Float64MultiArray, "/dog_joint_pos", 1)

        # 文件名要带"本次实际生效的 PD": 默认直接从运行中的 arm_control_node 读真实 kps/kds[joint],
        # 而不是靠命令行手填(手填易和 yaml 对不上)。读不到才回退。
        self.kp_lbl, self.kd_lbl = self._resolve_gains()

        self.sub = self.create_subscription(JointState, "/left_joint_states", self._on_state, 10)

        # 安全: 读 URDF 限位, 按 base 两侧余量限幅(支持非零 base / 非对称限位 / 下界=0 的膝关节)
        base_j = float(args.base[self.j]) if args.base is not None else 0.0
        try:
            jname, lo, hi = read_joint_limit(self.j, args.urdf_dir)
            room = min(base_j - lo, hi - base_j)          # base 到上/下限位的较小余量
            self.safe_amp = max(0.0, args.limit_frac * room)
            self.get_logger().info(
                f"关节 {jname} 限位 [{lo:.2f},{hi:.2f}] base={base_j:.3f} → "
                f"{args.limit_frac*100:.0f}%×余量 安全幅值 ±{self.safe_amp:.3f}")
        except Exception as e:
            self.safe_amp = args.max_dev
            self.get_logger().warn(f"读 URDF 限位失败({e}), 退回 max_dev={args.max_dev}")
        if args.mode == "step":
            levels = [float(x) for x in args.levels.split(",")]
            mx = max((abs(v) for v in levels), default=0.0)
            if mx > self.safe_amp and mx > 0:
                sc = self.safe_amp / mx
                args.levels = ",".join(f"{v*sc:.4f}" for v in levels)
                self.get_logger().warn(f"step 幅值 {mx:.3f} 超安全值, 整体×{sc:.2f} → 最大 {self.safe_amp:.3f}")
        elif args.amp > self.safe_amp:   # sine / chirp
            self.get_logger().warn(f"amp {args.amp:.3f} 超安全值, 收紧到 {self.safe_amp:.3f}")
            args.amp = self.safe_amp
        args.max_dev = min(args.max_dev, self.safe_amp)   # 硬限幅 backstop

        self.offset_fn, self.exc_dur = build_excitation(args)
        self.base = None              # 12 维起始姿态 (来自首帧状态)
        self.start_q = None
        self.cmd_log = []             # (t, phase, q_cmd[12])
        self.state_log = []           # (t, phase, q[12], v[12], tau[12])
        self.bad = 0
        self.nframe = 0

        self.phase = "wait"
        self.done = False
        self.t0 = None                # 整个流程起点 (monotonic)
        self.phase_t0 = None
        self.dt = 1.0 / args.rate
        self.timer = self.create_timer(self.dt, self._tick)
        self.get_logger().info("等待 /left_joint_states 首帧 ...")

    # ---- 解析本次实际 kp/kd(用于文件名): 从运行中的 arm_control_node 读真实参数 ----
    def _resolve_gains(self):
        a = self.args
        # 从运行中的控制节点读 kps/kds 的第 joint 项(用底层 GetParameters 服务, 跨 rclpy 版本通用)
        try:
            from rcl_interfaces.srv import GetParameters
            cli = self.create_client(GetParameters, f"/{a.node_name}/get_parameters")
            if cli.wait_for_service(timeout_sec=a.param_timeout):
                req = GetParameters.Request(names=["kps", "kds"])
                fut = cli.call_async(req)
                rclpy.spin_until_future_complete(self, fut, timeout_sec=a.param_timeout)
                res = fut.result()
                if res is not None and len(res.values) == 2:
                    kps = list(res.values[0].double_array_value)
                    kds = list(res.values[1].double_array_value)
                    kp, kd = kps[self.j], kds[self.j]
                    self.get_logger().info(
                        f"已从 [{a.node_name}] 读到实际增益: kp[{self.j}]={kp:g}  kd[{self.j}]={kd:g}")
                    return f"{kp:g}", f"{kd:g}"
            self.get_logger().warn(
                f"无法从 [{a.node_name}] 读取 kps/kds(超时或未声明),文件名增益标记为 NA")
        except Exception as e:
            self.get_logger().warn(f"读取节点增益异常: {e};文件名增益标记为 NA")
        return "NA", "NA"

    # ---- 状态回调: 记录 + 坏帧检测 ----
    def _on_state(self, msg):
        if len(msg.position) < 12:
            return
        q = np.array(msg.position[:12], dtype=float)
        v = np.array(msg.velocity[:12], dtype=float) if len(msg.velocity) >= 12 else np.full(12, np.nan)
        tau = np.array(msg.effort[:12], dtype=float) if len(msg.effort) >= 12 else np.full(12, np.nan)

        if self.base is None:
            self.start_q = q.copy()
            self.base = np.zeros(12, dtype=float) if self.args.base is None else np.array(self.args.base, dtype=float)
            self.get_logger().info(f"首帧到达, 复位基准 base[{self.j}]={self.base[self.j]:.4f} rad (全关节归0), 开始缓升...")
            self.t0 = time.monotonic()
            self.phase_t0 = self.t0
            self.phase = "ramp"

        # 坏帧: 任一关节 railing 到量程
        self.nframe += 1
        rail = (np.abs(q) >= 0.99 * Q_MAX) | (np.abs(v) >= 0.99 * DQ_MAX) | (np.abs(tau) >= 0.99 * TAU_MAX)
        if np.any(rail):
            self.bad += 1

        if self.t0 is not None:
            self.state_log.append((time.monotonic() - self.t0, self.phase, q, v, tau))

    # ---- 控制循环: 状态机 + 发布命令 ----
    def _tick(self):
        if self.base is None:
            return
        now = time.monotonic()
        tph = now - self.phase_t0
        cmd = self.base.copy()

        if self.phase == "ramp":
            a = min(tph / self.args.ramp, 1.0)
            cmd = self.start_q + a * (self.base - self.start_q)
            if a >= 1.0:
                self._goto("settle", now)
        elif self.phase == "settle":
            if tph >= self.args.settle:
                self.get_logger().info(f"开始激励 mode={self.args.mode} 时长~{self.exc_dur:.1f}s")
                self._goto("excite", now)
        elif self.phase == "excite":
            off = self.offset_fn(tph)
            cmd[self.j] = self._clamp(self.base[self.j] + off)
            if tph >= self.exc_dur:
                self._goto("return", now)
        elif self.phase == "return":
            a = min(tph / 1.0, 1.0)
            last = self._clamp(self.base[self.j] + self.offset_fn(self.exc_dur))
            cmd[self.j] = last + a * (self.base[self.j] - last)
            if a >= 1.0:
                self._finish()
                return

        m = Float64MultiArray()
        m.data = cmd.tolist()
        self.pub.publish(m)
        if self.t0 is not None:
            self.cmd_log.append((now - self.t0, self.phase, cmd.copy()))

    def _clamp(self, x):
        lo = self.base[self.j] - self.args.max_dev
        hi = self.base[self.j] + self.args.max_dev
        return float(min(max(x, lo), hi))

    def _goto(self, phase, now):
        self.phase = phase
        self.phase_t0 = now

    def _finish(self):
        self.phase = "done"
        self._save()
        rate = 100.0 * self.bad / max(self.nframe, 1)
        self.get_logger().info(f"完成。状态帧={self.nframe}  坏帧率={rate:.2f}%  "
                               f"({'数据可信' if rate < 1 else '坏帧偏高, 数据需谨慎(疑似串口后端)'})")
        self.done = True   # 通知 main 循环退出(不在回调里直接 shutdown, 否则 spin 不干净返回)

    def _save(self):
        os.makedirs(self.args.out, exist_ok=True)
        suf = f"_{self.args.suffix}" if self.args.suffix else ""
        tag = f"j{self.j}_kp{self.kp_lbl}_kd{self.kd_lbl}_{self.args.mode}{suf}"
        cpath = os.path.join(self.args.out, f"{tag}_cmd.csv")
        spath = os.path.join(self.args.out, f"{tag}_state.csv")
        with open(cpath, "w") as f:
            f.write("t,phase," + ",".join(f"qd{i}" for i in range(12)) + "\n")
            for t, ph, q in self.cmd_log:
                f.write(f"{t:.6f},{ph}," + ",".join(f"{x:.6f}" for x in q) + "\n")
        with open(spath, "w") as f:
            cols = [f"q{i}" for i in range(12)] + [f"v{i}" for i in range(12)] + [f"tau{i}" for i in range(12)]
            f.write("t,phase," + ",".join(cols) + "\n")
            for t, ph, q, v, tau in self.state_log:
                vals = list(q) + list(v) + list(tau)
                f.write(f"{t:.6f},{ph}," + ",".join(f"{x:.6f}" for x in vals) + "\n")
        self.get_logger().info(f"已保存:\n  {cpath}\n  {spath}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--joint", type=int, required=True, help="被测关节 idx 0-11 (实机序 L1..L6,R1..R6)")
    p.add_argument("--mode", choices=["step", "sine", "chirp"], required=True)
    p.add_argument("--node-name", type=str, default="armcontrol_node", help="被查询增益的控制节点名")
    p.add_argument("--param-timeout", type=float, default=5.0, help="读取节点 kps/kds 参数的超时(秒)")
    p.add_argument("--rate", type=float, default=200.0, help="命令发布频率 Hz")
    p.add_argument("--max-dev", type=float, default=0.4, help="被测关节相对 base 的硬限幅 (rad)")
    p.add_argument("--ramp", type=float, default=2.0, help="起始缓升时长 s")
    p.add_argument("--settle", type=float, default=1.0, help="缓升后静置 s")
    p.add_argument("--base", type=str, default=None, help="可选: 12 维 base 姿态, 逗号分隔; 缺省=首帧实测")
    p.add_argument("--base-j", type=float, default=None,
                   help="只设被测关节的 base(其余=0); 走路中心不在0的关节用(如膝0.7/踝pitch-0.25)")
    p.add_argument("--out", type=str, default="sysid/data/real",
                   help="CSV 输出目录; 一次测试建议 sysid/data/real/<测试时间>/data")
    p.add_argument("--suffix", type=str, default="", help="文件名后缀(区分同关节同模式的多次实验,如 fwd/rev)")
    p.add_argument("--limit-frac", type=float, default=0.5, help="激励幅度上限占 base 两侧余量的比例(默认0.5)")
    p.add_argument("--urdf-dir", type=str,
                   default="control_ws/install/armcontrol/share/armcontrol/urdf/v3.2",
                   help="读关节限位的 URDF 目录")
    # step
    p.add_argument("--levels", type=str, default="0.1,0.2,0.3,-0.1,-0.2,-0.3")
    p.add_argument("--hold", type=float, default=1.5)
    # sine
    p.add_argument("--freqs", type=str, default="0.5,1,2,3")
    p.add_argument("--amp", type=float, default=0.15)
    p.add_argument("--cycles", type=float, default=6)
    # chirp
    p.add_argument("--f0", type=float, default=0.3)
    p.add_argument("--f1", type=float, default=5.0)
    p.add_argument("--dur", type=float, default=20.0)
    args = p.parse_args()
    if args.base is not None:
        args.base = [float(x) for x in args.base.split(",")]
        assert len(args.base) == 12
    elif args.base_j is not None:
        b = [0.0] * 12
        b[args.joint] = args.base_j
        args.base = b

    rclpy.init()
    node = ExciteRecorder(args)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info("中断, 保存已采数据...")
        node._save()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
