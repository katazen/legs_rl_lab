#!/usr/bin/env python3
"""平衡姿态 <-> 走路默认姿态 的慢速同步插值工具(含力矩看门狗与全程记录)。

背景: 机器人靠外置限位模块卡在一个"平衡姿态", 需要慢速起立到走路默认姿态,
并能原路返回。难点是脚底受地面摩擦锁定, 同时改变脚间距与高度容易崴倒。

路径选择 —— 12 关节同步走一段, 不拆路点:
  曾经为了绕开"承重挪脚力矩不够"而设计过交替卸载(侧倾把一只脚卸载再蹭它)的
  15 路点序列。实机跑到第 13 段跳闸, 而且中途往后倒。拿那次的实测轨迹逐帧回算,
  原因很清楚: 侧倾本身把质心推到了支撑面后缘 —— 最差边距只剩 +18.0 mm,
  质心 x 到了 -26.2 mm(已落到踝轴之后), 脚倾也到 10~12°(脚在边缘承重)。
  同一套模型算 12 关节同步插值: 全程边距 >= +53.4 mm, 质心 x 始终 +8.9~+31.2 mm
  稳稳落在踝前, 躯干俯仰 0->24.6° 单调平滑, 脚倾不超过 6.1°。
  为了省力矩而引入失稳, 是笔亏本买卖 -> 回到同步插值。

  代价是脚全程承重挪位(两个姿态的脚间距 265 vs 451 mm, 每只脚还要原地扭 40°)。
  主障碍是髋roll 的静摩擦: 沿这条路径需求 17.6~20.2 N·m(μ=0.65 估), 而实测反推的
  静摩擦门槛在 12.6~17.7 N·m(R2 用 17.7 挪到 97%, L2 只有 12.6 就冻在 0.30 rad)。
  摩擦门槛由位置误差决定、与速度无关, 所以放慢并不能规避 -> 靠看门狗放到全额解决。
  膝反而完全不吃力: 需求只有 1.4~5.9 N·m(质心几乎压在膝轴正上方, 力臂 23~99 mm)。

安全设计:
- armcontrol 生效参数 max_vel=0 -> 纯位置 PD: tau = kp*(q_des-q) - kd*dq
- 力矩看门狗: 两路并行, 任一路超限立即冻结目标于实测位置并退出
    路1 位置误差: 阈值 = effort_limit/kp(全额)
    路2 电机自报 effort: 阈值 = effort_limit 本身(26 / 踝roll 5.8 N·m)
- smoothstep 起止导数为 0 -> 起步与到位都没有速度台阶, 也便于从 rl_real 热交接
- 到位后 **持续保持下发直到 Ctrl+C**(--no-keep 关闭)。切记: 这个节点没有命令超时
  保护, 但达妙电机固件自己有(err_code 0xD=通讯丢失) -> 脚本一退出、停发 control_mit,
  电机就自行失能, 腿失去支撑直接塌下去(实测髋pitch 塌了 0.47 rad)。它不会报错,
  只会让 effort 静静变成 0 -> 只看位置读数发现不了。
  推论: 这个进程一定要显式收干净, 否则它会一直以 200 Hz 发着旧姿态, 和后来启动的
  rl_real 争抢 /dog_joint_pos(armcontrol QoS depth=1 只取最新), 表现为"启动后停在旧姿态"。
- 同时记录推算力矩与电机自报 effort, 并交叉验证符号链(详见 FB_SIGN)

用法:
  python3 pose_move.py selfcheck                 # 原地保持自检 + 记录平衡姿态
  python3 pose_move.py to_balance                # 走路准备 -> 下蹲平衡(顶挡块)
  python3 pose_move.py to_stand                  # 退回走路准备姿态
  python3 pose_move.py roundtrip --dwell 5       # 一次跑完往返
  python3 pose_move.py to_balance --frac 0.3     # 只走行程前 30%(试探, 不顶挡块)
  python3 pose_move.py to_balance --move 90      # 更慢(缺省 60s)
"""
import argparse, json, os, sys, time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

from com_check import real_limits

REAL = ["L1","L2","L3","L4","L5","L6","R1","R2","R3","R4","R5","R6"]
DESC = ["髋pitch","髋roll","髋yaw","膝","踝pitch","踝roll"] * 2

# armcontrol install/share/config/arm_control_node.yaml 生效值(实机序)
KP = np.array([200.,100.,100.,250.,40.,40., 200.,100.,100.,250.,40.,40.])
KD = np.array([5.,5.,5.,5.,2.,0.5, 5.,5.,5.,5.,2.,0.5])
# effort_limit: legs/踝pitch 26 N·m, 踝roll 5.8 N·m (来自 NLEGS_CFG)
EFF = np.array([26.,26.,26.,26.,26.,5.8, 26.,26.,26.,26.,26.,5.8])
# 逐关节误差阈值(rad) = 全额 effort_limit 对应的误差。
# 曾经乘过 0.6 留余量, 但那个 0.6 是拍的, 而且把髋roll 卡在 15.6 N·m ——
# 正好落在实测静摩擦门槛 12.6~17.7 N·m 中间, 结果一条腿挪得动一条挪不动。
# 现在放到全额: 髋roll 0.26 rad(26 N·m), 4340 的 CAN 编码上限 T_MAX=30 N·m 仍在之上。
# 代价: 踝pitch 阈值变成 0.65 rad, 已超过它自己的行程 ±0.4 -> 该关节实质上没有
# 看门狗了。要重新收紧就用 --err-cap。
ERR_LIMIT = EFF / KP

# armcontrol 的 kLegCmdSign / kLegFbSign (两者完全相同): 命令乘 s 下发, 反馈乘 s 发布,
# 两次翻转互相抵消 -> "读数 vs 目标"永远自洽。自洽不等于正确: 若某位与真实物理方向
# 不符, 就构成一个自洽的镜像世界。而 effort 发布时没乘 s(裸值), 于是有精确预测:
#   tau_推算 = kp(q_des - s*q_raw) - kd*s*dq_raw,  tau_自报 = kp(s*q_des - q_raw) - kd*dq_raw
#   ∵ s^2 = 1  ⟹  tau_推算 = s * tau_自报
# 任何偏离都说明该关节的符号链有问题 -> _report 里逐关节校验。
FB_SIGN = np.array([1.,1.,-1.,1.,-1.,1., -1.,1.,-1.,-1.,1.,1.])

# 走路默认姿态(实机序), 取自 deploy.yaml default_joint_pos 重排
STAND = np.array([-0.1,0.,0.,0.2,-0.1,0., -0.1,0.,0.,0.2,-0.1,0.])

# 全关节零位: 只用于零点诊断(悬空时两腿应笔直铅垂), 不是行走姿态
ZERO = np.zeros(12)

# 单段同步插值的缺省总时长(s)
MOVE_T = 60.0

RATE = 200.0
HERE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.join(HERE, "pose_logs")
BALANCE_JSON = os.path.join(LOGDIR, "balance_pose.json")


def smoothstep(a):
    a = min(max(a, 0.0), 1.0)
    return a * a * (3.0 - 2.0 * a)


class PoseMover(Node):
    def __init__(self, mode, hold_dur, overshoot, dwell, move_t, frac, only, err_cap):
        super().__init__("pose_mover")
        self.mode = mode
        self.hold_dur, self.overshoot = hold_dur, overshoot
        self.dwell = dwell               # 往返中点停留 s
        self.move_t = move_t             # 单段过渡总时长 s
        self.frac = frac                 # 只走行程的前 frac(试探用), 1.0=走完
        self.only = only                 # 只动这些关节(实机名), None=全动
        self.err_lim = np.minimum(ERR_LIMIT, err_cap)   # 看门狗阈值封顶
        self.q = None
        self.dq = np.zeros(12)
        self.tau_fb = np.zeros(12)       # 电机自报力矩 msg.effort (裸值, 未乘 FB_SIGN)
        self.pub = self.create_publisher(Float64MultiArray, "/dog_joint_pos", 10)
        self.create_subscription(JointState, "/left_joint_states", self._on_joint, 50)
        # armcontrol 已在主循环逐帧检查 12 个电机的 GetErrCode(), 并在"全失能/全使能/
        # 单机故障码变化"时往 /motor_warn 发 JSON。但它只在跳变时发一次 -> 必须全程订阅并
        # 带时间戳记下来, 否则事后根本无法知道"动作中途是不是有电机失能了"。
        self.create_subscription(String, "/motor_warn", self._on_warn, 20)
        self.warns = []
        self.rows, self.tripped, self.done = [], None, False
        self.q_start = None
        self.segs = []                   # [(名字, 目标12, 时长, 停留)]
        self.target = None
        self.hold_target = None          # 退出后由 main 持续保持下发的目标(防失能塌陷)
        self.t0 = None
        self.phase = "wait"
        os.makedirs(LOGDIR, exist_ok=True)
        self.timer = self.create_timer(1.0 / RATE, self._tick)

    def _on_joint(self, m):
        if len(m.position) >= 12:
            self.q = np.array(m.position[:12])
            if len(m.velocity) >= 12:
                self.dq = np.array(m.velocity[:12])
            if len(m.effort) >= 12:
                self.tau_fb = np.array(m.effort[:12])

    def _on_warn(self, m):
        t = (time.monotonic() - self.t0) if self.t0 is not None else -1.0
        self.warns.append((t, m.data))
        print(f"\n!! /motor_warn t={t:.2f}s  {m.data}")

    # ---------------- 目标规划 ----------------
    def _balance_goal(self):
        """读录制的平衡姿态, 夹到实机可达行程。

        录的是实机读数: 挡块承力时电机停在挡块上, 读数会略微顶过软限位
        (实测踝pitch -0.402/-0.413)。差这一点不影响断电后的自锁。
        """
        with open(BALANCE_JSON) as f:
            bal = np.array(json.load(f)["balance_pose_real_order"])
        lo, hi = real_limits()
        return np.clip(bal, lo, hi)

    def _leg(self, name, q0, goal, tail_overshoot):
        """一段到底: 12 关节同步 smoothstep 从 q0 走到 goal。

        --frac 只走行程的前一部分(按比例缩短时长, 保持同样的速度), 用来试探;
        试探时不顶挡块。
        """
        if self.frac < 1.0:
            return [(name + f" {self.frac*100:.0f}%",
                     q0 + self.frac * (goal - q0), self.move_t * self.frac, 0.0)]
        if tail_overshoot and self.overshoot > 0:
            # 朝"顶上挡块"方向微量过冲: 让外置限位模块承力, 断电后腿不会掉。
            # 过冲量很小(默认 0.01 rad), 对应力矩 kp*0.01 = 0.4~2.5 N·m, 既顶得住又安全。
            goal = goal + np.sign(goal - q0) * self.overshoot
        return [(name, goal, self.move_t, 0.0)]

    def _plan(self):
        """返回段列表 [(名字, 目标12, 时长, 后置停留)]。

        to_balance: 实测起点 -> 平衡姿态(顶挡块)。
        to_stand:   实测位置 -> STAND。不管当前歪在哪(比如上次跳闸冻在半途),
                    都是同一条同步插值回去, 不依赖"来时走过什么路径"。
        roundtrip:  去程(顶挡块) -> 停留 dwell -> 回到出发姿态。
        """
        if self.mode in ("selfcheck", "relax"):
            return []
        if self.mode == "to_stand":
            return self._leg("move 走路准备", self.q_start.copy(), STAND, False)
        if self.mode == "to_zero":
            return self._leg("move 全关节零位", self.q_start.copy(), ZERO, False)
        bal = self._balance_goal()
        if self.mode == "to_balance":
            return self._leg("move 下蹲平衡", self.q_start.copy(), bal, True)
        out = self._leg("move 下蹲平衡", self.q_start.copy(), bal, True)
        nm, q, d, _ = out[0]
        return [(nm, q, d, self.dwell),
                ("back 走路准备", self.q_start.copy(), self.move_t, 0.0)]

    def _apply_only(self):
        """--joints 试探模式: 指定关节以外全程冻结在起点。"""
        if not self.only:
            return
        keep = [i for i, n in enumerate(REAL) if n not in self.only]
        self.segs = [(nm, self._freeze(q, keep), d, w) for nm, q, d, w in self.segs]

    def _freeze(self, q, keep):
        q = q.copy()
        q[keep] = self.q_start[keep]
        return q

    def _target_at(self, t):
        """逐段 smoothstep: 每段起止速度都为 0, 所以起步和到位都没有速度台阶。"""
        prev = self.q_start
        for k, (nm, goal, dur, dwell) in enumerate(self.segs):
            if t < dur:
                s = smoothstep(t / dur) if dur > 0 else 1.0
                return prev + s * (goal - prev), k, nm, s
            t -= dur
            if t < dwell:
                return goal.copy(), k, nm + "(停留)", 1.0
            t -= dwell
            prev = goal
        return prev.copy(), len(self.segs), "hold", 1.0

    @property
    def total_move_time(self):
        return sum(d + w for _, _, d, w in self.segs)

    # ---------------- 主循环 ----------------
    def _tick(self):
        if self.q is None:
            return
        if self.q_start is None:
            self.q_start = self.q.copy()
            self.segs = self._plan()
            self._apply_only()
            self.goal = self.segs[-1][1] if self.segs else self.q_start.copy()
            self.t0 = time.monotonic()
            self.phase = "move" if self.segs else "hold"
            print(f"起点(实机序): [{', '.join(f'{x:+.4f}' for x in self.q_start)}]")
            print(f"终点(实机序): [{', '.join(f'{x:+.4f}' for x in self.goal)}]")
            if self.frac < 1.0:
                print(f"** 试探模式: 只走行程的前 {self.frac*100:.0f}%, 不顶挡块")
            if self.only:
                print(f"** 只动关节: {','.join(self.only)}  阈值封顶 {self.err_lim.max():.3f} rad")
            if self.segs:
                print(f"\n共 {len(self.segs)} 段 {self.total_move_time:.0f}s"
                      f" + 保持 {self.hold_dur:.0f}s:")
                for k, (nm, q, d, w) in enumerate(self.segs):
                    print(f"  {k+1:2d}. {nm:<26}{d:5.1f}s"
                          f"{'  停留 %.0fs' % w if w else ''}")
            else:
                print("原地保持")
            print()

        t = time.monotonic() - self.t0
        if self.mode == "relax":
            # 目标每帧跟随实测 -> 误差恒 0 -> 力矩恒 0, 等效断电但可随时收回控制权
            self.target, k, seg, s = self.q.copy(), -1, "hold", 1.0
        elif self.segs:
            self.target, k, seg, s = self._target_at(t)
        else:
            self.target, k, seg, s = self.q_start.copy(), -1, "hold", 1.0

        if seg != self.phase:
            print(f"  t={t:6.1f}s  -> {seg}")
            self.phase = seg

        err = self.target - self.q
        tau = KP * err - KD * self.dq

        # ---- 力矩看门狗: 两路判定, 任一路超限即停 ----
        # 路1(位置误差): tau = kp*err - kd*dq 推算。
        # 路2(电机自报): msg.effort 裸值, 取绝对值故无需管 FB_SIGN。
        # 为什么要路2: 实测证明路1 会系统性低估电机真实输出(同一帧 R2 推算 21.8 而自报
        # 30.0, L5 推算 3.9 而自报 7.3)。所以只按 kp*err 设阈值拦不住过温 —— L2 线圈
        # 就是在路1 显示"没超"的情况下烧的。自报值取自电机侧, 是唯一可信的力矩来源。
        over = np.abs(err) > self.err_lim
        over_fb = np.abs(self.tau_fb) > EFF
        if (over.any() or over_fb.any()) and self.tripped is None:
            if over_fb.any():      # 自报优先: 它比推算可信
                j, why = int(np.argmax(np.abs(self.tau_fb) / EFF)), "自报力矩"
            else:
                j, why = int(np.argmax(np.abs(err) / self.err_lim)), "位置误差"
            self.tripped = (t, REAL[j], err[j], tau[j])
            self.target = self.q.copy()          # 冻结在实测位置 -> 零误差
            print(f"\n!! 看门狗触发[{why}] t={t:.2f}s  {REAL[j]}({DESC[j]}) "
                  f"err={err[j]:+.4f} rad (阈值 {self.err_lim[j]:.3f})  "
                  f"tau 推算={tau[j]:+.1f} / 自报={self.tau_fb[j]:+.1f} N·m (上限 {EFF[j]:.1f})")
            print("   已冻结目标于实测位置, 准备退出")

        if self.tripped is not None:
            self.target = self.q.copy()

        m = Float64MultiArray(); m.data = self.target.tolist()
        self.pub.publish(m)

        self.rows.append(np.concatenate(([t], self.target, self.q, err, self.dq, tau, self.tau_fb,
                                         [float(k), s])))

        if self.tripped is not None and t - self.tripped[0] > 0.5:
            self._finish("看门狗中断")
        elif t > self.total_move_time + self.hold_dur:
            self._finish("正常结束")

    def _finish(self, why):
        if self.done:
            return
        self.timer.cancel()
        # 看门狗跳闸时保持在实测位置(零误差); 正常结束时保持在目标位置(继续承重)。
        self.hold_target = (self.q.copy() if self.tripped is not None else self.target.copy())
        self._report(why)
        self.done = True          # 由 main 循环检测退出, 不在回调里 shutdown

    # ---------------- 报告 ----------------
    def _report(self, why):
        d = np.array(self.rows)
        ts = time.strftime("%Y%m%d_%H%M%S")
        csv = os.path.join(LOGDIR, f"{self.mode}_{ts}.csv")
        hdr = (["t"] + [f"tgt_{n}" for n in REAL] + [f"q_{n}" for n in REAL]
               + [f"err_{n}" for n in REAL] + [f"dq_{n}" for n in REAL]
               + [f"tau_{n}" for n in REAL] + [f"taufb_{n}" for n in REAL] + ["seg", "s"])
        np.savetxt(csv, d, delimiter=",", header=",".join(hdr), comments="", fmt="%.6f")

        # 列布局: t(1)+tgt(12)+q(12)+err(12)+dq(12)+tau(12)+taufb(12)+seg,s(2) = 75
        q_f = d[-1, 13:25]
        err_all = d[:, 25:37]
        tau_all = d[:, 49:61]
        taufb_all = d[:, 61:73]
        print(f"\n===== {self.mode} / {why} =====")
        print(f"时长 {d[-1,0]:.2f}s, {len(d)} 帧 -> {csv}\n")
        print(f"{'关节':<4}{'部位':<9}{'起点':>9}{'目标':>9}{'实到':>9}{'到位误差':>10}"
              f"{'|err|max':>10}{'阈值':>8}{'|tau|max':>10}")
        print("-" * 90)
        for i, nm in enumerate(REAL):
            print(f"{nm:<4}{DESC[i]:<9}{self.q_start[i]:>9.4f}{self.goal[i]:>9.4f}{q_f[i]:>9.4f}"
                  f"{q_f[i]-self.goal[i]:>10.4f}{np.abs(err_all[:,i]).max():>10.4f}"
                  f"{ERR_LIMIT[i]:>8.3f}{np.abs(tau_all[:,i]).max():>10.1f}")
        print(f"\n最大 |到位误差| = {np.abs(q_f-self.goal).max():.4f} rad "
              f"({np.rad2deg(np.abs(q_f-self.goal).max()):.2f} deg) @ {REAL[int(np.abs(q_f-self.goal).argmax())]}")
        print(f"全程最大 |tau| = {np.abs(tau_all).max():.1f} N·m @ {REAL[int(np.abs(tau_all).max(0).argmax())]}")
        self._check_sign(tau_all, taufb_all)
        self._check_mirror(q_f)
        if self.warns:
            print(f"\n全程 /motor_warn 事件 {len(self.warns)} 条:")
            for tw, txt in self.warns:
                print(f"  t={tw:7.2f}s  {txt}")
        else:
            print("\n/motor_warn: 全程无事件(使能状态未发生跳变)")
        if self.mode == "selfcheck":
            with open(BALANCE_JSON, "w") as f:
                json.dump({"balance_pose_real_order": self.q_start.tolist(),
                           "joint_order": REAL, "recorded": ts}, f, indent=2)
            print(f"\n平衡姿态已记录 -> {BALANCE_JSON}")

    def _check_sign(self, tau_all, taufb_all):
        """符号链交叉验证: 应有 tau_推算 ≈ FB_SIGN * tau_自报 (推导见文首 FB_SIGN)。

        这是唯一能突破"自洽镜像世界"的手段 —— 只比目标 vs 读数永远看不出问题。
        失能判据必须看"指令力矩显著而自报力矩为 0": selfcheck 时目标=实测, PD 力矩
        本来就≈0, effort 也≈0, 那是正常的, 不能据此判失能。
        """
        cmd_amp = np.abs(tau_all).max()
        fb_amp = np.abs(taufb_all).max()
        if cmd_amp < 1.0:
            print(f"\n符号链校验: 跳过 —— 指令力矩全程最大仅 {cmd_amp:.2f} N·m, "
                  f"没有足够激励(effort max {fb_amp:.2f})。需带负载的动作才能判定。")
            return
        if fb_amp < 0.2:
            print(f"\n!! 指令力矩最大 {cmd_amp:.1f} N·m 但电机自报 effort 全程≈0 "
                  f"-> 电机未使能/已失能, 位置环根本没在工作")
            return
        print("\n符号链校验 (预期 tau_推算 = FB_SIGN * tau_自报, 只统计 |tau_自报|>0.5 的帧):")
        bad = []
        for i, nm in enumerate(REAL):
            m = np.abs(taufb_all[:, i]) > 0.5
            if m.sum() < 20:
                print(f"  {nm:<4}{DESC[i]:<9}样本不足({int(m.sum())} 帧), 跳过")
                continue
            # 相关系数符号即可判定, 不依赖幅值标定是否精确
            a, b = tau_all[m, i], taufb_all[m, i]
            r = float(np.corrcoef(a, b)[0, 1])
            got = 1.0 if r > 0 else -1.0
            ok = (got == FB_SIGN[i])
            if not ok:
                bad.append(nm)
            print(f"  {nm:<4}{DESC[i]:<9}预期{FB_SIGN[i]:+.0f} 实测{got:+.0f} "
                  f"corr={r:+.3f}  |tau_推算|max={np.abs(a).max():.1f} "
                  f"|tau_自报|max={np.abs(b).max():.1f}{'' if ok else '   <-- 不符!'}")
        if bad:
            print(f"  !! 符号链异常关节: {','.join(bad)} —— 该关节可能存在镜像错配, 读数不可信")
        else:
            print("  OK: 12 关节符号链全部符合预期")

    def _check_mirror(self, q_f):
        """左右镜像残差: 两条腿可以各自都在容差内, 却一前一后差几 cm。

        关节1髋pitch/4膝/5踝pitch 左右同号 -> 看 L-R; 2髋roll/3髋yaw/6踝roll 反号 -> 看 L+R。
        """
        same = (0, 3, 4)
        print("\n左右镜像残差(到位后):")
        worst, worst_j = 0.0, 0
        for j in range(6):
            l, r = q_f[j], q_f[j + 6]
            res = (l - r) if j in same else (l + r)
            if abs(res) > worst:
                worst, worst_j = abs(res), j
            print(f"  关节{j+1} {DESC[j]:<9}L{l:+.4f} R{r:+.4f} "
                  f"{'差' if j in same else '和'}={res:+.4f}")
        print(f"  最大镜像残差 {worst:.4f} rad ({np.rad2deg(worst):.2f} deg) @ 关节{worst_j+1} {DESC[worst_j]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["selfcheck", "to_stand", "to_zero", "to_balance", "roundtrip", "relax"])
    p.add_argument("--move", type=float, default=MOVE_T,
                   help=f"单段同步插值总时长 s(缺省 {MOVE_T:.0f})")
    p.add_argument("--dwell", type=float, default=5.0, help="roundtrip 在平衡姿态停留 s")
    p.add_argument("--hold", type=float, default=3.0, help="到位后保持 s")
    p.add_argument("--overshoot", type=float, default=0.01,
                   help="终点朝挡块方向的过冲量 rad(让限位模块承力)")
    p.add_argument("--frac", type=float, default=1.0,
                   help="只走行程的前这么多(0~1, 试探用, 不顶挡块); 1=走完")
    p.add_argument("--joints", type=str, default="",
                   help="只动这些关节, 逗号分隔如 L5,R5; 空=全动")
    p.add_argument("--err-cap", type=float, default=10.0,
                   help="看门狗误差阈值封顶 rad(与逐关节阈值取较小值)")
    p.add_argument("--no-keep", dest="keep", action="store_false",
                   help="到位后立即退出(默认会持续保持下发直到 Ctrl+C)。"
                        "注意: 停发后电机固件会因通讯丢失自行失能, 腿会塌下去")
    a = p.parse_args()
    if a.mode in ("to_balance", "roundtrip") and not os.path.exists(BALANCE_JSON):
        sys.exit(f"缺少 {BALANCE_JSON}, 请先跑 selfcheck 记录平衡姿态")
    only = [s.strip() for s in a.joints.split(",") if s.strip()]
    rclpy.init()
    n = PoseMover(a.mode, a.hold, a.overshoot, a.dwell, a.move, a.frac, only, a.err_cap)
    try:
        while rclpy.ok() and not n.done:
            rclpy.spin_once(n, timeout_sec=0.1)
        # 不松手: 持续下发最终目标维持电机使能与承重, 直到人工接手后 Ctrl+C。
        if a.keep and n.hold_target is not None:
            print("\n>>> 持续保持下发中 (维持电机使能、腿不会塌)。扣住机器人后按 Ctrl+C 退出。")
            print(">>> 一旦退出, 电机将因通讯丢失在短时间内自行失能 -> 腿失去支撑。\n")
            m = Float64MultiArray(); m.data = n.hold_target.tolist()
            while rclpy.ok():
                n.pub.publish(m)
                rclpy.spin_once(n, timeout_sec=0.0)
                time.sleep(1.0 / RATE)
    except KeyboardInterrupt:
        print("\n用户中断 -> 停止下发, 电机即将失能")
    finally:
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
