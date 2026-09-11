#!/usr/bin/env python3
"""DM4340 上电位置锁存诊断工具 (只订阅 /left_joint_states, 绝不发布 -> 永不驱动电机)。

背景: 2026-09-03 实测确认电机上电时位置初始化会偶发跳变, 幅度是离散量子的整数倍
      (当次 L1髋pitch 与 R3髋yaw 同时跳 0.13047 rad = 342 LSB)。跳变发生在电机上电,
      与 armcontrol 重启无关, 与徒手搬动无关。本工具用于抓取和量化这类跳变。

子命令:
  record <csv>          全程按原始速率记录 12 关节位置/力矩, 阶段标记由 MARKFILE 注入
  snap   <tag> [秒数]   采样并存快照 JSON, 打印均值/抖动/力矩
  diff   <tagA> <tagB>  逐关节比对两个快照 (B - A)
  jump   <csv> [阈值]   扫描记录中的离散跳变, 阈值默认 0.025 rad
"""
import json
import os
import sys
import time

import numpy as np

NM = ['L1髋p', 'L2髋r', 'L3髋y', 'L4膝', 'L5踝p', 'L6踝r',
      'R1髋p', 'R2髋r', 'R3髋y', 'R4膝', 'R5踝p', 'R6踝r']
HERE = os.path.dirname(os.path.abspath(__file__))
SNAPDIR = os.path.join(HERE, "pose_logs", "snaps")
MARKFILE = "/tmp/latch_mark"
LSB = 25.0 / 65535.0          # DM 电机 P_MAX=12.5 -> 位置量化步长 rad


def _snap_path(tag):
    return os.path.join(SNAPDIR, "snap_%s.json" % tag)


def _collect(dur, node_name):
    """采样 dur 秒, 返回 (位置数组, 力矩数组)。只订阅。"""
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    q, tau = [], []

    class Sub(Node):
        def __init__(self):
            super().__init__(node_name)
            self.create_subscription(JointState, "/left_joint_states", self.cb, 10)

        def cb(self, m):
            if len(m.position) >= 12:
                q.append(list(m.position[:12]))
                tau.append(list(m.effort[:12]) if len(m.effort) >= 12 else [0.0] * 12)

    rclpy.init()
    n = Sub()
    t0 = time.time()
    while time.time() - t0 < dur:
        rclpy.spin_once(n, timeout_sec=0.1)
    n.destroy_node()
    rclpy.shutdown()
    return np.array(q), np.array(tau)


def cmd_snap(tag, dur=5.0):
    q, tau = _collect(dur, "latch_snap")
    if not len(q):
        sys.exit("没收到关节数据! armcontrol 在跑吗?")
    os.makedirs(SNAPDIR, exist_ok=True)
    mu, sd = q.mean(0), q.std(0)
    json.dump({"tag": tag, "t": time.time(), "n": len(q),
               "q": [round(float(v), 6) for v in mu],
               "std": [round(float(v), 6) for v in sd],
               "tau_absmax": [round(float(v), 3) for v in np.abs(tau).max(0)]},
              open(_snap_path(tag), "w"), indent=1)
    print("快照 %s  (%d 帧, %.1f s)" % (tag, len(q), dur))
    print("%-8s %11s %9s %9s" % ("关节", "位置rad", "抖动std", "|tau|max"))
    for i, nm in enumerate(NM):
        print("%-8s %11.5f %9.5f %9.2f" % (nm, mu[i], sd[i], np.abs(tau[:, i]).max()))
    m = np.abs(tau).max()
    print("\nmax|tau| = %.2f N·m -> %s" % (m, "自由(可徒手搬动)" if m < 1.0 else "有力矩, 关节被撑住!"))
    print("存至 %s" % _snap_path(tag))


def cmd_diff(ta, tb):
    A, B = (json.load(open(_snap_path(t))) for t in (ta, tb))
    qa, qb = np.array(A["q"]), np.array(B["q"])
    d = qb - qa
    print("A = %s    B = %s" % (A["tag"], B["tag"]))
    print("%-8s %11s %11s %10s %9s %8s  %s"
          % ("关节", "A", "B", "差rad", "差deg", "差LSB", "判定"))
    for i, nm in enumerate(NM):
        a = abs(d[i])
        v = "跳变!!" if a > 0.05 else ("可疑" if a > 0.01 else "一致")
        print("%-8s %11.5f %11.5f %10.5f %9.3f %8.0f  %s"
              % (nm, qa[i], qb[i], d[i], np.degrees(d[i]), d[i] / LSB, v))
    k = int(np.argmax(np.abs(d)))
    print("\n最大差异: %s  %.5f rad = %.3f deg" % (NM[k], d[k], np.degrees(d[k])))
    for th, lab in ((0.05, "3deg"), (0.01, "0.6deg")):
        print("超 %s 的关节: %s" % (lab, [NM[i] for i in range(12) if abs(d[i]) > th] or "无"))


def _load(csv):
    rows, marks = [], []
    with open(csv) as f:
        f.readline()
        for ln in f:
            p = ln.rstrip("\n").split(",")
            if len(p) < 26:
                continue
            rows.append([float(p[0])] + [float(v) for v in p[1:13]])
            marks.append(p[25])
    return np.array(rows), marks


def cmd_jump(csv, th=0.025):
    a, marks = _load(csv)
    t, q = a[:, 0] - a[0, 0], a[:, 1:13]
    d = np.diff(q, axis=0)
    print("总帧 %d, 时长 %.1f s, 阈值 %.3f rad" % (len(a), t[-1], th))
    hits = []
    for i in range(12):
        for k in np.where(np.abs(d[:, i]) > th)[0]:
            hits.append((t[k + 1], NM[i], q[k, i], q[k + 1, i], d[k, i],
                         t[k + 1] - t[k], marks[k + 1]))
    hits.sort()
    print("%-9s %-7s %10s %10s %9s %8s %7s %7s %s"
          % ("t(s)", "关节", "前", "后", "跳变rad", "跳变deg", "LSB", "dt(s)", "阶段"))
    for h in hits[:80]:
        print("%-9.3f %-7s %10.5f %10.5f %9.5f %8.3f %7.0f %7.4f %s"
              % (h[0], h[1], h[2], h[3], h[4], np.degrees(h[4]), h[4] / LSB, h[5], h[6]))
    print("\n共 %d 处" % len(hits))
    gap = np.diff(t)
    i = int(np.argmax(gap))
    print("采样间隔: 中位 %.4f s, 最大 %.3f s (t=%.1f, 通常是节点重启/断电空窗)"
          % (np.median(gap), gap.max(), t[i]))


def cmd_record(csv):
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    os.makedirs(os.path.dirname(os.path.abspath(csv)), exist_ok=True)
    f = open(csv, "w", buffering=1)
    f.write("t_wall," + ",".join("q_" + n for n in NM) + "," +
            ",".join("tau_" + n for n in NM) + ",mark\n")
    cnt = [0]

    class Rec(Node):
        def __init__(self):
            super().__init__("latch_record")
            self.create_subscription(JointState, "/left_joint_states", self.cb, 10)

        def cb(self, m):
            if len(m.position) < 12:
                return
            mark = ""
            if os.path.exists(MARKFILE):
                try:
                    mark = open(MARKFILE).read().strip().replace(",", ";")
                except OSError:
                    pass
            f.write("%.3f," % time.time() +
                    ",".join("%.4f" % v for v in m.position[:12]) + "," +
                    ",".join("%.3f" % v for v in (m.effort[:12] if len(m.effort) >= 12
                                                  else [0.0] * 12)) + "," + mark + "\n")
            cnt[0] += 1

    rclpy.init()
    n = Rec()
    print("记录中 -> %s  (阶段标记写入 %s)" % (csv, MARKFILE))
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        print("共 %d 帧" % cnt[0])
        f.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    c, rest = sys.argv[1], sys.argv[2:]
    if c == "snap":
        cmd_snap(rest[0], float(rest[1]) if len(rest) > 1 else 5.0)
    elif c == "diff":
        cmd_diff(rest[0], rest[1])
    elif c == "jump":
        cmd_jump(rest[0], float(rest[1]) if len(rest) > 1 else 0.025)
    elif c == "record":
        cmd_record(rest[0])
    else:
        sys.exit(__doc__)
