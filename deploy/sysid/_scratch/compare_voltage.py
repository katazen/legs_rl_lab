#!/usr/bin/env python3
"""24V vs 48V 部署日志对比: 关节速度上限/力矩-速度包络/跟踪质量。"""
import csv
import sys

import numpy as np

# effort 通道漏乘 kLegFbSign(arm_control_node.cpp), 带符号分析前先翻正
KSIGN = np.array([+1, +1, -1, +1, -1, +1, -1, +1, -1, -1, +1, +1], float)
NAMES = ["L1hipP", "L2hipR", "L3hipY", "L4knee", "L5ankP", "L6ankR",
         "R1hipP", "R2hipR", "R3hipY", "R4knee", "R5ankP", "R6ankR"]


def load(path):
    rows = list(csv.DictReader(open(path)))
    def col(fmt, n=12):
        return np.array([[float(r[fmt.format(i)]) for i in range(n)] for r in rows])
    d = {
        "t": np.array([float(r["t"]) for r in rows]),
        "cmd_vx": np.array([float(r["cmd_vx"]) for r in rows]),
        "cmd_yaw": np.array([float(r["cmd_yaw"]) for r in rows]),
        "w": np.array([[float(r[k]) for k in ("wx", "wy", "wz")] for r in rows]),
        "g": np.array([[float(r[k]) for k in ("gx", "gy", "gz")] for r in rows]),
        "q": col("q{}"), "qd": col("qd{}"), "mvel": col("mvel{}"),
        "tau": col("tau{}") * KSIGN, "cmd": col("cmd{}"),
    }
    return d


def summarize(tag, d):
    t = d["t"]
    dt = np.diff(t)
    moving = np.abs(d["cmd_vx"]) > 0.05
    print(f"\n===== {tag} =====")
    print(f"时长 {t[-1]:.1f}s  帧率~{1/np.median(dt):.1f}Hz  "
          f"cmd_vx范围 [{d['cmd_vx'].min():.2f},{d['cmd_vx'].max():.2f}]  "
          f"行走占比 {moving.mean()*100:.0f}%")
    print(f"姿态: |gx|p95 {np.percentile(np.abs(d['g'][:,0]),95):.3f}  "
          f"|gy|p95 {np.percentile(np.abs(d['g'][:,1]),95):.3f}  "
          f"wx std {d['w'][:,0].std():.2f}  wy std {d['w'][:,1].std():.2f} rad/s")
    print(f"{'joint':>7} {'|v|p99':>7} {'|v|max':>7} {'|tau|p99':>8} {'|tau|max':>8} "
          f"{'跟踪RMSE':>9} {'跟踪p99':>8}")
    for j in range(12):
        v = d["mvel"][:, j]
        tau = d["tau"][:, j]
        err = d["cmd"][:, j] - d["q"][:, j]
        print(f"{NAMES[j]:>7} {np.percentile(np.abs(v),99):>7.2f} {np.abs(v).max():>7.2f} "
              f"{np.percentile(np.abs(tau),99):>8.2f} {np.abs(tau).max():>8.2f} "
              f"{np.sqrt((err**2).mean()):>9.4f} {np.percentile(np.abs(err),99):>8.4f}")
    return moving


def envelope(tag, d, joints=(0, 3, 6, 9)):
    """高速区力矩可用性: 按速度分箱看同向力矩的 p90 (驱动象限 v·tau>0)。"""
    print(f"-- {tag} 驱动象限包络 (v 分箱 -> 同向|tau| p90) --")
    bins = np.arange(0, 8.5, 1.0)
    header = "  ".join(f"{lo:.0f}-{hi:.0f}" for lo, hi in zip(bins[:-1], bins[1:]))
    print(f"{'joint':>7}  {header}")
    for j in joints:
        v, tau = d["mvel"][:, j], d["tau"][:, j]
        drive = v * tau > 0
        cells = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = drive & (np.abs(v) >= lo) & (np.abs(v) < hi)
            cells.append(f"{np.percentile(np.abs(tau[m]), 90):4.1f}" if m.sum() > 20 else "   -")
        print(f"{NAMES[j]:>7}  " + "  ".join(cells))


f1, f2 = sys.argv[1], sys.argv[2]
d1, d2 = load(f1), load(f2)
summarize(f1.split('/')[-1], d1)
summarize(f2.split('/')[-1], d2)
envelope(f1.split('/')[-1], d1)
envelope(f2.split('/')[-1], d2)
