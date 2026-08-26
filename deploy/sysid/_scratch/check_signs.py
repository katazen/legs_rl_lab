#!/usr/bin/env python3
"""符号一致性体检: v vs dq/dt, tau vs PD 重建, 覆盖 stage1b 4 关节与 stage3a 踝关节。"""
import csv
import os

import numpy as np

DOC = os.path.join(os.path.dirname(__file__), "..", "doc")

CASES = [
    ("stage1b", "j0_kp200_kd5_step_step_bi", 0, 200.0, 5.0),
    ("stage1b", "j6_kp200_kd5_step_step_bi", 6, 200.0, 5.0),
    ("stage1b", "j3_kp250_kd5_step_step_bi", 3, 250.0, 5.0),
    ("stage1b", "j9_kp250_kd5_step_step_bi", 9, 250.0, 5.0),
    ("stage3a", "j4_kp40_kd2_sine_frf_ankle_small_r1", 4, 40.0, 2.0),
    ("stage3a", "j10_kp40_kd2_sine_frf_ankle_small_r1", 10, 40.0, 2.0),
    ("stage3a", "j5_kp40_kd0.5_sine_frf_ankle_small_r1", 5, 40.0, 0.5),
    ("stage3a", "j11_kp40_kd0.5_sine_frf_ankle_small_r1", 11, 40.0, 0.5),
    ("stage3a", "j1_kp100_kd5_sine_frf_hiproll_back30_r1", 1, 100.0, 5.0),
    ("stage3a", "j7_kp100_kd5_sine_frf_hiproll_back30_r1", 7, 100.0, 5.0),
]


def load(stage, stem, j):
    root = os.path.join(DOC, stage, "data", stem)
    with open(root + "_cmd.csv") as f:
        cmd = list(csv.DictReader(f))
    with open(root + "_state.csv") as f:
        state = list(csv.DictReader(f))
    ct = np.array([float(r["t"]) for r in cmd])
    qd = np.array([float(r[f"qd{j}"]) for r in cmd])
    st = np.array([float(r["t"]) for r in state])
    q = np.array([float(r[f"q{j}"]) for r in state])
    v = np.array([float(r[f"v{j}"]) for r in state])
    tau = np.array([float(r[f"tau{j}"]) for r in state])
    sp = np.array([r["phase"] for r in state])
    m = sp == "excite"
    return ct, qd, st[m], q[m], v[m], tau[m]


print(f"{'stage':>8} {'j':>3} | {'corr(v, dq/dt)':>14} | {'corr(tau, PD重建)':>16} "
      f"{'斜率':>7} | 判定")
for stage, stem, j, kp, kd in CASES:
    ct, qd, st, q, v, tau = load(stage, stem, j)
    dqdt = np.gradient(q, st)
    qd_i = np.interp(st, ct, qd)
    pd = kp * (qd_i - q) - kd * v
    cv = np.corrcoef(v, dqdt)[0, 1]
    mask = np.abs(pd) > 0.2          # 排除死区附近样本
    ctau = np.corrcoef(tau[mask], pd[mask])[0, 1]
    slope = np.polyfit(pd[mask], tau[mask], 1)[0]
    verdict = "OK" if (cv > 0.5 and ctau > 0.5) else "!! 异常"
    print(f"{stage:>8} {j:>3} | {cv:>14.3f} | {ctau:>16.3f} {slope:>7.3f} | {verdict}")
