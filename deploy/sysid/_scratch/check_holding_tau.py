#!/usr/bin/env python3
"""Stage1B 稳态保持力矩分析: 区分 hip/knee DC 增益缺口来自重力还是摩擦。"""
import csv
import os

import numpy as np

DOC = os.path.join(os.path.dirname(__file__), "..", "doc")
JOINTS = [("hipL", 0, "j0_kp200_kd5_step_step_bi", 200.0),
          ("hipR", 6, "j6_kp200_kd5_step_step_bi", 200.0),
          ("kneeL", 3, "j3_kp250_kd5_step_step_bi", 250.0),
          ("kneeR", 9, "j9_kp250_kd5_step_step_bi", 250.0)]


def read(stem, j):
    root = os.path.join(DOC, "stage1b", "data", stem)
    with open(root + "_cmd.csv") as f:
        cmd = list(csv.DictReader(f))
    with open(root + "_state.csv") as f:
        state = list(csv.DictReader(f))
    return (np.array([float(r["t"]) for r in cmd]),
            np.array([r["phase"] for r in cmd]),
            np.array([float(r[f"qd{j}"]) for r in cmd]),
            np.array([float(r["t"]) for r in state]),
            np.array([float(r[f"q{j}"]) for r in state]),
            np.array([float(r[f"tau{j}"]) for r in state]))


for label, j, stem, kp in JOINTS:
    ct, cp, qd, st, q, tau = read(stem, j)
    m = cp == "excite"
    t, target = ct[m], qd[m]
    k = np.where(np.abs(np.diff(target)) > 0.005)[0]
    rows = {}
    for i in k:
        te, ap, an = t[i + 1], target[i], target[i + 1]
        if not (abs(ap) < 0.02 and 0.02 < abs(an) < 0.15):
            continue
        ss = (st >= te + 1.0) & (st <= te + 1.4)
        key = round(an, 2)
        rows.setdefault(key, []).append((q[ss].mean(), tau[ss].mean(),
                                         kp * (an - q[ss].mean())))
    # 0 位稳态 tau (回0段)
    zero_ss = []
    for i in k:
        te, ap, an = t[i + 1], target[i], target[i + 1]
        if abs(an) < 0.02 and abs(ap) > 0.02:
            ss = (st >= te + 1.0) & (st <= te + 1.4)
            zero_ss.append(tau[ss].mean())
    print(f"\n== {label} (kp={kp:g})  0位稳态tau均值 {np.mean(zero_ss):+.3f} N·m ==")
    print(f"{'target':>8} {'q_ss':>9} {'tau_ss(反馈)':>12} {'kp·(qd-q)(重建)':>15} {'err(rad)':>9}")
    for key in sorted(rows):
        arr = np.array(rows[key])
        print(f"{key:>8.2f} {arr[:,0].mean():>9.4f} {arr[:,1].mean():>12.3f} "
              f"{arr[:,2].mean():>15.3f} {key-arr[:,0].mean():>9.4f}")
