#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""走路工况全 12 关节: 实机 target/real 与 Isaac(吊起来)sim 跟踪对比。
用法: python3 plot_walk_replay_all.py <sim2real.csv> <SIM_walk.csv> [out.png]
"""
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JN = ["L1-HipPitch", "L2-HipRoll", "L3-HipYaw", "L4-Knee", "L5-AnkPitch", "L6-AnkRoll",
      "R1-HipPitch", "R2-HipRoll", "R3-HipYaw", "R4-Knee", "R5-AnkPitch", "R6-AnkRoll"]


def read(p):
    r = list(csv.DictReader(open(p)))
    return {k: np.array([float(x[k]) for x in r]) for k in r[0]}


def gain_lag_rmse(tt, tgt, tq, q):
    """gain=幅度比, lag=互相关滞后(ms), rmse(mrad); 都相对 target。"""
    qi = np.interp(tt, tq, q)
    g0 = tgt - tgt.mean(); r0 = qi - qi.mean()
    gain = r0.std() / (g0.std() + 1e-9)
    dt = np.median(np.diff(tt)); n = len(tt)
    lag = (np.argmax(np.correlate(r0, g0, "full")) - (n - 1)) * dt * 1000
    rmse = np.sqrt(np.mean((qi - tgt) ** 2)) * 1000
    return gain, lag, rmse


def main():
    real_csv, sim_csv = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else "walk_replay_all.png"
    R = read(real_csv); S = read(sim_csv)
    tr = R["t"] - R["t"][0]; ts = S["t"] - S["t"][0]

    fig, ax = plt.subplots(6, 2, figsize=(17, 18), sharex=True)
    print(f"{'joint':<13}{'real gain/lag/rmse':>26}{'sim gain/lag/rmse':>26}")
    summ = []
    for i in range(12):
        r, c = i % 6, i // 6
        a = ax[r, c]
        tgt = R[f"cmd{i}"]; real = R[f"q{i}"]; sim = S[f"q{i}"]
        a.plot(tr, tgt, "k--", lw=1.3, label="target (real cmd)", zorder=5)
        a.plot(tr, real, "-", color="tab:blue", lw=1.0, label="real")
        a.plot(ts, sim, "-", color="tab:red", lw=1.0, label="sim (hung, legs.py cfg)")
        gr, lr, er = gain_lag_rmse(tr, tgt, tr, real)
        gs, ls, es = gain_lag_rmse(tr, tgt, ts, sim)
        summ.append((JN[i], gr, lr, er, gs, ls, es))
        a.set_title(f"{JN[i]}   real g{gr:.2f}/{lr:+.0f}ms/{er:.0f}mrad  |  sim g{gs:.2f}/{ls:+.0f}ms/{es:.0f}mrad",
                    fontsize=8.5)
        a.grid(alpha=0.3)
        if i == 0:
            a.legend(fontsize=8, loc="upper right")
        print(f"{JN[i]:<13}{f'{gr:.2f}/{lr:+.0f}/{er:.0f}':>26}{f'{gs:.2f}/{ls:+.0f}/{es:.0f}':>26}")
    for c in range(2):
        ax[5, c].set_xlabel("time [s]")
    fig.suptitle("Walking joint tracking: real (on ground) vs Isaac sim (hung, identified legs.py config)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print("saved", out)


if __name__ == "__main__":
    main()
