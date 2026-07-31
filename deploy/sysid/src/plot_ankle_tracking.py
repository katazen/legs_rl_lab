#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""踝 pitch 最终配置 sim/real 时域跟踪对比: 频率(行)×幅值(列)网格, 每格叠 cmd/real/sim。
选 0.3(准静态)/0.5(步态)/0.9(共振)/1.4Hz(滚降)四个关键频段, 各取丢2周期后~2.5周期窗口。
用最终 clean 配置的 sim: arm0.0509 + viscous0 + 库仑(左L5=0.4/右R5=0.6)。

用法: python3 plot_ankle_tracking.py <idx> <fc> <out.png>
"""
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from analyze_stepsine import _read_cols, seg_bounds, excite_start

j = int(sys.argv[1]); fc = sys.argv[2]; out = sys.argv[3]
D = "/home/woan/workspace/legs_rl_lab/deploy/sysid/data/real/2026-07-29_ankle/data"
FREQS = [0.3, 0.5, 0.7, 0.9, 1.1, 1.4, 1.8, 2.4, 3.2]; CYC = 10
SHOW_F = [0.3, 0.5, 0.9, 1.4]          # rows: quasi-static / gait / resonance / rolloff
AMPS = [("a03", "small amp"), ("a07", "mid amp"), ("a14", "large amp")]  # cols
bnd = seg_bounds(FREQS, CYC)

fig, ax = plt.subplots(len(SHOW_F), len(AMPS), figsize=(15, 12), sharex=True)
for ci, (a, alab) in enumerate(AMPS):
    cmd = f"{D}/j{j}_kp40_kd2_sine_{a}_cmd.csv"
    rs = f"{D}/j{j}_kp40_kd2_sine_{a}_state.csv"
    ss = f"{D}/SIM_j{j}_kp40_kd2_sine_{a}_fc{fc}_v0_a0.0509_state.csv"
    c = _read_cols(cmd, [f"qd{j}", "t"]); r = _read_cols(rs, [f"q{j}", "t"]); s = _read_cols(ss, [f"q{j}", "t"])
    t0c = excite_start(c["t"], c.get("phase")); t0r = excite_start(r["t"], r.get("phase")); t0s = excite_start(s["t"], s.get("phase"))
    lc = c["t"] - t0c; lr = r["t"] - t0r; ls = s["t"] - t0s
    for ri, ftd in enumerate(SHOW_F):
        i = int(np.argmin([abs(f - ftd) for f in FREQS])); f = FREQS[i]
        lo = bnd[i] + 2.0 / f; hi = min(bnd[i + 1], lo + 2.5 / f)
        mc = (lc >= lo) & (lc < hi); mr = (lr >= lo) & (lr < hi); ms = (ls >= lo) & (ls < hi)
        base = c[f"qd{j}"][mc].mean()
        A = ax[ri, ci]
        A.plot((lc[mc] - lo) * f, c[f"qd{j}"][mc] - base, ":", color="0.5", lw=1.4, label="cmd")
        A.plot((lr[mr] - lo) * f, r[f"q{j}"][mr] - base, "-", color="#1b7837", lw=2.0, label="real")
        A.plot((ls[ms] - lo) * f, s[f"q{j}"][ms] - base, "--", color="#c51b7d", lw=2.0, label="sim")
        A.grid(alpha=.3); A.axhline(0, color="k", lw=0.4)
        if ci == 0: A.set_ylabel(f"{f:g}Hz\nq-base [rad]", fontsize=10)
        if ri == 0: A.set_title(f"{alab} ({a})", fontsize=11)
        if ri == 0 and ci == 0: A.legend(fontsize=9, loc="upper right")
        if ri == len(SHOW_F) - 1: A.set_xlabel("cycles")
side = "L5 ankle-pitch (idx4)" if j == 4 else "R5 ankle-pitch (idx10)"
fig.suptitle(f"{side}  sim vs real tracking [final cfg: arm0.0509 + Coulomb Fc={fc} + viscous0, kp40/kd2]", fontsize=13)
fig.tight_layout(); fig.savefig(out, dpi=110); print("saved", out)
