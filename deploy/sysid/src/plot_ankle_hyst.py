#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""踝 pitch 低频迟滞可视化: 命令 qd -> 响应 q 的 Lissajous + 力矩 tau vs 关节偏差。
死区/齿隙 -> q-vs-qd 出现平段(命令走间隙时 q 不动); 摩擦 -> tau 迟滞方框。
用法: python3 plot_ankle_hyst.py <idx> <freqs> <cycles> <fseg> <kp> <out.png> <lbl>=<cmd>,<state> ...
"""
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from analyze_stepsine import _read_cols, seg_bounds, excite_start

j = int(sys.argv[1])
freqs = [float(x) for x in sys.argv[2].split(",")]
cycles = float(sys.argv[3])
fseg = float(sys.argv[4])      # 取哪个频率段做迟滞图
kp = float(sys.argv[5])
out = sys.argv[6]
curves = [(a.split("=", 1)[0], a.split("=", 1)[1].split(",")) for a in sys.argv[7:]]

iseg = freqs.index(fseg)
bounds = seg_bounds(freqs, cycles)
fig, ax = plt.subplots(1, 2, figsize=(13, 6))
cmap = plt.get_cmap("viridis"); cols = [cmap(x) for x in np.linspace(0.05, 0.8, len(curves))]

for (lbl, paths), col in zip(curves, cols):
    cmd_csv, state_csv = paths
    c = _read_cols(cmd_csv, [f"qd{j}", "t"])
    s = _read_cols(state_csv, [f"q{j}", "t", f"tau{j}"])
    lt = c["t"] - excite_start(c["t"], c.get("phase"))
    lr = s["t"] - excite_start(s["t"], s.get("phase"))
    lo = bounds[iseg] + 2.0 / fseg; hi = bounds[iseg + 1]   # 丢前2周期
    mc = (lt >= lo) & (lt < hi); ms = (lr >= lo) & (lr < hi)
    qd = c[f"qd{j}"][mc]; qd -= qd.mean()
    q = s[f"q{j}"][ms];  qc = q.mean(); q = q - qc
    tau = s[f"tau{j}"][ms]
    # 命令与响应时间轴不同, 用响应自身相位画 q-vs-qd 需插值命令到响应时刻
    qd_i = np.interp(lr[ms], lt[mc], c[f"qd{j}"][mc]); qd_i -= qd_i.mean()
    ax[0].plot(qd_i, q, "-", color=col, lw=1, alpha=.8, label=lbl)
    ax[1].plot(q, tau, "-", color=col, lw=1, alpha=.8, label=lbl)

lim = max(abs(np.array(ax[0].get_xlim())).max(), abs(np.array(ax[0].get_ylim())).max())
ax[0].plot([-lim, lim], [-lim, lim], "k:", lw=0.8, label="ideal q=qd")
ax[0].set_xlabel("commanded qd - mean [rad]"); ax[0].set_ylabel("measured q - mean [rad]")
ax[0].set_title(f"idx{j} Lissajous @ {fseg}Hz  (flat step = deadband/backlash)")
ax[0].grid(alpha=.3); ax[0].legend(fontsize=8); ax[0].set_aspect("equal", "box")
ax[1].set_xlabel("q - mean [rad]"); ax[1].set_ylabel(f"tau{j} [Nm]")
ax[1].set_title("torque vs deflection (loop area = friction/hysteresis)")
ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(out, dpi=120); print("saved", out)
