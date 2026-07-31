#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""踝 pitch sim/real 对比图: 验证 sim 配置(库仑摩擦)是否复现实机的变幅非线性。
左列 FRF: 实机(实线)vs sim(虚线), 三档幅值叠加 —— 看 sim 是否复现"低频增益随幅值升、
表观 fn 随幅值升"的描述函数特征。右列时域跟踪: 选低/中频段, cmd vs real vs sim 叠加。

用法:
  python3 plot_sim_real_ankle.py <idx> <freqs逗号> <cycles> <kp> <out.png> \
      <amp标签>:<cmd>,<real_state>,<sim_state> ...
例:
  python3 plot_sim_real_ankle.py 4 0.3,0.5,0.7,0.9,1.1,1.4,1.8,2.4,3.2 10 40 out.png \
      a03:.../j4..a03_cmd.csv,.../j4..a03_state.csv,.../SIM_j4..a03_fc0.6_state.csv ...
"""
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from analyze_stepsine import _read_cols, frf_stepsine, seg_bounds, excite_start

j = int(sys.argv[1])
freqs = [float(x) for x in sys.argv[2].split(",")]
cycles = float(sys.argv[3])
kp = float(sys.argv[4])
out = sys.argv[5]
# 每档: label:cmd,real_state,sim_state
groups = []
for a in sys.argv[6:]:
    lbl, paths = a.split(":", 1)
    cmd, rs, ss = paths.split(",")
    groups.append((lbl, cmd, rs, ss))

TD_FREQS = [0.5, 1.4]   # 时域展示的两个频段
fig, ax = plt.subplots(2, 2, figsize=(15, 9))
cmap = plt.get_cmap("viridis")
cols = [cmap(x) for x in np.linspace(0.05, 0.75, len(groups))]

print(f"=== idx{j} kp{kp} sim/real 对比 ===")
for (lbl, cmd, rs, ss), col in zip(groups, cols):
    c = _read_cols(cmd, [f"qd{j}", "t"])
    r = _read_cols(rs, [f"q{j}", "t"])
    s = _read_cols(ss, [f"q{j}", "t"])
    t0c = excite_start(c["t"], c.get("phase"))
    t0r = excite_start(r["t"], r.get("phase"))
    t0s = excite_start(s["t"], s.get("phase"))
    fcr, gr, lagr = frf_stepsine(c["t"], c[f"qd{j}"], r["t"], r[f"q{j}"], freqs, cycles, 2.0, t0c, t0r)
    fcs, gs, lags = frf_stepsine(c["t"], c[f"qd{j}"], s["t"], s[f"q{j}"], freqs, cycles, 2.0, t0c, t0s)
    print(f"[{lbl}] real低频增益{gr[0]:.3f} 峰{gr.max():.3f}@{fcr[np.argmax(gr)]:.2f} | "
          f"sim低频增益{gs[0]:.3f} 峰{gs.max():.3f}@{fcs[np.argmax(gs)]:.2f}")
    ax[0, 0].semilogx(fcr, gr, "-o", ms=5, color=col, label=f"real {lbl}")
    ax[0, 0].semilogx(fcs, gs, "--s", ms=4, color=col, alpha=.8, label=f"sim {lbl}")
    ax[1, 0].semilogx(fcr, lagr, "-o", ms=5, color=col, label=f"real {lbl}")
    ax[1, 0].semilogx(fcs, lags, "--s", ms=4, color=col, alpha=.8)

    # 时域: 两个频段各画一小窗(丢前2周期后取~2.5周期)
    bnd = seg_bounds(freqs, cycles)
    for axi, ftd in zip((ax[0, 1], ax[1, 1]), TD_FREQS):
        iseg = int(np.argmin([abs(f - ftd) for f in freqs])); fseg = freqs[iseg]
        lo = bnd[iseg] + 2.0 / fseg; hi = min(bnd[iseg + 1], lo + 2.5 / fseg)
        lc = c["t"] - t0c; lr = r["t"] - t0r; ls = s["t"] - t0s
        mc = (lc >= lo) & (lc < hi); mr = (lr >= lo) & (lr < hi); ms = (ls >= lo) & (ls < hi)
        base = c[f"qd{j}"][mc].mean()
        if lbl == groups[-1][0]:   # 命令只画一次(最大幅值那档), 其余档命令形状相同仅幅值不同
            pass
        axi.plot((lc[mc] - lo) * fseg, c[f"qd{j}"][mc] - base, ":", color=col, lw=1.0, alpha=.5)
        axi.plot((lr[mr] - lo) * fseg, r[f"q{j}"][mr] - base, "-", color=col, lw=1.4, label=f"real {lbl}")
        axi.plot((ls[ms] - lo) * fseg, s[f"q{j}"][ms] - base, "--", color=col, lw=1.4, alpha=.8, label=f"sim {lbl}")
        axi.set_title(f"time tracking @ {fseg:g}Hz  (dotted=cmd)")
        axi.set_xlabel("cycles"); axi.set_ylabel("q - base [rad]"); axi.grid(alpha=.3)

ax[0, 0].axhline(1.0, color="k", lw=0.5); ax[0, 0].axhline(0.707, color="gray", ls=":", lw=1)
ax[0, 0].set_ylabel("gain |q/target|"); ax[0, 0].grid(alpha=.3, which="both")
ax[0, 0].legend(fontsize=7, ncol=2); ax[0, 0].set_title(f"idx{j} FRF gain  real(solid) vs sim(dashed)")
ax[1, 0].axhline(0, color="k", lw=0.5); ax[1, 0].set_ylabel("phase lag [deg]")
ax[1, 0].set_xlabel("frequency [Hz]"); ax[1, 0].grid(alpha=.3, which="both"); ax[1, 0].legend(fontsize=7)
ax[0, 1].legend(fontsize=7); ax[1, 1].legend(fontsize=7)
fig.suptitle(f"ankle pitch idx{j}: sim(Coulomb) vs real amplitude-dependence", fontsize=13)
fig.tight_layout(); fig.savefig(out, dpi=120); print("saved", out)
