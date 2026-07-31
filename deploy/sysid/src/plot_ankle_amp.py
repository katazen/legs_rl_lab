#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""踝 pitch 变幅线性度叠图: 同一频率网格下多个激励幅值的 stepped-sine FRF 叠在一张图。
判据: 曲线重合 -> 传动柔性(LTI, 真实动力学); 小幅增益低/共振下移 -> 齿隙/死区(非线性伪惯量)。

用法:
  python3 plot_ankle_amp.py <idx> <freqs逗号> <cycles> <kp> <out.png> \
      <label>=<cmd.csv>,<state.csv>  [<label>=...]  ...
"""
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from analyze_stepsine import _read_cols, frf_stepsine, excite_start, fit_2nd_order_delay, _model

j = int(sys.argv[1])
freqs = [float(x) for x in sys.argv[2].split(",")]
cycles = float(sys.argv[3])
kp = float(sys.argv[4])
out = sys.argv[5]
curves = []
for a in sys.argv[6:]:
    lbl, paths = a.split("=", 1)
    curves.append((lbl, paths.split(",")))

fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
cmap = plt.get_cmap("viridis")
cols = [cmap(x) for x in np.linspace(0.05, 0.8, len(curves))]

print(f"=== joint idx {j}  kp={kp}  变幅线性度 ===")
for (lbl, paths), col in zip(curves, cols):
    cmd_csv, state_csv = paths
    c = _read_cols(cmd_csv, [f"qd{j}", "t"])
    s = _read_cols(state_csv, [f"q{j}", "t"])
    t0t = excite_start(c["t"], c.get("phase"))
    t0r = excite_start(s["t"], s.get("phase"))
    fc, g, lag = frf_stepsine(c["t"], c[f"qd{j}"], s["t"], s[f"q{j}"],
                              freqs, cycles, 2.0, t0t, t0r)
    ipk = int(np.argmax(g))
    print(f"\n[{lbl}] 低频增益{g[0]:.3f}  峰 {g[ipk]:.3f}@{fc[ipk]:.2f}Hz")
    for f, gg, ll in zip(fc, g, lag):
        print(f"    {f:6.2f}Hz  gain={gg:.3f}  lag={ll:6.1f}°")
    ax[0].semilogx(fc, g, "-o", ms=5, color=col, label=lbl)
    ax[1].semilogx(fc, lag, "-o", ms=5, color=col, label=lbl)
    if len(fc) >= 4:
        J, cc, td = fit_2nd_order_delay(fc, g, lag, kp)
        wn = np.sqrt(kp / J); zeta = cc / (2 * np.sqrt(kp * J))
        print(f"    -> fit: J={J:.4f}  c={cc:.2f}  τ={td*1000:.1f}ms  fn={wn/2/np.pi:.2f}Hz  ζ={zeta:.3f}")
        ff = np.logspace(np.log10(fc[0]), np.log10(fc[-1]), 200)
        Hm = _model(2*np.pi*ff, kp, J, cc, td)
        ax[0].semilogx(ff, np.abs(Hm), "--", color=col, lw=1, alpha=.6,
                       label=f"{lbl} fit fn={wn/2/np.pi:.2f} ζ={zeta:.2f}")
        ax[1].semilogx(ff, -np.degrees(np.angle(Hm)), "--", color=col, lw=1, alpha=.6)

ax[0].axhline(1.0, color="k", lw=0.5); ax[0].axhline(0.707, color="gray", ls=":", lw=1)
ax[0].set_ylabel("gain |q/target|"); ax[0].grid(alpha=.3, which="both"); ax[0].legend(fontsize=8)
ax[0].set_title(f"ankle pitch amplitude-linearity  idx {j}  (overlap=LTI compliance / shift=backlash)")
ax[1].axhline(0, color="k", lw=0.5); ax[1].set_ylabel("phase lag [deg]")
ax[1].set_xlabel("frequency [Hz]"); ax[1].grid(alpha=.3, which="both")
fig.tight_layout(); fig.savefig(out, dpi=120); print("\nsaved", out)
