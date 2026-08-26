#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zoom plot explaining HOW the pure delay T_d was measured in Experiment 1.

Same criterion as analyze_step_bi.py:
    t_edge  = time the command level changes (0 -> A)
    t_onset = first time |v| exceeds max(0.02, 10% * peak|v|) after the edge
    T_d     = t_onset - t_edge

Shows one single small-amplitude rising edge, blown up, with both times marked.
English labels (CJK renders as boxes here).
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = os.path.join(os.path.dirname(__file__), "..", "doc", "stage1", "data")
PNG = os.path.join(os.path.dirname(__file__), "..", "doc", "stage1", "png")

STEM = "j0_kp200_kd5_step_step_bi"
J = 0
LABEL = "hip_pitch_L"
EDGE_PICK = 1          # which small-amp rising edge to show (0-based)
PRE, POST = 0.030, 0.120   # zoom window around the edge [s]


def load(stem, j):
    cr = list(csv.reader(open(os.path.join(DATA, stem + "_cmd.csv"))))
    sr = list(csv.reader(open(os.path.join(DATA, stem + "_state.csv"))))
    ci = {n: k for k, n in enumerate(cr[0])}
    si = {n: k for k, n in enumerate(sr[0])}
    cd, sd = cr[1:], sr[1:]
    return dict(
        ct=np.array([float(r[ci["t"]]) for r in cd]),
        cph=[r[ci["phase"]] for r in cd],
        qd=np.array([float(r[ci[f"qd{j}"]]) for r in cd]),
        st=np.array([float(r[si["t"]]) for r in sd]),
        q=np.array([float(r[si[f"q{j}"]]) for r in sd]),
        v=np.array([float(r[si[f"v{j}"]]) for r in sd]),
    )


d = load(STEM, J)
m = np.array([p == "excite" for p in d["cph"]])
ct, qd = d["ct"][m], d["qd"][m]
st, q, v = d["st"], d["q"], d["v"]

# --- detect 0 -> A rising edges, keep small amplitudes (linear region) ---
dq = np.diff(qd)
edges = []
for i in np.where(np.abs(dq) > 0.005)[0]:
    a_prev, a_new = round(qd[i], 3), round(qd[i + 1], 3)
    if abs(a_prev) < 0.02 and 0.02 < abs(a_new) < 0.15:
        edges.append((ct[i + 1], a_new))
te, amp = edges[EDGE_PICK]

# --- delay criterion, identical to analyze_step_bi.py ---
w = (st >= te - 0.05) & (st <= te + 0.5)
wt, wv = st[w], v[w]
vpk = np.max(np.abs(wv))
thr = max(0.02, 0.10 * vpk)
over = np.where(np.abs(wv) > thr)[0]
t_on = wt[over[0]]
Td = t_on - te

# --- zoom window ---
z = (st >= te - PRE) & (st <= te + POST)
zt, zq, zv = st[z], q[z], v[z]
zc = (ct >= te - PRE) & (ct <= te + POST)

fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

# top: position
ax[0].step(ct[zc] - te, qd[zc], "k--", where="post", lw=1.6, label="cmd q_des (step)")
ax[0].plot(zt - te, zq, "b.-", ms=6, lw=1.4, label="meas q (100Hz samples)")
ax[0].set_ylabel("position [rad]")
ax[0].set_title(f"Exp1 {LABEL}: how pure delay T_d is measured "
                f"(single {amp:+.2f} rad step, zoomed)")

# bottom: velocity + threshold
ax[1].plot(zt - te, np.abs(zv), "g.-", ms=6, lw=1.4, label="meas |v| (100Hz samples)")
ax[1].axhline(thr, color="orange", ls="-.", lw=1.4,
              label=f"onset threshold = max(0.02, 10%*peak) = {thr:.3f} rad/s")
ax[1].set_ylabel("|velocity| [rad/s]")
ax[1].set_xlabel("t - t_cmd_edge [s]")
ax[1].set_ylim(0, max(0.35, thr * 4))

# mark the two instants on both panels
for a in ax:
    a.axvline(0.0, color="k", lw=2.0)
    a.axvline(Td, color="r", lw=2.0, ls="--")
    a.grid(alpha=0.3)
    a.legend(fontsize=8, loc="upper left")

# annotate the delay bracket on the velocity panel
ybr = ax[1].get_ylim()[1] * 0.55
ax[1].annotate("", xy=(0.0, ybr), xytext=(Td, ybr),
               arrowprops=dict(arrowstyle="<->", color="r", lw=2.0))
ax[1].text(Td / 2, ybr * 1.10, f"T_d = {Td*1e3:.0f} ms", color="r",
           ha="center", fontsize=13, fontweight="bold")
ax[0].text(0.0, ax[0].get_ylim()[1] * 0.96, " (1) cmd edge", color="k",
           ha="left", va="top", fontsize=10)
ax[0].text(Td, ax[0].get_ylim()[1] * 0.96, " (2) motion onset", color="r",
           ha="left", va="top", fontsize=10)

fig.tight_layout()
out = os.path.join(PNG, "exp1_delay_zoom.png")
fig.savefig(out, dpi=120)
plt.close(fig)

print(f"edge #{EDGE_PICK}: t_edge={te:.4f}s amp={amp:+.3f} rad")
print(f"peak|v| in window = {vpk:.3f} rad/s -> threshold = {thr:.4f} rad/s")
print(f"t_onset = {t_on:.4f}s  ->  T_d = {Td*1e3:.1f} ms")
print("samples right after the edge (t-te [ms], |v|):")
for tt, vv in zip(zt, np.abs(zv)):
    if -0.005 <= tt - te <= 0.055:
        mark = "  <-- first over threshold" if abs(tt - t_on) < 1e-9 else ""
        print(f"   {(tt-te)*1e3:7.1f}   {vv:.4f}{mark}")
print("saved:", out)
