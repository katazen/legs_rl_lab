#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Friction pre-check on Experiment 1 data (no new hardware run).
Decide whether stage-2 (low-speed / constant-torque friction id) is actually
needed for these 4 joints, or whether it degrades to a quick confirmation.

Coulomb-friction signatures (PDF fig.4 / fig.5):
  - normalized small-amp step curves do NOT overlap across amplitudes
    (smaller amp is slower, larger steady-state error)  -> friction present
  - DC gain (settle/A) < 1 and gets worse for smaller A -> friction present
Linear region check uses ONLY 0.05 vs 0.10 (0.20/0.35 are velocity-saturated,
which also breaks overlap but for a different reason).
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = os.path.join(os.path.dirname(__file__), "..", "doc", "stage1", "data")
PNG  = os.path.join(os.path.dirname(__file__), "..", "doc", "stage1", "png")

JOINTS = [
    ("hip_pitch_L", 0, "j0_kp200_kd5_step_step_bi"),
    ("hip_pitch_R", 6, "j6_kp200_kd5_step_step_bi"),
    ("knee_L",      3, "j3_kp250_kd5_step_step_bi"),
    ("knee_R",      9, "j9_kp250_kd5_step_step_bi"),
]
LIN = [0.05, 0.10]   # linear-region amps only


def load(stem, j):
    cr = list(csv.reader(open(os.path.join(DATA, stem + "_cmd.csv"))))
    sr = list(csv.reader(open(os.path.join(DATA, stem + "_state.csv"))))
    ci = {n: k for k, n in enumerate(cr[0])}
    si = {n: k for k, n in enumerate(sr[0])}
    ct = np.array([float(r[ci["t"]]) for r in cr[1:]])
    cph = [r[ci["phase"]] for r in cr[1:]]
    qd = np.array([float(r[ci[f"qd{j}"]]) for r in cr[1:]])
    st = np.array([float(r[si["t"]]) for r in sr[1:]])
    q  = np.array([float(r[si[f"q{j}"]]) for r in sr[1:]])
    return ct, cph, qd, st, q


def seg(t, x, t0, t1):
    m = (t >= t0) & (t <= t1)
    return t[m], x[m]


def edges(ct, cph, qd):
    m = np.array([p == "excite" for p in cph])
    ct, qd = ct[m], qd[m]
    out = []
    idx = np.where(np.abs(np.diff(qd)) > 0.005)[0]
    for i in idx:
        out.append((ct[i + 1], round(qd[i], 3), round(qd[i + 1], 3)))
    return out


print("\n=== EXP1 friction pre-check (DC gain by amplitude & direction) ===")
print(f"{'joint':12s} {'dir':>4s} {'A':>5s} {'DCgain':>7s} {'n':>3s}")
tg = np.arange(0.0, 0.90, 0.005)
overlays = {}
for lab, j, stem in JOINTS:
    ct, cph, qd, st, q = load(stem, j)
    ed = edges(ct, cph, qd)
    rising = [(te, an) for (te, ap, an) in ed if abs(ap) < 0.02 and abs(an) > 0.02]
    # DC gain per (amp, sign)
    bucket = {}
    norm = {a: [] for a in LIN}
    for te, an in rising:
        a = min(LIN + [0.20, 0.35], key=lambda x: abs(x - abs(an)))
        bt, bq = seg(st, q, te - 0.15, te - 0.02)
        _, ss = seg(st, q, te + 1.0, te + 1.4)
        if not len(bq) or not len(ss):
            continue
        base, settle = bq.mean(), ss.mean()
        amp = settle - base
        gain = amp / an
        sgn = "+" if an > 0 else "-"
        bucket.setdefault((abs(a) if a in LIN else a, sgn), []).append(gain)
        if a in LIN and abs(amp) > 0.02 and np.sign(amp) == np.sign(an):
            wt, wq = seg(st, q, te, te + 0.89)
            y = (wq - base) / amp
            norm[a].append(np.interp(tg, wt - te, y, left=0.0, right=y[-1]))
    for (a, sgn) in sorted(bucket):
        g = bucket[(a, sgn)]
        print(f"{lab:12s} {sgn:>4s} {a:5.2f} {np.mean(g):7.3f} {len(g):3d}")
    overlays[lab] = {a: (np.mean(np.vstack(v), 0) if v else None) for a, v in norm.items()}

# overlay 0.05 vs 0.10 normalized curves per joint (friction => no overlap)
fig, axs = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
for ax, (lab, j, stem) in zip(axs.ravel(), JOINTS):
    for a, c in zip(LIN, ["tab:blue", "tab:red"]):
        y = overlays[lab][a]
        if y is not None:
            ax.plot(tg, y, color=c, lw=1.4, label=f"A={a}")
    ax.axhline(1.0, color="k", ls=":", lw=0.7)
    ax.set_title(lab); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_xlabel("t since edge [s]"); ax.set_ylabel("normalized q")
fig.suptitle("Exp1 friction pre-check: A=0.05 vs 0.10 normalized step overlap\n"
             "(overlap + DC~1 => linear-region friction negligible)")
fig.tight_layout()
out = os.path.join(PNG, "exp1_friction_precheck.png")
fig.savefig(out, dpi=110)
print(f"\nsaved overlay -> {out}")
print("Read: if the two curves sit on top of each other and level at ~1.0,")
print("Coulomb friction is negligible in the linear region -> stage-2 = quick confirm.\n")
