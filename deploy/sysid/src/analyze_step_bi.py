#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze Experiment 1: bidirectional multi-amplitude position steps (+ torque channel).
Numpy + matplotlib only (system python3). English labels (CJK renders as boxes here).

For each of the 4 joints (hip pitch L/R idx0/6, knee L/R idx3/9):
  - detect 0->A step edges from the command channel
  - response onset   : time from cmd edge to velocity threshold (small amps, both dirs)
  - torque ceiling   : peak |tau| vs |A| (plateau = saturation)
  - velocity ceiling : peak |v|   vs |A| (plateau start; sampling-limited estimate)
  - Kp/J, (Kd+B)/J   : 2nd-order+delay fit to the bidirection-averaged small-amp step
  - L/R symmetry     : overlay + parameter compare

Outputs plots to sysid/doc/<stage>/png/ and prints a summary block for the doc.
"""
import argparse, csv, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--stage", default="stage1", help="doc 下的阶段目录")
ap.add_argument("--prefix", default="exp1", help="输出图文件名前缀")
args = ap.parse_args()
DATA = os.path.join(os.path.dirname(__file__), "..", "doc", args.stage, "data")
PNG  = os.path.join(os.path.dirname(__file__), "..", "doc", args.stage, "png")
os.makedirs(PNG, exist_ok=True)

JOINTS = [
    # label,            idx, file-stem,                         kp
    ("hip_pitch_L", 0, "j0_kp200_kd5_step_step_bi", 200.0),
    ("hip_pitch_R", 6, "j6_kp200_kd5_step_step_bi", 200.0),
    ("knee_L",      3, "j3_kp250_kd5_step_step_bi", 250.0),
    ("knee_R",      9, "j9_kp250_kd5_step_step_bi", 250.0),
]
AMPS = [0.05, 0.10, 0.20, 0.35]
SMALL = [0.05, 0.10]        # linear-region amplitudes used for the 2nd-order fit


def load(stem, j):
    cpath = os.path.join(DATA, stem + "_cmd.csv")
    spath = os.path.join(DATA, stem + "_state.csv")
    cr = list(csv.reader(open(cpath)))
    sr = list(csv.reader(open(spath)))
    ch, cd = cr[0], cr[1:]
    sh, sd = sr[0], sr[1:]
    ci = {n: k for k, n in enumerate(ch)}
    si = {n: k for k, n in enumerate(sh)}
    ct = np.array([float(r[ci["t"]]) for r in cd])
    cph = [r[ci["phase"]] for r in cd]
    qd = np.array([float(r[ci[f"qd{j}"]]) for r in cd])
    st = np.array([float(r[si["t"]]) for r in sd])
    sph = [r[si["phase"]] for r in sd]
    q  = np.array([float(r[si[f"q{j}"]]) for r in sd])
    v  = np.array([float(r[si[f"v{j}"]]) for r in sd])
    tau= np.array([float(r[si[f"tau{j}"]]) for r in sd])
    return dict(ct=ct, cph=cph, qd=qd, st=st, sph=sph, q=q, v=v, tau=tau)


def detect_edges(ct, cph, qd):
    """Return list of (t_edge, a_prev, a_new) for command level changes in excite phase."""
    m = np.array([p == "excite" for p in cph])
    ct, qd = ct[m], qd[m]
    edges = []
    dq = np.diff(qd)
    idx = np.where(np.abs(dq) > 0.005)[0]
    for i in idx:
        a_prev = round(qd[i], 3)
        a_new = round(qd[i + 1], 3)
        edges.append((ct[i + 1], a_prev, a_new))
    return edges


def seg(t, x, t0, t1):
    m = (t >= t0) & (t <= t1)
    return t[m], x[m]


def analyze_joint(label, j, stem, kp):
    d = load(stem, j)
    edges = detect_edges(d["ct"], d["cph"], d["qd"])
    # keep 0 -> A rising edges
    rising = [(te, an) for (te, ap, an) in edges if abs(ap) < 0.02 and abs(an) > 0.02]

    st, q, v, tau = d["st"], d["q"], d["v"], d["tau"]

    def match_amp(a):  # nearest nominal amplitude
        return min(AMPS, key=lambda x: abs(x - abs(a)))

    # ---- per-edge metrics ----
    peak_tau = {a: [] for a in AMPS}
    peak_v   = {a: [] for a in AMPS}
    delays   = []
    norm_curves = []   # (tg, y) resampled small-amp normalized steps for the fit
    tg = np.arange(0.0, 0.90, 0.005)

    for te, an in rising:
        a = match_amp(an)
        tt, vv = seg(st, v, te - 0.05, te + 0.6)
        _, tq = seg(st, tau, te, te + 0.25)
        if len(vv) and len(tq):
            peak_v[a].append(np.max(np.abs(vv)))
            peak_tau[a].append(np.max(np.abs(tq)))
        # delay from velocity onset (small amps, cleanest)
        if a in SMALL:
            wt, wv = seg(st, v, te - 0.05, te + 0.5)
            if len(wv) > 3:
                vpk = np.max(np.abs(wv))
                thr = max(0.02, 0.10 * vpk)
                over = np.where(np.abs(wv) > thr)[0]
                if len(over):
                    t_on = wt[over[0]]
                    if 0 <= t_on - te <= 0.2:
                        delays.append(t_on - te)
        # normalized step for fit (small amps only)
        if a in SMALL:
            bt, bq = seg(st, q, te - 0.15, te - 0.02)
            _stt, ss = seg(st, q, te + 1.0, te + 1.4)
            if len(bq) and len(ss):
                base = bq.mean(); settle = ss.mean(); amp = settle - base
                if np.sign(amp) == np.sign(an) and abs(amp) > 0.02:
                    wt, wq = seg(st, q, te, te + 0.89)
                    y = (wq - base) / amp
                    yg = np.interp(tg, wt - te, y, left=0.0, right=y[-1] if len(y) else 1.0)
                    norm_curves.append(yg)

    y_avg = np.mean(np.vstack(norm_curves), axis=0) if norm_curves else None

    # ---- 2nd-order + delay fit to averaged normalized step ----
    fit = None
    if y_avg is not None:
        fit = fit_second_order(tg, y_avg)

    d_mean = float(np.mean(delays)) if delays else float("nan")
    d_std  = float(np.std(delays)) if delays else float("nan")

    return dict(label=label, j=j, kp=kp, st=st, q=q, v=v, tau=tau,
                d=d, edges=edges, rising=rising,
                peak_tau={a: (np.mean(x) if x else np.nan) for a, x in peak_tau.items()},
                peak_v={a: (np.mean(x) if x else np.nan) for a, x in peak_v.items()},
                delay_mean=d_mean, delay_std=d_std, n_delay=len(delays),
                tg=tg, y_avg=y_avg, fit=fit)


def step_response(tg, wn, zeta, Td):
    """Unit step response of Kp*e^{-sTd}/(J s^2+(Kd+B)s+Kp), normalized DC=1."""
    y = np.zeros_like(tg)
    tau = tg - Td
    m = tau > 0
    t = tau[m]
    if zeta < 1.0:
        wd = wn * math.sqrt(1 - zeta * zeta)
        y[m] = 1 - np.exp(-zeta * wn * t) * (np.cos(wd * t) + (zeta * wn / wd) * np.sin(wd * t))
    elif abs(zeta - 1.0) < 1e-6:
        y[m] = 1 - np.exp(-wn * t) * (1 + wn * t)
    else:
        s = math.sqrt(zeta * zeta - 1)
        r1 = -wn * (zeta - s); r2 = -wn * (zeta + s)
        c1 = -r2 / (r1 - r2); c2 = r1 / (r1 - r2)
        y[m] = 1 + c1 * np.exp(r1 * t) + c2 * np.exp(r2 * t)
    return y


def fit_second_order(tg, y_avg):
    fitmask = tg <= 0.75
    wns = np.logspace(math.log10(3), math.log10(80), 90)
    zes = np.linspace(0.30, 2.6, 90)
    tds = np.arange(0.0, 0.045, 0.005)
    best = (1e9, None)
    for Td in tds:
        for wn in wns:
            for ze in zes:
                ym = step_response(tg, wn, ze, Td)
                e = np.sum((ym[fitmask] - y_avg[fitmask]) ** 2)
                if e < best[0]:
                    best = (e, (wn, ze, Td))
    wn, ze, Td = best[1]
    return dict(wn=wn, zeta=ze, Td=Td, sse=best[0],
                KpoverJ=wn * wn, KdBoverJ=2 * ze * wn)


# ============================ run + plots ============================
res = {r["label"]: r for r in (analyze_joint(*j) for j in JOINTS)}

# ---- per-joint tracking overview + step fit + delay zoom ----
for lab, r in res.items():
    d = r["d"]; st, q, v, tau = r["st"], r["q"], r["v"], r["tau"]
    m = np.array([p == "excite" for p in d["sph"]])
    # command resampled onto state time for overlay
    qd_on = np.interp(st, d["ct"], d["qd"])
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(st[m], qd_on[m], "k--", lw=0.8, label="cmd q_des")
    ax[0].plot(st[m], q[m], "b", lw=0.8, label="meas q")
    ax[0].set_ylabel("pos [rad]"); ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_title(f"Exp1  {lab}  (kp={r['kp']:g}, kd=5)  bidirectional step ladder")
    ax[1].plot(st[m], v[m], "g", lw=0.7); ax[1].set_ylabel("vel [rad/s]")
    ax[2].plot(st[m], tau[m], "r", lw=0.7)
    ax[2].axhline(30, color="k", ls=":", lw=0.7); ax[2].axhline(-30, color="k", ls=":", lw=0.7)
    ax[2].set_ylabel("torque [N.m]"); ax[2].set_xlabel("t [s]")
    fig.tight_layout(); fig.savefig(os.path.join(PNG, f"{args.prefix}_{lab}_tracking.png"), dpi=110)
    plt.close(fig)

    # step fit
    if r["fit"]:
        f = r["fit"]; tg = r["tg"]
        fig, a2 = plt.subplots(figsize=(7, 5))
        a2.plot(tg, r["y_avg"], "b", lw=1.6, label="measured (bidir-avg, small amp)")
        a2.plot(tg, step_response(tg, f["wn"], f["zeta"], f["Td"]), "r--", lw=1.4,
                label=f"fit wn={f['wn']:.1f} zeta={f['zeta']:.2f} Td={f['Td']*1e3:.0f}ms")
        a2.set_xlabel("t since cmd edge [s]"); a2.set_ylabel("normalized q")
        a2.set_title(f"Exp1 {lab}  2nd-order+delay fit\nKp/J={f['KpoverJ']:.0f}  (Kd+B)/J={f['KdBoverJ']:.1f}")
        a2.legend(fontsize=8); a2.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(PNG, f"{args.prefix}_{lab}_step_fit.png"), dpi=110)
        plt.close(fig)

# ---- peak torque / peak vel vs amplitude (all joints) ----
for metric, fn, ylab, title in [
    ("peak_tau", "peaktorque_vs_amp", "peak |torque| [N.m]", "Torque ceiling (saturation)"),
    ("peak_v",   "peakvel_vs_amp",    "peak |vel| [rad/s]",  "Velocity ceiling (sampled estimate)")]:
    fig, ax = plt.subplots(figsize=(7, 5))
    for lab, r in res.items():
        ys = [r[metric][a] for a in AMPS]
        ax.plot(AMPS, ys, "-o", label=lab)
    if metric == "peak_tau":
        ax.axhline(30, color="k", ls=":", label="TMAX=30")
    ax.set_xlabel("step amplitude |A| [rad]"); ax.set_ylabel(ylab)
    ax.set_title(f"Exp1  {title}"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(PNG, f"{args.prefix}_{fn}.png"), dpi=110)
    plt.close(fig)

# ---- L/R symmetry overlays ----
for pair, name in [(("hip_pitch_L", "hip_pitch_R"), "hip_pitch"),
                   (("knee_L", "knee_R"), "knee")]:
    fig, ax = plt.subplots(figsize=(7, 5))
    for lab in pair:
        r = res[lab]
        if r["fit"]:
            ax.plot(r["tg"], r["y_avg"], lw=1.5, label=f"{lab} (wn={r['fit']['wn']:.1f}, z={r['fit']['zeta']:.2f})")
    ax.set_xlabel("t since cmd edge [s]"); ax.set_ylabel("normalized q")
    ax.set_title(f"Exp1  L/R symmetry: {name}  (small-amp avg step)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(PNG, f"{args.prefix}_{name}_lr.png"), dpi=110)
    plt.close(fig)

# ============================ summary ============================
print("\n================= EXP1 SUMMARY =================")
hdr = f"{'joint':12s} {'kp':>4s} {'onset_ms':>10s} {'n':>3s} {'wn':>6s} {'zeta':>5s} {'Kp/J':>7s} {'(Kd+B)/J':>9s}"
print(hdr)
for lab, r in res.items():
    f = r["fit"]
    print(f"{lab:12s} {r['kp']:4.0f} {r['delay_mean']*1e3:5.1f}+-{r['delay_std']*1e3:4.1f} "
          f"{r['n_delay']:3d} {f['wn']:6.1f} {f['zeta']:5.2f} {f['KpoverJ']:7.0f} {f['KdBoverJ']:9.1f}")
print("\npeak torque [N.m] vs amplitude:")
print(f"{'joint':12s} " + " ".join(f"{a:>6.2f}" for a in AMPS))
for lab, r in res.items():
    print(f"{lab:12s} " + " ".join(f"{r['peak_tau'][a]:6.1f}" for a in AMPS))
print("\npeak vel [rad/s] vs amplitude:")
print(f"{'joint':12s} " + " ".join(f"{a:>6.2f}" for a in AMPS))
for lab, r in res.items():
    print(f"{lab:12s} " + " ".join(f"{r['peak_v'][a]:6.2f}" for a in AMPS))
# implied J = kp / (Kp/J)
print("\nimplied J = kp/(Kp/J) [kg.m^2]:")
for lab, r in res.items():
    print(f"  {lab:12s} J ~= {r['kp']/r['fit']['KpoverJ']:.4f}")
print("================================================\n")
