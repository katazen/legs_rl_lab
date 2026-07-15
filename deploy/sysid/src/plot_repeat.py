#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同一电机两次膝辨识的重复性对比: run1 vs run2, 同 kp 同频率的真机 gain/lag 并排 + 叠图。
每个膝一张图: 按频率分子图, 每子图 target(黑虚) + run1 real(蓝) + run2 real(橙)。
用法: python3 sysid/src/plot_repeat.py <run1_session> <run2_session> --kp 250 --joints 3,9
"""
import argparse, csv, os, glob
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SEG = [(0, 12, "0.5 Hz"), (12, 18, "1.0 Hz"), (18, 22, "1.5 Hz"), (22, 25, "2.0 Hz")]
JNAME = {3: "L4 knee", 9: "R4 knee"}


def read(p):
    rows = list(csv.DictReader(open(p)))
    d = {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != "phase"}
    d["phase"] = np.array([r["phase"] for r in rows], dtype=object)
    return d


def gain_lag(t, g, tq, q):
    r = np.interp(t, tq, q); g0, r0 = g - g.mean(), r - r.mean()
    if g0.std() < 1e-6:
        return float("nan"), float("nan")
    gain = r0.std() / (g0.std() + 1e-9)
    dt = np.median(np.diff(t)); n = len(t)
    lag = (np.argmax(np.correlate(r0, g0, mode="full")) - (n - 1)) * dt * 1000
    return gain, lag


def load(session, J, kp):
    fs = glob.glob(f"{session}/data/j{J}_kp{kp}_kd*_sine_cmd.csv")
    if not fs:
        return None
    tag = os.path.basename(fs[0])[:-len("_cmd.csv")]
    return read(fs[0]), read(f"{session}/data/{tag}_state.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run1"); ap.add_argument("run2")
    ap.add_argument("--kp", default="250"); ap.add_argument("--joints", default="3,9")
    a = ap.parse_args()
    pdir = os.path.join(a.run2, "png"); os.makedirs(pdir, exist_ok=True)
    tbl = [f"\n重复性 run1 vs run2  kp={a.kp}   (gain / lag[ms])", "-" * 64]
    for J in [int(x) for x in a.joints.split(",")]:
        d1 = load(a.run1, J, a.kp); d2 = load(a.run2, J, a.kp)
        if d1 is None or d2 is None:
            print(f"j{J}: 缺文件"); continue
        (c1, s1), (c2, s2) = d1, d2
        ex1 = c1["phase"] == "excite"; t01 = c1["t"][ex1][0]
        ex2 = c2["phase"] == "excite"; t02 = c2["t"][ex2][0]
        fig, ax = plt.subplots(len(SEG), 1, figsize=(13, 3.2 * len(SEG)))
        for i, (lo, hi, fl) in enumerate(SEG):
            # run1
            m1 = ex1 & (c1["t"] >= t01 + lo) & (c1["t"] < t01 + hi)
            m2 = ex2 & (c2["t"] >= t02 + lo) & (c2["t"] < t02 + hi)
            has1, has2 = m1.sum() > 10, m2.sum() > 10
            if has1:
                w = (c1["t"] >= t01 + lo) & (c1["t"] <= t01 + hi)
                ax[i].plot(c1["t"][w] - t01, c1[f"qd{J}"][w], "k--", lw=1.6, label="target", zorder=5)
                ax[i].plot(s1["t"][(s1["t"] >= t01 + lo) & (s1["t"] <= t01 + hi)] - t01,
                           s1[f"q{J}"][(s1["t"] >= t01 + lo) & (s1["t"] <= t01 + hi)], "-", color="tab:blue", lw=1.4, label="run1")
                g1, l1 = gain_lag(c1["t"][m1], c1[f"qd{J}"][m1], s1["t"], s1[f"q{J}"])
            else:
                g1 = l1 = float("nan")
            if has2:
                ax[i].plot(s2["t"][(s2["t"] >= t02 + lo) & (s2["t"] <= t02 + hi)] - t02,
                           s2[f"q{J}"][(s2["t"] >= t02 + lo) & (s2["t"] <= t02 + hi)], "-", color="tab:orange", lw=1.4, label="run2")
                if not has1:  # run1 无此频(2Hz), 用 run2 的 target
                    w = (c2["t"] >= t02 + lo) & (c2["t"] <= t02 + hi)
                    ax[i].plot(c2["t"][w] - t02, c2[f"qd{J}"][w], "k--", lw=1.6, label="target", zorder=5)
                g2, l2 = gain_lag(c2["t"][m2], c2[f"qd{J}"][m2], s2["t"], s2[f"q{J}"])
            else:
                g2 = l2 = float("nan")
            ax[i].set_title(f"{fl}   run1: gain {g1:.2f} lag {l1:.0f}ms  |  run2: gain {g2:.2f} lag {l2:.0f}ms", fontsize=11)
            ax[i].grid(alpha=.3); ax[i].legend(loc="upper right", fontsize=9); ax[i].set_ylabel("pos [rad]")
            tbl.append(f"{JNAME[J]} {fl}: run1 {g1:.2f}/{l1:.0f}   run2 {g2:.2f}/{l2:.0f}   Δgain {abs(g1-g2):.2f}"
                       if has1 and has2 else f"{JNAME[J]} {fl}: run1 {'-' if not has1 else f'{g1:.2f}'}  run2 {g2:.2f}/{l2:.0f} (新增)")
        ax[-1].set_xlabel("t within segment [s]")
        fig.suptitle(f"Repeatability run1 vs run2 — {JNAME[J]} kp={a.kp}", fontsize=13)
        plt.tight_layout(); out = f"{pdir}/repeat_j{J}_kp{a.kp}.png"; plt.savefig(out, dpi=110); plt.close(); print(f"  {out}")
    print("\n".join(tbl))


if __name__ == "__main__":
    main()
