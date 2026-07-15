#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""走路膝目标喂进 DelayedDCMotor 的跟踪对比图。2 子图(L4/R4), 三线 target/real/sim。
用法: python3 sysid/src/plot_walk_replay.py <session> --suffix ddcv2.3s26d8 --kp 300
"""
import argparse, csv, os, glob
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

JNAME = {3: "L4 knee (left)", 9: "R4 knee (right)"}


def read(p):
    rows = list(csv.DictReader(open(p)))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != "phase"}


def gain_lag(t, g, tq, q):
    r = np.interp(t, tq, q); g0, r0 = g - g.mean(), r - r.mean()
    gain = r0.std() / (g0.std() + 1e-9)
    dt = np.median(np.diff(t)); n = len(t)
    lag = (np.argmax(np.correlate(r0, g0, mode="full")) - (n - 1)) * dt * 1000
    return gain, lag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session"); ap.add_argument("--suffix", required=True); ap.add_argument("--kp", default="300")
    a = ap.parse_args()
    ddir = os.path.join(a.session, "data"); pdir = os.path.join(a.session, "png"); os.makedirs(pdir, exist_ok=True)
    joints = [3, 9]
    fig, ax = plt.subplots(len(joints), 1, figsize=(15, 4.2 * len(joints)))
    for i, J in enumerate(joints):
        tag = f"j{J}_kp{a.kp}_kd5_walk"
        cmd = read(f"{ddir}/{tag}_cmd.csv"); real = read(f"{ddir}/{tag}_state.csv")
        sim = read(f"{ddir}/SIM_{tag}_{a.suffix}_state.csv")
        ax[i].plot(cmd["t"], cmd[f"qd{J}"], "k--", lw=1.7, label="target (walk cmd)", zorder=5)
        ax[i].plot(real["t"], real[f"q{J}"], "-", color="tab:blue", lw=1.5, label="real (walk)")
        ax[i].plot(sim["t"], sim[f"q{J}"], "-", color="tab:red", lw=1.5, label="sim (DelayedDCMotor)")
        gr, lr = gain_lag(cmd["t"], cmd[f"qd{J}"], real["t"], real[f"q{J}"])
        gs, ls = gain_lag(cmd["t"], cmd[f"qd{J}"], sim["t"], sim[f"q{J}"])
        ax[i].set_title(f"{JNAME[J]}   real: gain {gr:.2f} lag {lr:.0f}ms  |  sim: gain {gs:.2f} lag {ls:.0f}ms", fontsize=12)
        ax[i].set_ylabel("pos [rad]"); ax[i].grid(alpha=.3); ax[i].legend(loc="upper right", fontsize=9)
    ax[-1].set_xlabel("t [s]")
    fig.suptitle(f"Walking knee tracking replayed through DelayedDCMotor ({a.suffix}, kp={a.kp}, hung)", fontsize=13)
    plt.tight_layout()
    out = f"{pdir}/walk_ddc_knees.png"; plt.savefig(out, dpi=115); plt.close(); print(f"  {out}")


if __name__ == "__main__":
    main()
