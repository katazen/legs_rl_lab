#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DelayedDCMotor 结果出图: 左右膝各 2 张(阶跃1张=正反2子图, 正弦1张=3频率3子图)。
每子图 3 条线: target / real(真机) / sim(DelayedDCMotor)。英文标签。
用法: python3 sysid/src/plot_ddc.py <session> --suffix ddcv2.3s26d8 --kp 250
"""
import argparse, csv, os, glob
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SINE_SEG = [(0, 12, "0.5 Hz"), (12, 18, "1.0 Hz"), (18, 22, "1.5 Hz"), (22, 25, "2.0 Hz")]
JNAME = {0: "L1 hip-pitch", 6: "R1 hip-pitch", 1: "L2 hip-roll", 7: "R2 hip-roll",
         2: "L3 hip-yaw", 8: "R3 hip-yaw", 3: "L4 knee (left)", 9: "R4 knee (right)",
         4: "L5 ankle-pitch", 10: "R5 ankle-pitch", 5: "L6 ankle-roll", 11: "R6 ankle-roll"}


def read(p):
    rows = list(csv.DictReader(open(p)))
    d = {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != "phase"}
    d["phase"] = np.array([r["phase"] for r in rows], dtype=object)
    return d


def gain_lag(t, g, tq, q):
    r = np.interp(t, tq, q); g0, r0 = g - g.mean(), r - r.mean()
    gain = r0.std() / (g0.std() + 1e-9)
    dt = np.median(np.diff(t)); n = len(t)
    lag = (np.argmax(np.correlate(r0, g0, mode="full")) - (n - 1)) * dt * 1000
    return gain, lag


def lines(ax, cmd, real, sim, J, t0=None, t1=None):
    def mask(t):
        return np.ones_like(t, bool) if t0 is None else ((t >= t0) & (t <= t1))
    ax.plot(cmd["t"][mask(cmd["t"])], cmd[f"qd{J}"][mask(cmd["t"])], "k--", lw=1.8, label="target", zorder=5)
    ax.plot(real["t"][mask(real["t"])], real[f"q{J}"][mask(real["t"])], "-", color="tab:blue", lw=1.5, label="real")
    ax.plot(sim["t"][mask(sim["t"])], sim[f"q{J}"][mask(sim["t"])], "-", color="tab:red", lw=1.5, label="sim (DelayedDCMotor)")
    ax.grid(alpha=.3); ax.legend(loc="upper right", fontsize=9); ax.set_ylabel("pos [rad]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session"); ap.add_argument("--suffix", required=True, help="SIM 文件的参数后缀, 如 ddcv2.3s26d8")
    ap.add_argument("--joints", default="3,9"); ap.add_argument("--kp", default="*")
    a = ap.parse_args()
    ddir = os.path.join(a.session, "data"); pdir = os.path.join(a.session, "png"); os.makedirs(pdir, exist_ok=True)
    for J in [int(x) for x in a.joints.split(",")]:
        cf = glob.glob(f"{ddir}/j{J}_kp{a.kp}_kd*_sine_cmd.csv")[0]
        tag = os.path.basename(cf)[:-len("_cmd.csv")]
        kp_lbl = tag.split("_kp")[1].split("_kd")[0]
        title = f"{JNAME.get(J, J)}  kp={kp_lbl}  |  sim = DelayedDCMotor ({a.suffix})  vs real"
        # ---- 正弦: 频率子图 ----
        cmd = read(cf); real = read(f"{ddir}/{tag}_state.csv"); sim = read(f"{ddir}/SIM_{tag}_{a.suffix}_state.csv")
        ex = cmd["phase"] == "excite"; tex0 = cmd["t"][ex][0]
        fig, ax = plt.subplots(len(SINE_SEG), 1, figsize=(13, 3.7 * len(SINE_SEG)))
        for i, (lo, hi, fl) in enumerate(SINE_SEG):
            t0, t1 = tex0 + lo, tex0 + hi
            lines(ax[i], cmd, real, sim, J, t0, t1)
            m = ex & (cmd["t"] >= t0) & (cmd["t"] < t1); tt, gg = cmd["t"][m], cmd[f"qd{J}"][m]
            gr, lr = gain_lag(tt, gg, real["t"], real[f"q{J}"]); gs, ls = gain_lag(tt, gg, sim["t"], sim[f"q{J}"])
            ax[i].set_title(f"{fl}   real: gain {gr:.2f} lag {lr:.0f}ms  |  sim: gain {gs:.2f} lag {ls:.0f}ms", fontsize=11)
        ax[-1].set_xlabel("t [s]"); fig.suptitle("Sine by frequency — " + title, fontsize=13)
        plt.tight_layout(); out = f"{pdir}/ddc_sine_j{J}_{a.suffix}.png"; plt.savefig(out, dpi=110); plt.close(); print(f"  {out}")

        # ---- 阶跃: 正/反 并排 ----
        fig, ax = plt.subplots(1, 2, figsize=(18, 6))
        for k, (mode, lab) in enumerate([("step_fwd", "Step forward"), ("step_rev", "Step reverse")]):
            cfs = glob.glob(f"{ddir}/j{J}_kp{a.kp}_kd*_{mode}_cmd.csv")
            if not cfs:
                ax[k].set_title(f"{lab} (no data)"); ax[k].grid(alpha=.3); continue
            tg = os.path.basename(cfs[0])[:-len("_cmd.csv")]
            c = read(cfs[0]); r = read(f"{ddir}/{tg}_state.csv"); s = read(f"{ddir}/SIM_{tg}_{a.suffix}_state.csv")
            lines(ax[k], c, r, s, J); ax[k].set_title(lab, fontsize=12); ax[k].set_xlabel("t [s]")
        fig.suptitle("Step response — " + title, fontsize=13)
        plt.tight_layout(); out = f"{pdir}/ddc_step_j{J}_{a.suffix}.png"; plt.savefig(out, dpi=110); plt.close(); print(f"  {out}")


if __name__ == "__main__":
    main()
