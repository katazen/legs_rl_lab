#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DelayedDCMotor vlim 扫描出图(拟合用). 对每个关节:
  - 每个 vlim 一张正弦分频图(target/real/sim);
  - 一张 gain-vs-vlim 汇总(各频率 sim gain 随 vlim, 叠真机水平线), 一眼挑最贴的 vlim。
用法: python3 sysid/src/plot_ddc_sweep.py <session> --joints 0,6 --kp 200
"""
import argparse, csv, os, glob, re
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SINE_SEG = [(0, 12, "0.5 Hz"), (12, 18, "1.0 Hz"), (18, 22, "1.5 Hz"), (22, 25, "2.0 Hz")]
JNAME = {0: "L1 hip-pitch", 6: "R1 hip-pitch", 3: "L4 knee", 9: "R4 knee"}


def read(p):
    r = list(csv.DictReader(open(p)))
    d = {k: np.array([float(x[k]) for x in r]) for k in r[0] if k != "phase"}
    d["phase"] = np.array([x["phase"] for x in r], dtype=object)
    return d


def gain_lag(t, g, tq, q):
    r = np.interp(t, tq, q); g0, r0 = g - g.mean(), r - r.mean()
    gg = r0.std() / (g0.std() + 1e-9)
    dt = np.median(np.diff(t)); n = len(t)
    lag = (np.argmax(np.correlate(r0, g0, mode="full")) - (n - 1)) * dt * 1000
    return gg, lag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session"); ap.add_argument("--joints", default="0,6"); ap.add_argument("--kp", default="200")
    a = ap.parse_args()
    ddir = os.path.join(a.session, "data"); pdir = os.path.join(a.session, "png"); os.makedirs(pdir, exist_ok=True)
    for J in [int(x) for x in a.joints.split(",")]:
        cf = glob.glob(f"{ddir}/j{J}_kp{a.kp}_kd*_sine_cmd.csv")[0]
        tag = os.path.basename(cf)[:-len("_cmd.csv")]
        cmd = read(cf); real = read(f"{ddir}/{tag}_state.csv")
        ex = cmd["phase"] == "excite"; tex0 = cmd["t"][ex][0]
        sims = {}
        for f in glob.glob(f"{ddir}/SIM_{tag}_ddcv*s*d*_state.csv"):
            v = float(re.search(r"ddcv([0-9.]+)s", os.path.basename(f)).group(1)); sims[v] = f
        # 真机各频 gain
        rg = []
        for lo, hi, _ in SINE_SEG:
            m = ex & (cmd["t"] >= tex0 + lo) & (cmd["t"] < tex0 + hi)
            rg.append(gain_lag(cmd["t"][m], cmd[f"qd{J}"][m], real["t"], real[f"q{J}"])[0])
        # 每个 vlim 一张正弦分频图
        summ = {f: [] for _, _, f in SINE_SEG}
        for v in sorted(sims):
            sim = read(sims[v])
            fig, ax = plt.subplots(len(SINE_SEG), 1, figsize=(13, 3.6 * len(SINE_SEG)))
            for i, (lo, hi, fl) in enumerate(SINE_SEG):
                t0, t1 = tex0 + lo, tex0 + hi
                for d, c, lb in [(cmd, "k", "target"), (real, "tab:blue", "real"), (sim, "tab:red", f"sim vlim={v:g}")]:
                    m = (d["t"] >= t0) & (d["t"] <= t1)
                    col = f"qd{J}" if lb == "target" else f"q{J}"
                    ax[i].plot(d["t"][m], d[col][m], ("k--" if lb == "target" else "-"),
                               color=(None if lb == "target" else c), lw=(1.7 if lb == "target" else 1.5), label=lb)
                mm = ex & (cmd["t"] >= t0) & (cmd["t"] < t1)
                gr, _ = gain_lag(cmd["t"][mm], cmd[f"qd{J}"][mm], real["t"], real[f"q{J}"])
                gs, _ = gain_lag(cmd["t"][mm], cmd[f"qd{J}"][mm], sim["t"], sim[f"q{J}"])
                summ[fl].append(gs)
                ax[i].set_title(f"{fl}   real gain {gr:.2f}  |  sim gain {gs:.2f}", fontsize=11)
                ax[i].grid(alpha=.3); ax[i].legend(loc="upper right", fontsize=9); ax[i].set_ylabel("pos [rad]")
            ax[-1].set_xlabel("t [s]")
            fig.suptitle(f"{JNAME.get(J,J)} kp={a.kp}  DelayedDCMotor vlim={v:g} sat26 delay8", fontsize=13)
            plt.tight_layout(); out = f"{pdir}/hipfit_j{J}_vlim{v:g}.png"; plt.savefig(out, dpi=110); plt.close(); print(f"  {out}")
        # gain-vs-vlim 汇总
        vs = sorted(sims)
        fig, ax = plt.subplots(figsize=(8, 5.2))
        for k, (_, _, fl) in enumerate(SINE_SEG):
            ax.plot(vs, summ[fl], "o-", label=f"sim {fl}")
            ax.axhline(rg[k], ls=":", color=ax.lines[-1].get_color(), alpha=.7)
        ax.set_xlabel("velocity_limit (vlim)"); ax.set_ylabel("tracking gain")
        ax.set_title(f"{JNAME.get(J,J)} kp={a.kp}: sim gain vs vlim (dotted = real target)")
        ax.grid(alpha=.3); ax.legend(fontsize=8)
        plt.tight_layout(); out = f"{pdir}/hipfit_gain_vs_vlim_j{J}.png"; plt.savefig(out, dpi=120); plt.close(); print(f"  {out}")


if __name__ == "__main__":
    main()
