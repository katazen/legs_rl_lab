#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""膝关节辨识细分对比图(英文标签, 规避 CJK 方框)。每个被测膝的每个 kp 出两张:
  sine_j{J}_kp{KP}.png : 正弦, 按频率分 3 个子图(0.5 / 1.0 / 1.5 Hz), 每子图 target/real/Isaac + 分频 gain/lag。
  step_j{J}_kp{KP}.png : 正、反阶跃并排一张, 每子图 target/real/Isaac。
用法: python3 sysid/src/plot_kp_detail.py sysid/data/real/knee_kp_sweep_<...>
"""
import argparse, csv, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JNAME = {0: "L1 hip-pitch", 6: "R1 hip-pitch", 1: "L2 hip-roll", 7: "R2 hip-roll",
         2: "L3 hip-yaw", 8: "R3 hip-yaw", 3: "L4 knee", 9: "R4 knee",
         4: "L5 ankle-pitch", 10: "R5 ankle-pitch"}
# 正弦扫频窗口(相对 excite 起点): 0.5Hz 6周期=12s, 1Hz=6s, 1.5Hz=4s, 2Hz=3s
SINE_SEG = [(0, 12, "0.5 Hz"), (12, 18, "1.0 Hz"), (18, 22, "1.5 Hz"), (22, 25, "2.0 Hz")]


def read(p):
    rows = list(csv.DictReader(open(p)))
    ph = np.array([r["phase"] for r in rows], dtype=object)
    d = {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != "phase"}
    d["phase"] = ph
    return d


def gain_lag(t, g, tq, q):
    r = np.interp(t, tq, q)
    g0, r0 = g - g.mean(), r - r.mean()
    gain = r0.std() / (g0.std() + 1e-9)
    dt = np.median(np.diff(t)); n = len(t)
    lag = (np.argmax(np.correlate(r0, g0, mode="full")) - (n - 1)) * dt * 1000
    return gain, lag


def plot_line(ax, cmd, real, sim, J, t0=None, t1=None):
    def clip(d):
        if t0 is None:
            return d["t"], d
        m = (d["t"] >= t0) & (d["t"] <= t1)
        return d["t"][m], m
    tc = cmd["t"]; mc = np.ones_like(tc, bool)
    if t0 is not None:
        mc = (tc >= t0) & (tc <= t1)
    ax.plot(cmd["t"][mc], cmd[f"qd{J}"][mc], "k--", lw=1.8, label="target", zorder=5)
    mr = (real["t"] >= t0) & (real["t"] <= t1) if t0 is not None else np.ones_like(real["t"], bool)
    ax.plot(real["t"][mr], real[f"q{J}"][mr], "-", color="tab:blue", lw=1.5, label="real")
    if sim is not None:
        ms = (sim["t"] >= t0) & (sim["t"] <= t1) if t0 is not None else np.ones_like(sim["t"], bool)
        ax.plot(sim["t"][ms], sim[f"q{J}"][ms], "-", color="tab:red", lw=1.5, label="Isaac sim")
    ax.grid(alpha=.3); ax.legend(loc="upper right", fontsize=9); ax.set_ylabel("pos [rad]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session"); ap.add_argument("--joints", default="3,9")
    a = ap.parse_args()
    ddir = os.path.join(a.session, "data")
    pdir = os.path.join(a.session, "png"); os.makedirs(pdir, exist_ok=True)
    joints = [int(x) for x in a.joints.split(",")]
    n = 0
    for J in joints:
        kps = [os.path.basename(cf).split("_kp")[1].split("_kd")[0]
               for cf in sorted(glob.glob(os.path.join(ddir, f"j{J}_kp*_kd*_sine_cmd.csv")))]
        for kp in kps:
            title0 = f"{JNAME.get(J,'idx'+str(J))} (idx{J})  kp={kp} kd=5  (hung)"
            # ---------- 正弦: 按频率分子图 ----------
            cf = glob.glob(os.path.join(ddir, f"j{J}_kp{kp}_kd*_sine_cmd.csv"))[0]
            tag = os.path.basename(cf)[:-len("_cmd.csv")]
            cmd = read(cf); real = read(os.path.join(ddir, f"{tag}_state.csv"))
            simp = os.path.join(ddir, f"SIM_{tag}_state.csv")
            sim = read(simp) if os.path.exists(simp) else None
            ex = cmd["phase"] == "excite"; tex0 = cmd["t"][ex][0]
            fig, ax = plt.subplots(len(SINE_SEG), 1, figsize=(13, 3.7 * len(SINE_SEG)))
            for i, (lo, hi, fl) in enumerate(SINE_SEG):
                t0, t1 = tex0 + lo, tex0 + hi
                plot_line(ax[i], cmd, real, sim, J, t0, t1)
                # 分频指标
                mm = ex & (cmd["t"] >= t0) & (cmd["t"] < t1)
                tt, gg = cmd["t"][mm], cmd[f"qd{J}"][mm]
                sub = f"{fl}"
                if len(tt) > 10:
                    gr, lr = gain_lag(tt, gg, real["t"], real[f"q{J}"])
                    sub += f"   real: gain {gr:.2f}, lag {lr:.0f}ms"
                    if sim is not None:
                        gs, ls = gain_lag(tt, gg, sim["t"], sim[f"q{J}"])
                        sub += f"   |  sim: gain {gs:.2f}, lag {ls:.0f}ms"
                ax[i].set_title(sub, fontsize=11)
            ax[-1].set_xlabel("t [s]")
            fig.suptitle("Sine sweep by frequency — " + title0, fontsize=13)
            plt.tight_layout()
            out = os.path.join(pdir, f"sine_j{J}_kp{kp}.png"); plt.savefig(out, dpi=110); plt.close(); n += 1
            print(f"  {out}")

            # ---------- 阶跃: 正/反 并排 ----------
            fig, ax = plt.subplots(1, 2, figsize=(18, 6))
            for k, (mode, lab) in enumerate([("step_fwd", "Step forward"), ("step_rev", "Step reverse")]):
                cfs = glob.glob(os.path.join(ddir, f"j{J}_kp{kp}_kd*_{mode}_cmd.csv"))
                if not cfs:
                    ax[k].set_title(f"{lab} (no data)"); ax[k].grid(alpha=.3); continue
                tg = os.path.basename(cfs[0])[:-len("_cmd.csv")]
                c = read(cfs[0]); r = read(os.path.join(ddir, f"{tg}_state.csv"))
                sp = os.path.join(ddir, f"SIM_{tg}_state.csv")
                s = read(sp) if os.path.exists(sp) else None
                plot_line(ax[k], c, r, s, J)
                ax[k].set_title(lab, fontsize=12); ax[k].set_xlabel("t [s]")
            fig.suptitle("Step response — " + title0, fontsize=13)
            plt.tight_layout()
            out = os.path.join(pdir, f"step_j{J}_kp{kp}.png"); plt.savefig(out, dpi=110); plt.close(); n += 1
            print(f"  {out}")
    print(f"完成 {n} 张 -> {pdir}")


if __name__ == "__main__":
    main()
