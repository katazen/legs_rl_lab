#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DCMotor 参数扫描 vs 真机 出图。每个 (vlim,sat) 组合一张图: 正弦按频率分 3 子图,
三条线 target/real/sim-DC, 标 real 与 sim 的分频 gain/lag。末尾打印汇总表(对照真机)。
文件名带 vlim 和 sat(dcv{vlim}s{sat}), 不同参数互不覆盖, 全部保留。
用法: python3 sysid/src/plot_dc_sweep.py <session> --joint 3 --kp 250
"""
import argparse, csv, os, glob, re
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SINE_SEG = [(0, 12, "0.5 Hz"), (12, 18, "1.0 Hz"), (18, 22, "1.5 Hz"), (22, 25, "2.0 Hz")]
JNAME = {3: "L4 knee", 9: "R4 knee"}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session"); ap.add_argument("--joint", type=int, default=3); ap.add_argument("--kp", default="250")
    a = ap.parse_args()
    ddir = os.path.join(a.session, "data"); pdir = os.path.join(a.session, "png"); os.makedirs(pdir, exist_ok=True)
    J = a.joint
    cf = glob.glob(f"{ddir}/j{J}_kp{a.kp}_kd*_sine_cmd.csv")[0]
    tag = os.path.basename(cf)[:-len("_cmd.csv")]
    cmd = read(cf); real = read(f"{ddir}/{tag}_state.csv")
    ex = cmd["phase"] == "excite"; tex0 = cmd["t"][ex][0]
    # 找所有 dcv{vlim}s{sat} 组合(新命名); 兼容旧命名 dcv{vlim}(无 sat)
    combos = []  # (vlim, sat_or_None, path)
    for f in glob.glob(f"{ddir}/SIM_{tag}_dcv*_state.csv"):
        b = os.path.basename(f)
        m = re.search(r"_dcv([0-9.]+)s([0-9.]+)_state", b)
        if m:
            combos.append((float(m.group(1)), float(m.group(2)), f))
        else:
            m2 = re.search(r"_dcv([0-9.]+)_state", b)
            if m2:
                combos.append((float(m2.group(1)), None, f))
    combos.sort(key=lambda x: (x[1] or 0, x[0]))
    tbl = [f"\n{'vlim':>5} {'sat':>4} | {'sim 0.5Hz':>9} {'1.0Hz':>6} {'1.5Hz':>6}   (real: 0.99 / 0.94 / 0.68)",
           "-" * 58]
    for vlim, sat, f in combos:
        sim = read(f)
        slab = f"s{sat:g}" if sat is not None else "s?"
        fig, ax = plt.subplots(len(SINE_SEG), 1, figsize=(13, 3.7 * len(SINE_SEG))); row = []
        for i, (lo, hi, fl) in enumerate(SINE_SEG):
            t0, t1 = tex0 + lo, tex0 + hi
            mc = ex & (cmd["t"] >= t0) & (cmd["t"] < t1)
            tt, gg = cmd["t"][mc], cmd[f"qd{J}"][mc]
            mm = (cmd["t"] >= t0) & (cmd["t"] <= t1)
            ax[i].plot(cmd["t"][mm], cmd[f"qd{J}"][mm], "k--", lw=1.8, label="target", zorder=5)
            for d, c, lb in [(real, "tab:blue", "real"), (sim, "tab:red", f"sim DC vlim={vlim:g} sat={sat:g}" if sat else f"sim DC vlim={vlim:g}")]:
                m = (d["t"] >= t0) & (d["t"] <= t1)
                ax[i].plot(d["t"][m], d[f"q{J}"][m], "-", color=c, lw=1.5, label=lb)
            gr, lr = gain_lag(tt, gg, real["t"], real[f"q{J}"])
            gs, ls = gain_lag(tt, gg, sim["t"], sim[f"q{J}"])
            ax[i].set_title(f"{fl}   real: gain {gr:.2f} lag {lr:.0f}ms  |  sim: gain {gs:.2f} lag {ls:.0f}ms", fontsize=11)
            ax[i].grid(alpha=.3); ax[i].legend(loc="upper right", fontsize=9); ax[i].set_ylabel("pos [rad]")
            row.append(gs)
        ax[-1].set_xlabel("t [s]")
        fig.suptitle(f"DCMotor vlim={vlim:g} sat={(sat if sat else 0):g}  —  {JNAME.get(J, J)} kp={a.kp} (no delay)", fontsize=13)
        plt.tight_layout()
        out = f"{pdir}/dc_v{vlim:g}{slab}_j{J}_kp{a.kp}.png"; plt.savefig(out, dpi=110); plt.close()
        print(f"  {out}")
        tbl.append(f"{vlim:>5g} {(sat if sat else 0):>4g} | {row[0]:>9.2f} {row[1]:>6.2f} {row[2]:>6.2f}")
    print("\n".join(tbl))


if __name__ == "__main__":
    main()
