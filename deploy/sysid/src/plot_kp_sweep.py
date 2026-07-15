#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""膝关节 kp 扫描对比图 + 定量指标(英文标签, 规避 CJK 字体方框)。

对每个被测膝(默认 L4=idx3, R4=idx9), 把不同 kp 的跟踪叠在一起看:
  - sine: target(黑虚) + 各 kp 的 real(彩色), 一眼看幅值/相位随 kp 变化
  - step_fwd / step_rev: 各 kp 的 real 阶跃响应叠加
并算每个 kp 在 sine 激励段的定量指标:
  gain = std(real)/std(target)        (幅值跟踪比, 1=理想)
  lag  = 互相关峰值对应的时间滞后 [ms]
  rmse = sqrt(mean((real-target)^2))  [rad]
  step: 末端稳态误差 ss_err [rad]

输入: sweep 会话目录 sysid/data/real/knee_kp_sweep_<...>  (内含 data/*.csv)
输出: <session>/png/kp_sweep_j{J}.png  +  <session>/kp_sweep_summary.md
用法: python3 sysid/src/plot_kp_sweep.py sysid/data/real/knee_kp_sweep_<...>
"""
import argparse
import csv
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JNAME = {3: "L4 knee", 9: "R4 knee"}
MODES = ["sine", "step_fwd", "step_rev"]
FN = re.compile(r"j(\d+)_kp([0-9.eE+-]+|NA)_kd([0-9.eE+-]+|NA)_(step_fwd|step_rev|sine)_(cmd|state)\.csv$")


def read(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    ph = np.array([r["phase"] for r in rows], dtype=object)
    d = {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != "phase"}
    d["phase"] = ph
    return d


def sine_metrics(cmd, st, J):
    """在 excite 段: gain/lag/rmse。real 插值到 cmd 时间轴。"""
    m = cmd["phase"] == "excite"
    t = cmd["t"][m]
    tgt = cmd[f"qd{J}"][m]
    if len(t) < 10:
        return None
    real = np.interp(t, st["t"], st[f"q{J}"])
    tgt0, real0 = tgt - tgt.mean(), real - real.mean()
    gain = real0.std() / (tgt0.std() + 1e-9)
    rmse = float(np.sqrt(np.mean((real - tgt) ** 2)))
    # lag: 互相关(real 相对 target 滞后为正)
    dt = np.median(np.diff(t))
    n = len(t)
    xc = np.correlate(real0, tgt0, mode="full")
    lag_idx = np.argmax(xc) - (n - 1)
    lag_ms = lag_idx * dt * 1000.0
    return dict(gain=gain, rmse=rmse, lag_ms=lag_ms)


def step_ss_err(cmd, st, J):
    """阶跃末端 0.3s 的稳态误差(real-target 均值)。"""
    m = cmd["phase"] == "excite"
    if m.sum() < 5:
        return None
    t = cmd["t"][m]; tgt = cmd[f"qd{J}"][m]
    t_end = t[-1]
    sel = t >= (t_end - 0.3)
    real = np.interp(t[sel], st["t"], st[f"q{J}"])
    return float(np.mean(real - tgt[sel]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--joints", default="3,9")
    a = ap.parse_args()
    ddir = os.path.join(a.session, "data")
    pdir = os.path.join(a.session, "png"); os.makedirs(pdir, exist_ok=True)
    joints = [int(x) for x in a.joints.split(",")]

    # (J, kp, mode) -> base_tag
    g = {}
    for fn in os.listdir(ddir):
        m = FN.search(fn)
        if not m or m.group(5) != "cmd":
            continue
        J, kp, kd, mode = int(m.group(1)), m.group(2), m.group(3), m.group(4)
        g[(J, kp, mode)] = (fn[:-len("_cmd.csv")], kd)

    lines = ["# 膝关节 kp 扫描辨识小结\n",
             f"会话: `{a.session}`\n",
             "激励(左右统一): sine amp0.5 freqs[0.5,1,1.5] 6cyc, step ±0.15/0.3, 中心0.55, 吊起, kd=5\n",
             "\n| joint | kp | gain | lag[ms] | sine_rmse[rad] | step_fwd_ss[rad] | step_rev_ss[rad] |",
             "|---|---|---|---|---|---|---|"]

    for J in joints:
        kps = sorted({kp for (jj, kp, md) in g if jj == J}, key=lambda x: float(x) if x != "NA" else 0)
        if not kps:
            continue
        colors = plt.cm.viridis(np.linspace(0, 0.85, len(kps)))
        fig, ax = plt.subplots(1, 3, figsize=(21, 6))
        for mi, mode in enumerate(MODES):
            for kp, c in zip(kps, colors):
                if (J, kp, mode) not in g:
                    continue
                tag, kd = g[(J, kp, mode)]
                cmd = read(os.path.join(ddir, f"{tag}_cmd.csv"))
                st = read(os.path.join(ddir, f"{tag}_state.csv"))
                if kp == kps[0]:
                    ax[mi].plot(cmd["t"], cmd[f"qd{J}"], "k--", lw=1.4, label="target", zorder=5)
                ax[mi].plot(st["t"], st[f"q{J}"], "-", color=c, lw=1.3, label=f"kp{kp}")
            ax[mi].set_title(mode, fontsize=13)
            ax[mi].set_xlabel("t [s]"); ax[mi].grid(alpha=.3)
            ax[mi].legend(loc="upper right", fontsize=9)
        ax[0].set_ylabel("position [rad]")
        fig.suptitle(f"{JNAME.get(J, 'idx'+str(J))} (idx{J})  kp sweep  (kd=5, hung, center 0.55)", fontsize=15)
        plt.tight_layout()
        out = os.path.join(pdir, f"kp_sweep_j{J}.png")
        plt.savefig(out, dpi=110); plt.close()
        print(f"  {out}")

        for kp in kps:
            sm = ff = fr = None
            kd = "5"
            if (J, kp, "sine") in g:
                tag, kd = g[(J, kp, "sine")]
                sm = sine_metrics(read(os.path.join(ddir, f"{tag}_cmd.csv")),
                                  read(os.path.join(ddir, f"{tag}_state.csv")), J)
            if (J, kp, "step_fwd") in g:
                tag, kd = g[(J, kp, "step_fwd")]
                ff = step_ss_err(read(os.path.join(ddir, f"{tag}_cmd.csv")),
                                 read(os.path.join(ddir, f"{tag}_state.csv")), J)
            if (J, kp, "step_rev") in g:
                tag, kd = g[(J, kp, "step_rev")]
                fr = step_ss_err(read(os.path.join(ddir, f"{tag}_cmd.csv")),
                                 read(os.path.join(ddir, f"{tag}_state.csv")), J)
            gg = f"{sm['gain']:.3f}" if sm else "-"
            ll = f"{sm['lag_ms']:.0f}" if sm else "-"
            rr = f"{sm['rmse']:.4f}" if sm else "-"
            ffs = f"{ff:+.4f}" if ff is not None else "-"
            frs = f"{fr:+.4f}" if fr is not None else "-"
            lines.append(f"| {JNAME.get(J, J)} | {kp} | {gg} | {ll} | {rr} | {ffs} | {frs} |")

    md = os.path.join(a.session, "kp_sweep_summary.md")
    with open(md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n{md}\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
