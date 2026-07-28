#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分频驻留正弦: 按频率分子图画 target/real/sim 时域跟踪曲线, 验证 sim 参数对不对得上实机。
每个频率取稳态若干周期(丢前1周期暂态), 叠 target(黑)/real(红)/sim(蓝), 标各自跟踪幅值比。
用法:
  python3 plot_sine_tracking.py <joint_idx> <freqs逗号> <cycles> <out.png> \
      real_cmd=<cmd.csv> real_state=<state.csv> sim=<sim_replay.csv> [show_cycles=3]
"""
import sys, csv
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt


def rd(path, cols):
    r = list(csv.DictReader(open(path)))
    out = {c: np.array([float(x[c]) for x in r]) for c in cols}
    if "phase" in r[0]:
        out["phase"] = np.array([x["phase"] for x in r])
    return out


def excite_t0(d):
    if "phase" in d:
        m = d["phase"] == "excite"
        return d["t"][m][0] if m.any() else d["t"][0]
    return d["t"][0]


def seg_bounds(freqs, cycles):
    return np.concatenate([[0.0], np.cumsum([cycles / f for f in freqs])])


def amp_of(t, y, f):
    """在频率 f 上拟合正弦幅值(去均值)。"""
    y = y - y.mean()
    A = np.vstack([np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return np.hypot(a, b)


def main():
    j = int(sys.argv[1])
    freqs = [float(x) for x in sys.argv[2].split(",")]
    cycles = float(sys.argv[3])
    out = sys.argv[4]
    show_cycles = 3.0
    P = {}
    for a in sys.argv[5:]:
        k, v = a.split("=", 1)
        if k == "show_cycles": show_cycles = float(v)
        else: P[k] = v

    rc = rd(P["real_cmd"], [f"qd{j}", "t"]); rc_t0 = excite_t0(rc)
    rs = rd(P["real_state"], [f"q{j}", "t"]); rs_t0 = excite_t0(rs)
    sm = rd(P["sim"], ["t", "target", "q"]); sm_t0 = excite_t0(sm)
    lt_c = rc["t"] - rc_t0; lt_s = rs["t"] - rs_t0; lt_m = sm["t"] - sm_t0

    bounds = seg_bounds(freqs, cycles)
    n = len(freqs)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.4 * n))
    if n == 1: axes = [axes]
    for i, f in enumerate(freqs):
        lo, hi = bounds[i], bounds[i + 1]
        w0 = lo + 1.0 / f                      # 丢第1个周期暂态
        w1 = min(hi, w0 + show_cycles / f)     # 展示 show_cycles 个周期
        ax = axes[i]
        mc = (lt_c >= w0) & (lt_c < w1); ms = (lt_s >= w0) & (lt_s < w1); mm = (lt_m >= w0) & (lt_m < w1)
        # target 用实机命令
        ax.plot(lt_c[mc], rc[f"qd{j}"][mc], "k-", lw=1.2, label="target")
        ax.plot(lt_s[ms], rs[f"q{j}"][ms], "r-", lw=1.6, label="real")
        ax.plot(lt_m[mm], sm["q"][mm], "b--", lw=1.6, label="sim")
        # 幅值比(整段稳态, 不只展示窗)
        segc = (lt_c >= w0) & (lt_c < hi); segs = (lt_s >= w0) & (lt_s < hi); segm = (lt_m >= w0) & (lt_m < hi)
        At = amp_of(lt_c[segc], rc[f"qd{j}"][segc], f)
        Ar = amp_of(lt_s[segs], rs[f"q{j}"][segs], f)
        Am = amp_of(lt_m[segm], sm["q"][segm], f)
        ax.set_title(f"{f:g} Hz   跟踪幅值比  real={Ar/At:.3f}  sim={Am/At:.3f}   (target幅值 {At:.3f}rad)",
                     fontsize=10)
        ax.set_ylabel("angle [rad]"); ax.grid(alpha=.3)
        if i == 0: ax.legend(loc="upper right", ncol=3, fontsize=9)
        print(f"[{f:g}Hz] target={At:.3f}  real gain={Ar/At:.3f}  sim gain={Am/At:.3f}  差={Ar/At-Am/At:+.3f}")
    axes[-1].set_xlabel("excite-local time [s]")
    fig.suptitle(f"sine tracking real vs sim  joint idx {j}", y=1.0)
    fig.tight_layout(); fig.savefig(out, dpi=120, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    main()
