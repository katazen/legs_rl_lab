#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分频驻留正弦 (stepped-sine) FRF 分析。
每个频率在【稳态段】(丢掉每段前 skip 个周期的暂态)对 target 与 response 各做一次
单频最小二乘正弦拟合, 直接得增益 |q/target| 与相位滞后。不用 chirp 的宽带 FFT ->
低频不再"周期饥饿"。可同时叠 sim 曲线对比。

用法:
  python3 analyze_stepsine.py <joint_idx> <freqs逗号> <cycles> <out.png> \
        real=<cmd.csv>,<state.csv>  [sim=<sim_state.csv>,<sim_target_col>] [skip=2]

  - real: excite_record 产出的 <..._cmd.csv>,<..._state.csv> 一对
  - sim (可选): isaac chirp_replay --replay_csv 回放产出的 csv (含 t,target,q), 只给这一个文件
  - skip: 每段丢掉的前导周期数 (默认 2)
"""
import sys, csv
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt


def _read_cols(path, cols):
    r = list(csv.DictReader(open(path)))
    out = {}
    for c in cols:
        out[c] = np.array([float(x[c]) for x in r])
    if "phase" in r[0]:
        out["phase"] = np.array([x["phase"] for x in r])
    return out


def seg_bounds(freqs, cycles):
    """复刻 excite_record.build_excitation(sine) 的分段边界(激励局部时间)。"""
    durs = np.array([cycles / f for f in freqs])
    return np.concatenate([[0.0], np.cumsum(durs)])  # len = n+1


def sine_phasor(t, x, f):
    """在已知频率 f 上对 x(t) 做最小二乘正弦拟合, 返回复相量 a+jb (x≈a·sin+b·cos)。"""
    x = x - x.mean()
    A = np.vstack([np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)]).T
    a, b = np.linalg.lstsq(A, x, rcond=None)[0]
    return a + 1j * b


def frf_stepsine(t_tgt, tgt, t_resp, resp, freqs, cycles, skip, t0_tgt, t0_resp):
    """返回 (fc, gain, lag_deg)。t0_* = 各自流的激励起点(局部时间基准)。"""
    bounds = seg_bounds(freqs, cycles)
    lt = t_tgt - t0_tgt
    lr = t_resp - t0_resp
    fc, gain, lag = [], [], []
    for i, f in enumerate(freqs):
        lo, hi = bounds[i], bounds[i + 1]
        win_lo = lo + skip / f                      # 丢前 skip 个周期的暂态
        mt = (lt >= win_lo) & (lt < hi)
        mr = (lr >= win_lo) & (lr < hi)
        if mt.sum() < 8 or mr.sum() < 8:
            print(f"  [skip] {f:g}Hz 稳态样本不足 (tgt={mt.sum()}, resp={mr.sum()})")
            continue
        Ht = sine_phasor(lt[mt], tgt[mt], f)
        Hr = sine_phasor(lr[mr], resp[mr], f)
        if abs(Ht) < 1e-6:
            continue
        H = Hr / Ht
        fc.append(f); gain.append(abs(H)); lag.append(-np.degrees(np.angle(H)))
    return np.array(fc), np.array(gain), np.array(lag)


def excite_start(t, phase):
    """激励段(phase=='excite')首帧时间; 无 phase 列则取 0。"""
    if phase is None:
        return t[0]
    m = phase == "excite"
    return t[m][0] if m.any() else t[0]


def report(label, fc, g, lag):
    if len(fc) == 0:
        print(f"[{label}] 无有效频点"); return
    g0 = g[0]
    ipk = int(np.argmax(g))
    below = np.where(g < 0.707 * g0)[0]
    bw = fc[below[0]] if len(below) else None
    pk = f"共振峰 {g[ipk]:.2f}@{fc[ipk]:.2f}Hz" if g[ipk] > 1.05 * g0 else "无明显共振峰"
    bwtxt = f"{bw:.2f}Hz" if bw else f">{fc[-1]:.2f}Hz"
    print(f"[{label}] 低频增益{g0:.2f}  {pk}  -3dB带宽~{bwtxt}")
    print(f"       {'f(Hz)':>7}{'gain':>8}{'lag(deg)':>10}")
    for f, gg, ll in zip(fc, g, lag):
        print(f"       {f:>7.2f}{gg:>8.3f}{ll:>10.1f}")


def _model(w, kp, J, c, td):
    return kp * np.exp(-1j * w * td) / (kp - J * w ** 2 + 1j * c * w)


def fit_2nd_order_delay(fc, gain, lag_deg, kp):
    """在离散 FRF 点上最小二乘拟合 H=kp·e^{-jωτ}/(kp − Jω² + jcω), 解 J,c,τ。
    纯 numpy 粗网格 + 局部细化, 复频响误差(增益与相位同时约束)。"""
    w = 2 * np.pi * fc
    Hd = gain * np.exp(-1j * np.radians(lag_deg))   # angle(H) = -lag
    Js = np.linspace(0.005, 0.8, 90); cs = np.linspace(0.3, 25, 90); tds = np.linspace(0, 0.03, 31)
    best = (1e18, None)
    for J in Js:
        for c in cs:
            for td in tds:
                e = np.sum(np.abs(Hd - _model(w, kp, J, c, td)) ** 2)
                if e < best[0]: best = (e, (J, c, td))
    J, c, td = best[1]
    for _ in range(4):
        Js = np.linspace(J * 0.7, J * 1.3, 25); cs = np.linspace(c * 0.7, c * 1.3, 25)
        tds = np.linspace(max(0, td - 0.004), td + 0.004, 21)
        best = (1e18, None)
        for JJ in Js:
            for cc in cs:
                for tt in tds:
                    e = np.sum(np.abs(Hd - _model(w, kp, JJ, cc, tt)) ** 2)
                    if e < best[0]: best = (e, (JJ, cc, tt))
        J, c, td = best[1]
    return J, c, td


def main():
    j = int(sys.argv[1])
    freqs = [float(x) for x in sys.argv[2].split(",")]
    cycles = float(sys.argv[3])
    out = sys.argv[4]
    skip = 2.0
    kp = None
    fitmin = 0.0
    fitmax = 1e9
    curves = []  # (label, cmd_or_state_paths)
    for a in sys.argv[5:]:
        if a.startswith("skip="):
            skip = float(a.split("=", 1)[1])
        elif a.startswith("kp="):
            kp = float(a.split("=", 1)[1])
        elif a.startswith("fitmin="):
            fitmin = float(a.split("=", 1)[1])
        elif a.startswith("fitmax="):
            fitmax = float(a.split("=", 1)[1])
        elif a.startswith("real="):
            curves.append(("real", a.split("=", 1)[1].split(",")))
        elif a.startswith("sim="):
            curves.append(("sim", a.split("=", 1)[1].split(",")))

    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    cols = {"real": "tab:red", "sim": "tab:blue"}
    for label, paths in curves:
        if label == "real":
            cmd_csv, state_csv = paths
            c = _read_cols(cmd_csv, [f"qd{j}", "t"])
            s = _read_cols(state_csv, [f"q{j}", "t"])
            t0t = excite_start(c["t"], c.get("phase"))
            t0r = excite_start(s["t"], s.get("phase"))
            fc, g, lag = frf_stepsine(c["t"], c[f"qd{j}"], s["t"], s[f"q{j}"],
                                      freqs, cycles, skip, t0t, t0r)
        else:  # sim: 单文件, 列名 target,q
            sim_csv = paths[0]
            d = _read_cols(sim_csv, ["t", "target", "q"])
            t0 = excite_start(d["t"], d.get("phase"))
            fc, g, lag = frf_stepsine(d["t"], d["target"], d["t"], d["q"],
                                      freqs, cycles, skip, t0, t0)
        report(label, fc, g, lag)
        col = cols.get(label, "tab:green")
        ax[0].semilogx(fc, g, "-o", ms=4, color=col, label=label)
        ax[1].semilogx(fc, lag, "-o", ms=4, color=col, label=label)

        # 在离散点上拟合二阶+延迟模型, 叠加平滑模型曲线(可限拟合频段, 避开次模态/高频噪声)
        if kp is not None and len(fc) >= 4:
            fm = (fc >= fitmin) & (fc <= fitmax)
            if fm.sum() >= 4:
                J, c_, td = fit_2nd_order_delay(fc[fm], g[fm], lag[fm], kp)
                wn = np.sqrt(kp / J); zeta = c_ / (2 * np.sqrt(kp * J))
                rng = f"[{fitmin:g}-{fitmax:g}Hz]" if (fitmin > 0 or fitmax < 1e8) else ""
                print(f"[{label} fit{rng}] J={J:.4f} kg·m²  c(总阻尼)={c_:.2f}  τ={td * 1000:.1f}ms"
                      f"  ->  fn={wn / 2 / np.pi:.2f}Hz  ζ={zeta:.3f}")
                ff = np.logspace(np.log10(fc[0]), np.log10(fc[-1]), 200)
                Hm = _model(2 * np.pi * ff, kp, J, c_, td)
                ax[0].semilogx(ff, np.abs(Hm), "--", color=col, lw=1.2, alpha=.8,
                               label=f"{label} fit (fn={wn/2/np.pi:.1f}Hz, ζ={zeta:.2f})")
                ax[1].semilogx(ff, -np.degrees(np.angle(Hm)), "--", color=col, lw=1.2, alpha=.8)

    ax[0].axhline(1.0, color="k", lw=0.5); ax[0].axhline(0.707, color="gray", ls=":", lw=1, label="-3dB")
    ax[0].set_ylabel("gain |q/target|"); ax[0].grid(alpha=.3, which="both"); ax[0].legend()
    ax[0].set_title(f"stepped-sine FRF  joint idx {j}")
    ax[1].axhline(0, color="k", lw=0.5); ax[1].set_ylabel("phase lag [deg]")
    ax[1].set_xlabel("frequency [Hz]"); ax[1].grid(alpha=.3, which="both"); ax[1].legend()
    fig.tight_layout(); fig.savefig(out, dpi=120); print("saved", out)


if __name__ == "__main__":
    main()
