#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PD 辨识网格分析: 处理 excite_record 录的 CSV, 出跟踪指标 + 每关节推荐 PD。

输入目录里应有成对文件(excite_record 产出):
  j{J}_kp{KP}_kd{KD}_sine[_suffix]_cmd.csv / _state.csv   (正弦: 增益/相位延迟/力矩)
  j{J}_kp{KP}_kd{KD}_step[_suffix]_cmd.csv / _state.csv    (阶跃: 超调/残振/力矩)

正弦增益/相位: 在每个频率窗口内对 cmd 与 real 各做该频率的最小二乘正弦拟合
  y ≈ a·cos(2πf t)+b·sin(2πf t)+c → 幅值=hypot(a,b), 相位=atan2(a,b);
  增益=幅real/幅cmd, 相位滞后 Δφ → 延迟=−Δφ/(2πf)。

用法:
  python3 analyze_pd_grid.py <data_dir> --freqs 0.5,1,2,3 --cycles 6
输出: <data_dir>/analysis/pd_grid_metrics.csv + 每关节对比图 + 终端推荐。
"""
import argparse
import csv
import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP",
                                   "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 判据阈值
GAIN_OK = 0.8            # ≤3Hz 幅值达成率下限
OVERSHOOT_OK = 0.15      # 阶跃超调上限
TORQUE_FRAC = 0.8        # 峰值力矩 ≤ TORQUE_FRAC × effort
FNAME = re.compile(r"j(\d+)_kp([0-9.eE+-]+|NA)_kd([0-9.eE+-]+|NA)_(sine|step|chirp)(?:_(.+))?_cmd\.csv$")


def effort_of(j):
    return 7.0 if j % 6 == 5 else 27.0   # 实机序: 每腿第6个(踝roll)=DM4310 effort 7, 其余 27


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    cols = {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != "phase"}
    cols["phase"] = np.array([r["phase"] for r in rows], dtype=object)
    return cols


def sine_fit(t, y, f):
    """在频率 f 拟合 y≈a cos+b sin+c, 返回 (幅值, 相位 rad)。"""
    w = 2 * np.pi * f
    A = np.column_stack([np.cos(w * t), np.sin(w * t), np.ones_like(t)])
    a, b, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(np.hypot(a, b)), float(np.arctan2(a, b))


def analyze_sine(cmd, state, j, freqs, cycles):
    """按频率窗口分段, 返回 [(f, gain, delay_ms, peak_tau), ...]。"""
    qd = cmd[f"qd{j}"]; tc = cmd["t"]; phc = cmd["phase"]
    q = state[f"q{j}"]; ts = state["t"]; phs = state["phase"]
    tau = state.get(f"tau{j}")
    # 激励段起点(cmd/state 各自)
    exc_c = np.where(phc == "excite")[0]
    exc_s = np.where(phs == "excite")[0]
    if len(exc_c) == 0 or len(exc_s) == 0:
        return []
    t0c, t0s = tc[exc_c[0]], ts[exc_s[0]]
    durs = [cycles / f for f in freqs]
    bounds = np.concatenate([[0.0], np.cumsum(durs)])
    out = []
    for i, f in enumerate(freqs):
        lo, hi = bounds[i], bounds[i + 1]
        skip = min(1.0 / f, 0.4 * (hi - lo))          # 丢头 1 个周期的暂态
        cm = (tc - t0c >= lo + skip) & (tc - t0c < hi) & (phc == "excite")
        sm = (ts - t0s >= lo + skip) & (ts - t0s < hi) & (phs == "excite")
        if cm.sum() < 8 or sm.sum() < 8:
            continue
        Ac, pc = sine_fit(tc[cm] - t0c, qd[cm], f)
        Ar, pr = sine_fit(ts[sm] - t0s, q[sm], f)
        if Ac < 1e-4:
            continue
        gain = Ar / Ac
        dphi = (pr - pc + np.pi) % (2 * np.pi) - np.pi  # 绕到 [-π,π]
        delay_ms = -dphi / (2 * np.pi * f) * 1000.0
        peak_tau = float(np.max(np.abs(tau[sm]))) if tau is not None else np.nan
        out.append((f, gain, delay_ms, peak_tau))
    return out


def analyze_step(cmd, state, j):
    """每个阶跃台阶: 超调%、是否残振、峰值力矩。返回 (max_overshoot, ring_flag, peak_tau)。"""
    qd = cmd[f"qd{j}"]; tc = cmd["t"]
    q = state[f"q{j}"]; ts = state["t"]; phs = state["phase"]
    tau = state.get(f"tau{j}")
    sm = phs == "excite"
    if sm.sum() < 10:
        return np.nan, False, np.nan
    # 找 cmd 的台阶跳变时刻
    edges = np.where(np.abs(np.diff(qd)) > 0.02)[0]
    over = []
    ring = False
    for e in edges:
        t_e = tc[e]
        seg = (ts >= t_e) & (ts < t_e + 1.2) & sm
        if seg.sum() < 5:
            continue
        y = q[seg]
        y0 = y[0]
        yf = np.median(y[-max(3, len(y)//5):])   # 末端稳态
        step = yf - y0
        if abs(step) < 0.02:
            continue
        peak = (np.max(y) - yf) if step > 0 else (yf - np.min(y))
        over.append(max(0.0, peak / abs(step)))
        # 残振: 稳态后 (q-yf) 过零次数
        tail = y[len(y)//2:] - yf
        zc = np.sum(np.diff(np.sign(tail)) != 0)
        if zc >= 4:
            ring = True
    peak_tau = float(np.max(np.abs(tau[sm]))) if tau is not None else np.nan
    return (max(over) if over else np.nan), ring, peak_tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--freqs", default="0.5,1,2,3")
    ap.add_argument("--cycles", type=float, default=6)
    a = ap.parse_args()
    freqs = [float(x) for x in a.freqs.split(",")]
    outdir = os.path.join(a.data_dir, "analysis")
    os.makedirs(outdir, exist_ok=True)

    # 收集所有 sine/step 配对
    recs = {}   # (j, kp, kd) -> {'sine': [...], 'step': (...)}
    for fn in sorted(os.listdir(a.data_dir)):
        m = FNAME.search(fn)
        if not m:
            continue
        j = int(m.group(1))
        if m.group(2) == "NA" or m.group(3) == "NA":
            print(f"[跳过] {fn}: kp/kd=NA(录制时没读到节点增益)")
            continue
        kp, kd, mode = float(m.group(2)), float(m.group(3)), m.group(4)
        cmd_p = os.path.join(a.data_dir, fn)
        st_p = cmd_p[:-len("_cmd.csv")] + "_state.csv"
        if not os.path.exists(st_p):
            print(f"[跳过] {fn}: 缺 state 配对")
            continue
        cmd, state = read_csv(cmd_p), read_csv(st_p)
        key = (j, kp, kd)
        recs.setdefault(key, {})
        if mode == "sine":
            recs[key]["sine"] = analyze_sine(cmd, state, j, freqs, a.cycles)
        elif mode == "step":
            recs[key]["step"] = analyze_step(cmd, state, j)

    if not recs:
        print("没找到可分析的数据(检查文件名格式 j{J}_kp*_kd*_{sine,step}_*_cmd/state.csv)")
        return

    # 写指标 CSV
    mpath = os.path.join(outdir, "pd_grid_metrics.csv")
    with open(mpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["joint", "kp", "kd", "freq_Hz", "gain", "delay_ms", "peak_tau",
                    "step_overshoot", "step_ring", "effort_lim"])
        for (j, kp, kd), d in sorted(recs.items()):
            eff = effort_of(j)
            ov, ring, ptau = d.get("step", (np.nan, False, np.nan))
            for (fr, g, dl, pt) in d.get("sine", []):
                w.writerow([j, kp, kd, fr, f"{g:.3f}", f"{dl:.1f}", f"{pt:.2f}",
                            f"{ov:.3f}" if ov == ov else "", int(ring), eff])
            if "sine" not in d:   # 只有 step 的也记一行
                w.writerow([j, kp, kd, "", "", "", f"{ptau:.2f}" if ptau==ptau else "",
                            f"{ov:.3f}" if ov==ov else "", int(ring), eff])
    print(f"[metrics] {mpath}")

    # 每关节: 对比图 + 推荐
    joints = sorted(set(j for (j, _, _) in recs))
    for j in joints:
        eff = effort_of(j)
        pds = sorted([(kp, kd) for (jj, kp, kd) in recs if jj == j])
        # ---- 对比图: 增益-频率 / 延迟-频率 ----
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        for (kp, kd) in pds:
            s = recs[(j, kp, kd)].get("sine", [])
            if not s:
                continue
            fs = [x[0] for x in s]; gs = [x[1] for x in s]; dls = [x[2] for x in s]
            ax[0].plot(fs, gs, "o-", label=f"kp{kp:g}/kd{kd:g}")
            ax[1].plot(fs, dls, "o-", label=f"kp{kp:g}/kd{kd:g}")
        ax[0].axhline(GAIN_OK, color="k", ls=":", lw=1); ax[0].set_ylim(0, 1.3)
        ax[0].set_title(f"关节 idx{j}: 幅值达成率(增益) vs 频率  (虚线={GAIN_OK})")
        ax[0].set_xlabel("Hz"); ax[0].set_ylabel("real/cmd"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=9)
        ax[1].set_title(f"关节 idx{j}: 相位滞后→延迟(ms) vs 频率")
        ax[1].set_xlabel("Hz"); ax[1].set_ylabel("delay ms"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=9)
        plt.tight_layout(); plt.savefig(f"{outdir}/joint{j}_pd_compare.png", dpi=110); plt.close()

        # ---- 推荐: 满足硬约束里取最低 kp ----
        cand = []
        for (kp, kd) in pds:
            d = recs[(j, kp, kd)]
            s = d.get("sine", [])
            g_ok = all(g >= GAIN_OK for (fr, g, _, _) in s if fr <= 3.0) and len(s) > 0
            tau_ok = all((pt <= TORQUE_FRAC * eff) for (_, _, _, pt) in s if pt == pt) if s else True
            ov, ring, ptau = d.get("step", (np.nan, False, np.nan))
            ov_ok = (ov <= OVERSHOOT_OK) if ov == ov else True
            ok = g_ok and tau_ok and ov_ok and (not ring)
            reason = []
            if not g_ok: reason.append("跟不动(增益<0.8)")
            if not tau_ok: reason.append("力矩饱和")
            if not ov_ok: reason.append(f"超调{ov*100:.0f}%")
            if ring: reason.append("残振/起振")
            cand.append((ok, kp, kd, ";".join(reason) or "OK"))
        print(f"\n===== 关节 idx{j} (effort={eff:g}) 推荐 =====")
        good = [c for c in cand if c[0]]
        if good:
            best = min(good, key=lambda c: (c[1], c[2]))   # 最低 kp, 再最低 kd
            print(f"  ★推荐 PD: kp={best[1]:g} kd={best[2]:g}  (满足约束里最低 kp)")
        else:
            print("  ⚠ 没有 PD 同时满足所有约束, 看下表放宽哪条:")
        for ok, kp, kd, r in cand:
            print(f"    {'✓' if ok else '✗'} kp{kp:g}/kd{kd:g}: {r}")
    print(f"\n[图] {outdir}/joint*_pd_compare.png")


if __name__ == "__main__":
    main()
