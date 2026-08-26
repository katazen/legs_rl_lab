#!/usr/bin/env python3
"""独立复算 Stage1B：起动延迟(速度阈值口径) + 新分析(tau起动/DC增益幅值依赖->摩擦)。"""
import csv
import os

import numpy as np

DOC = os.path.join(os.path.dirname(__file__), "..", "doc")
JOINTS = [("hipL", 0, "j0_kp200_kd5_step_step_bi", 200.0),
          ("hipR", 6, "j6_kp200_kd5_step_step_bi", 200.0),
          ("kneeL", 3, "j3_kp250_kd5_step_step_bi", 250.0),
          ("kneeR", 9, "j9_kp250_kd5_step_step_bi", 250.0)]


def read(stage, stem, j):
    root = os.path.join(DOC, stage, "data", stem)
    with open(root + "_cmd.csv") as f:
        cmd = list(csv.DictReader(f))
    with open(root + "_state.csv") as f:
        state = list(csv.DictReader(f))
    return {
        "ct": np.array([float(r["t"]) for r in cmd]),
        "cp": np.array([r["phase"] for r in cmd]),
        "qd": np.array([float(r[f"qd{j}"]) for r in cmd]),
        "st": np.array([float(r["t"]) for r in state]),
        "q": np.array([float(r[f"q{j}"]) for r in state]),
        "v": np.array([float(r[f"v{j}"]) for r in state]),
        "tau": np.array([float(r[f"tau{j}"]) for r in state]),
    }


def edges_of(d):
    excite = d["cp"] == "excite"
    t, qd = d["ct"][excite], d["qd"][excite]
    k = np.where(np.abs(np.diff(qd)) > 0.005)[0]
    out = [(t[i + 1], qd[i], qd[i + 1]) for i in k]
    return [(te, an) for te, ap, an in out if abs(ap) < 0.02 and 0.02 < abs(an) < 0.15]


def onset(t, y, te, thr_abs, frac=0.10, win=0.5):
    """命令沿 te 后 |y| 过 max(thr_abs, frac*peak) 的首样本与线性插值时刻。"""
    w = np.where((t >= te - 0.05) & (t <= te + win))[0]
    ay = np.abs(y[w])
    # 扣除沿前基线(对 tau 很重要: 静摩擦下稳态 tau 未必为 0)
    pre = ay[t[w] < te]
    base = np.median(pre) if len(pre) else 0.0
    ay = np.abs(ay - base)
    thr = max(thr_abs, frac * ay[t[w] >= te].max())
    idx = np.where((ay > thr) & (t[w] >= te))[0]
    if not len(idx):
        return np.nan, np.nan
    i = idx[0]
    raw = t[w[i]] - te
    if i and ay[i] > ay[i - 1]:
        tc = t[w[i - 1]] + (thr - ay[i - 1]) / (ay[i] - ay[i - 1]) * (t[w[i]] - t[w[i - 1]])
    else:
        tc = t[w[i]]
    return raw, tc - te


print(f"{'joint':>6} {'n':>3} | {'v首样本(ms)':>14} {'v插值(ms)':>12} | "
      f"{'tau首样本(ms)':>14} {'tau插值(ms)':>13} | {'DCg@.05':>8} {'DCg@.10':>8} "
      f"{'err.05(rad)':>11} {'err.10(rad)':>11} {'Fc_est(N·m)':>11}")
for label, j, stem, kp in JOINTS:
    d = read("stage1b", stem, j)
    ed = edges_of(d)
    vraw, vint, traw, tint = [], [], [], []
    gain = {0.05: [], 0.10: []}
    err = {0.05: [], 0.10: []}
    for te, amp in ed:
        r, c = onset(d["st"], d["v"], te, 0.02)
        vraw.append(r); vint.append(c)
        r2, c2 = onset(d["st"], d["tau"], te, 0.05)
        traw.append(r2); tint.append(c2)
        pre = (d["st"] >= te - 0.15) & (d["st"] <= te - 0.02)
        ss = (d["st"] >= te + 1.0) & (d["st"] <= te + 1.4)
        dq = d["q"][ss].mean() - d["q"][pre].mean()
        a = round(abs(amp), 2)
        if a in gain:
            gain[a].append(dq / amp)
            err[a].append(abs(amp) - abs(dq))
    vraw, vint = np.array(vraw) * 1e3, np.array(vint) * 1e3
    traw, tint = np.array(traw) * 1e3, np.array(tint) * 1e3
    e05, e10 = np.mean(err[0.05]), np.mean(err[0.10])
    # 若为库仑摩擦: 稳态 kp*err = Fc, 与幅值无关 -> 两档 err 应近似相等
    fc_est = kp * 0.5 * (e05 + e10)
    print(f"{label:>6} {len(ed):>3} | {np.mean(vraw):6.2f}±{np.std(vraw):4.2f} "
          f"{np.mean(vint):6.2f}±{np.std(vint):4.2f} | "
          f"{np.nanmean(traw):6.2f}±{np.nanstd(traw):4.2f} "
          f"{np.nanmean(tint):6.2f}±{np.nanstd(tint):4.2f} | "
          f"{np.mean(gain[0.05]):8.3f} {np.mean(gain[0.10]):8.3f} "
          f"{e05:11.5f} {e10:11.5f} {fc_est:11.3f}")

print("\n判据: 若 err(0.05)≈err(0.10) [绝对稳态误差与幅值无关] -> 库仑摩擦主导, Fc≈kp*err;"
      "\n      若 gain 两档相同(err 随幅值等比) -> 比例/标定误差而非摩擦。")
