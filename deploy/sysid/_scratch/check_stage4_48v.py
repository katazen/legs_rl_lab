#!/usr/bin/env python3
"""Stage4 48V 文档数字独立复算 + 新结论挖掘.

- 4A: 膝/踝roll 阶跃逐幅值峰速、起动延迟(自实现口径)、峰值力矩
- 4A+: 膝 0.35rad 阶跃 transit 段 τ–v 驱动象限包络 → 直线外推零力矩转速 (velocity_limit 数据估计)
- 4B/4C: 用独立相量拟合复算 FRF 增益/相位, 对照 *_frf_24v_vs_48v.csv
- tau 通道: 全程乘 KSIGN 翻正 (arm_control_node effort 漏乘 kLegFbSign)
"""
import csv
import numpy as np

ROOT = "/home/woan/workspace/legs_rl_lab/deploy/sysid/doc/stage4_48v"
KSIGN = np.array([+1, +1, -1, +1, -1, +1, -1, +1, -1, -1, +1, +1], float)


def load(path):
    with open(path) as f:
        r = csv.reader(f)
        next(r)
        ph, rows = [], []
        for row in r:
            if not row:
                continue
            ph.append(row[1])
            rows.append([float(x) for x in row[:1] + row[2:]])
    return np.array(rows), np.array(ph)


def onset_ms(t, v, t_edge, pre_mask):
    """命令沿 -> |v| 过 max(0.02, 10% 峰速) 的首样本, 扣基线."""
    m = (t >= t_edge) & (t < t_edge + 0.4)
    if m.sum() < 5:
        return None
    av = np.abs(v[m] - np.median(v[pre_mask]))
    thr = max(0.02, 0.1 * av.max())
    idx = np.where(av > thr)[0]
    return (t[m][idx[0]] - t_edge) * 1000 if len(idx) else None


def step_check(stem, j, amps):
    st, _ = load(f"{ROOT}/4a/data/{stem}_state.csv")
    cm, _ = load(f"{ROOT}/4a/data/{stem}_cmd.csv")
    t, q, v, tau = st[:, 0], st[:, 1 + j], st[:, 13 + j], KSIGN[j] * st[:, 25 + j]
    tc, qd = cm[:, 0], cm[:, 1 + j]
    base = np.median(qd[: int(len(qd) * 0.1)])
    rel = qd - base
    edges = np.where(np.abs(np.diff(rel)) > 0.02)[0]
    edges = edges[np.insert(np.diff(tc[edges]) > 0.5, 0, True)]  # 去抖
    print(f"-- {stem} (j{j}): {len(edges)} 个命令沿")
    delays, drive_pts = [], []
    peaks = {a: [] for a in amps}
    for e in edges:
        te = tc[e]
        target = rel[min(e + 5, len(rel) - 1)]
        a = round(abs(target), 2)
        m = (t >= te) & (t < te + 0.5)
        pre = (t >= te - 0.2) & (t < te)
        if m.sum() < 10:
            continue
        pv = np.abs(v[m]).max()
        if a in peaks:
            peaks[a].append(pv)
        d = onset_ms(t, v, te, pre)
        if d is not None and a <= 0.10:  # 文档口径: 只用小幅值沿
            delays.append(d)
        # transit 驱动象限样本 (速度与力矩同向)
        mm = m & (np.abs(v) > 0.3) & (v * tau > 0)
        drive_pts.append(np.column_stack([np.abs(v[mm]), np.abs(tau[mm])]))
    for a in amps:
        if peaks[a]:
            print(f"   A={a:4.2f}: peak|v| {np.mean(peaks[a]):.3f} ± {np.std(peaks[a]):.3f}  (n={len(peaks[a])})")
    print(f"   onset delay: {np.mean(delays):.2f} ± {np.std(delays):.2f} ms (n={len(delays)})")
    return np.vstack(drive_pts) if drive_pts else None


def envelope_fit(pts, vmin, label):
    """按 |v| 分箱取 p90 |tau|, 高速段直线拟合外推零力矩转速."""
    bins = np.arange(0, pts[:, 0].max() + 0.25, 0.25)
    ctr, p90 = [], []
    for lo in bins[:-1]:
        m = (pts[:, 0] >= lo) & (pts[:, 0] < lo + 0.25)
        if m.sum() >= 8:
            ctr.append(lo + 0.125)
            p90.append(np.percentile(pts[m, 1], 90))
    ctr, p90 = np.array(ctr), np.array(p90)
    print(f"   {label} 包络: " + "  ".join(f"{c:.2f}:{p:.1f}" for c, p in zip(ctr, p90)))
    m = ctr >= vmin
    if m.sum() >= 3:
        k, b = np.polyfit(ctr[m], p90[m], 1)
        if k < 0:
            print(f"   高速段(≥{vmin}) 拟合 τ={b:.1f}{k:+.2f}·v  →  零力矩转速 ≈ {-b / k:.2f} rad/s, 零速截距 ≈ {b:.1f} N·m")


def phasor(t, x, f):
    A = np.column_stack([np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t), np.ones_like(t)])
    c, *_ = np.linalg.lstsq(A, x, rcond=None)
    return complex(c[1], c[0])  # 与 analyze_stepsine 同构: H 相位差给滞后


def frf_check(stem, sub, j, freqs, cycles=6, skip=2):
    st, phs = load(f"{ROOT}/{sub}/data/{stem}_state.csv")
    cm, phc = load(f"{ROOT}/{sub}/data/{stem}_cmd.csv")
    t0s = st[phs == "excite", 0][0]
    t0c = cm[phc == "excite", 0][0]
    lt_c, tgt = cm[:, 0] - t0c, cm[:, 1 + j]
    lt_s, q = st[:, 0] - t0s, st[:, 1 + j]
    bounds = np.concatenate([[0], np.cumsum([cycles / f for f in freqs])])
    out = []
    for i, f in enumerate(freqs):
        lo, hi = bounds[i] + skip / f, bounds[i + 1]
        mc = (lt_c >= lo) & (lt_c < hi)
        ms = (lt_s >= lo) & (lt_s < hi)
        if mc.sum() < 8 or ms.sum() < 8:
            continue
        H = phasor(lt_s[ms], q[ms], f) / phasor(lt_c[mc], tgt[mc], f)
        out.append((f, abs(H), -np.degrees(np.angle(H))))
    return out


FRFS = [0.5, 0.8, 1.2, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0]
print("========== 4A 阶跃复算 ==========")
pl = step_check("j3_kp250_kd5_step_bigmotor48_r1", 3, [0.05, 0.10, 0.20, 0.35])
pr = step_check("j9_kp250_kd5_step_bigmotor48_r1", 9, [0.05, 0.10, 0.20, 0.35])
step_check("j5_kp40_kd0.5_step_smallmotor48_r1", 5, [0.05, 0.10, 0.15, 0.20])
step_check("j5_kp40_kd0.5_step_smallmotor48_high_r1", 5, [0.25, 0.30])
step_check("j11_kp40_kd0.5_step_smallmotor48_r1", 11, [0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
print("========== 4A+ 膝 τ–v 包络外推 velocity_limit ==========")
envelope_fit(pl, 2.5, "膝L 48V")
envelope_fit(pr, 2.5, "膝R 48V")
print("========== 4B FRF 独立复算 (48V) ==========")
for stem, j in [("j3_kp250_kd5_sine_frf_small48_r1", 3), ("j9_kp250_kd5_sine_frf_small48_r1", 9),
                ("j5_kp40_kd0.5_sine_frf_ankle_small48_r1", 5), ("j11_kp40_kd0.5_sine_frf_ankle_small48_r1", 11)]:
    res = frf_check(stem, "4b", j, FRFS)
    print(f"-- j{j}: " + "  ".join(f"{f:g}Hz {g:.3f}/{p:.1f}°" for f, g, p in res[:4]) + " ...")
print("========== 4C 踝pitch 三幅值复算 (48V, cycles=8) ==========")
for j in (4, 10):
    for tag in ("amp003", "amp005", "amp010"):
        res = frf_check(f"j{j}_kp40_kd2_sine_{tag}_48_r1", "4c", j, [0.5, 0.8, 1.2, 2.0], cycles=8)
        s = "  ".join(f"{f:g}Hz {g:.3f}/{p:.1f}°" for f, g, p in res)
        print(f"-- j{j} {tag}: {s}")

print("========== 1-DOF 仿真反推膝 velocity_limit ==========")
# 模型: J q̈ = clip(Kp(qd(t-Td)-q) - Kd v, ±τmax(v)) - Fc sgn(v)
# τmax(v) = clip(sat*(1 - v/vlim), 0, eff)  (驱动方向),  制动方向 = eff
def sim_step(A, J, vlim, sat=26.0, eff=26.0, kp=250.0, kd=5.0, fc=1.0, td=0.017, T=0.5, dt=1e-4):
    n = int(T / dt)
    nd = int(td / dt)
    q = np.zeros(n); v = np.zeros(n)
    for k in range(1, n):
        qd = A if k - 1 >= nd else 0.0
        tau = kp * (qd - q[k-1]) - kd * v[k-1]
        cap_pos = np.clip(sat * (1 - v[k-1] / vlim), 0, eff)
        cap_neg = -np.clip(sat * (1 + v[k-1] / vlim), 0, eff)
        tau = np.clip(tau, cap_neg, cap_pos)
        tau -= fc * np.sign(v[k-1])
        v[k] = v[k-1] + tau / J * dt
        q[k] = q[k-1] + v[k] * dt
    return np.abs(v).max()

# 先用未饱和的 A=0.05/0.10 标定 J (两电压一致 → 线性区)
meas = {0.05: 1.05, 0.10: 2.11, 0.20: 4.13, 0.35: 5.08}  # 左右均值 48V
best_J, best_e = None, 9e9
for J in np.arange(0.02, 0.30, 0.005):
    e = sum((sim_step(a, J, vlim=99) - meas[a])**2 for a in (0.05, 0.10))
    if e < best_e:
        best_e, best_J = e, J
print(f"线性区标定 J = {best_J:.3f} kg·m²  (A=0.05/0.10 峰速残差²={best_e:.4f})")
for vl in (3.4, 5.5, 6.0, 6.5, 6.8, 7.5, 8.5, 10.0):
    p20 = sim_step(0.20, best_J, vl)
    p35 = sim_step(0.35, best_J, vl)
    print(f"  vlim={vl:4.1f}:  A=0.20 → {p20:.2f} (实测4.13)   A=0.35 → {p35:.2f} (实测5.08)")
