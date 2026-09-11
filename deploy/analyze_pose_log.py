#!/usr/bin/env python3
"""分析 pose_move.py 产生的 CSV: 到位精度、力矩峰值、分段时序。"""
import sys
import numpy as np

REAL = ["L1","L2","L3","L4","L5","L6","R1","R2","R3","R4","R5","R6"]
DESC = ["髋pitch","髋roll","髋yaw","膝","踝pitch","踝roll"] * 2
KP = np.array([200.,100.,100.,250.,40.,40., 200.,100.,100.,250.,40.,40.])
EFF = np.array([26.,26.,26.,26.,26.,5.8, 26.,26.,26.,26.,26.,5.8])
ERR_LIMIT = 0.6 * EFF / KP
# armcontrol kLegFbSign; effort 发布时未乘此符号 -> 预期 tau_推算 = FB_SIGN * tau_自报
FB_SIGN = np.array([1.,1.,-1.,1.,-1.,1., -1.,1.,-1.,-1.,1.,1.])

d = np.loadtxt(sys.argv[1], delimiter=",", skiprows=1)
t = d[:, 0]
tgt, q, err, dq, tau = (d[:, 1:13], d[:, 13:25], d[:, 25:37], d[:, 37:49], d[:, 49:61])
# 旧日志 63 列(无 effort), 新日志 75 列(taufb 在 61:73) -> 按列数自适应
has_fb = d.shape[1] >= 75
taufb = d[:, 61:73] if has_fb else None
seg, s = (d[:, 73], d[:, 74]) if has_fb else (d[:, 61], d[:, 62])

print(f"文件 {sys.argv[1].split('/')[-1]}   时长 {t[-1]:.2f}s   {len(t)} 帧")
segs = [(1.0, "段1 整形(roll/yaw)"), (2.0, "段2 升降(pitch/膝)"), (0.0, "保持")]
for v, nm in segs:
    m = seg == v
    if m.any():
        print(f"  {nm}: t={t[m][0]:.2f}~{t[m][-1]:.2f}s")

print("\n关节  部位          " + "     起点     目标     实到    到位误差     行程  完成率 |err|max   阈值 |tau|max")
print("-" * 100)
for i in range(12):
    trav = tgt[-1, i] - q[0, i]
    done = q[-1, i] - q[0, i]
    rate = done / trav * 100 if abs(trav) > 1e-4 else 100.0
    print(f"{REAL[i]:<4}{DESC[i]:<9}{q[0,i]:>9.4f}{tgt[-1,i]:>9.4f}{q[-1,i]:>9.4f}"
          f"{q[-1,i]-tgt[-1,i]:>10.4f}{trav:>9.4f}{rate:>7.1f}%"
          f"{np.abs(err[:,i]).max():>9.4f}{ERR_LIMIT[i]:>7.3f}{np.abs(tau[:,i]).max():>9.1f}")

fin = np.abs(q[-1] - tgt[-1])
print(f"\n到位误差: max {fin.max():.4f} rad ({np.rad2deg(fin.max()):.2f}°) @ {REAL[int(fin.argmax())]}"
      f"   RMS {np.sqrt((fin**2).mean()):.4f} rad")
print(f"跟踪误差: max {np.abs(err).max():.4f} rad @ {REAL[int(np.abs(err).max(0).argmax())]}"
      f"   (阈值最紧 0.062 @ 膝)")
print(f"力矩峰值: max {np.abs(tau).max():.1f} N·m @ {REAL[int(np.abs(tau).max(0).argmax())]}"
      f"   (effort_limit 26 / 踝roll 5.8)")
print(f"关节速度: max {np.abs(dq).max():.3f} rad/s @ {REAL[int(np.abs(dq).max(0).argmax())]}")

# 左右镜像残差: 两条腿可以各自都在容差内, 却一前一后差几 cm。
# 关节1髋pitch/4膝/5踝pitch 左右同号 -> 看 L-R; 2髋roll/3髋yaw/6踝roll 反号 -> 看 L+R
same = (0, 3, 4)
print("\n左右镜像残差(末帧):")
worst, worst_j = 0.0, 0
for j in range(6):
    l, r = q[-1, j], q[-1, j + 6]
    res = (l - r) if j in same else (l + r)
    if abs(res) > worst:
        worst, worst_j = abs(res), j
    print(f"  关节{j+1} {DESC[j]:<9}L{l:+.4f} R{r:+.4f} {'差' if j in same else '和'}={res:+.4f}")
print(f"  最大 {worst:.4f} rad ({np.rad2deg(worst):.2f}°) @ 关节{worst_j+1} {DESC[worst_j]}")

# 符号链交叉验证 + 使能状态: 只比"目标 vs 读数"发现不了自洽的镜像错配。
# 失能判据必须看"指令力矩显著而自报力矩为0": 目标=实测时 PD 力矩本来就≈0。
if has_fb:
    cmd_amp, fb_amp = np.abs(tau).max(), np.abs(taufb).max()
    if cmd_amp < 1.0:
        print(f"\n符号链校验: 跳过 —— 指令力矩全程最大仅 {cmd_amp:.2f} N·m, 没有足够激励")
    elif fb_amp < 0.2:
        print(f"\n!! 指令力矩最大 {cmd_amp:.1f} N·m 但 effort 全程≈0 -> 电机未使能/已失能")
    else:
        print("\n符号链校验 (预期 tau_推算 = FB_SIGN * tau_自报):")
        bad = []
        for i in range(12):
            m = np.abs(taufb[:, i]) > 0.5
            if m.sum() < 20:
                print(f"  {REAL[i]:<4}{DESC[i]:<9}样本不足({int(m.sum())} 帧), 跳过")
                continue
            r = float(np.corrcoef(tau[m, i], taufb[m, i])[0, 1])
            got = 1.0 if r > 0 else -1.0
            ok = got == FB_SIGN[i]
            if not ok:
                bad.append(REAL[i])
            print(f"  {REAL[i]:<4}{DESC[i]:<9}预期{FB_SIGN[i]:+.0f} 实测{got:+.0f} corr={r:+.3f}"
                  f"  |推算|max={np.abs(tau[m,i]).max():>5.1f} |自报|max={np.abs(taufb[m,i]).max():>5.1f}"
                  f"{'' if ok else '   <-- 不符!'}")
        print(f"  {'!! 异常关节: ' + ','.join(bad) if bad else 'OK: 12 关节符号链全部符合预期'}")
else:
    print("\n(旧格式日志, 无电机自报 effort, 无法做符号链校验)")

# 到位后是否漂移(末尾保持段)
hold = seg == 0.0
if hold.sum() > 100:
    qh = q[hold]
    drift = qh[-1] - qh[0]
    print(f"\n保持段({hold.sum()/200:.1f}s)漂移: max {np.abs(drift).max():.4f} rad "
          f"@ {REAL[int(np.abs(drift).argmax())]}")
    print("  逐关节漂移: " + "  ".join(f"{REAL[i]}{drift[i]:+.4f}" for i in range(12)))

# 可选: 指定关节的时间历程, 用于判断是"走一段后卡住"还是"全程跟不上"
for nm in sys.argv[2:]:
    i = REAL.index(nm)
    print(f"\n--- {nm}({DESC[i]}) 时间历程 ---")
    head = f"{'t':>7}{'目标':>10}{'实测':>10}{'误差':>9}{'速度':>8}{'tau':>7}"
    print(head + (f"{'tau自报':>9}" if has_fb else ""))
    for tt in np.linspace(0, t[-1], 16):
        k = int(np.argmin(np.abs(t - tt)))
        line = (f"{t[k]:>7.2f}{tgt[k,i]:>10.4f}{q[k,i]:>10.4f}{err[k,i]:>9.4f}"
                f"{dq[k,i]:>8.3f}{tau[k,i]:>7.1f}")
        print(line + (f"{taufb[k,i]:>9.1f}" if has_fb else ""))
