#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实机观测数据出图 (按用户规范)。
图存到 <数据同级目录>/<数据时间戳>/ 内, 每 10s 一段, 每段 8 张:
  6 张关节跟踪 (同名关节左右对比, 上=左脚 Lx 下=右脚 Rx; target 黑虚, real 蓝)
  1 张角速度 (wx/wy/wz)
  1 张重力分量 (gx/gy/gz)
图片名以时间段开头: 如 0-10s-joint1.png / 0-10s-angvel.png / 0-10s-gravity.png
用法: python plot_real_log.py <obs_xxx.csv> [窗口秒数=10]
"""
import sys, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# 中文字体 (系统装了 Noto Sans CJK), 修负号方框
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP",
                                   "WenQuanYi Zen Hei", "AR PL UMing CN", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CSV = sys.argv[1]
WIN = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0

d = np.genfromtxt(CSV, delimiter=",", names=True, dtype=None, encoding=None)
def c(k): return np.array(d[k], dtype=float)
t = c("t"); t = t - t[0]

# 数据时间戳 -> 同级文件夹
m = re.search(r"(\d{8}_\d{6})", os.path.basename(CSV))
stamp = m.group(1) if m else os.path.splitext(os.path.basename(CSV))[0]
OUT = os.path.join(os.path.dirname(os.path.abspath(CSV)), stamp)
os.makedirs(OUT, exist_ok=True)

# q/cmd 实机序: 0-5=L1..L6, 6-11=R1..R6
T = t[-1]
nwin = int(np.ceil(T / WIN))
n = 0
for w in range(nwin):
    t0, t1 = w * WIN, min((w + 1) * WIN, T)
    sel = (t >= t0) & (t < t1)
    if sel.sum() < 2:
        continue
    tt = t[sel]
    tag = f"{int(round(t0))}-{int(round(t1))}s"
    # ---- 6 张关节跟踪 (上 L, 下 R) ----
    for j in range(1, 7):
        li, ri = j - 1, j - 1 + 6
        fig, ax = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
        ax[0].plot(tt, c(f"cmd{li}")[sel], "k--", lw=1.6, label="target")
        ax[0].plot(tt, c(f"q{li}")[sel], "-", color="tab:blue", lw=1.6, label="real")
        ax[0].set_title(f"L{j}  (左脚)", fontsize=14); ax[0].set_ylabel("pos [rad]")
        ax[0].grid(alpha=.3); ax[0].legend(loc="upper right", fontsize=12)
        ax[1].plot(tt, c(f"cmd{ri}")[sel], "k--", lw=1.6, label="target")
        ax[1].plot(tt, c(f"q{ri}")[sel], "-", color="tab:red", lw=1.6, label="real")
        ax[1].set_title(f"R{j}  (右脚)", fontsize=14); ax[1].set_ylabel("pos [rad]")
        ax[1].grid(alpha=.3); ax[1].legend(loc="upper right", fontsize=12)
        ax[1].set_xlabel("t [s]")
        fig.suptitle(f"joint {j} 跟踪  {tag}  (target=黑虚, real=实线)", fontsize=15)
        plt.tight_layout(); plt.savefig(f"{OUT}/{tag}-joint{j}.png", dpi=110); plt.close(); n += 1
    # ---- 角速度 ----
    fig, a = plt.subplots(figsize=(18, 8))
    for k, col in zip(["wx", "wy", "wz"], ["tab:red", "tab:green", "tab:blue"]):
        a.plot(tt, c(k)[sel], lw=1.5, label=k, color=col)
    a.set_title(f"base angular velocity [rad/s]  {tag}", fontsize=15)
    a.set_xlabel("t [s]"); a.grid(alpha=.3); a.legend(loc="upper right", fontsize=12)
    plt.tight_layout(); plt.savefig(f"{OUT}/{tag}-angvel.png", dpi=110); plt.close(); n += 1
    # ---- 重力分量 ----
    fig, a = plt.subplots(figsize=(18, 8))
    for k, col in zip(["gx", "gy", "gz"], ["tab:red", "tab:green", "tab:blue"]):
        a.plot(tt, c(k)[sel], lw=1.5, label=k, color=col)
    a.axhline(-1, color="k", ls=":", lw=1.0)
    a.set_title(f"projected gravity  {tag}  (直立: gx,gy≈0, gz≈-1)", fontsize=15)
    a.set_xlabel("t [s]"); a.grid(alpha=.3); a.legend(loc="upper right", fontsize=12)
    plt.tight_layout(); plt.savefig(f"{OUT}/{tag}-gravity.png", dpi=110); plt.close(); n += 1

print(f"完成 {n} 张 -> {OUT}")
