#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单关节 sysid 实验出图: 每个关节的一次实验(同 kp/kd)一张图, 3 子图。

一次"实验" = 单关节 正向阶跃(step_fwd) + 反向阶跃(step_rev) + 正弦扫频(sine)。
图: 3 子图分别画 正阶跃 / 反阶跃 / 正弦扫频 的 target(cmd, 黑虚) vs real(q, 蓝) 跟踪。

输入: 一次测试的会话目录 sysid/data/real/<测试时间>/  (内含 data/*.csv)
输出: 会话目录下 png/j{J}_kp{KP}_kd{KD}.png

用法:
  python3 sysid/src/plot_experiment.py sysid/data/real/<测试时间>
  python3 sysid/src/plot_experiment.py sysid/data/real/<测试时间> --joints 0,3,4
"""
import argparse
import csv
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP",
                                   "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

JOINT_NAME = ["L1 髋pitch", "L2 髋roll", "L3 髋yaw", "L4 膝", "L5 踝pitch", "L6 踝roll",
              "R1 髋pitch", "R2 髋roll", "R3 髋yaw", "R4 膝", "R5 踝pitch", "R6 踝roll"]
MODES = [("step_fwd", "正向阶跃"), ("step_rev", "反向阶跃"), ("sine", "正弦扫频")]
FN = re.compile(r"j(\d+)_kp([0-9.eE+-]+|NA)_kd([0-9.eE+-]+|NA)_(step_fwd|step_rev|sine)_cmd\.csv$")


def read(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != "phase"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", help="会话目录 sysid/data/real/<测试时间>")
    ap.add_argument("--joints", default="all", help="'all' 或 '0,3,4'")
    a = ap.parse_args()
    ddir = os.path.join(a.session, "data")
    pdir = os.path.join(a.session, "png")
    os.makedirs(pdir, exist_ok=True)

    # 收集 (J,kp,kd) -> {mode: base_tag}
    groups = {}
    for fn in sorted(os.listdir(ddir)):
        m = FN.search(fn)
        if not m:
            continue
        J = int(m.group(1))
        key = (J, m.group(2), m.group(3))
        groups.setdefault(key, {})[m.group(4)] = fn[:-len("_cmd.csv")]

    want = None if a.joints == "all" else set(int(x) for x in a.joints.split(","))
    n = 0
    for (J, kp, kd), modes in sorted(groups.items()):
        if want is not None and J not in want:
            continue
        fig, ax = plt.subplots(1, 3, figsize=(21, 6))
        for i, (mode, label) in enumerate(MODES):
            if mode not in modes:
                ax[i].set_title(f"{label} (无数据)"); ax[i].grid(alpha=.3); continue
            tag = modes[mode]
            cmd = read(os.path.join(ddir, f"{tag}_cmd.csv"))
            st = read(os.path.join(ddir, f"{tag}_state.csv"))
            ax[i].plot(cmd["t"], cmd[f"qd{J}"], "k--", lw=1.6, label="target")
            ax[i].plot(st["t"], st[f"q{J}"], "-", color="tab:blue", lw=1.4, label="real")
            ax[i].set_title(label, fontsize=13)
            ax[i].set_xlabel("t [s]"); ax[i].grid(alpha=.3); ax[i].legend(loc="upper right", fontsize=11)
        ax[0].set_ylabel("pos [rad]")
        nm = JOINT_NAME[J] if J < 12 else f"idx{J}"
        fig.suptitle(f"{nm}  (idx{J})  kp={kp}/kd={kd}   target=黑虚线  real=蓝实线", fontsize=15)
        plt.tight_layout()
        out = os.path.join(pdir, f"j{J}_kp{kp}_kd{kd}.png")
        plt.savefig(out, dpi=110); plt.close(); n += 1
        print(f"  {os.path.basename(out)}")
    print(f"完成 {n} 张 -> {pdir}")


if __name__ == "__main__":
    main()
