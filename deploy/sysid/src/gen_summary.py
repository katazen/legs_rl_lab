#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成/更新实机 sysid 实验汇总文档 sysid/doc/实机实验汇总.md。

每次测试(data/real/<测试时间>/)作为一个 ## 子标题, 内容:
  - 本次测的关节 + PD 列表
  - 测量内容说明(自动占位, 你后续补充解释)
  - 附该次每个关节的实验图(png/)

已存在的 ## <测试时间> 小节不会被覆盖(保留你已填的说明); 只追加新测试。
先跑 plot_experiment.py 出图, 再跑本脚本。

用法: python3 sysid/src/gen_summary.py
"""
import os
import re

SYSID = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
REAL = os.path.join(SYSID, "data", "real")
DOC = os.path.join(SYSID, "doc", "实机实验汇总.md")
FN = re.compile(r"j(\d+)_kp([0-9.eE+-]+|NA)_kd([0-9.eE+-]+|NA)_(step_fwd|step_rev|sine)_cmd\.csv$")
PNG_FN = re.compile(r"^j\d+_kp[0-9.eENA+-]+_kd[0-9.eENA+-]+\.png$")   # 只认 plot_experiment 出的图
JOINT_NAME = ["L1髋pitch", "L2髋roll", "L3髋yaw", "L4膝", "L5踝pitch", "L6踝roll",
              "R1髋pitch", "R2髋roll", "R3髋yaw", "R4膝", "R5踝pitch", "R6踝roll"]


def session_section(ts):
    ddir = os.path.join(REAL, ts, "data")
    pdir = os.path.join(REAL, ts, "png")
    # 关节+PD
    groups = {}
    if os.path.isdir(ddir):
        for fn in sorted(os.listdir(ddir)):
            m = FN.search(fn)
            if m:
                groups.setdefault((int(m.group(1)), m.group(2), m.group(3)), set()).add(m.group(4))
    lines = [f"## {ts}\n"]
    if not groups:
        lines.append("_(该会话 data/ 下无有效 CSV)_\n")
        return "\n".join(lines)
    tested = ", ".join(f"{JOINT_NAME[J] if J < 12 else 'idx'+str(J)}(kp{kp}/kd{kd})"
                       for (J, kp, kd) in sorted(groups))
    lines.append(f"- **测试关节/PD**: {tested}")
    lines.append("- **测量内容**: 单关节 正向阶跃 + 反向阶跃 + 正弦扫频, 看该 PD 下的跟踪/延迟/超调。")
    lines.append("- **说明(待填)**: \n")
    # 附图(相对 doc/ 的路径)
    pngs = sorted(f for f in os.listdir(pdir) if PNG_FN.match(f)) if os.path.isdir(pdir) else []
    for png in pngs:
        rel = os.path.join("..", "data", "real", ts, "png", png)
        lines.append(f"![{png}]({rel})\n")
    if not pngs:
        lines.append("_(png/ 下暂无图, 先跑 plot_experiment.py)_\n")
    return "\n".join(lines)


def main():
    sessions = sorted(d for d in os.listdir(REAL) if os.path.isdir(os.path.join(REAL, d))) \
        if os.path.isdir(REAL) else []
    existing = ""
    if os.path.exists(DOC):
        existing = open(DOC, encoding="utf-8").read()
    else:
        existing = "# 实机 sysid 实验汇总\n\n每次测试按时间为小节; 单关节 正反阶跃 + 正弦扫频。\n\n"

    added = []
    for ts in sessions:
        if re.search(rf"^## {re.escape(ts)}\b", existing, re.M):
            continue                       # 已有该小节, 保留(不覆盖你填的说明)
        existing = existing.rstrip() + "\n\n" + session_section(ts) + "\n"
        added.append(ts)

    os.makedirs(os.path.dirname(DOC), exist_ok=True)
    open(DOC, "w", encoding="utf-8").write(existing)
    print(f"汇总 -> {DOC}")
    print(f"新增小节: {added if added else '(无, 均已存在)'}")


if __name__ == "__main__":
    main()
