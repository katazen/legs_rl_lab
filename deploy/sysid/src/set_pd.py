#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PD 辨识网格用: 给单个关节设定 kp/kd, 写进 armcontrol 参数 yaml(源+install)。

只改指定关节的一项, 其余保留。改完需重启 armcontrol 才生效。
关节 idx = 实机序 L1..L6,R1..R6 (0-11)。--both 同时设对称腿的同名关节。

用法:
  python3 set_pd.py --joint 0 --kp 200 --kd 5
  python3 set_pd.py --joint 4 --kp 40 --kd 0.5 --both     # L5+R5 一起设
  python3 set_pd.py --show                                # 只打印当前 arm yaml 的 PD

辨识全部跑完后, 用 sync_pd.py 可把 PD 恢复回 deploy.yaml 的值。
"""
import argparse
import os
import yaml

H1 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))   # deploy 目录 (本文件在 deploy/sysid/src/)
ARM_SRC = f"{H1}/control_ws/src/armcontrol/config/arm_control_node.yaml"
ARM_INSTALL = f"{H1}/control_ws/install/armcontrol/share/armcontrol/config/arm_control_node.yaml"


def load():
    src = ARM_SRC if os.path.exists(ARM_SRC) else ARM_INSTALL
    p = yaml.safe_load(open(src))["armcontrol_node"]["ros__parameters"]
    kps = [float(x) for x in p["kps"]]
    kds = [float(x) for x in p["kds"]]
    max_vel = p.get("max_vel", 0.0)
    return kps, kds, max_vel


def write(kps, kds, max_vel, note):
    text = (
        "armcontrol_node:\n"
        "  ros__parameters:\n"
        f"    # {note}\n"
        f"    kps: {kps}\n"
        f"    kds: {kds}\n"
        f"    max_vel: {max_vel}\n"
    )
    for pth in (ARM_SRC, ARM_INSTALL):
        if os.path.isdir(os.path.dirname(pth)):
            open(pth, "w").write(text)
            print(f"[set_pd] 写入 {pth}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint", type=int, help="实机序 idx 0-11 (L1..L6,R1..R6)")
    ap.add_argument("--kp", type=float)
    ap.add_argument("--kd", type=float)
    ap.add_argument("--both", action="store_true", help="同时设对称腿的同名关节")
    ap.add_argument("--show", action="store_true", help="只打印当前 PD")
    a = ap.parse_args()

    kps, kds, max_vel = load()
    if a.show:
        print(f"kps={kps}\nkds={kds}\nmax_vel={max_vel}")
        return
    assert a.joint is not None and a.kp is not None and a.kd is not None, "需 --joint --kp --kd"
    js = [a.joint]
    if a.both:
        js.append(a.joint + 6 if a.joint < 6 else a.joint - 6)
    for j in js:
        kps[j] = a.kp
        kds[j] = a.kd
    write(kps, kds, max_vel, f"PD辨识手动设定 joint={js} kp={a.kp} kd={a.kd}(勿用于部署, 恢复用 sync_pd.py)")
    print(f"[set_pd] 已设 joint {js}: kp={a.kp} kd={a.kd}  → 重启 armcontrol 生效")


if __name__ == "__main__":
    main()
