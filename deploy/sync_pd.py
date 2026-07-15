#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从训练 run 的 deploy.yaml 同步 PD 到 armcontrol 参数 yaml。

单一真源 = deploy.yaml 的 stiffness/damping (训练导出)。
- run 取自 rl_real 的 common.yaml (与部署节点用同一个)。
- 顺序换算: deploy.yaml 为 SDK/mjc 序 [R1..R6, L1..L6];
  armcontrol 双腿为 [L1..L6, R1..R6] -> 交换两个 6 元块。
- 写入 armcontrol yaml 的【源】与【install】两份 (armcontrol 运行时读 install)。
  max_vel 保留原值 (部署侧选择, 不在 deploy.yaml)。

在 start_real.sh 启动 armcontrol 之前调用。
"""
import os
import yaml

H1 = os.path.dirname(os.path.abspath(__file__))   # deploy 目录(本文件所在)
COMMON_INSTALL = f"{H1}/rl_real_py/install/rl_real_py/share/rl_real_py/configs/common.yaml"
COMMON_SRC = f"{H1}/rl_real_py/configs/common.yaml"
ARM_SRC = f"{H1}/control_ws/src/armcontrol/config/arm_control_node.yaml"
ARM_INSTALL = f"{H1}/control_ws/install/armcontrol/share/armcontrol/config/arm_control_node.yaml"


def main():
    common = COMMON_INSTALL if os.path.exists(COMMON_INSTALL) else COMMON_SRC
    cfg = yaml.safe_load(open(common))
    logs_root = cfg["logs_root"]
    if not os.path.isabs(logs_root):
        logs_root = os.path.join(os.path.dirname(H1), logs_root)   # H1=deploy → 上级=legs_rl_lab 根
    run_dir = os.path.join(logs_root, cfg["run"])
    dep = yaml.safe_load(open(os.path.join(run_dir, "params", "deploy.yaml")))

    st = [float(x) for x in dep["stiffness"]]
    dm = [float(x) for x in dep["damping"]]
    # SDK 序 [R1..R6, L1..L6] -> armcontrol 序 [L1..L6, R1..R6]
    kps = st[6:12] + st[0:6]
    kds = dm[6:12] + dm[0:6]

    # 保留原 max_vel
    max_vel = 0.0
    if os.path.exists(ARM_SRC):
        prev = yaml.safe_load(open(ARM_SRC)) or {}
        max_vel = prev.get("armcontrol_node", {}).get("ros__parameters", {}).get("max_vel", 0.0)

    text = (
        "armcontrol_node:\n"
        "  ros__parameters:\n"
        f"    # 本文件由 sync_pd.py 从 {cfg['run']}/params/deploy.yaml 自动生成, 勿手改\n"
        f"    kps: {kps}\n"
        f"    kds: {kds}\n"
        f"    max_vel: {max_vel}\n"
    )
    for p in (ARM_SRC, ARM_INSTALL):
        if os.path.isdir(os.path.dirname(p)):
            with open(p, "w") as f:
                f.write(text)
            print(f"[sync_pd] 写入 {p}")
    print(f"[sync_pd] run={cfg['run']}")
    print(f"[sync_pd] kps={kps}")
    print(f"[sync_pd] kds={kds}  max_vel={max_vel}")


if __name__ == "__main__":
    main()
