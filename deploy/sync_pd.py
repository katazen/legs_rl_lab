#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从训练 run 的 deploy.yaml 同步 PD 到 armcontrol 参数 yaml。

单一真源 = deploy.yaml 的 stiffness/damping (训练导出)。
- --config 与部署节点选择同一份完整配置，默认 common.yaml。
- 顺序换算: deploy.yaml 为 SDK/mjc 序 [R1..R6, L1..L6];
  armcontrol 双腿为 [L1..L6, R1..R6] -> 交换两个 6 元块。
- 写入 armcontrol yaml 的【源】与【install】两份 (armcontrol 运行时读 install)。
  max_vel 与 enable_dual_leg_diag 保留原值 (部署侧选择, 不在 deploy.yaml)。

在 start_real.sh 启动 armcontrol 之前调用。
"""
import os
import argparse
from pathlib import Path
import sys

import numpy as np
import yaml

H1 = os.path.dirname(os.path.abspath(__file__))   # deploy 目录(本文件所在)
COMMON_SRC = f"{H1}/rl_real_py/configs/common.yaml"
ARM_SRC = f"{H1}/control_ws/src/armcontrol/config/arm_control_node.yaml"
ARM_INSTALL = f"{H1}/control_ws/install/armcontrol/share/armcontrol/config/arm_control_node.yaml"
sys.path.insert(0, str(Path(H1) / "rl_real_py"))
from rl_real_py.deployment_config import check_multi_pd, load_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=COMMON_SRC)
    parser.add_argument("--check-only", action="store_true", help="只校验和显示，不修改 PD 配置")
    args = parser.parse_args()
    cfg, run_dir, _ = load_settings(args.config)
    check_multi_pd(cfg)
    with (run_dir / "params/deploy.yaml").open() as stream:
        dep = yaml.safe_load(stream)

    st = [float(x) for x in dep["stiffness"]]
    dm = [float(x) for x in dep["damping"]]
    sdk = dep["joint_names"]
    if len(sdk) != 12 or len(set(sdk)) != 12 or len(st) != 12 or len(dm) != 12:
        raise ValueError("PD 必须覆盖 12 个唯一关节")
    order = [sdk.index("joint_" + name) for name in cfg["joint_index_in_real"]]
    kps, kds = [st[i] for i in order], [dm[i] for i in order]
    if not np.isfinite([kps, kds]).all() or np.any(np.asarray([kps, kds]) < 0):
        raise ValueError("PD 参数必须为有限非负数")

    # 保留部署侧参数
    max_vel = 0.0
    enable_diag = False
    if os.path.exists(ARM_SRC):
        prev = yaml.safe_load(open(ARM_SRC)) or {}
        params = prev.get("armcontrol_node", {}).get("ros__parameters", {})
        max_vel = params.get("max_vel", 0.0)
        enable_diag = params.get("enable_dual_leg_diag", False)
    if max_vel != 0.0:
        raise ValueError("训练为零目标速度 PD；armcontrol max_vel 必须为 0，不能额外添加差分速度前馈")

    text = (
        "armcontrol_node:\n"
        "  ros__parameters:\n"
        f"    # 本文件由 sync_pd.py 从 {run_dir.name}/params/deploy.yaml 自动生成, 勿手改\n"
        f"    kps: {kps}\n"
        f"    kds: {kds}\n"
        f"    max_vel: {max_vel}\n"
        f"    enable_dual_leg_diag: {str(enable_diag).lower()}\n"
    )
    if not args.check_only:
        if not all(Path(p).parent.is_dir() for p in (ARM_SRC, ARM_INSTALL)):
            raise FileNotFoundError("armcontrol 源/安装配置目录缺失，请先编译")
        for p in (ARM_SRC, ARM_INSTALL):
            with open(p, "w") as f:
                f.write(text)
            print(f"[sync_pd] 写入 {p}")
    else:
        print("[sync_pd] CHECK ONLY：未修改文件")
    print(f"[sync_pd] walk={run_dir}")
    print(f"[sync_pd] kps={kps}")
    print(f"[sync_pd] kds={kds}  max_vel={max_vel}  enable_dual_leg_diag={enable_diag}")


if __name__ == "__main__":
    main()
