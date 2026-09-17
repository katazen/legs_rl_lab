"""Shared configuration resolution for the RL node and PD synchronization."""

from pathlib import Path

import yaml
import numpy as np


def load_settings(config_file):
    path = Path(config_file).expanduser().resolve()
    with path.open() as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict) or not isinstance(cfg.get("tasks"), dict) or set(cfg["tasks"]) != {"walk", "crouch", "rise"}:
        raise ValueError("配置必须包含 walk / crouch / rise 三个 tasks 路径；旧单策略配置已退役")
    for name, value in cfg["tasks"].items():
        if name == "crouch" and value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"tasks.{name} 必须是模型路径；仅 crouch 可设为 null 禁用")
    roots = (*path.parents, *Path(__file__).resolve().parents)
    root = next((p for p in roots if (p / "source/legs_rl_lab").is_dir()), None)
    if root is None:
        raise ValueError("找不到 legs_rl_lab 根目录")
    cfg["tasks"] = {name: None if value is None else str((root / Path(value).expanduser()).resolve())
                    for name, value in cfg["tasks"].items()}
    return cfg, Path(cfg["tasks"]["walk"]), root


def check_multi_pd(cfg):
    """驱动只设一次 PD；按关节名核对启用的模型，禁止带着旧 PD 切模型。"""
    baseline = None
    for name, path in cfg["tasks"].items():
        if name == "crouch" and path is None:
            continue
        with (Path(path) / "params/deploy.yaml").open() as stream:
            dep = yaml.safe_load(stream)
        sdk = dep["joint_names"]
        if len(sdk) != 12 or len(set(sdk)) != 12:
            raise ValueError(f"{name} 必须包含 12 个唯一关节")
        order = [sdk.index("joint_" + n) for n in cfg["joint_index_in_real"]]
        values = np.r_[np.asarray(dep["stiffness"])[order], np.asarray(dep["damping"])[order], dep["step_dt"]]
        if not np.isfinite(values).all() or np.any(values < 0) or dep["step_dt"] <= 0:
            raise ValueError(f"{name} PD / step_dt 无效")
        if baseline is not None and not np.array_equal(values, baseline):
            raise ValueError(f"{name} 的 PD 或策略周期不同，不能共用驱动切换")
        baseline = values
