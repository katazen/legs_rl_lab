"""Shared configuration resolution for the RL node and PD synchronization."""

from pathlib import Path
import xml.etree.ElementTree as ET

import yaml
import numpy as np


def read_joint_limits(xml_path, joint_names):
    """Read MJCF hinge ranges in the requested hardware order (radians)."""
    tree = ET.parse(xml_path).getroot()
    compiler = tree.find("compiler")
    angle = compiler.get("angle", "degree") if compiler is not None else "degree"
    if angle not in ("radian", "degree"):
        raise ValueError(f"{xml_path}: 无效 compiler angle={angle}")
    joints = [j for j in tree.findall(".//worldbody//joint") if j.get("name")]
    by_name = {j.get("name"): j for j in joints}
    expected = {"joint_" + name for name in joint_names}
    if len(joint_names) != 12 or len(expected) != 12 or len(by_name) != len(joints) or set(by_name) != expected:
        raise ValueError(f"{xml_path}: 必须包含与实机匹配的 12 个唯一关节")
    limits = []
    for name in joint_names:
        joint = by_name["joint_" + name]
        if joint.get("type", "hinge") != "hinge" or joint.get("limited") == "false":
            raise ValueError(f"{xml_path}: {name} 必须为有限位的 hinge")
        limits.append([float(value) for value in joint.get("range", "").split()])
    limits = np.asarray(limits, dtype=float)
    if limits.shape != (12, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError(f"{xml_path}: 关节 range 无效")
    if angle == "degree":
        limits = np.deg2rad(limits)
    return limits[:, 0].tolist(), limits[:, 1].tolist()


def load_settings(config_file):
    path = Path(config_file).expanduser().resolve()
    with path.open() as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict) or not isinstance(cfg.get("tasks"), dict) or set(cfg["tasks"]) != {"walk", "crouch", "rise"}:
        raise ValueError("配置必须包含 walk / crouch / rise 三个任务；后两者可设为 null 禁用")
    for name, value in cfg["tasks"].items():
        if name != "walk" and value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"tasks.{name} 必须是模型路径；仅 crouch / rise 可设为 null 禁用")
    roots = (*path.parents, *Path(__file__).resolve().parents)
    root = next((p for p in roots if (p / "source/legs_rl_lab").is_dir()), None)
    if root is None:
        raise ValueError("找不到 legs_rl_lab 根目录")
    cfg["tasks"] = {name: None if value is None else str((root / Path(value).expanduser()).resolve())
                    for name, value in cfg["tasks"].items()}
    xml = root / "source/legs_rl_lab/legs_rl_lab/assets/nlegs/mjcf/nlegs_limit.xml"
    # Legacy YAML arrays never override the XML; all consumers share the parsed ranges.
    cfg["joint_lower_limits"], cfg["joint_upper_limits"] = read_joint_limits(xml, cfg["joint_index_in_real"])
    cfg["joint_limits_xml"] = str(xml)
    return cfg, Path(cfg["tasks"]["walk"]), root


def check_multi_pd(cfg):
    """驱动只设一次 PD；按关节名核对启用的模型，禁止带着旧 PD 切模型。"""
    baseline = None
    for name, path in cfg["tasks"].items():
        if path is None:
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
