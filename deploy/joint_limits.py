"""MJCF limits for standalone experiment tools; not used by the RL node."""
import xml.etree.ElementTree as ET
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
