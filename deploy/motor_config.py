"""Self-developed motor mapping and feedback-frame checks, without hardware access."""
from pathlib import Path

import numpy as np
import yaml

CONFIG = Path(__file__).resolve().parent / "control_ws/src/armcontrol/config/motors.yaml"
REAL_NAMES = [f"{side}{i}" for side in "LR" for i in range(1, 7)]


def load_motors(path=CONFIG):
    cfg = yaml.safe_load(Path(path).read_text())
    if set(cfg["models"]) != {"large", "small"}:
        raise ValueError("电机型号必须为 large / small")
    for limits in cfg["models"].values():
        values = [limits[key] for key in ("position_range", "velocity_range", "effort_range")]
        if not np.isfinite(values).all() or np.any(np.asarray(values) <= 0):
            raise ValueError("协议编码范围必须为有限正数")
    joints = cfg["joints"]
    if [j["name"] for j in joints] != REAL_NAMES:
        raise ValueError("电机顺序必须为 L1..L6,R1..R6")
    for i, joint in enumerate(joints):
        if joint["model"] != ("small" if i % 6 == 5 else "large"):
            raise ValueError("只有 ankle roll 使用 small")
        if joint["direction"] not in (-1, 1):
            raise ValueError("direction 必须为 +/-1")
        if (joint["slave_id"], joint["master_id"]) != (
                [1, 2, 3, 4, 6, 7][i % 6], [17, 18, 19, 20, 22, 23][i % 6]):
            raise ValueError("CAN ID 与双腿接线映射不符")
    return cfg


FB_SIGN = np.array([j["direction"] for j in load_motors()["joints"]], dtype=float)


def detect_effort_frame(pd_effort, feedback):
    """Check frame consistency; correlations cannot establish physical joint direction."""
    pd_effort, feedback = np.asarray(pd_effort), np.asarray(feedback)
    scores = []
    for i in np.flatnonzero(FB_SIGN < 0):
        a, b = pd_effort[:, i], feedback[:, i]
        valid = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > .5)
        if valid.sum() < 20 or np.std(a[valid]) < 1e-9 or np.std(b[valid]) < 1e-9:
            continue
        corr = float(np.corrcoef(a[valid], b[valid])[0, 1])
        if abs(corr) >= .3:
            scores.append(corr)
    if not scores or abs(np.mean(scores)) < .3:
        raise ValueError("effort 坐标无法判定：激励不足或方向相关性不一致")
    if np.mean(scores) < 0:
        return FB_SIGN.copy(), "电机帧 effort；乘 direction 后转为关节帧"
    return np.ones(12), "关节帧 effort；无需再次取反"
