"""窄本体变体 (nlegs)：完全复用 legs 的 env 配置，仅两处不同——
机器人资产 (A1_legs_V2_narrow, 脚间距 0.2) 和 feet_y_distance 的目标间距。
其余 scene / events / rewards / observations / commands / 步态参数全部继承 legs。
"""

from isaaclab.utils import configclass

from legs_rl_lab.assets.legs_URDF.nlegs import NLEGS_CFG
from legs_rl_lab.tasks.legs_task.task.legs.legs_env_cfg import (
    RobotEnvCfg as LegsEnvCfg,
    RobotPlayEnvCfg as LegsPlayEnvCfg,
)


def _apply_nlegs(cfg) -> None:
    """把 legs 配置改成窄本体：换机器人资产 + 改 feet_y_distance 目标间距。"""
    cfg.scene.robot = NLEGS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.rewards.feet_y_distance.params["threshold"] = 0.2


@configclass
class RobotEnvCfg(LegsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_nlegs(self)


@configclass
class RobotPlayEnvCfg(LegsPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_nlegs(self)
