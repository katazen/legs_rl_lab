"""rough_static：零速站稳，行走时保留 rough 地形；运行时应用实测单侧限位。"""

from isaaclab.utils import configclass

from legs_rl_lab.tasks.nlegs_task.agents.rsl_rl_ppo_cfg import NlegsRoughPPORunnerCfg
from legs_rl_lab.tasks.nlegs_task.task.flat.static_env_cfg import FlatStaticEnvCfg
from .rough_env_cfg import _apply_rough


@configclass
class RoughStaticEnvCfg(FlatStaticEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_rough(self, knee_stops=True)


@configclass
class RoughStaticPlayEnvCfg(RoughStaticEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.max_init_terrain_level = 4


@configclass
class NlegsRoughStaticPPORunnerCfg(NlegsRoughPPORunnerCfg):
    experiment_name = "nlegs_rough_static"
