"""rough_nogait：复杂地形盲走，以接触和离地高度奖励迈步，不提供步态时钟。"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import feet_air_time_positive_biped

from legs_rl_lab.tasks.nlegs_task import mdp
from legs_rl_lab.tasks.nlegs_task.agents.rsl_rl_ppo_cfg import NlegsRoughPPORunnerCfg

from .rough_env_cfg import RewardsCfg, RoughEnvCfg, RoughPlayEnvCfg


@configclass
class NoGaitRewardsCfg(RewardsCfg):
    gait = None
    feet_air_time = RewTerm(
        func=feet_air_time_positive_biped,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "threshold": 0.3,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Link_R6", "Link_L6"], preserve_order=True),
        },
    )
    feet_clearance = RewTerm(
        func=mdp.feet_clearance_nogait,
        weight=1.0,
        params={
            "target_height": 0.12,
            "std": 0.05,
            "asset_cfg": SceneEntityCfg("robot", body_names=["Link_R6", "Link_L6"], preserve_order=True),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["Link_R6", "Link_L6"], preserve_order=True
            ),
        },
    )


def _remove_gait(cfg):
    cfg.gait = None
    cfg.observations.policy.gait_phase = None
    cfg.observations.critic.gait_phase = None
    # 接触传感器和资产的默认 body 顺序不同，滑移项也必须逐脚对齐。
    for key in ("asset_cfg", "sensor_cfg"):
        cfg.rewards.feet_slide.params[key].body_names = ["Link_R6", "Link_L6"]
        cfg.rewards.feet_slide.params[key].preserve_order = True


@configclass
class RoughNoGaitEnvCfg(RoughEnvCfg):
    rewards: NoGaitRewardsCfg = NoGaitRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        _remove_gait(self)


@configclass
class RoughNoGaitPlayEnvCfg(RoughPlayEnvCfg):
    rewards: NoGaitRewardsCfg = NoGaitRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        _remove_gait(self)


@configclass
class RoughNoGaitPPORunnerCfg(NlegsRoughPPORunnerCfg):
    experiment_name = "rough_nogait"
    resume = False

    def __post_init__(self):
        super().__post_init__()
        # 原镜像布局含相位，不能用于无相位观测；保留 rough 的 16 个 mini-batch。
        self.algorithm.symmetry_cfg = None
