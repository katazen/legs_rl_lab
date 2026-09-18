"""nlegs_flat_static：有速度命令时行走，零速时双脚落地、回默认站姿并停稳。"""

from pathlib import Path

from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

from legs_rl_lab.assets.nlegs import nlegs
from legs_rl_lab.tasks.nlegs_task.agents.rsl_rl_ppo_cfg import NlegsFlatPPORunnerCfg
from .flat_env_cfg import FlatEnvCfg, RewardsCfg
from . import static_mdp


@configclass
class StaticRewardsCfg(RewardsCfg):
    base_height = RewTerm(func=static_mdp.moving_base_height_l2, weight=-5.0, params={"target_height": 0.58})
    standing_joint_pos = RewTerm(func=static_mdp.standing_joint_pos_l1, weight=-1.0)
    standing_joint_vel = RewTerm(func=static_mdp.standing_joint_vel_l2, weight=-0.02)
    standing_feet_contact = RewTerm(
        func=static_mdp.standing_feet_contact, weight=0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*6")},
    )


@configclass
class FlatStaticEnvCfg(FlatEnvCfg):
    rewards: StaticRewardsCfg = StaticRewardsCfg()
    # 仅容忍数值残差；小速度、纯转向仍是行走，观测与奖励共用同一个阈值。
    command_threshold: float = 1e-6

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.usd_path = str(Path(nlegs.__file__).parent / "mjcf/nlegs_limit/nlegs_limit.usd")
        self.commands.base_velocity.rel_standing_envs = 0.35
        self.commands.base_velocity.resampling_time_range = (3.0, 6.0)
        for group in (self.observations.policy, self.observations.critic):
            group.gait_phase.params.update(gate_by_cmd=True, command_threshold=self.command_threshold)
        for name in ("gait", "feet_clearance"):
            getattr(self.rewards, name).params.update(moving_only=True, command_threshold=self.command_threshold)
        for name in ("base_height", "standing_joint_pos", "standing_joint_vel", "standing_feet_contact"):
            getattr(self.rewards, name).params["command_threshold"] = self.command_threshold
        # 保留速度跟踪、平衡、防滑和扰动；静止是软奖励，受扰动时仍可迈步救回平衡。


@configclass
class FlatStaticPlayEnvCfg(FlatStaticEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False


@configclass
class NlegsFlatStaticPPORunnerCfg(NlegsFlatPPORunnerCfg):
    experiment_name = "nlegs_flat_static"
