"""legs_static —— 在 legs_dr(域随机化)基础上, 让零速命令时保持 default 姿态静止, 而非原地踏步。

改动(相对 legs_dr):
  - 相位相关奖励(gait/feet_clearance)加"速度开关": |cmd|<0.1 时不给踏步/抬脚奖励;
  - 新增 stand_still 惩罚: 零速时惩罚关节偏离 default -> 拉回静止默认姿态;
  - gait 相位观测零速归零(gate_by_cmd): 让策略据此判断"该站住";
  - 提高 rel_standing_envs, 多练站立。
其余(域随机化/scene/其他奖励/gait 周期/机器人)全部继承 legs_dr。
"""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from legs_rl_lab.tasks.legs_task import mdp

from .legs_dr_env_cfg import RobotDREnvCfg, RobotDRPlayEnvCfg


def _apply_static(cfg) -> None:
    """把 legs_dr 配置改成'零速站立'语义。"""
    # 相位奖励只在有速度指令时生效(零速不逼踏步/抬脚)
    cfg.rewards.gait.params["moving_only"] = True
    cfg.rewards.feet_clearance.params["moving_only"] = True
    # 零速时惩罚关节偏离 default -> 保持静止默认姿态
    cfg.rewards.stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-1.0,
        params={"command_name": "base_velocity"},
    )
    # gait 相位观测零速归零(策略/critic 同步)
    cfg.observations.policy.gait_phase = ObsTerm(func=mdp.gait_phase_obs, params={"gate_by_cmd": True})
    cfg.observations.critic.gait_phase = ObsTerm(func=mdp.gait_phase_obs, params={"gate_by_cmd": True})
    # 多分配站立环境, 练好静止
    cfg.commands.base_velocity.rel_standing_envs = 0.25

    # base_height 加运动门控: 零速时不管高度(交给 stand_still 保持关节), 只在运动时约束
    cfg.rewards.base_height = RewTerm(
        func=mdp.base_height_l2_moving,
        weight=-5.0,
        params={"target_height": 0.45},
    )
    # 加大抖动/速度惩罚, 压制站立时膝盖高频颤动
    cfg.rewards.action_rate.weight = -0.3
    cfg.rewards.joint_vel.weight = -0.005
    cfg.rewards.feet_clearance.params["target_height"] = 0.2


@configclass
class RobotStaticEnvCfg(RobotDREnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_static(self)


@configclass
class RobotStaticPlayEnvCfg(RobotDRPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_static(self)
