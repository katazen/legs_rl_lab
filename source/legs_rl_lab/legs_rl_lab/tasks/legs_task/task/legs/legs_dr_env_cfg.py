"""legs_dr —— 在 legs 基础上补充域随机化(对齐 engineai 走路任务的 DR 项)。

只新增/加强域随机化，其余(scene/rewards/obs/commands/gait)全部继承 legs。
新增项(legs 原本没有的)：
  - 连杆质量整体缩放(所有 body)      randomize_rigid_body_mass  scale
  - base 质心(COM)偏移               randomize_rigid_body_com
  - PD 增益(刚度/阻尼)缩放           randomize_actuator_gains   scale
  - 关节摩擦 / armature 随机          randomize_joint_parameters
legs 已有(继承, 不重复)：地面/材质摩擦、base 质量加减、电机零位偏置(joint_zero_bias)、随机推力。
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from legs_rl_lab.tasks.legs_task import mdp

from .legs_env_cfg import EventCfg, RobotEnvCfg, RobotPlayEnvCfg


@configclass
class DomainRandEventCfg(EventCfg):
    """在 legs 的 EventCfg 上追加域随机化项。"""

    # 连杆质量整体缩放(所有 body) [0.9, 1.1]，模拟建模/装配质量误差
    randomize_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # base 质心偏移 [-0.05, 0.05] m，模拟负载/装配质心不准
    randomize_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # PD 增益(刚度/阻尼)相对默认值缩放 [0.8, 1.2]，模拟电机增益不确定
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # 关节摩擦 / armature 随机(绝对值范围，对齐 engineai)
    randomize_joint_params = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class RobotDREnvCfg(RobotEnvCfg):
    """legs + 域随机化。"""

    events: DomainRandEventCfg = DomainRandEventCfg()


@configclass
class RobotDRPlayEnvCfg(RobotPlayEnvCfg):
    """play/eval：继承 legs 的 play(少环境、命令放开)，不加额外 DR。"""

    pass
