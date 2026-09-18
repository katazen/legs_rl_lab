"""nlegs 窄本体机器人定义(自包含资产包).

从 assets/legs_narrow 复制并改名而来, 只保留 nlegs_flat 任务实际使用的配置链:
nlegs.xml(MJCF) -> mjcf/nlegs/nlegs.usd(--import-sites 转换, 与 legs_narrow.usd 逐字节同源).

与 legs_narrow.NLEGS_FIX_CFG 的差异: 踝 pitch(.*5) 从 4340 组拆出单独的 "ankle" 组,
参数直接在这里定义(原先在 env cfg 的 __post_init__ 里运行时拆分):
- 摩擦 0.5->0.55 (48V 辨识回线复现)
- 延迟 max 6->8 (20~40ms, 覆盖实测起动延迟 14-18ms + 余量)
- 明确不碰 armature —— 抬它会欠阻尼过冲、放大幅值, 与实机磨圆衰减反向。
踝增益域随机化(randomize_ankle_gains)属于 DR, 放在任务的 EventCfg 里, 不在这里。
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from legs_rl_lab.actuators import DelayedDCMotorCfg  # 自定义: 延迟 + 转矩-转速滚降(膝辨识用)

# 资产相对本文件定位，避免硬编码绝对路径（换机器/换用户名都能用）
_ASSET_DIR = os.path.dirname(os.path.abspath(__file__))


@configclass
class UnitreeArticulationCfg(ArticulationCfg):
    """Configuration for Unitree articulations."""

    joint_sdk_names: list[str] = None
    soft_joint_pos_limit_factor = 0.95


@configclass
class UnitreeUsdFileCfg(sim_utils.UsdFileCfg):
    activate_contact_sensors: bool = True
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
    )


NLEGS_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUsdFileCfg(
        usd_path=os.path.join(_ASSET_DIR, "mjcf/nlegs_limit/nlegs_limit.usd"),
    ),
    # articulation root (PhysicsArticulationRootAPI) 在 `base` body 上,
    # 即 /<defaultPrim>/base/base, 相对 spawn 出的 Robot prim 是 /base/base。
    articulation_root_prim_path='/base/base',
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.62),  # 次姿态站立本体高度大约0.58
        joint_pos={
            ".*1": -0.1,
            ".*4": 0.2,
            ".*5": -0.1,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": DelayedDCMotorCfg(
            joint_names_expr=[".*1", ".*2", ".*3", ".*4"],
            effort_limit=26.0,
            saturation_effort=26.0,
            velocity_limit=7.0,
            stiffness={
                ".*1": 200.0,
                ".*2": 100.0,
                ".*3": 100.0,
                ".*4": 250.0,
            },
            damping={
                ".*1": 5.0,
                ".*2": 5.0,
                ".*3": 5.0,
                ".*4": 5.0,
            },
            armature=0.0509,
            friction=0.5,
            dynamic_friction=0.5,
            min_delay=4,
            max_delay=6,
        ),
        # 踝 pitch 单独一组(原 _apply_ankle_dr 运行时拆分, 现直接定义)
        "ankle_pitch": DelayedDCMotorCfg(
            joint_names_expr=[".*5"],
            effort_limit=26.0,
            saturation_effort=26.0,
            velocity_limit=7.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.0509,
            friction=0.55,
            dynamic_friction=0.55,
            min_delay=4,
            max_delay=8,
        ),
        "ankle_roll": DelayedPDActuatorCfg(
            joint_names_expr=[".*6"],
            stiffness=40.0,
            damping=0.5,
            armature=0.00219,
            effort_limit_sim=5.8,
            velocity_limit_sim=14.0,
            viscous_friction=0.54,  # 踝roll 粘滞摩擦, Stage3B 回放验证(RMSE 0.19->0.05)
            min_delay=4,
            max_delay=6,
        ),
    },
    joint_sdk_names=['joint_R1',
                     'joint_R2',
                     'joint_R3',
                     'joint_R4',
                     'joint_R5',
                     'joint_R6',
                     'joint_L1',
                     'joint_L2',
                     'joint_L3',
                     'joint_L4',
                     'joint_L5',
                     'joint_L6'],
)
