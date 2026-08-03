import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DelayedPDActuatorCfg
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
        usd_path=os.path.join(_ASSET_DIR, "mjcf/legs_narrow/legs_narrow.usd"),
    ),
    # In A1_legs_V2_mjcf.usd the articulation root (PhysicsArticulationRootAPI) is on the
    # `base` body at /<defaultPrim>/base/base, so relative to the spawned Robot prim it is /base/base.
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
        "delayed_pd": DelayedPDActuatorCfg(
            joint_names_expr=[".*6"],
            stiffness=40.0,
            damping=0.5,
            armature=0.00219,
            effort_limit_sim=5.8,
            velocity_limit_sim=14.0,
            min_delay=4,
            max_delay=6,
        ),
        "delayed_dcmotor": DelayedDCMotorCfg(
            joint_names_expr=[".*1", ".*2", ".*3", ".*4", ".*5"],
            effort_limit=26.0,
            saturation_effort=26.0,
            velocity_limit=3.4,
            velocity_limit_sim=14.0,
            stiffness={
                ".*1": 200.0,
                ".*2": 200.0,
                ".*3": 200.0,
                ".*4": 250.0,
                ".*5": 40.0
            },
            damping={
                ".*1": 5.0,
                ".*2": 5.0,
                ".*3": 5.0,
                ".*4": 5.0,
                ".*5": 2.0
            },
            armature=0.0509,
            friction={
                ".*1": 0.0,
                ".*2": 0.0,
                ".*3": 0.0,
                ".*4": 0.0,
                "joint_L5": 0.4,
                "joint_R5": 0.6
            },
            dynamic_friction={
                ".*1": 0.0,
                ".*2": 0.0,
                ".*3": 0.0,
                ".*4": 0.0,
                "joint_L5": 0.4,
                "joint_R5": 0.6
            },
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
NLEGS_FIX_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUsdFileCfg(
        usd_path=os.path.join(_ASSET_DIR, "mjcf/legs_narrow/legs_narrow.usd"),
    ),
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
        "4310": DelayedPDActuatorCfg(
            joint_names_expr=[".*6"],
            stiffness=40.0,
            damping=0.5,
            armature=0.00219,
            effort_limit_sim=5.8,
            velocity_limit_sim=14.0,
            min_delay=4,
            max_delay=6,
        ),
        "4340": DelayedDCMotorCfg(
            joint_names_expr=[".*1", ".*2", ".*3", ".*4", ".*5"],
            effort_limit=26.0,
            saturation_effort=26.0,
            velocity_limit=3.4,
            stiffness={
                ".*1": 200.0,
                ".*2": 100.0,
                ".*3": 100.0,
                ".*4": 250.0,
                ".*5": 40.0
            },
            damping={
                ".*1": 5.0,
                ".*2": 5.0,
                ".*3": 5.0,
                ".*4": 5.0,
                ".*5": 2.0
            },
            armature=0.0509,
            friction=0.5,
            dynamic_friction=0.5,
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
