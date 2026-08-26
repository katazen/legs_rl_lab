import os

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from legs_rl_lab.actuators import DelayedDCMotorCfg
from isaaclab.actuators import ImplicitActuatorCfg


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
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )


NLEGS_CODEX_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUsdFileCfg(
        usd_path=os.path.join(_ASSET_DIR, "mjcf/legs_narrow/legs_narrow.usd"),
    ),
    articulation_root_prim_path="/base/base",
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.62),
        joint_pos={
            ".*1": -0.1,
            ".*4": 0.2,
            ".*5": -0.1,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # 前 10 个 48:1 行星电机: 客观硬件参数保持一致, 左右差异交给 DR。
        "motor_48": DelayedDCMotorCfg(
            joint_names_expr=[".*1", ".*2", ".*3", ".*4", ".*5"],
            effort_limit=26.0,
            saturation_effort=26.0,
            velocity_limit=3.4,
            velocity_limit_sim=14.0,
            stiffness={
                ".*1": 200.0,
                ".*2": 100.0,
                ".*3": 100.0,
                ".*4": 250.0,
                ".*5": 40.0,
            },
            damping={
                ".*1": 5.0,
                ".*2": 5.0,
                ".*3": 5.0,
                ".*4": 5.0,
                ".*5": 2.0,
            },
            armature=0.0509,
            friction=0.5,
            dynamic_friction=0.5,
            min_delay=4,
            max_delay=6,
        ),
        # 踝 roll 是 10:1 行星电机: 同一转子惯量按减速比反射, 力矩/速度按 10:1 侧建模。
        "motor_10": DelayedDCMotorCfg(
            joint_names_expr=[".*6"],
            effort_limit=5.8,
            saturation_effort=5.8,
            velocity_limit=16.3,
            velocity_limit_sim=20.0,
            stiffness=40.0,
            damping=0.5,
            armature=0.00219,
            friction=0.5,
            dynamic_friction=0.5,
            min_delay=4,
            max_delay=6,
        ),
    },
    joint_sdk_names=[
        "joint_R1",
        "joint_R2",
        "joint_R3",
        "joint_R4",
        "joint_R5",
        "joint_R6",
        "joint_L1",
        "joint_L2",
        "joint_L3",
        "joint_L4",
        "joint_L5",
        "joint_L6",
    ],
)


NLEGS_IDEAL_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUsdFileCfg(
        usd_path=os.path.join(_ASSET_DIR, "mjcf/legs_narrow/legs_narrow.usd"),
    ),
    articulation_root_prim_path="/base/base",
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.62),
        joint_pos={
            ".*1": -0.1,
            ".*4": 0.2,
            ".*5": -0.1,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "ideal": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim={
                ".*1": 80.0,
                ".*2": 80.0,
                ".*3": 50.0,
                ".*4": 100.0,
                ".*5": 60.0,
                ".*6": 40.0,
            },
            velocity_limit_sim=30.0,
            stiffness={
                ".*1": 220.0,
                ".*2": 140.0,
                ".*3": 100.0,
                ".*4": 300.0,
                ".*5": 80.0,
                ".*6": 60.0,
            },
            damping={
                ".*1": 6.0,
                ".*2": 5.0,
                ".*3": 3.0,
                ".*4": 7.0,
                ".*5": 3.0,
                ".*6": 2.0,
            },
            armature=0.0,
            friction=0.0,
            dynamic_friction=0.0,
            viscous_friction=0.0,
        ),
    },
    joint_sdk_names=[
        "joint_R1",
        "joint_R2",
        "joint_R3",
        "joint_R4",
        "joint_R5",
        "joint_R6",
        "joint_L1",
        "joint_L2",
        "joint_L3",
        "joint_L4",
        "joint_L5",
        "joint_L6",
    ],
)
