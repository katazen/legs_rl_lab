import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from legs_rl_lab.actuators import DelayedDCMotorCfg


_ASSET_DIR = os.path.dirname(os.path.abspath(__file__))


@configclass
class NlegsBodyArticulationCfg(ArticulationCfg):
    joint_sdk_names: list[str] | None = None
    soft_joint_pos_limit_factor = 0.95


NLEGS_BODY_CFG = NlegsBodyArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(_ASSET_DIR, "mjcf", "nlegs_body_base2kg", "nlegs_body_base2kg.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    articulation_root_prim_path="/base/base",
    init_state=ArticulationCfg.InitialStateCfg(
        # 此站姿平地脚底贴地时 root_z≈0.449893 m；出生时留约 4 cm 净空。
        pos=(0.0, 0.0, 0.49),
        joint_pos={".*1": -0.1, ".*4": 0.4, ".*5": -0.3},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "4310": DelayedDCMotorCfg(
            joint_names_expr=[".*6"],
            effort_limit=5.8,
            effort_limit_sim=5.8,
            saturation_effort=5.8,
            velocity_limit=29.0,
            velocity_limit_sim=29.0,
            stiffness=40.0,
            damping=0.5,
            armature=0.00219,
            min_delay=1,
            max_delay=4,
        ),
        "4340": DelayedDCMotorCfg(
            joint_names_expr=[".*1", ".*2", ".*3", ".*4", ".*5"],
            effort_limit=26.0,
            saturation_effort=26.0,
            velocity_limit=6.0,
            stiffness={".*1": 200.0,
                       ".*2": 200.0,
                       ".*3": 200.0,
                       ".*4": 250.0,
                       ".*5": 200.0,},
            damping={".*1": 5.0,
                     ".*2": 5.0,
                     ".*3": 5.0,
                     ".*4": 5.0,
                     ".*5": 5.0,},
            armature=0.0509,
            # friction=0.5,
            # dynamic_friction=0.5,
            min_delay=1,
            max_delay=4,
        ),
    },
    joint_sdk_names=[
        "joint_R1", "joint_R2", "joint_R3", "joint_R4", "joint_R5", "joint_R6",
        "joint_L1", "joint_L2", "joint_L3", "joint_L4", "joint_L5", "joint_L6",
    ],
)
