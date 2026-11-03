"""W1196 measured-mass asset with the current body-rough actuator baseline."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from legs_rl_lab.actuators import DelayedDCMotorCfg


_ASSET_DIR = Path(__file__).resolve().parent
# CAD geometry at J1/J4/J5 = -0.1/0.2/-0.1 rad, with capsule soles on a plane.
W1196_NOMINAL_BASE_HEIGHT = 0.51722
W1196_FEET_WIDTH = 0.251803
W1196_FOOT_HEIGHT_OFFSET = 0.015


@configclass
class W1196ArticulationCfg(ArticulationCfg):
    joint_sdk_names: list[str] | None = None
    soft_joint_pos_limit_factor = 0.95


W1196_CFG = W1196ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(_ASSET_DIR / "w1196" / "w1196.usd"),
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
        # Nominal sole clearance is about 3.8 cm; all values are metres/radians.
        pos=(0.0, 0.0, 0.555),
        joint_pos={".*1": -0.1, ".*4": 0.2, ".*5": -0.1},
        joint_vel={".*": 0.0},
    ),
    # shortcut: retain prior motor identification until W1196 drives are measured.
    actuators={
        "legs": DelayedDCMotorCfg(
            joint_names_expr=[".*1", ".*2", ".*3", ".*4", ".*5"],
            effort_limit=26.0,
            effort_limit_sim=26.0,
            saturation_effort=26.0,
            velocity_limit=6.0,
            velocity_limit_sim=6.0,
            stiffness={".*1": 300.0, ".*2": 300.0, ".*3": 300.0, ".*4": 300.0, ".*5": 300.0},
            damping={".*1": 5.0, ".*2": 5.0, ".*3": 5.0, ".*4": 5.0, ".*5": 5.0},
            armature=0.0509,
            friction=0.5,
            dynamic_friction=0.5,
            min_delay=1,
            max_delay=5,
        ),
        "ankle_roll": DelayedPDActuatorCfg(
            joint_names_expr=[".*6"],
            stiffness=50.0,
            damping=1.0,
            armature=0.00219,
            effort_limit=5.8,
            effort_limit_sim=5.8,
            velocity_limit=29.0,
            velocity_limit_sim=29.0,
            viscous_friction=0.54,
            min_delay=1,
            max_delay=5,
        ),
    },
    joint_sdk_names=[
        "joint_R1", "joint_R2", "joint_R3", "joint_R4", "joint_R5", "joint_R6",
        "joint_L1", "joint_L2", "joint_L3", "joint_L4", "joint_L5", "joint_L6",
    ],
)
