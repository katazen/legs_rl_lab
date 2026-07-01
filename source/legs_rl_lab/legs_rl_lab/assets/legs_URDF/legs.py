import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg,DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass


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


LEGS_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUsdFileCfg(
        usd_path="/home/woan/workspace/legs_rl_lab/source/legs_rl_lab/legs_rl_lab/assets/legs_URDF/mjcf/A1_legs_V2_mjcf/A1_legs_V2_mjcf.usd",
    ),
    # In A1_legs_V2_mjcf.usd the articulation root (PhysicsArticulationRootAPI) is on the
    # `base` body at /<defaultPrim>/base/base, so relative to the spawned Robot prim it is /base/base.
    articulation_root_prim_path='/base/base',
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            ".*1": -0.1,
            ".*4": 0.2,
            ".*5": -0.1,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "4340": DelayedPDActuatorCfg(
            joint_names_expr=[".*1", ".*2", ".*3", ".*4"],
            effort_limit_sim=27.0,
            velocity_limit_sim=14.0,
            stiffness=200.0,
            damping=5.0,
            armature=0.032,
            min_delay=9,
            max_delay=12,
        ),
        "4310": DelayedPDActuatorCfg(
            joint_names_expr=[".*5", ".*6"],
            effort_limit_sim=7.0,
            velocity_limit_sim=14.0,
            stiffness=40.0,
            damping=0.5,
            armature=0.0018,
            min_delay=9,
            max_delay=12,
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
