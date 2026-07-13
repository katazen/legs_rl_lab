import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg,DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from legs_rl_lab.actuators import DelayedDCMotorCfg   # 自定义: 延迟 + 转矩-转速滚降(膝辨识用)

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


LEGS_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUsdFileCfg(
        usd_path=os.path.join(_ASSET_DIR, "mjcf/A1_legs_V2_mjcf/A1_legs_V2_mjcf.usd"),
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
        # ① DelayedPD 组: 不需要转矩-转速滚降的关节(实测 gain≈1 或柔性共振, DCMotor 治不了).
        #    髋roll(.*2)/髋yaw(.*3)/踝pitch(.*5)=4340; 踝roll(.*6)=4310. 每关节参数用字典.
        #    踝pitch 用 kp40/kd2(kd 压掉柔性共振过冲, sim/real 最贴); delay 全 4-6(实测传输延迟~20ms).
        "delayed_pd": DelayedPDActuatorCfg(
            joint_names_expr=[".*2", ".*3", ".*5", ".*6"],
            stiffness={".*2": 200.0, ".*3": 200.0, ".*5": 40.0, ".*6": 40.0},
            damping={".*2": 5.0, ".*3": 5.0, ".*5": 2.0, ".*6": 0.5},
            armature={".*2": 0.0509, ".*3": 0.0509, ".*5": 0.0509, ".*6": 0.00219},
            effort_limit_sim={".*2": 26.0, ".*3": 26.0, ".*5": 26.0, ".*6": 5.8},
            velocity_limit_sim=14.0,
            min_delay=4,
            max_delay=6,
        ),
        # ② DelayedDCMotor 组: 需要转矩-转速滚降(高频幅值滚降)的关节. 髋pitch(.*1) + 膝(.*4), 均 4340.
        #    velocity_limit 用【模型量字典】(非 _sim): 髋pitch 2.2; 膝左右不对称 L4=2.3/R4=3.0. sat/effort=26.
        "delayed_dcmotor": DelayedDCMotorCfg(
            joint_names_expr=[".*1", ".*4"],
            effort_limit=26.0,
            saturation_effort=26.0,
            velocity_limit={".*1": 2.2, "joint_L4": 2.3, "joint_R4": 3.0},
            stiffness={".*1": 200.0, ".*4": 250.0},
            damping={".*1": 5.0, ".*4": 5.0},
            armature=0.0509,
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
