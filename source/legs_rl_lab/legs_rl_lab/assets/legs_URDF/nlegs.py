import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from legs_rl_lab.assets.legs_URDF.legs import UnitreeArticulationCfg, UnitreeUsdFileCfg

# 与 legs.py 完全相同的机器人配置，唯一区别是指向 A1_legs_V2_narrow_mjcf.xml 转出的 USD。
# TODO: USD 还没转好，转好后把下面的 usd_path 填上（narrow 版 USD 的绝对路径）。

NLEGS_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUsdFileCfg(
        usd_path="/home/woan/workspace/legs_rl_lab/source/legs_rl_lab/legs_rl_lab/assets/legs_URDF/mjcf/A1_legs_V2_narrow_mjcf/A1_legs_V2_narrow_mjcf.usd",
    ),
    # narrow USD 与原 USD 结构一致：articulation root 在 base 上，相对 spawn 的 Robot prim 为 /base/base。
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
            min_delay=11,
            max_delay=15,
        ),
        "4310": DelayedPDActuatorCfg(
            joint_names_expr=[".*5", ".*6"],
            effort_limit_sim=7.0,
            velocity_limit_sim=14.0,
            stiffness=40.0,
            damping=0.5,
            armature=0.0018,
            min_delay=11,
            max_delay=15,
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
