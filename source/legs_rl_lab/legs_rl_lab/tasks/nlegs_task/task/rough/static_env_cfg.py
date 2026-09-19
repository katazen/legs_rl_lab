"""rough_static：零速站稳，行走时保留 rough 地形；运行时应用实测单侧限位。"""

from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.utils import configclass

from legs_rl_lab.tasks.nlegs_task import mdp
from legs_rl_lab.tasks.nlegs_task.agents.rsl_rl_ppo_cfg import NlegsRoughPPORunnerCfg
from legs_rl_lab.tasks.nlegs_task.task.flat.static_env_cfg import FlatStaticEnvCfg
from .rough_env_cfg import _apply_rough


# 2026-09-17 实测下蹲端点，与部署 crouch_calibration 相同；另一端及踝 roll 不新增约束。
CROUCH_LIMITS = {
    "joint_L1": (-0.9752422370, None),
    "joint_L2": (None, 0.4964904250),
    "joint_L3": (None, 0.4983978027),
    "joint_L4": (None, 1.0561150530),
    "joint_L5": (-0.3763256275, None),
    "joint_R1": (-0.9676127260, None),
    "joint_R2": (-0.5270084688, None),
    "joint_R3": (-0.6185626001, None),
    "joint_R4": (None, 1.0912108034),
    "joint_R5": (-0.4224841688, None),
}


@configclass
class RoughStaticEnvCfg(FlatStaticEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_rough(self)
        self.actions.JointPositionAction.clip = {
            name: (lo if lo is not None else -float("inf"), hi if hi is not None else float("inf"))
            for name, (lo, hi) in CROUCH_LIMITS.items()
        }
        self.events.crouch_joint_limits = EventTerm(
            func=mdp.set_joint_position_limits, mode="startup", params={"joint_limits": CROUCH_LIMITS},
        )
        # 小幅辨识支持更快的等效响应；只扩膝的延迟覆盖，不改髋或名义 PD/惯量。
        actuators = self.scene.robot.actuators
        actuators["knees"] = actuators["legs"].replace(
            joint_names_expr=[".*4"], stiffness=250.0, damping=5.0, min_delay=1, max_delay=6,
        )
        actuators["legs"].joint_names_expr = [".*1", ".*2", ".*3"]
        actuators["legs"].stiffness.pop(".*4")
        actuators["legs"].damping.pop(".*4")
        actuators["ankle_pitch"].min_delay = 1
        actuators["ankle_roll"].min_delay = 2
        # 上报力矩的留出 PD 回归支持较低 Kd；不把它等同于独立力矩标定。
        self.events.randomize_ankle_gains.params["damping_distribution_params"] = (0.7, 1.2)
        self.events.randomize_ankle_roll_gains = EventTerm(
            func=mdp.randomize_actuator_gains, mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*6"]),
                "stiffness_distribution_params": (0.85, 1.1),
                "damping_distribution_params": (0.5, 1.0),
                "operation": "scale", "distribution": "uniform",
            },
        )


@configclass
class RoughStaticPlayEnvCfg(RoughStaticEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.max_init_terrain_level = 4


@configclass
class NlegsRoughStaticPPORunnerCfg(NlegsRoughPPORunnerCfg):
    experiment_name = "nlegs_rough_static"

    def __post_init__(self):
        super().__post_init__()
        # 实测左右限位不同，不强制学习严格镜像动作。
        self.algorithm.symmetry_cfg = None
