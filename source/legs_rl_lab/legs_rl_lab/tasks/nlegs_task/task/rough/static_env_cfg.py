"""rough_static：零速站稳，行走时保留 rough 地形；运行时应用实测单侧限位。"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from legs_rl_lab.tasks.nlegs_task import mdp

from legs_rl_lab.tasks.nlegs_task.agents.rsl_rl_ppo_cfg import NlegsRoughPPORunnerCfg
from legs_rl_lab.tasks.nlegs_task.task.flat.static_env_cfg import FlatStaticEnvCfg
from .rough_env_cfg import NLEGS_ROUGH_TERRAINS_CFG


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


def _apply_rough_static(cfg) -> None:
    """保留 rough_static 原有参数和膝限位，不跟随 rough 的悬空辨识候选。"""
    limits = CROUCH_LIMITS
    cfg.actions.JointPositionAction.clip = {
        name: (lo if lo is not None else -float("inf"), hi if hi is not None else float("inf"))
        for name, (lo, hi) in limits.items()
    }
    cfg.events.crouch_joint_limits = EventTerm(
        func=mdp.set_joint_position_limits, mode="startup", params={"joint_limits": limits},
    )
    # 09-16/17 小幅辨识只支持扩展等效延迟与增益覆盖，保留名义 PD、惯量和摩擦。
    actuators = cfg.scene.robot.actuators
    actuators["knees"] = actuators["legs"].replace(
        joint_names_expr=[".*4"], stiffness=250.0, damping=5.0, min_delay=1, max_delay=6,
    )
    actuators["legs"].joint_names_expr = [".*1", ".*2", ".*3"]
    actuators["legs"].stiffness.pop(".*4")
    actuators["legs"].damping.pop(".*4")
    actuators["ankle_pitch"].min_delay = 1
    actuators["ankle_roll"].min_delay = 2
    cfg.events.randomize_ankle_gains.params["damping_distribution_params"] = (0.7, 1.2)
    cfg.events.randomize_ankle_roll_gains = EventTerm(
        func=mdp.randomize_actuator_gains, mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*6"]),
            "stiffness_distribution_params": (0.85, 1.1),
            "damping_distribution_params": (0.5, 1.0),
            "operation": "scale", "distribution": "uniform",
        },
    )
    # --- 地形: plane -> generator(rough) ---
    # 用 .replace() 拿独立副本，避免 play/train 两个 env cfg 共享并互相改写同一个地形对象
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = NLEGS_ROUGH_TERRAINS_CFG.replace()
    cfg.scene.terrain.terrain_generator.curriculum = True
    cfg.scene.terrain.max_init_terrain_level = 5  # 初始铺在 0~5 级，course 再上下调

    # --- 地形课程: 走得远升难度、走不动降难度 ---
    cfg.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

    # --- critic 特权观测: 187 点局部高度图(非对称 actor-critic, actor 仍盲走) ---
    # offset=0.58(名义 base 高度)使数值围绕 0; clip 兜住未命中射线的 inf
    cfg.observations.critic.height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.58},
        clip=(-1.0, 1.0),
    )

    # --- 因地形调整的奖励 ---
    # base_height 用高度扫描做地形相对(否则起伏地形上 base 世界系绝对高度恒被罚)
    cfg.rewards.base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")
    cfg.rewards.base_height.weight = -2.0
    cfg.rewards.flat_orientation.weight = -1.0
    cfg.rewards.base_linear_velocity.weight = -0.5
    cfg.rewards.feet_flat.weight = -0.3
    # feet_clearance 同样地形相对化(每只脚取扫描点中水平最近命中点作脚下地面高度)
    cfg.rewards.feet_clearance.params["sensor_cfg"] = SceneEntityCfg("height_scanner")
    cfg.rewards.feet_clearance.params["target_height"] = 0.12
    cfg.rewards.lateral_move.weight = -2.0

    # --- 终止: 摔倒(躯干严重倾斜)即终止 ---
    cfg.terminations.bad_orientation = DoneTerm(
        func=mdp.bad_orientation, params={"limit_angle": 1.0}
    )


@configclass
class RoughStaticEnvCfg(FlatStaticEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_rough_static(self)


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
