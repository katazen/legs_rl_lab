"""窄本体变体 (nlegs)：完全复用 legs 的 env 配置，仅两处不同——
机器人资产 (A1_legs_V2_narrow, 脚间距 0.2) 和 feet_y_distance 的目标间距。
其余 scene / events / rewards / observations / commands / 步态参数全部继承 legs。
"""

from isaaclab.utils import configclass

from legs_rl_lab.assets.legs_narrow.nlegs import NLEGS_CFG, NLEGS_FIX_CFG
from legs_rl_lab.tasks.legs_task import mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from legs_rl_lab.tasks.legs_task.task.legs.legs_dr_env_cfg import (
    RobotEnvCfg as LegsEnvCfg,
    RobotPlayEnvCfg as LegsPlayEnvCfg,
)


def _apply_ankle_dr(cfg) -> None:
    """踝域随机化(仅 nlegs, 不动共享的 nlegs.py / legs_dr)。

    24V 时代结论是"踝 pitch 比 sim 多滞后 ~60ms + 幅值腰斩", 据此给了 20~80ms 延迟
    + kp 0.6~1.0 的悲观 DR。Stage4C(48V) 证明那个迟钝大半是 24V 电压瓶颈不是执行器
    本性(2Hz 增益 0.161->0.658, 相位 134°->95°; 起动延迟 ~14-18ms 不随电压变),
    故 48V 下 DR 收窄到温和偏保守:
      1) 延迟: min 4, max 16->8 (20~40ms), 覆盖实测 14-18ms + 余量;
      2) 带宽: kp (0.6,1.0)->(0.7,1.1) 仍略偏下(残余摩擦削幅), kd (1.0,1.8)->(0.9,1.5)。
    摩擦 0.5->0.55: 3C(24V)与4C(48V)两轮回线复现 L0.45-0.50/R0.60-0.70。
    明确不碰 armature —— 抬它会欠阻尼过冲、放大幅值, 与实机磨圆衰减反向。
    """
    # ---- 1. 拆踝执行器组, 延迟适度拉宽 ----
    acts = cfg.scene.robot.actuators
    dc = acts["4340"]
    dc.joint_names_expr = [".*1", ".*2", ".*3", ".*4"]
    if isinstance(dc.stiffness, dict):
        dc.stiffness = {k: v for k, v in dc.stiffness.items() if k != ".*5"}
    if isinstance(dc.damping, dict):
        dc.damping = {k: v for k, v in dc.damping.items() if k != ".*5"}
    acts["ankle"] = type(dc)(
        joint_names_expr=[".*5"],
        effort_limit=dc.effort_limit, saturation_effort=dc.saturation_effort,
        velocity_limit=dc.velocity_limit,
        stiffness=40.0, damping=2, armature=0.0509,
        friction=0.55, dynamic_friction=0.55,
        min_delay=4, max_delay=8,
    )

    # ---- 2. DR 事件: 踝从全局 gains/armature 里排除, 单独随机 ----
    # 全局 gains(.*  0.8~1.2)与 armature 随机都排除 .*5, 避免两条 reset 事件互相覆盖。
    for term_name in ("randomize_actuator_gains", "randomize_joint_params"):
        term = getattr(cfg.events, term_name, None)
        if term is not None:
            term.params["asset_cfg"] = SceneEntityCfg(
                "robot", joint_names=[".*1", ".*2", ".*3", ".*4", ".*6"])
    # 踝专属带宽随机化(48V 后收窄: kp 略偏下 + kd 略偏高)。下限 kp scale 0.7 远在发散区(kp≈10-15)之上。
    cfg.events.randomize_ankle_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*5"]),
            "stiffness_distribution_params": (0.7, 1.1),
            "damping_distribution_params": (0.9, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


def _apply_nlegs(cfg) -> None:
    """把 legs 配置改成窄本体：换机器人资产 + 改 feet_y_distance 目标间距。"""
    cfg.scene.robot = NLEGS_FIX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.events.add_base_mass.params["mass_distribution_params"] = (1.0, 4.0)
    cfg.events.reset_robot_joints.params["position_range"] = (-0.2, 0.2)
    cfg.gait.period = 0.6
    cfg.gait.stance_ratio = 0.55
    cfg.rewards.flat_orientation.weight = -4.0
    cfg.rewards.base_height.params["target_height"] = 0.58
    cfg.rewards.feet_y_distance.params["threshold"] = 0.222
    cfg.rewards.joint_deviation_legs.weight = -0.5
    # cfg.rewards.joint_deviation_hip = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.05,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*1", ".*4"])},
    # )
    # 无 y 速度命令时抑制身体左右平移(体系 y 线速度罚), 有横移命令时自动关闭
    cfg.rewards.lateral_move = RewTerm(
        func=mdp.base_lateral_move_l2,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # cfg.commands.base_velocity.ranges.lin_vel_x = (-0.3, 1.0)
    # cfg.commands.base_velocity.limit_ranges.lin_vel_x = (-0.3, 1.0)
    cfg.rewards.undesired_contacts = None

@configclass
class RobotEnvCfg(LegsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_nlegs(self)
        _apply_ankle_dr(self)  # 迟钝踝域随机化(仅训练)


@configclass
class RobotPlayEnvCfg(LegsPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = NLEGS_FIX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
