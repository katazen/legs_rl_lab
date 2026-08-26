"""nlegs + EngineAI 风格 AMP: 复用 nlegs 走路任务, 把 AMP 观测组 + 奖励整套换成 EngineAI 范式。

与 amp_task/task/nlegs 的区别:
  - AMP 观测组 = EngineAI 风格单帧 15 维 [joint_pos*9(12), base_lin_vel_b*7(3)],
    由 history_length=5 堆成 75 维 obs["amp"] (history-window 范式)。
  - 奖励整套换成 EngineAI 的 feet_air_time 驱动奖励(EngineaiRewardsCfg), 去掉 PM01 的腰/臂项,
    常数按 nlegs 尺寸(站高~0.58, 脚横距~0.20)重调。风格奖励仍由 AMP 判别器加性提供。
  - 终止追加 illegal_contact(base+膝), 对齐 EngineAI。
  - 策略/评论观测、动作缩放(0.25)、控制频率(50Hz)、平地、命令+课程、事件等保持本仓原样
    (不改部署代码)。

判别器/专家数据/加性 style reward 由 agents 里的 NlegsAmpEngineaiPPORunnerCfg +
legs_rl_lab.amp_engineai.amp_ppo.AMPPPO 负责。
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from legs_rl_lab.tasks.amp_task import mdp
from legs_rl_lab.tasks.amp_task.task.nlegs.nlegs_env_cfg import EventCfg, RobotEnvCfg, RobotPlayEnvCfg

# nlegs 尺寸(用于重调 EngineAI 里 PM01 专属常数)
_NLEGS_BASE_HEIGHT = 0.58     # 站姿本体高(资产 init z=0.62, 注释"约0.58")
_NLEGS_ANKLE_DIST = 0.222      # 脚横向间距(MuJoCo FK 实测 |L6.y-R6.y|=0.2218)
_NLEGS_FOOT_OFFSET = 0.012     # 站姿时脚连杆原点距地高度(FK: base=0.58 时脚原点 z≈0.0117; PM01 原值 0.045)
_CURR = {"start_scale": 0.1, "power": 0.8, "interval_epochs": 200 * 24}  # 能耗/动作课程


@configclass
class AmpEngineaiObsCfg(ObsGroup):
    """EngineAI 风格 AMP 观测组: 单帧 15 维, 5 帧堆叠 -> 判别器输入 75 维。"""

    amp = ObsTerm(func=mdp.amp_obs_engineai)

    def __post_init__(self):
        self.history_length = 5          # history-window: 与 data_loader/判别器 frame_length 一致
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class EngineaiRewardsCfg:
    """整套照搬 EngineAI(PM01Rewards) 的奖励, 适配 nlegs(仅下肢 12 DoF)。

    对齐关系(见 nlegs 关节/连杆命名: 髋pitch=.*1, 髋roll=.*2, 髋yaw=.*3, 膝=.*4,
    踝pitch=.*5, 踝roll=.*6; 脚=Link_[LR]6=.*6; 膝连杆=.*4; base=base):
      - 去掉 PM01 的 waist_pos / arm_* 项(nlegs 无腰无臂)。
      - leg_joint_position 作用于髋roll/髋yaw/踝roll = [.*2, .*3, .*6]。
      - base_height / feet_position 的高度/横距常数按 nlegs 重调。
    """

    # -- 速度跟踪(任务主目标; 名称须与 lin/ang_vel_cmd_levels 课程默认项名一致)
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "sigma": 5},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=2.5,
        params={"command_name": "base_velocity", "sigma": 5},
    )

    # -- base 姿态/高度
    base_orientation = RewTerm(func=mdp.base_orientation, weight=1.0)
    base_height = RewTerm(
        func=mdp.base_height_tracking,
        weight=0.4,
        params={"target_height": _NLEGS_BASE_HEIGHT},
    )

    # -- 脚位姿/接触(显式左右脚: +y=左脚 Link_L6, -y=右脚 Link_R6)
    #    preserve_order=True 保证 body_ids 顺序=[L6, R6], 否则默认按 body index 升序解析成
    #    [R6, L6](R6 在 MJCF 树中排前), 会把 desired_y 的 +y 错配给右脚 -> 奖励交叉腿。
    foot_position = RewTerm(
        func=mdp.feet_position,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["Link_L6", "Link_R6"], preserve_order=True),
            "command_name": "base_velocity",
            "stand_threshold": 0.1,
            "ankle_distance": _NLEGS_ANKLE_DIST,
            "base_height_target": _NLEGS_BASE_HEIGHT,
            "foot_origin_offset": _NLEGS_FOOT_OFFSET,
        },
    )
    feet_orientation = RewTerm(
        func=mdp.feet_orientation,
        weight=0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*6"),
            "command_name": "base_velocity",
            "stand_threshold": 0.1,
        },
    )
    feet_contact = RewTerm(
        func=mdp.feet_contact_fixed,
        weight=0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*6"),
            "command_name": "base_velocity",
            "stand_threshold": 0.1,
            "force_threshold": 5.0,
        },
    )

    # -- 关节回默认位(髋roll/髋yaw/踝roll)
    leg_joint_position = RewTerm(
        func=mdp.joint_deviation_exp,
        weight=0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*2", ".*3", ".*6"]), "scale": 3.0},
    )

    # -- air time(feet_air_time 驱动步态; 含 yaw 门控版, 纯旋转命令也发抬脚奖励 -> 旋转靠踏步而非抖动)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_engineai,
        weight=10.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*6"),
            "threshold": 0.5,
        },
    )
    feet_air_time_dense = RewTerm(
        func=mdp.feet_air_time_positive_biped_engineai,
        weight=1.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*6"),
            "threshold": 0.5,
        },
    )
    # -- 抬脚高度: 落地时结算本段摆动峰值高度对 0.10m 的偏差(惩罚), 直接锚定抬脚高度;
    #    只在落地事件结算 -> 不会被抖脚 farming, 权重不敏感; 含 yaw 门控 -> 直走+旋转都抬脚。
    # feet_swing_height = RewTerm(
    #     func=mdp.feet_swing_height,
    #     weight=-50.0,
    #     params={
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*6"),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*6"),
    #         "target_height": 0.10,
    #     },
    # )

    # -- 脚绊/滑
    foot_stumble = RewTerm(
        func=mdp.feet_stumble_engineai,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*6"),
            "tangential_threshold": 2.0,
            "normal_threshold": 1.0,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*6"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*6"),
        },
    )
    # 横向脚距: 无横向速度命令(|cmd_y|<0.1, 含前后走)时惩罚偏离 0.222, 防止走路收脚。
    # feet_position 只在完全站立时约束脚距, 走路时失效 -> 这里补上。取 |L.y-R.y| 绝对差, 与左右顺序无关。
    feet_y_distance = RewTerm(
        func=mdp.feet_y_distance,
        weight=-4.0,
        params={"threshold": 0.222, "asset_cfg": SceneEntityCfg("robot", body_names=".*6")},
    )

    # -- 正则(关节限位/速度/加速度/力矩)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-10.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    dof_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-5, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")})
    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-8, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")})
    dof_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-6, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")})

    # -- 能耗/动作平滑(带 epoch 课程)
    energy_cost = RewTerm(
        func=mdp.energy_cost_with_curriculum,
        weight=-0.004,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), **_CURR},
    )
    action_rate = RewTerm(func=mdp.action_rate_with_curriculum, weight=-0.06, params=dict(_CURR))
    action_smoothness = RewTerm(func=mdp.action_smoothness_with_curriculum, weight=-0.04, params=dict(_CURR))

    # -- 终止惩罚
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class EngineaiTerminationsCfg:
    """对齐 EngineAI: 追加 illegal_contact(base+膝), 保留 root_height 兜底。"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*4"]), "threshold": 1.0},
    )
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})


@configclass
class EngineaiEventCfg(EventCfg):
    """在 nlegs 基础 EventCfg 上追加域随机化(对齐 legs_dr / EngineAI 的 DR 项)。

    基础版已有(继承, 不重复): 地面/材质摩擦、base 质量加减、根位姿/速度随机、
    关节初值随机(by_offset)、电机零位偏置(joint_zero_bias)、随机推力。
    这里新增 legs_dr 的 4 项: 连杆质量整体缩放、base 质心偏移、PD 增益缩放、关节摩擦/armature 随机。
    """

    # 连杆质量整体缩放(所有 body) [0.9, 1.1], 模拟建模/装配质量误差
    randomize_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # base 质心偏移 [-0.05, 0.05] m, 模拟负载/装配质心不准
    randomize_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # PD 增益(刚度/阻尼)相对默认值缩放 [0.8, 1.2], 模拟电机增益不确定
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # 关节摩擦 / armature 随机(相对缩放, 对齐 legs_dr)
    randomize_joint_params = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "armature_distribution_params": (0.8, 1.2),
            "friction_distribution_params": (0.6, 1.4),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class RobotEngineaiEnvCfg(RobotEnvCfg):
    """训练配置: 换 EngineAI 风格 AMP 观测组 + EngineAI 奖励 + 终止 + 域随机化。"""

    def __post_init__(self):
        super().__post_init__()
        self.observations.amp = AmpEngineaiObsCfg()
        self.rewards = EngineaiRewardsCfg()
        self.terminations = EngineaiTerminationsCfg()
        self.events = EngineaiEventCfg()


@configclass
class RobotEngineaiPlayEnvCfg(RobotPlayEnvCfg):
    """Play/评估配置: 同样换 EngineAI 风格观测组 + 奖励 + 终止。"""

    def __post_init__(self):
        super().__post_init__()
        self.observations.amp = AmpEngineaiObsCfg()
        self.rewards = EngineaiRewardsCfg()
        self.terminations = EngineaiTerminationsCfg()
