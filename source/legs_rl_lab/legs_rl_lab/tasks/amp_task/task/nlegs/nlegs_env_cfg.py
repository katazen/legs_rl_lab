"""nlegs + AMP：自包含的窄本体走路任务配置。

设计（本文件不再依赖已删除的 legs 目录）：
  - scene / observations / actions / commands / gait / terminations / curriculum：沿用本项目自己的配置。
  - rewards：整体移植 TienKung-Lab walk_cfg.LiteRewardCfg（见 AmpRewardsCfg，去上肢项、适配 nlegs）。
  - events：参考 TienKung walk_cfg 的 DomainRandCfg，但保留两个 nlegs 特有项——
        ① reset_robot_joints 用 reset_joints_by_offset（nlegs 有零默认角关节，by_scale 随机不到它们）；
        ② joint_zero_bias（支撑 joint_pos_rel_biased 观测，消实机零速漂移）。
  - AMP：追加名为 "amp" 的 30 维观测组，供判别器经 obs["amp"] 取用（不改 train.py）。
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from legs_rl_lab.assets.legs_narrow.nlegs import NLEGS_FIX_CFG
from legs_rl_lab.tasks.amp_task import mdp


@configclass
class GaitCfg:
    """步态时钟参数（会被 train.py 的 dump_yaml 写进 env.yaml，训练/部署可复现）。"""

    period: float = 0.85          # 步态周期 (s)
    stance_ratio: float = 0.62    # 支撑相占比
    feet_offset: list = [0.0, 0.5]  # 左右腿相位偏移


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """带足式机器人的地形场景配置。"""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # 窄本体机器人资产（脚间距 0.2）
    robot: ArticulationCfg = NLEGS_FIX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # history_length=4=decimation: net_forces_w_history 正好覆盖一个控制步的 4 个物理子步,
    # 供 gait 力奖励对子步取均值(对齐 TienKung avg_feet_force_per_step)。
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/base/.*", history_length=4, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """域随机化配置（参考 TienKung walk_cfg 的 DomainRandCfg；两处 nlegs 特有项见文件头说明）。"""

    # startup: 地面/材质摩擦随机（TienKung 范围）
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.1, 1.3),
            "dynamic_friction_range": (0.1, 1.3),
            "restitution_range": (0.0, 0.005),
            "num_buckets": 64,
        },
    )

    # startup: base 质量加减（TienKung 在 pelvis，这里映射到 base）
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset: 根位姿/速度随机
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    # reset: 关节初值随机（nlegs 特有：用 by_offset，绝对偏移，能覆盖默认=0 的髋roll/yaw、踝roll）
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # reset: 标0偏置随机化（nlegs 特有：每环境每关节恒定偏置，模拟实机编码器零位标定误差，
    # 仅作用于策略观测 joint_pos_rel_biased -> 消除实机零速漂移）
    joint_zero_bias = EventTerm(
        func=mdp.randomize_joint_zero_bias,
        mode="reset",
        params={"bias_range": (-0.05, 0.05)},
    )

    # interval: 随机推力（TienKung 范围）
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class CommandsCfg:
    """MDP 命令规格。"""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.5), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.5, 0.5)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.5), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.5, 0.5)
        ),
    )


@configclass
class ActionsCfg:
    """MDP 动作规格。"""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """MDP 观测规格（策略 / 评论 / AMP 判别器三组）。"""

    @configclass
    class PolicyCfg(ObsGroup):
        """策略观测组。"""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel_biased, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)
        gait_phase = ObsTerm(func=mdp.gait_phase_obs)

        def __post_init__(self):
            self.history_length = 10
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """评论（特权）观测组。"""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01)
        last_action = ObsTerm(func=mdp.last_action)
        gait_phase = ObsTerm(func=mdp.gait_phase_obs)

        def __post_init__(self):
            self.history_length = 10

    critic: CriticCfg = CriticCfg()

    @configclass
    class AmpCfg(ObsGroup):
        """AMP 判别器观测组: 30 维 = 关节位置(12) + 关节速度(12) + 左右脚 base 系位置(6)。"""

        amp = ObsTerm(func=mdp.amp_obs)

        def __post_init__(self):
            self.history_length = 1
            self.enable_corruption = False
            self.concatenate_terms = True

    amp: AmpCfg = AmpCfg()


@configclass
class AmpRewardsCfg:
    """移植自 TienKung-Lab walk_cfg.LiteRewardCfg 的奖励, 适配 nlegs(仅下肢 12 DoF)。

    对齐关系:
      - 去掉 TienKung 的上肢项(shoulder / elbow / arm deviation), nlegs 无手臂。
      - body / joint 名映射到 nlegs: base 躯干; 脚=".*6"(ankle roll link); 髋roll=".*2", 髋yaw=".*3",
        髋pitch=".*1", 膝=".*4", 踝pitch=".*5", 踝roll=".*6"。
      - 步态周期奖励(frc/spd/support)配合 nlegs 的 stance-first 相位时钟(env.cfg.gait)。
      - AMP 的风格奖励由判别器另行提供, 这里只保留任务/正则项。
    """

    # -- 速度跟踪(任务主目标)
    track_lin_vel_xy = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=1.0)
    track_ang_vel_z = RewTerm(func=mdp.track_ang_vel_z_exp, weight=1.0)

    # -- base 正则
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    action_acc = RewTerm(func=mdp.action_acc_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*6).*"])},
    )
    # base 是 articulation root, body_orientation_l2(base) 与 flat_orientation_l2 逐位相同
    # (都取 root 系单位重力的 xy 平方和), 故合并为一项, 权重取二者之和(-1.0 + -2.0 = -3.0)。
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # -- 脚
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*6"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*6"),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*6"), "threshold": 500, "max_reward": 400},
    )
    # feet_too_near = RewTerm(
    #     func=mdp.feet_too_near_humanoid,
    #     weight=-2.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=".*6"), "threshold": 0.2},
    # )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*6")},
    )
    feet_y_distance = RewTerm(
        func=mdp.feet_y_distance,
        weight=-2.0,
        params={"threshold": 0.222, "asset_cfg": SceneEntityCfg("robot", body_names=".*6")},
    )
    # 前向零速命令时惩罚前后脚错位(原地踏步保持双脚同一 x 线); 有前进命令时自动关闭
    feet_x_distance = RewTerm(
        func=mdp.feet_x_distance,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*6")},
    )

    # -- 关节回默认位(仅站立时约束, 行走放开)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*2", ".*3"]), "standing_only": True},
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*1", ".*4", ".*5", ".*6"]), "standing_only": True},
    )

    # -- 步态周期奖励(与相位时钟耦合)
    gait_feet_frc_perio = RewTerm(
        func=mdp.gait_feet_frc_perio,
        weight=1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*6"), "delta_t": 0.02},
    )
    gait_feet_spd_perio = RewTerm(
        func=mdp.gait_feet_spd_perio,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*6"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*6"),
            "delta_t": 0.02,
        },
    )
    gait_feet_frc_support_perio = RewTerm(
        func=mdp.gait_feet_frc_support_perio,
        weight=0.6,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*6"), "delta_t": 0.02},
    )

    # -- 踝柔顺 + 髋侧摆/内扣抑制
    ankle_torque = RewTerm(
        func=mdp.ankle_torque,
        weight=-0.0005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*5", ".*6"])},
    )
    ankle_action = RewTerm(
        func=mdp.ankle_action,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*5", ".*6"])},
    )
    hip_roll_action = RewTerm(
        func=mdp.joint_action_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*2"])},
    )
    hip_yaw_action = RewTerm(
        func=mdp.joint_action_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*3"])},
    )


@configclass
class TerminationsCfg:
    """MDP 终止条件。"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})


@configclass
class CurriculumCfg:
    """MDP 课程项。"""

    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)
    ang_vel_cmd_levels = CurrTerm(mdp.ang_vel_cmd_levels)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """nlegs + AMP 走路环境（自包含）。"""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: AmpRewardsCfg = AmpRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    # 步态时钟参数（会被 dump 到 env.yaml）
    gait: GaitCfg = GaitCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods（按最小更新周期 = 物理步长）
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # 地形课程开关
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
