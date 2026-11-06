"""nlegs_body_flat：W1196 平地行走/零速踏步，独立 MDP/PPO 配置。

共同奖励权重取自 2026-08-26_13-57-34 flat；其余使用当前 rough，零速继续踏步。
共享 W1196_CFG 及基础 MDP 函数，不继承旧 flat 或 rough 的环境/算法配置。
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
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

from legs_rl_lab.assets.nlegs_v2.nlegs_v2 import (
    W1196_CFG,
    W1196_FEET_WIDTH,
    W1196_FOOT_HEIGHT_OFFSET,
    W1196_NOMINAL_BASE_HEIGHT,
)
from legs_rl_lab.tasks.nlegs_task import mdp
from legs_rl_lab.tasks.nlegs_task.mdp.symmetry import compute_symmetric_states

from ..rough import body_rewards


@configclass
class FlatBodyGaitCfg:
    """步态时钟参数, 会被 dump 到 env.yaml; reward / observation 通过 env.cfg.gait.* 读取。"""

    period: float = 0.6             # 步态周期 (s)
    stance_ratio: float = 0.55      # 支撑相占比
    feet_offset: list = [0.0, 0.5]  # 右、左足顺序，与奖励实体选择一致


@configclass
class FlatBodySceneCfg(InteractiveSceneCfg):
    """W1196 机器人与无限平面；不需要地形高度扫描。"""

    # ground terrain
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
    # 与当前 nlegs_body_rough 共用同一资产，执行器参数不另写一套。
    robot: ArticulationCfg = W1196_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/base/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class FlatBodyEventCfg:
    """当前 W1196 随机化基准；不额外改变执行器 PD 增益。"""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # reset
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            # 约为名义 base 质量的 ±20%；保留 USD 实测质量，不把未知差额固定加到机身。
            "mass_distribution_params": (-0.9, 0.9),
            "operation": "add",
        },
    )

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

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            # 绝对偏移(rad); 换成 offset 后默认=0 的关节(髋roll/yaw、踝roll)也会被随机化
            "position_range": (-0.2, 0.2),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # 仅左右踝 pitch：观测加零偏 b，物理目标减同一个 b。
    joint_zero_bias = EventTerm(
        func=mdp.randomize_joint_zero_bias,
        mode="reset",
        params={
            "bias_range": (-0.05, 0.05),
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*5"]),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class FlatBodyCommandsCfg:
    """Command specifications for the MDP."""

    # 普通速度命令；零速样本仍学习踏步，不切换站立模式。
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.5), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-1.0, 1.0)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.5), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-1.0, 1.0)
        ),
    )


@configclass
class FlatBodyActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        class_type=mdp.ZeroBiasJointPositionAction,
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class FlatBodyObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel_biased, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)
        gait_phase = ObsTerm(func=mdp.gait_phase_obs, params={"gate_by_cmd": False})

        def __post_init__(self):
            self.history_length = 15
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01)
        last_action = ObsTerm(func=mdp.last_action)
        gait_phase = ObsTerm(func=mdp.gait_phase_obs, params={"gate_by_cmd": False})

        def __post_init__(self):
            self.history_length = 15

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class FlatBodyRewardsCfg:
    """旧 flat 共同权重；保留 rough 防偏航项，低前速双支撑时约束前后错脚。"""

    # -- task
    track_lin_vel_xy = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=1.0)
    track_ang_vel_z = RewTerm(func=mdp.track_ang_vel_z_exp, weight=1.0)
    alive = RewTerm(func=mdp.is_alive, weight=0.15)
    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    base_yaw = RewTerm(func=body_rewards.base_yaw_no_command_l2, weight=-1.0, params={"command_threshold": 0.1})
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.15)
    action_acc = RewTerm(func=mdp.action_acc_l2, weight=-0.05)   # 二阶平滑, 压高频颤动(膝 5Hz)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-4.0)

    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*2", ".*6"])},
    )
    joint_deviation_hip_yaw = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*3"])},
    )

    # 与当前 rough 一致，轻量惩罚踝 pitch/roll 动作及力矩。
    ankle_action = RewTerm(
        func=mdp.ankle_action,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*5", ".*6"])},
    )
    ankle_torque = RewTerm(
        func=mdp.ankle_torque,
        weight=-0.0005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*5", ".*6"])},
    )

    # -- robot
    base_height = RewTerm(func=mdp.base_height_l2, weight=-5.0, params={"target_height": W1196_NOMINAL_BASE_HEIGHT})

    # -- feet
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=1.0,
        params={"moving_only": False, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Link_R6", "Link_L6"], preserve_order=True)},
    )

    feet_y_distance = RewTerm(
        func=mdp.feet_y_distance,
        weight=-2.0,
        params={"command_threshold": 0.1, "threshold": W1196_FEET_WIDTH, "asset_cfg": SceneEntityCfg("robot", body_names=["Link_R6", "Link_L6"], preserve_order=True)},
    )

    feet_x_distance = RewTerm(
        func=mdp.feet_x_distance,
        weight=-2.0,
        params={
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=["Link_R6", "Link_L6"], preserve_order=True),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Link_R6", "Link_L6"], preserve_order=True),
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["Link_R6", "Link_L6"], preserve_order=True),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Link_R6", "Link_L6"], preserve_order=True),
        },
    )

    feet_clearance = RewTerm(
        func=mdp.feet_clearance,
        weight=1.0,
        params={"moving_only": False, "target_height": 0.06, "foot_height_offset": W1196_FOOT_HEIGHT_OFFSET, "asset_cfg": SceneEntityCfg("robot", body_names=["Link_R6", "Link_L6"], preserve_order=True)},
    )

    feet_contact_yaw = RewTerm(
        func=body_rewards.feet_contact_yaw_l2,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["Link_R6", "Link_L6"], preserve_order=True),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Link_R6", "Link_L6"], preserve_order=True),
            "min_force": 5.0,
            "full_force": 20.0,
        },
    )

    feet_contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=-0.0002,
        params={"threshold": 300, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Link_R6", "Link_L6"], preserve_order=True)},
    )

    feet_flat = RewTerm(
        func=mdp.feet_flat,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["Link_R6", "Link_L6"], preserve_order=True)},
    )

    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Link_R6", "Link_L6"], preserve_order=True)},
    )

@configclass
class FlatBodyTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})
    # 与当前 W1196 rough 相同的倒地判据；base 原点位于髋轴。
    base_height = DoneTerm(
        func=mdp.base_height_below_feet,
        params={"minimum_height": 0.14, "asset_cfg": SceneEntityCfg("robot", body_names=["Link_R6", "Link_L6"], preserve_order=True)},
    )


@configclass
class FlatBodyCurriculumCfg:
    """Curriculum terms for the MDP."""
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)
    ang_vel_cmd_levels = CurrTerm(mdp.ang_vel_cmd_levels)


@configclass
class FlatBodyEnvCfg(ManagerBasedRLEnvCfg):
    """W1196 平地行走/零速踏步；所有任务配置在本文件明确列出。"""

    # Scene settings
    scene: FlatBodySceneCfg = FlatBodySceneCfg(num_envs=2048, env_spacing=2.5)
    # Basic settings
    observations: FlatBodyObservationsCfg = FlatBodyObservationsCfg()
    actions: FlatBodyActionsCfg = FlatBodyActionsCfg()
    commands: FlatBodyCommandsCfg = FlatBodyCommandsCfg()
    # MDP settings
    rewards: FlatBodyRewardsCfg = FlatBodyRewardsCfg()
    terminations: FlatBodyTerminationsCfg = FlatBodyTerminationsCfg()
    events: FlatBodyEventCfg = FlatBodyEventCfg()
    curriculum: FlatBodyCurriculumCfg = FlatBodyCurriculumCfg()
    # 步态时钟参数（会被 dump 到 env.yaml）
    gait: FlatBodyGaitCfg = FlatBodyGaitCfg()

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

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class FlatBodyPlayEnvCfg(FlatBodyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32


@configclass
class FlatBodyMLPActorCfg:
    """Only the fields that rsl-rl 5.x MLPModel.__init__ accepts."""
    class_name: str = "MLPModel"
    hidden_dims: list = None
    activation: str = "elu"
    obs_normalization: bool = False
    distribution_cfg: dict = None


@configclass
class FlatBodyMLPCriticCfg:
    class_name: str = "MLPModel"
    hidden_dims: list = None
    activation: str = "elu"
    obs_normalization: bool = False


@configclass
class FlatBodyPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """独立 PPO 配置；15 帧、705D actor / 930D critic，无高度图。"""

    class_name = "OnPolicyRunner"
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    clip_actions = 5.0
    check_for_nan = True
    experiment_name = "nlegs_body_flat"
    run_name = "w1196"
    logger = "tensorboard"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    actor = FlatBodyMLPActorCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
    )
    critic = FlatBodyMLPCriticCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            mirror_loss_coeff=1.0,
            data_augmentation_func=compute_symmetric_states,
        ),
    )
