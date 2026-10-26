"""nlegs_body_rough：独立的新本体崎岖地形步行配置，全部任务与 PPO 参数就地定义。"""

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
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
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

from legs_rl_lab.assets.nlegs_body.nlegs_body import NLEGS_BODY_CFG
from legs_rl_lab.tasks.nlegs_task import mdp
from legs_rl_lab.tasks.nlegs_task.mdp.symmetry import compute_symmetric_states
from . import body_rewards


# 平地、随机崎岖路面和上下坡；从完整难度范围随机生成，无地形课程。
NLEGS_BODY_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    difficulty_range=(0.0, 1.0),
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.06), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.3, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.3, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
        ),
    },
)


@configclass
class BodyGaitCfg:
    """步态时钟参数, 会被 dump 到 env.yaml; reward / observation 通过 env.cfg.gait.* 读取。"""

    period: float = 0.6             # 完整周期(s)：支撑0.33、摆动0.27、每次双支撑0.03
    stance_ratio: float = 0.55      # 支撑相占比
    feet_offset: list = [0.0, 0.5]  # 右腿、左腿相位偏移，与足部实体顺序一致


@configclass
class RoughBodySceneCfg(InteractiveSceneCfg):
    """nlegs_body 机器人与不含楼梯的生成式地形。"""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=NLEGS_BODY_ROUGH_TERRAINS_CFG,
        max_init_terrain_level=None,
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
    robot: ArticulationCfg = NLEGS_BODY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
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
class BodyEventCfg:
    """本任务的材料、质心、质量、零偏、踝 PD 与外部扰动随机化。"""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.1, 1.3),
            "dynamic_friction_range": (0.1, 1.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # 局部质心偏移只在初始化时叠加一次，避免 reset 时逐次累积。
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.03, 0.01), "y": (-0.01, 0.01), "z": (-0.02, 0.02)},
        },
    )

    # reset
    # 新资产已包含躯干；围绕名义质量缩放，避免重复叠加负载。
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-2, 2),
            "operation": "add",
        },
    )

    link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Link_[LR][1-6]"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
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

    # 同一零偏 b：观测加 b，执行器目标减 b。
    joint_zero_bias = EventTerm(
        func=mdp.randomize_joint_zero_bias,
        mode="reset",
        params={"bias_range": (-0.05, 0.05)},
    )

    # randomize_ankle_gains = EventTerm(
    #     func=mdp.randomize_actuator_gains, mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*5"]),
    #         "stiffness_distribution_params": (0.7, 1.3),
    #         "damping_distribution_params": (0.7, 1.5),
    #         "operation": "scale", "distribution": "uniform",
    #     },
    # )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class BodyCommandsCfg:
    """Command specifications for the MDP."""

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
class BodyActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        class_type=mdp.ZeroBiasJointPositionAction,
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True,
    )


@configclass
class BodyObservationsCfg:
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
        gait_phase = ObsTerm(func=mdp.gait_phase_obs)

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
        gait_phase = ObsTerm(func=mdp.gait_phase_obs)

        # 187 点高度图只进入 critic，actor 保持盲走。
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.45},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.history_length = 15

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class BodyRewardsCfg:
    """水平机身、横向站距和三 pitch 软参考；零速时仍按相位踏步。"""

    # -- task
    track_lin_vel_xy = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=1.0)
    track_ang_vel_z = RewTerm(func=mdp.track_ang_vel_z_exp, weight=1.0)
    alive = RewTerm(func=mdp.is_alive, weight=0.15)
    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    base_angular = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.3)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.15)
    action_acc = RewTerm(func=mdp.action_acc_l2, weight=-0.05)   # 二阶平滑, 压高频颤动(膝 5Hz)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    # 无倾角容忍带：总倾角 10° 时，未乘 step_dt 的惩罚约为 -0.45。
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-15.0)
    # 摆动相内 sin²(pi*u)，增量 [-a, 2a, -a]；不直接叠加到 action。
    # a=0.2 的固定 base 几何抬升仅约 2.63 cm，允许偏离此软参考达到足高目标。
    pitch_reference = RewTerm(
        func=body_rewards.pitch_reference,
        weight=0.5,
        params={"amplitude": 0.2, "std": 0.2},
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*2", ".*3", ".*6"])},
    )

    # 动作偏离只约束踝 roll，避免抵消踝 pitch 周期参考；保留轻量力矩惩罚。
    ankle_action = RewTerm(
        func=mdp.ankle_action,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*6"])},
    )
    ankle_torque = RewTerm(
        func=mdp.ankle_torque,
        weight=-0.0005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*5", ".*6"])},
    )

    # 新站姿实际 USD 脚底贴地时 root 高度约为 0.449893 m。
    base_height = RewTerm(
        func=mdp.base_height_l2, weight=-2.0,
        params={"target_height": 0.45, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )

    # -- feet
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=1.0,
        params={"sensor_cfg": SceneEntityCfg(
            "contact_forces", body_names=["Link_R6", "Link_L6"], preserve_order=True
        )},
    )

    # 恢复原 body_rough：机身坐标系横向脚距绝对误差，仅 |cmd_y| < 0.1 时生效。
    feet_y_distance = RewTerm(
        func=mdp.feet_y_distance,
        weight=-1.0,
        params={"threshold": 0.266, "asset_cfg": SceneEntityCfg("robot", body_names=".*6")},
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            # 相位、脚速与接触都显式采用右脚、左脚顺序。
            "asset_cfg": SceneEntityCfg("robot", body_names=["Link_R6", "Link_L6"], preserve_order=True),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Link_R6", "Link_L6"], preserve_order=True),
        },
    )

    feet_clearance = RewTerm(
        func=body_rewards.feet_clearance,
        weight=1.5,
        params={
            "target_height": 0.06,
            "asset_cfg": SceneEntityCfg("robot", body_names=["Link_R6", "Link_L6"], preserve_order=True),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )

    feet_contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=-0.0002,
        params={"threshold": 200, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*6")},
    )

    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*6"])},
    )

    # 允许保持水平时为换脚而横向转移重心；横向行走速度由 tracking 项约束。


@configclass
class BodyTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})
    base_height = DoneTerm(
        func=mdp.base_height_below_feet,
        params={"minimum_height": 0.16, "asset_cfg": SceneEntityCfg("robot", body_names=".*6")},
    )


@configclass
class BodyCurriculumCfg:
    """仅保留速度命令课程接口；不更新地形等级。"""
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)
    ang_vel_cmd_levels = CurrTerm(mdp.ang_vel_cmd_levels)


@configclass
class RoughBodyEnvCfg(ManagerBasedRLEnvCfg):
    """独立步行任务；配置均在本文件定义，0 速仍原地踏步。"""

    # Scene settings
    scene: RoughBodySceneCfg = RoughBodySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: BodyObservationsCfg = BodyObservationsCfg()
    actions: BodyActionsCfg = BodyActionsCfg()
    commands: BodyCommandsCfg = BodyCommandsCfg()
    # MDP settings
    rewards: BodyRewardsCfg = BodyRewardsCfg()
    terminations: BodyTerminationsCfg = BodyTerminationsCfg()
    events: BodyEventCfg = BodyEventCfg()
    curriculum: BodyCurriculumCfg = BodyCurriculumCfg()
    # 步态时钟参数（会被 dump 到 env.yaml）
    gait: BodyGaitCfg = BodyGaitCfg()

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
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt


@configclass
class RoughBodyPlayEnvCfg(RoughBodyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.max_init_terrain_level = None


@configclass
class BodyMLPActorCfg:
    """Only the fields that rsl-rl 5.x MLPModel.__init__ accepts."""
    class_name: str = "MLPModel"
    hidden_dims: list = None
    activation: str = "elu"
    obs_normalization: bool = False
    distribution_cfg: dict = None


@configclass
class BodyMLPCriticCfg:
    class_name: str = "MLPModel"
    hidden_dims: list = None
    activation: str = "elu"
    obs_normalization: bool = False


@configclass
class RoughBodyPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """独立 PPO 配置；12 个 mini-batch，actor/critic 均使用 15 帧历史。"""

    class_name = "OnPolicyRunner"
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    clip_actions = 5.0
    check_for_nan = True
    experiment_name = "nlegs_body_rough"
    run_name = ""
    logger = "tensorboard"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    actor = BodyMLPActorCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
    )
    critic = BodyMLPCriticCfg(
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
        num_mini_batches=16,
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
