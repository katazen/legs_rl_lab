"""nlegs_rough：零速原地踏步；机器人、执行器、地形和全部 MDP 配置在本文件定义。

不继承 flat 或公共机器人配置。复用 USD 几何资产和 mdp/执行器实现。
2026-09-20 左右踝 pitch 使用悬空步态回放的等效候选，固定 PD 与延迟作对照；
7 rad/s、26 N·m 仍是未辨识的模型假设，不代表实测速度/力矩上限。
膝限位块已拆除，L4/R4 不加目标裁剪或实测端点，保留 USD 原有活动范围。
"""

from pathlib import Path

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.actuators import DelayedPDActuatorCfg
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

from legs_rl_lab.actuators import DelayedDCMotorCfg
from legs_rl_lab.tasks.nlegs_task import mdp


# 实测单侧止挡；未列出的端点保留 USD 原值。膝 L4/R4 不在本表中。
CROUCH_LIMITS = {
    "joint_L1": (-0.9752422370, None),
    "joint_L2": (None, 0.4964904250),
    "joint_L3": (None, 0.4983978027),
    "joint_L5": (-0.3763256275, None),
    "joint_R1": (-0.9676127260, None),
    "joint_R2": (-0.5270084688, None),
    "joint_R3": (-0.6185626001, None),
    "joint_R5": (-0.4224841688, None),
}


# 按 0.58m 小机身温和缩放的 rough 地形。num_rows = 难度等级(课程沿行递增)，num_cols = 每级的变体数。
NLEGS_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,  # 行=难度，配合 terrain_levels_vel 课程
    sub_terrains={
        # 20% 平地：保底，避免一上来全是难地形
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        # 随机起伏(碎石感)：温和 2~6cm
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.06), noise_step=0.02, border_width=0.25
        ),
        # 金字塔坡 / 反金字塔坡：坡度 ≤0.3(≈17°)
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
        ),
        # 随机方块：矮块 2~8cm
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15, grid_width=0.45, grid_height_range=(0.02, 0.08), platform_width=2.0
        ),
        # 台阶 / 反台阶：单级高 3~12cm(远低于 g1 的 5~23cm)
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.075,
            step_height_range=(0.03, 0.12),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.075,
            step_height_range=(0.03, 0.12),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


@configclass
class RoughRobotCfg(ArticulationCfg):
    joint_sdk_names: list[str] = [
        "joint_R1", "joint_R2", "joint_R3", "joint_R4", "joint_R5", "joint_R6",
        "joint_L1", "joint_L2", "joint_L3", "joint_L4", "joint_L5", "joint_L6",
    ]
    soft_joint_pos_limit_factor: float = 0.95


@configclass
class GaitCfg:
    """步态时钟参数, 会被 dump 到 env.yaml; reward / observation 通过 env.cfg.gait.* 读取。"""

    period: float = 0.6             # 步态周期 (s)
    stance_ratio: float = 0.55      # 支撑相占比
    feet_offset: list = [0.0, 0.5]  # 左右腿相位偏移


@configclass
class RoughSceneCfg(InteractiveSceneCfg):
    """窄本体机器人与生成式 rough 地形。"""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=NLEGS_ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
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
    robot: RoughRobotCfg = RoughRobotCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(__file__).resolve().parents[4] / "assets/nlegs/mjcf/nlegs/nlegs.usd"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        articulation_root_prim_path="/base/base",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.62),
            joint_pos={".*1": -0.1, ".*4": 0.2, ".*5": -0.1},
            joint_vel={".*": 0.0},
        ),
        # delay 单位为 5 ms 物理步。髋/膝/踝 roll 保留原参数。
        # pitch 的 26/7 是暂用假设；本轮只支持下列整组等效参数，不支持真实上限或 DR 区间。
        actuators={
            "legs": DelayedDCMotorCfg(
                joint_names_expr=[".*1", ".*2", ".*3"],
                effort_limit=26.0,
                saturation_effort=26.0,
                velocity_limit=7.0,
                stiffness={".*1": 200.0, ".*2": 100.0, ".*3": 100.0},
                damping={".*1": 5.0, ".*2": 5.0, ".*3": 5.0},
                armature=0.0509,
                friction=0.5,
                dynamic_friction=0.5,
                viscous_friction=0.0,
                min_delay=4,
                max_delay=6,
            ),
            "knees": DelayedDCMotorCfg(
                joint_names_expr=[".*4"],
                effort_limit=26.0,
                saturation_effort=26.0,
                velocity_limit=7.0,
                stiffness=250.0,
                damping=5.0,
                armature=0.0509,
                friction=0.5,
                dynamic_friction=0.5,
                viscous_friction=0.0,
                min_delay=1,
                max_delay=6,
            ),
            "ankle_pitch_left": DelayedDCMotorCfg(
                joint_names_expr=["joint_L5"],
                effort_limit=26.0,
                saturation_effort=26.0,
                velocity_limit=7.0,
                stiffness=40.0,
                damping=2.0,
                armature=0.035,
                friction=0.51,
                dynamic_friction=0.26,
                viscous_friction=0.0,
                min_delay=3,
                max_delay=3,
            ),
            "ankle_pitch_right": DelayedDCMotorCfg(
                joint_names_expr=["joint_R5"],
                effort_limit=26.0,
                saturation_effort=26.0,
                velocity_limit=7.0,
                stiffness=40.0,
                damping=2.0,
                armature=0.0509,
                friction=0.55,
                dynamic_friction=0.385,
                viscous_friction=0.0,
                min_delay=2,
                max_delay=2,
            ),
            "ankle_roll": DelayedPDActuatorCfg(
                joint_names_expr=[".*6"],
                stiffness=40.0,
                damping=0.5,
                armature=0.00219,
                effort_limit_sim=5.8,
                velocity_limit_sim=14.0,
                friction=0.0,
                dynamic_friction=0.0,
                viscous_friction=0.54,
                min_delay=2,
                max_delay=6,
            ),
        },
    )

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
class EventCfg:
    """rough 的全部事件；保留既有扰动，踝 pitch 固定为本轮候选 PD。"""

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

    crouch_joint_limits = EventTerm(
        func=mdp.set_joint_position_limits, mode="startup", params={"joint_limits": CROUCH_LIMITS},
    )

    # 机身局部坐标系的质心偏移（m）；工程初始范围，非实测辨识区间。
    # 每环境初始化时独立均匀采样一次；该函数为增量写入，不在 reset 时累加。
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.01, 0.01), "z": (-0.02, 0.02)},
        },
    )

    # reset
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (1.0, 5.0),
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

    # 标0偏置随机化: 每环境每关节一个恒定偏置(reset 采样, 整幕不变), 模拟实机编码器零位
    # 标定误差, 让策略对恒定零偏鲁棒；不保证消除实机零速漂移。仅作用于策略观测的 joint_pos_rel。
    # 对称 ±0.05rad(≈±2.9°): 均值0 是有意的(目的是不敏感, 非拟合实机特定偏置)。
    joint_zero_bias = EventTerm(
        func=mdp.randomize_joint_zero_bias,
        mode="reset",
        params={"bias_range": (-0.05, 0.05)},
    )

    # 对照配置固定 pitch 的 Kp=40、Kd=2；没有已验证的增益随机化范围。
    randomize_ankle_gains = None
    randomize_ankle_roll_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*6"]),
            "stiffness_distribution_params": (0.85, 1.1),
            "damping_distribution_params": (0.5, 1.0),
            "operation": "scale",
            "distribution": "uniform",
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
class CommandsCfg:
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
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True,
        clip={
            name: (lo if lo is not None else -float("inf"), hi if hi is not None else float("inf"))
            for name, (lo, hi) in CROUCH_LIMITS.items()
        },
    )


@configclass
class ObservationsCfg:
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
            self.history_length = 10
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
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.58},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.history_length = 10

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """rough 奖励完整定义；零速时仍运行步态、抬脚和高度奖励。"""

    # -- task
    track_lin_vel_xy = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=1.0)
    track_ang_vel_z = RewTerm(func=mdp.track_ang_vel_z_exp, weight=1.0)
    alive = RewTerm(func=mdp.is_alive, weight=0.15)
    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    base_angular = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.15)
    action_acc = RewTerm(func=mdp.action_acc_l2, weight=-0.05)   # 二阶平滑, 压高频颤动(膝 5Hz)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)

    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*2", ".*3", ".*6"])},
    )

    # 让踝(pitch .*5 + roll .*6)被动，脚触地自然贴合，抑制踝滚转侧崴（移植自 TienKung，沿用其权重）
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
    base_height = RewTerm(
        func=mdp.base_height_l2, weight=-2.0,
        params={"target_height": 0.58, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )

    # -- feet
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*6")},
    )

    feet_y_distance = RewTerm(
        func=mdp.feet_y_distance,
        weight=-2.0,
        params={"threshold": 0.222, "asset_cfg": SceneEntityCfg("robot", body_names=".*6")},
    )

    feet_x_distance = RewTerm(
        func=mdp.feet_x_distance,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*6")},
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*6"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*6"),
        },
    )

    feet_clearance = RewTerm(
        func=mdp.feet_clearance,
        weight=1.0,
        params={
            "target_height": 0.12,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*6"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )

    feet_contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=-0.0002,
        params={"threshold": 200, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*6")},
    )

    feet_flat = RewTerm(
        func=mdp.feet_flat,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*6")},
    )

    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*6"])},
    )

    # -- other
    # 无 y 速度命令时抑制身体左右平移(体系 y 线速度罚), 有横移命令时自动关闭
    lateral_move = RewTerm(
        func=mdp.base_lateral_move_l2,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # 注: nlegs 已删 undesired_contacts, 故这里不声明。


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})
    # 相对量 base_z - min(feet_z) < 0.2m 即终止(身体塌到脚边), 与地形绝对高度无关
    base_height = DoneTerm(
        func=mdp.base_height_below_feet,
        params={"minimum_height": 0.2, "asset_cfg": SceneEntityCfg("robot", body_names=".*6")},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)
    ang_vel_cmd_levels = CurrTerm(mdp.ang_vel_cmd_levels)
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class RoughEnvCfg(ManagerBasedRLEnvCfg):
    """独立 rough 配置；0 速仍原地踏步。"""

    # Scene settings
    scene: RoughSceneCfg = RoughSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
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

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt


@configclass
class RoughPlayEnvCfg(RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.max_init_terrain_level = 4
