"""nlegs_rough_zero_start：完整独立配置，零速站稳，有命令行走。

本文件包含机器人、地形、命令、动作、观测、奖励、事件、终止、课程及 PPO 参数。
通用 MDP 算法仍复用 mdp；修改本任务参数无需修改 rough_env_cfg.py。
以真机较好的 2026-09-29_14-52-25 为动力学参考，保留零速站立门控。
2026-09-29 审计候选：解除额外踝目标约束，按有效运动累计地形课程。
此配置尚不是经过真机验证的改善结果；试验依据见 analysis/robot_host_2026-09-29。
"""

from pathlib import Path
from dataclasses import MISSING
import torch

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg
from legs_rl_lab.actuators import DelayedDCMotorCfg
from legs_rl_lab.tasks.nlegs_task.mdp.symmetry import compute_symmetric_states

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

from legs_rl_lab.tasks.nlegs_task import mdp

# ====================== 机器人与零速启动参数 ======================
# 当前默认模型；换同构模型可直接改此路径。不同结构还需同步下方关节、
# 执行器、传感器路径、足部选择器、站姿和高度奖励，以及 sim2sim 的 SCENE_XML。
ROBOT_USD_PATH = str(Path(__file__).resolve().parents[4] / "assets/nlegs_body/mjcf/nlegs_body/nlegs_body.usd")
ROBOT_INIT_HEIGHT = 0.50  # reset 留出少量落地空间，与 14-52-25 一致
ROBOT_TARGET_HEIGHT = 0.46  # 新 base 原点下的站立高度，不随 reset 高度改变
STARTUP_STAND_SECONDS = 2.0
COMMAND_THRESHOLD = 1e-6  # 仅忽略浮点残差，小速度和纯转向也算移动
# 默认保留 USD 的膝限位。只有确认编码器零位、连杆干涉及有效行程后才设为 1.45。
# 开启时还必须同步 sim2sim 的 MJCF 限位；此事件不修改原始 USD 或真实电机限位。
VERIFIED_KNEE_UPPER_LIMIT = None


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

NLEGS_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUsdFileCfg(
        usd_path=ROBOT_USD_PATH,
    ),
    articulation_root_prim_path='/base/base',
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, ROBOT_INIT_HEIGHT),  # 默认站姿贴近平地
        joint_pos={
            ".*1": -0.1,
            ".*4": 0.2,
            ".*5": -0.1,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": DelayedDCMotorCfg(
            joint_names_expr=[".*1", ".*2", ".*3", ".*4"],
            effort_limit=26.0,
            saturation_effort=26.0,
            velocity_limit=7.0,
            stiffness={
                ".*1": 200.0,
                ".*2": 200.0,
                ".*3": 200.0,
                ".*4": 250.0,
            },
            damping={
                ".*1": 5.0,
                ".*2": 5.0,
                ".*3": 5.0,
                ".*4": 5.0,
            },
            armature=0.0509,
            friction=0.5,
            dynamic_friction=0.5,
            min_delay=3,
            max_delay=7,
        ),
        "ankle_pitch": DelayedDCMotorCfg(
            joint_names_expr=[".*5"],
            effort_limit=26.0,
            saturation_effort=26.0,
            velocity_limit=7.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.0509,
            friction=0.5,
            dynamic_friction=0.5,
            min_delay=3,
            max_delay=7,
        ),
        "ankle_roll": DelayedPDActuatorCfg(
            joint_names_expr=[".*6"],
            # 显式 PD 内部限矩与 PhysX 限矩分别设置，避免从 USD 继承零限矩。
            effort_limit=7,
            velocity_limit=14.0,
            stiffness=40.0,
            damping=0.5,
            armature=0.00219,
            effort_limit_sim=5.8,
            velocity_limit_sim=14.0,
            min_delay=3,
            max_delay=7,
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

class ZeroStartVelocityCommand(UniformVelocityCommand):
    """每次 reset 先零速站立，随后正常采样走/停命令。"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.expected_distance = torch.zeros(self.num_envs, device=self.device)
        self.directed_distance = torch.zeros_like(self.expected_distance)
        self.moving_seconds = torch.zeros_like(self.expected_distance)
        self.max_displacement = torch.zeros_like(self.expected_distance)

    def _update_metrics(self):
        super()._update_metrics()
        speed = self.vel_command_b[:, :2].norm(dim=-1)
        # 排除 reset 后新一幕的首帧、零速和纯转向。命令在本方法之后重采样。
        active = (speed > 0.05) & (self._env.episode_length_buf > 0)
        dt = self._env.step_dt
        direction = self.vel_command_b[:, :2] / speed.clamp_min(1e-6).unsqueeze(-1)
        along = (self.robot.data.root_lin_vel_b[:, :2] * direction).sum(-1)
        self.expected_distance += speed * dt * active
        self.directed_distance += along * dt * active
        self.moving_seconds += dt * active
        displacement = (self.robot.data.root_pos_w[:, :2] - self._env.scene.env_origins[:, :2]).norm(dim=-1)
        self.max_displacement = torch.maximum(self.max_displacement, displacement * active)

    def reset(self, env_ids=None):
        metrics = super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        for value in (self.expected_distance, self.directed_distance, self.moving_seconds, self.max_displacement):
            value[ids] = 0.0
        self.vel_command_b[ids] = 0.0
        self.is_standing_env[ids] = True
        self.time_left[ids] = self.cfg.startup_stand_seconds
        return metrics

def terrain_levels_active_motion(env, env_ids):
    """只用有效平移时段评价；需要离开中央平台，站立/纯转向不降级。"""
    term = env.command_manager.get_term("base_velocity")
    expected = term.expected_distance[env_ids]
    ratio = term.directed_distance[env_ids] / expected.clamp_min(1e-6)
    enough = (term.moving_seconds[env_ids] >= 2.0) & (expected >= 0.5)
    failed = env.termination_manager.terminated[env_ids]
    move_up = enough & (expected >= 2.0) & (ratio >= 0.8) & (term.max_displacement[env_ids] >= 2.4) & ~failed
    move_down = enough & ((ratio < 0.5) | failed) & ~move_up
    env.scene.terrain.update_env_origins(env_ids, move_up, move_down)
    return env.scene.terrain.terrain_levels.float().mean()

@configclass
class ZeroStartVelocityCommandCfg(UniformVelocityCommandCfg):
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
    class_type: type = ZeroStartVelocityCommand
    startup_stand_seconds: float = STARTUP_STAND_SECONDS

def standing_joint_pos_l1(env, command_threshold: float):
    return mdp.joint_deviation_l1(env) * ~mdp.command_is_moving(env, command_threshold)

def standing_joint_vel_l2(env, command_threshold: float):
    return mdp.joint_vel_l2(env) * ~mdp.command_is_moving(env, command_threshold)

def ankle_roll_target_l2(env, asset_cfg: SceneEntityCfg):
    """惩罚裁剪前目标偏移；脚被接触约束挡住时仍能约束策略输出。"""
    term = env.action_manager.get_term("JointPositionAction")
    # 本任务动作覆盖全部关节，且动作顺序等于 articulation 关节顺序。
    delta = term.raw_actions[:, asset_cfg.joint_ids] * term.cfg.scale
    return delta.square().sum(dim=1)

def standing_feet_contact(env, sensor_cfg: SceneEntityCfg, command_threshold: float):
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, 2]
    return (forces > 1.0).all(dim=1).float() * ~mdp.command_is_moving(env, command_threshold)

# nlegs_body 默认站姿高度约 0.46m。num_rows = 难度等级，num_cols = 每级变体数。
NLEGS_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,  # 行=难度，配合 terrain_levels_active_motion
    sub_terrains={
        # 20% 平地：保底，避免一上来全是难地形
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.02, 0.06), noise_step=0.02, border_width=0.25
        ),
        # 金字塔坡 / 反金字塔坡：坡度 ≤0.3(≈17°)
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, 0.3), platform_width=1.5, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.3), platform_width=1.5, border_width=0.25
        ),
        # 台阶 / 反台阶：单级高 4~10cm
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.03, 0.12),
            step_width=0.3,
            platform_width=1.5,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.03, 0.12),
            step_width=0.3,
            platform_width=1.5,
            border_width=1.0,
            holes=False,
        ),
    },
)

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
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.6,
            dynamic_friction=0.6,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = NLEGS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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
    """rough 的全部事件；执行器名义 PD 来自 NLEGS_CFG。"""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.5),
            "dynamic_friction_range": (0.3, 1.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # 机身局部坐标系的质心偏移（m）；工程初始范围，非实测辨识区间。
    # 每环境初始化时独立均匀采样一次；该函数为增量写入，不在 reset 时累加。
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
        },
    )

    # reset
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.4, 0.4), "y": (-0.4, 0.4), "yaw": (-3.14, 3.14)},
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
            # 从默认站姿启动；可在这里增加关节重置随机范围
            "position_range": (-0.2, 0.2),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # 标0偏置随机化: 每环境每关节一个恒定偏置(reset 采样, 整幕不变), 模拟实机编码器零位
    # q_encoder = q_physical + b：策略观测加 b，电机物理目标减同一个 b；critic 保持真值。
    # ±0.05rad(≈±2.9°)沿用工程覆盖范围，并非实测置信区间；不保证消除实机零速漂移。
    joint_zero_bias = EventTerm(
        func=mdp.randomize_joint_zero_bias,
        mode="reset",
        params={"bias_range": (-0.04, 0.04)},
    )

    # 每关节、每环境、每回合独立采样；scale 始终基于 NLEGS_CFG 名义值，不逐回合累乘。
    randomize_leg_gains = EventTerm(
        func=mdp.randomize_actuator_gains, mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*[1-4]"]),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale", "distribution": "uniform",
        },
    )
    randomize_ankle_pitch_gains = EventTerm(
        func=mdp.randomize_actuator_gains, mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*5"]),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale", "distribution": "uniform",
        },
    )
    randomize_ankle_roll_gains = EventTerm(
        func=mdp.randomize_actuator_gains, mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*6"]),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale", "distribution": "uniform",
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = ZeroStartVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=ZeroStartVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.5), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.5, 0.5)
        ),
        limit_ranges=ZeroStartVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.5), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.5, 0.5)
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        class_type=mdp.ZeroBiasJointPositionAction,
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True,
        clip=None,  # 与两版较好策略一致；仍保留 PPO 原始动作裁剪和关节物理限位
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
        gait_phase = ObsTerm(
            func=mdp.gait_phase_obs,
            params={"gate_by_cmd": True, "command_threshold": COMMAND_THRESHOLD},
        )

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
        gait_phase = ObsTerm(
            func=mdp.gait_phase_obs,
            params={"gate_by_cmd": True, "command_threshold": COMMAND_THRESHOLD},
        )

        # 187 点高度图只进入 critic，actor 保持盲走。
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": ROBOT_TARGET_HEIGHT},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.history_length = 10

    # privileged observations
    critic: CriticCfg = CriticCfg()

@configclass
class RewardsCfg:
    """完整奖励：零速关闭踏步/抬脚，保留平衡和高度约束并鼓励站稳。"""

    ankle_roll_target = None  # 解除 zero-start 独有的额外裁剪前动作惩罚

    # -- 零速专用站立奖励（受扰失衡时仍允许策略调整脚步）
    standing_joint_pos = RewTerm(
        func=standing_joint_pos_l1, weight=-1.0,
        params={"command_threshold": COMMAND_THRESHOLD},
    )
    standing_joint_vel = RewTerm(
        func=standing_joint_vel_l2, weight=-0.02,
        params={"command_threshold": COMMAND_THRESHOLD},
    )
    standing_feet_contact = RewTerm(
        func=standing_feet_contact, weight=0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*6"),
                "command_threshold": COMMAND_THRESHOLD},
    )

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
    action_acc = RewTerm(func=mdp.action_acc_l2, weight=-0.05)   # 二阶平滑；不能由此推断真实颤动频率
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)

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
        params={"target_height": ROBOT_TARGET_HEIGHT, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )

    # -- feet
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*6"),
                "moving_only": True, "command_threshold": COMMAND_THRESHOLD},
    )

    feet_y_distance = RewTerm(
        func=mdp.feet_y_distance,
        weight=-2.0,
        params={"threshold": 0.25, "asset_cfg": SceneEntityCfg("robot", body_names=".*6")},
    )

    feet_x_distance = RewTerm(
        func=mdp.feet_x_distance,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*6")},
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*6"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*6"),
        },
    )

    feet_clearance = RewTerm(
        func=mdp.feet_clearance,
        weight=1.0,
        params={
            "target_height": 0.14,  # 软奖励目标，不是实际足底离地高度保证
            "moving_only": True, "command_threshold": COMMAND_THRESHOLD,
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
        weight=-0.2,
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
    # nlegs_body 默认 base 到脚部 body 约 0.445m，塌陷终止高度与 rough 对齐。
    base_height = DoneTerm(
        func=mdp.base_height_below_feet,
        params={"minimum_height": 0.16, "asset_cfg": SceneEntityCfg("robot", body_names=".*6")},
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    # 命令范围固定；不让零速站立期间的速度奖励推动命令课程。
    lin_vel_cmd_levels = None
    ang_vel_cmd_levels = None
    terrain_levels = CurrTerm(func=terrain_levels_active_motion)

@configclass
class RoughZeroStartEnvCfg(ManagerBasedRLEnvCfg):
    """完整零速静止任务；训练 reset 先站立，再采样走/停命令。"""

    command_threshold: float = COMMAND_THRESHOLD

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
        if VERIFIED_KNEE_UPPER_LIMIT is not None:
            self.events.verified_knee_limits = EventTerm(
                func=mdp.set_joint_position_limits, mode="startup",
                params={"joint_limits": {name: (None, VERIFIED_KNEE_UPPER_LIMIT)
                                         for name in ("joint_L4", "joint_R4")}},
            )

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

@configclass
class RoughZeroStartPlayEnvCfg(RoughZeroStartEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Play 使用名义 PD，避免回放对照每次重置得到不同的执行器。
        self.events.randomize_leg_gains = None
        self.events.randomize_ankle_pitch_gains = None
        self.events.randomize_ankle_roll_gains = None
        self.observations.policy.enable_corruption = False
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.max_init_terrain_level = 4

# ====================== 本任务独立 PPO 参数 ======================
@configclass
class MLPActorCfg:
    """Only the fields that rsl-rl 5.x MLPModel.__init__ accepts."""
    class_name: str = "MLPModel"
    hidden_dims: list = None
    activation: str = "elu"
    obs_normalization: bool = False
    distribution_cfg: dict = None

@configclass
class MLPCriticCfg:
    class_name: str = "MLPModel"
    hidden_dims: list = None
    activation: str = "elu"
    obs_normalization: bool = False

@configclass
class NlegsRoughZeroStartPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    clip_actions = 5.0
    experiment_name = "nlegs_rough_zero_start"
    actor = MLPActorCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "log"},
    )
    critic = MLPCriticCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
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
