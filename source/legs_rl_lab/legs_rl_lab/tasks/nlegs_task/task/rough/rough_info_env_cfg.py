"""nlegs rough_info —— rough 的"非盲走"版本: actor 也吃地形高度图。

与 nlegs_rough (rough_env_cfg.py) 的**三处**区别, 其余全部照搬:

1) 楼梯比例 15% -> 40%。原配方里 pyramid_stairs / _inv 各 0.075, 台阶样本太稀; 提到各 0.20
   并从 flat(0.2->0.1) 与两个坡(各 0.15->0.10)里匀出来。刻意**不做成纯楼梯**: 本任务保留
   完整速度跟踪命令, 目标是一个通用 rough 策略, 全是楼梯会丢掉其它地形上的能力。

2) height_scan 进 **actor** 观测(不再是盲走)。这是本任务存在的唯一理由 —— rough 盲走策略
   上不去 5cm 台阶的根因之一是它对前方地形只能事后感知(脚撞上立面才知道), 给了高度图就能
   事前预判。代价: **策略不再能直接部署到没有地形传感器的实机**, 实机需要 LiDAR/深度相机
   建高度图, 或只用于 sim2sim 验证"有地形信息能到什么上限"(隔离"感知问题 vs 能力问题")。

3) 扫描范围改为机身前 1.0m / 后 0.5m / 左右各 0.5m, 且 **actor 与 critic 都只用当前帧**。
   - grid_pattern 是以中心对称生成的(arange(-size/2, +size/2)), 表达不了前后不等的范围,
     故用 offset.pos 的 x 分量把整个网格前移 0.25m: size=[1.5, 1.0] + offset x=0.25
     -> x 从 -0.50 到 +1.00(16 点), y 从 -0.50 到 +0.50(11 点), 共 **176 点**。
     offset.pos 在 ray_alignment="yaw" 下会跟着机身朝向一起转(ray_caster.py: ray_starts_w
     = quat_apply_yaw(quat_w, ray_starts) + pos_w), 所以"前方 1m"是机身前方而非世界 +x。
   - offset.pos 的 z=20.0 是**射线起点抬升高度**, 不是扫描范围, 也不进 height_scan 的数值:
     height_scan = pos_w.z - hit.z - offset, 而 pos_w 取自 base prim 位姿, 不含 cfg.offset
     (cfg.offset 只加到 ray_starts 上, 见 ray_caster.py L218-224 vs L241-249)。
   - 只用当前帧: 地形本身不动, 历史高度图价值低; 10 帧会把 actor 撑到 470+1760=2230 维。
     改法与 rough_step 一致 —— group 级 history_length 留 None(它会覆盖 term 级设置), 改为
     逐 term 声明 history_length=10, height_scan 不声明(=0)。
     actor 470 -> 646, critic 620 -> 796。镜像布局见 mdp/symmetry.py。

保留不变: 速度跟踪命令(lin_vel_x/y + ang_vel_z 全范围, heading_command=False)、命令课程、
地形课程、rough 的全套奖励调整(含 feet_drag)与 bad_orientation 终止。

奖励上的唯一差异: feet_stumble -> feet_kick(同一摩擦锥判据, 但改成按撞击功率连续加权、
并读 net_forces_w_history 避开毫秒级脉冲被降采样漏掉)。

不继承任何现有 env cfg(全量展开), 改这里不会影响 nlegs_flat / nlegs_rough / nlegs_rough_step。
"""

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

from legs_rl_lab.assets.nlegs.nlegs import NLEGS_CFG
from legs_rl_lab.tasks.nlegs_task import mdp

# --- 高度扫描几何(改这里必须同步 mdp/symmetry.py 的 _HSCAN_INFO_* 与 sim2sim_info.py) ---
# 机身前 1.0m / 后 0.5m / 左右各 0.5m: 中心对称的 1.5x1.0 网格 + x 前移 0.25m
SCAN_RESOLUTION = 0.1
SCAN_SIZE = [1.5, 1.0]
SCAN_OFFSET = (0.25, 0.0, 20.0)   # z 是射线起点抬升, 不影响范围与数值
SCAN_HEIGHT_OFFSET = 0.58         # 名义 base 高度, 使 height_scan 数值围绕 0


# ============================== 地形 ==============================
# 与 NLEGS_ROUGH_TERRAINS_CFG 同一套子地形参数, 只改 proportion(楼梯 15% -> 40%)。
NLEGS_ROUGH_INFO_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,  # 行=难度, 配合 terrain_levels_vel 课程
    sub_terrains={
        # 10% 平地: 保底(rough 版是 20%, 匀给楼梯)
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.10),
        # 随机起伏(碎石感): 温和 2~6cm
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(0.02, 0.06), noise_step=0.02, border_width=0.25
        ),
        # 金字塔坡 / 反金字塔坡: 坡度 ≤0.3(≈17°)
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.10, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.10, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
        ),
        # 随机方块: 矮块 2~8cm
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15, grid_width=0.45, grid_height_range=(0.02, 0.08), platform_width=2.0
        ),
        # 台阶 / 反台阶: 各 0.075 -> 0.20(合计 40%)。单级高 3~12cm, 参数与 rough 版一致。
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.03, 0.12),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.03, 0.12),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


@configclass
class GaitCfg:
    """步态时钟参数, 会被 dump 到 env.yaml; reward / observation 通过 env.cfg.gait.* 读取。"""

    period: float = 0.7             # 步态周期 (s)
    stance_ratio: float = 0.55      # 支撑相占比
    feet_offset: list = [0.0, 0.5]  # 左右腿相位偏移


@configclass
class RoughInfoSceneCfg(InteractiveSceneCfg):
    """生成器 rough 地形(楼梯加重) + 窄本体机器人 + 高度扫描。"""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=NLEGS_ROUGH_INFO_TERRAINS_CFG,
        max_init_terrain_level=5,   # 初始铺在 0~5 级, 课程再上下调
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
    robot: ArticulationCfg = NLEGS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 高度扫描: 机身前 1.0m / 后 0.5m / 左右各 0.5m, 16x11=176 点
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/base",
        offset=RayCasterCfg.OffsetCfg(pos=SCAN_OFFSET),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=SCAN_RESOLUTION, size=SCAN_SIZE),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/.*", history_length=3, track_air_time=True
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """事件项(与 flat/rough 完全一致, 共 7 项)。"""

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

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (1.0, 4.0),
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
        params={"position_range": (-0.2, 0.2), "velocity_range": (-1.0, 1.0)},
    )

    # 标0偏置随机化: 每环境每关节一个恒定偏置, 模拟实机编码器零位标定误差
    joint_zero_bias = EventTerm(
        func=mdp.randomize_joint_zero_bias,
        mode="reset",
        params={"bias_range": (-0.05, 0.05)},
    )

    # 踝(pitch .*5)专属带宽随机化(48V 辨识后收窄)
    randomize_ankle_gains = EventTerm(
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

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class CommandsCfg:
    """速度跟踪命令 —— 与 flat/rough 完全一致(全范围 vx/vy/wz, 无 heading 命令)。"""

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
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """观测: actor 与 critic 都带 height_scan(仅当前帧), 其余本体项 10 帧历史。

    两个组都**逐 term 声明 history_length**, group 级刻意留 None —— ObservationManager
    里 group.history_length 若非 None 会覆盖所有 term 的设置(observation_manager.py
    _prepare_terms), 那样就无法让 height_scan 单独不带历史。
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """actor 观测: 本体 47 维 x 10 帧 + 当前帧 176 点高度图 = 646 维。"""

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2), history_length=10
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05), history_length=10
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}, history_length=10
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel_biased, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=10
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5), history_length=10
        )
        last_action = ObsTerm(func=mdp.last_action, history_length=10)
        gait_phase = ObsTerm(func=mdp.gait_phase_obs, history_length=10)
        # 176 点局部高度图, 仅当前帧(history_length 默认 0)。
        # noise ±0.05m: actor 要用它做决策, 无噪声会让策略过度依赖精确高度图;
        # 实机的高度图来自 LiDAR/深度建图, 厘米级误差是常态。clip 兜住未命中射线的 inf。
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": SCAN_HEIGHT_OFFSET},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            # 不设 self.history_length(留 None): 否则会覆盖上面各 term 的逐项历史设置
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """critic 特权观测: 本体 62 维 x 10 帧 + 当前帧 176 点高度图 = 796 维。

        critic 的 height_scan 不加噪声(特权观测本就该拿真值降低价值估计方差)。
        """

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, history_length=10)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, history_length=10)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, history_length=10)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}, history_length=10
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, history_length=10)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, history_length=10)
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, history_length=10)
        last_action = ObsTerm(func=mdp.last_action, history_length=10)
        gait_phase = ObsTerm(func=mdp.gait_phase_obs, history_length=10)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": SCAN_HEIGHT_OFFSET},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            # 不设 self.history_length(留 None): 否则会覆盖上面各 term 的逐项历史设置
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """奖励 = flat 的全套 + rough 因地形做的调整(权重已就地写死)。"""

    # -- task
    track_lin_vel_xy = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=1.0)
    track_ang_vel_z = RewTerm(func=mdp.track_ang_vel_z_exp, weight=1.0)
    alive = RewTerm(func=mdp.is_alive, weight=0.15)
    # -- base (rough 调整: base_linear_velocity -2.0 -> -0.5, flat_orientation -4.0 -> -1.0)
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    base_angular = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.15)
    action_acc = RewTerm(func=mdp.action_acc_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)

    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*2", ".*3", ".*6"])},
    )

    # 让踝(pitch .*5 + roll .*6)被动, 脚触地自然贴合
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

    # -- robot (rough 调整: 加 height_scanner 做地形相对高度, 权重 -5 -> -2)
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-2.0,
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

    # rough 调整: 地形相对化 + target 0.1 -> 0.12
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

    # rough 调整: -1.0 -> -0.3
    feet_flat = RewTerm(
        func=mdp.feet_flat,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*6")},
    )

    # 踢立面惩罚: 替掉 flat/rough 用的 feet_stumble(同一摩擦锥判据, 但那个是 torch.any 二值化,
    # 且只读 net_forces_w 最新一帧会漏掉毫秒级撞击脉冲)。这里按撞击功率连续加权 + 读历史窗口。
    feet_kick = RewTerm(
        func=mdp.feet_kick,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*6"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*6"]),
            "cone_ratio": 4.0,
            "saturation": 50.0,
        },
    )

    # 低空拖脚惩罚: 摆动相脚底离地 <6cm 还水平挥就罚, 逼"先抬后挥"(防上台阶踢立面)
    feet_drag = RewTerm(
        func=mdp.feet_drag,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*6"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "height_threshold": 0.06,
        },
    )

    # -- other (rough 调整: -5.0 -> -2.0)
    lateral_move = RewTerm(
        func=mdp.base_lateral_move_l2,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 相对量 base_z - min(feet_z) < 0.2m 即终止(身体塌到脚边), 与地形绝对高度无关
    base_height = DoneTerm(
        func=mdp.base_height_below_feet,
        params={"minimum_height": 0.2, "asset_cfg": SceneEntityCfg("robot", body_names=".*6")},
    )
    # 摔倒(躯干严重倾斜)即终止
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})


@configclass
class CurriculumCfg:
    """命令课程 + 地形课程(走得远升难度, 走不动降难度)。"""

    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)
    ang_vel_cmd_levels = CurrTerm(func=mdp.ang_vel_cmd_levels)
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class RoughInfoEnvCfg(ManagerBasedRLEnvCfg):
    """楼梯加重的 rough 地形 + actor 带地形高度图。"""

    scene: RoughInfoSceneCfg = RoughInfoSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    gait: GaitCfg = GaitCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt


@configclass
class RoughInfoPlayEnvCfg(RoughInfoEnvCfg):
    """play/eval: 少环境 + 少地形块, 便于观察; 关掉观测噪声看策略真实水平。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        # 用 .replace() 拿独立副本, 避免与 train cfg 共享同一个地形对象
        self.scene.terrain.terrain_generator = NLEGS_ROUGH_INFO_TERRAINS_CFG.replace()
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.max_init_terrain_level = 4
