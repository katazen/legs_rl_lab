"""nlegs_rough_step: **专用上台阶**任务, 全量展开配置(不继承 flat/rough 任何 env cfg)。

目标很窄: 让实机盲走上一部 5cm 高 / 20cm 深 / 4 级 的楼梯。因此有意放弃通用性,
换取该单一任务上的样本效率。与 nlegs_rough 的三处本质区别:

1) 地形: 只有"反金字塔台阶(坑)" + 少量平地, 删掉坡/方块/随机起伏。
   - 反金字塔(坑)的生成原点在**坑底中心**(见 isaaclab mesh_terrains.inverted_pyramid_stairs_terrain,
     origin z = -(num_steps+1)*step_height), 四个方向全是**上行**台阶 -> 无论朝哪走都在上楼,
     天然解决"必须先朝向楼梯"的问题。(正金字塔原点在塔顶, 只能练下楼, 故只留 10% 作下楼样本。)
   - 台阶级数 = (size - 2*border_width - platform_width) // (2*step_width) + 1, 三个 step_width
     变体各自配 platform_width, 都精确落在 **4 级**(与实机一致)。
   - step_height_range=(0.04, 0.06) 且生成器 curriculum=False -> difficulty 每块随机采样,
     即台阶高度在 4~6cm 均匀随机(实机 5cm 落在中间)。step_width 开 0.18/0.20/0.24 三档。
     这是 Cassie(RSS 2021, arXiv:2105.08328)的"每级加 ±1cm 噪声"思想: 防止策略靠本体感知
     反推出台阶精确尺寸后过拟合成"数节拍", 实机差几毫米就崩。**这条随机化不能省。**
   - 无地形课程(terrain_levels): Cassie 原文结论是随机楼梯 + 无课程即可; 且 4 级 5cm 总高
     仅 20cm, "走够半个地块才升级"的判据不成立。

2) 命令: 退化为"恒定向前 + 朝向纠偏", 不再全向随机。
   - lin_vel_x=(0.3, 0.5)(给范围而非单点: Cassie 实测存在最优接近速度, 太慢没动量、太快步态
     太动态都会失败; 实机也留调速余量), lin_vel_y=(0, 0)。
   - heading_command=True + rel_heading_envs=1.0 + ranges.heading=(0, 0): ang_vel_z 由朝向误差
     自动算出(世界系 +x 为目标朝向), 实机放歪了会自己纠回来。
     **注意 ang_vel_z 此时是纠偏量的 clip 上下界(isaaclab velocity_command.py), 必须留 ±0.5,
     置 (0,0) 会把纠偏夹成 0。**
   - rel_standing_envs=0.0: 一启动就走, 不要站立样本。
   - 无命令课程(lin_vel/ang_vel_cmd_levels): 命令已固定, 课程无意义。
   - reset 的 yaw 收窄到 ±0.4rad: 坑里空间有限, 别让它先原地转 180°。

3) critic 的 height_scan **只用当前帧**(其余本体项仍保留 10 帧历史)。
   地形高度图的历史价值低(地形本身不动, 历史信息基本可由本体历史+当前地形推出), 而 187 维 x 10
   帧会把 critic 输入撑到 2490 维。改法: group 级 history_length 留 None(它会覆盖 term 级设置),
   改为逐 term 声明 history_length=10, height_scan 不声明(=0)。critic 维度 2490 -> 807。
   对应 mdp/symmetry.py 的镜像布局已按"逐项历史"扩展。

actor 仍是**盲走**(仅 IMU+关节+步态时钟, 可直接部署实机); height_scanner 只服务 critic 特权
观测与 base_height / feet_clearance / feet_drag 的地形相对化。

奖励沿用 nlegs_rough 的一套(权重已就地写死), 含 feet_drag 低空拖脚惩罚; feet_clearance 暂时
沿用现有 exp 形式(target 0.12), 以便单独验证"专用地形"这一个变量的效果。
"""

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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


@configclass
class GaitCfg:
    """步态时钟参数, 会被 dump 到 env.yaml; reward / observation 通过 env.cfg.gait.* 读取。"""

    period: float = 0.7             # 步态周期 (s), 与 nlegs_rough 一致
    stance_ratio: float = 0.55      # 支撑相占比
    feet_offset: list = [0.0, 0.5]  # 左右腿相位偏移


# 专用上台阶地形: 反金字塔台阶(坑, 四面上行) 为主 + 少量平地 + 少量下楼。
# num_rows/num_cols 只是地块变体数量(curriculum=False -> 行不代表难度, difficulty 每块随机)。
# 级数校验(size 5.0 - 2*0.5 = 4.0 可用跨度):
#   step_width 0.18, platform 2.7 -> (4.0-2.7)//0.36 + 1 = 4 级, 起步到第一级 1.35m
#   step_width 0.20, platform 2.6 -> (4.0-2.6)//0.40 + 1 = 4 级, 起步到第一级 1.30m
#   step_width 0.24, platform 2.4 -> (4.0-2.4)//0.48 + 1 = 4 级, 起步到第一级 1.20m
# 起步点在坑底平台中心, 叠加 reset_base 的 x/y ±0.5m -> 到第一级的距离天然随机 0.7~1.9m。
NLEGS_ROUGH_STEP_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(5.0, 5.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,  # 无地形课程: difficulty 每块随机 -> 台阶高度 4~6cm 均匀采样
    sub_terrains={
        # 15% 平地: 保住平地步态, 别让策略彻底忘了怎么在平地走
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.15),
        # 上行台阶(坑) 三种踏面深度, 共 75%
        "stairs_up_w18": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.30,
            step_height_range=(0.04, 0.06),
            step_width=0.18,
            platform_width=2.7,
            border_width=0.5,
            holes=False,
        ),
        "stairs_up_w20": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.30,
            step_height_range=(0.04, 0.06),
            step_width=0.20,
            platform_width=2.6,
            border_width=0.5,
            holes=False,
        ),
        "stairs_up_w24": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.04, 0.06),
            step_width=0.24,
            platform_width=2.4,
            border_width=0.5,
            holes=False,
        ),
        # 10% 下楼(正金字塔, 原点在塔顶): 上下同训, Cassie 亦如此; 实机上楼后总要下来
        "stairs_down_w20": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.04, 0.06),
            step_width=0.20,
            platform_width=2.6,
            border_width=0.5,
            holes=False,
        ),
    },
)


@configclass
class RoughStepSceneCfg(InteractiveSceneCfg):
    """场景: 窄本体机器人 + 专用台阶地形。"""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=NLEGS_ROUGH_STEP_TERRAINS_CFG,
        # curriculum=False 时行不代表难度, None -> 环境铺满所有行列(各种难度均匀混合)
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
    # robots (踝执行器拆组等驱动器参数已在资产 NLEGS_CFG 里定义)
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
    """事件项(与 nlegs_flat/rough 同 7 项; 仅 reset_base 的 yaw 收窄)。"""

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

    # reset
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (1.0, 4.0),
            "operation": "add",
        },
    )

    # x/y ±0.5m 提供"到第一级台阶的距离"随机化(坑底平台半宽 1.2~1.35m);
    # yaw 收窄到 ±0.4rad(≈±23°): 起步大致朝向台阶, 剩下的偏差由 heading 命令纠偏,
    # 避免在坑底先原地转半圈(既浪费 episode 也学不到上台阶)。
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.4, 0.4)},
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
    # 标定误差, 让策略对恒定零偏鲁棒 -> 消除实机零速漂移。仅作用于策略观测的 joint_pos_rel。
    joint_zero_bias = EventTerm(
        func=mdp.randomize_joint_zero_bias,
        mode="reset",
        params={"bias_range": (-0.05, 0.05)},
    )

    # 踝(pitch .*5)专属带宽随机化(48V 辨识后收窄: kp 略偏下 + kd 略偏高)。
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

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class CommandsCfg:
    """命令: 恒定向前 + 朝向纠偏(不再全向随机采样)。"""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,   # 一启动就走, 无站立样本
        rel_heading_envs=1.0,    # 全部环境用朝向命令
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        # heading=(0,0): 目标朝向锁死世界系 +x(所有地块轴对齐, +x 恒指向一面台阶)。
        # ang_vel_z 在 heading_command 下是**纠偏量的 clip 界**, 必须留 ±0.5(置 0 会夹死纠偏)。
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.3, 0.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.5, 0.5), heading=(0.0, 0.0)
        ),
        # 无命令课程, limit_ranges 与 ranges 保持一致(该字段是 MISSING 必填)
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.3, 0.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.5, 0.5), heading=(0.0, 0.0)
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
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """actor 观测: 盲走(仅 IMU + 关节 + 命令 + 步态时钟), 10 帧历史。"""

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
        """critic 特权观测。

        **逐 term 声明 history_length**, group 级刻意留 None —— ObservationManager 里
        group.history_length 若非 None 会覆盖所有 term 的设置(observation_manager.py
        _prepare_terms), 那样就无法让 height_scan 单独不带历史。
        本体项 10 帧, height_scan 只用当前帧(地形不动, 历史价值低, 且省下 187*9=1683 维)。
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
        # 187 点局部高度图, 仅当前帧(history_length 默认 0); offset=0.58(名义 base 高度)使数值
        # 围绕 0; clip 兜住未命中射线的 inf。镜像布局见 mdp/symmetry.py
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.58},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            # 不设 self.history_length(留 None): 否则会覆盖上面各 term 的逐项历史设置
            self.concatenate_terms = True

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """奖励项(沿用 nlegs_rough 的一套, 地形相关权重已就地写死)。"""

    # -- task
    track_lin_vel_xy = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=1.0)
    track_ang_vel_z = RewTerm(func=mdp.track_ang_vel_z_exp, weight=1.0)
    alive = RewTerm(func=mdp.is_alive, weight=0.15)
    # -- base (台阶上需要竖直速度/身体倾斜, 故这几项比平地放松)
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

    # 让踝(pitch .*5 + roll .*6)被动，脚触地自然贴合，抑制踝滚转侧崴
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

    # -- robot (地形相对高度: 否则台阶上 base 的世界系绝对高度会被恒定惩罚)
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

    # 脚高地形相对化(每只脚取扫描点中水平最近命中点作脚下地面高度), target 0.12 抬高摆动腿
    feet_clearance = RewTerm(
        func=mdp.feet_clearance,
        weight=1.0,
        params={
            "target_height": 0.12,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*6"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
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
    # 无 y 速度命令时抑制身体左右平移; 台阶上需要一定横向调整来平衡, 故比平地放松
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
    """无课程项。

    - 地形课程(terrain_levels): 生成器 curriculum=False, 难度每块随机(Cassie 结论: 随机楼梯
      无需课程), 且 4 级 x 5cm 总高仅 20cm, "走够半个地块才升级"的判据不成立。
    - 命令课程(lin_vel/ang_vel_cmd_levels): 命令已固定为恒定前进 + 朝向纠偏, 无可扩展区间。
    """


@configclass
class RoughStepEnvCfg(ManagerBasedRLEnvCfg):
    """窄本体专用上台阶环境(全量展开版, 不继承 flat/rough)。"""

    # Scene settings
    scene: RoughStepSceneCfg = RoughStepSceneCfg(num_envs=4096, env_spacing=2.5)
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

        # 取地形生成器的独立副本, 避免 play/train 两个 env cfg 共享并互相改写同一个对象
        self.scene.terrain.terrain_generator = NLEGS_ROUGH_STEP_TERRAINS_CFG.replace()


@configclass
class RoughStepPlayEnvCfg(RoughStepEnvCfg):
    """play/eval: 少环境 + 少地形块, 便于观察上台阶行为。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
