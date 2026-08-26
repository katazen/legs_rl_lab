"""nlegs_rough —— 在 nlegs(平地)基础上换成生成器 rough 地形 + 地形课程，并按地形调整奖励/终止。

地形做法参照 g1 的 velocity_env_cfg（terrain_type="generator" + terrain_levels 课程 + max_init_terrain_level），
但子地形按 0.58m 小机身**温和缩放**，不照搬 g1/IsaacLab 默认的 0.23m 台阶、0.4 坡度：
  台阶 ≤0.12m、坡 ≤0.3、方块 0.02~0.08m、随机起伏 0.02~0.06m，并保留 20% 平地。
策略保持**盲走**（仅 IMU+关节，可直接部署到真机；真机无地形传感器），
height_scanner 只用于 base_height 奖励（地形相对高度）和地形课程 terrain_levels_vel。

因地形改动的奖励（相对 nlegs 平地版）：
  - base_height          : 加 height_scanner 传感器 -> 地形相对高度；权重 -5 -> -2
                           （否则起伏地形上 base 的世界系绝对高度会被恒定惩罚）
  - flat_orientation     : -4.0 -> -1.0   （上坡时身体自然倾斜，重罚会与爬坡冲突）
  - base_linear_velocity : -2.0 -> -0.5   （爬台阶/坡需要竖直方向速度，别罚太狠）
  - feet_flat            : -1.0 -> -0.3   （坡/台阶上脚无法始终贴平地面）
  - feet_clearance       : target 0.1 -> 0.12（抬高摆动腿以跨越起伏/矮台阶）
  - lateral_move         : -5.0 -> -2.0   （地形上需要一定横向调整来平衡，别压死）
新增：
  - curriculum.terrain_levels  （走得远 -> 升难度；走不动 -> 降难度）
  - terminations.bad_orientation（摔倒即终止，平地版里是注释掉的）
"""

import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

from legs_rl_lab.tasks.legs_task import mdp
from legs_rl_lab.tasks.legs_task.task.nlegs.nlegs_env_cfg import (
    RobotEnvCfg as NlegsEnvCfg,
    RobotPlayEnvCfg as NlegsPlayEnvCfg,
)


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


def _apply_rough(cfg) -> None:
    """把 nlegs 平地配置改成生成器 rough 地形 + 地形课程 + 因地形调整奖励/终止。"""
    # --- 地形: plane -> generator(rough) ---
    # 用 .replace() 拿独立副本，避免 play/train 两个 env cfg 共享并互相改写同一个地形对象
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = NLEGS_ROUGH_TERRAINS_CFG.replace()
    cfg.scene.terrain.terrain_generator.curriculum = True
    cfg.scene.terrain.max_init_terrain_level = 5  # 初始铺在 0~5 级，course 再上下调

    # --- 地形课程: 走得远升难度、走不动降难度 ---
    cfg.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

    # --- 因地形调整的奖励 ---
    # base_height 用高度扫描做地形相对(否则起伏地形上 base 世界系绝对高度恒被罚)
    cfg.rewards.base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")
    cfg.rewards.base_height.weight = -2.0
    cfg.rewards.flat_orientation.weight = -1.0
    cfg.rewards.base_linear_velocity.weight = -0.5
    cfg.rewards.feet_flat.weight = -0.3
    cfg.rewards.feet_clearance.params["target_height"] = 0.12
    # lateral_move 是 nlegs 训练版 _apply_nlegs 动态加的；play 用 base legs 奖励集里没有，需守卫
    if getattr(cfg.rewards, "lateral_move", None) is not None:
        cfg.rewards.lateral_move.weight = -2.0

    # --- 终止: 摔倒(躯干严重倾斜)即终止 ---
    cfg.terminations.bad_orientation = DoneTerm(
        func=mdp.bad_orientation, params={"limit_angle": 1.0}
    )


@configclass
class RobotRoughEnvCfg(NlegsEnvCfg):
    """nlegs + rough 地形。"""

    def __post_init__(self):
        super().__post_init__()
        _apply_rough(self)


@configclass
class RobotRoughPlayEnvCfg(NlegsPlayEnvCfg):
    """play/eval：继承 nlegs play(少环境、命令放开)，同样切 rough 地形，但地形块更少便于可视化。"""

    def __post_init__(self):
        super().__post_init__()
        _apply_rough(self)
        # play: 少量地形行列 + 固定较低初始难度，便于观察
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.max_init_terrain_level = 4
