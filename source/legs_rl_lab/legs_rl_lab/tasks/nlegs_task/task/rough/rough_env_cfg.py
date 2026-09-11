"""nlegs rough —— 在 nlegs_flat(平地)基础上换成生成器阶梯地形，并按地形调整奖励/动作/命令。

地形：20% 平地 + 40% 正金字塔台阶 + 40% 反金字塔台阶，单级高固定 0.08m、踏面 0.2m（无难度课程）。
坡面/碎石/方块已移除——目标是专攻上下台阶，且坡面会与"脚掌全程水平"的目标直接冲突。
actor 保持**盲走**（仅 IMU+关节，可直接部署到真机；真机无地形传感器）；
height_scanner 用于 critic 特权观测 height_scan（非对称 actor-critic，降价值估计方差）
和 base_height / feet_clearance / feet_drag 奖励（地形相对高度），不进 actor 观测。

目标步态：0.7 m/s 前进 + 高抬腿（0.18m）+ 脚掌尽可能水平。0.7 这个上限由"一步一级"定出：
  步长 = v × period / 2 要等于 step_width，即 v = 2 × step_width / period = 0.4 / 0.6 = 0.667 m/s。
最硬的约束来自腿长而非踝限位：
  支撑相脚底相对髋要前后扫过 v × period × stance_ratio = 0.7 × 0.6 × 0.5 = 0.21m，
  而 base 高度恒定时该跨度的上界 = 2·√(L² − (h − 0.1296)²)（L = 0.2335 + 0.221 = 0.4545，
  0.1296 = 髋pitch轴下偏移 0.061 + 踝到脚底 0.0686）。要够 0.21m 需 h ≲ 0.572，
  而平地版的 0.58 只给 0.122m —— 这是本任务把 base 目标高度压到 0.55 的唯一理由。
  （0.55 时跨度上界 0.345m、留 64% 余量，摆动腿抬脚上限 0.238m 仍高于 0.18 目标。）
  脚掌水平只作软目标：0.55 处“严格全程平脚”只覆盖 0.164m，策略必然让踝在支撑相两端
  饱和、脚掌倾斜约 2°，feet_flat 会稳定在一个非零地板值上，这是预期行为。

因地形/目标步态调整的项（相对 nlegs_flat）：
  - commands             : 对齐 SSR(arXiv:2605.30770) 附录 A.2 的地形穿越命令设计——
                           lin_vel_x (-0.3,0.5) -> (0.0,0.7)：砍倒走（盲走倒下台阶是学不会的长尾，
                           正/反金字塔各 40% 使前进已同时覆盖上/下台阶）；上限 0.7 而非 1.0，
                           是让一步一级对应的 0.667m/s 落在区间内并靠近上端（见 gait 条目）；
                           lin_vel_y ±0.3 -> ±0.1：收窄侧移（SSR 楼梯上无独立侧向命令）；
                           heading 混合模式：75% 环境 wz 由航向误差生成（直线穿越台阶），
                           25% 保持均匀采样 wz（保住原地转能力，替代 SSR 的毕业阶段）。
                           ranges 与 limit_ranges 保持相等 -> 速度课程空转，从第一步即全范围。
  - gait                 : period 0.8 -> 0.6；stance_ratio 0.55 -> 0.50
                           （0.667m/s 时步长 0.20m 恰等于 step_width，一步一级，落点相位锁定）
  - actions.scale        : 髋pitch/膝 0.25 -> 0.5（clip_actions=5.0 硬钳位下，0.25 在 base 0.55
                           处的抬脚上限只有 0.145m，够不到 0.18 目标；0.5 后上限 0.238m）
  - base_height          : 加 height_scanner 传感器 -> 地形相对高度；权重 -5 -> -2
                           target 0.58 -> 0.55（0.58 处支撑跨度上界仅 0.122m，够不到 0.21m）
  - flat_orientation     : -4.0 -> -1.0   （上台阶时身体自然倾斜，重罚会与攀爬冲突）
  - base_linear_velocity : -2.0 -> -0.5   （爬台阶需要竖直方向速度，别罚太狠）
  - feet_flat            : -1.0 -> -0.6，pitch_scale 0.1 -> 1.0
                           （原有效俯仰系数只有 0.03 等于没管；不上 -1.0 因支撑末期有运动学地板值）
  - feet_clearance       : 加 height_scanner 传感器 -> 脚高地形相对化（否则凸起地形自动
                           满分/凹陷地形恒零分）；target 0.1 -> 0.18（高抬腿风格目标）
  - lateral_move         : -5.0 -> -2.0   （地形上需要一定横向调整来平衡，别压死）
新增：
  - rewards.feet_drag（摆动相低空拖脚惩罚：脚底离地 <8cm 还水平挥就罚，逼"先抬后挥"，
                       针对上台阶时脚尖踢立面的失败模式；同样地形相对化）
  - observations.critic.height_scan（187 点局部高度图，仅训练用；镜像增强见 mdp/symmetry.py；
                       offset 必须含 scanner 的 20m 挂载高度，详见 _apply_rough 内注释）
  - terminations.bad_orientation（摔倒即终止，平地版里没有）
不含地形课程（curriculum.terrain_levels 未注册）——难度直接钉在最大值。
注：flat 的 base_height 终止项已是相对量 base_z - min(feet_z)，与地形无关，这里直接继承。
"""

import math

import isaaclab.terrains as terrain_gen
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

from legs_rl_lab.tasks.nlegs_task import mdp
from legs_rl_lab.tasks.nlegs_task.task.flat.flat_env_cfg import FlatEnvCfg, FlatPlayEnvCfg


# 20% 平地 + 80% 阶梯(正/反各 40%)，台阶高度固定 0.08m（无难度课程，直接最大难度）。
# 注: curriculum=True 保留不是为了课程，而是它走 _add_terrains_by_curriculum 按 proportion 精确切列;
# 改成 False 会走 _add_terrains_by_proportion 按概率随机采样，20/40/40 就只是期望值。
# difficulty_range=(1.0, 1.0) 使每行难度恒为 1.0 -> step_height 全取 step_height_range 上端。
NLEGS_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,  # 仅用于按 proportion 精确切列，难度已被 difficulty_range 钉死
    difficulty_range=(1.0, 1.0),
    sub_terrains={
        # 20% 平地：保底基线，避免纯阶梯上学出畸形步态
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        # 台阶 / 反台阶各 40%：difficulty 恒 1.0 -> 单级高固定 0.08m
        # step_width 0.3 -> 0.2 对齐目标实物台阶(踏面 0.20m / 单级高 0.05m)。与 period 0.6 配对时
        # 一步一级对应 v = 2*0.2/0.6 = 0.667m/s，落点相位与台阶锁定(盲走没有地形感知，步长只由
        # 速度命令×周期决定，不会自适应踏面；踏面与步长不整除就会周期性踩到台阶边缘)。
        # 0.2 已经是下限: 脚的碰撞几何 fromto -0.04~0.14 + 胶囊半径 0.012 -> 脚长 0.204m，
        # 与踏面等长、落点零容错(踝在跟后 0.052/趾前 0.152，踝前可支撑段 0.148m)。
        # 单级高仍留 0.08(> 实际 0.05)不改: 偏保守，训得动更高的台阶只是余量。
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.4,
            step_height_range=(0.03, 0.08),
            step_width=0.2,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.4,
            step_height_range=(0.03, 0.08),
            step_width=0.2,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


def _apply_rough(cfg) -> None:
    """把 nlegs_flat 平地配置改成生成器阶梯地形(固定最大难度) + 按地形调整奖励/动作/终止。"""
    # --- 地形: plane -> generator(阶梯) ---
    # 用 .replace() 拿独立副本，避免 play/train 两个 env cfg 共享并互相改写同一个地形对象
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = NLEGS_ROUGH_TERRAINS_CFG.replace()
    # 无地形课程: 不设 max_init_terrain_level(默认 None = 从 0~num_rows-1 全范围采样)，
    # 因为所有行难度相同，用满全部行能把机器人摊平到更大的地形面积上。
    # 也不注册 curriculum.terrain_levels：难度已统一，升降级只是无意义的位置扰动。

    # --- 速度命令: 对齐 SSR 附录 A.2 的地形穿越命令设计（台阶专精线）---
    # 砍倒走 + 收窄侧移。ranges 与 limit_ranges 必须同步收：lin_vel_cmd_levels 课程会把
    # ranges 向 limit_ranges 双向扩张(每次 ±0.1 后 clamp)，只收 ranges 会被课程慢慢加回来；
    # 两者相等则课程空转，从第一步即全范围。倒走/大侧移能力由 nlegs_flat 线负责。
    # 部署侧导出的是 limit_ranges，故实机/sim2sim 的命令 clamp 自动变为前进 only。
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.7)
    cfg.commands.base_velocity.limit_ranges.lin_vel_x = (0.0, 0.7)
    cfg.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)
    cfg.commands.base_velocity.limit_ranges.lin_vel_y = (-0.1, 0.1)
    # heading 混合模式(一个旋钮近似 SSR 的两阶段命令)：75% 环境 wz = clip(航向误差×0.5,
    # ranges.ang_vel_z=±0.5)——机器人收敛到目标朝向后直线穿越台阶，"步长 0.20m = 台阶宽"
    # 的落点相位锁定才真正成立(独立采样 wz 时走圆弧，大量时间斜切/沿台阶边缘平行走)；
    # 其余 25% 仍均匀采样 wz，保住原地转/大角速度跟踪(SSR 毕业后自由采样阶段的作用)。
    # 观测仍是 [vx,vy,wz] 三维，heading 只改训练期 wz 的生成方式，部署接口与镜像增广不变。
    cfg.commands.base_velocity.heading_command = True
    cfg.commands.base_velocity.rel_heading_envs = 0.75
    cfg.commands.base_velocity.heading_control_stiffness = 0.5
    cfg.commands.base_velocity.ranges.heading = (-math.pi, math.pi)

    # --- critic 特权观测: 187 点局部高度图(非对称 actor-critic, actor 仍盲走) ---
    # height_scan 返回 sensor.pos_w.z - hit_z - offset，而 scanner 虽挂在 base 上但
    # RayCasterCfg.OffsetCfg(pos=(0,0,20.0))，故 pos_w.z = base_z + 20。
    # offset 必须含这 20m 挂载高度: 20 + 0.55(名义 base 高度) = 20.55，这样平地上恒为 0、
    # 台阶处直接给出相对地形起伏。(旧值 0.58 漏了 20m，实际值≈20.0 被 clip 压成恒 1.0，
    # 特权观测等于没有; base_height/feet_clearance/feet_drag 直接用 ray_hits_w 绝对高度，不受影响。)
    # clip 兜住未命中射线的 inf(未命中 -> -inf -> 钳到 -1.0)。
    cfg.observations.critic.height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 20.55},
        clip=(-1.0, 1.0),
    )

    # --- 步态: period 0.8 -> 0.6，stance_ratio 0.55 -> 0.50 ---
    # 一步一级要求步长 = v*period/2 = step_width，即 v = 2*step_width/period = 0.4/period。
    # period 0.6 -> 0.667m/s，落在命令区间 (0,0.7) 内且靠近上端; period 0.8 -> 0.5m/s 偏慢，
    # period 0.4 -> 1.0m/s 但步频 2.5Hz、摆动相只剩 0.2s，膝转速吹顶。
    # 代价: 摆动相只剩 0.6*0.5 = 0.30s，这 0.30s 内脚要抬起再落回、同时相对地面前移
    # v*period = 0.42m(平均水平速度 1.4m/s)。
    # 注: 上 0.05m 台阶时腿的折叠需求远小于总爬升——摆动的 0.30s 内身体自己也升了 0.05m，
    # 脚只需比中间那级的鼻线再高出一点(约 0.025+余量)，所以 0.05m 台阶对膝转速几乎无压力。
    # stance 0.50 是下限，再低会出现双脚同时离地的飞行相(变成跑)。
    cfg.gait.period = 0.6
    cfg.gait.stance_ratio = 0.50

    # --- 动作缩放: 髋pitch(.*1)/膝(.*4) 0.25 -> 0.5，其余保持 0.25 ---
    # policy_action_clip=5.0 在训练(vecenv_wrapper)和部署(sim2sim)都硬钳位，故 scale 直接决定
    # 可达关节角: 默认姿态 θ1=-0.1/θ4=0.2，scale 0.25 -> 髋pitch 最多 -1.35、膝最多 1.45，
    # base 0.55 下抬脚硬上限只有 0.145m(0.58 下也只有 0.175m)，够不到 0.18 目标且动作永久
    # 贴 -5 无平衡余量; scale 0.5 后膝能到软限位 1.872，上限 0.238m，限制转移到力矩/摆动时间。
    cfg.actions.JointPositionAction.scale = {".*[14]": 0.5, ".*[2356]": 0.25}

    # --- 因地形调整的奖励 ---
    # base_height 用高度扫描做地形相对(否则起伏地形上 base 世界系绝对高度恒被罚)
    cfg.rewards.base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")
    cfg.rewards.base_height.weight = -2.0
    # target 0.58 -> 0.55: base 高度恒定时，脚底相对髋的前后可达跨度上界 = 2·√(L²-(h-0.1296)²)，
    # L=0.4545(大腿 0.2335 + 小腿 0.221)，0.1296 = 髋pitch轴下偏移 0.061 + 踝到脚底 0.0686。
    # 该上界(脚掌保持水平): 0.58->0.122m、0.565->0.261m、0.56->0.292m、0.55->0.345m、0.54->0.391m。
    # 0.7m/s 支撑相需 0.21m，故 h 必须 ≲0.572；取 0.55 留 64% 余量，且抬脚上限仍有 0.238m。
    # 注: 这是腿长约束，与踝限位无关——0.57 以上放宽踝限位也救不回来。
    cfg.rewards.base_height.params["target_height"] = 0.55
    cfg.rewards.flat_orientation.weight = -1.0
    cfg.rewards.base_linear_velocity.weight = -0.5
    # 脚掌水平: 坡面地形已删，平地与台阶顶面都水平，故不再与地形冲突。
    # pitch_scale 0.1 -> 1.0(原来 0.1 × 权重 0.3 = 有效系数 0.03，等于没管俯仰);
    # 权重 -0.3 -> -0.6 而非 -1.0: base 0.55 处“严格全程平脚”只覆盖 0.164m < 需要的 0.21m，
    # 策略必然让踝在支撑相两端撞 ±0.38 软限位、脚掌倾斜约 2°，即俯仰项有降不到 0 的地板值;
    # 权重过大只会让它靠缩短步长来买水平度，反而拖累速度跟踪与抬脚。
    cfg.rewards.feet_flat.weight = -0.6
    cfg.rewards.feet_flat.params["pitch_scale"] = 1.0
    # feet_clearance 同样地形相对化(每只脚取扫描点中水平最近命中点作脚下地面高度)
    cfg.rewards.feet_clearance.params["sensor_cfg"] = SceneEntityCfg("height_scanner")
    # 0.12 -> 0.18: 台阶最高只有 8cm，这里是高抬腿风格目标而非通行需求
    cfg.rewards.feet_clearance.params["target_height"] = 0.18
    cfg.rewards.lateral_move.weight = -2.0

    # --- 低空拖脚惩罚: 摆动相脚底离地 <8cm 还水平挥就罚, 逼"先抬后挥"(防上台阶踢立面) ---
    # 阈值 8cm 取训练地形的单级高(实物台阶只 5cm，偏保守): 脚必须先抬过台阶顶面再往前挥。
    # weight 保持 -1.0 而不上调: 该项罚的是**世界系**脚速，命令速度越高惩罚天然越大
    # (0.7m/s 下摆动相 0.30s 内脚要相对地面前进 v*period = 0.42m，平均 1.4m/s，是命令
    # 速度的 2 倍)，再加权等于给高速命令叠税。
    cfg.rewards.feet_drag = RewTerm(
        func=mdp.feet_drag,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*6"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "height_threshold": 0.08,
        },
    )

    # --- 终止: 摔倒(躯干严重倾斜)即终止 ---
    cfg.terminations.bad_orientation = DoneTerm(
        func=mdp.bad_orientation, params={"limit_angle": 1.0}
    )


@configclass
class RoughEnvCfg(FlatEnvCfg):
    """nlegs_flat + rough 地形。"""

    def __post_init__(self):
        super().__post_init__()
        _apply_rough(self)


@configclass
class RoughPlayEnvCfg(FlatPlayEnvCfg):
    """play/eval：继承 flat play(少环境)，同样切 rough 地形，但地形块更少便于可视化。"""

    def __post_init__(self):
        super().__post_init__()
        _apply_rough(self)
        # play: 少量地形行列便于可视化(难度已统一为 0.08m 台阶，无需限制初始等级)
        # 注: 5 列下比例仍精确为 1/2/2 = 20%/40%/40%
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
