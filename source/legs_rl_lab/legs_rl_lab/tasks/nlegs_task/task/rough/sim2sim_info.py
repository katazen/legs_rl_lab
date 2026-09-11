"""nlegs_rough_info 的 MuJoCo sim2sim 回放器 —— 复用 flat 版全部逻辑, 只加地形与高度图观测。

用法(按文件路径直接运行, 不要 python -m 走包导入):
    python source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/task/rough/sim2sim_info.py \
        --run RUN [--headless] [--duration S] [--save-data] [--no-scan-marker]

与 rough/sim2sim.py 的唯一实质区别: **actor 的观测里多一项 176 点局部高度图**, 而 flat 版的
观测流水线只认本体项(_OBS_FEATURES 白名单里没有 height_scan)。本文件在 MuJoCo 里复刻
Isaac Lab RayCaster + mdp.height_scan 的全过程:

  1) 网格点(sensor 局部系): isaaclab patterns.grid_pattern, ordering="xy"
         x = arange(-size_x/2, +size_x/2, resolution)   -> nx 点
         y = arange(-size_y/2, +size_y/2, resolution)   -> ny 点
         meshgrid(indexing="xy") 后按行主序展平 => idx = iy * nx + ix
     再整体加上 cfg.offset.pos(含把网格前移的 x 分量, 以及 z=20 的射线起点抬升)。
  2) 射线起点(世界系): ray_alignment="yaw" -> 局部点先绕 z 转 base 的 yaw, 再加 base 世界位置
         ray_starts_w = quat_apply_yaw(quat_w, ray_starts) + pos_w
     射线方向恒为世界 -z(yaw 模式下方向不旋转)。
  3) 命中: MuJoCo mj_ray。只对 geom group 0 求交 —— nlegs.xml 里机器人的可视 geom 是
     group 1、碰撞 geom 是 group 3, 地面与本文件加的地形 box 都是 group 0, 所以这个掩码
     天然把机器人自身滤掉了(否则 20m 高处往下打的射线会先命中机器人)。
  4) 数值: height_scan = pos_w.z - hit.z - offset(0.58), 与 isaaclab mdp.height_scan 一致。
     未命中(mj_ray 返回 -1)时给 +inf, 由观测项自带的 clip=(-1,1) 兜住 —— 与 Isaac 侧
     raycast 打空返回 inf 后被同一个 clip 截断的行为对齐。

网格几何(resolution / size / offset)从 deploy.yaml 的 height_scanner 段读, 由导出器写入,
不需要手工同步; 旧 run 没有这一段时回退到下方 FALLBACK_SCANNER 并打印提示。读到之后会用
"nx*ny == height_scan 观测项的维度"做一次交叉校验, 几何写错会当场报错而不是静默跑偏。

地形参数与 rough/sim2sim.py **逐字一致**(楼梯 4 级 x 5cm/20cm + 10° 斜坡), 这样同一套地形
上可以直接对比"盲走 rough"与"非盲走 rough_info"。键盘与可视化也沿用 flat 版(小键盘
8/2 4/6 7/9 调速), 因为本任务保留了完整的速度跟踪命令。
额外的可视化: viewer 里把 176 个采样点的命中位置画成小球, 颜色按 height_scan 数值从蓝(低)
到红(高), 用来肉眼确认高度图对齐没错位。
"""

import argparse
import importlib.util
import os

import mujoco
import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_flat_sim2sim():
    """按文件路径加载 flat/sim2sim.py（包导入会拉起 Isaac Lab，故走 importlib）。"""
    path = os.path.abspath(os.path.join(_HERE, "..", "flat", "sim2sim.py"))
    spec = importlib.util.spec_from_file_location("nlegs_flat_sim2sim", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


flat = _load_flat_sim2sim()
# load_config 会拒绝它不认识的观测项。height_scan 的数值由本文件自己算(父类的 features 里
# 没有它, 见 RoughInfoRunner._observation_terms), 这里只是把它登记进白名单让校验放行。
flat._OBS_FEATURES["height_scan"] = "height_scan"

# ===================== 需要自己填/改的部分（全部集中在这里） =====================
# 要回放的训练 run（logs/rsl_rl/nlegs_rough_info/ 下的目录名，需先用 play 导出 exported/policy.pt）
RUN = ""
# run 所在的 logs 根目录
LOGS_ROOT = os.path.join(flat._REPO_ROOT, "logs", "rsl_rl", "nlegs_rough_info")

# --- 楼梯（沿 +X 行进, 宽度沿 Y 居中）: 与 rough/sim2sim.py 一致 ---
STAIR_START = 2.0        # 近端边缘距原点(m)
STAIR_WIDTH = 2.0        # 宽(m)
STAIR_STEP_H = 0.05      # 每级高(m)
STAIR_STEP_D = 0.2       # 每级深(m)
STAIR_NUM_STEPS = 4      # 上行级数(第 NUM 级踏面 = 顶部平台)
STAIR_PLATFORM_D = 0.4   # 顶部平台深(m), 高 = NUM_STEPS * STEP_H
STAIR_RGBA = (0.7, 0.7, 0.7, 1.0)

# --- 斜坡（沿 +Y 行进, 宽度沿 X 居中）: 与 rough/sim2sim.py 一致 ---
RAMP_START = 2.0         # 近端边缘距原点(m)
RAMP_WIDTH = 2.0         # 宽(m)
RAMP_SLOPE_DEG = 10.0    # 坡度(°)
RAMP_RUN = 2.0           # 单边坡的水平投影长度(m), 顶高 = RUN * tan(坡度)
RAMP_PLATFORM_D = 0.4    # 顶部平台深(m)
RAMP_THICKNESS = 0.01    # 坡面板厚(m), 只影响外观/底部
RAMP_RGBA = (0.55, 0.5, 0.45, 1.0)

# --- 高度图 ---
# deploy.yaml 没有 height_scanner 段(旧导出器)时的回退值, 必须与 rough_info_env_cfg.py 一致
FALLBACK_SCANNER = {
    "offset": (0.25, 0.0, 20.0),
    "resolution": 0.1,
    "size": (1.5, 1.0),
    "ordering": "xy",
    "ray_alignment": "yaw",
}
SCAN_MARKER = True        # viewer 里画出采样点命中位置
SCAN_MARKER_RADIUS = 0.015
SCAN_MARKER_RANGE = 0.2   # 着色饱和的 height_scan 绝对值(m): 蓝 <= -RANGE, 红 >= +RANGE
# ==============================================================================

# flat.load_config 读取模块全局 LOGS_ROOT
flat.LOGS_ROOT = LOGS_ROOT

# mj_ray 的 geom group 掩码: 只留 group 0(地面 + 本文件加的地形)。
# nlegs.xml: 可视 geom group=1, 碰撞 geom group=3, 所以机器人自身不会被射线命中。
_TERRAIN_GROUPS = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)


def _add_box(spec, name, pos, size, rgba, quat=(1.0, 0.0, 0.0, 0.0)):
    spec.worldbody.add_geom(
        name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=list(pos), size=list(size), quat=list(quat), rgba=list(rgba),
    )


def _add_stairs(spec):
    """+X 方向: 上行踏面 1..N-1 -> 顶部平台(第 N 级) -> 下行踏面 N-1..1。"""
    half_w = STAIR_WIDTH / 2
    x = STAIR_START
    for i in range(1, STAIR_NUM_STEPS):
        h = i * STAIR_STEP_H
        _add_box(spec, f"stair_up_{i}", (x + STAIR_STEP_D / 2, 0.0, h / 2),
                 (STAIR_STEP_D / 2, half_w, h / 2), STAIR_RGBA)
        x += STAIR_STEP_D
    h_top = STAIR_NUM_STEPS * STAIR_STEP_H
    _add_box(spec, "stair_platform", (x + STAIR_PLATFORM_D / 2, 0.0, h_top / 2),
             (STAIR_PLATFORM_D / 2, half_w, h_top / 2), STAIR_RGBA)
    x += STAIR_PLATFORM_D
    for i in range(STAIR_NUM_STEPS - 1, 0, -1):
        h = i * STAIR_STEP_H
        _add_box(spec, f"stair_down_{i}", (x + STAIR_STEP_D / 2, 0.0, h / 2),
                 (STAIR_STEP_D / 2, half_w, h / 2), STAIR_RGBA)
        x += STAIR_STEP_D


def _add_ramp(spec):
    """+Y 方向: 上坡 -> 顶部平台 -> 下坡。坡面用绕 X 轴旋转的板状 box。"""
    angle = np.deg2rad(RAMP_SLOPE_DEG)
    rise = RAMP_RUN * np.tan(angle)
    slope_len = RAMP_RUN / np.cos(angle)
    half_w, half_t = RAMP_WIDTH / 2, RAMP_THICKNESS / 2
    half_angle = angle / 2

    top_mid_y = RAMP_START + RAMP_RUN / 2
    _add_box(spec, "ramp_up",
             (0.0, top_mid_y + half_t * np.sin(angle), rise / 2 - half_t * np.cos(angle)),
             (half_w, slope_len / 2, half_t), RAMP_RGBA,
             quat=(np.cos(half_angle), np.sin(half_angle), 0.0, 0.0))
    plat_y = RAMP_START + RAMP_RUN + RAMP_PLATFORM_D / 2
    _add_box(spec, "ramp_platform", (0.0, plat_y, rise / 2),
             (half_w, RAMP_PLATFORM_D / 2, rise / 2), RAMP_RGBA)
    top_mid_y = RAMP_START + RAMP_RUN + RAMP_PLATFORM_D + RAMP_RUN / 2
    _add_box(spec, "ramp_down",
             (0.0, top_mid_y - half_t * np.sin(angle), rise / 2 - half_t * np.cos(angle)),
             (half_w, slope_len / 2, half_t), RAMP_RGBA,
             quat=(np.cos(half_angle), -np.sin(half_angle), 0.0, 0.0))


def add_terrain(spec):
    _add_stairs(spec)
    _add_ramp(spec)


def read_scanner_cfg(run_dir):
    """从 deploy.yaml 读高度图网格几何；旧 run 缺该段时回退到 FALLBACK_SCANNER。"""
    with open(os.path.join(run_dir, "params", "deploy.yaml")) as file:
        deploy = yaml.safe_load(file)
    scanner = deploy.get("height_scanner")
    if scanner is None:
        print("[sim2sim] deploy.yaml 缺 height_scanner 段(旧导出器)，回退到脚本内的 "
              f"FALLBACK_SCANNER={FALLBACK_SCANNER}")
        scanner = dict(FALLBACK_SCANNER)
    if scanner.get("ordering", "xy") != "xy":
        raise ValueError(f"本脚本只实现了 ordering='xy'，收到 {scanner.get('ordering')}")
    if scanner.get("ray_alignment", "yaw") != "yaw":
        raise ValueError(f"本脚本只实现了 ray_alignment='yaw'，收到 {scanner.get('ray_alignment')}")
    return scanner


def grid_ray_starts(scanner):
    """复刻 isaaclab patterns.grid_pattern(ordering='xy') + cfg.offset.pos。

    返回 (starts[N,3] 在 sensor 局部系, nx, ny)；展平序 idx = iy * nx + ix。
    """
    resolution = float(scanner["resolution"])
    size_x, size_y = (float(v) for v in scanner["size"])
    # arange 的上界加 1e-9 与 isaaclab 一致(保证端点被取到)
    x = np.arange(-size_x / 2, size_x / 2 + 1.0e-9, resolution)
    y = np.arange(-size_y / 2, size_y / 2 + 1.0e-9, resolution)
    grid_x, grid_y = np.meshgrid(x, y, indexing="xy")   # 形状 (ny, nx)
    starts = np.zeros((grid_x.size, 3), dtype=np.float64)
    starts[:, 0] = grid_x.reshape(-1)
    starts[:, 1] = grid_y.reshape(-1)
    starts += np.asarray(scanner["offset"], dtype=np.float64)
    return starts, x.size, y.size


class RoughInfoRunner(flat.MujocoRunner):
    """flat 的 MujocoRunner + 楼梯/斜坡地形 + MuJoCo 版 height_scan 观测。"""

    def __init__(self, cfg, scanner, show_viewer=True, save_data=False, scan_marker=SCAN_MARKER):
        # 这几项要在 super().__init__ 之前准备好: 它末尾会调 _update_markers
        self._scan_starts, self._scan_nx, self._scan_ny = grid_ray_starts(scanner)
        self._scan_offset = float(cfg.observations["height_scan"]["params"].get("offset", 0.5))
        self._scan_hits = None
        self._scan_values = None
        self._scan_marker = scan_marker
        self._ray_geomid = np.zeros(1, dtype=np.int32)
        super().__init__(cfg, show_viewer=show_viewer, save_data=save_data)

        # 把 height_scan 从"父类逐项处理"的表里摘出来: 父类的 features 里没有这一项。
        # TermGroupedHistory 已在 super().__init__ 里按含它的顺序建好 buffer(它排在最后),
        # 所以摘掉之后 _observation_terms 再补上, 拼接顺序不变。
        self._scan_cfg = self.cfg.observations.pop("height_scan")
        num_rays = self._scan_nx * self._scan_ny
        if len(self._scan_cfg["scale"]) != num_rays:
            raise ValueError(
                f"高度图几何与训练不符: 网格 {self._scan_nx}x{self._scan_ny}={num_rays} 点, "
                f"但 deploy.yaml 的 height_scan 是 {len(self._scan_cfg['scale'])} 维"
            )
        if self._scan_cfg["history_length"] != 1:
            raise ValueError(
                f"本脚本只支持单帧高度图，deploy.yaml 里 history_length="
                f"{self._scan_cfg['history_length']}"
            )
        print(f"[sim2sim] 高度图: {self._scan_nx}x{self._scan_ny}={num_rays} 点, "
              f"分辨率 {scanner['resolution']}m, 机身系 x∈"
              f"[{self._scan_starts[:, 0].min():+.2f},{self._scan_starts[:, 0].max():+.2f}] "
              f"y∈[{self._scan_starts[:, 1].min():+.2f},{self._scan_starts[:, 1].max():+.2f}], "
              f"高度基准 {self._scan_offset}m")

    def _make_model(self, cfg):
        spec = mujoco.MjSpec.from_file(cfg.scene_xml)
        add_terrain(spec)
        rise = RAMP_RUN * np.tan(np.deg2rad(RAMP_SLOPE_DEG))
        print(f"[sim2sim] 地形: 楼梯 +X {STAIR_START}m 起(顶高 {STAIR_NUM_STEPS * STAIR_STEP_H:.2f}m), "
              f"斜坡 +Y {RAMP_START}m 起(顶高 {rise:.3f}m)")
        return spec.compile()

    # ------------------------------ 高度图 ------------------------------

    def _height_scan(self):
        """MuJoCo 版 mdp.height_scan: base_z - hit_z - offset, 未命中给 +inf。"""
        base_pos = self.data.xpos[self.base_body_id]
        quat = self.data.qpos[self.base_qpos_adr + 3:self.base_qpos_adr + 7]
        yaw = flat.yaw_from_quat(quat)   # 与 isaaclab yaw_quat 同一个 yaw 提取式
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        local = self._scan_starts
        starts = np.empty_like(local)
        starts[:, 0] = cos_y * local[:, 0] - sin_y * local[:, 1] + base_pos[0]
        starts[:, 1] = sin_y * local[:, 0] + cos_y * local[:, 1] + base_pos[1]
        starts[:, 2] = local[:, 2] + base_pos[2]

        direction = np.array([0.0, 0.0, -1.0])   # yaw 模式下方向不随机身旋转
        hits = starts.copy()
        for index in range(starts.shape[0]):
            distance = mujoco.mj_ray(
                self.model, self.data, starts[index], direction,
                _TERRAIN_GROUPS, 1, -1, self._ray_geomid,
            )
            hits[index, 2] = starts[index, 2] - distance if distance >= 0.0 else -np.inf
        self._scan_hits = hits
        self._scan_values = (base_pos[2] - hits[:, 2] - self._scan_offset).astype(np.float32)
        return self._scan_values

    def _observation_terms(self):
        terms = super()._observation_terms()
        value = self._height_scan()
        if self._scan_cfg.get("clip") is not None:
            value = np.clip(value, *self._scan_cfg["clip"])
        terms["height_scan"] = value * np.asarray(self._scan_cfg["scale"], dtype=np.float32)
        return terms

    def _update_markers(self):
        super()._update_markers()
        if not self._scan_marker or self._scan_hits is None:
            return
        scene = self.viewer.user_scn
        clip = self._scan_cfg.get("clip")
        for point, value in zip(self._scan_hits, self._scan_values):
            if not np.isfinite(point[2]):
                continue
            if clip is not None:
                value = float(np.clip(value, *clip))
            # -RANGE -> 蓝, 0 -> 灰, +RANGE -> 红
            level = float(np.clip(value / SCAN_MARKER_RANGE, -1.0, 1.0))
            rgba = (0.5 + 0.5 * level, 0.5 - 0.5 * abs(level), 0.5 - 0.5 * level, 0.9)
            _add_sphere(scene, point, SCAN_MARKER_RADIUS, rgba)


def _add_sphere(scene, pos, radius, rgba):
    """往 viewer.user_scn 里追加一个小球(flat 版只提供了箭头)。"""
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_SPHERE,
        np.full(3, radius, dtype=np.float64), np.asarray(pos, dtype=np.float64),
        np.eye(3).reshape(-1), np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=RUN)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--no-scan-marker", action="store_true", help="不画高度图采样点")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit(f"请在脚本顶部 RUN 或 --run 指定 {LOGS_ROOT} 下的训练 run 目录名")
    flat._self_check()
    duration = args.duration if args.duration is not None else (10.0 if args.headless else flat.SIM_DURATION)
    config = flat.load_config(args.run)
    if "height_scan" not in config.observations:
        raise SystemExit(
            f"{args.run} 的 deploy.yaml 里 actor 观测没有 height_scan —— 这是盲走策略, "
            "请用 rough/sim2sim.py 回放"
        )
    scanner = read_scanner_cfg(config.run_dir)
    print(f"[sim2sim] run={args.run}, action_clip=±{config.policy_action_clip}")
    runner = RoughInfoRunner(
        config, scanner, show_viewer=not args.headless, save_data=args.save_data,
        scan_marker=SCAN_MARKER and not args.no_scan_marker,
    )
    runner.run(duration=duration, realtime=not args.headless)
    print(f"[sim2sim] 完成 {runner.data.time:.2f}s，base_z={runner.data.xpos[runner.base_body_id, 2]:.3f}m")


if __name__ == "__main__":
    main()
