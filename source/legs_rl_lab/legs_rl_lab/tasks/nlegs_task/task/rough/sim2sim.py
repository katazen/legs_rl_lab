"""nlegs_rough 的 MuJoCo sim2sim 回放器 —— 复用 flat 版全部逻辑，只在场景上叠加两块验证地形。

用法(按文件路径直接运行, 不要 python -m 走包导入):
    python source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/task/rough/sim2sim.py \
        [--run RUN] [--headless] [--duration S] [--save-data]

地形(参数见下方可改块):
  1. 楼梯(+X 方向, 近端距原点 2m): 宽 150cm, 每级高 5cm/深 20cm 共 4 级上行,
     最高处是 20cm 高、40cm 深的平台(即第 4 级踏面), 随后以相同台阶下行回到地面。
  2. 斜坡(+Y 方向, 近端距原点 2m): 宽 150cm, 5° 上坡 2m -> 40cm 深平台(高约 17.5cm)
     -> 5° 下坡回到地面。
键盘/可视化与 flat 版一致: 小键盘 8/2 4/6 7/9 调速, 原点三轴 + 速度跟踪箭头。
"""

import argparse
import importlib.util
import os

import mujoco
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_flat_sim2sim():
    """按文件路径加载 flat/sim2sim.py（包导入会拉起 Isaac Lab，故走 importlib）。"""
    path = os.path.abspath(os.path.join(_HERE, "..", "flat", "sim2sim.py"))
    spec = importlib.util.spec_from_file_location("nlegs_flat_sim2sim", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


flat = _load_flat_sim2sim()

# ===================== 需要自己填/改的部分（全部集中在这里） =====================
# 要回放的训练 run（logs/rsl_rl/nlegs_rough/ 下的目录名，需先用 play 导出 exported/policy.pt）
RUN = "2026-08-27_19-45-40"
# run 所在的 logs 根目录（想用平地策略先试地形，可临时指到 .../nlegs_flat）
LOGS_ROOT = os.path.join(flat._REPO_ROOT, "logs", "rsl_rl", "nlegs_rough")

# --- 楼梯（沿 +X 行进, 宽度沿 Y 居中）---
STAIR_START = 2.0        # 近端边缘距原点(m)
STAIR_WIDTH = 2.0        # 宽(m)
STAIR_STEP_H = 0.05      # 每级高(m)
STAIR_STEP_D = 0.2       # 每级深(m)
STAIR_NUM_STEPS = 4      # 上行级数(第 NUM 级踏面 = 顶部平台)
STAIR_PLATFORM_D = 0.4   # 顶部平台深(m), 高 = NUM_STEPS * STEP_H
STAIR_RGBA = (0.7, 0.7, 0.7, 1.0)

# --- 斜坡（沿 +Y 行进, 宽度沿 X 居中）---
RAMP_START = 2.0         # 近端边缘距原点(m)
RAMP_WIDTH = 2.0         # 宽(m)
RAMP_SLOPE_DEG = 10.0     # 坡度(°)
RAMP_RUN = 2.0           # 单边坡的水平投影长度(m), 顶高 = RUN * tan(坡度)
RAMP_PLATFORM_D = 0.4    # 顶部平台深(m)
RAMP_THICKNESS = 0.01     # 坡面板厚(m), 只影响外观/底部
RAMP_RGBA = (0.55, 0.5, 0.45, 1.0)
# ==============================================================================

# flat.load_config 读取模块全局 LOGS_ROOT
flat.LOGS_ROOT = LOGS_ROOT


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
    """+Y 方向: 5° 上坡 -> 顶部平台 -> 5° 下坡。坡面用绕 X 轴旋转的板状 box。"""
    angle = np.deg2rad(RAMP_SLOPE_DEG)
    rise = RAMP_RUN * np.tan(angle)
    slope_len = RAMP_RUN / np.cos(angle)
    half_w, half_t = RAMP_WIDTH / 2, RAMP_THICKNESS / 2
    half_angle = angle / 2

    # 上坡: 绕 +X 转 +angle, 板顶面从 (y=START,z=0) 升到 (y=START+RUN,z=rise)
    top_mid_y = RAMP_START + RAMP_RUN / 2
    _add_box(spec, "ramp_up",
             (0.0, top_mid_y + half_t * np.sin(angle), rise / 2 - half_t * np.cos(angle)),
             (half_w, slope_len / 2, half_t), RAMP_RGBA,
             quat=(np.cos(half_angle), np.sin(half_angle), 0.0, 0.0))
    # 顶部平台
    plat_y = RAMP_START + RAMP_RUN + RAMP_PLATFORM_D / 2
    _add_box(spec, "ramp_platform", (0.0, plat_y, rise / 2),
             (half_w, RAMP_PLATFORM_D / 2, rise / 2), RAMP_RGBA)
    # 下坡: 绕 +X 转 -angle, 镜像
    top_mid_y = RAMP_START + RAMP_RUN + RAMP_PLATFORM_D + RAMP_RUN / 2
    _add_box(spec, "ramp_down",
             (0.0, top_mid_y - half_t * np.sin(angle), rise / 2 - half_t * np.cos(angle)),
             (half_w, slope_len / 2, half_t), RAMP_RGBA,
             quat=(np.cos(half_angle), -np.sin(half_angle), 0.0, 0.0))


def add_terrain(spec):
    _add_stairs(spec)
    _add_ramp(spec)


class RoughRunner(flat.MujocoRunner):
    """flat 的 MujocoRunner + 程序化叠加楼梯/斜坡地形。"""

    def _make_model(self, cfg):
        spec = mujoco.MjSpec.from_file(cfg.scene_xml)
        add_terrain(spec)
        rise = RAMP_RUN * np.tan(np.deg2rad(RAMP_SLOPE_DEG))
        print(f"[sim2sim] 地形: 楼梯 +X {STAIR_START}m 起(顶高 {STAIR_NUM_STEPS * STAIR_STEP_H:.2f}m), "
              f"斜坡 +Y {RAMP_START}m 起(顶高 {rise:.3f}m)")
        return spec.compile()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=RUN)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--save-data", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit(f"请在脚本顶部 RUN 或 --run 指定 {LOGS_ROOT} 下的训练 run 目录名")
    flat._self_check()
    duration = args.duration if args.duration is not None else (10.0 if args.headless else flat.SIM_DURATION)
    config = flat.load_config(args.run)
    print(f"[sim2sim] run={args.run}, action_clip=±{config.policy_action_clip}")
    runner = RoughRunner(config, show_viewer=not args.headless, save_data=args.save_data)
    runner.run(duration=duration, realtime=not args.headless)
    print(f"[sim2sim] 完成 {runner.data.time:.2f}s，base_z={runner.data.xpos[runner.base_body_id, 2]:.3f}m")


if __name__ == "__main__":
    main()
