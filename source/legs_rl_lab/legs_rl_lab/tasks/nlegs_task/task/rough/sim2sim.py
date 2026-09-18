"""nlegs_rough 的 MuJoCo sim2sim 回放器 —— 上坡、平台、下楼梯连续地形。

用法(按文件路径直接运行, 不要 python -m 走包导入):
    python source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/task/rough/sim2sim.py \
        [--run RUN] [--headless] [--duration S] [--save-data] [--check-terrain]

地形(参数见下方可改块):
  沿 +X 行进，距原点 2m 开始，整条路线宽 2m：
  上坡水平长 2m、抬高 20cm(约 5.71°) -> 30cm 长平台 -> 每级下降 5cm、踏面深 20cm。
  下行踏面高度依次为 15/10/5cm，最后落回地面，共 4 次落差；踏面越高灰度越浅。
  --check-terrain 只检查地形碰撞面高度与衔接，不加载策略或打开窗口。
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
RUN = "2026-09-18_10-10-59"
# run 所在的 logs 根目录（想用平地策略先试地形，可临时指到 .../nlegs_flat）
LOGS_ROOT = os.path.join(flat._REPO_ROOT, "logs", "rsl_rl", "nlegs_rough")

# --- 平台与下楼梯（接在上坡末端，沿 +X 行进）---
STAIR_STEP_H = 0.05      # 每级高(m)
STAIR_STEP_D = 0.2       # 每级深(m)
STAIR_NUM_STEPS = 4      # 下行落差次数(含最后到地面)，也决定上坡总高度
STAIR_PLATFORM_D = 0.3   # 顶部平台长(m), 高 = NUM_STEPS * STEP_H = 0.20m
STAIR_TREAD_GRAY = (0.35, 0.95)   # 踏面灰度区间: (最低一级, 顶部平台), 按级数线性变亮
STAIR_RISER_RGBA = (0.05, 0.05, 0.05, 1.0)   # 立面(前后/侧面)
STAIR_TREAD_T = 0.004    # 踏面薄板厚(m), 从黑色主体高度里扣掉, 总高不变

# --- 上坡（沿 +X 行进, 宽度沿 Y 居中）---
RAMP_START = 2.0         # 近端边缘距原点(m)
RAMP_WIDTH = 2.0         # 整条路线宽(m)，平台与下楼梯使用相同宽度
RAMP_RUN = 2.0           # 上坡水平投影长度(m)，坡度由总高度/此长度计算
RAMP_THICKNESS = 0.01     # 坡面板厚(m), 只影响外观/底部
RAMP_RGBA = (0.55, 0.5, 0.45, 1.0)

# --- 出生高度（base 原点的世界 z, 覆盖 deploy.yaml 的 base_init_state.pos[2]）---
# 默认站姿 base_z ≈ 0.582, 故 0.82 = 约 0.24m 自由下落, 用来验证落地抗冲击/自恢复。
DROP_HEIGHT = 0.62
# ==============================================================================

# flat.load_config 读取模块全局 LOGS_ROOT
flat.LOGS_ROOT = LOGS_ROOT


def _add_box(spec, name, pos, size, rgba, quat=(1.0, 0.0, 0.0, 0.0)):
    spec.worldbody.add_geom(
        name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=list(pos), size=list(size), quat=list(quat), rgba=list(rgba),
    )


def _tread_rgba(level):
    """第 level 级(1..N)踏面的灰度: 最低一级最深、顶部平台最浅。"""
    lo, hi = STAIR_TREAD_GRAY
    gray = lo + (hi - lo) * (level - 1) / max(STAIR_NUM_STEPS - 1, 1)
    return (gray, gray, gray, 1.0)


def _add_step(spec, name, x0, depth, h, half_w, tread_rgba):
    """一级台阶 = 黑色立面主体(高 h - t) + 顶面灰阶踏面薄板(厚 t)。

    拆成两块而不是直接叠一层薄板: 黑块高度扣掉板厚, 两者拼起来总高仍精确为 h,
    碰撞面高度与改色前一致, 也不会因共面产生 z-fighting。
    """
    t = min(STAIR_TREAD_T, h)
    body_h = h - t
    if body_h > 0:
        _add_box(spec, f"{name}_riser", (x0 + depth / 2, 0.0, body_h / 2),
                 (depth / 2, half_w, body_h / 2), STAIR_RISER_RGBA)
    _add_box(spec, f"{name}_tread", (x0 + depth / 2, 0.0, h - t / 2),
             (depth / 2, half_w, t / 2), tread_rgba)


def _add_stairs(spec):
    """从上坡末端接平台，再逐级下行回到地面。"""
    half_w = RAMP_WIDTH / 2
    x = RAMP_START + RAMP_RUN
    h_top = STAIR_NUM_STEPS * STAIR_STEP_H
    _add_step(spec, "stair_platform", x, STAIR_PLATFORM_D, h_top, half_w,
              _tread_rgba(STAIR_NUM_STEPS))
    x += STAIR_PLATFORM_D
    for i in range(STAIR_NUM_STEPS - 1, 0, -1):
        _add_step(spec, f"stair_down_{i}", x, STAIR_STEP_D, i * STAIR_STEP_H, half_w, _tread_rgba(i))
        x += STAIR_STEP_D


def _add_ramp(spec):
    """沿 +X 上坡，板顶面从地面准确接到平台高度。"""
    rise = STAIR_NUM_STEPS * STAIR_STEP_H
    angle = np.arctan2(rise, RAMP_RUN)
    slope_len = RAMP_RUN / np.cos(angle)
    half_w, half_t = RAMP_WIDTH / 2, RAMP_THICKNESS / 2
    half_angle = angle / 2

    # 绕 Y 轴转 -angle，并沿法线下移半个板厚，保证顶面两端没有额外台阶。
    top_mid_x = RAMP_START + RAMP_RUN / 2
    _add_box(spec, "ramp_up",
             (top_mid_x + half_t * np.sin(angle), 0.0, rise / 2 - half_t * np.cos(angle)),
             (slope_len / 2, half_w, half_t), RAMP_RGBA,
             quat=(np.cos(half_angle), 0.0, -np.sin(half_angle), 0.0))


def add_terrain(spec):
    _add_ramp(spec)
    _add_stairs(spec)


def _check_terrain():
    """用 MuJoCo 射线检查坡面、平台、各级踏面及接缝两侧。"""
    spec = mujoco.MjSpec.from_string(
        '<mujoco><worldbody><geom type="plane" size="10 10 0.1"/></worldbody></mujoco>'
    )
    add_terrain(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rise = STAIR_NUM_STEPS * STAIR_STEP_H
    epsilon = 1e-6
    samples = [(RAMP_START - epsilon, 0.0)]
    samples += [(RAMP_START + s, rise * s / RAMP_RUN)
                for s in (epsilon, RAMP_RUN / 2, RAMP_RUN - epsilon)]
    x = RAMP_START + RAMP_RUN
    for level in range(STAIR_NUM_STEPS, 0, -1):
        depth = STAIR_PLATFORM_D if level == STAIR_NUM_STEPS else STAIR_STEP_D
        samples += [(x + s, level * STAIR_STEP_H)
                    for s in (epsilon, depth / 2, depth - epsilon)]
        x += depth
    samples.append((x + epsilon, 0.0))
    geom_id = np.zeros(1, dtype=np.int32)
    for x, height in samples:
        for y in (-RAMP_WIDTH / 2 + epsilon, 0.0, RAMP_WIDTH / 2 - epsilon,
                  RAMP_WIDTH / 2 + epsilon):
            expected = height if abs(y) < RAMP_WIDTH / 2 else 0.0
            distance = mujoco.mj_ray(model, data, np.array([x, y, rise + 1.0]),
                                    np.array([0.0, 0.0, -1.0]), None, 1, -1, geom_id)
            assert distance >= 0.0 and np.isclose(rise + 1.0 - distance, expected, atol=1e-8), (
                x, y, expected, rise + 1.0 - distance
            )
    assert model.ngeom == 2 + 2 * STAIR_NUM_STEPS
    print("[sim2sim] 地形检查通过：上坡、平台、下楼梯高度/宽度及接缝正确")


class RoughRunner(flat.MujocoRunner):
    """flat 的 MujocoRunner + 连续上坡/平台/下楼梯地形。"""

    def _make_model(self, cfg):
        spec = mujoco.MjSpec.from_file(cfg.scene_xml)
        add_terrain(spec)
        rise = STAIR_NUM_STEPS * STAIR_STEP_H
        angle = np.rad2deg(np.arctan2(rise, RAMP_RUN))
        print(f"[sim2sim] 地形: +X {RAMP_START}m 起，宽 {RAMP_WIDTH}m，上坡 {RAMP_RUN}m "
              f"抬高 {rise:.2f}m ({angle:.2f}°) → 平台 {STAIR_PLATFORM_D}m → "
              f"下楼 {STAIR_NUM_STEPS} 级，每级高 {STAIR_STEP_H}m / 深 {STAIR_STEP_D}m")
        return spec.compile()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=RUN)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--check-terrain", action="store_true", help="仅检查地形几何，不加载策略")
    args = parser.parse_args()
    if args.check_terrain:
        _check_terrain()
        return
    if not args.run:
        raise SystemExit(f"请在脚本顶部 RUN 或 --run 指定 {LOGS_ROOT} 下的训练 run 目录名")
    flat._self_check()
    duration = args.duration if args.duration is not None else (10.0 if args.headless else flat.SIM_DURATION)
    config = flat.load_config(args.run)
    config.base_pos[2] = DROP_HEIGHT  # 覆盖 deploy.yaml 的出生高度, 从空中自由下落
    print(f"[sim2sim] run={args.run}, action_clip=±{config.policy_action_clip}, "
          f"出生高度 {config.base_pos[2]:.3f}m")
    runner = RoughRunner(config, show_viewer=not args.headless, save_data=args.save_data)
    runner.run(duration=duration, realtime=not args.headless)
    print(f"[sim2sim] 完成 {runner.data.time:.2f}s，base_z={runner.data.xpos[runner.base_body_id, 2]:.3f}m")


if __name__ == "__main__":
    main()
