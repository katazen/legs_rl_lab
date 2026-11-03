"""nlegs_rough 的 MuJoCo sim2sim 回放器 —— 上坡、平台、下坡连续地形。

用法(按文件路径直接运行, 不要 python -m 走包导入):
    python source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/task/rough/sim2sim.py \
        [--run RUN] [--xml XML] [--headless] [--duration S] [--save-data] [--check-terrain]

地形(参数见下方可改块):
  沿 +X 行进，距原点 2m 开始，整条路线宽 2m：
  上坡水平长 2m、抬高 20cm(约 5.71°) -> 60cm 长平台 -> 同坡度下坡回到地面。
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
# 要回放的训练 run（logs/rsl_rl/nlegs_body_rough/ 下的目录名，需已有 exported/policy.pt）
RUN = "2026-11-03_11-24-26_w1196"
# run 所在的 logs 根目录（想用平地策略先试地形，可临时指到 .../nlegs_flat）
LOGS_ROOT = os.path.join(flat._REPO_ROOT, "logs", "rsl_rl", "nlegs_body_rough")
# 机器人场景 XML（包含地面）；可直接改这里，或用 --xml / --scene-xml 覆盖。
SCENE_XML = os.path.join(
    flat._REPO_ROOT, "source", "legs_rl_lab", "legs_rl_lab", "assets", "nlegs_v2",
    "w1196_scene.xml",
)

# --- 顶部平台（接在上坡末端，高度与宽度沿用坡体）---
PLATFORM_LENGTH = 0.60
PLATFORM_RGBA = (1.0, 0.8, 0.05, 1.0)  # 亮黄色，区分平台与两侧坡面

# --- 上坡（沿 +X 行进, 宽度沿 Y 居中）---
RAMP_START = 2.0         # 近端边缘距原点(m)
RAMP_WIDTH = 2.0         # 整条路线宽(m)，平台与下坡使用相同宽度
RAMP_RUN = 2.0           # 上坡水平投影长度(m)，坡度由总高度/此长度计算
RAMP_HEIGHT = 0.20       # 上坡抬高(m)，下坡使用相同高度和水平长度
RAMP_THICKNESS = 0.30    # 地下厚度(m)：底面埋入地面，避免足底下陷时碰到坡体背面
RAMP_RGBA = (0.55, 0.5, 0.45, 1.0)

# --- 出生高度（base 原点的世界 z, 覆盖 deploy.yaml 的 base_init_state.pos[2]）---
# 与 W1196 训练初始高度一致，默认站姿足底距地约 3.7cm。
DROP_HEIGHT = 0.555
# ==============================================================================

# flat.load_config 读取模块全局 LOGS_ROOT
flat.LOGS_ROOT = LOGS_ROOT


def add_terrain(spec):
    """用单一凸棱柱连接上下坡与平台，避免分块内部端面产生横向接触力。"""
    up_end = RAMP_START + RAMP_RUN
    down_start = up_end + PLATFORM_LENGTH
    end = down_start + RAMP_RUN
    profile = [(RAMP_START, -RAMP_THICKNESS), (end, -RAMP_THICKNESS),
               (end, 0.0), (down_start, RAMP_HEIGHT),
               (up_end, RAMP_HEIGHT), (RAMP_START, 0.0)]
    vertices = [[x, y, z] for y in (-RAMP_WIDTH / 2, RAMP_WIDTH / 2) for x, z in profile]
    spec.add_mesh(name="ramp_course_mesh", uservert=np.asarray(vertices).ravel().tolist())
    spec.worldbody.add_geom(
        name="ramp_course", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="ramp_course_mesh",
        rgba=list(RAMP_RGBA), contype=1, conaffinity=1, group=0,
    )
    # 仅用于着色；抬高 0.2mm 避免共面闪烁，碰撞仍使用完整坡体。
    spec.worldbody.add_geom(
        name="platform_color", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[(up_end + down_start) / 2, 0.0, RAMP_HEIGHT + 0.0001],
        size=[PLATFORM_LENGTH / 2, RAMP_WIDTH / 2, 0.0001],
        rgba=list(PLATFORM_RGBA), contype=0, conaffinity=0, group=2,
    )


def _check_terrain():
    """用 MuJoCo 射线检查上下坡、平台及接缝两侧，并检查坡面接触法向。"""
    spec = mujoco.MjSpec.from_string(
        '<mujoco><worldbody><geom type="plane" size="10 10 0.1"/></worldbody></mujoco>'
    )
    add_terrain(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rise = RAMP_HEIGHT
    # 射线避开 mesh 共享边的浮点歧义；精确接缝另用下方接触检查覆盖。
    epsilon = 1e-6
    samples = [(RAMP_START - epsilon, 0.0)]
    samples += [(RAMP_START + s, rise * s / RAMP_RUN)
                for s in (epsilon, RAMP_RUN / 2, RAMP_RUN - epsilon)]
    x = RAMP_START + RAMP_RUN
    samples += [(x + s, rise) for s in (epsilon, PLATFORM_LENGTH / 2, PLATFORM_LENGTH - epsilon)]
    x += PLATFORM_LENGTH
    samples += [(x + s, rise * (1 - s / RAMP_RUN))
                for s in (epsilon, RAMP_RUN / 2, RAMP_RUN - epsilon)]
    samples.append((x + RAMP_RUN + epsilon, 0.0))
    geom_id = np.zeros(1, dtype=np.int32)
    collision_groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    for sample_x, height in samples:
        for y in (-RAMP_WIDTH / 2 - epsilon, -RAMP_WIDTH / 2 + epsilon, 0.0, RAMP_WIDTH / 2 - epsilon,
                  RAMP_WIDTH / 2 + epsilon):
            expected = height if abs(y) < RAMP_WIDTH / 2 else 0.0
            distance = mujoco.mj_ray(model, data, np.array([sample_x, y, rise + 1.0]),
                                    np.array([0.0, 0.0, -1.0]), collision_groups, 1, -1, geom_id)
            assert distance >= 0.0 and np.isclose(rise + 1.0 - distance, expected, atol=1e-8), (
                sample_x, y, expected, rise + 1.0 - distance
            )
    assert model.ngeom == 3  # 地面、连续坡体和无碰撞的着色面

    # 原 10mm 薄板在脚底球心穿过中面后会产生向下接触；复现该侵入量。
    assert RAMP_THICKNESS > 0.0, "坡体底面必须埋入地面"
    probe = spec.worldbody.add_body(name="terrain_contact_probe")
    probe.add_freejoint()
    probe.add_geom(name="probe", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                   size=[0.0035, 0.0, 0.0], mass=1.0)
    model = spec.compile()
    data = mujoco.MjData(model)
    ramp_id, probe_id = model.geom("ramp_course").id, model.geom("probe").id
    up_end = RAMP_START + RAMP_RUN
    down_start = up_end + PLATFORM_LENGTH
    contact_x = [RAMP_START + RAMP_RUN / 2, down_start + RAMP_RUN / 2]
    contact_x += [seam + offset for seam in (up_end, down_start) for offset in (-0.001, 0.0, 0.001)]
    for sample_x in contact_x:
        height = np.interp(sample_x, [RAMP_START, up_end, down_start, down_start + RAMP_RUN],
                           [0.0, rise, rise, 0.0])
        data.qpos[:3] = (sample_x, 0.0, height - 0.0055)
        mujoco.mj_forward(model, data)
        contacts = [c for c in data.contact if {c.geom1, c.geom2} == {ramp_id, probe_id}]
        assert contacts, f"侵入坡面 x={sample_x} 的探针必须产生接触"
        for contact in contacts:
            sign = 1.0 if contact.geom2 == probe_id else -1.0
            assert sign * contact.frame[2] > 0.99, "坡面及接缝接触法向必须向上"
    print(f"[sim2sim] 地形检查通过：上下坡/{PLATFORM_LENGTH:g}m平台高度、宽度及接缝正确，"
          "坡面和接缝侵入9mm仍向上支撑")


class RoughRunner(flat.MujocoRunner):
    """flat 的 MujocoRunner + 连续上坡/平台/下坡地形。"""

    def _make_model(self, cfg):
        spec = mujoco.MjSpec.from_file(cfg.scene_xml)
        add_terrain(spec)
        rise = RAMP_HEIGHT
        angle = np.rad2deg(np.arctan2(rise, RAMP_RUN))
        print(f"[sim2sim] 地形: +X {RAMP_START}m 起，宽 {RAMP_WIDTH}m，上坡 {RAMP_RUN}m "
              f"抬高 {rise:.2f}m ({angle:.2f}°) → 平台 {PLATFORM_LENGTH}m → "
              f"下坡 {RAMP_RUN}m 降低 {rise:.2f}m ({angle:.2f}°)")
        return spec.compile()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=RUN)
    parser.add_argument("--xml", "--scene-xml", dest="scene_xml", default=SCENE_XML,
                        help="机器人场景 XML（含地面），命令行路径优先于顶部 SCENE_XML")
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
    config.scene_xml = os.path.abspath(os.path.expanduser(args.scene_xml))
    config.base_pos[2] = DROP_HEIGHT  # 覆盖 deploy.yaml 的出生高度, 从空中自由下落
    print(f"[sim2sim] run={args.run}, action_clip=±{config.policy_action_clip}, "
          f"出生高度 {config.base_pos[2]:.3f}m")
    print(f"[sim2sim] XML={config.scene_xml}")
    runner = RoughRunner(config, show_viewer=not args.headless, save_data=args.save_data)
    runner.run(duration=duration, realtime=not args.headless)
    print(f"[sim2sim] 完成 {runner.data.time:.2f}s，base_z={runner.data.xpos[runner.base_body_id, 2]:.3f}m")


if __name__ == "__main__":
    main()
