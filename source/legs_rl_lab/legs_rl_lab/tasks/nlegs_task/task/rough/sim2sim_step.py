"""nlegs_rough_step 的 MuJoCo sim2sim 回放器 —— 复用 flat 版全部逻辑, 只改地形与命令生成方式。

用法(按文件路径直接运行, 不要 python -m 走包导入):
    python source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/task/rough/sim2sim_step.py \
        --run RUN [--start-yaw 0.3] [--speed 0.4] [--headless] [--duration S] [--save-data]

与 rough/sim2sim.py 的三处区别(对应训练侧 rough_step_env_cfg.py 的三处区别):

1) 命令由**朝向纠偏**自动生成, 不再由键盘给常量 yaw 速度。
   训练用 heading_command=True + rel_heading_envs=1.0 + ranges.heading=(0,0), 即
       wz = clip(heading_control_stiffness * wrap_to_pi(heading_target - yaw_w),
                 ang_vel_z[0], ang_vel_z[1])
   这里逐控制步复刻该式(见 _update_heading_command)。若沿用 rough/sim2sim.py 的键盘
   常量 yaw, wz 的分布与训练完全不同, sim2sim 结果不可信。
   - heading_target 从 deploy.yaml 的 commands.base_velocity.ranges.heading 读(导出器已导出)。
   - ang_vel_z 的 clip 界直接用 cfg.command_ranges[2](同样来自 deploy.yaml)。
   - **heading_control_stiffness 导出器没有导出**, 故下方 HEADING_STIFFNESS 需手工与
     rough_step_env_cfg.py 保持一致。这是本脚本唯一的手工同步点。
   - vx 恒定(默认取 ranges.lin_vel_x 中点), vy 恒 0; 小键盘 8/2 可在 ranges 内微调 vx,
     7/9 改的是**目标朝向** heading_target(直接改 wz 会被下一帧纠偏覆盖)。

2) 地形只留楼梯, **去掉了 rough 版的斜坡** —— rough_step 训练集里没有坡, 测一个没训过的
   地形只会制造困惑。台阶参数直接对齐实机(5cm 高 / 20cm 深 / 4 级)。
   顶部平台加长到 2m(rough 版是 0.4m): 训练用的是"反金字塔坑", 走完台阶后是无限大的外围
   平地; 平台太短会让机器人刚上顶就被迫下楼, 偏离训练分布, 也没法判断上楼是否真的成功。

3) 支持 --start-yaw 指定初始偏航(训练 reset 有 ±0.4rad 随机)。实机放歪是必然的, 用它验证
   heading 纠偏是否真能把机器人拽回正对楼梯。

actor 观测与 flat/rough 完全一致(盲走, 47 维 x 10 帧), 故 flat 版的 _OBS_FEATURES /
TermGroupedHistory / 力矩模型 / 延迟模型全部原样复用, 本文件不改动任何共享代码。

若拿 nlegs_rough 或 nlegs_flat 的策略跑本脚本(想在同一楼梯上做对比), deploy.yaml 里
heading 会是 null, 此时自动退回"键盘常量命令"模式并给出提示。
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

# ===================== 需要自己填/改的部分（全部集中在这里） =====================
# 要回放的训练 run（logs/rsl_rl/nlegs_rough_step/ 下的目录名，需先用 play 导出 exported/policy.pt）
RUN = "2026-08-31_15-51-22"
# run 所在的 logs 根目录（想拿 rough/flat 策略在同一楼梯上做对比，把这里临时指过去即可）
LOGS_ROOT = os.path.join(flat._REPO_ROOT, "logs", "rsl_rl", "nlegs_rough_step")

# ⚠️ 必须与 rough_step_env_cfg.py 的 CommandsCfg.base_velocity.heading_control_stiffness 一致。
#    导出器不导出该字段，改了训练侧记得同步这里。
HEADING_STIFFNESS = 0.5
# 初始偏航(rad)，训练 reset 范围是 ±0.4；设非 0 可验证 heading 纠偏
START_YAW = 0.0
# 恒定前进速度(m/s)；None = 取 deploy.yaml 里 ranges.lin_vel_x 的中点
FORWARD_SPEED = None
# 小键盘 7/9 每次改变目标朝向的量(rad)
HEADING_KEY_STEP = 0.1

# --- 楼梯（沿 +X 行进, 宽度沿 Y 居中）---
STAIR_START = 2.0        # 近端边缘距原点(m), 兼作助跑距离
STAIR_WIDTH = 2.0        # 宽(m)
STAIR_STEP_H = 0.05      # 每级高(m), 对齐实机
STAIR_STEP_D = 0.2       # 每级深(m), 对齐实机
STAIR_NUM_STEPS = 4      # 上行级数, 对齐实机
STAIR_PLATFORM_D = 2.0   # 顶部平台深(m), 高 = NUM_STEPS * STEP_H
STAIR_ADD_DESCENT = True  # 平台后是否接下行台阶(训练里下楼占 10%)
STAIR_RGBA = (0.7, 0.7, 0.7, 1.0)
# ==============================================================================

# flat.load_config 读取模块全局 LOGS_ROOT
flat.LOGS_ROOT = LOGS_ROOT
# 去掉 vy(4/6): 训练 ranges.lin_vel_y=(0,0), 改了也是零。
# 8/2 调 vx; 7/9 不再直接改 wz(会被朝向纠偏立刻覆盖), 改的是**目标朝向** heading_target。
flat.KEY_BINDINGS = {
    "8": (0, 0.05), "2": (0, -0.05),
    "7": (2, HEADING_KEY_STEP), "9": (2, -HEADING_KEY_STEP),
}


def _wrap_to_pi(angle):
    """与 isaaclab.utils.math.wrap_to_pi 逐位等价（π 映射到 π，−π 映射到 −π）。"""
    wrapped = (angle + np.pi) % (2.0 * np.pi)
    if wrapped == 0.0 and angle > 0.0:
        return np.pi
    return wrapped - np.pi


def _read_heading_target(run_dir):
    """从 deploy.yaml 读 ranges.heading；null(未开 heading_command) 返回 None。

    训练侧 ranges.heading=(0,0) 表示目标朝向锁死世界系 +x，与本脚本楼梯朝向一致。
    """
    with open(os.path.join(run_dir, "params", "deploy.yaml")) as file:
        deploy = yaml.safe_load(file)
    heading = deploy["commands"]["base_velocity"]["ranges"].get("heading")
    if heading is None:
        return None
    low, high = float(heading[0]), float(heading[1])
    if not np.isclose(low, high):
        print(f"[sim2sim] ranges.heading=({low}, {high}) 非单点，取中点 {(low + high) / 2:.3f}")
    return (low + high) / 2.0


def _add_box(spec, name, pos, size, rgba, quat=(1.0, 0.0, 0.0, 0.0)):
    spec.worldbody.add_geom(
        name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=list(pos), size=list(size), quat=list(quat), rgba=list(rgba),
    )


def add_terrain(spec):
    """+X 方向: 上行踏面 1..N-1 -> 顶部平台(第 N 级) -> 可选下行踏面 N-1..1。"""
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
    if STAIR_ADD_DESCENT:
        for i in range(STAIR_NUM_STEPS - 1, 0, -1):
            h = i * STAIR_STEP_H
            _add_box(spec, f"stair_down_{i}", (x + STAIR_STEP_D / 2, 0.0, h / 2),
                     (STAIR_STEP_D / 2, half_w, h / 2), STAIR_RGBA)
            x += STAIR_STEP_D


class RoughStepRunner(flat.MujocoRunner):
    """flat 的 MujocoRunner + 楼梯地形 + 朝向纠偏命令。"""

    def __init__(self, cfg, show_viewer=True, save_data=False,
                 heading_target=None, start_yaw=0.0, forward_speed=None):
        super().__init__(cfg, show_viewer=show_viewer, save_data=save_data)
        self.heading_target = heading_target
        # 恒定前进命令；vy 恒 0（训练 ranges.lin_vel_y=(0,0)）
        vx_low, vx_high = cfg.command_ranges[0]
        vx = (vx_low + vx_high) / 2.0 if forward_speed is None else forward_speed
        self.command[0] = np.clip(vx, vx_low, vx_high)
        self.command[1] = 0.0
        if heading_target is None:
            print("[sim2sim] deploy.yaml 里 ranges.heading=null（该 run 未开 heading_command），"
                  "退回键盘常量命令模式；wz 保持 0")
        else:
            self._apply_start_yaw(start_yaw)
            self._update_heading_command()
        print(f"[sim2sim] 命令: vx={self.command[0]:+.2f}(范围 {vx_low:+.2f}~{vx_high:+.2f}) "
              f"vy=0.00 wz=朝向纠偏(stiffness={HEADING_STIFFNESS}, "
              f"clip=±{cfg.command_ranges[2, 1]:.2f})")
        # 上楼进度追踪（起点为基准）
        self._init_base_z = float(self.data.xpos[self.base_body_id, 2])
        self.max_base_x = float(self.data.xpos[self.base_body_id, 0])
        self.max_base_z = self._init_base_z

    def _make_model(self, cfg):
        spec = mujoco.MjSpec.from_file(cfg.scene_xml)
        add_terrain(spec)
        tail = "，平台后接下行台阶" if STAIR_ADD_DESCENT else ""
        print(f"[sim2sim] 地形: 楼梯 +X {STAIR_START}m 起，{STAIR_NUM_STEPS} 级 x "
              f"{STAIR_STEP_H * 100:.0f}cm 高 / {STAIR_STEP_D * 100:.0f}cm 深，"
              f"顶高 {STAIR_NUM_STEPS * STAIR_STEP_H:.2f}m，顶部平台 {STAIR_PLATFORM_D}m{tail}")
        return spec.compile()

    def _apply_start_yaw(self, yaw):
        """在 cfg.base_quat 上叠加绕 z 的初始偏航（训练 reset 的 yaw 随机对应项）。"""
        if abs(yaw) < 1e-9:
            return
        q_yaw = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float64)
        q_out = np.zeros(4, dtype=np.float64)
        mujoco.mju_mulQuat(q_out, q_yaw, self.cfg.base_quat.astype(np.float64))
        self.data.qpos[self.base_qpos_adr + 3:self.base_qpos_adr + 7] = q_out
        mujoco.mj_forward(self.model, self.data)
        print(f"[sim2sim] 初始偏航 {yaw:+.3f} rad ({np.rad2deg(yaw):+.1f}°)，靠朝向纠偏拉回")

    def _update_heading_command(self):
        """复刻 isaaclab UniformVelocityCommand 的 heading 分支（velocity_command.py）。"""
        quat = self.data.qpos[self.base_qpos_adr + 3:self.base_qpos_adr + 7]
        error = _wrap_to_pi(self.heading_target - flat.yaw_from_quat(quat))
        self.command[2] = np.clip(
            HEADING_STIFFNESS * error,
            self.cfg.command_ranges[2, 0],
            self.cfg.command_ranges[2, 1],
        )

    def _adjust_command(self, index, increment):
        """7/9 改目标朝向而非 wz —— 直接改 command[2] 会在下一帧被纠偏覆盖。"""
        if index == 2 and self.heading_target is not None:
            self.heading_target = _wrap_to_pi(self.heading_target + increment)
            self._update_heading_command()
            print(f"[cmd] vx={self.command[0]:+.2f} heading_target="
                  f"{np.rad2deg(self.heading_target):+.1f}° wz={self.command[2]:+.2f}")
            return
        super()._adjust_command(index, increment)

    def _observation_terms(self):
        # 命令在观测被读取之前刷新，保证 wz 与当前 yaw 同步（与训练的 _update_command 时序一致）
        if self.heading_target is not None:
            self._update_heading_command()
        base = self.data.xpos[self.base_body_id]
        self.max_base_x = max(self.max_base_x, float(base[0]))
        self.max_base_z = max(self.max_base_z, float(base[2]))
        return super()._observation_terms()

    def report(self):
        """判定是否真的上到了顶部平台。"""
        climbed = self.max_base_z - self._init_base_z
        expected = STAIR_NUM_STEPS * STAIR_STEP_H
        platform_x = STAIR_START + (STAIR_NUM_STEPS - 1) * STAIR_STEP_D
        base = self.data.xpos[self.base_body_id]
        print(f"[sim2sim] 爬升 {climbed:+.3f}m (预期 {expected:.2f}m = "
              f"{STAIR_NUM_STEPS} 级 x {STAIR_STEP_H * 100:.0f}cm)，"
              f"最远 x={self.max_base_x:.2f}m (平台起点 x={platform_x:.2f}m)")
        if self.max_base_x > platform_x + 0.3 and climbed > expected * 0.7:
            print("[sim2sim] 判定: 上楼成功（已站上顶部平台）")
        elif climbed > STAIR_STEP_H * 0.7:
            level = int(round(climbed / STAIR_STEP_H))
            print(f"[sim2sim] 判定: 上到约第 {level} 级后未能继续（卡住或摔倒）")
        else:
            print("[sim2sim] 判定: 未能上第一级台阶")
        print(f"[sim2sim] 结束姿态: x={base[0]:.2f} y={base[1]:.2f} z={base[2]:.3f}, "
              f"yaw={np.rad2deg(flat.yaw_from_quat(self.data.qpos[self.base_qpos_adr + 3:self.base_qpos_adr + 7])):+.1f}°")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=RUN)
    parser.add_argument("--start-yaw", type=float, default=START_YAW,
                        help="初始偏航(rad)，训练 reset 范围 ±0.4")
    parser.add_argument("--speed", type=float, default=FORWARD_SPEED,
                        help="恒定前进速度(m/s)，默认取 ranges.lin_vel_x 中点")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--save-data", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit(f"请在脚本顶部 RUN 或 --run 指定 {LOGS_ROOT} 下的训练 run 目录名")
    flat._self_check()
    duration = args.duration if args.duration is not None else (20.0 if args.headless else flat.SIM_DURATION)
    config = flat.load_config(args.run)
    heading_target = _read_heading_target(config.run_dir)
    print(f"[sim2sim] run={args.run}, action_clip=±{config.policy_action_clip}, "
          f"gait_period={config.gait_period}s")
    runner = RoughStepRunner(
        config, show_viewer=not args.headless, save_data=args.save_data,
        heading_target=heading_target, start_yaw=args.start_yaw, forward_speed=args.speed,
    )
    runner.run(duration=duration, realtime=not args.headless)
    print(f"[sim2sim] 完成 {runner.data.time:.2f}s")
    runner.report()


if __name__ == "__main__":
    main()
