"""rough_nogait 的 MuJoCo 回放：上坡、平台、下楼梯，不使用相位观测。

按文件路径运行，避免包导入启动 Isaac Lab：
python source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/task/rough/sim2sim_nogait.py
--run RUN 支持目录名或绝对路径；先用 play 导出 exported/policy.pt。
--headless --duration 10 --cmd 0.2 0 0 可离线回放，--save-data 保存跟踪 CSV。
键盘和可视化沿用 rough：8/2 前后、4/6 左右、7/9 转向。
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np


# 按文件路径复用地形、动力学、控制和历史观测，不导入 Isaac Lab。
spec = importlib.util.spec_from_file_location("rough_nogait_replay", Path(__file__).with_name("sim2sim.py"))
rough = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rough)
flat = rough.flat

# 默认回放 run；更换模型可修改此处或传 --run。
RUN = "2026-10-06_18-20-58"
LOGS_ROOT = str(Path(flat._REPO_ROOT) / "logs/rsl_rl/rough_nogait")
DROP_HEIGHT = rough.DROP_HEIGHT
flat.LOGS_ROOT = LOGS_ROOT


def load_config(run):
    cfg = flat.load_config(run)
    if "gait_phase" in cfg.observations:
        raise ValueError("rough_nogait 需要无相位策略，请选择 rough_nogait 的导出模型")
    if set(cfg.commands) != {"base_velocity"}:
        raise ValueError("rough_nogait 需要 base_velocity 行走命令")
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=RUN)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--cmd", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("VX", "VY", "YAW"))
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--check-terrain", action="store_true")
    args = parser.parse_args()
    if args.check_terrain:
        rough._check_terrain()
        return
    duration = args.duration if args.duration is not None else (10.0 if args.headless else flat.SIM_DURATION)
    if not np.isfinite(duration) or duration <= 0:
        parser.error("--duration 必须是有限正数")
    if not np.isfinite(args.cmd).all():
        parser.error("--cmd 必须为三个有限数值")
    cfg = load_config(args.run)
    if not Path(cfg.model_path).is_file():
        parser.error(f"没有 {cfg.model_path}；请先用 play 导出该 run 的策略")
    flat._self_check()
    cfg.base_pos[2] = DROP_HEIGHT
    print(f"[rough_nogait] run={cfg.run_dir}，无相位，出生高度 {DROP_HEIGHT:.3f}m")
    runner = rough.RoughRunner(cfg, show_viewer=not args.headless, save_data=args.save_data)
    runner.command[:] = np.clip(args.cmd, cfg.command_ranges[:, 0], cfg.command_ranges[:, 1])
    print(f"[rough_nogait] 初始速度 vx/vy/yaw={runner.command.tolist()}，指令按训练范围截断")
    runner.run(duration=duration, realtime=not args.headless)
    print(f"[rough_nogait] 完成 {runner.data.time:.2f}s，base_z={runner.data.xpos[runner.base_body_id, 2]:.3f}m")


if __name__ == "__main__":
    main()
