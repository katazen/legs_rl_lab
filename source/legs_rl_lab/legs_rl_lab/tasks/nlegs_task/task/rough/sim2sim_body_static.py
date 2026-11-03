"""W1196 零速站立策略的 MuJoCo 回放；不启动 ROS、不连接实机。

必须用 --run 指定已训练导出的 nlegs_body_rough_static run（目录名或绝对路径）。
8/2 前后、4/6 左右、7/9 转向、0 清零命令，支持主键盘和小键盘。
0 不停策略、不覆盖关节目标，策略自行完成停步并站稳。
"""

import argparse
import importlib.util
from pathlib import Path

import glfw
import mujoco
import numpy as np


spec = importlib.util.spec_from_file_location("nlegs_body_static_replay", Path(__file__).with_name("sim2sim.py"))
rough = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rough)
flat = rough.flat
flat.LOGS_ROOT = str(Path(flat._REPO_ROOT) / "logs/rsl_rl/nlegs_body_rough_static")
SCENE_XML = str(Path(flat._REPO_ROOT) / "source/legs_rl_lab/legs_rl_lab/assets/nlegs_v2/w1196_scene.xml")


class BodyStaticRunner(rough.RoughRunner):
    def __init__(self, cfg, show_viewer=True, save_data=False):
        params = cfg.observations.get("gait_phase", {}).get("params", {})
        if not params.get("gate_by_cmd", False) or params.get("stop_transition_time", 0.0) <= 0.0:
            raise ValueError("需要新任务训练导出的渐变站立策略；不能直接回放旧踏步模型")
        super().__init__(cfg, show_viewer=False, save_data=save_data)
        if show_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=self._on_key)
            for name, value in flat.CAMERA.items():
                setattr(self.viewer.cam, name, value)
            self.viewer.cam.lookat[:] = self.data.xpos[self.base_body_id]
            self._update_markers()

    def _on_key(self, keycode):
        if glfw.KEY_KP_0 <= keycode <= glfw.KEY_KP_9:
            digit = str(keycode - glfw.KEY_KP_0)
        elif ord("0") <= keycode <= ord("9"):
            digit = chr(keycode)
        else:
            return
        if digit == "0":
            self.command[:] = 0.0
            print("[cmd] 零速：策略继续运行并收步站稳")
        elif digit in flat.KEY_BINDINGS:
            self._adjust_command(*flat.KEY_BINDINGS[digit])

    def _start_keyboard(self):
        # 只接收当前 MuJoCo 窗口按键，避免后台全局监听影响其他程序。
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="新任务导出的 run 目录名或绝对路径")
    parser.add_argument("--xml", "--scene-xml", dest="scene_xml", default=SCENE_XML)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--save-data", action="store_true")
    args = parser.parse_args()
    duration = args.duration if args.duration is not None else (10.0 if args.headless else flat.SIM_DURATION)
    if not np.isfinite(duration) or duration <= 0:
        parser.error("--duration 必须是有限正数")
    cfg = flat.load_config(args.run)
    cfg.scene_xml = str(Path(args.scene_xml).expanduser().resolve())
    if not Path(cfg.model_path).is_file():
        parser.error(f"没有 {cfg.model_path}；请先用 play.py 导出训练后的策略")
    flat._self_check()
    print(f"[body_static] run={cfg.run_dir}")
    print("[body_static] 8/2前后 4/6左右 7/9转向 0收步站稳；策略保持运行")
    runner = BodyStaticRunner(cfg, show_viewer=not args.headless, save_data=args.save_data)
    runner.run(duration=duration, realtime=not args.headless)
    print(f"[body_static] 完成 {runner.data.time:.2f}s，base_z={runner.data.xpos[runner.base_body_id, 2]:.3f}m")


if __name__ == "__main__":
    main()
