"""flat_crouch 的限位下蹲起步回放；按文件路径运行，不需要启动 Isaac Sim。

python source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/task/flat/sim2sim_crouch.py --run RUN
等待阶段暂停物理与策略；聚焦 MuJoCo 窗口，按小键盘 5 开始。
8/2、4/6、7/9 调前后、左右、转向速度，与 flat 一致；主键盘数字也可使用。
--pose-only 仅计算并打印姿态；无界面回放使用 --headless --auto-start --duration 10。
"""

import argparse
import importlib.util
from pathlib import Path
import sys
from threading import Event
import time

import glfw
import mujoco
import numpy as np


def _load_file(name, path):
    # 包导入会加载 Isaac Lab；沿用 rough 回放器的文件导入方式。
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


flat = _load_file("nlegs_crouch_flat_replay", Path(__file__).with_name("sim2sim.py"))
pose_generator = _load_file(
    "nlegs_crouch_pose_generator", Path(flat._REPO_ROOT) / "scripts/generate_crouch_pose_bank.py"
)

RUN = "2026-09-09_20-27-00"
# RUN = "2026-08-26_13-57-34"
LOGS_ROOT = Path(flat._REPO_ROOT) / "logs/rsl_rl/nlegs_flat_crouch"
flat.LOGS_ROOT = str(LOGS_ROOT)
flat.SCENE_XML = str(pose_generator.ASSET / "mjcf/nlegs_limit_scene.xml")


def print_pose(q, state):
    angles = pose_generator.Rotation.from_quat(state["quat"], scalar_first=True).as_euler("xyz", degrees=True)
    print(f"[crouch] base roll/pitch/yaw = {angles.round(4).tolist()} deg; base_z={state['height']:.6f} m")
    print(f"[crouch] quat WXYZ = {state['quat'].round(8).tolist()}")
    for side in ("L", "R"):
        values = [q[pose_generator.NAMES.index(f"joint_{side}{i}")] for i in range(1, 7)]
        print(f"[crouch] {side}1..{side}6 = {np.round(values, 6).tolist()} rad")
    print(f"[crouch] 最大脚底倾角={state['tilt_deg']:.4f} deg, 足底高度差={state['height_error'] * 1000:.4f} mm, "
          f"出生间隙={pose_generator.CLEARANCE * 1000:.1f} mm")


class CrouchRunner(flat.MujocoRunner):
    def __init__(self, cfg, show_viewer=True, save_data=False):
        self.start_requested = Event()
        self.policy_started = False
        super().__init__(cfg, show_viewer=False, save_data=save_data)
        solver = pose_generator.PoseSolver()
        q, state = solver.solve_limit()
        if set(cfg.joint_names) != set(pose_generator.NAMES):
            raise ValueError("策略关节名称与限位模型不一致")
        order = [pose_generator.NAMES.index(name) for name in cfg.joint_names]
        if not np.allclose(self.model.jnt_range[self.joint_ids], solver.limits[order], rtol=0, atol=1e-6):
            raise ValueError("回放场景的关节限位与 nlegs_limit.xml 不一致")
        self.data.qpos[self.qpos_adr] = q[order]
        self.data.qpos[self.base_qpos_adr:self.base_qpos_adr + 3] = [0, 0, state["height"]]
        self.data.qpos[self.base_qpos_adr + 3:self.base_qpos_adr + 7] = state["quat"]
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        print_pose(q, state)
        if show_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=self._on_key)
            self.viewer.cam.distance = flat.CAMERA["distance"]
            self.viewer.cam.elevation = flat.CAMERA["elevation"]
            self.viewer.cam.azimuth = flat.CAMERA["azimuth"]
            self.viewer.cam.lookat[:] = self.data.xpos[self.base_body_id]
            self._update_markers()
            self.viewer.sync()

    def _on_key(self, keycode):
        if glfw.KEY_KP_0 <= keycode <= glfw.KEY_KP_9:
            digit = str(keycode - glfw.KEY_KP_0)
        elif ord("0") <= keycode <= ord("9"):
            digit = chr(keycode)
        else:
            return
        if digit == "5":
            self.start_requested.set()
        elif digit in flat.KEY_BINDINGS:
            self._adjust_command(*flat.KEY_BINDINGS[digit])

    def _start_keyboard(self):
        # 使用当前窗口回调，避免与 flat 的全局 pynput 监听重复处理。
        pass

    def _wait_for_start(self):
        if self.start_requested.is_set():
            return True
        if self.viewer is None:
            raise ValueError("无界面模式需要 --auto-start")
        print("[crouch] 物理与策略已暂停。聚焦 MuJoCo 窗口，按小键盘 5 开始 RL 控制。")
        while not self.start_requested.is_set():
            if not self.viewer.is_running():
                return False
            self._update_markers()
            self.viewer.sync()
            time.sleep(self.cfg.step_dt)
        return self.viewer.is_running()

    def run(self, duration=flat.SIM_DURATION, realtime=True):
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("duration 必须是有限正数")
        if not self.policy_started:
            try:
                ready = self._wait_for_start()
            except BaseException:
                if self.viewer is not None:
                    self.viewer.close()
                raise
            if not ready:
                self.viewer.close()
                return
            self.policy_started = True
            print("[crouch] RL 已启动：从当前下蹲状态直接接管，步态相位从 0 开始。")
        super().run(duration=duration, realtime=realtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=RUN)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--auto-start", action="store_true", help="跳过按键等待，显式启动 RL（自动化测试用）")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--pose-only", action="store_true", help="只计算姿态，不加载策略或启动窗口")
    args = parser.parse_args()
    if args.pose_only:
        print_pose(*pose_generator.PoseSolver().solve_limit())
        return
    if args.headless and not args.auto_start:
        parser.error("--headless 必须同时指定 --auto-start；仅检查姿态请用 --pose-only")
    duration = args.duration if args.duration is not None else (10.0 if args.headless else flat.SIM_DURATION)
    if not np.isfinite(duration) or duration <= 0:
        parser.error("--duration 必须是有限正数")
    config = flat.load_config(args.run)
    if not Path(config.model_path).is_file():
        parser.error(f"没有导出的策略 {config.model_path}；请先运行 scripts/rsl_rl/play.py "
                     f"--task nlegs_flat_crouch --load_run {Path(config.run_dir).name} --num_envs 1，导出后再回放")
    flat._self_check()
    print(f"[sim2sim_crouch] run={config.run_dir}, policy={config.model_path}")
    runner = CrouchRunner(config, show_viewer=not args.headless, save_data=args.save_data)
    if args.auto_start:
        runner.start_requested.set()
    runner.run(duration=duration, realtime=not args.headless)
    print(f"[sim2sim_crouch] 完成 {runner.data.time:.2f}s，base_z={runner.data.xpos[runner.base_body_id, 2]:.3f}m")


if __name__ == "__main__":
    main()
