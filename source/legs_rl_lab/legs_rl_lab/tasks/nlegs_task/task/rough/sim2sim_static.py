"""rough_static 的 MuJoCo 回放：零速站稳，有指令行走，不连接实机。

python source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/task/rough/sim2sim_static.py
--run RUN 可选择其他已导出的 run，也接受绝对路径。
启动即运行策略；聚焦窗口，8/2 前后、4/6 左右、7/9 转向、0 清零速度。
支持主键盘和小键盘。地形沿用 rough：上坡 → 30cm 平台 → 下楼梯。
离线检查：--headless --duration 10；--save-data 保存关节跟踪数据。
"""

import argparse
import importlib.util
from pathlib import Path

import glfw
import mujoco
import numpy as np


# 按路径复用回放器，避免包导入启动 Isaac Lab。
spec = importlib.util.spec_from_file_location("nlegs_rough_static_replay", Path(__file__).with_name("sim2sim.py"))
rough = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rough)
flat = rough.flat
RUN = "2026-09-18_18-36-41"
flat.LOGS_ROOT = str(Path(flat._REPO_ROOT) / "logs/rsl_rl/nlegs_rough_static")
flat.SCENE_XML = str(Path(flat.SCENE_XML).with_name("nlegs_limit_scene.xml"))


class RoughStaticRunner(rough.RoughRunner):
    def __init__(self, cfg, show_viewer=True, save_data=False):
        super().__init__(cfg, show_viewer=False, save_data=save_data)
        if show_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=self._on_key)
            for name, value in flat.CAMERA.items():
                setattr(self.viewer.cam, name, value)
            self.viewer.cam.lookat[:] = self.data.xpos[self.base_body_id]
            self._update_markers()

    def _make_model(self, cfg):
        phase = cfg.observations.get("gait_phase", {}).get("params", {})
        if not phase.get("gate_by_cmd", False):
            raise ValueError("该 run 没有零速相位门控，请选择 rough_static 的导出模型")
        clip = cfg.action_term_clip
        if (clip is None or clip.shape != (len(cfg.joint_names), 2)
                or np.isnan(clip).any() or (clip[:, 0] >= clip[:, 1]).any()
                or not np.isfinite(clip).any()):
            raise ValueError("rough_static 需要有效的单侧关节限位 clip，请重新导出 deploy.yaml")
        model = super()._make_model(cfg)
        # clip 是 policy 顺序；按名字写入模型，只覆盖有限端，另一端保留 XML 原值。
        for policy_id, sdk_id in enumerate(cfg.policy_to_sdk):
            name = cfg.joint_names[sdk_id]
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"MuJoCo 场景缺少转动关节 {name}")
            ends = np.isfinite(clip[policy_id])
            model.jnt_range[joint, ends] = clip[policy_id, ends]
            if ends.any():
                model.jnt_limited[joint] = True
            lo, hi = model.jnt_range[joint]
            if not lo < hi or not lo <= cfg.default_sdk[sdk_id] <= hi:
                raise ValueError(f"{name} 限位 [{lo}, {hi}] 无效或不包含默认站姿")
        print("[rough_static] 已同步目标角与物理单端限位；XML/USD 不改写")
        return model

    def _on_key(self, keycode):
        if glfw.KEY_KP_0 <= keycode <= glfw.KEY_KP_9:
            digit = str(keycode - glfw.KEY_KP_0)
        elif ord("0") <= keycode <= ord("9"):
            digit = chr(keycode)
        else:
            return
        if digit == "0":
            self.command[:] = 0
            print("[cmd] 零速站稳")
        elif digit in flat.KEY_BINDINGS:
            self._adjust_command(*flat.KEY_BINDINGS[digit])

    def _start_keyboard(self):
        # 只响应当前 MuJoCo 窗口，禁用父类的全局键盘监听。
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=RUN)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--check-terrain", action="store_true")
    args = parser.parse_args()
    if args.check_terrain:
        rough._check_terrain()
        return
    duration = args.duration if args.duration is not None else (10.0 if args.headless else flat.SIM_DURATION)
    if not np.isfinite(duration) or duration <= 0:
        parser.error("--duration 必须是有限正数")
    config = flat.load_config(args.run)
    if not Path(config.model_path).is_file():
        parser.error(f"没有 {config.model_path}；请先用 play 导出该 run 的策略")
    flat._self_check()
    print(f"[rough_static] run={config.run_dir}")
    print("[rough_static] 策略已启动：8/2前后 4/6左右 7/9转向 0站稳")
    runner = RoughStaticRunner(config, show_viewer=not args.headless, save_data=args.save_data)
    runner.run(duration=duration, realtime=not args.headless)
    print(f"[rough_static] 完成 {runner.data.time:.2f}s，base_z={runner.data.xpos[runner.base_body_id, 2]:.3f}m")


if __name__ == "__main__":
    main()
