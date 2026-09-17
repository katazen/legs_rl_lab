"""nlegs_mimic_crouch 的 MuJoCo 策略回放，不启动 Isaac Sim、不连接实机。

conda activate unitree_lab
python source/legs_rl_lab/legs_rl_lab/tasks/mimic_task/task/nlegs_crouch/sim2sim.py --run 新训练目录
必须指定已导出策略的 --run；也接受 run 的绝对路径，不再默认加载旧限位模型。
v3 为 341 帧 / 3.4s；以所选 run 的 deploy.yaml 记录的参考为准。
站立等待时暂停物理与参考时钟，聚焦窗口按小键盘/主键盘 5 开始。
参考播放一遍后停在最后一帧，策略继续控制，不自动重置机器人。
离线测试：--headless --auto-start --duration 6；--save-data 保存 PD 目标/实际角度/力矩。
"""

import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys
from threading import Event
import time

import glfw
import mujoco
import numpy as np


TASK_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "nlegs_mimic_flat_replay", TASK_DIR.parents[2] / "nlegs_task/task/flat/sim2sim.py",
)
flat = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = flat
spec.loader.exec_module(flat)
flat.LOGS_ROOT = str(Path(flat._REPO_ROOT) / "logs/rsl_rl/nlegs_mimic_crouch")
flat.SCENE_XML = str(Path(flat.SCENE_XML).with_name("nlegs_limit_scene.xml"))
flat._OBS_FEATURES.update(motion_command="motion_command", motion_anchor_ori_b="motion_anchor_ori_b")
RUN = None


def rotation_matrix(quat_wxyz):
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, np.asarray(quat_wxyz, dtype=np.float64))
    return matrix.reshape(3, 3)


def motion_path(recorded_path):
    path = Path(recorded_path)
    if path.is_file():
        return path
    # 允许训练日志随仓库换机器；只重定位 source 下的原相对路径，不替换成另一份动作。
    relative = path.as_posix().partition("/source/")[2] if path.is_absolute() else ""
    path = Path(flat._REPO_ROOT) / ("source/" + relative if relative else path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到训练使用的参考动作: {recorded_path}")
    return path


class MimicCrouchRunner(flat.MujocoRunner):
    def __init__(self, cfg, show_viewer=True, save_data=False, base_mass_add=2.5):
        expected = {"motion_command": 24, "motion_anchor_ori_b": 6, "base_ang_vel": 3,
                    "joint_pos_rel": 12, "joint_vel_rel": 12, "last_action": 12}
        if set(cfg.commands) != {"motion"} or list(cfg.observations) != list(expected):
            raise ValueError("请选择 nlegs Mimic 模型：需要按训练顺序排列的 69D mimic 观测")
        for name, width in expected.items():
            term = cfg.observations[name]
            if len(term["scale"]) != width or term["history_length"] != 1:
                raise ValueError(f"{name} 必须为 {width}D 单帧观测")
        motion_cfg = cfg.commands["motion"]
        self.track_heading = motion_cfg.get("track_heading", True)
        if not isinstance(self.track_heading, bool):
            raise ValueError("motion.track_heading 必须是布尔值")
        self.yaw_alignment = np.eye(3)
        if motion_cfg["anchor_body_name"] != "base":
            raise ValueError("此回放器要求 motion 的 anchor_body_name=base")
        if not np.isfinite(base_mass_add) or base_mass_add < 0:
            raise ValueError("base_mass_add 必须是非负有限数")
        self.base_mass_add = base_mass_add
        self.motion_file = motion_path(motion_cfg["motion_file"])
        with np.load(self.motion_file, allow_pickle=False) as motion:
            model_file = Path(cfg.scene_xml).with_name("nlegs_limit.xml")
            if str(motion["model_sha256"]) != hashlib.sha256(model_file.read_bytes()).hexdigest():
                raise ValueError("参考动作与 nlegs_limit.xml 校验不一致，请重新生成对应模型的动作")
            self.fps = float(motion["fps"].item())
            if not np.isfinite(self.fps) or self.fps <= 0:
                raise ValueError("参考动作 fps 必须是有限正数")
            stride = self.fps * cfg.step_dt
            if stride < 1 or not np.isclose(stride, round(stride), rtol=0, atol=1e-6):
                raise ValueError("参考动作 fps 必须是策略频率的整数倍，与训练端一致")
            self.frame_stride = round(stride)
            names = motion["joint_names"].tolist()
            bodies = motion["body_names"].tolist()
            if len(names) != 12 or len(set(names)) != 12 or set(names) != set(cfg.joint_names):
                raise ValueError("参考动作必须包含策略对应的 12 个唯一关节名称")
            if len(set(bodies)) != len(bodies) or "base" not in bodies:
                raise ValueError("参考动作 body_names 必须唯一且包含 base")
            order = [names.index(cfg.joint_names[i]) for i in cfg.policy_to_sdk]
            frames = motion["joint_pos"].shape[0]
            if frames < 2 or str(motion["velocity_frame"]) != "world_link_origin":
                raise ValueError("参考动作需至少两帧，速度需为 world_link_origin")
            for field, shape in (("joint_pos", (frames, 12)), ("joint_vel", (frames, 12)),
                                 ("body_pos_w", (frames, len(bodies), 3)),
                                 ("body_quat_w", (frames, len(bodies), 4)),
                                 ("body_lin_vel_w", (frames, len(bodies), 3)),
                                 ("body_ang_vel_w", (frames, len(bodies), 3))):
                if motion[field].shape != shape or not np.isfinite(motion[field]).all():
                    raise ValueError(f"参考动作字段 {field} 形状错误或包含 NaN/Inf")
            self.ref_pos = motion["joint_pos"][:, order].astype(np.float32)
            self.ref_vel = motion["joint_vel"][:, order].astype(np.float32)
            base = bodies.index("base")
            self.ref_quat = motion["body_quat_w"][:, base].copy()
            if not np.allclose(np.linalg.norm(self.ref_quat, axis=1), 1., atol=1e-5):
                raise ValueError("参考 base 四元数必须是单位 WXYZ")
            initial_pos = motion["body_pos_w"][0, base].copy()
            initial_lin_vel = motion["body_lin_vel_w"][0, base].copy()
            initial_ang_vel = motion["body_ang_vel_w"][0, base].copy()

        self.start_requested = Event()
        self.policy_started = False
        self.end_announced = False
        super().__init__(cfg, show_viewer=False, save_data=save_data)
        limits = self.model.jnt_range[self.joint_ids[cfg.policy_to_sdk]]
        if (cfg.action_term_clip is None or cfg.action_term_clip.shape != (12, 2)
                or not np.allclose(cfg.action_term_clip, limits, atol=1e-5)):
            raise ValueError("训练导出的动作限位与 MuJoCo nlegs_limit.xml 不一致")
        if ((self.ref_pos < limits[:, 0] - 1e-5) | (self.ref_pos > limits[:, 1] + 1e-5)).any():
            raise ValueError("参考关节角超出 MuJoCo 限位")
        self.data.qpos[self.qpos_adr[cfg.policy_to_sdk]] = self.ref_pos[0]
        self.data.qvel[self.dof_adr[cfg.policy_to_sdk]] = self.ref_vel[0]
        bq, bv = self.base_qpos_adr, self.base_dof_adr
        self.data.qpos[bq:bq + 3] = initial_pos
        self.data.qpos[bq + 3:bq + 7] = self.ref_quat[0]
        self.data.qvel[bv:bv + 3] = initial_lin_vel
        self.data.qvel[bv + 3:bv + 6] = rotation_matrix(self.ref_quat[0]).T @ initial_ang_vel
        mujoco.mj_forward(self.model, self.data)
        print(f"[mimic] {self.motion_file.name}: {frames} 帧 / {self.fps:g}Hz；"
              f"策略 {1 / cfg.step_dt:g}Hz，每步推进 {self.frame_stride} 帧；base 额外质量 {base_mass_add:g}kg")
        if show_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=self._on_key)
            for name, value in flat.CAMERA.items():
                setattr(self.viewer.cam, name, value)
            self.viewer.cam.lookat[:] = self.data.xpos[self.base_body_id]
            self.viewer.sync()

    def _make_model(self, cfg):
        model = super()._make_model(cfg)
        base = model.body("base").id
        # 训练 base 加重 1–4kg；默认取中值，惯量按 Isaac 的 mass randomization 同比缩放。
        model.body_inertia[base] *= (model.body_mass[base] + self.base_mass_add) / model.body_mass[base]
        model.body_mass[base] += self.base_mass_add
        mujoco.mj_setConst(model, mujoco.MjData(model))
        return model

    @property
    def reference_frame(self):
        return min(self.episode_step * self.frame_stride, len(self.ref_pos) - 1)

    def _observation_features(self):
        features = super()._observation_features()
        frame = self.reference_frame
        quat = self.data.qpos[self.base_qpos_adr + 3:self.base_qpos_adr + 7]
        actual, reference = rotation_matrix(quat), rotation_matrix(self.ref_quat[frame])
        alignment = self.yaw_alignment
        if not self.track_heading:
            yaw = np.arctan2(actual[1, 0], actual[0, 0]) - np.arctan2(reference[1, 0], reference[0, 0])
            c, s = np.cos(yaw), np.sin(yaw)
            alignment = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        relative = actual.T @ alignment @ reference
        features["motion_command"] = np.concatenate((self.ref_pos[frame], self.ref_vel[frame]))
        features["motion_anchor_ori_b"] = relative[:, :2].reshape(-1).astype(np.float32)
        features["ang_vel"] = self.data.qvel[self.base_dof_adr + 3:self.base_dof_adr + 6].astype(np.float32)
        return features

    def _infer(self):
        if self.reference_frame == len(self.ref_pos) - 1 and not self.end_announced:
            print("[mimic] 参考动作已到末帧，继续 RL 保持目标；这不代表机器人已经到位。")
            self.end_announced = True
        return super()._infer()

    def _on_key(self, keycode):
        if keycode in (glfw.KEY_KP_5, ord("5")):
            self.start_requested.set()

    def _start_keyboard(self):
        pass  # 本任务没有行走速度命令。

    def _wait_for_start(self):
        if self.start_requested.is_set():
            return True
        if self.viewer is None:
            raise ValueError("无界面模式需要 --auto-start")
        print("[mimic] 参考首帧等待：物理、策略和参考时钟暂停。聚焦窗口按 5 开始。")
        while not self.start_requested.is_set():
            if not self.viewer.is_running():
                return False
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
            print("[mimic] RL 已接管，参考从第 0 帧开始；再次按 5 不会重置。")
        super().run(duration=duration, realtime=realtime)


def main(default_run=RUN, description=__doc__):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=default_run, required=default_run is None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--base-mass-add", type=float, default=2.5, help="base 额外质量 kg；训练范围 1–4")
    args = parser.parse_args()
    if args.headless and not args.auto_start:
        parser.error("--headless 必须同时指定 --auto-start")
    duration = args.duration if args.duration is not None else (6.0 if args.headless else flat.SIM_DURATION)
    if not np.isfinite(duration) or duration <= 0:
        parser.error("--duration 必须是有限正数")
    cfg = flat.load_config(args.run)
    if not Path(cfg.model_path).is_file():
        parser.error(f"缺少 {cfg.model_path}；请先用对应任务的 play 导出策略")
    print(f"[mimic] policy={cfg.model_path}")
    runner = MimicCrouchRunner(cfg, show_viewer=not args.headless, save_data=args.save_data,
                              base_mass_add=args.base_mass_add)
    if args.auto_start:
        runner.start_requested.set()
    runner.run(duration=duration, realtime=not args.headless)
    print(f"[mimic] 完成 {runner.data.time:.2f}s；参考帧 {runner.reference_frame}；"
          f"base_z={runner.data.qpos[runner.base_qpos_adr + 2]:.3f}m")


if __name__ == "__main__":
    main()
