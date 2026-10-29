"""Mimic reference and validation; NumPy only, no ROS, Isaac or MuJoCo runtime."""

import hashlib
from pathlib import Path

import numpy as np


OBSERVATIONS = {"motion_command": 24, "motion_anchor_ori_b": 6, "base_ang_vel": 3,
                "joint_pos_rel": 12, "joint_vel_rel": 12, "last_action": 12}


def rotation_matrix(quat):
    q = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(q)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(norm - 1.) > .1:
        raise ValueError("IMU 四元数必须为有效的单位 WXYZ")
    w, x, y, z = q / norm
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                     [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                     [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]])


def resolve_recorded_path(recorded, root):
    path = Path(recorded)
    if not path.is_absolute():
        path = root / path
    if path.is_file():
        return path
    relative = path.as_posix().partition("/source/")[2]
    candidate = root / "source" / relative
    if not relative or not candidate.is_file():
        raise FileNotFoundError(f"找不到训练资源: {recorded}")
    return candidate


class MotionReference:
    def __init__(self, deploy, root):
        if deploy.get("format_version") != 2 or deploy.get("task_type") != "reference_motion_tracking":
            raise ValueError("mimic 需要 format_version=2 的 reference_motion_tracking 导出")
        if set(deploy["commands"]) != {"motion"} or list(deploy["observations"]) != list(OBSERVATIONS):
            raise ValueError("mimic 需要训练顺序的 69D 观测与唯一 motion 命令")
        sdk = deploy["joint_names"]
        indexes = deploy["joint_ids_map"]
        if len(sdk) != 12 or len(set(sdk)) != 12 or sorted(indexes) != list(range(12)):
            raise ValueError("joint_ids_map 必须是 12 关节完整排列")
        self.joint_names = [sdk[i] for i in indexes]
        self.step_dt = float(deploy["step_dt"])
        config = deploy["commands"]["motion"]
        self.track_heading = config.get("track_heading", True)
        if config["anchor_body_name"] != "base":
            raise ValueError("mimic 仅支持 base 参考姿态")
        model_file = resolve_recorded_path(config["model_file"], root)
        self.path = resolve_recorded_path(config["motion_file"], root)
        with np.load(self.path, allow_pickle=False) as data:
            if str(data["model_sha256"]) != hashlib.sha256(model_file.read_bytes()).hexdigest():
                raise ValueError("参考动作与模型校验不一致")
            self.fps = float(data["fps"].item())
            stride = self.fps * self.step_dt
            if stride < 1 or not np.isclose(stride, round(stride), rtol=0, atol=1e-6):
                raise ValueError("参考帧率必须为策略频率的整数倍")
            self.stride = round(stride)
            names, bodies = data["joint_names"].tolist(), data["body_names"].tolist()
            if len(names) != 12 or set(names) != set(self.joint_names) or len(set(bodies)) != len(bodies):
                raise ValueError("参考动作名称重复或与策略不匹配")
            frames = data["joint_pos"].shape[0]
            for key, shape in (("joint_pos", (frames, 12)), ("joint_vel", (frames, 12)),
                               ("body_quat_w", (frames, len(bodies), 4))):
                if data[key].shape != shape or not np.isfinite(data[key]).all():
                    raise ValueError(f"参考动作字段无效: {key}")
            order = [names.index(n) for n in self.joint_names]
            self.positions = data["joint_pos"][:, order].astype(np.float32)
            self.velocities = data["joint_vel"][:, order].astype(np.float32)
            quats = data["body_quat_w"][:, bodies.index("base")]
            self.rotations = np.array([rotation_matrix(q) for q in quats])
            self.quaternions = quats.copy()
        if not np.allclose(self.velocities[-1], 0., atol=1e-5):
            raise ValueError("末帧保持要求参考末帧关节速度为零")
        self.steps = 0
        self.yaw_alignment = np.eye(3)

    @property
    def frame(self):
        return min(self.steps * self.stride, len(self.positions) - 1)

    def reset(self, actual_quat):
        actual = rotation_matrix(actual_quat)
        reference = self.rotations[0]
        yaw = np.arctan2(actual[1, 0], actual[0, 0]) - np.arctan2(reference[1, 0], reference[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        self.yaw_alignment = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
        self.steps = 0

    def features(self, actual_quat):
        actual, reference = rotation_matrix(actual_quat), self.rotations[self.frame]
        alignment = self.yaw_alignment
        if not self.track_heading:
            yaw = np.arctan2(actual[1, 0], actual[0, 0]) - np.arctan2(reference[1, 0], reference[0, 0])
            c, s = np.cos(yaw), np.sin(yaw)
            alignment = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        relative = actual.T @ alignment @ reference
        return {"motion_command": np.r_[self.positions[self.frame], self.velocities[self.frame]],
                "motion_anchor_ori_b": relative[:, :2].reshape(-1).astype(np.float32)}
