"""nlegs_body 策略的 MuJoCo sim2sim。只需修改 RUN 后运行本文件。"""

import os
import time
import types
from collections import deque
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml

try:
    from pynput import keyboard
except ImportError:
    keyboard = None


def _find_repo_root(start):
    path = os.path.dirname(os.path.abspath(start))
    while path != os.path.dirname(path):
        if os.path.isdir(os.path.join(path, "source")) and os.path.isdir(os.path.join(path, "scripts")):
            return path
        path = os.path.dirname(path)
    raise RuntimeError("找不到 legs_rl_lab 仓库根")


_REPO_ROOT = _find_repo_root(__file__)
RUN = "2026-08-24_16-52-23"

LOGS_ROOT = os.path.join(_REPO_ROOT, "logs", "rsl_rl", "nlegs_body")
SCENE_XML = os.path.join(
    _REPO_ROOT, "source", "legs_rl_lab", "legs_rl_lab", "assets", "nlegs", "mjcf", "nlegs_body_scene.xml"
)
SIM_DURATION = 10000.0
ZERO_OFFSET = {}

_OBS_FEATURES = {
    "base_ang_vel": "ang_vel",
    "projected_gravity": "gravity",
    "velocity_commands": "command",
    "joint_pos_rel": "dof_pos",
    "joint_vel_rel": "dof_vel",
    "last_action": "last_action",
    "gait_phase": "gait",
}


def _as_vector(value, size, name):
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.size == 1:
        vector = np.full(size, vector.item(), dtype=np.float32)
    if vector.size != size:
        raise ValueError(f"{name} 数量 {vector.size} != {size}")
    return vector


@dataclass
class ActuatorGroup:
    name: str
    joint_ids: np.ndarray
    torque_model: str
    effort_limit: np.ndarray
    effort_limit_sim: np.ndarray
    velocity_limit: np.ndarray
    velocity_limit_sim: np.ndarray
    saturation_effort: np.ndarray | None
    delay: tuple[int, int]


def build_cfg(run):
    run_dir = os.path.join(LOGS_ROOT, run)
    deploy_path = os.path.join(run_dir, "params", "deploy.yaml")
    with open(deploy_path) as file:
        deploy = yaml.safe_load(file)

    required = {
        "format_version", "joint_names", "joint_ids_map", "physics_dt", "step_dt",
        "policy_action_clip", "stiffness", "damping", "default_joint_pos", "base_init_state",
        "armature", "friction", "dynamic_friction", "viscous_friction", "actuators",
        "gait_period", "actions", "observations", "commands",
    }
    missing = sorted(required - set(deploy))
    if missing:
        raise KeyError(f"{deploy_path} 缺字段 {missing}；请用新版导出器重新生成")
    if deploy["format_version"] != 2:
        raise ValueError(f"不支持 deploy.yaml format_version={deploy['format_version']}")

    joint_names = list(deploy["joint_names"])
    short_joint_names = [name.removeprefix("joint_") for name in joint_names]
    action_dim = len(joint_names)
    policy_to_sdk = np.asarray(deploy["joint_ids_map"], dtype=np.int32)
    if sorted(policy_to_sdk.tolist()) != list(range(action_dim)):
        raise ValueError("joint_ids_map 必须是 policy->SDK 的完整排列")

    default_policy = _as_vector(deploy["default_joint_pos"], action_dim, "default_joint_pos")
    default_sdk = np.empty(action_dim, dtype=np.float32)
    default_sdk[policy_to_sdk] = default_policy
    action_cfg = deploy["actions"].get("JointPositionAction")
    if action_cfg is None or len(deploy["actions"]) != 1:
        raise ValueError("sim2sim 当前要求唯一的 JointPositionAction")
    if action_cfg.get("joint_ids") is not None:
        raise ValueError("sim2sim 当前要求 JointPositionAction 覆盖全部关节")

    physics_dt = float(deploy["physics_dt"])
    step_dt = float(deploy["step_dt"])
    decimation = round(step_dt / physics_dt)
    if decimation < 1 or not np.isclose(decimation * physics_dt, step_dt):
        raise ValueError(f"step_dt={step_dt} 不是 physics_dt={physics_dt} 的整数倍")

    observations = deploy["observations"]
    unsupported = sorted(set(observations) - set(_OBS_FEATURES))
    if unsupported:
        raise KeyError(f"sim2sim 尚未实现观测项: {unsupported}")
    command_ranges = deploy["commands"]["base_velocity"]["ranges"]

    actuators = []
    covered = []
    for name, group in deploy["actuators"].items():
        joint_ids = np.asarray(group["joint_ids"], dtype=np.int32)
        size = joint_ids.size
        torque_model = group["torque_model"]
        if torque_model not in ("ideal_pd", "dc_motor"):
            raise ValueError(f"不支持执行器力矩模型 {torque_model}")
        saturation = group.get("saturation_effort")
        actuators.append(ActuatorGroup(
            name=name,
            joint_ids=joint_ids,
            torque_model=torque_model,
            effort_limit=_as_vector(group["effort_limit"], size, f"{name}.effort_limit"),
            effort_limit_sim=_as_vector(group["effort_limit_sim"], size, f"{name}.effort_limit_sim"),
            velocity_limit=_as_vector(group["velocity_limit"], size, f"{name}.velocity_limit"),
            velocity_limit_sim=_as_vector(group["velocity_limit_sim"], size, f"{name}.velocity_limit_sim"),
            saturation_effort=(
                None if saturation is None else _as_vector(saturation, size, f"{name}.saturation_effort")
            ),
            delay=tuple(int(value) for value in group["delay"]),
        ))
        covered.extend(joint_ids.tolist())
    if sorted(covered) != list(range(action_dim)):
        raise ValueError("actuators.*.joint_ids 必须无重叠地覆盖全部 SDK 关节")

    base = deploy["base_init_state"]
    action_term_clip = action_cfg.get("clip")
    return types.SimpleNamespace(
        run_dir=run_dir,
        model_path=os.path.join(run_dir, "exported", "policy.pt"),
        scene_xml=SCENE_XML,
        physics_dt=physics_dt,
        step_dt=step_dt,
        decimation=decimation,
        action_dim=action_dim,
        joint_names=joint_names,
        short_joint_names=short_joint_names,
        policy_to_sdk=policy_to_sdk,
        default_policy=default_policy,
        default_sdk=default_sdk,
        base_pos=_as_vector(base["pos"], 3, "base_init_state.pos"),
        base_quat=_as_vector(base["rot"], 4, "base_init_state.rot"),
        stiffness=_as_vector(deploy["stiffness"], action_dim, "stiffness"),
        damping=_as_vector(deploy["damping"], action_dim, "damping"),
        armature=_as_vector(deploy["armature"], action_dim, "armature"),
        friction=_as_vector(deploy["friction"], action_dim, "friction"),
        dynamic_friction=_as_vector(deploy["dynamic_friction"], action_dim, "dynamic_friction"),
        viscous_friction=_as_vector(deploy["viscous_friction"], action_dim, "viscous_friction"),
        action_scale=_as_vector(action_cfg["scale"], action_dim, "action scale"),
        action_offset=_as_vector(action_cfg["offset"], action_dim, "action offset"),
        action_term_clip=(
            None if action_term_clip is None else np.asarray(action_term_clip, dtype=np.float32)
        ),
        policy_action_clip=(
            None if deploy["policy_action_clip"] is None else float(deploy["policy_action_clip"])
        ),
        observations=observations,
        gait_period=float(deploy["gait_period"]),
        command_ranges=np.asarray([
            command_ranges["lin_vel_x"], command_ranges["lin_vel_y"], command_ranges["ang_vel_z"]
        ], dtype=np.float32),
        actuators=actuators,
    )


class TermGroupedHistory:
    def __init__(self, observations):
        self.buffers = {
            name: np.zeros((int(term["history_length"]), len(term["scale"])), dtype=np.float32)
            for name, term in observations.items()
        }
        self.ready = False

    def update(self, terms):
        for name, buffer in self.buffers.items():
            if not self.ready:
                buffer[:] = terms[name]
            else:
                buffer[:-1] = buffer[1:]
                buffer[-1] = terms[name]
        self.ready = True
        return np.concatenate([buffer.reshape(-1) for buffer in self.buffers.values()])


class ActuatorLatency:
    """每次 reset 为每个 Isaac actuator group 各采一次固定延迟。"""

    def __init__(self, groups, action_dim, announce=True):
        self.groups = groups
        self.announce = announce
        self.delay = np.zeros(action_dim, dtype=np.int32)
        self.buffer = deque(maxlen=max(group.delay[1] for group in groups) + 1)
        self.reset()

    def reset(self):
        self.buffer.clear()
        labels = []
        for group in self.groups:
            low, high = group.delay
            sampled = np.random.randint(low, high + 1)
            self.delay[group.joint_ids] = sampled
            labels.append(f"{group.name}={sampled}")
        if self.announce:
            print(f"[Latency] {', '.join(labels)} physics steps")

    def process(self, target):
        self.buffer.append(np.asarray(target, dtype=np.float32).copy())
        result = target.copy()
        for delay in np.unique(self.delay):
            valid_delay = min(int(delay), len(self.buffer) - 1)
            mask = self.delay == delay
            result[mask] = self.buffer[-1 - valid_delay][mask]
        return result


def gravity_from_quat(q):
    w, x, y, z = q
    return np.array([
        2.0 * (-z * x + w * y),
        -2.0 * (z * y + w * x),
        1.0 - 2.0 * (w * w + z * z),
    ], dtype=np.float32)


class MujocoRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        if not os.path.isfile(cfg.model_path):
            raise FileNotFoundError(f"没有导出的策略 {cfg.model_path}；先运行 play 导出 policy.pt")
        self.model = mujoco.MjModel.from_xml_path(cfg.scene_xml)
        self.model.opt.timestep = cfg.physics_dt
        self.data = mujoco.MjData(self.model)
        self.policy = torch.jit.load(cfg.model_path, map_location="cpu").eval()
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.joint_ids = np.array([
            self._id(mujoco.mjtObj.mjOBJ_JOINT, f"joint_{name}") for name in cfg.short_joint_names
        ])
        self.qpos_adr = self.model.jnt_qposadr[self.joint_ids]
        self.dof_adr = self.model.jnt_dofadr[self.joint_ids]
        self.actuator_ids = np.array([
            self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"joint_{name}_servo") for name in cfg.short_joint_names
        ])
        free_joints = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if free_joints.size != 1:
            raise ValueError("MuJoCo 模型必须恰好有一个 freejoint")
        self.base_joint_id = free_joints[0]
        self.base_body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "base")
        self._id(mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat")
        self._id(mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")

        effort_limit_sim = np.empty(cfg.action_dim, dtype=np.float32)
        for group in cfg.actuators:
            effort_limit_sim[group.joint_ids] = group.effort_limit_sim
        self.effort_limit_sim = effort_limit_sim
        self.zero_offset = np.array([
            ZERO_OFFSET.get(name, 0.0) for name in cfg.short_joint_names
        ], dtype=np.float32)
        self.model.actuator_gainprm[self.actuator_ids, 0] = 1.0
        self.model.actuator_biasprm[self.actuator_ids, 1] = 0.0
        self.model.actuator_forcelimited[self.actuator_ids] = 1
        self.model.actuator_forcerange[self.actuator_ids] = np.column_stack(
            (-effort_limit_sim, effort_limit_sim)
        )
        self.model.dof_armature[self.dof_adr] = cfg.armature
        self.model.dof_damping[self.dof_adr] = cfg.viscous_friction
        frictionloss = np.where(cfg.dynamic_friction > 0.0, cfg.dynamic_friction, cfg.friction)
        self.model.dof_frictionloss[self.dof_adr] = frictionloss
        if not np.allclose(cfg.friction, cfg.dynamic_friction):
            print("[sim2sim] MuJoCo 仅有一个 dry friction；优先使用 dynamic_friction，零值回退 friction")

        mujoco.mj_resetData(self.model, self.data)
        base_qpos = self.model.jnt_qposadr[self.base_joint_id]
        self.data.qpos[base_qpos:base_qpos + 3] = cfg.base_pos
        self.data.qpos[base_qpos + 3:base_qpos + 7] = cfg.base_quat
        self.data.qpos[self.qpos_adr] = cfg.default_sdk
        mujoco.mj_forward(self.model, self.data)

        self.latency = ActuatorLatency(cfg.actuators, cfg.action_dim)
        self.history = TermGroupedHistory(cfg.observations)
        self.last_action = np.zeros(cfg.action_dim, dtype=np.float32)
        self.command = np.zeros(3, dtype=np.float32)
        self.episode_step = 0
        self.listener = None
        self.viewer.cam.distance = 3.0
        self.viewer.cam.elevation = -20
        self.viewer.cam.azimuth = 90

    def _id(self, object_type, name):
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"nlegs_body_scene.xml 缺少 {name}")
        return object_id

    def _joint_state(self):
        q_sdk = self.data.qpos[self.qpos_adr].astype(np.float32) - self.zero_offset
        qd_sdk = self.data.qvel[self.dof_adr].astype(np.float32)
        return q_sdk, qd_sdk

    def _observation_terms(self):
        q_sdk, qd_sdk = self._joint_state()
        q_policy = q_sdk[self.cfg.policy_to_sdk]
        qd_policy = qd_sdk[self.cfg.policy_to_sdk]
        phase = self.episode_step * self.cfg.step_dt / self.cfg.gait_period
        gait_params = self.cfg.observations.get("gait_phase", {}).get("params", {})
        if gait_params.get("gate_by_cmd", False) and np.linalg.norm(self.command) < 1e-6:
            phase = 0.0
        features = {
            "ang_vel": self.data.sensor("imu_gyro").data.astype(np.float32),
            "gravity": gravity_from_quat(self.data.sensor("imu_quat").data),
            "command": self.command,
            "dof_pos": q_policy - self.cfg.default_policy,
            "dof_vel": qd_policy,
            "last_action": self.last_action,
            "gait": np.array([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)], dtype=np.float32),
        }
        terms = {}
        for name, term_cfg in self.cfg.observations.items():
            value = features[_OBS_FEATURES[name]]
            if term_cfg.get("clip") is not None:
                value = np.clip(value, *term_cfg["clip"])
            terms[name] = value * np.asarray(term_cfg["scale"], dtype=np.float32)
        return terms

    def _target_sdk(self, action):
        target_policy = action * self.cfg.action_scale + self.cfg.action_offset
        if self.cfg.action_term_clip is not None:
            target_policy = np.clip(
                target_policy,
                self.cfg.action_term_clip[:, 0],
                self.cfg.action_term_clip[:, 1],
            )
        target_sdk = np.empty(self.cfg.action_dim, dtype=np.float32)
        target_sdk[self.cfg.policy_to_sdk] = target_policy
        return target_sdk

    def _torque(self, target):
        q_sdk, qd_sdk = self._joint_state()
        torque = self.cfg.stiffness * (target - q_sdk) - self.cfg.damping * qd_sdk
        for group in self.cfg.actuators:
            ids = group.joint_ids
            if group.torque_model == "ideal_pd":
                torque[ids] = np.clip(torque[ids], -group.effort_limit, group.effort_limit)
                continue
            velocity = qd_sdk[ids]
            saturation = group.saturation_effort
            velocity_at_limit = group.velocity_limit * (1.0 + group.effort_limit / saturation)
            velocity = np.clip(velocity, -velocity_at_limit, velocity_at_limit)
            torque_max = np.minimum(
                saturation * (1.0 - velocity / group.velocity_limit), group.effort_limit
            )
            torque_min = np.maximum(
                saturation * (-1.0 - velocity / group.velocity_limit), -group.effort_limit
            )
            torque[ids] = np.clip(torque[ids], torque_min, torque_max)
        return np.clip(torque, -self.effort_limit_sim, self.effort_limit_sim).astype(np.float32)

    def _adjust_command(self, index, increment):
        self.command[index] = np.clip(
            self.command[index] + increment,
            self.cfg.command_ranges[index, 0],
            self.cfg.command_ranges[index, 1],
        )
        print(f"[cmd] vx={self.command[0]:+.2f} vy={self.command[1]:+.2f} yaw={self.command[2]:+.2f}")

    def _draw_arrows(self):
        """世界原点 XYZ（红绿蓝），机身上方指令/实际速度（绿/蓝）。"""
        scn = self.viewer.user_scn
        origin = np.zeros(3)
        arrows = [
            (origin, np.array([0.35, 0.0, 0.0]), 0.012, [0.9, 0.1, 0.1, 1.0]),
            (origin, np.array([0.0, 0.35, 0.0]), 0.012, [0.1, 0.9, 0.1, 1.0]),
            (origin, np.array([0.0, 0.0, 0.35]), 0.012, [0.1, 0.4, 1.0, 1.0]),
        ]

        base = self.data.xpos[self.base_body_id]
        anchor = base + np.array([0.0, 0.0, 0.35])
        w, x, y, z = self.data.xquat[self.base_body_id]
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        c, s = np.cos(yaw), np.sin(yaw)
        vx, vy = self.command[:2]
        cmd_world = np.array([c * vx - s * vy, s * vx + c * vy, 0.0])
        base_dof = self.model.jnt_dofadr[self.base_joint_id]
        actual_world = np.r_[self.data.qvel[base_dof:base_dof + 2], 0.0]
        for vector, color in ((cmd_world, [0.1, 0.9, 0.1, 1.0]),
                              (actual_world, [0.1, 0.4, 1.0, 1.0])):
            if np.linalg.norm(vector) >= 1e-3:
                arrows.append((anchor, anchor + vector, 0.02, color))

        for geom, (start, end, width, color) in zip(scn.geoms, arrows):
            mujoco.mjv_initGeom(
                geom, mujoco.mjtGeom.mjGEOM_ARROW,
                np.zeros(3), np.zeros(3), np.zeros(9), np.asarray(color, dtype=np.float32),
            )
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, width, start, end)
        scn.ngeom = len(arrows)

    def _start_keyboard(self):
        if keyboard is None:
            print("[sim2sim] 未安装 pynput，键盘调速不可用")
            return

        def on_press(key):
            try:
                bindings = {
                    "8": (0, 0.05), "2": (0, -0.05),
                    "4": (1, -0.05), "6": (1, 0.05),
                    "7": (2, 0.05), "9": (2, -0.05),
                }
                if key.char in bindings:
                    self._adjust_command(*bindings[key.char])
            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press)
        self.listener.start()

    def run(self):
        self._start_keyboard()
        try:
            while self.viewer.is_running() and self.data.time < SIM_DURATION:
                started = time.monotonic()
                obs = self.history.update(self._observation_terms())
                with torch.no_grad():
                    action = self.policy(torch.as_tensor(obs, dtype=torch.float32)).cpu().numpy().reshape(-1)
                if action.size != self.cfg.action_dim or not np.isfinite(action).all():
                    raise RuntimeError("策略输出维度错误或包含 NaN/Inf")
                if self.cfg.policy_action_clip is not None:
                    action = np.clip(action, -self.cfg.policy_action_clip, self.cfg.policy_action_clip)
                action = action.astype(np.float32)
                self.last_action[:] = action
                target_sdk = self._target_sdk(action)

                for _ in range(self.cfg.decimation):
                    delayed_target = self.latency.process(target_sdk)
                    self.data.ctrl[self.actuator_ids] = self._torque(delayed_target)
                    mujoco.mj_step(self.model, self.data)

                self.episode_step += 1
                with self.viewer.lock():
                    self.viewer.cam.lookat[:] = self.data.xpos[self.base_body_id]
                    self._draw_arrows()
                self.viewer.sync()
                time.sleep(max(0.0, self.cfg.step_dt - (time.monotonic() - started)))
        finally:
            if self.listener is not None:
                self.listener.stop()
            self.viewer.close()


def _self_check():
    assert np.allclose(gravity_from_quat([1.0, 0.0, 0.0, 0.0]), [0.0, 0.0, -1.0])
    group = ActuatorGroup("test", np.array([0]), "ideal_pd", *[np.ones(1)] * 4, None, (2, 2))
    latency = ActuatorLatency([group], 1, announce=False)
    assert np.allclose(latency.process(np.array([3.0], dtype=np.float32)), [3.0])


if __name__ == "__main__":
    _self_check()
    MujocoRunner(build_cfg(RUN)).run()
