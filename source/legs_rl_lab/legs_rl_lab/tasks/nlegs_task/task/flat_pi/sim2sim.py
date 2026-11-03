"""Pi frame-stacked policy replay on W1196. Run this file directly, outside Isaac."""

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch
import yaml


def _load_sibling(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_HERE = Path(__file__).resolve().parent
flat = _load_sibling("nlegs_pi_flat_controller", _HERE.parent / "flat" / "sim2sim.py")
deploy_module = _load_sibling("nlegs_pi_deploy", _HERE / "deploy.py")
LOGS_ROOT = Path(flat._REPO_ROOT) / "logs/rsl_rl/nlegs_v2_pi"
RUN = "2026-11-03_17-35-22_v1"
SCENE_XML = Path(flat._REPO_ROOT) / "source/legs_rl_lab/legs_rl_lab/assets/nlegs_v2/w1196_scene.xml"


def euler_xyz(quaternion):
    """World attitude from wxyz quaternion, each angle in [-pi, pi]."""
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return np.array([
        np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y)),
        np.arcsin(np.clip(2 * (w*y - z*x), -1, 1)),
        np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z)),
    ], dtype=np.float32)


def actor_frame(phase, command, q_rel, qd, previous_actions, angular_velocity, quaternion):
    """The original Pi 47-value actor frame, in SDK joint order."""
    return np.concatenate((
        [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)],
        np.asarray(command) * [2., 2., 1.], q_rel, np.asarray(qd) * .05,
        previous_actions, angular_velocity, euler_xyz(quaternion),
    )).astype(np.float32)


class FrameHistory:
    def __init__(self, length, frame_size=47):
        self.buffer = np.zeros((length, frame_size), dtype=np.float32)

    def update(self, frame):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = frame
        return self.buffer.reshape(-1).copy()


def load_config(run=None, *, deploy=None, policy=None, scene_xml=None):
    if run is None and (deploy is None or policy is None):
        raise ValueError("Provide --run, or both --deploy and --policy")
    run_dir = LOGS_ROOT / run if run is not None else Path(deploy).resolve().parent.parent
    deploy_path = Path(deploy) if deploy else run_dir / "params/deploy.yaml"
    data = yaml.safe_load(deploy_path.read_text(encoding="utf-8"))
    if data.get("format_version") != 3 or data.get("task_type") != "nlegs_v2_pi":
        raise ValueError("Expected nlegs_v2_pi format 3; old 705-dimensional policies are incompatible")
    obs = data["observations"]
    expected = {"layout": "frame_major", "frame_size": 47, "history_order": "oldest_to_newest",
                "history_reset": "zero_padding", "frame_order": deploy_module.FRAME_ORDER,
                "quaternion_order": "wxyz", "euler_convention": "xyz_signed_radians"}
    if any(obs.get(key) != value for key, value in expected.items()):
        raise ValueError("Unsupported Pi observation layout")
    if obs["command_scale"] != [2., 2., 1.] or obs["joint_velocity_scale"] != .05:
        raise ValueError("Unsupported Pi observation scaling")
    if obs["actor_history"] <= 0 or obs["clip"] <= 0:
        raise ValueError("Observation history and clipping must be positive")
    names = data["joint_names"]
    if names != [f"joint_{side}{i}" for side in ("R", "L") for i in range(1, 7)]:
        raise ValueError("Pi requires SDK joint order R1..R6,L1..L6")
    if data["joint_ids_map"] != list(range(12)):
        raise ValueError("Pi policy/action order must be SDK order")
    vector = lambda value, name: flat._as_vector(value, 12, name)
    dt, step_dt = float(data["physics_dt"]), float(data["step_dt"])
    if not np.isfinite([dt, step_dt]).all() or dt <= 0 or step_dt <= 0:
        raise ValueError("Invalid controller time step")
    decimation = round(step_dt / dt)
    if decimation < 1 or not np.isclose(decimation * dt, step_dt):
        raise ValueError("step_dt must be an integer multiple of physics_dt")
    actuators, covered = [], []
    for name, group in data["actuators"].items():
        ids = np.asarray(group["joint_ids"], dtype=np.int32)
        values = lambda key: flat._as_vector(group[key], len(ids), f"{name}.{key}")
        if group["torque_model"] not in ("dc_motor", "ideal_pd"):
            raise ValueError(f"Unsupported torque model: {group['torque_model']}")
        delay = tuple(int(value) for value in group["delay"])
        if len(delay) != 2 or not 0 <= delay[0] <= delay[1]:
            raise ValueError("Invalid actuator delay")
        actuators.append(flat.ActuatorGroup(
            name, ids, group["torque_model"], values("effort_limit"), values("effort_limit_sim"),
            values("velocity_limit"), values("velocity_limit_sim"),
            None if group["saturation_effort"] is None else values("saturation_effort"), delay))
        covered.extend(ids.tolist())
    if sorted(covered) != list(range(12)):
        raise ValueError("Actuator groups must cover each joint exactly once")
    ranges = data["commands"]["base_velocity"]["ranges"]
    default = vector(data["default_joint_pos"], "default_joint_pos")
    config = flat.SimConfig(
        run_dir=str(run_dir), model_path=str(policy or run_dir / "exported/policy.pt"),
        scene_xml=str(scene_xml or SCENE_XML), physics_dt=dt, step_dt=step_dt, decimation=decimation,
        joint_names=names, short_joint_names=[name.removeprefix("joint_") for name in names],
        policy_to_sdk=np.arange(12), default_policy=default, default_sdk=default.copy(),
        base_pos=flat._as_vector(data["base_init_state"]["pos"], 3, "base pos"),
        base_quat=flat._as_vector(data["base_init_state"]["rot"], 4, "base quat"),
        **{key: vector(data[key], key) for key in ("stiffness", "damping", "armature", "friction",
                                                   "dynamic_friction", "viscous_friction")},
        action_scale=vector(data["action_scale"], "action_scale"), action_offset=default.copy(),
        action_term_clip=None, policy_action_clip=float(data["policy_action_clip"]), observations={},
        command_ranges=np.asarray([ranges[key] for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z")]),
        gait_period=float(data["gait_period"]), actuators=actuators, commands=data["commands"],
    )
    config.pi_observations = obs
    config.phase_offset = float(data["phase_offset"])
    config.contact_friction = float(data["contact_friction"])
    if not np.isfinite(config.contact_friction) or config.contact_friction < 0:
        raise ValueError("Invalid contact friction")
    return config


class PiMujocoRunner(flat.MujocoRunner):
    def __init__(self, cfg, show_viewer=True, save_data=False):
        super().__init__(cfg, show_viewer=show_viewer, save_data=save_data)
        self.history = FrameHistory(cfg.pi_observations["actor_history"])

    def _make_model(self, cfg):
        model = super()._make_model(cfg)
        colliding = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        model.geom_friction[colliding, 0] = cfg.contact_friction
        return model

    def _observation_frame(self):
        # mj_step integrates qpos after sensor evaluation; align IMU and joints at this tick.
        flat.mujoco.mj_forward(self.model, self.data)
        q, qd = self._joint_state()
        frame = actor_frame(
            self.episode_step * self.cfg.step_dt / self.cfg.gait_period + self.cfg.phase_offset, self.command,
            q - self.cfg.default_sdk, qd, self.last_action,
            self.data.sensor("imu_gyro").data, self.data.sensor("imu_quat").data,
        )
        return np.clip(frame, -self.cfg.pi_observations["clip"], self.cfg.pi_observations["clip"])

    def _infer(self):
        obs = self.history.update(self._observation_frame())
        if not np.isfinite(obs).all():
            raise RuntimeError("Non-finite Pi observation")
        with torch.inference_mode():
            action = self.policy(torch.from_numpy(obs).unsqueeze(0)).cpu().numpy().reshape(-1)
        if action.size != 12 or not np.isfinite(action).all():
            raise RuntimeError("Policy must return 12 finite actions")
        action = np.clip(action, -self.cfg.policy_action_clip, self.cfg.policy_action_clip).astype(np.float32)
        self.last_action[:] = action
        return action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="Run name under logs/rsl_rl/nlegs_v2_pi, or absolute run path; defaults to RUN unless --deploy and --policy are both given")
    parser.add_argument("--deploy", help="Explicit Pi params/deploy.yaml")
    parser.add_argument("--policy", help="Explicit exported TorchScript policy")
    parser.add_argument("--xml", "--scene-xml", dest="scene_xml")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--command", nargs=3, type=float, default=[0., 0., 0.], metavar=("VX", "VY", "WZ"))
    parser.add_argument("--seed", type=int, default=42, help="Actuator delay sampling seed")
    parser.add_argument("--save-data", action="store_true")
    args = parser.parse_args()
    run = args.run
    if run is None and not (args.deploy and args.policy):
        run = RUN
    cfg = load_config(run, deploy=args.deploy, policy=args.policy, scene_xml=args.scene_xml)
    command = np.asarray(args.command, dtype=np.float32)
    if not np.isfinite(command).all() or np.any(command < cfg.command_ranges[:, 0]) or np.any(command > cfg.command_ranges[:, 1]):
        parser.error("--command must lie inside the exported velocity ranges")
    duration = args.duration if args.duration is not None else (10. if args.headless else flat.SIM_DURATION)
    if not np.isfinite(duration) or duration <= 0:
        parser.error("--duration must be finite and positive")
    np.random.seed(args.seed)
    runner = PiMujocoRunner(cfg, show_viewer=not args.headless, save_data=args.save_data)
    runner.command[:] = command
    runner.run(duration, realtime=not args.headless)
    if not np.isfinite(runner.data.qpos).all() or not np.isfinite(runner.data.qvel).all():
        raise RuntimeError("MuJoCo state became non-finite")
    print(f"[nlegs_v2_pi] Completed {runner.data.time:.2f}s; base z={runner.data.xpos[runner.base_body_id, 2]:.3f}m")


if __name__ == "__main__":
    main()
