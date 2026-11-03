"""Export the Pi observation/action contract without importing Isaac in replay."""

from pathlib import Path

import numpy as np
import yaml


FRAME_ORDER = ["phase_sin_cos", "scaled_commands", "joint_pos_rel", "scaled_joint_vel",
               "previous_actions", "base_ang_vel", "base_euler_xyz"]


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _values(value):
    value = _array(value)
    return (value[0] if value.ndim > 1 else value).tolist()


def export_pi_deploy_cfg(env, output_dir):
    """Write ``params/deploy.yaml`` in a run directory; joint order is SDK order."""
    cfg, asset = env.cfg, env.robot
    robot_cfg = cfg.scene.robot
    joint_names = list(robot_cfg.joint_sdk_names)
    joint_ids = _array(env.joint_ids).astype(int).tolist()
    if [asset.data.joint_names[i] for i in joint_ids] != joint_names:
        raise ValueError("Pi policy joint_ids must follow joint_sdk_names exactly")
    asset_to_sdk = {asset_id: sdk_id for sdk_id, asset_id in enumerate(joint_ids)}

    def joints(name):
        return _array(getattr(asset.data, name))[0, joint_ids].tolist()

    actuators = {}
    for name, actuator in asset.actuators.items():
        ids = actuator.joint_indices
        if isinstance(ids, slice):
            ids = list(range(len(asset.data.joint_names)))[ids]
        ids = _array(ids).astype(int).tolist()
        actuators[name] = {
            "joint_ids": [asset_to_sdk[i] for i in ids],
            "torque_model": "dc_motor" if hasattr(actuator, "_saturation_effort") else "ideal_pd",
            **{key: _values(getattr(actuator, key)) for key in (
                "effort_limit", "effort_limit_sim", "velocity_limit", "velocity_limit_sim")},
            "saturation_effort": (
                _values(actuator._saturation_effort) if hasattr(actuator, "_saturation_effort") else None),
            "delay": [int(actuator.cfg.min_delay), int(actuator.cfg.max_delay)],
        }
    command_ranges = cfg.command_ranges
    if not isinstance(command_ranges, dict):
        command_ranges = dict(zip(("lin_vel_x", "lin_vel_y", "ang_vel_z"), command_ranges))
    result = {
        "format_version": 3,
        "task_type": "nlegs_v2_pi",
        "real_deployment_supported": False,
        "joint_names": joint_names,
        "joint_ids_map": list(range(len(joint_names))),
        "physics_dt": float(cfg.sim.dt),
        "step_dt": float(cfg.sim.dt * cfg.decimation),
        "base_init_state": {"pos": list(robot_cfg.init_state.pos), "rot": list(robot_cfg.init_state.rot)},
        "contact_friction": float(cfg.scene.terrain.physics_material.dynamic_friction),
        "default_joint_pos": joints("default_joint_pos"),
        **{key: joints(f"default_joint_{key}") for key in ("stiffness", "damping", "armature")},
        **{key: joints(f"default_joint_{key}_coeff") for key in (
            "friction", "dynamic_friction", "viscous_friction")},
        "actuators": actuators,
        "action_scale": float(cfg.action_scale),
        "policy_action_clip": float(cfg.clip_actions),
        "action_target": "default_joint_pos + action_scale * clipped_policy_action",
        "deployment_action_randomization": False,
        "training_action_randomization": {
            "enabled": bool(cfg.action_randomization), "interpolation": "uniform_per_env_0_1",
            "multiplicative_gaussian_std": float(cfg.dynamic_randomization),
            "clip_after_randomization": True,
        },
        "gait_period": float(cfg.gait.period),
        "phase_offset": 0.0,
        "commands": {"base_velocity": {"ranges": {key: list(value) for key, value in command_ranges.items()}}},
        "observations": {
            "layout": "frame_major", "frame_size": 47, "actor_history": int(cfg.actor_history),
            "critic_frame_size": 73, "critic_history": int(cfg.critic_history),
            "history_order": "oldest_to_newest", "history_reset": "zero_padding",
            "frame_order": FRAME_ORDER, "command_scale": [2.0, 2.0, 1.0],
            "joint_velocity_scale": 0.05, "clip": float(cfg.clip_observations),
            "quaternion_order": "wxyz", "euler_convention": "xyz_signed_radians",
        },
    }
    path = Path(output_dir) / "params" / "deploy.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(result, sort_keys=False), encoding="utf-8")
    return str(path)
