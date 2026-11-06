"""Task-local rewards for nlegs_body_rough."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_yaw_no_command_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize body-z angular speed squared below the yaw-command threshold.

    Angular speed is in rad/s. Translation commands do not disable this cost;
    yaw commands at or above the magnitude threshold do. This does not constrain absolute heading.
    """
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")
    return asset.data.root_ang_vel_b[:, 2].square() * (command[:, 2].abs() < command_threshold)


def feet_contact_yaw_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    min_force: float = 5.0,
    full_force: float = 20.0,
) -> torch.Tensor:
    """Penalize foot spin about the resultant contact normal, gated by vertical load.

    Forces are in N and angular velocities in rad/s. Asset and sensor selectors
    must resolve to the same foot order. Unloaded swing feet are unconstrained.
    Isaac's net_forces_w contains normal forces only, excluding friction forces.
    """
    if not 0.0 <= min_force < full_force:
        raise ValueError("feet contact yaw requires 0 <= min_force < full_force")
    asset = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    normal_force = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    vertical_force = normal_force[..., 2]
    contact_weight = ((vertical_force - min_force) / (full_force - min_force)).clamp(0.0, 1.0)
    # shortcut: non-coplanar contacts use a resultant normal; resolve per-contact normals if needed.
    normal = normal_force / torch.linalg.vector_norm(normal_force, dim=-1, keepdim=True).clamp_min(1e-6)
    yaw_rate = (asset.data.body_ang_vel_w[:, asset_cfg.body_ids, :] * normal).sum(dim=-1)
    return (contact_weight * yaw_rate.square()).sum(dim=1)


def _swing_weight(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Smooth swing weights in the configured right/left foot order."""
    gait = env.cfg.gait
    phase = (env.episode_length_buf * env.step_dt / gait.period).unsqueeze(-1)
    offsets = torch.as_tensor(gait.feet_offset, dtype=phase.dtype, device=phase.device)
    leg_phase = (phase + offsets) % 1.0
    progress = ((leg_phase - gait.stance_ratio) / (1.0 - gait.stance_ratio)).clamp(min=0.0)
    return torch.sin(torch.pi * progress).square()


def pitch_reference(
    env: ManagerBasedRLEnv,
    amplitude: float = 0.20,
    std: float = 0.20,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Soft physical hip/knee/ankle-pitch reference; encoder bias is deliberately excluded."""
    if std <= 0.0:
        raise ValueError("pitch reference std must be positive")
    asset = env.scene[asset_cfg.name]
    joint_ids = [asset.joint_names.index(f"joint_{side}{index}") for side in "RL" for index in (1, 4, 5)]
    positions = asset.data.joint_pos[:, joint_ids].reshape(-1, 2, 3)
    default = asset.data.default_joint_pos[:, joint_ids].reshape(-1, 2, 3)
    delta = positions.new_tensor([-amplitude, 2.0 * amplitude, -amplitude])
    reference = default + _swing_weight(env).unsqueeze(-1) * delta
    error = (positions - reference).square().mean(dim=(1, 2))
    return torch.exp(-error / std**2)


def feet_width_l2(
    env: ManagerBasedRLEnv,
    target_width: float = 0.265538,
    std: float = 0.02,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["Link_R6", "Link_L6"], preserve_order=True
    ),
) -> torch.Tensor:
    """Always penalize signed left-minus-right foot width in the base-yaw frame."""
    if std <= 0.0:
        raise ValueError("feet width std must be positive")
    asset = env.scene[asset_cfg.name]
    feet = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    delta = feet[:, 1] - feet[:, 0]
    w, x, y, z = asset.data.root_link_quat_w.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
    width = -yaw.sin() * delta[:, 0] + yaw.cos() * delta[:, 1]
    return ((width - target_width) / std).square()


def feet_clearance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float = 0.06,
    moving_only: bool = False,
    sensor_cfg: SceneEntityCfg | None = None,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Terrain-relative minimum clearance with a target-normalized shortfall."""
    if target_height <= 0.0:
        raise ValueError("feet clearance target_height must be positive")
    asset = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    feet_height = feet_pos[:, :, 2] - 0.0135
    if sensor_cfg is not None:
        ray_hits = env.scene.sensors[sensor_cfg.name].data.ray_hits_w
        distance = torch.cdist(feet_pos[:, :, :2], ray_hits[:, :, :2])
        ground_height = torch.gather(ray_hits[:, :, 2], 1, distance.argmin(dim=-1))
        feet_height = feet_height - torch.nan_to_num(ground_height, nan=0.0, posinf=0.0, neginf=0.0)
    shortfall = (target_height - feet_height).clamp(min=0.0) / target_height
    reward = torch.exp(-2.0 * _swing_weight(env) * shortfall.square()).mean(dim=1)
    if moving_only:
        commands = env.command_manager.get_command("base_velocity")
        reward = reward * (torch.linalg.vector_norm(commands, dim=1) >= command_threshold)
    return reward
