from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_apply_inverse, quat_conjugate, quat_apply, euler_xyz_from_quat
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .gait import get_phase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Joint penalties.
"""


def track_lin_vel_xy_exp(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command("base_velocity")
    lin_vel_error = torch.sum(torch.square(cmd[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-4 * lin_vel_error)


def track_ang_vel_z_exp(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command("base_velocity")
    ang_vel_error = torch.square(cmd[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-4 * ang_vel_error)


def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    return (~env.termination_manager.terminated).float()


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)

def ang_vel_y(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 1])

def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    diff = torch.clamp(env.action_manager.action - env.action_manager.prev_action, -1.0, 1.0)
    return torch.sum(torch.square(diff), dim=1)


def joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    out_of_limits = -(
            asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
            asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
        env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def feet_x_distance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    leftfoot = asset.data.body_pos_w[:, asset_cfg.body_ids[0], :] - asset.data.root_link_pos_w[:, :]
    rightfoot = asset.data.body_pos_w[:, asset_cfg.body_ids[1], :] - asset.data.root_link_pos_w[:, :]
    leftfoot_b = quat_apply(quat_conjugate(asset.data.root_link_quat_w[:, :]), leftfoot)
    rightfoot_b = quat_apply(quat_conjugate(asset.data.root_link_quat_w[:, :]), rightfoot)
    x_distance_b = torch.abs(leftfoot_b[:, 0] - rightfoot_b[:, 0])
    x_vel_flag = torch.abs(env.command_manager.get_command("base_velocity")[:, 0]) < 0.1
    return x_distance_b * x_vel_flag


def feet_y_distance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, threshold: float = 0.36) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    leftfoot = asset.data.body_pos_w[:, asset_cfg.body_ids[0], :] - asset.data.root_link_pos_w[:, :]
    rightfoot = asset.data.body_pos_w[:, asset_cfg.body_ids[1], :] - asset.data.root_link_pos_w[:, :]
    leftfoot_b = quat_apply(quat_conjugate(asset.data.root_link_quat_w[:, :]), leftfoot)
    rightfoot_b = quat_apply(quat_conjugate(asset.data.root_link_quat_w[:, :]), rightfoot)
    y_distance_b = torch.abs(torch.abs(leftfoot_b[:, 1] - rightfoot_b[:, 1]) - threshold)
    y_vel_flag = torch.abs(env.command_manager.get_command("base_velocity")[:, 1]) < 0.1
    return y_distance_b * y_vel_flag


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

def flat_orientation_x(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.projected_gravity_b[:, 0])

def height_l2(env: ManagerBasedRLEnv, target_height: float,
              asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2] - target_height)


def _get_leg_phases(env: ManagerBasedRLEnv):
    cycle_phase = get_phase(env)
    off_tensor = torch.tensor(env.cfg.gait.feet_offset, device=env.device).unsqueeze(0)
    leg_phases = (cycle_phase + off_tensor) % 1.0
    return leg_phases


def feet_gait(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    leg_phases = _get_leg_phases(env)
    should_be_stance = leg_phases < env.cfg.gait.stance_ratio
    match = (should_be_stance == is_contact)
    return torch.mean(torch.where(match, 1.0, -0.5), dim=1)


def body_sway(env: ManagerBasedRLEnv, amplitude: float = 0.1,
              asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    gravity_y = asset.data.projected_gravity_b[:, 1]
    phase = get_phase(env).squeeze(1)
    target_y = amplitude * torch.sin(2 * torch.pi * phase)
    error = torch.square(gravity_y - target_y)
    return torch.exp(-40.0 * error)


def arm_swing_vel(env: ManagerBasedRLEnv, leg_cfg: SceneEntityCfg, arm_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[leg_cfg.name]
    leg_vel = asset.data.joint_vel[:, leg_cfg.joint_ids]
    arm_vel = asset.data.joint_vel[:, arm_cfg.joint_ids]
    target_arm_vel = torch.flip(leg_vel, dims=[1]) * 0.5
    error = torch.square(arm_vel - target_arm_vel)
    return torch.exp(-torch.sum(error, dim=1) / 1.0)


def arm_swing_gait(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    d_pos = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    cmd_vel = env.command_manager.get_command("base_velocity")[:, 0]
    amp = torch.abs(cmd_vel) * 0.5  # 幅度系数
    amp = torch.clamp(amp, 0.0, 1.0).unsqueeze(1)
    cycle_phase = get_phase(env)
    off_tensor = torch.tensor(env.cfg.gait.feet_offset[::-1], device=env.device).unsqueeze(0)
    arm_phases = (cycle_phase + off_tensor) % 1.0
    target_wave = torch.sin(arm_phases * 2 * torch.pi)
    target_pos = amp * target_wave
    error = torch.square(d_pos - target_pos)
    reward = torch.exp(-torch.sum(error, dim=1) / 0.1)  # sigma=0.1
    return reward


def arm_swing_pos(env: ManagerBasedRLEnv, leg_cfg: SceneEntityCfg, arm_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[leg_cfg.name]
    delta_leg = asset.data.joint_pos[:, leg_cfg.joint_ids] - asset.data.default_joint_pos[:, leg_cfg.joint_ids]
    delta_arm = asset.data.joint_pos[:, arm_cfg.joint_ids] - asset.data.default_joint_pos[:, arm_cfg.joint_ids]
    # 左臂的目标 = 右腿的动作 * 比例
    # 右臂的目标 = 左腿的动作 * 比例
    # 使用 torch.flip 将 [Left, Right] 翻转为 [Right, Left]
    # 摆动比例0.5：腿摆10度，手摆5度
    target_arm_swing = torch.flip(delta_leg, dims=[1]) * 0.5
    error = torch.square(delta_arm - target_arm_swing)
    return torch.exp(-torch.sum(error, dim=1) / 0.2)


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def feet_clearance_old(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float = 0.1) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    feet_pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - 0.0135
    leg_phases = _get_leg_phases(env)
    target_curve = torch.zeros_like(leg_phases)
    in_swing = leg_phases > env.cfg.gait.stance_ratio
    swing_duration = 1.0 - env.cfg.gait.stance_ratio
    swing_progress = (leg_phases[in_swing] - env.cfg.gait.stance_ratio) / swing_duration * torch.pi
    target_curve[in_swing] = torch.sin(swing_progress) * target_height
    error = torch.square(target_curve - feet_pos_z)
    return torch.exp(-torch.sum(error, dim=1) / 0.02)

def feet_clearance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float = 0.1) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    feet_pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - 0.0135
    leg_phases = _get_leg_phases(env)
    swing_duration = 1.0 - env.cfg.gait.stance_ratio
    in_swing = leg_phases > env.cfg.gait.stance_ratio
    swing_progress = torch.zeros_like(leg_phases)
    swing_progress[in_swing] = (leg_phases[in_swing] - env.cfg.gait.stance_ratio) / swing_duration
    weight = torch.sin(swing_progress * torch.pi) ** 2
    shortfall = torch.clamp(target_height - feet_pos_z, min=0.0)
    error = weight * shortfall ** 2
    return torch.exp(-error / 0.005).mean(dim=1)

def contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - threshold
    return torch.sum(violation.clip(min=0.0), dim=1)


def feet_flat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    gravity_dir_w = torch.tensor([0.0, 0.0, -1.0], device=env.device)
    gravity_dir_w = gravity_dir_w.repeat(env.num_envs, 2, 1)
    gravity_b = quat_apply_inverse(foot_quat, gravity_dir_w)
    roll_error = torch.square(gravity_b[:, :, 1])
    pitch_error = torch.square(gravity_b[:, :, 0])
    return torch.sum(roll_error + 0.1 * pitch_error, dim=1)


# def feet_flat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     asset: RigidObject = env.scene[asset_cfg.name]
#     foot_quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
#     gravity_dir_w = torch.tensor([0.0, 0.0, -1.0], device=env.device)
#     gravity = quat_apply_inverse(foot_quat, gravity_dir_w.repeat(env.num_envs, 2, 1))
#     return torch.sum(torch.square(gravity[:, :, :2]), dim=(1, 2))


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)
