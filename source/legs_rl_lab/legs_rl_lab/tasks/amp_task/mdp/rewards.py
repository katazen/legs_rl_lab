from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_apply, quat_conjugate
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .gait import get_phase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""nlegs + AMP 任务用到的奖励项（其余未被 AmpRewardsCfg 引用的奖励已删除）。

分两部分:
  - 上半段: 本项目(legs_task)既有的通用/正则奖励。
  - 下半段(见分隔注释): 从 TienKung-Lab 移植的奖励, 适配 nlegs 的 stance-first 步态相位约定。
"""


# -- 速度跟踪 / base 正则 -------------------------------------------------------


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


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


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


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


# -- 关节 / 踝 正则 -------------------------------------------------------------


def ankle_action(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """惩罚踝关节的动作量，让踝被动、脚触地时自然贴合（移植自 TienKung）。"""
    return torch.sum(torch.abs(env.action_manager.action[:, asset_cfg.joint_ids]), dim=1)


def ankle_torque(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """惩罚踝关节输出力矩，使踝柔顺、不主动扭地面（移植自 TienKung）。"""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def joint_deviation_l1(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), standing_only: bool = False
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(angle), dim=1)
    if standing_only:  # 仅在零速命令(站立)时约束关节回默认位; 行走时放开, 允许摆腿(移植自 TienKung)
        cmd = env.command_manager.get_command("base_velocity")
        zero_flag = (torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])) < 0.1
        reward = reward * zero_flag
    return reward


def feet_y_distance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, threshold: float = 0.36) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    leftfoot = asset.data.body_pos_w[:, asset_cfg.body_ids[0], :] - asset.data.root_link_pos_w[:, :]
    rightfoot = asset.data.body_pos_w[:, asset_cfg.body_ids[1], :] - asset.data.root_link_pos_w[:, :]
    leftfoot_b = quat_apply(quat_conjugate(asset.data.root_link_quat_w[:, :]), leftfoot)
    rightfoot_b = quat_apply(quat_conjugate(asset.data.root_link_quat_w[:, :]), rightfoot)
    y_distance_b = torch.abs(torch.abs(leftfoot_b[:, 1] - rightfoot_b[:, 1]) - threshold)
    y_vel_flag = torch.abs(env.command_manager.get_command("base_velocity")[:, 1]) < 0.1
    return y_distance_b * y_vel_flag


def feet_x_distance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """惩罚两脚在 base 系前后(x)方向的错位, 仅在前向零速命令时生效(避免原地踏步时前后脚)。"""
    asset = env.scene[asset_cfg.name]
    leftfoot = asset.data.body_pos_w[:, asset_cfg.body_ids[0], :] - asset.data.root_link_pos_w[:, :]
    rightfoot = asset.data.body_pos_w[:, asset_cfg.body_ids[1], :] - asset.data.root_link_pos_w[:, :]
    leftfoot_b = quat_apply(quat_conjugate(asset.data.root_link_quat_w[:, :]), leftfoot)
    rightfoot_b = quat_apply(quat_conjugate(asset.data.root_link_quat_w[:, :]), rightfoot)
    x_distance_b = torch.abs(leftfoot_b[:, 0] - rightfoot_b[:, 0])
    x_vel_flag = torch.abs(env.command_manager.get_command("base_velocity")[:, 0]) < 0.1
    return x_distance_b * x_vel_flag


# -- 脚接触 --------------------------------------------------------------------


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


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


"""
以下为从 TienKung-Lab (legged_lab/mdp/rewards.py) 移植的奖励, 适配 nlegs 的 stance-first 步态相位约定。
"""


def body_force(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    """惩罚脚部触地冲击力(超过 threshold 的部分, 上限 max_reward), 抑制砸地(移植自 TienKung)。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1)
    reward = forces - threshold
    reward = reward.clip(min=0.0, max=max_reward)
    return torch.sum(reward, dim=1)


def feet_too_near_humanoid(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, threshold: float = 0.2
) -> torch.Tensor:
    """惩罚两脚水平间距小于 threshold(避免交叉/踩到自己), 移植自 TienKung。"""
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clip(min=0.0)


def joint_action_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """惩罚指定关节的动作绝对值(hip roll / hip yaw 侧摆内扣), 移植自 TienKung。"""
    return torch.sum(torch.abs(env.action_manager.action[:, asset_cfg.joint_ids]), dim=1)


def _get_leg_phases(env: ManagerBasedRLEnv):
    cycle_phase = get_phase(env)
    off_tensor = torch.tensor(env.cfg.gait.feet_offset, device=env.device).unsqueeze(0)
    leg_phases = (cycle_phase + off_tensor) % 1.0
    return leg_phases


def _gait_clock(env: ManagerBasedRLEnv, delta_t: float = 0.02):
    """按 nlegs stance-first 约定生成每只脚的支撑/摆动指示。

    相位 phase∈[0,1): [0, stance_ratio) 为支撑期, [stance_ratio, 1) 为摆动期。
    返回 (I_frc, I_spd): I_frc 摆动期力应为 0 的窗口, I_spd 支撑期速度应为 0 的窗口;
    在两期交界处用 delta_t 做平滑过渡, 数值∈[0,1], I_frc+I_spd≈1。
    形状均为 (N, 2), 列 0/1 对应 feet_offset 顺序(与传感器 body 顺序一致)。
    """
    leg_phases = _get_leg_phases(env)  # (N, 2)
    stance_ratio = env.cfg.gait.stance_ratio
    # 支撑期 (速度应为 0): 相位落在 [0, stance_ratio)
    I_spd = torch.clip(leg_phases / delta_t, 0, 1) * torch.clip((stance_ratio - leg_phases) / delta_t, 0, 1)
    I_frc = 1.0 - I_spd
    return I_frc, I_spd


def gait_feet_frc_perio(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, delta_t: float = 0.02) -> torch.Tensor:
    """摆动期奖励脚部接触力→0(抬脚离地)。力用大系数进 exp 近似二值(移植自 TienKung)。

    力对一个控制步的 decimation 个物理子步取均值(先逐子步取模, 再对子步维求均值),
    等价 TienKung 的 avg_feet_force_per_step; 需 contact sensor history_length>=decimation。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).mean(dim=1)  # (N, 2)
    I_frc, _ = _gait_clock(env, delta_t)
    return torch.sum(I_frc * torch.exp(-200.0 * torch.square(forces)), dim=1)


def gait_feet_spd_perio(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, delta_t: float = 0.02,
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """支撑期奖励脚部速度→0(不打滑)。速度用大系数进 exp 近似二值(移植自 TienKung)。

    脚速来自 articulation(无子步 history), 用当前控制步瞬时值; 速度连续无冲击尖峰, 瞬时与子步均值差异可忽略。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    speed = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :].norm(dim=-1)  # (N, 2)
    _, I_spd = _gait_clock(env, delta_t)
    return torch.sum(I_spd * torch.exp(-100.0 * torch.square(speed)), dim=1)


def gait_feet_frc_support_perio(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, delta_t: float = 0.02) -> torch.Tensor:
    """支撑期奖励脚部有接触力(1-exp, 有力则→1), 保证支撑腿实实在在踩地(移植自 TienKung)。

    力同 gait_feet_frc_perio: 对子步取均值(先取模再均值), 对齐 TienKung avg_feet_force_per_step。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).mean(dim=1)  # (N, 2)
    _, I_spd = _gait_clock(env, delta_t)
    return torch.sum(I_spd * (1.0 - torch.exp(-10.0 * torch.square(forces))), dim=1)
