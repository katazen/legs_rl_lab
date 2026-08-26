"""照搬 EngineAI(velocity/mdp/rewards.py) 的奖励函数, 供 nlegs_amp_engineai 任务使用。

原样移植, 仅两处适配本仓:
  1. EngineAI 内部用 `from isaaclab.envs import mdp` 调 action_rate_l2, 本仓改用
     `from isaaclab.envs import mdp as isaac_mdp` 避免与本仓 tasks.amp_task.mdp 命名冲突。
  2. EngineAI 的 feet_stumble 语义(切向力>阈值 且 法向力<阈值)与本仓 rewards.py 已有的
     feet_stumble(forces_xy>4*forces_z)不同, 为免 `import *` 遮蔽, 这里改名为
     feet_stumble_engineai(nlegs_amp 任务仍用旧的)。

其余函数名与本仓 rewards.py 无冲突(已核对), 可安全 `from .rewards_engineai import *`。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    quat_apply_inverse,
    quat_from_euler_xyz,
    wrap_to_pi,
    yaw_quat,
)

# feet_air_time* 定义在 locomotion-velocity 任务包(不在 isaaclab.envs.mdp), 显式再导出,
# 让本仓 amp_task.mdp 的 `from .rewards_engineai import *` 能解析 mdp.feet_air_time(_positive_biped)。
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import (  # noqa: F401
    feet_air_time,
    feet_air_time_positive_biped,
)


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# 动作平滑 / 课程缩放
# ---------------------------------------------------------------------------
def action_smoothness(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize action second-order differences to encourage smooth control."""
    action_manager = env.action_manager
    prev_prev_action = getattr(action_manager, "_prev_prev_action", None)
    if prev_prev_action is None:
        prev_prev_action = torch.zeros_like(action_manager.action)
        action_manager._prev_prev_action = prev_prev_action

    second_diff = action_manager.action + prev_prev_action - 2.0 * action_manager.prev_action
    reward = torch.sum(torch.square(second_diff), dim=1)

    # Update action history for the next step and clear reset environments.
    prev_prev_action.copy_(action_manager.prev_action)
    reset_buf = getattr(env, "reset_buf", None)
    if reset_buf is not None:
        reset_env_ids = reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if reset_env_ids.numel() > 0:
            prev_prev_action[reset_env_ids] = 0.0

    return reward


def _epoch_curriculum_scale(env: ManagerBasedRLEnv, start_scale: float, power: float, interval_epochs: int) -> float:
    """Compute epoch-based scale by exponentiating by power every interval_epochs."""
    num_step = env.common_step_counter
    interval = max(int(interval_epochs), 1)
    updates = num_step // interval
    return float(start_scale) ** (float(power) ** updates)


def action_smoothness_with_curriculum(
    env: ManagerBasedRLEnv, start_scale: float, power: float, interval_epochs: int
) -> torch.Tensor:
    """Action smoothness penalty with epoch-based curriculum scaling."""
    reward = action_smoothness(env)
    return reward * _epoch_curriculum_scale(env, start_scale, power, interval_epochs)


def action_rate_with_curriculum(
    env: ManagerBasedRLEnv, start_scale: float, power: float, interval_epochs: int
) -> torch.Tensor:
    """Action rate penalty with epoch-based curriculum scaling."""
    reward = isaac_mdp.action_rate_l2(env)
    return reward * _epoch_curriculum_scale(env, start_scale, power, interval_epochs)


def energy_cost_with_curriculum(
    env: ManagerBasedRLEnv,
    start_scale: float,
    power: float,
    interval_epochs: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Energy cost penalty with epoch-based curriculum scaling."""
    reward = energy_cost(env, asset_cfg=asset_cfg)
    return reward * _epoch_curriculum_scale(env, start_scale, power, interval_epochs)


# ---------------------------------------------------------------------------
# 速度跟踪(gravity-aligned / world-frame)
# ---------------------------------------------------------------------------
def track_lin_vel_xy_yaw_frame_exp(
    env, sigma: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), stand_threshold: float = 0.06
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    commands = env.command_manager.get_command(command_name)
    stand_command = (torch.norm(commands[:, :2], dim=1) < stand_threshold) & (
        torch.abs(commands[:, 2]) < stand_threshold
    )
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error_square = torch.sum(torch.square(commands[:, :2] - vel_yaw[:, :2]), dim=1)
    lin_vel_error_abs = torch.sum(torch.abs(commands[:, :2] - vel_yaw[:, :2]), dim=1)
    rew_square = torch.exp(-lin_vel_error_square * sigma)
    rew_abs = torch.exp(-lin_vel_error_abs * sigma)
    return torch.where(stand_command, rew_abs, rew_square)


def track_ang_vel_z_world_exp(
    env, command_name: str, sigma: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), stand_threshold: float = 0.06
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    commands = env.command_manager.get_command(command_name)
    stand_command = (torch.norm(commands[:, :2], dim=1) < stand_threshold) & (
        torch.abs(commands[:, 2]) < stand_threshold
    )
    asset = env.scene[asset_cfg.name]
    ang_vel_error_square = torch.square(commands[:, 2] - asset.data.root_ang_vel_w[:, 2])
    ang_vel_error_abs = torch.abs(commands[:, 2] - asset.data.root_ang_vel_w[:, 2])
    rew_square = torch.exp(-ang_vel_error_square * sigma)
    rew_abs = torch.exp(-ang_vel_error_abs * sigma)
    return torch.where(stand_command, rew_abs, rew_square)


# ---------------------------------------------------------------------------
# 脚: stumble / contact / position / orientation
# ---------------------------------------------------------------------------
def feet_stumble_engineai(
    env, sensor_cfg: SceneEntityCfg, tangential_threshold: float = 2.0, normal_threshold: float = 1.0
) -> torch.Tensor:
    """Penalize feet hitting vertical surfaces using contact forces.

    Flags a stumble when tangential force exceeds ``tangential_threshold`` while the normal force stays
    below ``normal_threshold``. Returns the count of stumbling feet per environment.
    (改名自 EngineAI 的 feet_stumble, 避免遮蔽本仓语义不同的同名函数。)
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    tangential = torch.norm(forces[..., :2], dim=-1) > tangential_threshold
    small_normal = torch.abs(forces[..., 2]) < normal_threshold
    stumble = tangential & small_normal
    return stumble.sum(dim=1)


def feet_contact_fixed(
    env, sensor_cfg: SceneEntityCfg, command_name: str, stand_threshold: float = 0.06, force_threshold: float = 5.0
) -> torch.Tensor:
    """Reward valid foot contacts during walking and standing.

    - When the command is effectively zero (stand), reward 1.
    - Otherwise, reward 1 if any recent timestep had exactly one foot in contact.
    Uses contact force history if available, else falls back to the latest forces.
    """
    commands = env.command_manager.get_command(command_name)
    stand_command = (torch.norm(commands[:, :2], dim=1) < stand_threshold) & (
        torch.abs(commands[:, 2]) < stand_threshold
    )

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_history = contact_sensor.data.net_forces_w_history
    if contact_history is None:
        contact_history = contact_sensor.data.net_forces_w.unsqueeze(1)

    contacts = contact_history[:, :, sensor_cfg.body_ids, 2] > force_threshold
    contact_num_buf = torch.sum(contacts, dim=-1)

    # For stand, require both feet in contact at the latest timestep.
    stand_contact = contact_num_buf[:, -1] == 2
    reward = (stand_command & stand_contact).float()
    contact_mask = (~stand_command) & torch.any(contact_num_buf == 1, dim=1)
    reward[contact_mask] = 1.0

    return reward


def feet_position(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    stand_threshold: float = 0.06,
    ankle_distance: float = 0.22,
    base_height_target: float = 0.82,
    foot_origin_offset: float = 0.045,
) -> torch.Tensor:
    """Reward keeping feet near a desired stance when standing; otherwise return 1.

    ``foot_origin_offset`` = 站姿时脚连杆原点距地面的高度(m); desired_z = -(base_height_target - offset)。
    PM01 原值 0.045; nlegs 经 MuJoCo FK 实测约 0.012(脚连杆原点几乎贴地), 由 cfg 传入。
    ``asset_cfg.body_ids`` 顺序决定 desired_y 的 +y/-y 分配(前一半 +y=左, 后一半 -y=右),
    故传入的 body_names 必须配 preserve_order=True 且按 [左脚, 右脚] 顺序列。
    """
    commands = env.command_manager.get_command(command_name)
    stand_command = (torch.norm(commands[:, :2], dim=1) < stand_threshold) & (
        torch.abs(commands[:, 2]) < stand_threshold
    )
    asset = env.scene[asset_cfg.name]

    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    base_pos_w = asset.data.root_pos_w
    base_quat_w = asset.data.root_quat_w

    # isolate yaw heading; zero roll/pitch
    r, p, y = euler_xyz_from_quat(base_quat_w)
    heading_quat = quat_from_euler_xyz(torch.zeros_like(r), torch.zeros_like(p), y)
    feet_pos_rel = feet_pos_w - base_pos_w.unsqueeze(1)
    # Expand heading quaternions per foot to satisfy broadcasting expected by quat_apply_inverse.
    num_envs, num_feet, _ = feet_pos_rel.shape
    heading_quat_per_foot = heading_quat.unsqueeze(1).expand(-1, num_feet, -1).reshape(-1, 4)
    feet_pos_rel_flat = feet_pos_rel.reshape(-1, 3)
    feet_pos_heading = quat_apply_inverse(heading_quat_per_foot, feet_pos_rel_flat).reshape(num_envs, num_feet, 3)

    desired_x = torch.zeros((num_envs, num_feet), device=feet_pos_heading.device)
    desired_y = torch.cat(
        (
            (ankle_distance * 0.5) * torch.ones((num_envs, num_feet // 2), device=feet_pos_heading.device),
            (-ankle_distance * 0.5) * torch.ones((num_envs, num_feet - num_feet // 2), device=feet_pos_heading.device),
        ),
        dim=1,
    )
    desired_z = -(base_height_target - foot_origin_offset) * torch.ones((num_envs, num_feet), device=feet_pos_heading.device)
    desired = torch.stack((desired_x, desired_y, desired_z), dim=-1)

    position_error = torch.sum(torch.abs(feet_pos_heading - desired), dim=(1, 2))
    reward_stand = torch.exp(-position_error * 3.0)
    return torch.where(stand_command, reward_stand, torch.ones_like(reward_stand))


def feet_orientation(env, asset_cfg: SceneEntityCfg, command_name: str, stand_threshold: float = 0.06) -> torch.Tensor:
    """Reward aligning feet orientation; ignore yaw error while turning."""
    commands = env.command_manager.get_command(command_name)
    yaw_command = torch.abs(commands[:, 2]) > stand_threshold

    asset = env.scene[asset_cfg.name]
    feet_quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    base_quat = asset.data.root_quat_w

    num_envs, num_feet, _ = feet_quat.shape
    feet_flat = feet_quat.reshape(-1, 4)
    roll, pitch, yaw = euler_xyz_from_quat(feet_flat)
    roll = roll.reshape(num_envs, num_feet)
    pitch = pitch.reshape(num_envs, num_feet)
    yaw = yaw.reshape(num_envs, num_feet)

    _, _, base_yaw = euler_xyz_from_quat(base_quat)

    feet_roll_pitch_error = torch.sum(torch.abs(torch.stack((roll, pitch), dim=-1)), dim=-1)
    feet_yaw_error = torch.abs(wrap_to_pi(yaw - base_yaw.unsqueeze(1)))

    rew = torch.sum(feet_roll_pitch_error + feet_yaw_error, dim=1)
    rew[yaw_command] = torch.sum(feet_roll_pitch_error[yaw_command], dim=1)
    return torch.exp(-rew * 2.0)


# ---------------------------------------------------------------------------
# air_time / clearance: 门控含 yaw, 让"纯旋转命令"也算在动 -> 旋转时抬脚踏步而非抖动
# ---------------------------------------------------------------------------
def _moving_command_mask(env, command_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    return (torch.norm(cmd[:, :2], dim=1) >= 0.1) | (torch.abs(cmd[:, 2]) >= 0.1)


def feet_air_time_engineai(
    env, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """feet_air_time(内建)的含 yaw 门控版: 触地时奖励 (last_air_time - threshold), 旋转命令也生效。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    reward *= _moving_command_mask(env, command_name)
    return reward


def feet_air_time_positive_biped_engineai(
env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """feet_air_time_positive_biped(内建)的含 yaw 门控版: 单脚支撑时长奖励(封顶 threshold), 旋转命令也生效。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    reward *= _moving_command_mask(env, command_name)
    return reward


def feet_swing_height(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    target_height: float = 0.10,
) -> torch.Tensor:
    """落地时结算"本段摆动峰值高度"对目标的偏差(惩罚, 配负权重), 直接锚定抬脚高度。

    机制(刻意模仿 feet_air_time 的落地事件结算, 与其正交互补):
      - 每步记录每只脚离地后的运行峰值高度 ``peak``;
      - 触地那一帧结算 ``(peak - target)^2`` 并把该脚峰值清零(下段摆动重新累计)。
    只在落地事件结算 -> 无法靠"抖脚/悬停"farming(抖脚只会让峰值偏离目标, 惩罚更大);
    权重不敏感, 收敛点稳定在 target 附近。想抬更高直接调大 target。
    仅"在动"(含 yaw)时计; 平地地面 z=0, 故 body_pos_w 的 z 即离地高度。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # (N, F)

    peak = getattr(env, "_feet_swing_peak", None)
    if peak is None or peak.shape != foot_z.shape:
        peak = foot_z.clone()
        env._feet_swing_peak = peak
    # 重开的环境: 清掉陈旧峰值, 用当前(落地/初始)脚高作为基线
    reset_buf = getattr(env, "reset_buf", None)
    if reset_buf is not None:
        reset_ids = reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            peak[reset_ids] = foot_z[reset_ids]
    # 累计本段摆动峰值
    torch.maximum(peak, foot_z, out=peak)

    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    error = torch.square(peak - target_height)
    reward = torch.sum(error * first_contact.float(), dim=1)
    # 落地脚峰值清零(重置为当前≈地面高), 下段摆动重新累计
    peak[first_contact] = foot_z[first_contact]

    reward *= _moving_command_mask(env, command_name)
    return reward


# ---------------------------------------------------------------------------
# base 高度 / 姿态 / 能耗 / 关节偏离
# ---------------------------------------------------------------------------
def base_height_tracking(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), target_height: float = 0.82
) -> torch.Tensor:
    """Reward keeping the base height near a target height."""
    asset = env.scene[asset_cfg.name]
    height_error = torch.abs(asset.data.root_pos_w[:, 2] - target_height)
    return torch.exp(-height_error * 30.0)


def energy_cost(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize energy consumption approximated by the sum of |torque * joint_vel|."""
    asset = env.scene[asset_cfg.name]
    joint_torques = asset.data.applied_torque[:, :]
    joint_vel = asset.data.joint_vel[:, :]
    power = joint_torques * joint_vel
    energy = torch.sum(torch.abs(power), dim=1)
    return energy


def base_orientation(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward keeping the base roll/pitch near zero."""
    asset = env.scene[asset_cfg.name]
    roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_quat_w)
    base_euler = torch.stack((roll, pitch, yaw), dim=-1)
    return torch.exp(-torch.sum(torch.abs(base_euler[:, :2]), dim=-1) * 10.0)


def joint_deviation_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tolerance: float = 0.1,
    scale: float = 3.0,
    max_err: float = 50.0,
) -> torch.Tensor:
    """Penalize joint positions deviating from defaults, in an exponential, configurable way."""
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    joint_pos = asset.data.joint_pos[:, joint_ids]
    default_pos = getattr(asset.data, "default_joint_pos", None)
    if default_pos is not None:
        default_pos = default_pos[:, joint_ids]
    else:
        default_pos = torch.zeros_like(joint_pos)
        print("Warning: joint_deviation_exp reward called but default_joint_pos not set in asset data; assuming zeros.")

    joint_error = torch.norm(joint_pos - default_pos, dim=1)
    joint_error = torch.clamp(joint_error - tolerance, min=0.0, max=max_err)
    return torch.exp(-joint_error * scale)
