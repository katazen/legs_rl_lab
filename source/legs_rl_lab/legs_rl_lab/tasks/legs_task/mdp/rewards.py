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

def base_lateral_move_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """无 y 速度命令时惩罚 base 体系左右(y)线速度, 抑制身体左右平移/漂移。
    体系 y 速度对纯前进/转向都应≈0, 故用 base 体系而非世界系; 有横移命令(|cmd_y|>=0.1)时自动关闭, 不干扰主动横移。"""
    asset: RigidObject = env.scene[asset_cfg.name]
    lateral_vel_sq = torch.square(asset.data.root_lin_vel_b[:, 1])
    y_vel_flag = torch.abs(env.command_manager.get_command("base_velocity")[:, 1]) < 0.1
    return lateral_vel_sq * y_vel_flag

def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    diff = torch.clamp(env.action_manager.action - env.action_manager.prev_action, -1.0, 1.0)
    return torch.sum(torch.square(diff), dim=1)


def action_acc_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """二阶动作平滑(动作"加速度")= a_t - 2·a_{t-1} + a_{t-2}, 专治高频颤动(方向反复)。
    a_{t-2} 用自维护缓冲(action_manager 只存到 prev_action=a_{t-1}); 每步只被 reward 管理器调一次, 故更新安全。
    注意: DCMotor 会把命令颤动滤掉→物理 joint_acc 抓不到, 必须在【动作】上惩罚。"""
    act = env.action_manager.action
    prev = env.action_manager.prev_action
    if (not hasattr(env, "_prev_prev_action")) or env._prev_prev_action.shape != act.shape:
        env._prev_prev_action = prev.clone()
    acc = torch.clamp(act - 2.0 * prev + env._prev_prev_action, -2.0, 2.0)
    env._prev_prev_action = prev.clone()
    return torch.sum(torch.square(acc), dim=1)


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


def ankle_action(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """惩罚踝关节的动作量，让踝被动、脚触地时自然贴合（移植自 TienKung）。"""
    return torch.sum(torch.abs(env.action_manager.action[:, asset_cfg.joint_ids]), dim=1)


def ankle_torque(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """惩罚踝关节输出力矩，使踝柔顺、不主动扭地面（移植自 TienKung）。"""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def stand_still(
        env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


def base_height_l2_moving(
        env: ManagerBasedRLEnv, target_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """base 高度 L2 惩罚, 仅运动时(|cmd|>=0.1)生效; 零速时不管高度, 交给 stand_still 保持姿态。"""
    asset: RigidObject = env.scene[asset_cfg.name]
    err = torch.square(asset.data.root_pos_w[:, 2] - target_height)
    return err * (torch.norm(env.command_manager.get_command("base_velocity"), dim=1) >= 0.1)


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


def feet_gait(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, moving_only: bool = False) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    leg_phases = _get_leg_phases(env)
    should_be_stance = leg_phases < env.cfg.gait.stance_ratio
    match = (should_be_stance == is_contact)
    reward = torch.mean(torch.where(match, 1.0, -0.5), dim=1)
    if moving_only:  # 速度开关: 零速命令时不奖励踏步(用于静止站立任务)
        reward = reward * (torch.norm(env.command_manager.get_command("base_velocity"), dim=1) >= 0.1)
    return reward


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

def feet_clearance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float = 0.1, moving_only: bool = False) -> torch.Tensor:
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
    reward = torch.exp(-error / 0.005).mean(dim=1)
    if moving_only:  # 速度开关: 零速命令时不奖励抬脚(用于静止站立任务)
        reward = reward * (torch.norm(env.command_manager.get_command("base_velocity"), dim=1) >= 0.1)
    return reward

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


def feet_flat_on_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    roll_weight: float = 1.0,
    pitch_weight: float = 0.3,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """脚触地时惩罚脚底相对地面的倾斜, 使脚底 3 根碰撞胶囊均匀受力、完全贴平。

    脚底碰撞模型是 3 根平行胶囊, 同属一个刚体 Link_*6:
      - 沿脚的 X 轴(前后, 脚跟->脚尖)延伸;
      - 在 Y 轴(左右)排布于 +0.03 / 0 / -0.03。
    接触传感器只能读整只脚的净力、读不到单根胶囊, 故用姿态等价刻画"均匀受力":
      - roll  (绕脚长轴 X 的侧倾, = 世界重力在脚系的 Y 分量): 侧倾会把一侧胶囊抬离地面、
              只剩另一侧受力 -> 主项, 管"左右三根均匀受力";
      - pitch (绕 Y 的前后倾, = 世界重力在脚系的 X 分量): 前后倾让每根胶囊只有脚跟或
              脚尖着地 -> 管"每根胶囊前后均匀受力"。
    只在脚**触地**(净接触力 > contact_threshold)时惩罚, 摆动腿姿态不管(不与抬脚/步态冲突)。

    注意:
      - roll_weight 应显著大于 pitch_weight —— 蹬地末期(toe-off)脚会自然前后倾一下,
        pitch 罚太重会压制正常蹬地推进; 而着地期侧倾(roll)几乎总是坏的(崴脚/单侧受力)。
      - 用世界重力方向 => 目标是"平行世界水平面"。仅平地正确; 斜坡/台阶(nlegs_rough)上应改成
        相对地形表面法线(需地形/接触法线), 届时可另写地形相对版或在坡上调低权重。
    返回每个环境两脚的倾斜惩罚之和(在 cfg 里配负权重)。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # 触地掩码: 该脚净接触力历史最大值 > 阈值(与 feet_slide 同款) -> [envs, feet]
    in_contact = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > contact_threshold
    )

    # 脚底相对世界水平面的倾斜: 把世界重力方向转到脚体坐标系
    num_feet = len(asset_cfg.body_ids)
    foot_quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]  # [envs, feet, 4]
    gravity_dir_w = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, num_feet, 1)
    gravity_b = quat_apply_inverse(foot_quat, gravity_dir_w)  # [envs, feet, 3]

    pitch_error = torch.square(gravity_b[:, :, 0])  # X 分量 -> 前后倾(脚跟/脚尖)
    roll_error = torch.square(gravity_b[:, :, 1])   # Y 分量 -> 侧倾(左右三根)
    tilt = roll_weight * roll_error + pitch_weight * pitch_error

    return torch.sum(tilt * in_contact.float(), dim=1)


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
