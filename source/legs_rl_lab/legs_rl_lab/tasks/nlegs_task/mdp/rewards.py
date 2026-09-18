"""flat 任务实际用到的奖励函数（从 legs_task/mdp/rewards.py 摘取, 未用到的不搬）。"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_apply_inverse, quat_conjugate, quat_apply
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .gait import command_is_moving, get_phase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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


def _get_leg_phases(env: ManagerBasedRLEnv):
    cycle_phase = get_phase(env)
    off_tensor = torch.tensor(env.cfg.gait.feet_offset, device=env.device).unsqueeze(0)
    leg_phases = (cycle_phase + off_tensor) % 1.0
    return leg_phases


def feet_gait(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, moving_only: bool = False, command_threshold: float = 0.1
) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    leg_phases = _get_leg_phases(env)
    should_be_stance = leg_phases < env.cfg.gait.stance_ratio
    match = (should_be_stance == is_contact)
    reward = torch.mean(torch.where(match, 1.0, -0.5), dim=1)
    if moving_only:  # 速度开关: 零速命令时不奖励踏步(用于静止站立任务)
        reward = reward * command_is_moving(env, command_threshold)
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def feet_clearance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float = 0.1, moving_only: bool = False, sensor_cfg: SceneEntityCfg | None = None, command_threshold: float = 0.1) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    feet_pos_z = feet_pos[:, :, 2] - 0.0135
    if sensor_cfg is not None:
        # 地形相对化: 每只脚取 height_scanner 中水平距离最近的命中点作为脚下地面高度(rough 用)
        ray_hits = env.scene.sensors[sensor_cfg.name].data.ray_hits_w  # (N, R, 3), 未命中为 inf
        dist = torch.cdist(feet_pos[:, :, :2], ray_hits[:, :, :2])  # (N, feet, R), inf 命中点不会被选中
        ground_z = torch.gather(ray_hits[:, :, 2], 1, dist.argmin(dim=-1))
        feet_pos_z = feet_pos_z - torch.nan_to_num(ground_z, nan=0.0, posinf=0.0, neginf=0.0)
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
        reward = reward * command_is_moving(env, command_threshold)
    return reward


def feet_drag(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, height_threshold: float = 0.06, sensor_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """摆动相低空拖脚惩罚: 脚底离地 < height_threshold 时惩罚脚的水平速度(rough 用)。
    逼策略"先抬后挥、抬着落"——上台阶失败多是摆动前期脚尖踢到立面, 而非最高点不够高。"""
    asset = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    feet_pos_z = feet_pos[:, :, 2] - 0.0135
    if sensor_cfg is not None:
        # 地形相对化: 同 feet_clearance, 每只脚取 height_scanner 中水平最近命中点作脚下地面高度
        ray_hits = env.scene.sensors[sensor_cfg.name].data.ray_hits_w
        dist = torch.cdist(feet_pos[:, :, :2], ray_hits[:, :, :2])
        ground_z = torch.gather(ray_hits[:, :, 2], 1, dist.argmin(dim=-1))
        feet_pos_z = feet_pos_z - torch.nan_to_num(ground_z, nan=0.0, posinf=0.0, neginf=0.0)
    in_swing = _get_leg_phases(env) > env.cfg.gait.stance_ratio
    feet_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1)
    dragging = in_swing & (feet_pos_z < height_threshold)
    return torch.sum(feet_vel_xy * dragging, dim=1)


def feet_kick(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    cone_ratio: float = 4.0,
    saturation: float = 50.0,
) -> torch.Tensor:
    """惩罚脚踢台阶立面: feet_stumble 的按撞击功率加权版(0/1 -> 连续, 并补上历史窗口)。

    两级判据, 各管一件事:

    1) 摩擦锥门控 |F_xy| > cone_ratio * |F_z| —— 判"这是不是立面"。
       平地上物理引擎强制 |F_xy| <= mu*|F_z|, 本项目 mu 随机化上界 1.3(events 的
       static_friction_range), 叠上 rough 最陡 16.7° 坡面也只到 tan(16.7+atan(1.3))~2.6,
       所以 cone_ratio=4 在平地/坡面不可能误触发; 而脚尖平踢立面时法向是水平的、既不承重
       也没有竖向滑移来产生竖向摩擦, F_z~0 使比值趋于很大。这一级与 feet_stumble 同判据。

    2) 水平负功率 relu(-F_xy . v_xy) —— 判"撞得多狠"。
       feet_stumble 是 torch.any 二值化: 轻擦与猛撞同分, 两脚齐踢与单脚同分, 策略只能学
       "踢/不踢"两档。改成力与脚速的负功率(W)后惩罚随撞击强度连续增长, 且天然把"脚尖静静
       抵着立面"(v~0, 无害)与"0.6m/s 撞上去"区分开。

    另外 feet_stumble 只读 net_forces_w 最新一帧, 而撞击是几毫秒的脉冲(物理 200Hz /
    控制 50Hz), 很容易整个落在没被采样的子步里被漏掉; 这里改读 net_forces_w_history。

    已知局限(与 feet_stumble 相同, 受限于净接触力传感): net_forces_w 是整个脚 body 上所有
    接触点的合力, 若脚已踩在踏面上(F_z 大)同时脚尖抵着立面, 摩擦锥门控会被踏面的 F_z 压住
    而漏判。这属于"已落地后顶住立面", 比摆动相踢击危害小, 暂不处理。

    saturation 把每只脚压到 [0,1): 撞击是脉冲, 不饱和会让尖峰主导梯度、价值函数震荡。
    50.0 是按 nlegs 标定的: 整机 8.48kg -> 单脚支撑 F_z~83N; 摆动腿 ~1.5kg 以 0.5m/s 在
    毫秒级内被止住 -> 峰值力 ~150N、功率 ~75W。故有效区间大约 2~100W, 取 50 能把轻擦
    (~5W -> 0.10)到猛撞(~100W -> 0.86)铺满; 取得太小会提前饱和、轻重不分。
    调法: 看 Episode_Reward/feet_kick, 平地(课程低 level)阶段应恒为 0(门控保证), 上台阶
    阶段若长期贴近每脚 1.0 说明 saturation 取小了。

    asset_cfg 与 sensor_cfg 的 body_names 必须写同一个模式, 否则两边 body 顺序对不上。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]  # (N,T,F,3)
    forces_xy, forces_z = forces[..., :2], forces[..., 2].abs()
    # 窗口只有 4 个物理步(20ms), 脚速变化很小, 用当前速度近似整个窗口
    feet_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2].unsqueeze(1)  # (N,1,F,2)
    power = -(forces_xy * feet_vel_xy).sum(dim=-1)          # (N,T,F), >0 表示力在阻碍脚的运动
    on_riser = forces_xy.norm(dim=-1) > cone_ratio * forces_z
    power = (power * on_riser).clamp(min=0.0).amax(dim=1)   # (N,F), 取窗口内最狠的一帧
    return torch.sum(1.0 - torch.exp(-power / saturation), dim=1)


def contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - threshold
    return torch.sum(violation.clip(min=0.0), dim=1)


def feet_flat(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pitch_scale: float = 0.1,
) -> torch.Tensor:
    """罚脚底板偏离水平：把重力方向转到脚坐标系，其水平分量即 sin(倾角)。

    pitch_scale 单独缩放俯仰项，因为俯仰受踝 pitch ±0.4rad 限位制约（支撑末期存在
    降不到 0 的运动学地板值），默认 0.1 只做弱约束；需要强调脚掌全程水平时上调。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    gravity_dir_w = torch.tensor([0.0, 0.0, -1.0], device=env.device)
    gravity_dir_w = gravity_dir_w.repeat(env.num_envs, 2, 1)
    gravity_b = quat_apply_inverse(foot_quat, gravity_dir_w)
    roll_error = torch.square(gravity_b[:, :, 1])
    pitch_error = torch.square(gravity_b[:, :, 0])
    return torch.sum(roll_error + pitch_scale * pitch_error, dim=1)


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward
