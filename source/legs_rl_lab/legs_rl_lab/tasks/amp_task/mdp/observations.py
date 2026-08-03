from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .gait import get_phase


def joint_pos_rel_biased(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """joint_pos - default_joint_pos, 外加每环境恒定的"标0偏置"(模拟编码器零位标定误差)。

    偏置由 events.randomize_joint_zero_bias 在 reset 时写入 env._joint_zero_bias。
    只用于策略观测(critic 用无偏的 mdp.joint_pos_rel), 让策略对恒定零偏鲁棒 -> 不再零速漂移。
    """
    asset = env.scene[asset_cfg.name]
    rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    bias = getattr(env, "_joint_zero_bias", None)
    if bias is not None:
        rel = rel + bias[:, asset_cfg.joint_ids]
    return rel


def gait_phase_obs(env: ManagerBasedRLEnv, gate_by_cmd: bool = False) -> torch.Tensor:
    phase_linear = get_phase(env)
    phase_obs = torch.zeros(env.num_envs, 2, device=env.device)
    phase_obs[:, 0] = torch.sin(phase_linear.squeeze(1) * 2 * torch.pi)
    phase_obs[:, 1] = torch.cos(phase_linear.squeeze(1) * 2 * torch.pi)
    if gate_by_cmd:  # 零速命令时相位归零, 让策略据此保持静止(用于静止站立任务)
        cmd_flag = (torch.norm(env.command_manager.get_command("base_velocity"), dim=1) >= 0.1).float()
        phase_obs = phase_obs * cmd_flag.unsqueeze(1)
    return phase_obs


def amp_obs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    right_foot: str = "Link_R6",
    left_foot: str = "Link_L6",
) -> torch.Tensor:
    """AMP 判别器观测 (30 维), 与专家数据 nlegs_walk.txt 及 convert_tienkung_motion.py 对齐:

        [0:12]  关节位置 (绝对角, Isaac DOF 序)
        [12:24] 关节速度 (Isaac DOF 序, rad/s)
        [24:27] 右脚位置 (base 系 xyz)
        [27:30] 左脚位置 (base 系 xyz)

    脚位置 = quat_apply_inverse(root_quat_w, foot_pos_w - root_pos_w), 转到 base 系。
    右脚在前、左脚在后, 顺序必须与转换脚本一致。脚体索引惰性缓存到 env 上, 避免每步正则匹配。
    """
    asset = env.scene[asset_cfg.name]
    ids = getattr(env, "_amp_foot_ids", None)
    if ids is None:
        ids = (asset.find_bodies(right_foot)[0][0], asset.find_bodies(left_foot)[0][0])
        env._amp_foot_ids = ids
    r_id, l_id = ids

    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]  # [N,12] 绝对角
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]  # [N,12]
    root_pos = asset.data.root_pos_w
    root_quat = asset.data.root_quat_w
    r_foot = quat_apply_inverse(root_quat, asset.data.body_pos_w[:, r_id, :] - root_pos)
    l_foot = quat_apply_inverse(root_quat, asset.data.body_pos_w[:, l_id, :] - root_pos)
    return torch.cat([joint_pos, joint_vel, r_foot, l_foot], dim=-1)