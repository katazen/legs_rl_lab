from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

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


def gait_phase_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    cmd_flag = torch.norm(env.command_manager.get_command("base_velocity"), dim=1) >= 0.1
    phase_linear = get_phase(env)
    phase_obs = torch.zeros(env.num_envs, 2, device=env.device)
    phase_obs[:, 0] = torch.sin(phase_linear.squeeze(1) * 2 * torch.pi)
    phase_obs[:, 1] = torch.cos(phase_linear.squeeze(1) * 2 * torch.pi)
    return phase_obs
    # return phase_obs * cmd_flag.unsqueeze(1)