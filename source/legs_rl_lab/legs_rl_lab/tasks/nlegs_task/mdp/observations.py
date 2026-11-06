from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .gait import gait_motion_weight, get_phase


def joint_pos_rel_biased(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """joint_pos - default_joint_pos, 外加每环境恒定的"标0偏置"(模拟编码器零位标定误差)。

    偏置由 events.randomize_joint_zero_bias 在 reset 时写入 env._joint_zero_bias。
    策略读 q_phys + b - default_joint_pos；critic 用无偏的 mdp.joint_pos_rel。
    rough 同时用 ZeroBiasJointPositionAction 换算目标；是否减少实机漂移需重新训练验证。
    """
    asset = env.scene[asset_cfg.name]
    rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    bias = getattr(env, "_joint_zero_bias", None)
    if bias is not None:
        rel = rel + bias[:, asset_cfg.joint_ids]
    return rel


def gait_phase_obs(
    env: ManagerBasedRLEnv, gate_by_cmd: bool = False, command_threshold: float = 0.1,
    stop_transition_time: float = 0.0, yaw_command_threshold: float | None = None,
) -> torch.Tensor:
    phase_linear = get_phase(env)
    phase_obs = torch.zeros(env.num_envs, 2, device=env.device)
    phase_obs[:, 0] = torch.sin(phase_linear.squeeze(1) * 2 * torch.pi)
    phase_obs[:, 1] = torch.cos(phase_linear.squeeze(1) * 2 * torch.pi)
    if gate_by_cmd:
        weight = gait_motion_weight(
            env, command_threshold, stop_transition_time, yaw_command_threshold, update_state=True,
        )
        phase_obs = phase_obs * weight.unsqueeze(1)
    return phase_obs
