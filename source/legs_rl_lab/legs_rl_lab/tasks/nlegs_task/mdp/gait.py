from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def get_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    """当前步态相位 in [0, 1)，形状 (num_envs, 1)。

    参数来自 env.cfg.gait（GaitCfg），因此会被 dump 到 env.yaml、可复现。
    每个 env 的相位偏移 phase_offsets 由 reset 事件 randomize_gait_phase 打乱；
    若还没被事件初始化则退化为 0（所有 env 同相）。
    """
    current_time = env.episode_length_buf * env.step_dt
    base_phase = current_time / env.cfg.gait.period
    if not hasattr(env, "phase_offsets"):
        env.phase_offsets = torch.zeros(env.num_envs, device=env.device)
    current_phase = (base_phase + env.phase_offsets) % 1.0
    return current_phase.unsqueeze(1)


def randomize_gait_phase(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """reset 事件：给被重置的 env 随机相位偏移（等价于原 LegsEnv._reset_idx 的行为）。"""
    if not hasattr(env, "phase_offsets"):
        env.phase_offsets = torch.zeros(env.num_envs, device=env.device)
    env.phase_offsets[env_ids] = torch.rand(len(env_ids), device=env.device)
