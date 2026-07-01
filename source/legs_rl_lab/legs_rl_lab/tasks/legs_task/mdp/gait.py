from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def get_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    """当前步态相位 in [0, 1)，形状 (num_envs, 1)。

    仅由 episode 时间 / 周期决定；参数来自 env.cfg.gait（GaitCfg），
    会被 dump 到 env.yaml、可复现。所有 env 同相（不做每-env 相位随机化）。
    """
    current_time = env.episode_length_buf * env.step_dt
    phase = (current_time / env.cfg.gait.period) % 1.0
    return phase.unsqueeze(1)
