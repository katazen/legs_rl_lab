from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    cmd_flag = torch.norm(env.command_manager.get_command("base_velocity"), dim=1) >= 0.1
    phase_linear = env.get_phase()
    phase_obs = torch.zeros(env.num_envs, 2, device=env.device)
    phase_obs[:, 0] = torch.sin(phase_linear.squeeze(1) * 2 * torch.pi)
    phase_obs[:, 1] = torch.cos(phase_linear.squeeze(1) * 2 * torch.pi)
    return phase_obs * cmd_flag.unsqueeze(1)