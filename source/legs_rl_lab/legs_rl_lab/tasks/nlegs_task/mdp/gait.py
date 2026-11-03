from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def command_is_moving(env: ManagerBasedRLEnv, command_threshold: float = 0.1) -> torch.Tensor:
    """按完整速度命令判断；原地转向也属于运动，不按实际速度切换。"""
    return torch.norm(env.command_manager.get_command("base_velocity"), dim=1) >= command_threshold


def gait_motion_weight(
    env: ManagerBasedRLEnv, command_threshold: float = 0.1, stop_transition_time: float = 0.0,
    *, update_state: bool = False,
) -> torch.Tensor:
    """Fade walking objectives after the last moving command; never override actions.

    Only observations record movement: Isaac evaluates rewards with the old
    command before resampling the next observation's command. Reward reads must
    not extend that observation's stop transition by one control step.
    """
    if not math.isfinite(stop_transition_time) or stop_transition_time < 0.0:
        raise ValueError("stop_transition_time must be finite and non-negative")
    moving = command_is_moving(env, command_threshold)
    if stop_transition_time == 0.0:
        return moving.float()
    steps = env.episode_length_buf
    now = steps.float() * env.step_dt
    state = getattr(env, "_gait_motion_state", None)
    if state is None:
        last_motion = torch.full_like(now, -float("inf"))
    else:
        previous_steps, last_motion = state
        # Zero catches same-length reset episodes, not only backwards clocks.
        reset = (steps < previous_steps) | (steps == 0)
        last_motion = torch.where(reset, -float("inf"), last_motion)
    if update_state:
        # shortcut: recorder observations before command resampling need a separate timing audit.
        last_motion = torch.where(moving, now, last_motion)
        env._gait_motion_state = (steps.clone(), last_motion)
    fading = (1.0 - (now - last_motion) / stop_transition_time).clamp(0.0, 1.0)
    return torch.where(moving, 1.0, fading)


def get_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    """当前步态相位 in [0, 1)，形状 (num_envs, 1)。

    仅由 episode 时间 / 周期决定；参数来自 env.cfg.gait（GaitCfg），
    会被 dump 到 env.yaml、可复现。所有 env 同相（不做每-env 相位随机化）。
    """
    current_time = env.episode_length_buf * env.step_dt
    phase = (current_time / env.cfg.gait.period) % 1.0
    return phase.unsqueeze(1)
