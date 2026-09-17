"""Reference-timed perturbations for crouch tracking."""

import torch

from isaaclab.envs.mdp import push_by_setting_velocity
from isaaclab.managers import ManagerTermBase


class PushDuringCrouch(ManagerTermBase):
    """One velocity push per episode, only inside the reference lowering window."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.push_at = torch.full((env.num_envs,), float("inf"), device=env.device)

    def reset(self, env_ids=None):
        command = self._env.command_manager.get_term(self.cfg.params["command_name"])
        lo, hi = self.cfg.params["motion_time_range_s"]
        if not (0 <= lo < hi <= command.motion.duration and hi - lo >= self._env.step_dt):
            raise ValueError("Push window must lie within the motion and span at least one policy step")
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        # CommandManager resets before EventManager: RSI has already selected its start frame.
        start = command.time_steps[env_ids] / command.motion.fps
        lower = (start + self._env.step_dt).clamp_min(lo)
        upper = lower.clamp_min(hi - self._env.step_dt)
        times = lower + torch.rand(len(env_ids), device=self.device) * (upper - lower)
        self.push_at[env_ids] = torch.where(lower < hi, times, float("inf"))

    def __call__(self, env, env_ids, motion_time_range_s, velocity_range, command_name):
        command = env.command_manager.get_term(command_name)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        time = command.time_steps[env_ids] / command.motion.fps
        lo, hi = motion_time_range_s
        due = (time >= self.push_at[env_ids]) & (time >= lo) & (time < hi)
        selected = env_ids[due]
        if selected.numel():
            push_by_setting_velocity(env, selected, velocity_range)
            self.push_at[selected] = float("inf")
