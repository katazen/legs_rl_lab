from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply_inverse
import torch


class QieEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        self.period = 0.75
        self.stance_ratio = 0.55
        self.feet_offset = [0.0, 0.5]
        super().__init__(cfg, render_mode, **kwargs)


    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        if not hasattr(self, "phase_offsets"):
            self.phase_offsets = torch.zeros(self.num_envs, device=self.device)
        self.phase_offsets[env_ids] = torch.rand(len(env_ids), device=self.device)

    def get_phase(self):
        current_time = self.episode_length_buf * self.step_dt
        base_phase = current_time / self.period
        if not hasattr(self, "phase_offsets"):
            self.phase_offsets = torch.zeros(self.num_envs, device=self.device)
        current_phase = (base_phase + self.phase_offsets) % 1.0
        return current_phase.unsqueeze(1)