from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING


class StandingVelocityCommand(UniformVelocityCommand):
    """Zero low translation/yaw commands before observations and tracking rewards."""

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        # Isaac's episode reset resamples without calling compute/_update_command.
        self._update_command()

    def _update_command(self):
        super()._update_command()
        standing = (torch.norm(self.vel_command_b[:, :2], dim=1) < self.cfg.command_threshold) & (
            self.vel_command_b[:, 2].abs() < self.cfg.yaw_command_threshold
        )
        self.vel_command_b[standing] = 0.0


@configclass
class StandingVelocityCommandCfg(UniformLevelVelocityCommandCfg):
    """Independent translation (m/s) and yaw (rad/s) deadbands for flat standing."""

    class_type: type = StandingVelocityCommand
    command_threshold: float = 0.1
    yaw_command_threshold: float = 0.1
