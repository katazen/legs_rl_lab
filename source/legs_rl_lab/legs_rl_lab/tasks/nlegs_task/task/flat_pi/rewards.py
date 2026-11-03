# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

"""Mini_Pi reward formulas with explicit nlegs_v2 joint and body mappings.

Adapted from HighTorque-Robotics/Mini_Pi_RL_Baseline, release_v1.0.0,
commit e8b787afa1e30da4f28f8cd7fefe542a3912ac27, humanoid/envs/pai/pai_env.py.
State tensors use Isaac Lab wxyz quaternions; velocities at columns 7:10 and
10:13 are world linear and angular velocities. The environment owns phase,
reference updates, reward scaling, and reset of all stateful reward buffers.
"""

import torch

from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi


class PiRewardsMixin:
    """The 25 active Pi reward terms, before configured weights and control dt."""

    def _reward_joint_pos(self):
        diff = (self.dof_pos - self.ref_dof_pos)[:, self.pitch_joint_indices]
        error = torch.norm(diff, dim=1)
        return torch.exp(-2 * error) - 0.2 * error.clamp(0, 0.5)

    def _reward_feet_distance(self):
        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        d_min = torch.clamp(foot_dist - self.cfg.rewards.min_dist, -0.5, 0.0)
        d_max = torch.clamp(foot_dist - self.cfg.rewards.max_dist, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_knee_distance(self):
        knee_pos = self.rigid_state[:, self.knee_indices, :2]
        knee_dist = torch.norm(knee_pos[:, 0, :] - knee_pos[:, 1, :], dim=1)
        d_min = torch.clamp(knee_dist - self.cfg.rewards.min_dist, -0.5, 0.0)
        d_max = torch.clamp(knee_dist - self.cfg.rewards.max_dist * 2.0, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_foot_slip(self):
        """Preserve Pi's angular xy penalty, despite its foot_slip name."""
        contact = self.contact_forces[:, self.feet_indices, 2] > self.cfg.rewards.contact_threshold
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 10:12], dim=2)
        return torch.sum(torch.sqrt(foot_speed_norm) * contact, dim=1)

    def _reward_feet_air_time(self):
        # Planned stance also ends air time in the original formulation.
        contact = self.contact_forces[:, self.feet_indices, 2] > self.cfg.rewards.contact_threshold
        stance_mask = self._get_gait_phase()
        self.contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * self.contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~self.contact_filt
        return air_time.sum(dim=1)

    def _reward_feet_contact_number(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > self.cfg.rewards.contact_threshold
        reward = torch.where(contact == self._get_gait_phase(), 1, -0.3)
        return torch.mean(reward, dim=1)

    def _reward_orientation(self):
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2.0

    def _reward_feet_contact_forces(self):
        return torch.sum(
            (torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
             - self.cfg.rewards.max_contact_force).clip(0, 400),
            dim=1,
        )

    def _reward_default_hip_roll_joint_pos(self):
        # Both hip and ankle roll remain centered on zero, as in Pi.
        error = torch.norm(self.dof_pos[:, self.roll_joint_indices], dim=1).clamp(0, 50)
        return torch.exp(-error / 0.1)

    def _reward_default_thigh_joint_pos(self):
        error = torch.norm(self.dof_pos[:, self.yaw_joint_indices], dim=1).clamp(0, 50)
        return torch.exp(-error / 0.1)

    def _reward_base_height(self):
        stance_mask = self._get_gait_phase()
        measured_heights = torch.sum(
            self.rigid_state[:, self.feet_indices, 2] * stance_mask, dim=1
        ) / torch.sum(stance_mask, dim=1)
        base_height = self.root_states[:, 2] - (measured_heights - self.cfg.rewards.foot_height_offset)
        return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 50)

    def _reward_base_acc(self):
        # Pi uses the per-control-step change of all six world velocities, not /dt.
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        return torch.exp(-torch.norm(root_acc, dim=1) * 3)

    def _reward_vel_mismatch_exp(self):
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.0)
        return (lin_mismatch + ang_mismatch) / 2.0

    def _reward_track_vel_hard(self):
        lin_vel_error = torch.norm(self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        ang_vel_error = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return ((torch.exp(-lin_vel_error * 10) + torch.exp(-ang_vel_error * 10)) / 2.0
                - 0.2 * (lin_vel_error + ang_vel_error))

    def _reward_tracking_lin_vel(self):
        error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        error *= 1.0 / (1.0 + torch.abs(self.commands[:, :2]))
        return self._neg_sqrd_exp(error, a=self.cfg.rewards.tracking_sigma_lin).sum(dim=1) / 2

    def _reward_tracking_ang_vel(self):
        error = self.commands[:, 2] - self.base_ang_vel[:, 2]
        error *= 1.0 / (1.0 + torch.abs(self.commands[:, 2]))
        return self._neg_sqrd_exp(error, a=self.cfg.rewards.tracking_sigma_ang)

    def _reward_feet_clearance(self):
        """Reward cumulative rise within the Pi band during any planned swing."""
        contact = self.contact_forces[:, self.feet_indices, 2] > self.cfg.rewards.clearance_contact_threshold
        feet_z = self.rigid_state[:, self.clearance_body_indices, 2] - self.cfg.rewards.clearance_height_offset
        self.feet_height += feet_z - self.last_feet_z
        self.last_feet_z = feet_z
        swing_mask = 1 - self._get_gait_phase()
        in_band = torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) < 0.01
        reward = torch.sum(in_band * swing_mask, dim=1)
        # Preserve evaluation before contact clears the accumulated rise.
        self.feet_height *= ~contact
        return reward

    def _reward_low_speed(self):
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)
        sign_mismatch = torch.sign(self.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])
        reward = torch.zeros_like(self.base_lin_vel[:, 0])
        reward[speed_too_low] = -1.0
        reward[speed_too_high] = 0.0
        reward[speed_desired] = 1.2
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.1)

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_collision(self):
        return torch.sum(
            1.0 * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1)
                   > self.cfg.rewards.collision_threshold),
            dim=1,
        )

    def _reward_action_smoothness(self):
        term_1 = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3

    def _reward_default_ankle_roll_pos(self):
        """Pi's foot world roll/pitch reward, with Isaac Lab wxyz quaternions."""
        e_1 = wrap_to_pi(torch.stack(euler_xyz_from_quat(self.rigid_state[:, self.feet_indices[0], 3:7]), dim=1))
        e_2 = wrap_to_pi(torch.stack(euler_xyz_from_quat(self.rigid_state[:, self.feet_indices[1], 3:7]), dim=1))
        roll = (torch.abs(e_1[:, 0]) + torch.abs(e_2[:, 0])) / 2
        pitch = (torch.abs(e_1[:, 1]) + torch.abs(e_2[:, 1])) / 2
        return (torch.exp(-roll / 0.1) + torch.exp(-pitch / 0.1)) / 2

    def _reward_termination(self):
        return -(self.reset_buf * ~self.time_out_buf).float()

    def _neg_sqrd_exp(self, x, a=1):
        """Keep Pi's actual exp(-x²/a³), including the extra division by a."""
        return torch.exp(-torch.square(x / a) / a)
