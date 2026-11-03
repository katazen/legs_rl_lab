"""Direct Isaac Lab implementation of the Pi gait, observations and step ordering.

Adapted from Mini_Pi_RL_Baseline e8b787a; see rewards.py for upstream BSD notice.
Joint actions/observations use W1196 SDK order (right leg, then left leg).
"""

import math

import torch
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, wrap_to_pi

from .env_cfg import PiEnvCfg
from .rewards import PiRewardsMixin


class PiEnv(PiRewardsMixin, DirectRLEnv):
    cfg: PiEnvCfg

    def __init__(self, cfg: PiEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["robot"]
        self.contact_sensor = self.scene["contact_forces"]
        self.joint_ids, names = self.robot.find_joints(cfg.scene.robot.joint_sdk_names, preserve_order=True)
        if names != cfg.scene.robot.joint_sdk_names or len(names) != 12:
            raise ValueError(f"W1196 SDK joint mapping failed: {names}")
        # Pi's first phase mask is LEFT stance; do not infer body order from PhysX indices.
        self.feet_indices, _ = self.robot.find_bodies(["Link_L6", "Link_R6"], preserve_order=True)
        self.clearance_body_indices, _ = self.robot.find_bodies(["Link_L5", "Link_R5"], preserve_order=True)
        self.knee_indices, _ = self.robot.find_bodies(["Link_L4", "Link_R4"], preserve_order=True)
        self.penalised_contact_indices, _ = self.robot.find_bodies("base")
        self._contact_ids, _ = self.contact_sensor.find_bodies(self.robot.body_names, preserve_order=True)
        self.pitch_joint_indices = [0, 3, 4, 6, 9, 10]
        self.roll_joint_indices = [1, 5, 7, 11]
        self.yaw_joint_indices = [2, 8]
        self.dt = self.step_dt
        self.default_dof_pos = self.robot.data.default_joint_pos[:, self.joint_ids].clone()
        self.default_joint_pd_target = self.default_dof_pos
        self.commands = torch.zeros(self.num_envs, 4, device=self.device)
        self.random_half_phase = torch.zeros(self.num_envs, device=self.device)
        self.ref_dof_pos = self.default_dof_pos.clone()
        self.last_actions = torch.zeros_like(self.actions)
        self.last_last_actions = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.last_root_vel = torch.zeros(self.num_envs, 6, device=self.device)
        self.feet_height = torch.zeros(self.num_envs, 2, device=self.device)
        self.last_feet_z = torch.zeros_like(self.feet_height)
        self.feet_air_time = torch.zeros_like(self.feet_height)
        self.last_contacts = torch.zeros_like(self.feet_height, dtype=torch.bool)
        self.rand_push_force = torch.zeros(self.num_envs, 2, device=self.device)
        self.rand_push_torque = torch.zeros(self.num_envs, 3, device=self.device)
        # Upstream allocates these two critic slots but never populates them.
        self.env_frictions = torch.zeros(self.num_envs, 1, device=self.device)
        self.body_mass = torch.zeros_like(self.env_frictions)
        self.policy_history = torch.zeros(self.num_envs, cfg.actor_history, 47, device=self.device)
        self.critic_history = torch.zeros(self.num_envs, cfg.critic_history, 73, device=self.device)
        self._obs_dirty = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._reset_pose_pending = torch.zeros_like(self._obs_dirty)
        self._observed_episode_step = torch.full_like(self.episode_length_buf, -1)
        self.noise_scale_vec = torch.zeros(47, device=self.device)
        self.noise_scale_vec[5:17] = cfg.noise_scales["joint_pos"]
        self.noise_scale_vec[17:29] = cfg.noise_scales["joint_vel"]
        self.noise_scale_vec[41:44] = cfg.noise_scales["ang_vel"]
        self.noise_scale_vec[44:47] = cfg.noise_scales["euler"]
        self.noise_scale_vec *= cfg.noise_level
        self._episode_sums = {
            name: torch.zeros(self.num_envs, device=self.device) for name in cfg.rewards.scales
        }
        self.reward_terms = {}
        self._randomize_physics()
        self._refresh_state()

    def _randomize_physics(self):
        """Set contact offsets, then Pi's startup material and base-mass randomization."""
        view = self.robot.root_physx_view
        ids = torch.arange(self.num_envs, dtype=torch.int, device="cpu")
        # USD schema modifiers skip W1196's instances; configure the live PhysX shapes instead.
        rest_offsets = torch.full_like(view.get_rest_offsets(), self.cfg.robot_rest_offset)
        view.set_rest_offsets(rest_offsets, ids)
        contact_offsets = torch.full_like(view.get_contact_offsets(), self.cfg.robot_contact_offset)
        view.set_contact_offsets(contact_offsets, ids)
        materials = view.get_material_properties()
        if self.cfg.randomization:
            low, high = self.cfg.friction_range
            buckets = torch.empty(self.cfg.friction_buckets).uniform_(low, high)
            friction = buckets[torch.randint(len(buckets), (self.num_envs,))]
        else:
            friction = torch.full((self.num_envs,), 0.6)
        materials[:, :, :2] = friction[:, None, None]
        materials[:, :, 2] = 0.0
        view.set_material_properties(materials, ids)
        self.sampled_friction = friction.to(self.device)
        masses = view.get_masses()
        base_id = self.penalised_contact_indices[0]
        if self.cfg.randomization:
            previous_mass = masses[:, base_id].clone()
            low, high = self.cfg.added_base_mass_range
            masses[:, base_id] += torch.empty(self.num_envs).uniform_(low, high)
            inertias = view.get_inertias()
            inertias[:, base_id] *= (masses[:, base_id] / previous_mass)[:, None]
            view.set_masses(masses, ids)
            view.set_inertias(inertias, ids)
        self.sampled_base_mass = masses[:, base_id].to(self.device)

    def _refresh_state(self):
        data = self.robot.data
        self.dof_pos = data.joint_pos[:, self.joint_ids]
        self.dof_vel = data.joint_vel[:, self.joint_ids]
        self.torques = data.applied_torque[:, self.joint_ids]
        self.root_states = data.root_state_w
        self.rigid_state = data.body_state_w
        self.base_lin_vel = quat_apply_inverse(data.root_quat_w, self.root_states[:, 7:10])
        self.base_ang_vel = data.root_ang_vel_b
        self.projected_gravity = data.projected_gravity_b
        self.base_euler_xyz = wrap_to_pi(torch.stack(euler_xyz_from_quat(data.root_quat_w), dim=-1))
        self.contact_forces = self.contact_sensor.data.net_forces_w[:, self._contact_ids]

    def _phase_angle(self):
        return 2 * math.pi * self.episode_length_buf * self.dt / self.cfg.gait.period + self.random_half_phase

    def _get_gait_phase(self):
        sine = torch.sin(self._phase_angle())
        stance = torch.stack((sine >= 0, sine < 0), dim=-1).float()
        stance[sine.abs() < 0.1] = 1.0
        return stance

    def compute_ref_state(self):
        sine = torch.sin(self._phase_angle())
        left_swing = sine.clamp(max=0)
        right_swing = -sine.clamp(min=0)
        delta = torch.zeros_like(self.ref_dof_pos)
        scale = self.cfg.rewards.target_joint_pos_scale
        for hip, knee, ankle, swing in ((0, 3, 4, right_swing), (6, 9, 10, left_swing)):
            delta[:, hip] = swing * scale
            delta[:, knee] = -swing * 2 * scale
            delta[:, ankle] = swing * scale
        delta[sine.abs() < 0.1] = 0
        self.ref_dof_pos = self.default_dof_pos + delta

    def _resample_commands(self, env_ids):
        for col, key in ((0, "lin_vel_x"), (1, "lin_vel_y"), (3, "heading") if self.cfg.heading_command else (2, "ang_vel_z")):
            low, high = self.cfg.command_ranges[key]
            self.commands[env_ids, col] = torch.empty(len(env_ids), device=self.device).uniform_(low, high)
        moving = self.commands[env_ids, :2].norm(dim=1) > self.cfg.small_command_threshold
        self.commands[env_ids, :2] *= moving[:, None]

    def _update_heading_command(self):
        if self.cfg.heading_command:
            error = wrap_to_pi(self.commands[:, 3] - self.base_euler_xyz[:, 2])
            limit = self.cfg.heading_yaw_limit
            self.commands[:, 2] = (self.cfg.heading_gain * error).clamp(-limit, limit)

    def _pre_physics_step(self, actions):
        self.last_last_actions.copy_(self.last_actions)
        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)
        self.last_root_vel.copy_(self.root_states[:, 7:13])
        if self.cfg.action_randomization:
            delay = torch.rand(self.num_envs, 1, device=self.device)
            actions = (1 - delay) * actions + delay * self.actions
            actions = actions + self.cfg.dynamic_randomization * torch.randn_like(actions) * actions
        self.actions = actions.clamp(-self.cfg.clip_actions, self.cfg.clip_actions)
        self._obs_dirty[:] = True

    def _apply_action(self):
        target = self.default_dof_pos + self.cfg.action_scale * self.actions
        self.robot.set_joint_position_target(target, joint_ids=self.joint_ids)

    def _get_dones(self):
        self._refresh_state()
        resample_steps = round(self.cfg.command_resampling_time / self.dt)
        ids = (self.episode_length_buf % resample_steps == 0).nonzero().flatten()
        self._resample_commands(ids)
        if self.cfg.randomization and self.common_step_counter % round(self.cfg.push_interval_s / self.dt) == 0:
            self.rand_push_force.uniform_(-self.cfg.max_push_vel_xy, self.cfg.max_push_vel_xy)
            self.rand_push_torque.uniform_(-self.cfg.max_push_ang_vel, self.cfg.max_push_ang_vel)
            velocity = self.root_states[:, 7:13].clone()
            velocity[:, :2] = self.rand_push_force
            velocity[:, 3:] = self.rand_push_torque
            self.robot.write_root_velocity_to_sim(velocity)
            self._refresh_state()
        self._update_heading_command()
        # Refresh BEFORE reward evaluation; the upstream reference was one tick late.
        self.compute_ref_state()
        base_contact = self.contact_forces[:, self.penalised_contact_indices].norm(dim=-1)
        fallen = (base_contact > self.cfg.termination_contact_threshold).any(dim=1)
        fallen |= (self.projected_gravity[:, :2].abs() > self.cfg.termination_gravity_xy).any(dim=1)
        fallen |= (self.root_states[:, 2] - self.scene.env_origins[:, 2]) < self.cfg.termination_height
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length
        return fallen, self.time_out_buf

    def _get_rewards(self):
        total = torch.zeros(self.num_envs, device=self.device)
        for name, weight in self.cfg.rewards.scales.items():
            if name == "termination":
                continue
            value = getattr(self, f"_reward_{name}")() * weight * self.dt
            self.reward_terms[name] = value
            self._episode_sums[name] += value
            total += value
        if self.cfg.rewards.only_positive_rewards:
            total.clamp_(min=0)
        terminal = self._reward_termination() * self.cfg.rewards.scales["termination"] * self.dt
        self.reward_terms["termination"] = terminal
        self._episode_sums["termination"] += terminal
        return total + terminal

    def _get_observations(self):
        # RSL-RL can randomize the initial episode counters after reset/get_observations.
        clock_changed = ~self._obs_dirty & (self._observed_episode_step != self.episode_length_buf)
        self.policy_history[clock_changed] = 0
        self.critic_history[clock_changed] = 0
        self._obs_dirty |= clock_changed
        ids = self._obs_dirty.nonzero().flatten()
        if len(ids):
            if self._reset_pose_pending.any():
                # Refresh FK without advancing physics, so reset motion cannot count as foot lift.
                self.sim.forward()
            self._refresh_state()
            reset_ids = self._reset_pose_pending.nonzero().flatten()
            self.last_feet_z[reset_ids] = (
                self.rigid_state[reset_ids][:, self.clearance_body_indices, 2]
                - self.cfg.rewards.clearance_height_offset
            )
            self._reset_pose_pending[:] = False
            self._update_heading_command()
            self.compute_ref_state()
            angle = self._phase_angle()
            command_input = torch.cat((
                torch.sin(angle)[:, None], torch.cos(angle)[:, None],
                self.commands[:, :3] * self.commands.new_tensor([2.0, 2.0, 1.0]),
            ), dim=-1)
            q = self.dof_pos - self.default_dof_pos
            dq = self.dof_vel * 0.05
            actor = torch.cat((command_input, q, dq, self.actions, self.base_ang_vel, self.base_euler_xyz), dim=-1)
            contact = self.contact_forces[:, self.feet_indices, 2] > self.cfg.rewards.contact_threshold
            contact[reset_ids] = False
            critic = torch.cat((
                command_input, q, dq, self.actions, self.dof_pos - self.ref_dof_pos,
                self.base_lin_vel * 2.0, self.base_ang_vel, self.base_euler_xyz,
                self.rand_push_force, self.rand_push_torque, self.env_frictions, self.body_mass / 30.0,
                self._get_gait_phase(), contact.float(),
            ), dim=-1)
            if self.cfg.noise_enabled:
                actor = actor + torch.randn_like(actor) * self.noise_scale_vec
            self.policy_history[ids, :-1] = self.policy_history[ids, 1:].clone()
            self.policy_history[ids, -1] = actor[ids]
            self.critic_history[ids, :-1] = self.critic_history[ids, 1:].clone()
            self.critic_history[ids, -1] = critic[ids]
            self._obs_dirty[ids] = False
            self._observed_episode_step[ids] = self.episode_length_buf[ids]
        return {
            "policy": self.policy_history.flatten(1).clamp(-self.cfg.clip_observations, self.cfg.clip_observations),
            "critic": self.critic_history.flatten(1).clamp(-self.cfg.clip_observations, self.cfg.clip_observations),
        }

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return
        super()._reset_idx(env_ids)
        self.extras["log"] = {}
        for name, values in self._episode_sums.items():
            self.extras["log"][f"Episode_Reward/{name}"] = values[env_ids].mean() / self.cfg.episode_length_s
            values[env_ids] = 0
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        low, high = self.cfg.reset_joint_position_range
        joint_pos += torch.empty_like(joint_pos).uniform_(low, high)
        limits = self.robot.data.joint_pos_limits[env_ids]
        joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])
        self.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
        root = self.robot.data.default_root_state[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:13], env_ids=env_ids)
        self.robot.set_joint_position_target(self.robot.data.default_joint_pos[env_ids], env_ids=env_ids)
        self.random_half_phase[env_ids] = torch.randint(2, (len(env_ids),), device=self.device) * math.pi
        self._resample_commands(env_ids)
        for buffer in (self.actions, self.last_actions, self.last_last_actions, self.last_dof_vel,
                       self.last_root_vel, self.feet_air_time, self.feet_height, self.last_contacts,
                       self.rand_push_force, self.rand_push_torque, self.policy_history, self.critic_history):
            buffer[env_ids] = 0
        self._obs_dirty[env_ids] = True
        self._reset_pose_pending[env_ids] = True
