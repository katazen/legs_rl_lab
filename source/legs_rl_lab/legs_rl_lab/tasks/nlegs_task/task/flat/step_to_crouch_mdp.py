"""站立→踏步下蹲→保持；参考只用于观测/奖励，绝不写入机器人状态或驱动目标。"""

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from legs_rl_lab.assets.nlegs import nlegs


class CrouchProgress(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not all(math.isfinite(x) and x > 0 for x in (
            cfg.prepare_s, cfg.lower_s, cfg.hold_s, cfg.success_hold_s,
            cfg.joint_tolerance, cfg.slip_budget_m,
        )) or cfg.hold_s < cfg.success_hold_s:
            raise ValueError("下蹲时间/容差必须为有限正数，保持阶段不能短于成功保持时间")
        if env.cfg.scene.terrain.terrain_type != "plane":
            raise ValueError("step_to_crouch 仅支持水平平地")
        env.cfg.episode_length_s = cfg.prepare_s + cfg.lower_s + cfg.hold_s
        self.robot = env.scene["robot"]
        self.feet, _ = self.robot.find_bodies(["Link_L6", "Link_R6"], preserve_order=True)
        self.sensors = [env.scene[name] for name in ("left_foot_contact", "right_foot_contact")]
        joints = ET.parse(cfg.xml_path).getroot().findall(".//worldbody//joint")
        ranges = {j.attrib["name"]: [float(x) for x in j.attrib["range"].split()] for j in joints}
        limits = torch.tensor([ranges[n] for n in self.robot.joint_names], device=self.device)
        if not torch.allclose(self.robot.data.joint_pos_limits, limits.expand(self.num_envs, -1, -1), atol=1e-5, rtol=0):
            raise ValueError("训练 USD 与 nlegs_limit.xml 的硬限位不一致，请重新转换 USD")
        sides = {"L1": 0, "L2": 1, "L3": 1, "L4": 1, "L5": 0,
                 "R1": 0, "R2": 0, "R3": 0, "R4": 1, "R5": 0}
        self.joints = [self.robot.joint_names.index(f"joint_{n}") for n in sides]
        self.goal = torch.stack([limits[i, side] for i, side in zip(self.joints, sides.values())])
        self.stand = self.robot.data.default_joint_pos[:, self.joints].clone()
        self.soft_limits = self.robot.data.soft_joint_pos_limits.clone()
        for i, side in zip(self.joints, sides.values()):
            self.soft_limits[:, i, side] = limits[i, side]
        self.start_step = torch.full((self.num_envs,), env.common_step_counter, device=self.device, dtype=torch.long)
        self.origin = self.robot.data.root_pos_w[:, :2].clone()
        self.slip_distance = torch.zeros(self.num_envs, 2, device=self.device)
        self.air_steps = torch.zeros_like(self.slip_distance)
        self.lifted = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.bool)
        self.steps = torch.zeros_like(self.slip_distance)
        self.stable_time = torch.zeros(self.num_envs, device=self.device)
        for name in ("joint_error_max_rad", "loaded_slip_max_m", "steps_left", "steps_right", "stable_time_s", "success"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    @property
    def elapsed(self):
        return (self._env.common_step_counter - self.start_step) * self._env.step_dt

    @property
    def fraction(self):
        return ((self.elapsed - self.cfg.prepare_s) / self.cfg.lower_s).clamp(0, 1)

    @property
    def command(self):
        u = self.fraction
        return (u * u * (3 - 2 * u)).unsqueeze(1)

    @property
    def stepping(self):
        fade_in = ((self.elapsed - self.cfg.prepare_s) / self._env.cfg.gait.period).clamp(0, 1)
        return fade_in * ((1 - self.fraction) / 0.2).clamp(0, 1)

    @property
    def goal_error(self):
        return (self.robot.data.joint_pos[:, self.joints] - self.goal).abs().amax(dim=1)

    def foot_state(self):
        position = self.robot.data.body_link_pos_w[:, self.feet]
        quat = self.robot.data.body_link_quat_w[:, self.feet]
        contact = torch.stack([s.data.contact_pos_w[:, 0, 0] for s in self.sensors], dim=1)
        force = torch.stack([s.data.force_matrix_w[:, 0, :, 2].sum(dim=1) for s in self.sensors], dim=1).clamp_min(0)
        contact = torch.where(torch.isfinite(contact), contact, position)
        omega = self.robot.data.body_ang_vel_w[:, self.feet]
        velocity = self.robot.data.body_link_lin_vel_w[:, self.feet] + torch.cross(omega, contact - position, dim=-1)
        slip = velocity[..., :2].norm(dim=-1)
        # XML 每脚三根胶囊的端部球心；用最低表面高度，避免脚尖还着地就算抬脚。
        centers = position.new_tensor([[x, y, -0.0015] for x in (-0.04, 0.14) for y in (-0.03, 0, 0.03)])
        offsets = quat_apply(quat[:, :, None].expand(-1, -1, 6, -1), centers.expand(self.num_envs, 2, -1, -1))
        height = (position[:, :, None, 2] + offsets[..., 2]).amin(dim=2) - 0.012
        height -= self._env.scene.env_origins[:, 2, None]
        vertical = torch.zeros_like(position)
        vertical[..., 2] = 1
        normals = quat_apply(quat, vertical)
        return force, slip, height, normals

    def stable(self):
        force, slip, _, normals = self.foot_state()
        robot = self.robot.data
        roll = torch.full_like(self.elapsed, self.cfg.goal_roll)
        pitch = torch.full_like(roll, self.cfg.goal_pitch)
        gravity = torch.stack((pitch.sin(), -roll.sin() * pitch.cos(), -roll.cos() * pitch.cos()), dim=1)
        return (
            (self.fraction >= 1) & (self.goal_error <= self.cfg.joint_tolerance)
            & ((robot.root_pos_w[:, 2] - self._env.scene.env_origins[:, 2] - self.cfg.goal_height).abs() < 0.025)
            & ((robot.projected_gravity_b - gravity).norm(dim=1) < 0.087)
            & (force > 10).all(dim=1) & (normals[..., 2] > math.cos(math.radians(7))).all(dim=1)
            & (robot.root_lin_vel_w.norm(dim=1) < 0.05) & (robot.root_ang_vel_w.norm(dim=1) < 0.1)
            & (robot.joint_vel.abs().amax(dim=1) < 0.15) & (slip.amax(dim=1) < 0.02)
        )

    def _resample_command(self, env_ids):
        self.start_step[env_ids] = self._env.common_step_counter
        self.origin[env_ids] = self.robot.data.root_pos_w[env_ids, :2]
        for buffer in (self.slip_distance, self.air_steps, self.lifted, self.steps, self.stable_time):
            buffer[env_ids] = 0
        if hasattr(self._env, "_prev_prev_action"):
            self._env._prev_prev_action[env_ids] = 0

    def _update_command(self):
        pass  # 进度直接来自 reset 后的实际步数，不使用 PPO 随机化的 episode_length_buf。

    def _update_metrics(self):
        force, slip, height, _ = self.foot_state()
        active = (self.elapsed >= self.cfg.prepare_s)[:, None]
        self.slip_distance += slip * (force > 10) * active * self._env.step_dt
        airborne = ((force < 1) & (height > 0.02) & (force.flip(dims=[1]) > 10)
                    & (self.stepping[:, None] > 0))
        self.air_steps = torch.where(airborne, self.air_steps + 1, 0)
        self.lifted |= self.air_steps * self._env.step_dt >= 0.04
        landed = self.lifted & (force > 10)
        self.steps += landed
        self.lifted &= ~landed
        self.stable_time = torch.where(self.stable(), self.stable_time + self._env.step_dt, 0)
        self.metrics["joint_error_max_rad"][:] = self.goal_error
        self.metrics["loaded_slip_max_m"][:] = self.slip_distance.amax(dim=1)
        self.metrics["steps_left"][:] = self.steps[:, 0]
        self.metrics["steps_right"][:] = self.steps[:, 1]
        self.metrics["stable_time_s"][:] = self.stable_time
        self.metrics["success"][:] = ((self.stable_time >= self.cfg.success_hold_s)
                                     & (self.steps >= 2).all(dim=1)
                                     & (self.slip_distance.amax(dim=1) <= self.cfg.slip_budget_m))


@configclass
class CrouchProgressCfg(CommandTermCfg):
    class_type: type = CrouchProgress
    resampling_time_range: tuple = (1.0e9, 1.0e9)
    xml_path: str = str(Path(nlegs.__file__).parent / "mjcf/nlegs_limit.xml")
    prepare_s: float = 1.0
    lower_s: float = 8.0
    hold_s: float = 3.0
    success_hold_s: float = 1.0
    joint_tolerance: float = 0.02
    slip_budget_m: float = 0.02
    # 由 generate_crouch_pose_bank.PoseSolver.solve_limit 求解，去掉出生离地间隙。
    goal_height: float = 0.4968918006856387
    goal_roll: float = 0.01058300513712913
    goal_pitch: float = 0.4020790749785923


def task(env) -> CrouchProgress:
    return env.command_manager.get_term("crouch_progress")


def gait_obs(env):
    term = task(env)
    phase = (term.elapsed - term.cfg.prepare_s).clamp_min(0) / env.cfg.gait.period * (2 * torch.pi)
    return torch.stack((phase.sin(), phase.cos()), dim=1) * term.stepping[:, None]


def posture(env):
    term = task(env)
    p = term.command
    target = term.stand + p * (term.goal - term.stand)
    error = (term.robot.data.joint_pos[:, term.joints] - target).square().mean(dim=1)
    return (1 + 3 * p[:, 0] ** 4) * torch.exp(-error / (0.25 - 0.20 * p[:, 0]).square())


def body_pose(env):
    term = task(env)
    p = term.command[:, 0]
    roll, pitch = p * term.cfg.goal_roll, p * term.cfg.goal_pitch
    gravity = torch.stack((pitch.sin(), -roll.sin() * pitch.cos(), -roll.cos() * pitch.cos()), dim=1)
    height = 0.581834209777198 + p * (term.cfg.goal_height - 0.581834209777198)
    height_error = term.robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - height
    return torch.exp(-height_error.square() / 0.04**2
                     - (term.robot.data.projected_gravity_b - gravity).square().sum(dim=1) / 0.15**2)


def stepping(env):
    term = task(env)
    force, _, height, _ = term.foot_state()
    phase = ((term.elapsed - term.cfg.prepare_s).clamp_min(0) / env.cfg.gait.period)[:, None]
    phase = (phase + phase.new_tensor(env.cfg.gait.feet_offset)) % 1
    swing = phase >= env.cfg.gait.stance_ratio
    swing_phase = ((phase - env.cfg.gait.stance_ratio) / (1 - env.cfg.gait.stance_ratio)).clamp(0, 1)
    desired_height = 0.06 * term.stepping[:, None] * (torch.pi * swing_phase).sin().square()
    match = torch.where(swing, force < 1, force > 10).float()
    clearance = torch.exp(-(height - desired_height).square() / 0.025**2)
    return term.stepping * (0.5 * match + 0.5 * clearance).mean(dim=1)


def settling(env):
    term = task(env)
    force, _, _, normals = term.foot_state()
    contact = (force > 10).all(dim=1).float()
    speed = term.robot.data.joint_vel.square().mean(dim=1)
    speed += term.robot.data.root_lin_vel_w.square().sum(dim=1) * 10
    speed += term.robot.data.root_ang_vel_w.square().sum(dim=1)
    return term.fraction**4 * contact * torch.exp(-speed - 10 * normals[..., :2].square().sum(dim=(1, 2)))


def loaded_slip(env):
    term = task(env)
    force, slip, _, _ = term.foot_state()
    spin = term.robot.data.body_ang_vel_w[:, term.feet, 2].abs() * 0.05
    return ((slip + spin) * (force > 10)).sum(dim=1)


def flight(env):
    force, _, _, _ = task(env).foot_state()
    return (force < 1).all(dim=1).float()


def drift(env):
    term = task(env)
    distance = (term.robot.data.root_pos_w[:, :2] - term.origin).norm(dim=1)
    return (distance - 0.15).clamp_min(0).square()


def joint_limits(env):
    term = task(env)
    q = term.robot.data.joint_pos
    return ((term.soft_limits[..., 0] - q).clamp_min(0)
            + (q - term.soft_limits[..., 1]).clamp_min(0)).sum(dim=1)


def time_out(env):
    term = task(env)
    return term.elapsed >= term.cfg.prepare_s + term.cfg.lower_s + term.cfg.hold_s


def excessive_slip(env):
    term = task(env)
    return term.slip_distance.amax(dim=1) > term.cfg.slip_budget_m
