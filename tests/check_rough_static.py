"""Bounded rough/static check: --task nlegs_rough (or nlegs_rough_static) --headless --device cuda:0."""
import argparse
import hashlib
import os
from pathlib import Path
import re
import tempfile
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=("nlegs_rough", "nlegs_rough_static"), default="nlegs_rough_static")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import yaml
from pxr import Usd, UsdPhysics
import legs_rl_lab.tasks
from isaaclab_tasks.utils import load_cfg_from_registry
from legs_rl_lab.tasks.nlegs_task import mdp
from legs_rl_lab.tasks.nlegs_task.task.rough.rough_env_cfg import CROUCH_LIMITS
from legs_rl_lab.utils.export_deploy_cfg import export_deploy_cfg


def main():
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    play = load_cfg_from_registry(args.task, "play_env_cfg_entry_point")
    agent = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    old = load_cfg_from_registry("nlegs_flat", "env_cfg_entry_point")
    is_static = args.task == "nlegs_rough_static"
    task_limits = {name: bounds for name, bounds in CROUCH_LIMITS.items()
                   if is_static or name not in ("joint_L4", "joint_R4")}
    assert play.scene.num_envs == 32
    assert agent.experiment_name == args.task and agent.algorithm.symmetry_cfg is None
    assert load_cfg_from_registry("nlegs_flat", "rsl_rl_cfg_entry_point").algorithm.symmetry_cfg is not None
    for config in (cfg, play):
        assert config.events.crouch_joint_limits.params["joint_limits"] == task_limits
        assert bool(config.observations.policy.gait_phase.params.get("gate_by_cmd", False)) == is_static
        assert config.scene.robot.actuators["knees"].min_delay == 1
    common = yaml.safe_load((Path(__file__).resolve().parents[1] / "deploy/rl_real_py/configs/common.yaml").read_text())
    calibration = common["crouch_calibration"]
    for name, bounds in CROUCH_LIMITS.items():
        short = name.removeprefix("joint_")
        side = 0 if calibration["stop_sides"][short] == "lower" else 1
        assert bounds[side] == calibration["joint_pos"][short] and bounds[1 - side] is None
    assert old.actions.JointPositionAction.clip is None
    assert not hasattr(old.events, "crouch_joint_limits")
    assert old.scene.robot.actuators["legs"].min_delay == 4
    assert old.scene.robot.actuators["legs"].stiffness[".*4"] == 250
    assert old.scene.robot.actuators["ankle_pitch"].min_delay == 4
    assert old.scene.robot.actuators["ankle_roll"].min_delay == 4
    assert old.events.randomize_ankle_gains.params["damping_distribution_params"] == (0.9, 1.5)
    assert old.rewards.base_height.func is mdp.base_height_l2
    assert cfg.scene.robot.spawn.usd_path == old.scene.robot.spawn.usd_path
    asset_dir = Path(cfg.scene.robot.spawn.usd_path).parents[2]
    files = [p for p in asset_dir.rglob("*") if p.suffix in (".xml", ".usd")]
    hashes = {p: hashlib.sha256(p.read_bytes()).digest() for p in files}
    stage = Usd.Stage.Open(cfg.scene.robot.spawn.usd_path)
    original = {
        p.GetName(): np.radians([UsdPhysics.RevoluteJoint(p).GetLowerLimitAttr().Get(),
                                UsdPhysics.RevoluteJoint(p).GetUpperLimitAttr().Get()])
        for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)
    }
    cfg.scene.num_envs = 16
    cfg.seed = 42
    cfg.sim.device = args.device
    cfg.scene.sky_light.spawn.texture_file = None
    cfg.scene.terrain.visual_material = None
    cfg.scene.terrain.terrain_generator.num_rows = 2
    cfg.scene.terrain.terrain_generator.num_cols = 10
    cfg.scene.terrain.max_init_terrain_level = 1
    cfg.commands.base_velocity.debug_vis = False
    env = gym.make(args.task, cfg=cfg).unwrapped
    try:
        obs, _ = env.reset()
        assert obs["policy"].shape == (16, 470) and obs["critic"].shape == (16, 2490)
        robot = env.scene["robot"]
        seen_delays = {name: set() for name in robot.actuators}
        for _ in range(12):
            env.reset()
            for name, actuator in robot.actuators.items():
                lags = actuator.positions_delay_buffer.time_lags
                assert (lags >= actuator.cfg.min_delay).all() and (lags <= actuator.cfg.max_delay).all()
                seen_delays[name].update(lags.cpu().tolist())
            for name, kp, kd in (("ankle_pitch", (28, 44), (1.4, 2.4)),
                                 ("ankle_roll", (34, 44), (0.25, 0.5))):
                actuator = robot.actuators[name]
                assert (actuator.stiffness >= kp[0]).all() and (actuator.stiffness <= kp[1]).all()
                assert (actuator.damping >= kd[0]).all() and (actuator.damping <= kd[1]).all()
        for name, bounds in (("legs", (4, 6)), ("knees", (1, 6)),
                             ("ankle_pitch", (1, 8)), ("ankle_roll", (2, 6))):
            assert seen_delays[name] == set(range(bounds[0], bounds[1] + 1))
        assert robot.actuators["knees"].stiffness.eq(250).all()
        assert robot.actuators["knees"].damping.eq(5).all()
        for j, name in enumerate(robot.joint_names):
            expected = original[name].copy()
            for side, value in enumerate(task_limits.get(name, (None, None))):
                if value is not None:
                    expected[side] = value
            np.testing.assert_allclose(robot.data.joint_pos_limits[:, j].cpu(),
                                       np.tile(expected, (16, 1)), atol=2e-6, rtol=0)
        limits = robot.data.joint_pos_limits.clone()
        physics_limits = robot.root_physx_view.get_dof_limits().to(env.device)
        assert torch.allclose(limits, physics_limits, atol=2e-6, rtol=0)
        midpoint = limits.mean(dim=-1)
        half_range = (limits[..., 1] - limits[..., 0]) * 0.5 * cfg.scene.robot.soft_joint_pos_limit_factor
        assert torch.allclose(robot.data.soft_joint_pos_limits,
                              torch.stack((midpoint - half_range, midpoint + half_range), dim=-1), atol=2e-6)
        expected_default = torch.zeros_like(robot.data.default_joint_pos)
        for pattern, value in cfg.scene.robot.init_state.joint_pos.items():
            for j, name in enumerate(robot.joint_names):
                if re.fullmatch(pattern, name):
                    expected_default[:, j] = value
        assert torch.equal(expected_default, robot.data.default_joint_pos)
        action = env.action_manager.get_term("JointPositionAction")
        for sign in (-1, 1):
            action.process_actions(torch.full((16, 12), sign * 100., device=env.device))
            for name, (lo, hi) in task_limits.items():
                j = action._joint_names.index(name)
                target = action.processed_actions[:, j]
                assert lo is None or (target >= lo).all()
                assert hi is None or (target <= hi).all()
            if not is_static:
                for name in ("joint_L4", "joint_R4"):
                    j = action._joint_names.index(name)
                    assert torch.isinf(action._clip[0, j]).all()
                    assert torch.allclose(action.processed_actions[:, j],
                                          action.raw_actions[:, j] * 0.25 + expected_default[:, j])
        cmd = env.command_manager.get_command("base_velocity")
        cmd.zero_()
        cmd[1, 0] = 0.02
        cmd[2, 2] = 0.02
        moving = mdp.command_is_moving(env, getattr(cfg, "command_threshold", 1e-6))
        phase = mdp.gait_phase_obs(env, **cfg.observations.policy.gait_phase.params)
        if is_static:
            assert phase[~moving].eq(0).all() and phase[moving].square().sum(1).allclose(torch.ones(2, device=env.device))
        else:
            assert phase.square().sum(1).allclose(torch.ones(16, device=env.device))
            env.episode_length_buf += 1
            advanced = mdp.gait_phase_obs(env, **cfg.observations.policy.gait_phase.params)
            assert not torch.allclose(phase[~moving], advanced[~moving])
            assert not hasattr(cfg.rewards, "standing_joint_pos")
            assert cfg.commands.base_velocity.rel_standing_envs == old.commands.base_velocity.rel_standing_envs
        for name, original_func in (("gait", mdp.feet_gait), ("feet_clearance", mdp.feet_clearance),
                                    ("base_height", mdp.base_height_l2)):
            term = env.reward_manager.get_term_cfg(name)
            params = term.params.copy()
            params.pop("command_threshold", None)
            params.pop("moving_only", None)
            expected = original_func(env, **params) * (moving if is_static else 1)
            assert torch.allclose(term.func(env, **term.params), expected)
        if is_static:
            robot.write_joint_state_to_sim(expected_default, torch.zeros_like(expected_default))
            term = env.reward_manager.get_term_cfg("standing_joint_pos")
            assert term.func(env, **term.params).eq(0).all()
        env.reset()
        for _ in range(60):
            obs, rewards, _, _, _ = env.step(torch.zeros((16, 12), device=env.device))
            assert torch.isfinite(rewards).all() and all(torch.isfinite(v).all() for v in obs.values())
        assert torch.equal(limits, robot.data.joint_pos_limits)
        with tempfile.TemporaryDirectory(prefix="rough-static-check-") as directory:
            export_deploy_cfg(env, directory, policy_action_clip=agent.clip_actions)
            exported = yaml.safe_load((Path(directory) / "params/deploy.yaml").read_text())
            np.testing.assert_array_equal(exported["actions"]["JointPositionAction"]["clip"], action._clip[0].cpu())
            assert bool(exported["observations"]["gait_phase"]["params"].get("gate_by_cmd", False)) == is_static
            assert exported["actuators"]["knees"]["delay"] == [1, 6]
            assert exported["actuators"]["ankle_pitch"]["delay"] == [1, 8]
            assert exported["actuators"]["ankle_roll"]["delay"] == [2, 6]
        assert all(hashlib.sha256(p.read_bytes()).digest() == h for p, h in hashes.items())
        print(f"PASS {args.task}: flat isolated; 470/2490D; deployment-matched PhysX/soft limits; opposite ends/default pose unchanged; "
              "action clip; task-specific zero-command reward/phase behavior; delay/PD randomization; finite 16-env rollout; "
              "exact exported limits; assets unchanged", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        os._exit(1)
    app.close()
