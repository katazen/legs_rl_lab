"""Bounded Isaac/PPO/export check for the independent W1196 zero-speed standing task.

This verifies implementation, not the locomotion quality of an untrained policy.
"""
import argparse
import json
from pathlib import Path
import tempfile
import traceback
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default=None, help="Optional directory for smoke-test exports, never a trained policy")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import yaml
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner

import legs_rl_lab.tasks  # noqa: F401
from legs_rl_lab.tasks.nlegs_task import mdp
from legs_rl_lab.utils.export_deploy_cfg import export_deploy_cfg


def canonical(value):
    return json.dumps(value, sort_keys=True, default=str)


def main():
    task = "nlegs_body_rough_static"
    baseline = load_cfg_from_registry("nlegs_body_rough", "env_cfg_entry_point")
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    old_agent = load_cfg_from_registry("nlegs_body_rough", "rsl_rl_cfg_entry_point")
    assert type(cfg).__bases__ == (ManagerBasedRLEnvCfg,)
    assert type(agent).__bases__ == (RslRlOnPolicyRunnerCfg,)
    for section in ("scene", "actions", "events", "terminations", "curriculum", "gait"):
        assert canonical(getattr(cfg, section).to_dict()) == canonical(getattr(baseline, section).to_dict()), section
    for section in ("actor", "critic", "algorithm"):
        assert canonical(getattr(agent, section).to_dict()) == canonical(getattr(old_agent, section).to_dict()), section
    for section in ("scene", "observations", "actions", "commands", "rewards", "terminations", "events", "curriculum", "gait"):
        assert type(getattr(cfg, section)).__module__ == type(cfg).__module__, section
    assert agent.experiment_name == task
    assert cfg.commands.base_velocity.rel_standing_envs == .35
    assert cfg.commands.base_velocity.resampling_time_range == (3., 6.)
    assert cfg.commands.base_velocity.ranges.to_dict() == baseline.commands.base_velocity.ranges.to_dict()
    for group in (cfg.observations.policy, cfg.observations.critic):
        assert group.history_length == 15
        assert group.gait_phase.params == dict(gate_by_cmd=True, command_threshold=1e-6, stop_transition_time=1.2)
    assert baseline.observations.policy.gait_phase.params == {}
    play = load_cfg_from_registry(task, "play_env_cfg_entry_point")
    assert play.scene.num_envs == 32 and cfg.scene.num_envs == 2048
    print("PASS: independent task, unchanged physics/PD/network, transition configuration", flush=True)

    cfg.seed = agent.seed
    cfg.scene.num_envs = 4
    cfg.sim.device = args.device
    cfg.scene.sky_light.spawn.texture_file = None
    cfg.scene.terrain.visual_material = None
    cfg.scene.terrain.terrain_generator.num_rows = 2
    cfg.scene.terrain.terrain_generator.num_cols = 4
    cfg.commands.base_velocity.debug_vis = False
    env = gym.make(task, cfg=cfg).unwrapped
    try:
        obs, _ = env.reset()
        assert obs["policy"].shape == (4, 705)
        assert obs["critic"].shape == (4, 3735)
        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(obs["critic"]).all()
        command = env.command_manager.get_command("base_velocity")
        env.episode_length_buf[:] = 0
        command.zero_()
        params = cfg.observations.policy.gait_phase.params
        torch.testing.assert_close(mdp.gait_phase_obs(env, **params), torch.zeros(4, 2, device=env.device))
        command[:, 1] = .2
        env.episode_length_buf[:] = 10
        moving = mdp.gait_phase_obs(env, **params)
        torch.testing.assert_close(moving.norm(dim=1), torch.ones(4, device=env.device))
        command.zero_()
        env.episode_length_buf[:] = 40
        mid = mdp.gait_phase_obs(env, **params)
        torch.testing.assert_close(mid.norm(dim=1), torch.full((4,), .5, device=env.device))
        for name in ("gait", "feet_clearance"):
            term = env.reward_manager.get_term_cfg(name)
            raw_params = dict(term.params, moving_only=False)
            torch.testing.assert_close(term.func(env, **term.params), .5 * term.func(env, **raw_params))
        env.episode_length_buf[:] = 70
        torch.testing.assert_close(mdp.gait_phase_obs(env, **params), torch.zeros(4, 2, device=env.device), atol=1e-6, rtol=0)
        pose = env.reward_manager.get_term_cfg("standing_joint_pos")
        torch.testing.assert_close(pose.func(env, **pose.params), mdp.joint_deviation_l1(env))
        height = env.reward_manager.get_term_cfg("base_height")
        torch.testing.assert_close(height.func(env, **height.params), torch.zeros(4, device=env.device))
        env.reset()
        for _ in range(64):
            obs, reward, _, _, _ = env.step(torch.zeros((4, 12), device=env.device))
            assert torch.isfinite(reward).all()
            assert all(torch.isfinite(v).all() for v in obs.values())
        print("PASS: runtime transition/reward gates, 705/3735 observations, 64-step finite rollout", flush=True)

        agent.device = env.device
        agent.num_steps_per_env = 8
        agent.algorithm.num_learning_epochs = 1
        agent.algorithm.num_mini_batches = 2
        with tempfile.TemporaryDirectory(prefix="body-static-smoke-") as temp:
            out = Path(args.output or temp).resolve()
            out.mkdir(parents=True, exist_ok=True)
            (out / "SMOKE_TEST_ONLY.txt").write_text("Untrained one-update implementation check. Do not deploy to hardware.\n")
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
            runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=str(out), device=env.device)
            runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)
            export_deploy_cfg(env, str(out), policy_action_clip=agent.clip_actions)
            (out / "params/agent.yaml").write_text(yaml.safe_dump(agent.to_dict()))
            dep = yaml.safe_load((out / "params/deploy.yaml").read_text())
            assert dep["observations"]["gait_phase"]["params"] == params
            assert dep["gait_period"] == .6 and dep["step_dt"] == .02
            exported = out / "exported"
            runner.export_policy_to_jit(str(exported))
            runner.export_policy_to_onnx(str(exported))
            scripted = torch.jit.load(str(exported / "policy.pt"), map_location="cpu")
            from onnx.reference import ReferenceEvaluator
            evaluator = ReferenceEvaluator(str(exported / "policy.onnx"))
            sample = torch.zeros(4, 705)
            with torch.no_grad():
                expected = scripted(sample).numpy()
            result = evaluator.run(None, {evaluator.input_names[0]: sample.numpy()})[0]
            assert expected.shape == (4, 12) and np.isfinite(expected).all()
            np.testing.assert_allclose(result, expected, atol=1e-5, rtol=1e-5)
        print("PASS: one PPO update, transition metadata, JIT/ONNX parity; policy behavior remains untrained", flush=True)
    finally:
        env.close()


try:
    main()
except Exception:
    traceback.print_exc()
    os._exit(1)  # Kit shutdown can otherwise hide a failed assertion.
else:
    app.close()
