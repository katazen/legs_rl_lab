"""Exercise nlegs_v2_pi resets, history, one PPO update, export and MuJoCo replay."""

import argparse
from datetime import datetime
import importlib.util
import math
import os
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, help="Keep the smoke-test model and deployment metadata here")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import yaml
from isaaclab.envs import DirectRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner

import legs_rl_lab.tasks  # noqa: F401
from legs_rl_lab.assets.nlegs_v2.nlegs_v2 import W1196_CFG, W1196_NOMINAL_BASE_HEIGHT
from legs_rl_lab.tasks.nlegs_task.task.flat_pi import env_cfg
from legs_rl_lab.utils.export_deploy_cfg import export_deploy_cfg


ROOT = Path(__file__).resolve().parents[1]
TASK = "nlegs_v2_pi"
POLICY_DIM, CRITIC_DIM = 705, 219
SDK_NAMES = [f"joint_{side}{index}" for side in "RL" for index in range(1, 7)]


def check_config():
    cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    play = load_cfg_from_registry(TASK, "play_env_cfg_entry_point")
    agent = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    assert env_cfg.PiEnvCfg.__bases__ == (DirectRLEnvCfg,)
    assert env_cfg.PiPPORunnerCfg.__bases__ == (RslRlOnPolicyRunnerCfg,)
    assert cfg.scene.num_envs == 4096
    for config in (cfg, play):
        assert config.scene.robot is not W1196_CFG
        assert config.scene.robot.spawn.usd_path == W1196_CFG.spawn.usd_path
        assert config.scene.robot.spawn.collision_props is None
        assert math.isclose(config.robot_contact_offset, 0.01)
        assert math.isclose(config.robot_rest_offset, 0.0)
        assert config.scene.robot.joint_sdk_names == SDK_NAMES
        assert math.isclose(config.sim.dt, 0.001)
        assert config.decimation == 20 and math.isclose(config.episode_length_s, 12.0)
        assert math.isclose(config.gait.period, 0.4)
        assert (config.observation_space, config.state_space) == (POLICY_DIM, CRITIC_DIM)
        assert math.isfinite(config.rewards.target_feet_height) and config.rewards.target_feet_height > 0
        assert math.isclose(config.rewards.base_height_target, W1196_NOMINAL_BASE_HEIGHT)
        for name, actuator in config.scene.robot.actuators.items():
            assert (actuator.min_delay, actuator.max_delay) == (5, 25)
            # The task changes delay units without mutating the shared robot or its drives.
            source = W1196_CFG.actuators[name]
            assert (source.min_delay, source.max_delay) == (1, 5)
            for field in ("stiffness", "damping", "effort_limit", "effort_limit_sim",
                          "velocity_limit", "velocity_limit_sim", "armature"):
                assert getattr(actuator, field) == getattr(source, field), (name, field)
    assert agent.experiment_name == TASK and not agent.resume
    assert agent.obs_groups == {"actor": ["policy"], "critic": ["critic"]}
    assert agent.actor.hidden_dims == [512, 256, 128]
    assert agent.critic.hidden_dims == [768, 256, 128]
    assert agent.algorithm.symmetry_cfg is None
    assert agent.clip_actions is None  # The environment clips after Pi's action blend.
    print("PASS independent Pi config: 705/219 observations, 50 Hz, 5--25 ms task-local delay", flush=True)
    return cfg, agent


def check_observations(env, obs):
    assert obs["policy"].shape == (4, POLICY_DIM)
    assert obs["critic"].shape == (4, CRITIC_DIM)
    assert all(torch.isfinite(value).all() for value in obs.values())
    torch.testing.assert_close(obs["policy"], env.policy_history.flatten(1).clamp(-18, 18))
    torch.testing.assert_close(obs["critic"], env.critic_history.flatten(1).clamp(-18, 18))
    last_frame = obs["policy"].reshape(4, 15, 47)[:, -1]
    q = env.robot.data.joint_pos[:, env.joint_ids]
    q_default = env.robot.data.default_joint_pos[:, env.joint_ids]
    qd = env.robot.data.joint_vel[:, env.joint_ids]
    torch.testing.assert_close(last_frame[:, 5:17], (q - q_default).clamp(-18, 18))
    torch.testing.assert_close(last_frame[:, 17:29], (qd * 0.05).clamp(-18, 18))
    torch.testing.assert_close(last_frame[:, 29:41], env.actions.clamp(-18, 18))

    # Observing twice (as the RSL-RL wrapper/exporter does) must not advance history.
    before = {key: value.clone() for key, value in obs.items()}
    repeated = env._get_observations()
    for key in before:
        torch.testing.assert_close(repeated[key], before[key], rtol=0, atol=0)


def check_reference(env, obs):
    # The observation clock and rewarded reference must describe the same step.
    sine = obs["policy"].reshape(4, 15, 47)[:, -1, 0]
    delta = torch.zeros((4, 12), device=env.device)
    right_swing = sine.clamp(min=0)
    left_swing = (-sine).clamp(min=0)
    for offset, swing in ((0, right_swing), (6, left_swing)):
        delta[:, offset] = -0.08 * swing
        delta[:, offset + 3] = 0.16 * swing
        delta[:, offset + 4] = -0.08 * swing
    delta[sine.abs() < 0.1] = 0
    expected = env.robot.data.default_joint_pos[:, env.joint_ids] + delta
    torch.testing.assert_close(env.ref_dof_pos, expected, rtol=1e-5, atol=1e-6)


def check_partial_reset(env):
    reset_ids = torch.tensor([1, 3], device=env.device)
    keep_ids = torch.tensor([0, 2], device=env.device)
    old_policy = env.policy_history[keep_ids].clone()
    old_critic = env.critic_history[keep_ids].clone()
    old_delay_pushes = {
        name: actuator.positions_delay_buffer._circular_buffer._num_pushes[keep_ids].clone()
        for name, actuator in env.robot.actuators.items()
    }
    env.feet_height[reset_ids] = 3.0
    env.last_feet_z[reset_ids] = -10.0
    env.last_contacts[reset_ids] = True
    env._reset_idx(reset_ids)
    assert torch.count_nonzero(env.feet_height[reset_ids]) == 0
    assert not env.last_contacts[reset_ids].any()
    obs = env._get_observations()
    assert torch.count_nonzero(env.policy_history[reset_ids, :-1]) == 0
    assert torch.count_nonzero(env.critic_history[reset_ids, :-1]) == 0
    torch.testing.assert_close(env.policy_history[keep_ids], old_policy, rtol=0, atol=0)
    torch.testing.assert_close(env.critic_history[keep_ids], old_critic, rtol=0, atol=0)
    # Reset FK must not write actuator targets or advance another environment's delay clock.
    for name, actuator in env.robot.actuators.items():
        pushes = actuator.positions_delay_buffer._circular_buffer._num_pushes
        torch.testing.assert_close(pushes[keep_ids], old_delay_pushes[name], rtol=0, atol=0)
        assert torch.count_nonzero(pushes[reset_ids]) == 0, name
    check_observations(env, obs)
    check_reference(env, obs)
    obs, _, _, _, _ = env.step(torch.zeros((4, 12), device=env.device))
    # A reset teleport must not count as a metre-scale foot lift on the next step.
    assert (env.feet_height[reset_ids].abs() < 0.08).all(), env.feet_height[reset_ids]
    print("PASS partial resets: histories and actuator delay clocks preserved; reset state cleared", flush=True)


def check_training_randomization(env):
    env.reset()
    saved = (env.cfg.randomization, env.cfg.action_randomization, env.cfg.noise_enabled,
             env.common_step_counter)
    try:
        env.cfg.randomization = True
        env.cfg.action_randomization = True
        env.cfg.noise_enabled = True
        env.common_step_counter = round(env.cfg.push_interval_s / env.step_dt) - 1
        requested_actions = torch.full((4, 12), 0.4, device=env.device)
        obs, reward, terminated, truncated, _ = env.step(requested_actions)
        assert torch.isfinite(reward).all()
        assert all(torch.isfinite(value).all() for value in obs.values())
        assert torch.isfinite(env.rand_push_force).all() and torch.isfinite(env.rand_push_torque).all()
        alive = ~(terminated | truncated)
        assert alive.any(), "No live environment available to verify the push"
        assert torch.count_nonzero(env.rand_push_force[alive]) > 0
        assert torch.count_nonzero(env.rand_push_torque[alive]) > 0
        torch.testing.assert_close(env.root_states[alive, 7:9], env.rand_push_force[alive])
        torch.testing.assert_close(env.root_states[alive, 10:13], env.rand_push_torque[alive])
        assert not torch.allclose(env.actions[alive], requested_actions[alive])
        noisy_q = obs["policy"].reshape(4, 15, 47)[alive, -1, 5:17]
        physical_q = (env.dof_pos - env.default_dof_pos)[alive]
        assert not torch.allclose(noisy_q, physical_q)
        print("PASS training branches: velocity pushes, action blend/noise and observation noise", flush=True)
    finally:
        (env.cfg.randomization, env.cfg.action_randomization, env.cfg.noise_enabled,
         env.common_step_counter) = saved


def check_external_episode_clock(env):
    obs, _ = env.reset()
    changed_ids = torch.tensor([1, 3], device=env.device)
    keep_ids = torch.tensor([0, 2], device=env.device)
    old_policy = env.policy_history[keep_ids].clone()
    old_critic = env.critic_history[keep_ids].clone()
    old_phase = obs["policy"].reshape(4, 15, 47)[changed_ids, -1, :2].clone()
    # RSL-RL replaces this tensor after the wrapper's initial observation.
    episode_lengths = env.episode_length_buf.clone()
    episode_lengths[changed_ids] += 3
    env.episode_length_buf = episode_lengths
    obs = env._get_observations()
    angle = 2 * math.pi * 3 * env.step_dt / env.cfg.gait.period
    expected_phase = torch.stack((
        old_phase[:, 0] * math.cos(angle) + old_phase[:, 1] * math.sin(angle),
        old_phase[:, 1] * math.cos(angle) - old_phase[:, 0] * math.sin(angle),
    ), dim=-1)
    torch.testing.assert_close(env.policy_history[changed_ids, -1, :2], expected_phase,
                               rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(env.critic_history[changed_ids, -1, :2], expected_phase,
                               rtol=1e-5, atol=1e-6)
    assert torch.count_nonzero(env.policy_history[changed_ids, :-1]) == 0
    assert torch.count_nonzero(env.critic_history[changed_ids, :-1]) == 0
    torch.testing.assert_close(env.policy_history[keep_ids], old_policy, rtol=0, atol=0)
    torch.testing.assert_close(env.critic_history[keep_ids], old_critic, rtol=0, atol=0)
    check_observations(env, obs)  # Also verifies the next read is idempotent.
    check_reference(env, obs)
    print("PASS external episode clock: fresh phase/reference, zero padding and idempotent reads", flush=True)


def check_exports(env, agent, runner, wrapped, run_dir):
    import onnx
    from onnx.reference import ReferenceEvaluator

    export_deploy_cfg(env, str(run_dir), policy_action_clip=agent.clip_actions)
    deploy_path = run_dir / "params/deploy.yaml"
    deploy = yaml.safe_load(deploy_path.read_text())
    assert deploy["format_version"] == 3 and deploy["task_type"] == TASK
    assert deploy["real_deployment_supported"] is False
    assert deploy["joint_names"] == SDK_NAMES
    assert deploy["joint_ids_map"] == list(range(12))
    assert math.isclose(deploy["physics_dt"], 0.001)
    assert math.isclose(deploy["step_dt"], 0.02)
    assert math.isclose(deploy["gait_period"], 0.4)
    for group in deploy["actuators"].values():
        assert group["delay"] == [5, 25]
    contract = deploy["observations"]
    assert (contract["layout"], contract["frame_size"], contract["actor_history"]) == ("frame_major", 47, 15)
    assert (contract["critic_frame_size"], contract["critic_history"]) == (73, 3)
    assert contract["history_order"] == "oldest_to_newest"
    assert contract["history_reset"] == "zero_padding"
    np.testing.assert_allclose(deploy["default_joint_pos"],
                               env.robot.data.default_joint_pos[0, env.joint_ids].cpu().numpy())

    model_dir = run_dir / "exported"
    runner.export_policy_to_jit(str(model_dir))
    runner.export_policy_to_onnx(str(model_dir))
    model = torch.jit.load(str(model_dir / "policy.pt"), map_location="cpu")
    current_obs = wrapped.get_observations()
    with torch.inference_mode():
        expected = runner.get_inference_policy()(current_obs).cpu()
        actual = model(current_obs["policy"].cpu())
    assert actual.shape == (4, 12) and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    graph = onnx.load(str(model_dir / "policy.onnx"))
    onnx.checker.check_model(graph)
    assert len(graph.graph.input) == 1 and len(graph.graph.output) == 1
    assert graph.graph.input[0].type.tensor_type.shape.dim[-1].dim_value == POLICY_DIM
    assert graph.graph.output[0].type.tensor_type.shape.dim[-1].dim_value == 12
    onnx_result, = ReferenceEvaluator(graph).run(
        None, {graph.graph.input[0].name: current_obs["policy"][:1].cpu().numpy()}
    )
    np.testing.assert_allclose(onnx_result, expected[:1].numpy(), rtol=1e-4, atol=1e-5)
    print("PASS one PPO update, deployment contract and JIT/ONNX inference parity", flush=True)

    replay_path = ROOT / "source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/task/flat_pi/sim2sim.py"
    spec = importlib.util.spec_from_file_location("pi_smoke_replay", replay_path)
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    # Independently assemble the replay input from the same physical state.
    to_numpy = lambda value: value.detach().cpu().numpy()
    replay_frame = replay.actor_frame(
        float(env._phase_angle()[0]) / (2 * math.pi),
        to_numpy(env.commands[0, :3]),
        to_numpy(env.dof_pos[0] - env.default_dof_pos[0]),
        to_numpy(env.dof_vel[0]), to_numpy(env.actions[0]),
        to_numpy(env.base_ang_vel[0]), to_numpy(env.robot.data.root_quat_w[0]),
    )
    np.testing.assert_allclose(replay_frame, to_numpy(env.policy_history[0, -1]), rtol=1e-5, atol=1e-5)
    config = replay.load_config(deploy=deploy_path, policy=model_dir / "policy.pt")
    mujoco_runner = replay.PiMujocoRunner(config, show_viewer=False)
    mujoco_runner.command[:] = [0.2, 0.0, 0.0]
    mujoco_runner.run(duration=0.4, realtime=False)
    assert np.isfinite(mujoco_runner.data.qpos).all()
    assert np.isfinite(mujoco_runner.data.qvel).all()
    assert np.isfinite(mujoco_runner.history.buffer).all()
    assert mujoco_runner.history.buffer.shape == (15, 47)
    assert mujoco_runner.data.time >= 0.39
    print("PASS 0.4 s MuJoCo replay: finite state and 15 x 47 history (plumbing check, not walking quality)", flush=True)


def main():
    cfg, agent = check_config()
    cfg.scene.num_envs = 4
    cfg.sim.device = args.device
    cfg.noise_enabled = False
    cfg.randomization = False
    cfg.action_randomization = False
    run_dir = args.output_dir or ROOT / "outputs" / f"pi_task_check_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make(TASK, cfg=cfg).unwrapped
    try:
        view = env.robot.root_physx_view
        contact_offsets = view.get_contact_offsets()
        rest_offsets = view.get_rest_offsets()
        assert contact_offsets.shape[0] == 4 and contact_offsets.shape[1] > 0
        torch.testing.assert_close(contact_offsets, torch.full_like(contact_offsets, 0.01))
        torch.testing.assert_close(rest_offsets, torch.zeros_like(rest_offsets))
        # Contact tuning must preserve the instanced USD asset rather than expand every mesh.
        stage = env.sim.stage
        for env_id in range(4):
            for body in env.robot.body_names:
                for group in ("visuals", "collisions"):
                    assert stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Robot/base/{body}/{group}").IsInstance()
        print(f"PASS PhysX contact/rest offsets: 0.01/0 m on {contact_offsets.numel()} shapes; USD instances preserved", flush=True)
        assert [env.robot.joint_names[i] for i in env.joint_ids] == SDK_NAMES
        obs, _ = env.reset()
        assert torch.count_nonzero(env.policy_history[:, :-1]) == 0
        assert torch.count_nonzero(env.critic_history[:, :-1]) == 0
        check_observations(env, obs)
        check_reference(env, obs)
        for _ in range(60):
            obs, reward, _, _, _ = env.step(torch.zeros((4, 12), device=env.device))
            assert torch.isfinite(reward).all()
            check_observations(env, obs)
            check_reference(env, obs)
        check_partial_reset(env)
        print("PASS 4-env/60-step rollout: finite values, SDK frame order and current-step reference", flush=True)
        check_external_episode_clock(env)
        check_training_randomization(env)

        agent.device = env.device
        agent.num_steps_per_env = 8
        agent.algorithm.num_learning_epochs = 1
        agent.algorithm.num_mini_batches = 2
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=str(run_dir), device=env.device)
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
        check_exports(env, agent, runner, wrapped, run_dir)
        torch.testing.assert_close(view.get_contact_offsets(), torch.full_like(contact_offsets, 0.01))
        torch.testing.assert_close(view.get_rest_offsets(), torch.zeros_like(rest_offsets))
        print(f"PASS nlegs_v2_pi smoke; artifacts: {run_dir}", flush=True)
    finally:
        env.close()


try:
    main()
except Exception:
    traceback.print_exc()
    # Kit shutdown can otherwise hide the Python failure's exit status.
    os._exit(1)
else:
    app.close()
