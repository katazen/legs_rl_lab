"""Bounded Isaac check: named FK/velocity mapping, reset, clock, rewards and PPO update.

conda activate unitree_lab
python tests/check_crouch_mimic.py --headless --device cuda:0
"""

import argparse
import tempfile
import traceback
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--export-check-dir", type=Path, help="可选：导出两轮冒烟模型到新目录，仅供接口测试，不能部署实机")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import yaml
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.math import quat_error_magnitude
from isaaclab.utils.io import dump_yaml

import legs_rl_lab.tasks
from legs_rl_lab.tasks.mimic_task.task.nlegs_crouch.tracking_env_cfg import NlegsCrouchPlayEnvCfg, NlegsCrouchEnvCfg, joint_ranges
from legs_rl_lab.tasks.mimic_task.agents.rsl_rl_ppo_cfg import NlegsCrouchPPORunnerCfg
from legs_rl_lab.tasks.mimic_task import mdp
from legs_rl_lab.utils.export_deploy_cfg import export_deploy_cfg


def check_crouch_events(env):
    command, robot = env.command_manager.get_term("motion"), env.scene["robot"]
    event_cfg = env.event_manager.get_term_cfg("push_robot")
    event = event_cfg.func
    lo, hi = event_cfg.params["motion_time_range_s"]
    com = robot.root_physx_view.get_coms().clone()
    base = robot.body_names.index("base")
    nominal = ET.parse(command.cfg.model_file).find(".//body[@name='base']/inertial").get("pos")
    offset = com[:, base, :3] - torch.tensor([float(x) for x in nominal.split()])
    bounds = torch.tensor(list(env.cfg.events.base_com.params["com_range"].values()))
    assert (offset >= bounds[:, 0] - 1e-6).all() and (offset <= bounds[:, 1] + 1e-6).all()
    assert (offset.std(dim=0) > 0).all(), "Base CoM must vary across environments"
    assert NlegsCrouchPlayEnvCfg().events.push_robot is None

    starts = torch.tensor([0, 50, 100, 147, 148, 160, 200, 210, 220, 224, 225, 226, 227, 228, 300, 340],
                          device=env.device)
    command.time_steps[:] = starts
    event.reset()
    scheduled = event.push_at.clone()
    eligible = (starts / command.motion.fps + env.step_dt) < hi
    assert torch.isfinite(scheduled).equal(eligible)
    assert (scheduled[eligible] >= lo).all() and (scheduled[eligible] < hi).all()
    counts = torch.zeros(16, dtype=torch.long, device=env.device)
    times = []
    for step in range(171):
        command.time_steps[:] = (starts + step * 2).clamp_max(340)
        before = robot.data.root_vel_w.clone()
        pending = torch.isfinite(event.push_at)
        env.event_manager.apply(mode="interval", dt=env.step_dt)
        fired = pending & torch.isinf(event.push_at)
        counts += fired
        delta = robot.data.root_vel_w - before
        assert torch.equal(before[~fired], robot.data.root_vel_w[~fired])
        assert (delta[:, 0].abs() <= .100001).all() and (delta[:, 1].abs() <= .050001).all()
        assert torch.equal(before[:, 2:], robot.data.root_vel_w[:, 2:]), "Only horizontal velocity may change"
        times.extend((command.time_steps[fired] / command.motion.fps).tolist())
        previous = robot.data.root_vel_w.clone()
        event(env, None, **event_cfg.params)
        assert torch.equal(previous, robot.data.root_vel_w), "Same reference frame must not push twice"
    assert counts.equal(eligible.long()), "Exactly one push for each episode with a remaining window"
    assert times and all(lo <= t < hi for t in times)
    previous = event.push_at[1:].clone()
    command.time_steps[0] = 0
    event.reset(torch.tensor([0], device=env.device))
    assert torch.isfinite(event.push_at[0]) and torch.equal(previous, event.push_at[1:])
    event.cfg.params["motion_time_range_s"] = (hi, lo)
    try:
        event.reset()
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid push window must fail")
    finally:
        event.cfg.params["motion_time_range_s"] = (lo, hi)
    env.reset()
    assert torch.equal(com, robot.root_physx_view.get_coms()), "Reset must not accumulate CoM offsets"
    print(f"EVENTS PASS: CoM bounds/no drift, one horizontal push in [{min(times):.2f}, {max(times):.2f}]s; "
          "standing/hold/late RSI excluded, partial reset and no duplicate push", flush=True)


def main():
    cfg = NlegsCrouchPlayEnvCfg()
    cfg.scene.num_envs = 16
    cfg.sim.device = args.device
    cfg.events.add_base_mass.params["mass_distribution_params"] = (2.5, 2.5)
    cfg.events.push_robot = NlegsCrouchEnvCfg().events.push_robot
    env = gym.make("nlegs_mimic_crouch", cfg=cfg).unwrapped
    try:
        obs, _ = env.reset(seed=42)
        term, robot = env.command_manager.get_term("motion"), env.scene["robot"]
        assert obs["policy"].shape == (16, 69) and obs["critic"].shape == (16, 138)
        assert term.motion.fps == 100 and env.step_dt == .02
        assert term.time_steps.eq(0).all() and torch.isfinite(obs["policy"]).all()
        assert Path(term.cfg.motion_file).name == "stand_to_crouch_v3.npz"
        assert term.cfg.track_heading and term.cfg.end_hold_s == 0.
        limits = joint_ranges(term.cfg.model_file)
        expected = torch.tensor([limits[n] for n in robot.joint_names], device=env.device).expand(16, -1, -1)
        torch.testing.assert_close(robot.data.joint_pos_limits, expected, atol=1e-6, rtol=0.)
        torch.testing.assert_close(env.action_manager.get_term("JointPositionAction")._clip, expected, atol=1e-6, rtol=0.)
        rise = Path(term.cfg.motion_file).parents[2] / "nlegs_stand/motions/crouch_to_stand_v2.npz"
        with np.load(rise, allow_pickle=False) as data:
            order = [data["joint_names"].tolist().index(n) for n in robot.joint_names]
            np.testing.assert_array_equal(term.motion.joint_pos[-1].cpu(), data["joint_pos"][0, order])
            np.testing.assert_array_equal(term.motion.joint_pos[0].cpu(), data["joint_pos"][-1, order])
        assert NlegsCrouchEnvCfg().commands.motion.start_probability == .5
        assert NlegsCrouchPPORunnerCfg().algorithm.symmetry_cfg is None
        check_crouch_events(env)
        # PhysX needs one physics step to refresh child-link velocity kinematics after set_coms.
        env.step(torch.zeros(16, 12, device=env.device))
        with np.load(cfg.commands.motion.motion_file, allow_pickle=False) as raw:
            order = [raw["joint_names"].tolist().index(n) for n in robot.joint_names]
            assert np.allclose(term.motion.joint_pos.cpu(), raw["joint_pos"][:, order])
        for frame in (0, 80, 128, 228, 340):
            pos = term.motion.body_pos_w[frame, 0].expand(16, -1) + env.scene.env_origins
            state = torch.cat((pos, term.motion.body_quat_w[frame, 0].expand(16, -1),
                               term.motion.body_lin_vel_w[frame, 0].expand(16, -1),
                               term.motion.body_ang_vel_w[frame, 0].expand(16, -1)), dim=-1)
            robot.write_joint_state_to_sim(term.motion.joint_pos[frame].expand(16, -1),
                                          term.motion.joint_vel[frame].expand(16, -1))
            robot.write_root_link_state_to_sim(state)
            env.sim.forward()
            env.scene.update(env.physics_dt)
            p_error = (term.robot_body_pos_w - env.scene.env_origins[:, None] - term.motion.body_pos_w[frame]).abs().max()
            q_error = quat_error_magnitude(term.robot_body_quat_w, term.motion.body_quat_w[frame].expand(16, -1, -1)).max()
            v_error = (term.robot_body_lin_vel_w - term.motion.body_lin_vel_w[frame]).abs().max()
            print(f"FK frame={frame} pos={p_error.item():.6f}m rot={q_error.item():.6f}rad velocity={v_error.item():.6f}m/s", flush=True)
            assert p_error < .002 and q_error < .005 and v_error < .01
            term.time_steps[:] = frame
            term._update_relative_pose()
            assert mdp.motion_joint_position_error_exp(env, "motion", .2).min() > .999
        env.reset()
        env.episode_length_buf.random_(0, env.max_episode_length)
        assert not mdp.motion_time_out(env).any(), "Runner's randomized episode buffer must not advance the reference"
        env.common_step_counter += 1
        term._update_command()
        assert term.time_steps.eq(2).all(), "100 Hz data must advance 2 frames per 50 Hz step"
        previous = robot.data.root_state_w[1:].clone()
        term._resample_command(torch.tensor([0], device=env.device))
        assert term.time_steps[0] == 0 and term.time_steps[1:].eq(2).all()
        assert torch.equal(previous, robot.data.root_state_w[1:])
        env.common_step_counter += 200
        previous = robot.data.root_state_w.clone()
        term._update_command()
        assert term.time_steps.eq(340).all() and mdp.motion_time_out(env).all()
        assert torch.equal(previous, robot.data.root_state_w), "End of clip must not teleport the robot"
        env.reset()
        term.cfg.start_probability = .5
        term._resample_command(torch.arange(16, device=env.device))
        assert term.time_steps.eq(0).any() and term.time_steps.gt(0).any()
        assert term.time_steps.max() < 228
        term.cfg.start_probability = 1.
        env.reset()
        for _ in range(40):
            obs, reward, _, _, _ = env.step(torch.zeros(16, 12, device=env.device))
            assert torch.isfinite(reward).all() and all(torch.isfinite(v).all() for v in obs.values())
        training_motion = NlegsCrouchEnvCfg().commands.motion
        for field in ("start_probability", "pose_range", "velocity_range", "joint_position_range"):
            setattr(term.cfg, field, getattr(training_motion, field))
        with tempfile.TemporaryDirectory(prefix="nlegs_mimic_check_") as log_dir:
            export_deploy_cfg(env, log_dir, policy_action_clip=5.)
            exported = yaml.safe_load((Path(log_dir) / "params/deploy.yaml").read_text())
            assert exported["real_deployment_supported"] is False and "motion" in exported["commands"]
            assert exported["commands"]["motion"]["motion_file"] == cfg.commands.motion.motion_file
            assert exported["commands"]["motion"]["track_heading"] is True
            agent = NlegsCrouchPPORunnerCfg()
            agent.device = env.device
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
            runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=log_dir, device=env.device)
            runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)
            if args.export_check_dir:
                output = args.export_check_dir
                output.mkdir(parents=True, exist_ok=False)
                export_deploy_cfg(env, str(output), policy_action_clip=agent.clip_actions)
                dump_yaml(str(output / "params/agent.yaml"), agent)
                (output / "exported").mkdir()
                torch.jit.script(runner.alg.actor.as_jit()).save(str(output / "exported/policy.pt"))
                model = runner.alg.actor.as_onnx(verbose=False).cpu()
                torch.onnx.export(model, torch.zeros(1, model.input_size), str(output / "exported/policy.onnx"),
                                  opset_version=11, input_names=["obs"], output_names=["actions"])
                print(f"SMOKE ONLY, not a trained crouch policy: {output}", flush=True)
        print("PASS: registration, 69/138D, named FK/velocity, 100-to-50 Hz clock, RSI/partial reset, end clamp, finite rollout, export and 2 PPO iterations", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        app.close()
