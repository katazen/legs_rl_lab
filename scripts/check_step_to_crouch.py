"""有界检查：任务隔离、时间表、目标限位、接触反馈、成功判据、部分 reset。

python scripts/check_step_to_crouch.py --headless --device cuda:0
只运行仿真；合成状态测试只验证判据，不代表策略已经学会下蹲。
"""

import argparse
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import traceback
from unittest.mock import patch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
import yaml
import legs_rl_lab.tasks
from legs_rl_lab.tasks.nlegs_task.task.flat.flat_env_cfg import FlatEnvCfg
from legs_rl_lab.tasks.nlegs_task.task.flat.crouch_env_cfg import FlatCrouchEnvCfg
from legs_rl_lab.tasks.nlegs_task.task.flat.step_to_crouch_env_cfg import (
    FlatStepToCrouchEnvCfg, FlatStepToCrouchPlayEnvCfg, NlegsFlatStepToCrouchPPORunnerCfg,
)
from legs_rl_lab.tasks.nlegs_task.task.flat import step_to_crouch_mdp as mdp
from legs_rl_lab.tasks.nlegs_task.agents.rsl_rl_ppo_cfg import NlegsFlatPPORunnerCfg
from legs_rl_lab.utils.export_deploy_cfg import export_deploy_cfg


def main():
    cfg = FlatStepToCrouchEnvCfg()
    assert cfg.events.reset_base is not None and cfg.events.reset_robot_joints is not None
    assert cfg.events.push_robot is None and cfg.curriculum is None
    assert not hasattr(cfg.commands, "base_velocity")
    assert NlegsFlatStepToCrouchPPORunnerCfg().algorithm.symmetry_cfg is None
    assert NlegsFlatPPORunnerCfg().algorithm.symmetry_cfg is not None
    assert FlatEnvCfg().scene.robot.spawn.usd_path != cfg.scene.robot.spawn.usd_path
    assert FlatEnvCfg().events.physics_material.params["static_friction_range"] == (0.1, 1.3)
    assert FlatCrouchEnvCfg().events.reset_crouch is not None
    assert FlatCrouchEnvCfg().events.push_robot is not None
    assert FlatStepToCrouchPlayEnvCfg().commands.to_dict() == cfg.commands.to_dict()
    spec = importlib.util.spec_from_file_location("pose_solver", Path(__file__).with_name("generate_crouch_pose_bank.py"))
    solver_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solver_module)
    solver = solver_module.PoseSolver()
    goal, geometry = solver.solve_limit()
    rpy = solver_module.Rotation.from_quat(geometry["quat"], scalar_first=True).as_euler("xyz")
    assert abs(cfg.commands.crouch_progress.goal_height - (geometry["height"] - solver_module.CLEARANCE)) < 1e-5
    assert abs(cfg.commands.crouch_progress.goal_roll - rpy[0]) < 1e-5
    assert abs(cfg.commands.crouch_progress.goal_pitch - rpy[1]) < 1e-5

    cfg.scene.num_envs = 16
    cfg.sim.device = args.device
    cfg.episode_length_s = 99.0  # 模拟命令行覆盖时间后旧的名义长度，初始化须重新对齐。
    env = gym.make("nlegs_flat_step_to_crouch", cfg=cfg).unwrapped
    try:
        assert env.max_episode_length_s == 12.0
        obs, _ = env.reset(seed=42)
        term = mdp.task(env)
        robot = env.scene["robot"]
        limits = robot.data.joint_pos_limits.clone()
        q = robot.data.joint_pos.clone()
        env.episode_length_buf.random_(0, env.max_episode_length)
        assert torch.count_nonzero(term.command) == 0
        assert not mdp.time_out(env).any()
        for elapsed, expected in ((0, 0), (1, 0), (5, 0.5), (9, 1), (12, 1)):
            term.start_step[:] = env.common_step_counter - round(elapsed / env.step_dt)
            assert torch.allclose(term.command, torch.full_like(term.command, expected))
        assert mdp.time_out(env).all()
        assert torch.count_nonzero(mdp.gait_obs(env)) == 0
        assert torch.count_nonzero(mdp.stepping(env)) == 0
        assert torch.equal(robot.data.joint_pos, q)
        assert torch.equal(robot.data.joint_pos_limits, limits)
        assert torch.count_nonzero(mdp.joint_limits(env)) == 0
        target = robot.data.default_joint_pos.clone()
        target[:, term.joints] = term.goal
        robot.write_joint_state_to_sim(target, torch.zeros_like(target))
        assert torch.count_nonzero(mdp.joint_limits(env)) == 0
        assert not term.stable().any()  # 只有关节到位，无稳定双脚接触，不能成功。

        obs, _ = env.reset(seed=42)
        touched = torch.zeros(2, device=env.device, dtype=torch.bool)
        for _ in range(40):
            obs, rewards, _, _, _ = env.step(torch.zeros_like(q))
            assert torch.isfinite(rewards).all()
            assert all(torch.isfinite(v).all() for v in obs.values())
            force, slip, height, _ = term.foot_state()
            assert force.shape == slip.shape == height.shape == (16, 2)
            touched |= (force > 10).any(dim=0)
            for i, sensor in enumerate(term.sensors):
                points = sensor.data.contact_pos_w[:, 0, 0]
                loaded = force[:, i] > 10
                assert torch.isfinite(points[loaded]).all()
                assert points[loaded, 2].abs().lt(0.01).all()
        assert touched.all(), "左右脚地面过滤器必须都收到真实接触力"

        force = torch.full((16, 2), 40.0, device=env.device)
        zeros = torch.zeros_like(force)
        normals = torch.zeros(16, 2, 3, device=env.device)
        normals[..., 2] = 1
        term.start_step[:] = env.common_step_counter - round(3 / env.step_dt)
        term.steps[:] = 0
        with patch.object(term, "foot_state", return_value=(zeros, zeros, zeros + 0.04, normals)):
            for _ in range(3):
                term._update_metrics()
        with patch.object(term, "foot_state", return_value=(force, zeros, zeros, normals)):
            term._update_metrics()
        assert not term.steps.any(), "双脚一起跳不算有效单脚卸载"
        for leg in (0, 1):
            support = force.clone()
            support[:, leg] = 0
            with patch.object(term, "foot_state", return_value=(support, zeros, zeros + 0.04, normals)):
                for _ in range(2):
                    term._update_metrics()
            with patch.object(term, "foot_state", return_value=(force, zeros, zeros, normals)):
                term._update_metrics()
        assert term.steps.eq(1).all()

        term.start_step[:] = env.common_step_counter - round(9.2 / env.step_dt)
        root = robot.data.default_root_state.clone()
        root[:, :3] = env.scene.env_origins
        root[:, 2] += term.cfg.goal_height
        root[:, 3:7] = torch.tensor(geometry["quat"], device=env.device)
        root[:, 7:] = 0
        order = [solver_module.NAMES.index(n) for n in robot.joint_names]
        target = torch.tensor(goal[order], dtype=torch.float32, device=env.device).repeat(16, 1)
        robot.write_root_state_to_sim(root)
        robot.write_joint_state_to_sim(target, torch.zeros_like(target))
        with patch.object(term, "foot_state", return_value=(force, zeros, zeros, normals)):
            assert term.stable().all()
            term.steps[:] = 2
            term.slip_distance[:] = 0
            term.stable_time[:] = 0
            for _ in range(10):
                term._update_metrics()
            assert not term.metrics["success"].any(), "瞬间到位不能算成功"
            for _ in range(45):
                term._update_metrics()
            assert term.metrics["success"].all()
            term.steps[0] = 0
            term.slip_distance[1] = term.cfg.slip_budget_m + 0.01
            term._update_metrics()
            assert not term.metrics["success"][:2].any(), "不踏步/超滑移预算均不能成功"
            assert mdp.excessive_slip(env)[1]
            target[:, term.joints[0]] += 0.03
            robot.write_joint_state_to_sim(target, torch.zeros_like(target))
            assert not term.stable().any(), "任一目标关节未到位即不能成功"
        before = term.start_step.clone()
        term.slip_distance[:] = 0.01
        env._prev_prev_action = torch.ones_like(q)
        env.reset(env_ids=torch.tensor([0, 3], device=env.device))
        assert term.command[[0, 3]].eq(0).all()
        assert torch.equal(term.start_step[[1, 2]], before[[1, 2]])
        assert term.slip_distance[[0, 3]].eq(0).all()
        assert term.slip_distance[[1, 2]].eq(0.01).all()
        assert env._prev_prev_action[[0, 3]].eq(0).all()
        assert env._prev_prev_action[[1, 2]].eq(1).all()

        with tempfile.TemporaryDirectory(prefix="nlegs-crouch-export-") as directory:
            export_deploy_cfg(env, directory, policy_action_clip=5.0)
            exported = yaml.safe_load((Path(directory) / "params/deploy.yaml").read_text())
            assert exported["commands"]["crouch_progress"]["lower_s"] == 8.0
            assert "base_velocity" not in exported["commands"]
            assert "crouch_progress" in exported["observations"]
        print("PASS: task isolation, XML/USD endpoints, FK goal, independent timing, stop stepping, "
              "ground contacts, finite rollout, strict hold/slip/step success, partial reset, export", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    app.close()
