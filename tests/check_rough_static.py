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
from legs_rl_lab.tasks.nlegs_task.task.rough.static_env_cfg import CROUCH_LIMITS as STATIC_LIMITS
from legs_rl_lab.utils.export_deploy_cfg import export_deploy_cfg


def main():
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    play = load_cfg_from_registry(args.task, "play_env_cfg_entry_point")
    agent = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    old = load_cfg_from_registry("nlegs_flat", "env_cfg_entry_point")
    is_static = args.task == "nlegs_rough_static"
    task_limits = STATIC_LIMITS if is_static else CROUCH_LIMITS
    assert set(CROUCH_LIMITS) == {f"joint_{side}{j}" for side in "LR" for j in (1, 2, 3, 5)}
    pitch_delays = {"ankle_pitch": (1, 8)} if is_static else {
        "ankle_pitch_left": (3, 3), "ankle_pitch_right": (2, 2),
    }
    delay_bounds = {"legs": (4, 6), "knees": (1, 6), "ankle_roll": (2, 6), **pitch_delays}
    gain_bounds = {"ankle_roll": ((34, 44), (0.25, 0.5))}
    gain_bounds.update({name: ((28, 44), (1.4, 2.4)) if is_static else ((40, 40), (2, 2))
                       for name in pitch_delays})
    assert play.scene.num_envs == 32
    assert agent.experiment_name == args.task and agent.algorithm.symmetry_cfg is None
    assert load_cfg_from_registry("nlegs_flat", "rsl_rl_cfg_entry_point").algorithm.symmetry_cfg is not None
    for config in (cfg, play):
        assert config.events.crouch_joint_limits.params["joint_limits"] == task_limits
        assert bool(config.observations.policy.gait_phase.params.get("gate_by_cmd", False)) == is_static
        assert config.scene.robot.actuators["knees"].min_delay == 1
        if not is_static:
            assert config.events.randomize_ankle_gains is None
            assert config.events.base_com.mode == "startup"
            assert not any(cls.__name__ in ("FlatEnvCfg", "FlatPlayEnvCfg") for cls in type(config).__mro__)
    assert cfg.scene.terrain.terrain_generator.num_rows == 10
    assert play.scene.terrain.terrain_generator.num_rows == 5
    common = yaml.safe_load((Path(__file__).resolve().parents[1] / "deploy/rl_real_py/configs/common.yaml").read_text())
    calibration = common["crouch_calibration"]
    for name, bounds in task_limits.items():
        short = name.removeprefix("joint_")
        side = 0 if calibration["stop_sides"][short] == "lower" else 1
        assert bounds[side] == calibration["joint_pos"][short] and bounds[1 - side] is None
    assert old.actions.JointPositionAction.clip is None
    assert old.actions.JointPositionAction.class_type is mdp.JointPositionAction
    assert load_cfg_from_registry("nlegs_rough_static", "env_cfg_entry_point").actions.JointPositionAction.class_type is mdp.JointPositionAction
    assert cfg.actions.JointPositionAction.class_type is (
        mdp.JointPositionAction if is_static else mdp.ZeroBiasJointPositionAction
    )
    assert not hasattr(old.events, "crouch_joint_limits")
    assert not hasattr(old.events, "base_com")
    assert not hasattr(load_cfg_from_registry("nlegs_rough_static", "env_cfg_entry_point").events, "base_com")
    assert old.scene.robot.actuators["legs"].min_delay == 4
    assert old.scene.robot.actuators["legs"].stiffness[".*4"] == 250
    assert old.scene.robot.actuators["ankle_pitch"].min_delay == 4
    assert old.scene.robot.actuators["ankle_roll"].min_delay == 4
    assert old.events.randomize_ankle_gains.params["damping_distribution_params"] == (0.9, 1.5)
    assert old.rewards.base_height.func is mdp.base_height_l2
    expected_asset = "nlegs_limit" if is_static else "nlegs"
    assert Path(cfg.scene.robot.spawn.usd_path).parts[-3:] == ("mjcf", expected_asset, f"{expected_asset}.usd")
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
        if not is_static:
            base = robot.body_names.index("base")
            com_before = robot.root_physx_view.get_coms().clone()
            base_prim = next(p for p in stage.Traverse()
                             if p.GetName() == "base" and p.HasAPI(UsdPhysics.MassAPI))
            nominal = torch.tensor(tuple(UsdPhysics.MassAPI(base_prim).GetCenterOfMassAttr().Get()))
            offset = com_before[:, base, :3] - nominal
            bounds = torch.tensor([cfg.events.base_com.params["com_range"][axis] for axis in "xyz"])
            assert (offset >= bounds[:, 0] - 1e-6).all() and (offset <= bounds[:, 1] + 1e-6).all()
            assert (offset.std(dim=0) > 0).all(), "Base CoM must vary across environments"
            assert env.event_manager.get_term_cfg("base_com").params["asset_cfg"].body_ids == [base]
        covered = [joint for actuator in robot.actuators.values() for joint in actuator.joint_names]
        assert len(covered) == len(set(covered)) == 12 and set(covered) == set(robot.joint_names)
        if not is_static:
            for name, joint, armature, friction in (
                ("ankle_pitch_left", "joint_L5", 0.035, (0.51, 0.26, 0.0)),
                ("ankle_pitch_right", "joint_R5", 0.0509, (0.55, 0.385, 0.0)),
            ):
                actuator = robot.actuators[name]
                assert actuator.joint_names == [joint]
                j = robot.joint_names.index(joint)
                np.testing.assert_allclose(robot.data.joint_armature[:, j].cpu(), armature, rtol=1e-6)
                for field, value in zip(("friction", "dynamic_friction", "viscous_friction"), friction):
                    np.testing.assert_allclose(getattr(robot.data, f"joint_{field}_coeff")[:, j].cpu(), value, atol=1e-7)
                assert actuator.effort_limit.eq(26).all() and actuator.velocity_limit.eq(7).all()
                assert actuator._saturation_effort == 26
        seen_delays = {name: set() for name in robot.actuators}
        for _ in range(12):
            env.reset()
            for name, actuator in robot.actuators.items():
                lags = actuator.positions_delay_buffer.time_lags
                assert (lags >= actuator.cfg.min_delay).all() and (lags <= actuator.cfg.max_delay).all()
                seen_delays[name].update(lags.cpu().tolist())
            for name, (kp, kd) in gain_bounds.items():
                actuator = robot.actuators[name]
                assert (actuator.stiffness >= kp[0]).all() and (actuator.stiffness <= kp[1]).all()
                assert (actuator.damping >= kd[0]).all() and (actuator.damping <= kd[1]).all()
        for name, bounds in delay_bounds.items():
            assert seen_delays[name] == set(range(bounds[0], bounds[1] + 1))
        if not is_static:
            com_after = robot.root_physx_view.get_coms().clone()
            # 质量随机化重算惯量时主轴四元数可有浮点变化；这里验证质心位置不累加。
            assert torch.equal(com_before[..., :3], com_after[..., :3]), "Reset must not accumulate CoM offsets"
            print("PASS base CoM: per-environment offsets within XYZ bounds; unchanged after 12 resets", flush=True)
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
        if not is_static:
            # 部分 reset 只重采样对应环境；同一偏置必须覆盖观测、目标及实际 PD 链路。
            saved_bias = env._joint_zero_bias.clone()
            env.reset(env_ids=torch.tensor([0, 2], device=env.device))
            assert not torch.equal(saved_bias[[0, 2]], env._joint_zero_bias[[0, 2]])
            assert torch.equal(saved_bias[1::2], env._joint_zero_bias[1::2])
            assert torch.equal(saved_bias[4:], env._joint_zero_bias[4:])
            assert env._joint_zero_bias.abs().le(0.05).all()
            for bias_value in (0.0, 0.05, -0.05):
                env._joint_zero_bias.fill_(bias_value)
                # 不同环境/关节也必须取各自的偏置。
                env._joint_zero_bias[1::2, ::2] *= -1
                action.process_actions(torch.zeros((16, 12), device=env.device))
                command = action.processed_actions.clone()
                true_pos = robot.data.joint_pos.clone()
                encoder_pos = mdp.joint_pos_rel_biased(env) + expected_default
                torch.testing.assert_close(encoder_pos, true_pos + env._joint_zero_bias)
                torch.testing.assert_close(mdp.joint_pos_rel(env), true_pos - expected_default)
                for _ in range(1 + max(hi for _, hi in delay_bounds.values())):
                    action.apply_actions()
                    robot.write_data_to_sim()  # 穿过原生延迟缓冲及执行器计算，不推进物理状态。
                torch.testing.assert_close(robot.data.joint_pos_target, command - env._joint_zero_bias)
                assert torch.equal(action.processed_actions, command), "Bias must not accumulate per substep"
                for actuator in robot.actuators.values():
                    ids = actuator.joint_indices
                    expected_pd = (actuator.stiffness * (command[:, ids] - encoder_pos[:, ids])
                                   - actuator.damping * robot.data.joint_vel[:, ids])
                    torch.testing.assert_close(actuator.computed_effort, expected_pd, atol=2e-5, rtol=1e-5)
                assert torch.equal(robot.data.joint_pos, true_pos)
            # 原限幅仍在编码器坐标；不能用已知随机偏置偷偷消除物理顶限位工况。
            for sign in (-1, 1):
                env._joint_zero_bias.fill_(-sign * 0.05)
                action.process_actions(torch.full((16, 12), sign * 100., device=env.device))
                action.apply_actions()
                torch.testing.assert_close(robot.data.joint_pos_target,
                                           action.processed_actions - env._joint_zero_bias)
            del env._joint_zero_bias
            action.apply_actions()
            torch.testing.assert_close(robot.data.joint_pos_target, action.processed_actions)
            env._joint_zero_bias = saved_bias
            assert torch.equal(robot.data.default_joint_pos, expected_default)
            assert torch.equal(robot.data.joint_pos_limits, limits)
            print("PASS zero bias: paired observation/PD target; both signs; no accumulation; partial reset; encoder clip; absent-bias fallback", flush=True)
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
        rollout_bias = env._joint_zero_bias.clone()
        for _ in range(60):
            obs, rewards, terminated, truncated, _ = env.step(torch.zeros((16, 12), device=env.device))
            assert torch.isfinite(rewards).all() and all(torch.isfinite(v).all() for v in obs.values())
            continuing = ~(terminated | truncated)
            assert torch.equal(env._joint_zero_bias[continuing], rollout_bias[continuing])
            rollout_bias = env._joint_zero_bias.clone()
        assert torch.equal(limits, robot.data.joint_pos_limits)
        with tempfile.TemporaryDirectory(prefix="rough-static-check-") as directory:
            export_deploy_cfg(env, directory, policy_action_clip=agent.clip_actions)
            exported = yaml.safe_load((Path(directory) / "params/deploy.yaml").read_text())
            np.testing.assert_array_equal(exported["actions"]["JointPositionAction"]["clip"], action._clip[0].cpu())
            np.testing.assert_allclose(exported["actions"]["JointPositionAction"]["offset"],
                                       expected_default[0].cpu(), atol=5e-4, rtol=0)
            assert bool(exported["observations"]["gait_phase"]["params"].get("gate_by_cmd", False)) == is_static
            for name, bounds in delay_bounds.items():
                assert exported["actuators"][name]["delay"] == list(bounds)
            if not is_static:
                for name, joint in (("ankle_pitch_left", "joint_L5"), ("ankle_pitch_right", "joint_R5")):
                    assert exported["actuators"][name]["joint_ids"] == [exported["joint_names"].index(joint)]
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
