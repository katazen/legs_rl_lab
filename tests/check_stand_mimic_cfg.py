"""Checks for stand task registration, configuration overrides and measured motion.

python tests/check_stand_mimic_cfg.py
python tests/check_stand_mimic_cfg.py --isaac  # 16-env headless reset/limits/PPO smoke test
"""

import ast
import hashlib
from pathlib import Path
import runpy
import sys
import xml.etree.ElementTree as ET

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "source/legs_rl_lab/legs_rl_lab/tasks/mimic_task"
TASK = TASKS / "task/nlegs_stand"
PACKAGE = "legs_rl_lab.tasks.mimic_task.task.nlegs_stand"


def main():
    runpy.run_path(str(TASK / "__init__.py"), run_name=PACKAGE)
    spec = gym.spec("nlegs_mimic_stand")
    tree = ast.parse((TASK / "tracking_env_cfg.py").read_text())
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    for key, name in (("env_cfg_entry_point", "NlegsStandEnvCfg"),
                      ("play_env_cfg_entry_point", "NlegsStandPlayEnvCfg")):
        assert spec.kwargs[key] == f"{PACKAGE}.tracking_env_cfg:{name}"
        assert name in classes
    assert ast.unparse(classes["NlegsStandEnvCfg"].bases[0]) == "NlegsCrouchEnvCfg"
    assert ast.unparse(classes["NlegsStandPlayEnvCfg"].bases[0]) == "NlegsStandEnvCfg"
    assignments = {ast.unparse(node.targets[0]): node.value for node in ast.walk(classes["NlegsStandEnvCfg"])
                   if isinstance(node, ast.Assign)}
    motion_expression = assignments.pop("self.commands.motion.motion_file")
    assert ast.unparse(motion_expression) == "str(Path(__file__).parent / 'motions/crouch_to_stand_v2.npz')"
    assert {key: ast.literal_eval(value) for key, value in assignments.items()} == {
        "self.commands.motion.sample_until_s": 2.6,
        "self.events.push_robot.params['motion_time_range_s']": (.5, 2.6),
        "self.episode_length_s": 3.35,
    }
    play = {ast.unparse(node.targets[0]): ast.literal_eval(node.value)
            for node in ast.walk(classes["NlegsStandPlayEnvCfg"]) if isinstance(node, ast.Assign)}
    assert play["self.commands.motion.start_probability"] == 1.
    assert play["self.commands.motion.pose_range"] == play["self.commands.motion.velocity_range"] == {}
    assert play["self.commands.motion.joint_position_range"] == (0., 0.)
    assert play["self.events.push_robot"] is None and not play["self.observations.policy.enable_corruption"]
    agents = ast.parse((TASKS / "agents/rsl_rl_ppo_cfg.py").read_text())
    agent = next(node for node in agents.body if isinstance(node, ast.ClassDef) and node.name == "NlegsStandPPORunnerCfg")
    assert ast.unparse(agent.bases[0]) == "NlegsCrouchPPORunnerCfg"
    assert ast.literal_eval(agent.body[0].value) == "nlegs_mimic_stand"
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(":NlegsStandPPORunnerCfg")

    motion = TASK / "motions/crouch_to_stand_v2.npz"
    with np.load(motion, allow_pickle=False) as data:
        assert float(data["fps"]) == 100 and len(data["joint_pos"]) == 336
        assert (len(data["joint_pos"]) - 1) / float(data["fps"]) == 3.35
        names = [f"joint_{side}{i}" for side in "LR" for i in range(1, 7)]
        assert data["joint_names"].tolist() == names
        assert str(data["velocity_frame"]) == "world_link_origin"
        np.testing.assert_allclose(data["joint_pos"][0], [
            -.975242237, .496490425, .498397803, 1.056115053, -.376325628, -.156650639,
            -.967612726, -.527008469, -.618562600, 1.091210803, -.422484169, .094029268,
        ], atol=1e-6, rtol=0.)
        np.testing.assert_allclose(data["joint_pos"][-1], [-.1, 0., 0., .2, -.1, 0.] * 2, atol=1e-6)
        for key in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
            np.testing.assert_allclose(data[key][[0, -1]], 0., atol=1e-6)
        assert all(np.isfinite(data[key]).all() for key in data.files if data[key].dtype.kind == "f")
        model = ROOT / "source/legs_rl_lab/legs_rl_lab/assets/nlegs/mjcf/nlegs_limit.xml"
        assert str(data["model_sha256"]) == hashlib.sha256(model.read_bytes()).hexdigest()
        ranges = {j.get("name"): tuple(map(float, j.get("range").split()))
                  for j in ET.parse(model).findall(".//worldbody//joint")}
        limits = np.asarray([ranges[name] for name in names])
        assert (data["joint_pos"] >= limits[:, 0] - 1e-6).all()
        assert (data["joint_pos"] <= limits[:, 1] + 1e-6).all()
    assert (TASK / "motions/crouch_to_stand_v1.npz").is_file(), "Keep old reference for old policies"
    print("PASS: registration, v2 measured endpoint, 3.35s timing, Play overrides, XML limits/hash and finite full-body data.")


def check_isaac():
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True).app
    import tempfile
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    import legs_rl_lab.tasks
    from legs_rl_lab.tasks.mimic_task.task.nlegs_stand.tracking_env_cfg import (
        NlegsStandEnvCfg, NlegsStandPlayEnvCfg,
    )
    from legs_rl_lab.tasks.mimic_task.task.nlegs_crouch.tracking_env_cfg import NlegsCrouchEnvCfg, joint_ranges, ASSET
    from legs_rl_lab.tasks.mimic_task.agents.rsl_rl_ppo_cfg import NlegsStandPPORunnerCfg

    cfg = NlegsStandEnvCfg()
    assert cfg.scene.robot.spawn.func is sim_utils.spawn_from_usd
    assert NlegsCrouchEnvCfg().commands.motion.sample_until_s == 2.28
    assert NlegsStandPlayEnvCfg().events.push_robot is None
    cfg.scene.num_envs = 16
    env = gym.make("nlegs_mimic_stand", cfg=cfg).unwrapped
    try:
        term, robot = env.command_manager.get_term("motion"), env.scene["robot"]
        ranges = joint_ranges(ASSET / "mjcf/nlegs_limit.xml")
        expected = torch.tensor([ranges[n] for n in robot.joint_names], device=env.device)
        torch.testing.assert_close(robot.data.joint_pos_limits, expected.expand(16, -1, -1), atol=1e-6, rtol=0.)
        torch.testing.assert_close(env.action_manager.get_term("JointPositionAction")._clip,
                                   expected.expand(16, -1, -1), atol=1e-6, rtol=0.)
        term.cfg.start_probability = 1.
        term.cfg.joint_position_range = (0., 0.)
        obs, _ = env.reset(seed=42)
        assert obs["policy"].shape == (16, 69) and obs["critic"].shape == (16, 138)
        torch.testing.assert_close(robot.data.joint_pos, term.motion.joint_pos[0].expand(16, -1))
        assert term.time_steps.eq(0).all() and term.motion.duration == 3.35
        env.common_step_counter += 168
        term._update_command()
        assert term.time_steps.eq(335).all(), "Odd last frame must not be skipped by 50Hz policy"
        term.cfg.start_probability = .5
        term.cfg.joint_position_range = (-.01, .01)
        env.reset()
        assert term.time_steps.max() < 260
        push = env.event_manager.get_term_cfg("push_robot").func.push_at
        finite = torch.isfinite(push)
        assert finite.any() and (push[finite] >= .5).all() and (push[finite] < 2.6).all()
        with tempfile.TemporaryDirectory(prefix="nlegs-stand-check-") as directory:
            agent = NlegsStandPPORunnerCfg()
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
            runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=directory, device=env.device)
            runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)
        print("PASS: 16-env USD/XML/action limits, measured crouch reset, 69/138D, final-frame clock, push timing and 2 PPO iterations.", flush=True)
    finally:
        env.close()
        app.close()


if __name__ == "__main__":
    main()
    if "--isaac" in sys.argv:
        sys.argv.remove("--isaac")
        check_isaac()
