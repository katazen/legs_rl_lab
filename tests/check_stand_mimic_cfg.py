"""CPU-only checks for stand task registration, configuration overrides and motion.

python tests/check_stand_mimic_cfg.py
Does not launch Isaac Sim or verify PhysX/PPO execution.
"""

import ast
import hashlib
from pathlib import Path
import runpy

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
    assert ast.unparse(motion_expression) == "str(Path(__file__).parent / 'motions/crouch_to_stand_v1.npz')"
    assert {key: ast.literal_eval(value) for key, value in assignments.items()} == {
        "self.commands.motion.sample_until_s": 2.08,
        "self.events.push_robot.params['motion_time_range_s']": (.4, 2.08),
        "self.episode_length_s": 2.68,
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

    motion = TASK / "motions/crouch_to_stand_v1.npz"
    with np.load(motion, allow_pickle=False) as data, np.load(
        TASKS / "task/nlegs_crouch/motions/stand_to_crouch_v2.npz", allow_pickle=False
    ) as original:
        assert float(data["fps"]) == 100 and len(data["joint_pos"]) == 269
        assert (len(data["joint_pos"]) - 1) / float(data["fps"]) == 2.68
        for key in ("joint_names", "body_names", "velocity_frame", "model_sha256"):
            np.testing.assert_array_equal(data[key], original[key])
        for key in ("joint_pos", "body_pos_w", "body_quat_w"):
            np.testing.assert_allclose(data[key], original[key][:269][::-1], atol=1e-6, rtol=0.)
        for key in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
            np.testing.assert_allclose(data[key], -original[key][:269][::-1], atol=1e-6, rtol=0.)
        assert all(np.isfinite(data[key]).all() for key in data.files if data[key].dtype.kind == "f")
        model = ROOT / "source/legs_rl_lab/legs_rl_lab/assets/nlegs/mjcf/nlegs_limit.xml"
        assert str(data["model_sha256"]) == hashlib.sha256(model.read_bytes()).hexdigest()
    print("PASS: Gym registration, config structure/timing, separate experiment, Play overrides and reversed full-body data.")
    print("CPU-only: Isaac initialization and PPO execution have not been tested here.")


if __name__ == "__main__":
    main()
