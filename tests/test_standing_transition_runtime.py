"""CPU checks for stop-phase observations; imports neither ROS nor MuJoCo.

python tests/test_standing_transition_runtime.py
"""

import ast
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "deploy/rl_real_py/rl_real_py/rl_real_common.py"
SIM = ROOT / "source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/task/flat/sim2sim.py"


def load_runtime(source, selected):
    # Compile the actual observation methods without importing hardware/UI modules.
    nodes = []
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name == "gravity_from_quat":
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in selected:
            methods = selected[node.name]
            if methods is not None:
                node.body = [child for child in node.body
                             if isinstance(child, ast.FunctionDef) and child.name in methods]
            nodes.append(node)
    namespace = {"np": np, "yaml": yaml, "os": os, "Path": Path, "SIM_DURATION": 10000.0}
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace


REAL_NS = load_runtime(REAL, {"Policy": {"__init__", "_build_terms"}, "TermGroupedHistory": None})
SIM_NS = load_runtime(SIM, {"MujocoRunner": {"__init__", "_observation_features", "run"}})
Policy = REAL_NS["Policy"]
Policy.motion = None
Policy._load_policy = lambda self, path: None
Runner = SIM_NS["MujocoRunner"]
SIM_NS["gravity_from_quat"] = REAL_NS["gravity_from_quat"]


def load_training_phase():
    namespace = {"torch": torch, "math": math}
    mdp = ROOT / "source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/mdp"
    for filename, names in (("gait.py", {"command_is_moving", "gait_motion_weight", "get_phase"}),
                            ("observations.py", {"gait_phase_obs"})):
        source = mdp / filename
        nodes = [node for node in ast.parse(source.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
        assert len(nodes) == len(names)
        module = ast.Module(body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *nodes,
        ], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["gait_phase_obs"]


TRAINING_PHASE = load_training_phase()


def phase_params(gated=True, duration=1.2):
    params = {"gate_by_cmd": gated, "command_threshold": 1e-6}
    if duration is not None:
        params["stop_transition_time"] = duration
    return params


def make_real(params):
    joints = [f"{side}{index}" for side in "RL" for index in range(1, 7)]
    observations = {
        "velocity_commands": {"scale": [1.0] * 3, "history_length": 15},
        "gait_phase": {"scale": [1.0] * 2, "history_length": 15, "params": params},
    }
    deploy = {
        "default_joint_pos": [0.0] * 12,
        "actions": {"JointPositionAction": {"scale": [0.25] * 12}},
        "gait_period": 0.6, "step_dt": 0.02, "observations": observations,
        "commands": {"base_velocity": {"ranges": {key: [-1.0, 1.0]
                     for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z")}}},
        "joint_names": [f"joint_{joint}" for joint in joints], "joint_ids_map": list(range(12)),
    }
    with TemporaryDirectory() as folder:
        path = Path(folder) / "params"
        path.mkdir()
        (path / "deploy.yaml").write_text(json.dumps(deploy))
        (path / "agent.yaml").write_text("{}")
        policy = Policy({"joint_index_in_real": joints}, folder, ROOT)
    policy.obs_raw[3] = 1.0
    return policy


def make_sim(params):
    runner = Runner.__new__(Runner)
    runner.cfg = SimpleNamespace(
        step_dt=0.02, gait_period=0.6,
        observations={"gait_phase": {"params": params}},
        policy_to_sdk=np.arange(12), default_policy=np.zeros(12, np.float32),
    )
    runner.gait_stop_transition_time = float(params.get("stop_transition_time", 0.0))
    runner.episode_step = 0
    runner._gait_last_motion_time = -np.inf
    runner._gait_last_step = -1
    runner.command = np.zeros(3, np.float32)
    runner.last_action = np.zeros(12, np.float32)
    runner._joint_state = lambda: (np.zeros(12, np.float32), np.zeros(12, np.float32))
    runner.data = SimpleNamespace(sensor=lambda name: SimpleNamespace(
        data=np.array([1.0, 0.0, 0.0, 0.0]) if name == "imu_quat" else np.zeros(3)))
    return runner


def make_training():
    command = torch.zeros(1, 3)
    return SimpleNamespace(
        num_envs=1, device="cpu", step_dt=0.02, command=command,
        episode_length_buf=torch.zeros(1, dtype=torch.long),
        command_manager=SimpleNamespace(get_command=lambda _: command),
        cfg=SimpleNamespace(gait=SimpleNamespace(period=0.6)),
    )


def check_frame(real, sim, step, command, expected_weight, training=None):
    real.policy_step = sim.episode_step = step
    real.cmd[:] = sim.command[:] = command
    expected = np.array([np.sin(2 * np.pi * step * 0.02 / 0.6),
                         np.cos(2 * np.pi * step * 0.02 / 0.6)]) * expected_weight
    for _ in range(2):  # Logging/extra observation reads cannot advance the fade.
        actual_real = real._build_terms()[-1]
        actual_sim = sim._observation_features()["gait"]
        np.testing.assert_allclose(actual_real, expected, atol=2e-7, rtol=0.0)
        np.testing.assert_allclose(actual_sim, expected, atol=2e-7, rtol=0.0)
        np.testing.assert_allclose(actual_real, actual_sim, atol=2e-7, rtol=0.0)
        if training is not None:
            training.episode_length_buf[:] = step
            training.command[:] = torch.tensor(command)
            actual_training = TRAINING_PHASE(training, **sim.cfg.observations["gait_phase"]["params"])[0]
            np.testing.assert_allclose(actual_training.numpy(), actual_real, atol=2e-6, rtol=0.0)


def test_transition_and_restart():
    real, sim = make_real(phase_params()), make_sim(phase_params())
    training = make_training()
    for step, command, weight in (
        (0, [0, 0, 0], 0.0),
        (1, [0.2, 0, 0], 1.0),
        (2, [0, 0, 0], 59 / 60),
        (31, [0, 0, 0], 0.5),
        (61, [0, 0, 0], 0.0),
        (62, [0, 0, -0.1], 1.0),  # Pure yaw remains motion.
        (63, [0, 0, 0], 59 / 60),
        (64, [1e-5, 0, 0], 1.0),  # Small commands are not silently discarded.
        (65, [5e-7, 0, 0], 59 / 60),
    ):
        check_frame(real, sim, step, command, weight, training)
    check_frame(real, sim, 0, [0, 0, 0], 0.0, training)  # Reset step count.
    check_frame(real, sim, 0, [0.2, 0, 0], 1.0)
    real.hist = REAL_NS["TermGroupedHistory"](real.term_dims, real.num_history)
    # Sim run restart clears fade even if no time elapsed (zero-length test run).
    sim.data.time = 0.0
    sim.viewer = sim.listener = None
    sim.save_data = False
    sim._start_keyboard = lambda: None
    sim.run(duration=0.0, realtime=False)
    check_frame(real, sim, 0, [0, 0, 0], 0.0)


def test_legacy_and_ungated():
    for gated, duration in ((True, None), (True, 0.0), (False, 0.0), (False, 1.2)):
        params = phase_params(gated, duration)
        real, sim = make_real(params), make_sim(params)
        training = make_training()
        check_frame(real, sim, 0, [0, 0, 0], 0.0 if gated else 1.0, training)
        check_frame(real, sim, 1, [0, 0, 0.2], 1.0, training)
        check_frame(real, sim, 2, [0, 0, 0], 0.0 if gated else 1.0, training)


def test_invalid_duration():
    for value in (-1.0, float("nan"), float("inf"), -float("inf")):
        params = phase_params(duration=value)
        for constructor in (
            lambda: make_real(params),
            lambda: Runner(SimpleNamespace(observations={"gait_phase": {"params": params}})),
        ):
            try:
                constructor()
            except ValueError as exc:
                assert "stop_transition_time" in str(exc)
            else:
                raise AssertionError(f"accepted invalid duration {value}")


if __name__ == "__main__":
    test_transition_and_restart()
    test_legacy_and_ungated()
    test_invalid_duration()
    print("PASS: Isaac/real/MuJoCo stop-phase parity, restart, repeated reads and legacy behavior")
