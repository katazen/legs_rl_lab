"""CPU checks for sway tolerance rewards; no Isaac app or robot assets required.

python tests/test_body_sway_rewards.py
"""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/mdp/rewards.py"
NAMES = {
    "flat_orientation_l2_with_tolerance",
    "ang_vel_xy_l2_with_tolerance",
    "base_lateral_move_l2_with_tolerance",
}


def load_rewards():
    # Isolate pure tensor functions so this check never imports Isaac or an asset.
    tree = ast.parse(SOURCE.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in NAMES]
    assert {node.name for node in functions} == NAMES
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                              *functions], type_ignores=[])
    namespace = {"torch": torch, "SceneEntityCfg": lambda name: SimpleNamespace(name=name)}
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in NAMES})


def make_env(**data):
    count = next(iter(data.values())).shape[0]
    commands = torch.zeros((count, 3), dtype=torch.float64)
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=SimpleNamespace(**data))},
        command_manager=SimpleNamespace(get_command=lambda _: commands),
    )
    return env, commands


def test_flat_orientation_tolerance():
    rewards = load_rewards()
    threshold = math.radians(8.0)
    angles = torch.tensor([0.0, threshold / 2, threshold, threshold + 0.2,
                           -(threshold + 0.2), math.pi], dtype=torch.float64)
    gravity = torch.stack((angles.sin(), torch.zeros_like(angles), -angles.cos()), dim=1)
    env, _ = make_env(projected_gravity_b=gravity)
    expected = torch.tensor([0.0, 0.0, 0.0, 0.04, 0.04, (math.pi - threshold) ** 2],
                            dtype=torch.float64)
    torch.testing.assert_close(rewards.flat_orientation_l2_with_tolerance(env, threshold), expected)
    # Changing the horizontal direction (including yaw) preserves total tilt.
    gravity[:, :2] = gravity[:, :2].flip(1)
    torch.testing.assert_close(rewards.flat_orientation_l2_with_tolerance(env, threshold), expected)
    gravity[0, 2] = -1.0 - 1e-10
    gravity[-1, 2] = 1.0 + 1e-10
    torch.testing.assert_close(rewards.flat_orientation_l2_with_tolerance(env, threshold), expected)


def test_angular_speed_tolerance():
    rewards = load_rewards()
    angular = torch.tensor([[0.0, 0.0, 5.0], [0.21, 0.28, 0.0], [0.3, 0.4, 3.0],
                            [-0.3, -0.4, -3.0]], dtype=torch.float64)
    env, _ = make_env(root_ang_vel_b=angular)
    torch.testing.assert_close(
        rewards.ang_vel_xy_l2_with_tolerance(env, 0.35),
        torch.tensor([0.0, 0.0, 0.15 ** 2, 0.15 ** 2], dtype=torch.float64),
    )


def test_lateral_speed_tolerance():
    rewards = load_rewards()
    lateral = torch.tensor([0.0, 0.05, 0.1, 0.3, -0.3, 0.3, 0.3, 0.3], dtype=torch.float64)
    linear = torch.stack((torch.full_like(lateral, 2.0), lateral, torch.full_like(lateral, 3.0)), dim=1)
    env, commands = make_env(root_lin_vel_b=linear)
    commands[:, 1] = torch.tensor([0.0, 0.0, 0.0, 0.099, -0.099, 0.1, -0.1, 0.2],
                                 dtype=torch.float64)
    torch.testing.assert_close(
        rewards.base_lateral_move_l2_with_tolerance(env, 0.1),
        torch.tensor([0.0, 0.0, 0.0, 0.04, 0.04, 0.0, 0.0, 0.0], dtype=torch.float64),
    )


if __name__ == "__main__":
    test_flat_orientation_tolerance()
    test_angular_speed_tolerance()
    test_lateral_speed_tolerance()
    print("PASS: tilt/angular/lateral tolerance boundaries, symmetry, yaw exclusion, and lateral command gate")
