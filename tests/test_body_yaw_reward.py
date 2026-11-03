"""CPU checks for base yaw cost gated by small turning commands.

python tests/test_body_yaw_reward.py
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import torch


SOURCE = (Path(__file__).resolve().parents[1]
          / "source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/task/rough")


def load_definition(filename, name, namespace):
    # Run the actual tensor code without starting Isaac; integration uses check_rough_body.py.
    path = SOURCE / filename
    definitions = [node for node in ast.parse(path.read_text()).body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name]
    assert len(definitions) == 1
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                              *definitions], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


REWARD = load_definition("body_rewards.py", "base_yaw_no_command_l2", {
    "torch": torch, "SceneEntityCfg": lambda name: SimpleNamespace(name=name),
})


def test_yaw_only_symmetric_cost_and_small_command_gate():
    angular = torch.tensor([[9., -7., 0.], [9., -7., 2.], [9., -7., -2.],
                            [0., 0., 2.], [0., 0., -2.], [0., 0., 2.], [0., 0., -2.]])
    command = torch.tensor([[0., 0., 0.], [.5, -.3, 0.], [-.3, .3, 0.],
                            [.5, 0., 1e-7], [.5, 0., -1e-7], [0., 0., .5], [0., 0., -.5]])
    def get_command(name):
        assert name == "base_velocity"
        return command
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=SimpleNamespace(root_ang_vel_b=angular))},
        command_manager=SimpleNamespace(get_command=get_command),
    )
    result = REWARD(env)
    torch.testing.assert_close(result, torch.tensor([0., 4., 4., 4., 4., 0., 0.]))
    assert result.shape == (7,) and result.dtype == angular.dtype and torch.isfinite(result).all()
    env.scene["other"] = env.scene.pop("robot")
    torch.testing.assert_close(REWARD(env, SimpleNamespace(name="other")), result)


def test_command_threshold_boundaries_and_translation_independence():
    yaw = torch.tensor([0., .099, -.099, .1, -.1, .101, -.101, .5, -.5], dtype=torch.float64)
    angular = torch.zeros((len(yaw), 3), dtype=torch.float64)
    angular[:, 2] = 2.0
    command = torch.zeros_like(angular)
    command[:, 2] = yaw
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=SimpleNamespace(root_ang_vel_b=angular))},
        command_manager=SimpleNamespace(get_command=lambda _: command),
    )
    expected = torch.tensor([4., 4., 4., 0., 0., 0., 0., 0., 0.], dtype=torch.float64)
    torch.testing.assert_close(REWARD(env), expected)
    command[:, :2] = torch.tensor([.5, -.3], dtype=torch.float64)
    original_command = command.clone()
    torch.testing.assert_close(REWARD(env), expected)
    torch.testing.assert_close(command, original_command)
    angular[:, 2] *= -1
    torch.testing.assert_close(REWARD(env), expected)


if __name__ == "__main__":
    tests = [value for name, value in globals().copy().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"PASS: {len(tests)} base yaw checks (axes, symmetry, strict +/-0.1 command gate, translation independence)")
