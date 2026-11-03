"""CPU checks for loaded-foot normal-axis twist; no Isaac app or robot connection.

python tests/test_body_foot_twist_rewards.py
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import torch


SOURCE = (Path(__file__).resolve().parents[1]
          / "source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/task/rough/body_rewards.py")


def load_reward():
    # Execute the actual tensor function, without importing Isaac or loading an asset.
    tree = ast.parse(SOURCE.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "feet_contact_yaw_l2"]
    assert len(functions) == 1
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                              *functions], type_ignores=[])
    namespace = {"torch": torch, "SceneEntityCfg": lambda name, **kwargs: SimpleNamespace(name=name, **kwargs)}
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace["feet_contact_yaw_l2"]


REWARD = load_reward()
ASSET = SimpleNamespace(name="robot", body_ids=[3, 1])
SENSOR = SimpleNamespace(name="contact_forces", body_ids=[0, 4])


class Scene(dict):
    pass


def make_env(count):
    # Contact sensor and articulation have different body counts and index orders.
    angular = torch.zeros((count, 4, 3), dtype=torch.float64)
    forces = torch.zeros((count, 5, 3), dtype=torch.float64)
    scene = Scene(robot=SimpleNamespace(data=SimpleNamespace(body_ang_vel_w=angular)))
    scene.sensors = {"contact_forces": SimpleNamespace(data=SimpleNamespace(net_forces_w=forces))}
    return SimpleNamespace(scene=scene), angular, forces


def test_load_thresholds_ramp_and_saturation():
    env, angular, forces = make_env(6)
    forces[:, 0, 2] = torch.tensor([-10., 0., 5., 12.5, 20., 100.])
    angular[:, 3, 2] = 2.0
    angular[:, 1, 2] = 9.0  # The other, unloaded foot is free to rotate.
    result = REWARD(env, ASSET, SENSOR)
    torch.testing.assert_close(result, torch.tensor([0., 0., 0., 2., 4., 4.], dtype=torch.float64))
    assert result.shape == (6,) and torch.isfinite(result).all()


def test_only_current_vertical_load_counts():
    env, angular, forces = make_env(4)
    angular[:, :, 2] = 3.0
    forces[1:, :, :2] = 1000.0  # A horizontal normal alone is not vertical support.
    forces[1, :, 2] = -100.0
    forces[2, :, 2] = 5.0
    # A previous contact must not suppress a foot that is currently unloaded.
    sensor_data = env.scene.sensors[SENSOR.name].data
    sensor_data.net_forces_w_history = torch.full((4, 4, 5, 3), 1000., dtype=torch.float64)
    result = REWARD(env, ASSET, SENSOR)
    assert torch.isfinite(result).all()  # Includes exactly zero force and normal length.
    torch.testing.assert_close(result, torch.zeros(4, dtype=torch.float64))


def test_roll_pitch_free_yaw_symmetric_and_additive():
    env, angular, forces = make_env(3)
    forces[:, :, 2] = 20.0
    angular[:, :, 0] = 7.0
    angular[:, :, 1] = -9.0
    angular[1, 3, 2], angular[1, 1, 2] = 0.75, -1.25
    angular[2, 3, 2], angular[2, 1, 2] = -0.75, 1.25
    # Tangential friction is a separate sensor field, never the ground-normal direction.
    env.scene.sensors[SENSOR.name].data.friction_forces_w = torch.full((3, 5, 3), 1000., dtype=torch.float64)
    torch.testing.assert_close(
        REWARD(env, ASSET, SENSOR),
        torch.tensor([0., 0.75**2 + 1.25**2, 0.75**2 + 1.25**2], dtype=torch.float64),
    )


def test_slope_normal_projection_and_tangent_rotation():
    env, angular, forces = make_env(5)
    normal = torch.tensor([0.6, 0., 0.8], dtype=torch.float64)
    tangent = torch.tensor([0.8, 0., -0.6], dtype=torch.float64)
    forces[:, 0] = normal * 25.0  # Fz=20 N, full gate.
    angular[0, 3] = 2.0 * normal
    angular[1, 3] = 3.0 * tangent
    forces[2, 0] = normal * (12.5 / 0.8)  # Fz=12.5 N, half gate.
    angular[2, 3] = 2.0 * normal
    forces[3, 0] = torch.tensor([-15., 0., 20.], dtype=torch.float64)
    angular[3, 3] = torch.tensor([-1.2, 0., 1.6], dtype=torch.float64)
    angular[4, 3] = torch.tensor([2., 0., 0.], dtype=torch.float64)
    torch.testing.assert_close(REWARD(env, ASSET, SENSOR),
                               torch.tensor([4., 0., 2., 4., 1.44], dtype=torch.float64))


def test_distinct_sensor_and_asset_indices():
    env, angular, forces = make_env(2)
    angular[:, [0, 2], 2] = 100.0  # Unselected body angular velocities must be ignored.
    angular[:, 3, 2] = 2.0
    angular[:, 1, 2] = 3.0
    forces[:, [1, 2, 3], 2] = 1000.0  # Unselected sensor contacts must also be ignored.
    forces[:, 0, 2] = torch.tensor([20., 5.])
    forces[:, 4, 2] = torch.tensor([12.5, 20.])
    expected = torch.tensor([4. + 0.5 * 9., 9.], dtype=torch.float64)
    torch.testing.assert_close(REWARD(env, ASSET, SENSOR), expected)
    # Reordering both selectors together preserves which load belongs to each foot.
    reversed_asset = SimpleNamespace(name="robot", body_ids=[1, 3])
    reversed_sensor = SimpleNamespace(name="contact_forces", body_ids=[4, 0])
    torch.testing.assert_close(REWARD(env, reversed_asset, reversed_sensor), expected)


def test_custom_thresholds_and_invalid_intervals():
    env, angular, forces = make_env(1)
    angular[0, 3, 2] = 2.0
    forces[0, 0, 2] = 1.0
    torch.testing.assert_close(REWARD(env, ASSET, SENSOR, min_force=0., full_force=2.),
                               torch.tensor([2.], dtype=torch.float64))
    for minimum, full in [(-1., 20.), (5., 5.), (20., 5.), (0., 0.)]:
        try:
            REWARD(env, ASSET, SENSOR, min_force=minimum, full_force=full)
        except ValueError:
            continue
        raise AssertionError(f"Expected ValueError for force interval {(minimum, full)}")


if __name__ == "__main__":
    tests = [value for name, value in globals().copy().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"PASS: {len(tests)} loaded-foot twist checks (load gate, current force, slope normal, symmetry, indexing, thresholds)")
