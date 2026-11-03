"""CPU checks for the shared swing-clearance reward's per-asset sole offset."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import torch


MDP = (Path(__file__).resolve().parents[1]
       / "source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/mdp")


def load_reward():
    namespace = {"torch": torch}
    for name, selected in (("gait.py", {"get_phase", "command_is_moving"}),
                           ("rewards.py", {"_get_leg_phases", "feet_clearance"})):
        source = MDP / name
        functions = [node for node in ast.parse(source.read_text()).body
                     if isinstance(node, ast.FunctionDef) and node.name in selected]
        assert len(functions) == len(selected)
        module = ast.Module(body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *functions,
        ], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["feet_clearance"]


REWARD = load_reward()
FEET = SimpleNamespace(name="robot", body_ids=[2, 0])
SENSOR = SimpleNamespace(name="height_scanner")


class Scene(dict):
    pass


def make_env(offset, terrain):
    # Right swing peak; left stance. Cover zero, partial, target and excess clearance.
    clearance = torch.tensor([0.0, 0.03, 0.06, 0.09], dtype=torch.float64)
    ground = torch.tensor([0.4, -0.2] if terrain else [0.0, 0.0], dtype=torch.float64)
    feet = torch.full((4, 3, 3), 100.0, dtype=torch.float64)  # Unselected body is irrelevant.
    feet[:, [2, 0], 0] = torch.tensor([-0.5, 0.5], dtype=torch.float64)
    feet[:, [2, 0], 1] = 0.0
    feet[:, [2, 0], 2] = ground + offset
    feet[:, 2, 2] += clearance
    hits = torch.zeros((4, 2, 3), dtype=torch.float64)
    hits[:, :, 0] = torch.tensor([0.5, -0.5], dtype=torch.float64)
    hits[:, :, 2] = ground.flip(0)  # Ray order differs from the foot order.
    scene = Scene(robot=SimpleNamespace(data=SimpleNamespace(body_pos_w=feet)))
    scene.sensors = {SENSOR.name: SimpleNamespace(data=SimpleNamespace(ray_hits_w=hits))}
    return SimpleNamespace(
        scene=scene, device="cpu", step_dt=0.02,
        episode_length_buf=torch.full((4,), 0.775 * 30.0, dtype=torch.float64),
        cfg=SimpleNamespace(gait=SimpleNamespace(period=0.6, stance_ratio=0.55, feet_offset=[0.0, 0.5])),
    )


def test_same_physical_clearance_across_assets_and_terrain():
    expected = torch.tensor([(1.0 + math.exp(-height**2 / 0.005)) / 2.0
                             for height in (0.06, 0.03, 0.0, 0.0)], dtype=torch.float64)
    for offset in (0.0, 0.0135, 0.03):
        for terrain in (False, True):
            env = make_env(offset, terrain)
            actual = REWARD(env, FEET, target_height=0.06,
                            sensor_cfg=SENSOR if terrain else None, foot_height_offset=offset)
            torch.testing.assert_close(actual, expected)


def test_default_preserves_old_sole_offset():
    for terrain in (False, True):
        env = make_env(0.0135, terrain)
        params = {"target_height": 0.06, "sensor_cfg": SENSOR if terrain else None}
        default = REWARD(env, FEET, **params)
        explicit = REWARD(env, FEET, **params, foot_height_offset=0.0135)
        torch.testing.assert_close(default, explicit, rtol=0.0, atol=0.0)
        # A wrong offset must change the swing-foot shortfall, not merely be accepted.
        wrong = REWARD(env, FEET, **params, foot_height_offset=0.03)
        assert wrong[0] < default[0]


if __name__ == "__main__":
    test_same_physical_clearance_across_assets_and_terrain()
    test_default_preserves_old_sole_offset()
    print("PASS: shared feet clearance offsets, terrain-relative heights and legacy default")
