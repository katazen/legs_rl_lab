"""CPU-only timing checks; no simulator or robot connection."""
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "gait", ROOT / "source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/mdp/gait.py"
)
gait = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gait)


def test_timing_and_independent_resets():
    command = torch.zeros(3, 3)
    env = SimpleNamespace(
        episode_length_buf=torch.zeros(3, dtype=torch.long), step_dt=0.02,
        command_manager=SimpleNamespace(get_command=lambda _: command),
    )
    weight = lambda: gait.gait_motion_weight(env, 1e-6, 1.2, update_state=True)
    torch.testing.assert_close(weight(), torch.zeros(3))
    command[0, 0], command[1, 2] = 0.2, 0.01
    env.episode_length_buf[:] = 1
    torch.testing.assert_close(weight(), torch.tensor([1., 1., 0.]))
    command.zero_()
    for step in (2, 15, 31, 60, 61, 80):
        env.episode_length_buf[:] = step
        expected = max(0., 1 - (step - 1) * .02 / 1.2)
        target = torch.tensor([expected, expected, 0.])
        torch.testing.assert_close(weight(), target)
        torch.testing.assert_close(weight(), target)  # reward + actor + critic
    command[0, 1] = -1e-4
    torch.testing.assert_close(weight(), torch.tensor([1., 0., 0.]))
    command.zero_()
    env.episode_length_buf[0] = 0
    torch.testing.assert_close(weight(), torch.zeros(3))
    command[1, 0] = .1
    weight()
    command.zero_()
    env.episode_length_buf[1] += 1
    torch.testing.assert_close(weight(), torch.tensor([0., 1 - .02 / 1.2, 0.]))
    # Other environments resetting must not erase an in-flight transition.
    env.episode_length_buf[2] = 0
    torch.testing.assert_close(weight(), torch.tensor([0., 1 - .02 / 1.2, 0.]))
    for bad in (-1., math.inf, math.nan):
        try:
            gait.gait_motion_weight(env, 1e-6, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid transition time {bad}")
    torch.testing.assert_close(gait.gait_motion_weight(env, 1e-6, 0.), torch.zeros(3))


if __name__ == "__main__":
    test_timing_and_independent_resets()
    print("PASS: stop timing, idempotence, partial resets, restart, yaw, tiny commands, legacy gate")
