"""Operator velocity-command ramping."""

import numpy as np


def limit_command_acceleration(current, target, max_accel, dt):
    """Limit only increasing command magnitude; zero/deceleration stays immediate."""
    current = np.asarray(current, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    step = np.asarray(max_accel, dtype=np.float32) * max(float(dt), 0.0)
    out = target.copy()
    reversing = current * target < 0.0
    rising = (~reversing) & (target != 0.0) & (np.abs(target) > np.abs(current))
    out[reversing] = 0.0
    out[rising] = current[rising] + np.clip(
        target[rising] - current[rising], -step[rising], step[rising]
    )
    return out


if __name__ == "__main__":
    rate = np.array([0.5, 0.5, 1.0], np.float32)
    np.testing.assert_allclose(
        limit_command_acceleration(np.zeros(3), np.array([0.3, -0.3, 0.5]), rate, 0.1),
        [0.05, -0.05, 0.1], atol=1e-6,
    )
    np.testing.assert_allclose(
        limit_command_acceleration(np.array([0.2, -0.2, 0.2]), np.zeros(3), rate, 0.1),
        0.0,
    )
    np.testing.assert_allclose(
        limit_command_acceleration(np.array([0.2, -0.2, 0.2]), np.array([0.1, -0.1, -0.3]), rate, 0.1),
        [0.1, -0.1, 0.0], atol=1e-6,
    )
    print("cmd ramp self-check passed")
