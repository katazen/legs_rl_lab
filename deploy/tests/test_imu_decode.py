"""Decode synthetic IMU frames; no serial port or ROS node is opened."""
from pathlib import Path
import sys
import struct
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'imu_ws/src/wit_ros2_imu'))
from wit_ros2_imu import wit_ros2_imu as imu


def feed(kind, values, bad=False):
    frame = bytearray([0x55, kind]) + struct.pack('<4h', *values)
    frame.append((sum(frame) + int(bad)) & 255)
    result = None
    for byte in frame: result = imu.handle_serial_data(byte)
    return result


def test_units_quaternion_and_checksum_are_preserved():
    imu.key, imu.buff = 0, {}
    feed(0x51, [16384, 0, -16384, 0])
    np.testing.assert_allclose(imu.acceleration, [78.4, 0, -78.4])
    feed(0x52, [16384, 0, -16384, 0])
    np.testing.assert_allclose(imu.angularVelocity, [1000 * np.pi / 180, 0, -1000 * np.pi / 180])
    assert feed(0x53, [16384, 0, 0, 0])
    np.testing.assert_allclose(imu.angle_degree, [90, 0, 0])
    assert not feed(0x53, [0, 0, 0, 0], bad=True)
    np.testing.assert_allclose(imu.angle_degree, [90, 0, 0])
    np.testing.assert_allclose(imu.get_quaternion_from_euler(np.pi, 0, 0), [1, 0, 0, 0], atol=1e-7)


def test_policy_imu_installation_rotation_is_independent_of_motor_signs():
    from rl_real_py.rl_real_common import RL_real
    node = SimpleNamespace(obs_raw=np.zeros(31), _last_imu_rx=None)
    message = SimpleNamespace(angular_velocity=SimpleNamespace(x=1, y=2, z=3),
                              orientation=SimpleNamespace(w=0, x=1, y=0, z=0))
    RL_real._on_imu(node, message)
    np.testing.assert_allclose(node.obs_raw[:7], [1, -2, -3, 1, 0, 0, 0])
