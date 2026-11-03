"""Run the compiled driver against two PTYs only. Never opens robot devices."""
from pathlib import Path
import os
import pty
import select
import signal
import struct
import subprocess
import threading
import time

import numpy as np
import rclpy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / 'deploy/control_ws/install/armcontrol/lib/armcontrol/arm_control_node'
SEND = struct.Struct('<2sBBIIBIBBBB8sB')
RX = struct.Struct('<BBBI8sB')
IDS = [1, 2, 3, 4, 6, 7]
SIGNS = np.array([1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, 1])


def test_driver_commands_feedback_and_warning_ids_use_twelve_real_motors(tmp_path, monkeypatch):
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    peers, handles, stop = [], [], threading.Event()
    commands = [{}, {}]
    failures = []
    paths = []
    for side in range(2):
        master, slave = pty.openpty()
        paths.append(os.ttyname(slave))
        assert paths[-1].startswith('/dev/pts/')
        handles.extend([master, slave])

        def firmware(fd=master, leg=side):
            buf = bytearray()
            try:
                while not stop.is_set():
                    if not select.select([fd], [], [], .1)[0]: continue
                    buf.extend(os.read(fd, 4096))
                    while len(buf) >= SEND.size:
                        packet = bytes(buf[:SEND.size]); del buf[:SEND.size]
                        header, _, _, _, _, _, can_id, _, _, _, _, data, _ = SEND.unpack(packet)
                        assert header == b'\x55\xaa'
                        slave_id = data[0] if can_id == 0x7ff else can_id
                        if slave_id not in IDS: continue
                        index = IDS.index(slave_id)
                        if can_id != 0x7ff and data[-1] not in (0xfc, 0xfd, 0xfe, 0xfb):
                            commands[leg][slave_id] = int.from_bytes(data[:2], 'big') / 65535 * 25 - 12.5
                        q = int((.1 * (index + 1) + 12.5) / 25 * 65535)
                        v = int((.2 + 45) / 90 * 4095)
                        effort = int((3. + 30) / 60 * 4095)
                        feedback = bytes([0x10 | slave_id, q >> 8, q & 255, v >> 4,
                                          (v & 15) << 4 | effort >> 8, effort & 255, 25, 26])
                        os.write(fd, RX.pack(0xaa, 0x11, 8, slave_id + 16, feedback, 0x55))
            except Exception as error:
                failures.append(error)
        peer = threading.Thread(target=firmware)
        peers.append(peer); peer.start()

    env = dict(os.environ, ROS_DOMAIN_ID='213', ROS_LOCALHOST_ONLY='1')
    log = (tmp_path / 'driver.log').open('w+')
    # The driver sees only the PTY paths created above, including during enable.
    proc = subprocess.Popen([str(DRIVER), '--ros-args',
        '-p', f'left_arm_device:={paths[0]}', '-p', f'right_arm_device:={paths[1]}'], env=env,
        stdout=log, stderr=subprocess.STDOUT)
    context = rclpy.Context()
    context.init(domain_id=213)
    node = rclpy.create_node('offline_driver_loopback', context=context)
    from rclpy.executors import SingleThreadedExecutor
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    states, health, warnings = [], [], []
    subscriptions = [
        node.create_subscription(JointState, '/left_joint_states', states.append, 10),
        node.create_subscription(Float64MultiArray, '/sysid_motor_health', health.append, 10),
        node.create_subscription(String, '/motor_warn', warnings.append, 10),
    ]
    publisher = node.create_publisher(Float64MultiArray, '/dog_joint_pos', 1)
    try:
        end = time.monotonic() + 8
        while not states and time.monotonic() < end and proc.poll() is None:
            executor.spin_once(timeout_sec=.1)
        assert proc.poll() is None and states, f'Driver failed or no virtual feedback: {failures}'
        assert states[-1].name == [f"{side}{i}" for side in "LR" for i in range(1, 7)]
        np.testing.assert_allclose(states[-1].position, np.tile(np.arange(1, 7) * .1, 2) * SIGNS, atol=.001)
        np.testing.assert_allclose(states[-1].velocity, .2 * SIGNS, atol=.03)
        np.testing.assert_allclose(states[-1].effort, 3 * SIGNS, atol=.03)
        target = np.arange(1, 13) * .02
        message = Float64MultiArray(data=target.tolist())
        end = time.monotonic() + 3
        while time.monotonic() < end and not all(len(c) == 6 for c in commands):
            publisher.publish(message)
            executor.spin_once(timeout_sec=.1)
        sent = np.array([commands[side][slave] for side in (0, 1) for slave in IDS])
        np.testing.assert_allclose(sent, target * SIGNS, atol=.001)
        previous = [dict(c) for c in commands]
        invalid = target.copy(); invalid[11] = float('nan')
        publisher.publish(Float64MultiArray(data=invalid.tolist()))
        until = time.monotonic() + .15
        while time.monotonic() < until: executor.spin_once(timeout_sec=.02)
        assert commands == previous
        assert health and len(health[-1].data) == 48
        assert all(np.array(health[-1].data).reshape(12, 4)[:, 1] == 1)
        import json
        assert all(len(json.loads(w.data)['errors']) <= 6 for w in warnings)
        assert not failures
    finally:
        proc.send_signal(signal.SIGINT)
        try: proc.wait(timeout=5)
        except subprocess.TimeoutExpired: proc.kill(); proc.wait()
        stop.set()
        for peer in peers: peer.join()
        for handle in handles: os.close(handle)
        executor.shutdown(); node.destroy_node(); context.shutdown()
        log.close()
    assert proc.returncode == 0, 'Driver thread did not stop cleanly'
