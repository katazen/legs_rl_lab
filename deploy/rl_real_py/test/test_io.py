"""离线检查 ROS 边界、配置与启动顺序；不连接机器人。"""

from pathlib import Path
import csv
import os
import pty
import queue
import select
import subprocess
import threading
import tty
from types import SimpleNamespace

import numpy as np
import pytest

from rl_real_py import rl_real_common as module
from rl_real_py.rl_real_common import RL_real
from rl_real_py.deployment_config import load_settings


def make_node(monkeypatch):
    monkeypatch.setattr(module.time, "monotonic", lambda: 100.0)
    n = object.__new__(RL_real)
    n.real_joint_names = [f"{leg}{i}" for leg in "LR" for i in range(1, 7)]
    n.lo, n.hi = np.full(12, -1.), np.full(12, 1.)
    n.lo[10] = -.4
    n.obs_raw = np.zeros(31, np.float32)
    n.obs_raw[3] = 1.
    n._last_joint_rx = n._last_imu_rx = 100.
    n.state_timeout = .2
    n.target_real = n.target_pub = np.zeros(12, np.float32)
    n.tick, n.pub_dt = 0, .005
    n.cmd = np.zeros(3, np.float32)
    n._kb_cmd = np.zeros(3, np.float32)
    n._joy_cmd = np.zeros(3, np.float32)
    n._last_joy_rx = -1e9
    n._prev_buttons = []
    n.deadzone = .12
    n.use_derived_vel = False
    n._motor_vel = n._motor_tau = np.zeros(12, np.float32)
    n._read_keys = lambda: None
    n.sent, n.warnings, n.log_events = [], [], []
    n.pub = SimpleNamespace(publish=lambda msg: n.sent.append(np.array(msg.data)))
    n.get_logger = lambda: SimpleNamespace(warn=lambda text, **kw: n.warnings.append(text))
    n._open_log = lambda *args: n.log_events.append("open")
    n._close_log = lambda: n.log_events.append("close")
    n._log_row = lambda *args: None
    return n


def test_keyboard_nonblocking_pty_idle_input_and_split_utf8(monkeypatch):
    master, slave = pty.openpty()
    received = []
    n = SimpleNamespace(multi=SimpleNamespace(keys=received.append))
    try:
        with os.fdopen(slave, "r", encoding="utf-8") as stream:
            tty.setcbreak(stream.fileno())
            os.set_blocking(stream.fileno(), False)
            monkeypatch.setattr(module.sys, "stdin", stream)
            for _ in range(200):
                RL_real._read_keys(n)  # 空终端曾在 TextIOWrapper.read 中抛 TypeError。
            assert not any(received)
            for data, expected in ((b"3p1r", "3p1r"), (b"\xe4", ""),
                                   (b"\xbd\xa0\x1b[A3", "\x1b[A3")):
                os.write(master, data)
                assert select.select([stream], [], [], 1.)[0]
                RL_real._read_keys(n)
                assert received[-1] == expected
                RL_real._read_keys(n)
                assert received[-1] == ""
    finally:
        os.close(master)


def test_async_log_writer(tmp_path):
    path = tmp_path / "log.csv"
    rows = queue.SimpleQueue()
    worker = threading.Thread(target=RL_real._write_log, args=(path, ["a", "b"], rows))
    worker.start()
    rows.put([1, 2])
    rows.put([3, 4])
    rows.put(None)
    worker.join(2)
    assert not worker.is_alive()
    with path.open() as stream:
        assert list(csv.reader(stream)) == [["a", "b"], ["1", "2"], ["3", "4"]]


@pytest.mark.parametrize("bad", ["missing_joint", "stale_joint", "missing_imu", "stale_imu",
                                  "nan_q", "inf_qd", "nan_imu", "zero_quat", "nonunit_quat", "outside"])
def test_invalid_current_state_cannot_be_captured(monkeypatch, bad):
    n = make_node(monkeypatch)
    if bad in ("missing_joint", "stale_joint"):
        n._last_joint_rx = None if bad.startswith("missing") else 99.
    elif bad in ("missing_imu", "stale_imu"):
        n._last_imu_rx = None if bad.startswith("missing") else 99.
    elif bad == "zero_quat":
        n.obs_raw[3:7] = 0.
    elif bad == "nonunit_quat":
        n.obs_raw[3] = 2.
    else:
        index, value = {"nan_q": (7, np.nan), "inf_qd": (19, np.inf),
                        "nan_imu": (0, np.nan), "outside": (17, -.425)}[bad]
        n.obs_raw[index] = value
    assert not n._capture_current_target() and n.warnings
    assert not n.sent
    np.testing.assert_array_equal(n.target_pub, 0.)


@pytest.mark.parametrize("positions,velocities", [(11, 12), (12, 11)])
def test_short_joint_state_is_rejected_without_crashing(monkeypatch, positions, velocities):
    n = make_node(monkeypatch)
    old = n.obs_raw.copy()
    n._on_joint(SimpleNamespace(position=[0.] * positions, velocity=[0.] * velocities, effort=[]))
    assert n._last_joint_rx is None
    np.testing.assert_array_equal(n.obs_raw, old)
    assert not n._capture_current_target()


@pytest.mark.parametrize("state", ["wait_current", "wait_feedback", "checking", "stand_ready",
                                    "crouch_ready", "walking", "lowering", "rising", "stopped"])
def test_publish_always_clips_to_common(monkeypatch, state):
    n = make_node(monkeypatch)
    n.mode = state
    for target, expected in ((n.lo - .1, n.lo), (n.hi + .1, n.hi)):
        n.target_pub = target.copy()
        n._publish_target()
        np.testing.assert_array_equal(n.sent[-1], expected.astype(np.float32))
    assert n.mode == state


@pytest.mark.parametrize("target", [np.full(12, np.nan), np.full(12, np.inf), np.zeros(11)])
def test_publish_rejects_invalid_target(monkeypatch, target):
    n = make_node(monkeypatch)
    n.target_pub = target
    n._publish_target()
    assert not n.sent and n.warnings


@pytest.mark.parametrize("fail_at,check_only", [(0, True), (0, False), (1, False), (2, False), (3, False)])
def test_launcher_prechecks_before_any_window(fail_at, check_only):
    deploy = Path(__file__).resolve().parents[2]
    shell = r'''
python3() { echo "PY|$*"; case "$*" in
 *preflight_only*) [[ "$FAIL_AT" != 1 ]] ;;
 *--check-only*) [[ "$FAIL_AT" != 2 ]] ;;
 *sync_pd*) [[ "$FAIL_AT" != 3 ]] ;;
 esac; }
gnome-terminal() { echo "WINDOW|$*"; }
sleep() { :; }
export -f python3 gnome-terminal sleep
bash "$@"
'''
    env = dict(os.environ, FAIL_AT=str(fail_at))
    command = ["bash", "-c", shell, "test", str(deploy / "start_real.sh")]
    if check_only:
        command.append("--check-only")
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    assert result.returncode == (1 if fail_at else 0)
    lines = result.stdout.splitlines()
    windows = [i for i, line in enumerate(lines) if line.startswith("WINDOW|")]
    assert len(windows) == (0 if fail_at or check_only else 3)
    calls = [line for line in lines if line.startswith("PY|")]
    assert calls and all("common.yaml" in call for call in calls)
    if windows:
        assert all(i < windows[0] for i, line in enumerate(lines) if line.startswith("PY|"))
        assert "wit_ros2_imu" in lines[windows[0]]
        assert "arm_control_node" in lines[windows[1]]
        assert "rl_real_common" in lines[windows[2]]


def test_single_config_and_legacy_entry_rejection(tmp_path):
    deploy = Path(__file__).resolve().parents[2]
    configs = deploy / "rl_real_py/configs"
    assert sorted(p.name for p in configs.glob("*.yaml")) == ["common.yaml"]
    assert not hasattr(RL_real, "_tick_prepare")
    for name in ("start_now.sh", "start_mimic.sh"):
        assert not (deploy / name).exists()
    for option in ("--multi", "--start-from-current"):
        result = subprocess.run(["bash", str(deploy / "start_real.sh"), option], capture_output=True, text=True)
        assert result.returncode == 2 and "未知参数" in result.stderr
    old = tmp_path / "old.yaml"
    old.write_text("run: old\nlogs_root: logs\n")
    with pytest.raises(ValueError, match="旧单策略配置已退役"):
        load_settings(old)
