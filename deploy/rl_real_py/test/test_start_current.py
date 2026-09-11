"""Offline state-machine checks: no Node initialization, ROS graph or motor I/O."""

from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from rl_real_py import rl_real_common as module
from rl_real_py.rl_real_common import RL_real, TermGroupedHistory


def make_node(monkeypatch, current=True):
    monkeypatch.setattr(module.time, "monotonic", lambda: 100.0)
    node = object.__new__(RL_real)
    node.start_from_current = current
    node.mode = "wait_current" if current else "prepare"
    node.real_joint_names = [f"{leg}{i}" for leg in "LR" for i in range(1, 7)]
    node.lo, node.hi = np.full(12, -1.0), np.full(12, 1.0)
    node.lo[10] = -0.4
    node.obs_raw = np.zeros(31, np.float32)
    node.obs_raw[0:3] = [0.01, 0.02, 0.03]
    node.obs_raw[3] = 1.0
    node.obs_raw[7:19] = np.linspace(0.1, 0.3, 12)
    node.obs_raw[19:31] = np.linspace(0.01, 0.12, 12)
    node._last_joint_rx = node._last_imu_rx = 100.0
    node.state_timeout = 0.2
    node.default_sim = np.full(12, 0.05, np.float32)
    node.real2sim = [6, 0, 7, 1, 8, 2, 9, 3, 10, 4, 11, 5]
    node.sim2real = np.argsort(node.real2sim)
    node.default_real = node.default_sim[node.sim2real].copy()
    node.target_real = node.default_real.copy()
    node.target_pub = node.default_real.copy()
    node.settle_bias = np.zeros(12, np.float32)
    node.settle_bias_max = np.full(12, 0.1)
    node.settle_ki, node.settle_tol, node.settle_time = 0.01, 0.01, 3.0
    node.prepare_t0 = node.settle_t0 = node.run_t0 = node.q_start_real = None
    node.prepare_time = 4.0
    node.tick, node.decimation, node.pub_dt = 0, 1, 0.005
    node.target_ema_alpha = 0.5
    node.num_actions, node.num_commands, node.num_history = 12, 3, 3
    node.action_scale, node.action_clip = np.full(12, 0.25), 5.0
    node.last_action = np.zeros(12, np.float32)
    node.gait_period, node.gait_gate_by_cmd = 0.8, False
    node.obs_names = ["base_ang_vel", "projected_gravity", "velocity_commands",
                      "joint_pos_rel", "joint_vel_rel", "last_action", "gait_phase"]
    node.term_dims = [3, 3, 3, 12, 12, 12, 2]
    node.term_scales = [np.ones(d, np.float32) for d in node.term_dims]
    node.hist = TermGroupedHistory(node.term_dims, node.num_history)
    node.cmd = np.zeros(3, np.float32)
    node.cmd_bias = np.zeros(3, np.float32)
    node._kb_cmd = np.zeros(3, np.float32)
    node._joy_cmd = np.zeros(3, np.float32)
    node._last_joy_rx = -1e9
    node._prev_buttons = []
    node.deadzone = 0.12
    node.cmd_min, node.cmd_max = np.full(3, -0.5), np.full(3, 0.5)
    node._read_keys = lambda: None
    node._update_cmd = lambda: None
    node._mirror_report = lambda q: None
    node.sent, node.inputs, node.warnings, node.log_events = [], [], [], []
    node.pub = SimpleNamespace(publish=lambda msg: node.sent.append(np.array(msg.data)))
    node.get_logger = lambda: SimpleNamespace(warn=lambda text, **kw: node.warnings.append(text))
    node._open_log = lambda: node.log_events.append("open")
    node._close_log = lambda: node.log_events.append("close")
    node._log_row = lambda terms: None

    def infer(x):
        node.inputs.append(x.copy())
        return np.full(12, 0.2, np.float32)

    node._infer = infer
    return node


def test_current_hold_start_uses_real_state_and_nominal_policy_frame(monkeypatch):
    n = make_node(monkeypatch)
    nominal = n.default_sim.copy()
    q = n.obs_raw[7:19].copy()
    n._tick()
    assert n.mode == "hold" and n.prepare_t0 is None and n.settle_t0 is None
    np.testing.assert_array_equal(n.sent[-1], q)
    assert not n.inputs and not n.log_events
    n._tick()
    np.testing.assert_array_equal(n.sent[-1], q)

    n.obs_raw[7:19] += 0.02  # P captures the current state, not the first startup snapshot.
    q = n.obs_raw[7:19].copy()
    n._kb_cmd[:] = n.cmd[:] = 0.3
    n._start_run()
    assert n.mode == "run"
    np.testing.assert_array_equal(n.cmd, 0.0)
    n._tick()
    np.testing.assert_array_equal(n.sent[-1], q)
    assert not n.inputs and n.log_events == ["open"]
    expected_terms = n._build_terms()
    n._tick()
    assert len(n.inputs) == 1
    for buf, term in zip(n.hist.buffers, expected_terms):
        np.testing.assert_allclose(buf, np.tile(term, (n.num_history, 1)), atol=1e-7)
    np.testing.assert_allclose(n.hist.buffers[3][0], q[n.real2sim] - nominal)
    np.testing.assert_array_equal(n.hist.buffers[5], 0.0)
    np.testing.assert_allclose(n.sent[-1], q + 0.5 * (0.1 - q), atol=1e-7)
    np.testing.assert_array_equal(n.default_sim, nominal)
    np.testing.assert_array_equal(n.default_real, nominal[n.sim2real])


@pytest.mark.parametrize("bad", ["missing_joint", "stale_joint", "missing_imu", "stale_imu",
                                  "nan_q", "inf_qd", "nan_imu", "zero_quat", "nonunit_quat", "outside"])
def test_invalid_current_state_never_publishes_or_starts(monkeypatch, bad):
    n = make_node(monkeypatch)
    if bad in ("missing_joint", "stale_joint"):
        n._last_joint_rx = None if bad.startswith("missing") else 99.0
    elif bad in ("missing_imu", "stale_imu"):
        n._last_imu_rx = None if bad.startswith("missing") else 99.0
    elif bad == "zero_quat":
        n.obs_raw[3:7] = 0.0
    elif bad == "nonunit_quat":
        n.obs_raw[3] = 2.0
    else:
        index, value = {"nan_q": (7, np.nan), "inf_qd": (19, np.inf),
                        "nan_imu": (0, np.nan), "outside": (17, -0.425)}[bad]
        n.obs_raw[index] = value
    n._tick()
    assert n.mode == "wait_current" and not n.sent and not n.inputs
    n.mode = "hold"
    n._start_run()
    assert n.mode == "hold" and not n.inputs
    assert n.warnings
    if bad == "outside":
        assert "R5=-0.42500" in n.warnings[-1]


@pytest.mark.parametrize("positions,velocities", [(11, 12), (12, 11)])
def test_short_joint_state_is_rejected_without_crashing(monkeypatch, positions, velocities):
    n = make_node(monkeypatch)
    old = n.obs_raw.copy()
    n._on_joint(SimpleNamespace(position=[0.0] * positions, velocity=[0.0] * velocities, effort=[]))
    assert n._last_joint_rx is None
    np.testing.assert_array_equal(n.obs_raw, old)
    n._tick()
    assert not n.sent


def test_pause_resume_and_reset_never_return_to_default(monkeypatch):
    n = make_node(monkeypatch)
    n._tick()
    n._on_joy(SimpleNamespace(axes=[], buttons=[1, 0, 0]))  # A starts.
    n._tick()
    n._tick()
    n.obs_raw[7:19] += 0.03
    n._on_joy(SimpleNamespace(axes=[], buttons=[0, 1, 0]))  # B locks actual q.
    assert n.mode == "hold"
    n._tick()
    np.testing.assert_array_equal(n.sent[-1], n.obs_raw[7:19])
    n._start_run()
    n._tick()
    frozen = n.target_pub.copy()
    n._last_joint_rx = 99.0
    n._stop_run()  # No valid q: retain the last actual published target.
    n._tick()
    np.testing.assert_array_equal(n.sent[-1], frozen)
    n._on_joy(SimpleNamespace(axes=[], buttons=[0, 0, 1]))  # X rearms, does not auto-run.
    assert n.mode == "wait_current" and n.run_t0 is None
    count = len(n.sent)
    n._tick()
    assert len(n.sent) == count
    n._last_joint_rx = 100.0
    n.obs_raw[7:19] += 0.01
    n._tick()
    assert n.mode == "hold"
    np.testing.assert_array_equal(n.sent[-1], n.obs_raw[7:19])


def test_invalid_run_state_freezes_actual_ema_target(monkeypatch):
    n = make_node(monkeypatch)
    n._tick()
    n._start_run()
    n._tick()
    n._tick()
    frozen, calls = n.target_pub.copy(), len(n.inputs)
    n.obs_raw[19] = np.nan
    n._tick()
    np.testing.assert_array_equal(n.sent[-1], frozen)
    assert len(n.inputs) == calls


def test_start_rechecks_freshness_before_first_run_publish(monkeypatch):
    n = make_node(monkeypatch)
    n._tick()
    n._start_run()
    count = len(n.sent)
    n._last_joint_rx = 99.0
    n._tick()
    assert len(n.sent) == count and not n.inputs and not n.log_events


def test_default_prepare_settle_hold_and_run_are_unchanged(monkeypatch):
    n = make_node(monkeypatch, current=False)
    q = n.obs_raw[7:19].copy()
    n._last_joint_rx = None
    n._tick()
    assert not n.sent
    n._last_joint_rx = 100.0
    n._tick()
    assert n.mode == "prepare"
    np.testing.assert_array_equal(n.sent[-1], q)
    n.prepare_t0 = 96.0
    n._tick()
    assert n.mode == "settle"
    np.testing.assert_array_equal(n.sent[-1], n.default_real)
    n.obs_raw[7:19] = n.default_real
    n._tick()
    assert n.mode == "hold"
    n._tick()
    np.testing.assert_array_equal(n.sent[-1], n.default_real)
    n._start_run()
    n._tick()
    assert len(n.inputs) == 1  # Original mode still infers on its first eligible tick.
    n._stop_run()
    n._tick()
    np.testing.assert_array_equal(n.sent[-1], n.default_real)
    n._reset()
    assert n.mode == "prepare"


@pytest.mark.parametrize("script,current", [("start_real.sh", False), ("start_now.sh", True)])
@pytest.mark.parametrize("fail_python", [False, True])
def test_launchers_only_opt_in_new_rl_mode(script, current, fail_python):
    deploy = Path(__file__).resolve().parents[2]
    # Only print terminal arguments: never execute the contained ROS commands or sync_pd.
    mock_shell = r'''
gnome-terminal() { printf 'WINDOW'; printf '|%s' "$@"; printf '\n'; }
sleep() { :; }
python3() { :; }
export -f gnome-terminal sleep python3
bash "$1"
'''
    if fail_python:
        mock_shell = mock_shell.replace("python3() { :; }", "python3() { return 1; }")
    result = subprocess.run(["bash", "-c", mock_shell, "launcher-test", str(deploy / script)],
                            capture_output=True, text=True)
    windows = [line for line in result.stdout.splitlines() if line.startswith("WINDOW|")]
    if current and fail_python:
        assert result.returncode == 1 and not windows
        return
    assert result.returncode == 0
    assert len(windows) == 3
    assert "wit_ros2_imu" in windows[0]
    assert "arm_control_node" in windows[1]
    assert "rl_real_common" in windows[2]
    assert not any("start_from_current:=" in line for line in windows[:2])
    assert ("--ros-args -p start_from_current:=true" in windows[2]) == current
