"""离线验证：运行中异常只告警，手柄 B 才进入中断锁存。"""

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from rl_real_py.multi_task import MultiTaskController
from rl_real_py.rl_real_common import Policy, RL_real
from rl_real_py.deployment_config import load_settings


def test_common_velocity_limits_override_training_limits_without_widening(monkeypatch):
    cfg, run_dir, repo = load_settings(Path(__file__).parent / "configs/common.yaml")
    cfg["velocity_command_limits"] = {
        "lin_vel_x": [-0.1, 0.2], "lin_vel_y": [-0.1, 0.1], "ang_vel_z": [-0.2, 0.2]
    }
    monkeypatch.setattr(Policy, "_load_policy", lambda *_: None)
    policy = Policy(cfg, run_dir, repo)
    np.testing.assert_allclose(policy.cmd_min, [-0.1, -0.1, -0.2])
    np.testing.assert_allclose(policy.cmd_max, [0.2, 0.1, 0.2])
    cfg["velocity_command_limits"]["lin_vel_x"] = [-0.4, 0.2]
    with pytest.raises(ValueError, match="不能超出训练范围"):
        Policy(cfg, run_dir, repo)


def test_initial_status_before_any_blend(capsys):
    m = object.__new__(MultiTaskController)
    m.state = "wait_current"
    m.policies = {"walk": object()}
    m.show_status()
    assert "等待关节/IMU" in capsys.readouterr().out


def test_runtime_faults_do_not_latch_but_gamepad_b_does():
    warnings = []
    n = SimpleNamespace(
        obs_raw=np.zeros(31, np.float32),
        target_pub=np.zeros(12, np.float32),
        target_real=np.zeros(12, np.float32),
        cmd=np.zeros(3, np.float32),
        _kb_cmd=np.zeros(3, np.float32),
        _joy_cmd=np.zeros(3, np.float32),
        _last_joy_rx=-1e9,
        _prev_buttons=[],
        ctrl_timeout=.4,
        deadzone=.12,
        cmd_min=np.full(3, -1., np.float32),
        cmd_max=np.full(3, 1., np.float32),
        tick=4,
        decimation=4,
        mode="walking",
        get_logger=lambda: SimpleNamespace(warn=lambda message, **kwargs: warnings.append(message),
                                           error=lambda message, **kwargs: warnings.append(message)),
        _publish_target=lambda: None,
        _close_log=lambda: None,
    )
    n._clear_cmd_sources = lambda: (n.cmd.fill(0), n._kb_cmd.fill(0), n._joy_cmd.fill(0))
    m = object.__new__(MultiTaskController)
    n.multi = m
    m.n, m.state, m.active, m.has_target = n, "walking", "walk", True
    m.cfg = {"handover_time": .2}
    m.buttons = {"a": 0, "b": 1, "x": 2, "y": 3, "lb": 9, "back": 4, "start": 6}
    m.policies = {"walk": SimpleNamespace(cmd=np.zeros(3, np.float32))}
    m.pending_target = None
    m.activated_at = 0.
    m.key_escape = False
    m.show_status = lambda: None
    m.policy_target = lambda policy: (_ for _ in ()).throw(ValueError("推理失败"))

    n.obs_raw[7] = 100.
    n.cmd_max[:] = [0.2, 0.1, 0.3]
    n._kb_cmd[:] = 1.
    RL_real._on_motor_warn(n, SimpleNamespace(data='{"errors":[{"err":"失能"}]}'))
    m.tick()
    np.testing.assert_allclose(n.cmd, n.cmd_max)
    assert m.state == "walking" and m.active == "walk"
    assert any("推理失败" in message for message in warnings)
    m.keys("p")
    assert m.state == "walking"

    RL_real._on_joy(n, SimpleNamespace(axes=[0.] * 6, buttons=[0, 1] + [0] * 13))
    assert m.state == "stopped" and m.active is None
    np.testing.assert_array_equal(n.target_real, n.target_pub)


def test_key_four_returns_to_stand_then_one_starts_walk():
    published = []
    n = SimpleNamespace(
        obs_raw=np.zeros(31, np.float32),
        target_pub=np.zeros(12, np.float32),
        target_real=np.zeros(12, np.float32),
        cmd=np.zeros(3, np.float32),
        _joy_cmd=np.zeros(3, np.float32),
        _prev_buttons=[],
        mode="crouch_ready",
        get_logger=lambda: SimpleNamespace(warn=lambda message: None),
        _clear_cmd_sources=lambda: None,
        _close_log=lambda: None,
        _open_log=lambda policy: None,
        _publish_target=lambda: published.append(n.target_pub.copy()),
    )
    m = object.__new__(MultiTaskController)
    m.n, m.state, m.active, m.has_target = n, "crouch_ready", None, True
    m.cfg = {"stop_blend_time": 0.5, "return_max_speed": 0.15}
    m.buttons = {"a": 0, "b": 1, "x": 2, "y": 3, "lb": 9, "back": 4, "start": 6}
    walk = SimpleNamespace(policy_step=0, last_action=np.zeros(12, np.float32),
                           term_dims=[12], num_history=1, motion=None, cmd=np.zeros(3, np.float32))
    m.policies = {"walk": walk}
    m.lo, m.hi = np.full(12, -1., np.float32), np.full(12, 1., np.float32)
    m.walk_limits = (m.lo, m.hi)
    m.pose_q = {"stand": np.full(12, 0.2, np.float32), "crouch": np.zeros(12, np.float32)}
    m.key_escape = False
    m.show_status = lambda: None
    m.policy_target = lambda policy: (m.pose_q["stand"].copy(), [])

    m.keys("4")
    assert m.state == "standing_transition"
    np.testing.assert_array_equal(m.blend_goal, m.pose_q["stand"])
    m.state_since -= m.stand_blend_time
    m.update_state()
    m.update_state()  # 当前观测还接近蹲姿，也不能覆盖人工回站姿的完成状态。
    assert m.state == "stand_ready"
    np.testing.assert_array_equal(n.target_pub, m.pose_q["stand"])
    m.keys("1")
    assert m.state == "walking"
    n.target_pub[:] = -0.4
    n.cmd[:] = 0.5
    n._clear_cmd_sources = lambda: n.cmd.fill(0)
    m.policy_target = lambda policy: (_ for _ in ()).throw(AssertionError("停止后不得运行策略"))
    m.keys("0")
    assert m.state == "stand_ready" and m.active is None
    np.testing.assert_array_equal(n.target_pub, m.pose_q["stand"])
    np.testing.assert_array_equal(n.cmd, 0)
    np.testing.assert_array_equal(published[-1], m.pose_q["stand"])
    m.tick()
    assert len(published) == 2
    m.keys("\n")
    assert m.state == "stand_ready" and len(published) == 2

    m.policy_target = lambda policy: (m.pose_q["stand"].copy(), [])
    m.keys("1")
    assert m.state == "walking"
    n.target_pub[:] = -0.4
    m.joy(SimpleNamespace(buttons=[0, 0, 0, 0, 0, 0, 1]))
    assert m.state == "stand_ready" and m.active is None
    np.testing.assert_array_equal(published[-1], m.pose_q["stand"])
