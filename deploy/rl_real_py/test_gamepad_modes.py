"""离线验证 SDL 手柄输入模式；不创建 ROS 节点或电机通信。"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rl_real_py.deployment_config import load_settings
from rl_real_py.multi_task import MultiTaskController
from rl_real_py.rl_real_common import Policy, RL_real


BUTTONS = {"a": 0, "b": 1, "x": 2, "y": 3, "lb": 9, "back": 4, "start": 6}


def message(axes=None, pressed=()):
    buttons = [0] * 21
    for index in pressed:
        buttons[index] = 1
    return SimpleNamespace(axes=[0.] * 6 if axes is None else axes, buttons=buttons)


@pytest.fixture
def controls(monkeypatch):
    clock, warnings, requests = [10.], [], []
    monkeypatch.setattr("rl_real_py.rl_real_common.time.monotonic", lambda: clock[0])
    n = SimpleNamespace(
        cmd=np.zeros(3, np.float32), _kb_cmd=np.zeros(3, np.float32),
        _joy_cmd=np.zeros(3, np.float32), _last_joy_rx=-1e9, _prev_buttons=[],
        cmd_min=np.array([-.2, -.1, -.4], np.float32),
        cmd_max=np.array([.3, .2, .5], np.float32),
        ctrl_timeout=.4, deadzone=.12, kb_step=.05, tick=1, decimation=2,
        get_logger=lambda: SimpleNamespace(warn=lambda msg, **kw: warnings.append(msg)),
        _publish_target=lambda: None,
    )
    n._clear_cmd_sources = lambda: RL_real._clear_cmd_sources(n)
    m = object.__new__(MultiTaskController)
    m.n, m.state, m.active, m.has_target = n, "walking", "walk", True
    m.buttons = MultiTaskController.gamepad_buttons(BUTTONS)
    m.policies = {"walk": SimpleNamespace(cmd=np.zeros(3, np.float32))}
    m.key_escape = False
    m.request = requests.append
    m.show_status = m.update_state = lambda: None
    n.multi = m
    return n, m, clock, warnings, requests


def test_default_stick_axes_deadzone_and_asymmetric_ranges(controls):
    n, m, *_ = controls
    assert m.joy_move_mode == "stick" and not m.joy_wait_neutral
    RL_real._on_joy(n, message([-.5, 1., -.5, 1., 1., -1.], pressed=[12, 13]))
    np.testing.assert_allclose(n._joy_cmd, [.3, -.05, -.2])
    RL_real._on_joy(n, message([.1, -.1, .1, 1., 1., -1.], pressed=[11, 14]))
    np.testing.assert_array_equal(n._joy_cmd, 0.)


@pytest.mark.parametrize("pressed, expected", [
    ([11], [.3, 0., .25]), ([12], [-.2, 0., .25]),
    ([13], [0., .2, .25]), ([14], [0., -.1, .25]),
    ([11, 13], [.3, .2, .25]), ([11, 12, 13, 14], [0., 0., .25]),
    ([], [0., 0., .25]),
])
def test_dpad_hold_release_opposites_and_right_stick_yaw(controls, pressed, expected):
    n, m, *_ = controls
    m.joy_move_mode = "dpad"
    # 左摇杆的偏转不能混入方向键模式；右 X 仍控制转向。
    msg = message([-1., -1., .5, 0., -1., 1.], pressed=pressed)
    for _ in range(3):
        RL_real._on_joy(n, msg)
        np.testing.assert_allclose(n._joy_cmd, expected)
    RL_real._on_joy(n, message())
    np.testing.assert_array_equal(n._joy_cmd, 0.)


def test_rb_edge_clears_only_joy_and_requires_all_controls_neutral(controls):
    n, m, *_ = controls
    n._kb_cmd[:] = [.05, -.05, .1]
    held = message([0., .8, 0., 0., 0., 0.], pressed=[10, 11])
    RL_real._on_joy(n, held)
    assert m.joy_move_mode == "dpad" and m.joy_wait_neutral
    np.testing.assert_array_equal(n._joy_cmd, 0.)
    np.testing.assert_allclose(n._kb_cmd, [.05, -.05, .1])
    for msg in (held, message(pressed=[10, 11, 12]),
                message([0., 0., -.2, 0., 0., 0.], pressed=[10]),
                message([.2, 0., 0., 0., 0., 0.], pressed=[10])):
        RL_real._on_joy(n, msg)
        assert m.joy_move_mode == "dpad" and m.joy_wait_neutral
        np.testing.assert_array_equal(n._joy_cmd, 0.)
    # RB 可保持按下；右 Y、扳机不用于移动，不阻止解除回中等待。
    RL_real._on_joy(n, message([0., 0., 0., 1., 1., -1.], pressed=[10]))
    assert not m.joy_wait_neutral and m.joy_move_mode == "dpad"
    np.testing.assert_array_equal(n._joy_cmd, 0.)
    RL_real._on_joy(n, message(pressed=[10, 11]))
    np.testing.assert_allclose(n._joy_cmd, [.3, 0., 0.])
    RL_real._on_joy(n, message())  # 松开 RB 后才能再次切换。
    RL_real._on_joy(n, message(pressed=[10]))
    assert m.joy_move_mode == "stick" and m.joy_wait_neutral
    np.testing.assert_array_equal(n._joy_cmd, 0.)


@pytest.mark.parametrize("pressed, expected", [
    ([1, 4, 6, 9, 0], "halt"), ([4, 6, 9, 0], "reset"),
    ([6], "stop"), ([9, 6], "return_stand"),
    ([9, 0], "walk"), ([9, 2], "crouch"), ([9, 3], "rise"),
    ([9], None),
])
def test_existing_buttons_take_priority_over_rb(controls, pressed, expected):
    n, m, _, _, requests = controls
    msg = message(pressed=[10, *pressed])
    RL_real._on_joy(n, msg)
    assert requests == ([] if expected is None else [expected])
    assert m.joy_move_mode == "stick" and not m.joy_wait_neutral
    RL_real._on_joy(n, msg)
    assert requests == ([] if expected is None else [expected])
    assert m.joy_move_mode == "stick"


def test_keyboard_independent_timeout_and_space(controls):
    n, m, clock, *_ = controls
    m.keys("wq")
    RL_real._on_joy(n, message([0., 1., 1., 0., 0., 0.]))
    m.tick()
    np.testing.assert_allclose(n.cmd, [.3, 0., .5])
    clock[0] += .41
    m.tick()
    np.testing.assert_allclose(n.cmd, [.05, 0., .05])
    np.testing.assert_array_equal(m.policies["walk"].cmd, n.cmd)
    RL_real._on_joy(n, message(pressed=[10]))
    m.tick()
    np.testing.assert_allclose(n.cmd, [.05, 0., .05])
    assert m.joy_wait_neutral
    m.keys(" ")
    m.tick()
    np.testing.assert_array_equal(n.cmd, 0.)
    np.testing.assert_array_equal(n._kb_cmd, 0.)
    np.testing.assert_array_equal(n._joy_cmd, 0.)
    assert m.joy_move_mode == "dpad"


@pytest.mark.parametrize("axes, count", [([0.] * 5, 21), ([np.nan] * 6, 21), ([0.] * 6, 14)])
def test_bad_message_cannot_toggle_or_refresh_timeout(controls, axes, count):
    n, m, _, warnings, requests = controls
    msg = message(axes, pressed=[10])
    msg.buttons = msg.buttons[:count]
    RL_real._on_joy(n, msg)
    assert m.joy_move_mode == "stick" and not m.joy_wait_neutral
    assert n._last_joy_rx == -1e9 and n._prev_buttons == []
    assert warnings and not requests


def test_optional_rb_default_and_custom_index(controls):
    n, m, *_ = controls
    assert MultiTaskController.gamepad_buttons(BUTTONS)["rb"] == 10
    assert "rb" not in BUTTONS
    m.buttons = MultiTaskController.gamepad_buttons({**BUTTONS, "rb": 5})
    RL_real._on_joy(n, message(pressed=[10]))
    assert m.joy_move_mode == "stick"
    RL_real._on_joy(n, message(pressed=[5]))
    assert m.joy_move_mode == "dpad"


@pytest.mark.parametrize("index", [0, 11, 12, 13, 14, -1, 21, True, 2.5, "10"])
def test_reject_conflicting_or_non_sdl_rb_indices(index):
    with pytest.raises(ValueError, match="gamepad_buttons"):
        MultiTaskController.gamepad_buttons({**BUTTONS, "rb": index})


def test_reject_missing_or_duplicate_existing_button():
    missing = {key: value for key, value in BUTTONS.items() if key != "b"}
    for buttons in (missing, {**BUTTONS, "lb": 6}):
        with pytest.raises(ValueError, match="gamepad_buttons"):
            MultiTaskController.gamepad_buttons(buttons)


def test_common_uses_selected_policy_training_velocity_ranges(monkeypatch):
    cfg, run_dir, repo = load_settings(Path(__file__).parent / "configs/common.yaml")
    assert "velocity_command_limits" not in cfg
    monkeypatch.setattr(Policy, "_load_policy", lambda *_: None)
    monkeypatch.setattr(Policy, "_check_policy", lambda *_: None)
    policy = Policy(cfg, run_dir, repo)
    ranges = policy.deploy["commands"]["base_velocity"]["ranges"]
    expected = np.asarray([ranges[name] for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z")], np.float32)
    np.testing.assert_array_equal(policy.cmd_min, expected[:, 0])
    np.testing.assert_array_equal(policy.cmd_max, expected[:, 1])
