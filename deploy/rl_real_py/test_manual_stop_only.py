"""离线验证：运行中异常只告警，手柄 B 才进入中断锁存。"""

from types import SimpleNamespace

import numpy as np

from rl_real_py.multi_task import MultiTaskController
from rl_real_py.rl_real_common import RL_real


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
    RL_real._on_motor_warn(n, SimpleNamespace(data='{"errors":[{"err":"失能"}]}'))
    m.tick()
    assert m.state == "walking" and m.active == "walk"
    assert any("推理失败" in message for message in warnings)
    m.keys("p")
    assert m.state == "walking"

    RL_real._on_joy(n, SimpleNamespace(axes=[0.] * 6, buttons=[0, 1] + [0] * 13))
    assert m.state == "stopped" and m.active is None
    np.testing.assert_array_equal(n.target_real, n.target_pub)
