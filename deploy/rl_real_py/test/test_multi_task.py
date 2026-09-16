"""离线切换测试：真实 ONNX 推理、合成反馈；不初始化 ROS 节点或连接电机。"""

import copy
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from rl_real_py.deployment_config import check_multi_pd, load_settings
from rl_real_py.multi_task import MultiTaskController
from rl_real_py.rl_real_common import Policy, RL_real
from test_io import make_node


CONFIG = Path(__file__).resolve().parents[1] / "configs/common.yaml"


def make_multi(monkeypatch, pose="stand"):
    n = make_node(monkeypatch)
    clock = [100.]
    monkeypatch.setattr("rl_real_py.rl_real_common.time.monotonic", lambda: clock[0])
    cfg, run, root = load_settings(CONFIG)
    Policy.__init__(n, cfg, run, root)
    n.target_pub = n.target_real = n.default_real.copy()
    n.cmd_accel_limit = np.asarray(cfg["cmd_accel_limit"], np.float32)
    n.ctrl_timeout, n.kb_step = cfg["ctrl_timeout"], cfg["keyboard"]["step"]
    n.state_timeout = cfg["state_timeout"]
    n._motor_faults = {}
    n._open_log = lambda *args: n.log_events.append("open")
    n._log_row = lambda *args: None
    n.multi = MultiTaskController(n, cfg, root)
    n.get_logger = lambda: SimpleNamespace(warn=lambda s, **kw: n.warnings.append(s),
                                          error=lambda s, **kw: n.warnings.append(s))
    set_pose(n, pose)
    return n, clock


def set_pose(n, pose):
    p = n.multi.policies["crouch" if pose == "stand" else "rise"]
    n.obs_raw[:] = 0.
    n.obs_raw[3:7] = p.motion.quaternions[0]
    n.obs_raw[7:19] = n.multi.pose_q[pose]


def advance(n, clock, seconds, follow=False, fresh=True):
    for _ in range(round(seconds / n.pub_dt)):
        clock[0] += n.pub_dt
        if fresh:
            n._last_joint_rx = n._last_imu_rx = clock[0]
        if follow and n.multi.active in ("crouch", "rise"):
            p = n.multi.policies[n.multi.active]
            n.obs_raw[7:19] = p.motion.positions[p.motion.frame, p.sim2real]
            n.obs_raw[3:7] = p.motion.quaternions[p.motion.frame]
        n._tick()
        if n.sent:
            assert np.isfinite(n.sent[-1]).all()
            assert np.all(n.sent[-1] >= n.lo) and np.all(n.sent[-1] <= n.hi)


def test_complete_cycle_separate_histories_and_manual_stop(monkeypatch):
    n, clock = make_multi(monkeypatch)
    m = n.multi
    assert not m.request("walk")
    advance(n, clock, .8)
    assert m.state == "stand_ready" and not m.request("rise")
    initial = n.target_pub.copy()
    assert m.request("crouch")
    assert not m.request("rise") and not m.request("crouch")
    p = m.policies["crouch"]
    advance(n, clock, .1, follow=True)
    assert p.motion.steps == 0
    assert np.max(np.abs(n.target_pub-initial)) < m.cfg["handover_max_delta"]
    advance(n, clock, 4., follow=True)
    assert m.state == "crouch_ready" and m.active == "crouch"
    assert p.motion.frame == len(p.motion.positions)-1
    assert not m.request("walk")
    before = p.motion.steps
    assert m.request("rise")
    assert m.policies["rise"].motion.steps == 0 and p.motion.steps == before
    advance(n, clock, 3.5, follow=True)
    assert m.state == "stand_ready" and m.active == "rise"
    assert m.request("walk")
    walk = m.policies["walk"]
    assert walk.hist is not m.policies["rise"].hist
    assert all(np.all(buf == buf[0]) for buf in walk.hist.buffers)
    m.keys("w")
    advance(n, clock, .5)
    assert n.cmd[0] > 0
    assert m.request("stop")
    advance(n, clock, .5)
    assert m.state == "stopping", "没有接触传感器，不能擅自宣告双脚承重并自动收脚"
    assert np.linalg.norm(n.cmd) < 1e-6
    assert not m.request("stop"), "长按 0 的重复字符不能被当成收脚确认"
    assert m.request("confirm_stop")
    advance(n, clock, 1.)
    assert m.state == "stand_ready" and m.active is None
    assert m.request("crouch")
    advance(n, clock, 4.1, follow=True)
    assert m.state == "crouch_ready"


def test_crouch_start_never_interpolates_to_standing(monkeypatch):
    n, clock = make_multi(monkeypatch, "crouch")
    q = n.obs_raw[7:19].copy()
    advance(n, clock, .8)
    assert n.multi.state == "crouch_ready"
    for target in n.sent:
        np.testing.assert_array_equal(target, q)
    assert not hasattr(n, "prepare_t0")
    assert n.multi.request("rise")


@pytest.mark.parametrize("bad", ["stale", "nan", "joint", "tilt", "motor", "action", "unsafe_target", "late", "slow"])
def test_fault_latches_keeps_last_target_and_never_auto_resumes(monkeypatch, bad):
    n, clock = make_multi(monkeypatch)
    advance(n, clock, .8)
    m = n.multi
    assert m.request("crouch")
    advance(n, clock, .3, follow=True)
    p = m.policies["crouch"]
    held, frame = n.target_pub.copy(), p.motion.steps
    if bad == "stale": n._last_imu_rx = clock[0] - 1.
    if bad == "nan": n.obs_raw[7] = np.nan
    if bad == "joint": n.obs_raw[8] = m.hi[1] + .04
    if bad == "tilt": n.obs_raw[3:7] = [np.cos(.5), 0., np.sin(.5), 0.]
    if bad == "motor":
        n._on_motor_warn(SimpleNamespace(data=json.dumps({"arm": "left", "errors": [{"id": 1, "err": "失能"}]})))
    if bad == "action": p._infer = lambda x: np.full(12, np.nan)
    if bad == "unsafe_target":
        action = np.zeros(12, np.float32)
        action[p.motion.joint_names.index("joint_L3")] = -5.
        p._infer = lambda x: action
    if bad == "late": m.last_policy = clock[0] - .1
    if bad == "slow":
        def slow(x):
            clock[0] += .1
            return np.zeros(12)
        p._infer = slow
    n.tick = 3
    n._tick()
    assert m.state == "stopped" and p.motion.steps == frame
    np.testing.assert_array_equal(n.sent[-1], held)
    set_pose(n, "stand")
    n._motor_faults.clear()
    advance(n, clock, .5)
    assert m.state == "stopped" and not m.request("walk")
    assert m.request("reset")
    advance(n, clock, .4)
    assert m.state == "stand_ready" and m.active is None


def test_first_command_enable_gap_holds_without_advancing(monkeypatch):
    n, clock = make_multi(monkeypatch)
    n._tick()
    assert n.multi.state == "wait_feedback"
    held = n.target_pub.copy()
    advance(n, clock, 1.2, fresh=False)
    assert n.multi.state == "wait_feedback"
    assert not n.multi.request("walk")
    np.testing.assert_array_equal(n.sent[-1], held)
    advance(n, clock, .8)
    assert n.multi.state == "stand_ready"


def test_operator_stop_priority_and_gamepad_edges(monkeypatch):
    n, clock = make_multi(monkeypatch)
    advance(n, clock, .8)
    m = n.multi
    buttons = [0]*8
    buttons[m.buttons["lb"]] = buttons[m.buttons["a"]] = 1
    n._on_joy(SimpleNamespace(axes=[0.]*3, buttons=buttons))
    assert m.state == "walking"
    advance(n, clock, .25)
    step = n.policy_step
    n._on_joy(SimpleNamespace(axes=[0.]*3, buttons=buttons))
    assert n.policy_step == step
    monkeypatch.setattr("sys.stdin", io.StringIO("2p1r"))
    RL_real._read_keys(n)
    assert m.state == "stopped" and m.active is None
    m.keys("w")
    assert not n._kb_cmd.any()


def test_unfinished_motion_and_unconfirmed_stop_time_out(monkeypatch):
    n, clock = make_multi(monkeypatch)
    advance(n, clock, .8)
    m = n.multi
    assert m.request("crouch")
    advance(n, clock, 8.)  # 反馈一直站立：不能只按参考时间宣布下蹲完成。
    assert m.state == "stopped"
    assert m.request("reset")
    advance(n, clock, .4)
    assert m.request("walk")
    advance(n, clock, .3)
    assert m.request("stop")
    advance(n, clock, 6.1)
    assert m.state == "stopped"


def test_pd_mismatch_rejected_before_synchronization(tmp_path):
    cfg, _, _ = load_settings(CONFIG)
    cfg = copy.deepcopy(cfg)
    for name, path in cfg["tasks"].items():
        target = tmp_path / name / "params"
        target.mkdir(parents=True)
        dep = yaml.safe_load((Path(path) / "params/deploy.yaml").read_text())
        if name == "rise":
            dep["stiffness"][0] += 1.
        (target / "deploy.yaml").write_text(yaml.safe_dump(dep))
        cfg["tasks"][name] = str(target.parent)
    with pytest.raises(ValueError, match="PD"):
        check_multi_pd(cfg)


@pytest.mark.parametrize("task", ["crouch", "rise"])
def test_onnx_reference_inputs_and_targets_match_sim2sim(monkeypatch, task):
    directory = os.environ.get("MULTI_PARITY_DIR")
    if not directory:
        pytest.skip("设置 MULTI_PARITY_DIR，包含 check_crouch_mimic_sim2sim 导出的 crouch.npz / rise.npz")
    n, _ = make_multi(monkeypatch)
    p = n.multi.policies[task]
    with np.load(Path(directory) / f"{task}.npz") as data:
        sdk = data["joint_names"].tolist()
        order = [sdk.index("joint_" + name) for name in n.real_joint_names]
        for i, step in enumerate(data["steps"]):
            p.motion.steps = int(step)
            p.obs_raw[:3] = data["omega"][i]
            p.obs_raw[3:7] = data["quat"][i]
            p.obs_raw[7:19] = data["q"][i, order]
            p.obs_raw[19:31] = data["qd"][i, order]
            p.last_action[:] = data["last_action"][i]
            obs = np.concatenate(p._build_terms())
            np.testing.assert_allclose(obs, data["observations"][i], rtol=1e-5, atol=1e-6)
            action = p._infer(obs)
            np.testing.assert_allclose(action, data["actions"][i], rtol=1e-4, atol=1e-5)
            target = p._target_from_action(action)
            np.testing.assert_allclose(target, data["targets"][i, order], rtol=1e-5, atol=5e-6)
        n.obs_raw[:] = p.obs_raw
        assert n.multi.pose_error("crouch" if task == "crouch" else "stand") is None


@pytest.mark.parametrize("bad", ["nan", "jump", "shape"])
def test_failed_handover_keeps_previous_policy_and_target(monkeypatch, bad):
    n, clock = make_multi(monkeypatch)
    advance(n, clock, .8)
    m = n.multi
    held = n.target_pub.copy()
    p = m.policies["crouch"]
    p._infer = lambda x: {"nan": np.full(12, np.nan), "jump": np.full(12, 5.), "shape": np.zeros(11)}[bad]
    assert not m.request("crouch")
    assert m.state == "stand_ready" and m.active is None
    np.testing.assert_array_equal(n.target_pub, held)


def test_zero_speed_gate_uses_exported_threshold_and_command_bias(monkeypatch):
    n, _ = make_multi(monkeypatch)
    n.gait_gate_by_cmd = True
    n.gait_command_threshold = 1e-6
    n.obs_raw[3] = 1.
    n.policy_step = 7
    n.cmd[:] = 0.
    assert not n._build_terms()[-1].any()
    n.cmd[2] = .02
    assert np.isclose(np.linalg.norm(n._build_terms()[-1]), 1.)
    n.cmd[:] = 0.
    n.cmd_bias[0] = .02
    assert np.isclose(np.linalg.norm(n._build_terms()[-1]), 1.)


def test_keyboard_escape_sequences_never_become_movement_keys(monkeypatch):
    n, clock = make_multi(monkeypatch)
    advance(n, clock, .8)
    assert n.multi.request("walk")
    n.multi.keys("\x1b[")
    n.multi.keys("A")
    assert not n._kb_cmd.any()
    n.multi.keys("\x1bOP")
    assert n.multi.state == "stopped"  # P 中断优先于任何字符序列。


def test_contact_tolerance_allows_rise_without_relaxing_published_limits(monkeypatch):
    n, clock = make_multi(monkeypatch, "crouch")
    joint = n.real_joint_names.index("L2")
    n.obs_raw[7 + joint] = n.multi.hi[joint] + .007
    advance(n, clock, .8)
    assert n.multi.state == "crouch_ready"
    assert n.multi.request("rise")
    advance(n, clock, .02)
    assert np.all(n.sent[-1] >= n.multi.lo) and np.all(n.sent[-1] <= n.multi.hi)
    n.obs_raw[7 + joint] = n.multi.hi[joint] + .031
    n._tick()
    assert n.multi.state == "stopped"


def test_task_feedback_tolerance_never_expands_hardware_bounds(monkeypatch):
    n, _ = make_multi(monkeypatch)
    for joint in range(12):
        for bound, direction in ((n.lo[joint], -1), (n.hi[joint], 1)):
            set_pose(n, "stand")
            n.obs_raw[7 + joint] = bound + direction * .005
            for task_bounds in (False, True):
                assert "common 硬件限位" in n.multi.state_error(task_bounds=task_bounds)


@pytest.mark.parametrize("bad", ["stale", "nan", "quat", "tilt", "moving", "angular", "not_standing"])
def test_task_key_rechecks_measured_pose(monkeypatch, bad):
    n, clock = make_multi(monkeypatch)
    advance(n, clock, .8)
    if bad == "stale": n._last_joint_rx = clock[0] - 1.
    if bad == "nan": n.obs_raw[3] = np.nan
    if bad == "quat": n.obs_raw[3:7] = 0.
    if bad == "tilt": n.obs_raw[3:7] = [np.cos(.2), 0., np.sin(.2), 0.]
    if bad == "moving": n.obs_raw[19] = .4
    if bad == "angular": n.obs_raw[0] = .5
    if bad == "not_standing": set_pose(n, "crouch")
    held = n.target_pub.copy()
    assert not n.multi.request("crouch") and n.multi.active is None
    np.testing.assert_array_equal(n.target_pub, held)
