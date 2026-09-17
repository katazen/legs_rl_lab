"""离线切换测试：真实 ONNX 推理、合成反馈；不初始化 ROS 节点或连接电机。"""

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from rl_real_py.deployment_config import check_multi_pd, load_settings
from rl_real_py.multi_task import MultiTaskController
from rl_real_py import motion_reference
from rl_real_py.rl_real_common import Policy, RL_real
from test_io import make_node


CONFIG = Path(__file__).resolve().parents[1] / "configs/common.yaml"


def make_multi(monkeypatch, pose="stand", calibrated=False, current=False):
    n = make_node(monkeypatch)
    clock = [100.]
    monkeypatch.setattr("rl_real_py.rl_real_common.time.monotonic", lambda: clock[0])
    cfg, run, root = load_settings(CONFIG)
    if not current:
        # 旧三任务/事故回归固定旧模型与原 XML，不跟随日常部署选择漂移。
        cfg["tasks"]["crouch"] = str(root / "logs/rsl_rl/nlegs_mimic_crouch/2026-09-15_15-36-16")
        cfg["tasks"]["rise"] = str(root / "logs/rsl_rl/nlegs_mimic_stand/2026-09-15_18-50-29")
        resolve = motion_reference.resolve_recorded_path
        monkeypatch.setattr(motion_reference, "resolve_recorded_path", lambda path, repo:
                            Path(__file__).parent / "fixtures/nlegs_limit_20260915.xml"
                            if Path(path).name == "nlegs_limit.xml" else resolve(path, repo))
    if not calibrated:
        cfg.pop("crouch_calibration", None)  # 原训练场景回归；实测标定另测。
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
    p = n.multi.policies["rise"]
    n.obs_raw[:] = 0.
    n.obs_raw[3:7] = p.motion.quaternions[-1 if pose == "stand" else 0]
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


def test_complete_cycle_separate_histories_and_manual_stop(monkeypatch, capsys):
    def hint(label, *controls):
        output = capsys.readouterr().out
        prompt = output.rsplit("[操作提示]", 1)[-1].strip()
        assert prompt.startswith(label + " | ")
        assert len(prompt.splitlines()) == 1 and len(prompt) < 140
        assert "P/B 中断" in prompt and "不是断电急停" not in prompt
        assert all(control in prompt for control in controls)
        return output

    n, clock = make_multi(monkeypatch)
    m = n.multi
    assert "不是断电急停" in hint("等待关节/IMU", "保持当前姿态")
    assert not m.request("walk")
    assert "不是断电急停" not in hint("等待关节/IMU")
    assert "走路未启动" in n.warnings[-1]
    advance(n, clock, .8)
    output = hint("站立就绪", "1/LB+A", "2/LB+X")
    assert "等待使能反馈" in output and "姿态验收中" in output
    assert m.state == "stand_ready" and not m.request("rise")
    hint("站立就绪", "1/LB+A", "2/LB+X")
    initial = n.target_pub.copy()
    assert m.request("crouch")
    hint("下蹲中", "不能切换")
    assert not m.request("rise") and not m.request("crouch")
    hint("下蹲中")
    p = m.policies["crouch"]
    advance(n, clock, .1, follow=True)
    assert p.motion.steps == 0
    assert np.max(np.abs(n.target_pub-initial)) < m.cfg["handover_max_delta"]
    advance(n, clock, 4., follow=True)
    assert m.state == "crouch_ready" and m.active == "crouch"
    hint("下蹲就绪", "3/LB+Y")
    assert p.motion.frame == len(p.motion.positions)-1
    assert not m.request("walk")
    before = p.motion.steps
    assert m.request("rise")
    hint("起身中", "不能切换")
    assert m.policies["rise"].motion.steps == 0 and p.motion.steps == before
    advance(n, clock, 3.5, follow=True)
    assert m.state == "stand_ready" and m.active == "rise"
    hint("站立就绪", "1/LB+A", "2/LB+X")
    assert m.request("walk")
    hint("行走中", "0/Start", "W/S", "摇杆", "空格")
    walk = m.policies["walk"]
    assert walk.hist is not m.policies["rise"].hist
    assert all(np.all(buf == buf[0]) for buf in walk.hist.buffers)
    m.keys("w")
    assert "[键盘速度设置]" in hint("行走中")
    advance(n, clock, .5)
    assert "[操作提示]" not in capsys.readouterr().out, "控制循环不能反复刷操作提示"
    assert n.cmd[0] > 0
    assert m.request("stop")
    hint("停步待确认", "Enter/再次 Start", "双脚落地")
    advance(n, clock, .5)
    assert m.state == "stopping", "没有接触传感器，不能擅自宣告双脚承重并自动收脚"
    assert np.linalg.norm(n.cmd) < 1e-6
    assert not m.request("stop"), "长按 0 的重复字符不能被当成收脚确认"
    assert m.request("confirm_stop")
    hint("慢回站姿中", "不能切换")
    advance(n, clock, 1.)
    assert m.state == "stand_ready" and m.active is None
    hint("站立就绪", "1/LB+A", "2/LB+X")
    assert m.request("crouch")
    advance(n, clock, 4.1, follow=True)
    assert m.state == "crouch_ready"
    hint("下蹲就绪", "3/LB+Y")
    m.keys("p")
    hint("中断锁存", "R/Back", "排障后")
    assert m.request("reset")
    hint("姿态验收中", "4/LB+Start")
    advance(n, clock, .4)
    hint("下蹲就绪", "3/LB+Y")
    m.keys(" ")
    assert "前后=0.00m/s" in hint("下蹲就绪")


@pytest.mark.parametrize("kind", ["hardware", "task_limit", "pose", "held_target"])
def test_rejection_lists_all_bad_joints_even_before_ready(monkeypatch, kind):
    n, clock = make_multi(monkeypatch)
    advance(n, clock, .8)
    m = n.multi
    joints = [n.real_joint_names.index(name) for name in ("L2", "R4")]
    if kind == "hardware":
        n.obs_raw[7 + np.array(joints)] = n.hi[joints] + .01
    elif kind == "task_limit":
        n.obs_raw[7 + np.array(joints)] = m.hi[joints] + .031
    elif kind == "pose":
        n.obs_raw[7 + np.array(joints)] += .2
    else:
        n.target_pub[joints] = m.hi[joints] + .031
    held = n.target_pub.copy()
    # 未就绪时的状态门槛也必须给出关节原因；不得放行或改写目标。
    for state in (("checking", "stand_ready", "stopped") if kind != "held_target" else ("stand_ready",)):
        m.state = state
        for task in ("walk", "crouch", "return_stand", "reset"):
            if (task == "reset" and state != "stopped") or (kind == "pose" and task == "return_stand"):
                continue  # 手动回站姿不要求匹配姿态模板。
            assert not m.request(task)
            warning = n.warnings[-1]
            assert all(name + "=" in warning for name in ("L2", "R4")), warning
            assert "rad" in warning and "范围[" in warning
            assert m.state == state and m.active is None
            np.testing.assert_array_equal(n.target_pub, held)
    if kind == "hardware":
        assert not n._current_state_ready()
        assert "L2=" in n.warnings[-1] and "R4=" in n.warnings[-1]
    if kind == "task_limit":
        assert "容差 ±0.030" in n.warnings[-1]


def test_crouch_start_never_interpolates_to_standing(monkeypatch):
    n, clock = make_multi(monkeypatch, "crouch")
    q = n.obs_raw[7:19].copy()
    advance(n, clock, .8)
    assert n.multi.state == "crouch_ready"
    for target in n.sent:
        np.testing.assert_array_equal(target, q)
    assert not hasattr(n, "prepare_t0")
    assert n.multi.request("rise")


@pytest.mark.parametrize("start", ["crouch", "arbitrary", "stopped", "completed_policy"])
def test_manual_stand_return_is_slow_synchronized_and_explicit(monkeypatch, start):
    n, clock = make_multi(monkeypatch, "crouch", calibrated=True)
    m = n.multi
    cfg, _, _ = load_settings(CONFIG)
    n.obs_raw[3:7] = cfg["crouch_calibration"]["body_quat_wxyz"]
    if start == "arbitrary":
        n.obs_raw[7:19] = .5 * (m.pose_q["stand"] + m.pose_q["crouch"])
        n.obs_raw[3:7] = [np.cos(.15), 0., np.sin(.15), 0.]
    advance(n, clock, 1.1)
    if start == "arbitrary":
        assert m.state == "checking" and all(m.pose_error(p) for p in m.pose_q)
    if start == "stopped":
        m.halt("测试人工中断")
        advance(n, clock, .35)
    if start == "completed_policy":
        m.active = "crouch"
        p = m.policies["crouch"]
        p.motion.steps = int(np.ceil((len(p.motion.positions) - 1) / p.motion.stride))
    original = n.target_pub.copy()
    if start == "stopped":
        buttons = [0] * 8
        buttons[m.buttons["lb"]] = buttons[m.buttons["start"]] = 1
        m.joy(SimpleNamespace(buttons=buttons))
        since = m.state_since
        m.joy(SimpleNamespace(buttons=buttons))  # 手柄长按不能重启插值。
        assert m.state_since == since
    else:
        m.keys("4")
    assert m.state == "standing_transition" and m.active is None
    np.testing.assert_array_equal(n.target_pub, original)  # 按键本身不跳目标。
    duration = m.stand_blend_time
    assert duration >= 1.5 * np.max(np.abs(original - m.pose_q["stand"])) / m.cfg["return_max_speed"] - 1e-6
    for p in m.policies.values():
        p._infer = lambda x: pytest.fail("手动回站姿不得运行策略推理")
    assert not m.request("return_stand") and not m.request("walk")
    frames = [original]
    for _ in range(int(np.ceil((duration + .5) / n.pub_dt))):
        # 理想跟随只测试调度、速度和状态机，不证明实机能独立平衡。
        n.obs_raw[7:19] = n.target_pub
        n.obs_raw[3:7] = [1., 0., 0., 0.]
        advance(n, clock, n.pub_dt)
        frames.append(n.target_pub.copy())
        assert m.state != "stopped", m.reason
    frames = np.array(frames)
    assert np.abs(np.diff(frames, axis=0)).max() / n.pub_dt <= m.cfg["return_max_speed"] + 5e-5
    assert np.all(frames >= m.lo) and np.all(frames <= m.hi)
    np.testing.assert_allclose(n.target_pub, m.pose_q["stand"], atol=1e-6)
    moving = np.abs(m.pose_q["stand"] - original) > 1e-4
    progress = (frames[:, moving] - original[moving]) / (m.pose_q["stand"][moving] - original[moving])
    assert np.ptp(progress, axis=1).max() < 1e-5
    assert m.state == "stand_ready" and m.active is None and not n.cmd.any()


@pytest.mark.parametrize("state", ["wait_current", "wait_feedback", "walking", "stopping", "lowering", "rising", "standing_transition"])
def test_manual_stand_return_never_interrupts_or_queues(monkeypatch, state):
    n, clock = make_multi(monkeypatch)
    advance(n, clock, 1.1)
    m = n.multi
    m.state = state
    held = n.target_pub.copy()
    assert not m.request("return_stand")
    assert m.state == state
    np.testing.assert_array_equal(n.target_pub, held)


@pytest.mark.parametrize("bad", ["stale", "nan", "quat", "hardware", "task_limit", "tilt", "motor", "moving", "angular", "held_target"])
def test_manual_stand_return_rechecks_faults_and_stability(monkeypatch, bad):
    n, clock = make_multi(monkeypatch)
    advance(n, clock, 1.1)
    m = n.multi
    if bad == "stale": n._last_imu_rx = clock[0] - 1.
    if bad == "nan": n.obs_raw[7] = np.nan
    if bad == "quat": n.obs_raw[3:7] = 0.
    if bad == "hardware": n.obs_raw[7] = n.lo[0] - .001
    if bad == "task_limit": n.obs_raw[8] = m.hi[1] + .031
    if bad == "tilt": n.obs_raw[3:7] = [np.cos(.5), 0., np.sin(.5), 0.]
    if bad == "motor": n._motor_faults["left:1"] = "失能"
    if bad == "moving": n.obs_raw[19] = .4
    if bad == "angular": n.obs_raw[0] = .5
    if bad == "held_target": n.target_pub[1] = m.hi[1] + .031
    held = n.target_pub.copy()
    assert not m.request("return_stand") and m.return_still_since is None
    np.testing.assert_array_equal(n.target_pub, held)
    set_pose(n, "stand")
    n._motor_faults.clear()
    n.target_pub = n.default_real.copy()
    advance(n, clock, .1)
    assert not m.request("return_stand"), "须重新累计稳定时间，不能沿用故障前计时"
    advance(n, clock, .25)
    assert m.state == "stand_ready", "之前拒绝的请求不得自动执行"
    assert m.request("return_stand")


@pytest.mark.parametrize("event", ["halt", "timeout", "stale", "scheduling", "motor", "task_limit", "tilt"])
def test_manual_stand_return_can_stop_and_never_declares_false_success(monkeypatch, event):
    n, clock = make_multi(monkeypatch, "crouch")
    advance(n, clock, 1.1)
    m = n.multi
    assert m.request("return_stand")
    advance(n, clock, .1)
    held = n.target_pub.copy()
    if event == "halt": m.keys("p")
    if event == "stale": n._last_joint_rx = clock[0] - 1.
    if event == "scheduling": clock[0] += .1
    if event == "motor": n._motor_faults["left:1"] = "失能"
    if event == "task_limit": n.obs_raw[8] = m.hi[1] + .031
    if event == "tilt": n.obs_raw[3:7] = [np.cos(.5), 0., np.sin(.5), 0.]
    if event == "timeout":
        advance(n, clock, m.stand_blend_time + m.cfg["end_timeout"] + .5)
    else:
        if event == "scheduling": n._last_joint_rx = n._last_imu_rx = clock[0]
        n._tick()
        np.testing.assert_array_equal(n.target_pub, held)
    assert m.state == "stopped" and m.active is None


def test_measured_crouch_handover_and_shared_bounds(monkeypatch):
    cfg, _, _ = load_settings(CONFIG)
    c = cfg["crouch_calibration"]
    n, clock = make_multi(monkeypatch, "crouch", calibrated=True)
    m, p = n.multi, n.multi.policies["rise"]
    n.obs_raw[3:7] = c["body_quat_wxyz"]
    n.obs_raw[19:31] = np.array([-1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, -1]) * (45 / 4095)
    measured = n.obs_raw.copy()
    reference = p.motion.positions.copy()
    clip = p.action_term_clip.copy()
    assert m.pose_error("crouch") is None
    advance(n, clock, .8)
    assert m.state == "crouch_ready"
    assert all(np.array_equal(target, measured[7:19]) for target in n.sent)
    assert m.request("rise")
    np.testing.assert_array_equal(m.handover_from, measured[7:19])
    np.testing.assert_array_equal(p.obs_raw, measured)  # 不伪造观测来适配旧网络。
    assert np.max(np.abs(m.pending_target - measured[7:19])) < m.cfg["handover_max_delta"]
    for name, side in c["stop_sides"].items():
        i = n.real_joint_names.index(name)
        assert (m.lo if side == "lower" else m.hi)[i] == measured[7 + i]
    for name in ("L6", "R6"):
        i = n.real_joint_names.index(name)
        assert m.lo[i] == max(n.lo[i], p.motion.limits[p.sim2real[i], 0])
        assert m.hi[i] == min(n.hi[i], p.motion.limits[p.sim2real[i], 1])
    for _ in range(700):
        # 合成反馈只验证切换和裁剪，不是动力学仿真或实机起身成功证明。
        n.obs_raw[7:19] = np.clip(p.motion.positions[p.motion.frame, p.sim2real], m.lo, m.hi)
        n.obs_raw[3:7] = p.motion.quaternions[p.motion.frame]
        n.obs_raw[19:31] = 0.
        advance(n, clock, n.pub_dt)
        assert m.state != "stopped", m.reason
        assert np.all(n.sent[-1] >= m.lo) and np.all(n.sent[-1] <= m.hi)
    assert m.state == "stand_ready" and m.request("walk")
    np.testing.assert_array_equal(p.motion.positions, reference)
    np.testing.assert_array_equal(p.action_term_clip, clip)
    np.testing.assert_array_equal(n.lo, np.asarray(cfg["joint_lower_limits"], np.float32))
    np.testing.assert_array_equal(n.hi, np.asarray(cfg["joint_upper_limits"], np.float32))
    n.obs_raw[7 + n.real_joint_names.index("R4")] = m.hi[n.real_joint_names.index("R4")] + .031
    advance(n, clock, n.pub_dt)
    assert m.state == "stopped" and "反馈超出任务限位" in m.reason


@pytest.mark.parametrize("bad", ["missing", "nan", "hardware", "ankle_stop", "side", "reversed", "quat", "tilt"])
def test_invalid_crouch_calibration_rejected(monkeypatch, bad):
    n, _ = make_multi(monkeypatch)
    cfg, _, root = load_settings(CONFIG)
    c = cfg["crouch_calibration"]
    if bad == "missing": c["joint_pos"].pop("L1")
    if bad == "nan": c["joint_pos"]["L1"] = float("nan")
    if bad == "hardware": c["joint_pos"]["L1"] = -1.051
    if bad == "ankle_stop": c["stop_sides"]["L6"] = "lower"
    if bad == "side": c["stop_sides"]["L1"] = "wrong"
    if bad == "reversed": c["stop_sides"]["L1"] = "upper"
    if bad == "quat": c["body_quat_wxyz"] = [0., 0., 0., 0.]
    if bad == "tilt": c["body_quat_wxyz"] = [np.cos(.5), 0., np.sin(.5), 0.]
    with pytest.raises(ValueError):
        MultiTaskController(n, cfg, root)
    assert not n.sent


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
    monkeypatch.setattr("sys.stdin", SimpleNamespace(fileno=lambda: 0))
    monkeypatch.setattr("rl_real_py.rl_real_common.os.read", lambda fd, size: b"2p1r")
    RL_real._read_keys(n)
    assert m.state == "stopped" and m.active is None
    m.keys("w")
    assert not n._kb_cmd.any()


def test_unfinished_motion_times_out_without_unsafe_rehandover(monkeypatch):
    n, clock = make_multi(monkeypatch)
    advance(n, clock, .8)
    m = n.multi
    assert m.request("crouch")
    advance(n, clock, 8.)  # 反馈一直站立：不能只按参考时间宣布下蹲完成。
    assert m.state == "stopped"
    assert m.request("reset")
    advance(n, clock, .4)
    assert m.state == "stand_ready"
    assert not m.request("walk"), "反馈站立但仍保持下蹲目标，不能绕过首拍跳变量保护"
    assert "首拍目标跳变过大" in n.warnings[-1]


def test_unconfirmed_stop_times_out(monkeypatch):
    n, clock = make_multi(monkeypatch)
    advance(n, clock, .8)
    m = n.multi
    assert m.request("walk")
    advance(n, clock, .3)
    assert m.request("stop")
    advance(n, clock, 6.1)
    assert m.state == "stopped"


def test_pd_mismatch_rejected_before_synchronization(tmp_path):
    cfg, _, _ = load_settings(CONFIG)
    cfg = copy.deepcopy(cfg)
    for name, path in cfg["tasks"].items():
        if path is None:
            continue
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
    if not (Path(directory) / f"{task}.npz").is_file():
        pytest.skip(f"没有 {task}.npz 回放数据")
    n, _ = make_multi(monkeypatch, current=(task == "rise"))
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


def test_current_rise_only_measured_start_and_disabled_crouch(monkeypatch, capsys):
    n, clock = make_multi(monkeypatch, "crouch", calibrated=True, current=True)
    cfg, _, _ = load_settings(CONFIG)
    m, p = n.multi, n.multi.policies["rise"]
    assert set(m.policies) == {"walk", "rise"} and cfg["tasks"]["crouch"] is None
    assert p.motion.path.name == "crouch_to_stand_v2.npz" and len(p.motion.positions) == 336
    n.obs_raw[3:7] = cfg["crouch_calibration"]["body_quat_wxyz"]
    measured = n.obs_raw.copy()
    for name, side in cfg["crouch_calibration"]["stop_sides"].items():
        i = n.real_joint_names.index(name)
        assert np.isclose(p.motion.positions[0, p.sim2real[i]], measured[7+i], atol=1e-6)
        assert (m.lo if side == "lower" else m.hi)[i] == measured[7+i]
    advance(n, clock, .8)
    assert m.state == "crouch_ready"
    assert all(np.array_equal(target, measured[7:19]) for target in n.sent)
    assert m.request("rise")
    for _ in range(850):
        advance(n, clock, n.pub_dt, follow=True)
        assert m.state != "stopped", m.reason
        assert np.all(n.sent[-1] >= m.lo) and np.all(n.sent[-1] <= m.hi)
    assert p.motion.frame == 335 and m.state == "stand_ready"
    held, active = n.target_pub.copy(), m.active
    m.keys("2")
    buttons = [0]*8
    buttons[m.buttons["lb"]] = buttons[m.buttons["x"]] = 1
    m.joy(SimpleNamespace(buttons=buttons))
    assert m.state == "stand_ready" and m.active == active
    np.testing.assert_array_equal(n.target_pub, held)
    assert "下蹲任务已禁用" in n.warnings[-1]
    output = capsys.readouterr().out
    assert "下蹲已禁用" in output
    assert "2/LB+X" not in output.rsplit("[操作提示]", 1)[-1]
    assert m.request("walk")
    n.obs_raw[7] = m.lo[0] - .031
    n.obs_raw[7+3] = m.hi[3] + .031
    advance(n, clock, n.pub_dt)
    assert m.state == "stopped" and "任务限位" in m.reason
    assert "L1=" in m.reason and "L4=" in m.reason
    held = n.target_pub.copy()
    m.keys("3")
    assert "起身未启动" in n.warnings[-1]
    assert "L1=" in n.warnings[-1] and "L4=" in n.warnings[-1]
    assert m.state == "stopped" and m.active is None
    np.testing.assert_array_equal(n.target_pub, held)


@pytest.mark.parametrize("task,value", [("walk", None), ("rise", None), ("crouch", ""), ("crouch", False)])
def test_only_crouch_can_be_explicitly_disabled(tmp_path, task, value):
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["tasks"][task] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="仅 crouch 可设为 null"):
        load_settings(path)


def test_reenabling_old_crouch_is_not_silently_accepted(monkeypatch):
    with monkeypatch.context() as legacy:
        old, _ = make_multi(legacy)
        down = old.multi.policies["crouch"]
    n, _ = make_multi(monkeypatch, current=True)
    cfg, _, root = load_settings(CONFIG)
    cfg["tasks"]["crouch"] = str(down.run_dir)
    up = n.multi.policies["rise"]
    monkeypatch.setattr("rl_real_py.multi_task.Policy", lambda c, path, r:
                        down if path == down.run_dir else up)
    with pytest.raises(ValueError, match="端点不衔接"):
        MultiTaskController(n, cfg, root)
    assert not n.sent
