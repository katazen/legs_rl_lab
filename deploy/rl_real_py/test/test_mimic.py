"""离线验证参考动作、训练裁剪和历史越限回归；不连接实机。"""

import copy
import csv

import numpy as np
import pytest

from rl_real_py.deployment_config import load_settings
from rl_real_py.motion_reference import MotionReference, rotation_matrix
from test_multi_task import CONFIG, make_multi


def mimic_policy(monkeypatch):
    n, _ = make_multi(monkeypatch)
    p = n.multi.policies["crouch"]
    p.obs_raw[:] = n.obs_raw
    cfg, _, root = load_settings(CONFIG)
    return p, cfg, p.deploy, p.run_dir, root


def test_capture_uses_common_without_relaxing_training_clip(monkeypatch):
    node, _ = make_multi(monkeypatch)
    n = node.multi.policies["crouch"]
    cfg, _, root = load_settings(CONFIG)
    dep = n.deploy
    joint = n.real_joint_names.index("L1")
    policy_joint = n.motion.joint_names.index("joint_L1")
    for angle in (-.97791, 1.045):
        node.obs_raw[7 + joint] = angle
        assert node._capture_current_target()
        assert np.isclose(node.target_pub[joint], angle)
        action = np.zeros(12, np.float32)
        action[policy_joint] = (angle - n.action_offset[policy_joint]) / n.action_scale[policy_joint]
        assert np.isclose(n._target_from_action(action)[joint], np.clip(angle, node.multi.lo[joint], node.multi.hi[joint]))
    node.obs_raw[7 + joint] = n.lo[joint] - .001
    assert not node._capture_current_target()
    assert not node.sent
    for name, action_value, expected in (("L2", 5., .51), ("R2", -5., -.54),
                                         ("L4", 5., 1.14), ("R4", 5., 1.19)):
        action = np.zeros(12, np.float32)
        action[n.motion.joint_names.index("joint_" + name)] = action_value
        assert np.isclose(n._target_from_action(action)[n.real_joint_names.index(name)], expected)
    tighter = n.lo.copy()
    tighter[joint] = -.96
    with pytest.raises(ValueError, match="参考动作超过硬件软件限位"):
        MotionReference(dep, root, cfg["joint_index_in_real"], tighter, n.hi)


def test_deploy_clip_is_not_compared_to_xml(monkeypatch):
    _, cfg, dep, _, root = mimic_policy(monkeypatch)
    dep["actions"]["JointPositionAction"]["clip"][0][1] = 1.04  # XML R1 upper is 1.05.
    motion = MotionReference(dep, root, cfg["joint_index_in_real"], cfg["joint_lower_limits"], cfg["joint_upper_limits"])
    assert np.isclose(motion.limits[0, 1], 1.04)


@pytest.mark.parametrize("clip", [None, [[0., 1.]], [[float("nan"), 1.]] * 12, [[1., 0.]] * 12])
def test_invalid_deploy_clip_still_rejected(monkeypatch, clip):
    _, cfg, dep, _, root = mimic_policy(monkeypatch)
    dep["actions"]["JointPositionAction"]["clip"] = clip
    with pytest.raises(ValueError, match="clip"):
        MotionReference(dep, root, cfg["joint_index_in_real"], cfg["joint_lower_limits"], cfg["joint_upper_limits"])


@pytest.mark.parametrize("filename,incident", [("20260915_161749_475060.csv", False),
                                               ("20260915_165008_038126.csv", True)])
def test_recorded_actions_restore_pre_incident_targets(monkeypatch, filename, incident):
    n, _, _, run, _ = mimic_policy(monkeypatch)
    path = run / "sim2real" / filename
    if not path.is_file():
        pytest.skip("Recorded regression log is not present in the selected run")
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    restored, recorded = [], []
    for row in rows:
        action = np.array([float(row[f"act{i}"]) for i in range(12)], np.float32)
        target = n._target_from_action(action)
        assert np.all(target >= n.lo - 1e-6) and np.all(target <= n.hi + 1e-6)
        restored.append(target)
        recorded.append([float(row[f"pub{i}"]) for i in range(12)])
    restored, recorded = np.asarray(restored), np.asarray(recorded)
    if not incident:
        np.testing.assert_allclose(restored, recorded, rtol=0., atol=1e-6)
        return
    assert np.abs(restored - recorded).max() > .5
    for name, lower, upper in (("L2", -.26, .51), ("R2", -.54, .26),
                               ("L4", 0., 1.14), ("R4", 0., 1.19)):
        joint = n.real_joint_names.index(name)
        assert restored[:, joint].min() >= lower - 1e-6
        assert restored[:, joint].max() <= upper + 1e-6
    # 历史实测越界仍由统一状态机拒绝。
    node, _ = make_multi(monkeypatch)
    row = next(row for row in rows if float(row["q1"]) > .51 + node.multi.cfg["joint_limit_tolerance"])
    node.obs_raw[7:19] = [float(row[f"q{i}"]) for i in range(12)]
    assert "反馈超出" in node.multi.state_error(task_bounds=True)


def test_named_observations_yaw_and_offsets(monkeypatch):
    n, _, dep, _, _ = mimic_policy(monkeypatch)
    yaw_quat = [np.cos(.6), 0., 0., np.sin(.6)]
    yaw = np.array([[np.cos(1.2), -np.sin(1.2), 0.], [np.sin(1.2), np.cos(1.2), 0.], [0., 0., 1.]])
    n.motion.reset(yaw_quat)
    n.motion.steps = 40
    body_quat = [np.cos(.6)*np.cos(.015), np.cos(.6)*np.sin(.015),
                 np.sin(.6)*np.sin(.015), np.sin(.6)*np.cos(.015)]
    roll = np.array([[1., 0., 0.], [0., np.cos(.03), -np.sin(.03)], [0., np.sin(.03), np.cos(.03)]])
    body = yaw @ roll
    n.obs_raw[3:7] = body_quat
    n.obs_raw[7:19] += np.arange(12) * .001
    n.obs_raw[19:31] = np.arange(12) * .01
    n.last_action[:] = np.arange(12) * .02
    terms = n._build_terms()
    with np.load(n.motion.path) as raw:
        names = raw["joint_names"].tolist()
        order = [names.index(name) for name in n.motion.joint_names]
        base = raw["body_names"].tolist().index("base")
        ref = rotation_matrix(raw["body_quat_w"][80, base])
        expected = (body.T @ yaw @ ref)[:, :2].reshape(-1)
        np.testing.assert_allclose(terms[0], np.r_[raw["joint_pos"][80, order], raw["joint_vel"][80, order]])
        np.testing.assert_allclose(terms[1], expected, atol=1e-6)
    np.testing.assert_allclose(terms[3], n.obs_raw[7:19][n.real2sim] - n.default_sim)
    assert np.concatenate(terms).shape == (69,)
    for i, name in enumerate(n.motion.joint_names):
        action = np.zeros(12, np.float32)
        action[i] = .5
        target = n._target_from_action(action)
        assert np.isclose(target[n.real2sim[i]], n.default_sim[i] + .125)
    # Yaw alignment must not erase real pitch/roll.
    n.motion.reset(body_quat)
    assert not np.allclose(n.motion.features(body_quat)["motion_anchor_ori_b"], [1, 0, 0, 1, 0, 0])


def test_reference_and_model_mismatch_rejected(monkeypatch, tmp_path):
    _, cfg, dep, _, root = mimic_policy(monkeypatch)
    broken = copy.deepcopy(dep)
    broken["observations"]["motion_command"]["history_length"] = 2
    with pytest.raises(ValueError, match="观测"):
        MotionReference(broken, root, cfg["joint_index_in_real"], cfg["joint_lower_limits"], cfg["joint_upper_limits"])
    broken = copy.deepcopy(dep)
    with np.load(dep["commands"]["motion"]["motion_file"]) as original:
        data = dict(original)
    data["model_sha256"] = np.array("wrong")
    np.savez(tmp_path / "wrong.npz", **data)
    broken["commands"]["motion"]["motion_file"] = str(tmp_path / "wrong.npz")
    with pytest.raises(ValueError, match="校验"):
        MotionReference(broken, root, cfg["joint_index_in_real"], cfg["joint_lower_limits"], cfg["joint_upper_limits"])


def test_heading_free_observation_keeps_tilt_joints_and_gyro(monkeypatch):
    node, _ = make_multi(monkeypatch, current=True)
    policy = node.multi.policies["rise"]
    motion = policy.motion
    assert motion.track_heading  # Existing exports without this field remain unchanged.
    motion.reset([1., 0., 0., 0.])
    motion.steps = 10_000
    policy.obs_raw[0:3] = [.2, -.3, .4]
    policy.obs_raw[7:19] = node.obs_raw[7:19]
    motion.track_heading = False
    terms = []
    for yaw in (-3., 0., 2.9):
        policy.obs_raw[3:7] = [np.cos(yaw/2)*np.cos(.1), np.cos(yaw/2)*np.sin(.1),
                              np.sin(yaw/2)*np.sin(.1), np.sin(yaw/2)*np.cos(.1)]
        terms.append(policy._build_terms())
    for term in terms[1:]:
        for expected, actual in zip(terms[0], term):
            np.testing.assert_allclose(actual, expected, atol=1e-6)
    np.testing.assert_allclose(terms[0][2], [.2, -.3, .4])
    assert not np.allclose(motion.features([1., 0., 0., 0.])["motion_anchor_ori_b"], terms[0][1])
    motion.track_heading = True
    assert not np.allclose(policy._build_terms()[1], terms[-1][1])


def test_heading_flag_must_be_boolean(monkeypatch):
    _, cfg, dep, _, root = mimic_policy(monkeypatch)
    dep["commands"]["motion"]["track_heading"] = "false"
    with pytest.raises(ValueError, match="track_heading"):
        MotionReference(dep, root, cfg["joint_index_in_real"], cfg["joint_lower_limits"], cfg["joint_upper_limits"])


def test_measured_crouch_v3_matches_rise_and_deployment_bounds(monkeypatch):
    node, _ = make_multi(monkeypatch, current=True, calibrated=True)
    cfg, _, root = load_settings(CONFIG)
    rise = node.multi.policies["rise"]
    dep = copy.deepcopy(rise.deploy)  # Test metadata compatibility only; no substitute policy is deployed.
    dep["commands"]["motion"]["motion_file"] = str(
        root / "source/legs_rl_lab/legs_rl_lab/tasks/mimic_task/task/nlegs_crouch/motions/stand_to_crouch_v3.npz")
    dep["commands"]["motion"]["track_heading"] = True
    down = MotionReference(dep, root, cfg["joint_index_in_real"], node.lo, node.hi)
    assert down.track_heading and down.stride == 2 and len(down.positions) == 341
    for first, last in ((0, -1), (-1, 0)):
        np.testing.assert_array_equal(down.positions[first], rise.motion.positions[last])
        np.testing.assert_array_equal(down.rotations[first], rise.motion.rotations[last])
    q = down.positions[:, rise.sim2real]
    assert (q >= node.multi.lo - 1e-6).all() and (q <= node.multi.hi + 1e-6).all()
    assert cfg["tasks"]["crouch"] is None, "Keep the obsolete real crouch policy disabled until retraining"
