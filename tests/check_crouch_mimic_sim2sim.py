"""离线检查 mimic sim2sim：69D、三种关节顺序、参考时钟、按键等待及真实策略回放。

conda activate unitree_lab
python tests/check_crouch_mimic_sim2sim.py [--task crouch|stand] [--run RUN]
需要已导出的 params/deploy.yaml 和 exported/policy.pt，不打开窗口、不连接实机。
"""

import argparse
import ast
import copy
import importlib.util
from pathlib import Path
import tempfile
import sys
from unittest.mock import patch

import glfw
import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("crouch", "stand"), default="crouch")
    parser.add_argument("--run")
    parser.add_argument("--parity-output", type=Path, help="导出真实回放的输入/输出，供 ROS 离线对齐测试")
    args = parser.parse_args()
    path = ROOT / f"source/legs_rl_lab/legs_rl_lab/tasks/mimic_task/task/nlegs_{args.task}/sim2sim.py"
    spec = importlib.util.spec_from_file_location("mimic_sim2sim_check", path)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    replay = entry.replay if args.task == "stand" else entry
    assert Path(replay.flat.LOGS_ROOT).name == f"nlegs_mimic_{args.task}"
    if not (args.run or entry.RUN):
        parser.error("下蹲检查需要 --run 指定新训练并导出的模型")
    np.random.seed(42)
    cfg = replay.flat.load_config(args.run or entry.RUN)
    runner = replay.MimicCrouchRunner(cfg, show_viewer=False)
    with np.load(runner.motion_file, allow_pickle=False) as data:
        motion = dict(data)
    names = motion["joint_names"].tolist()
    policy_names = [cfg.joint_names[i] for i in cfg.policy_to_sdk]
    order = [names.index(name) for name in policy_names]
    base = motion["body_names"].tolist().index("base")
    q0 = runner.data.qpos.copy()
    qd0 = runner.data.qvel.copy()
    assert runner.reference_frame == 0 and np.allclose(qd0, 0., atol=1e-10)
    assert np.array_equal(qd0[runner.dof_adr[cfg.policy_to_sdk]], motion["joint_vel"][0, order])
    assert np.allclose(qd0[:3], motion["body_lin_vel_w"][0, base])
    initial_rotation = Rotation.from_quat(motion["body_quat_w"][0, base], scalar_first=True)
    assert np.allclose(qd0[3:6], initial_rotation.inv().apply(motion["body_ang_vel_w"][0, base]))
    assert np.array_equal(q0[runner.qpos_adr[cfg.policy_to_sdk]], motion["joint_pos"][0, order])
    assert np.allclose(q0[:3], motion["body_pos_w"][0, base])
    assert np.allclose(q0[3:7], motion["body_quat_w"][0, base])
    if args.task == "stand":
        assert not np.allclose(q0[runner.qpos_adr], cfg.default_sdk)
        assert np.allclose(runner.ref_pos[-1], cfg.default_policy)
    else:
        assert np.array_equal(q0[runner.qpos_adr], cfg.default_sdk)
    assert not runner.history.ready and not runner.latency.buffer and not runner.policy_started
    nominal = mujoco.MjModel.from_xml_path(cfg.scene_xml)
    assert np.isclose(runner.model.body_mass[runner.base_body_id], nominal.body("base").mass[0] + 2.5)
    assert np.isclose(runner.model.body_subtreemass[runner.base_body_id], nominal.body("base").subtreemass[0] + 2.5)
    assert np.allclose(runner._target_sdk(np.zeros(12)), cfg.default_sdk)

    end_step = int(np.ceil((len(motion["joint_pos"]) - 1) / runner.frame_stride))
    for step in (0, 1, 40, 64, 114, end_step - 1, end_step, end_step + 1, 1000):
        runner.episode_step = step
        frame = min(round(step * cfg.step_dt * float(motion["fps"])), len(motion["joint_pos"]) - 1)
        assert runner.reference_frame == frame
        features = runner._observation_features()
        assert np.array_equal(features["motion_command"], np.concatenate(
            (motion["joint_pos"][frame, order], motion["joint_vel"][frame, order])))

    runner.episode_step = 40
    rotation = Rotation.from_euler("xyz", [.3, -.2, .4])
    runner.data.qpos[3:7] = rotation.as_quat(scalar_first=True)
    runner.data.qpos[runner.qpos_adr] += np.arange(12) * .002
    runner.data.qvel[runner.dof_adr] = np.arange(12) * .1
    runner.data.qvel[3:6] = [.2, -.3, .4]
    runner.last_action[:] = np.arange(12) * .3 - 1.5
    reference_rot = Rotation.from_quat(motion["body_quat_w"][80, base], scalar_first=True)
    if not runner.track_heading:
        delta = rotation.as_euler("xyz")[2] - reference_rot.as_euler("xyz")[2]
        reference_rot = Rotation.from_euler("z", delta) * reference_rot
    expected = {
        "motion_command": np.r_[motion["joint_pos"][80, order], motion["joint_vel"][80, order]],
        "motion_anchor_ori_b": (rotation.inv() * reference_rot).as_matrix()[:, :2].reshape(-1),
        "base_ang_vel": runner.data.qvel[3:6].copy(),
        "joint_pos_rel": np.array([runner.data.joint(n).qpos[0] for n in policy_names]) - cfg.default_policy,
        "joint_vel_rel": np.array([runner.data.joint(n).qvel[0] for n in policy_names]),
        "last_action": runner.last_action.copy(),
    }
    terms = runner._observation_terms()
    for name, values in expected.items():
        term = cfg.observations[name]
        if term["clip"] is not None:
            values = np.clip(values, *term["clip"])
        assert np.allclose(terms[name], values * term["scale"], atol=1e-6), name
    assert np.concatenate(list(terms.values())).size == 69
    # Same input semantics in standalone replay, unified replay and real deployment.
    sys.path.insert(0, str(ROOT / "deploy/rl_real_py"))
    from rl_real_py.motion_reference import MotionReference
    deployed = yaml.safe_load((Path(cfg.model_path).parents[1] / "params/deploy.yaml").read_text())
    heading_cfg = copy.deepcopy(cfg)
    heading_cfg.commands["motion"]["track_heading"] = False
    deployed["commands"]["motion"]["track_heading"] = False
    free = replay.MimicCrouchRunner(heading_cfg, show_viewer=False)
    reference = MotionReference(deployed, ROOT, cfg.short_joint_names, [-10.] * 12, [10.] * 12)
    free.data.qvel[3:6] = [.2, -.3, .4]
    for step in (0, 40, end_step, end_step + 100):
        free.episode_step = reference.steps = step
        observed = []
        for yaw in (-3., 0., 2.9):
            quat = Rotation.from_euler("xyz", [.1, -.2, yaw]).as_quat(scalar_first=True)
            free.data.qpos[3:7] = quat
            features = free._observation_features()
            for name, value in reference.features(quat).items():
                np.testing.assert_allclose(features[name], value, atol=1e-6)
            observed.append(features["motion_anchor_ori_b"])
            np.testing.assert_allclose(features["ang_vel"], [.2, -.3, .4])
        np.testing.assert_allclose(observed, np.broadcast_to(observed[0], (3, 6)), atol=1e-6)
    unified = ast.parse((ROOT / "scripts/sim2sim.py").read_text())
    assert any(isinstance(node, ast.Assign) and ast.unparse(node) == "MotionRunner = mimic.MimicCrouchRunner"
               for node in unified.body), "Unified replay must reuse the same observation implementation"
    action = np.linspace(-5, 5, 12)
    target = runner._target_sdk(action)
    for i, name in enumerate(policy_names):
        expected_target = np.clip(action[i] * cfg.action_scale[i] + cfg.action_offset[i],
                                  *runner.model.joint(name).range)
        assert np.isclose(target[cfg.joint_names.index(name)], expected_target)

    runner.episode_step = 0
    runner.data.qpos[:] = q0
    runner.data.qvel[:] = qd0
    runner.last_action[:] = 0
    mujoco.mj_forward(runner.model, runner.data)
    try:
        runner._wait_for_start()
    except ValueError as exc:
        assert "--auto-start" in str(exc)
    else:
        raise AssertionError("无界面时必须显式启动")

    class WaitingViewer:
        frames = 0

        def is_running(self):
            return True

        def sync(self):
            self.frames += 1
            assert runner.data.time == 0 and runner.episode_step == 0 and runner.reference_frame == 0
            assert np.array_equal(runner.data.qpos, q0) and not runner.history.ready
            if self.frames == 3:
                runner._on_key(glfw.KEY_KP_5)

    runner.viewer = WaitingViewer()
    with patch.object(replay.time, "sleep"):
        assert runner._wait_for_start()
    assert runner.viewer.frames == 3
    runner.viewer = None
    observations, frames, heights = [], [], []
    parity = {key: [] for key in ("steps", "q", "qd", "quat", "omega", "last_action", "actions", "targets")}
    policy = runner.policy

    def observe(obs):
        observations.append(obs.numpy().copy())
        frames.append(runner.reference_frame)
        heights.append(runner.data.qpos[2])
        assert obs.numel() == 69 and np.isfinite(obs.numpy()).all()
        assert np.isfinite(runner.last_torque).all()
        action = policy(obs)
        if args.parity_output:
            q, qd = runner._joint_state()
            parity["steps"].append(runner.episode_step)
            parity["q"].append(q.copy())
            parity["qd"].append(qd.copy())
            parity["quat"].append(runner.data.qpos[3:7].copy())
            parity["omega"].append(runner.data.qvel[3:6].copy())
            parity["last_action"].append(runner.last_action.copy())
            parity["actions"].append(action.numpy().copy())
            parity["targets"].append(runner._target_sdk(np.clip(action.numpy(), -cfg.policy_action_clip, cfg.policy_action_clip)))
        return action

    runner.policy = observe
    first = np.concatenate(list(runner._observation_terms().values()))
    runner.run(duration=6., realtime=False)
    assert np.array_equal(observations[0], first)
    last_frame = len(motion["joint_pos"]) - 1
    assert np.array_equal(frames, np.minimum(np.arange(len(frames)) * runner.frame_stride, last_frame))
    assert runner.end_announced and runner.policy_started and runner.reference_frame == last_frame
    assert np.isfinite(runner.data.qpos).all() and np.isclose(runner.data.time, runner.episode_step * cfg.step_dt)
    before = runner.data.qpos.copy(), runner.episode_step
    runner._on_key(ord("5"))
    runner._on_key(glfw.KEY_KP_8)
    assert np.array_equal(before[0], runner.data.qpos) and before[1] == runner.episode_step
    assert not runner.command.any()

    with tempfile.TemporaryDirectory(prefix="mimic-replay-check-") as directory:
        bad_cfg = copy.deepcopy(cfg)
        bad_motion = dict(motion, model_sha256=np.array("wrong-model"))
        bad_path = Path(directory) / "bad.npz"
        np.savez(bad_path, **bad_motion)
        bad_cfg.commands["motion"]["motion_file"] = str(bad_path)
        try:
            replay.MimicCrouchRunner(bad_cfg, show_viewer=False)
        except ValueError as exc:
            assert "校验不一致" in str(exc)
        else:
            raise AssertionError("错误模型校验值必须拒绝")
    relocated = "/old/machine/legs_rl_lab/" + str(runner.motion_file.relative_to(ROOT))
    assert replay.motion_path(relocated) == runner.motion_file
    error = runner.data.qpos[runner.qpos_adr[cfg.policy_to_sdk]] - runner.ref_pos[-1]
    if args.parity_output:
        np.savez_compressed(args.parity_output, **{k: np.asarray(v) for k, v in parity.items()},
                            observations=np.asarray(observations), joint_names=np.asarray(cfg.joint_names))
    print("PASS: 69D order/rotation, heading-free deployment parity, named joint/action mapping, frozen keypad wait, 100-to-50Hz clock, "
          "end clamp/no reset, model checksum and real-policy 6s finite rollout")
    print(f"Rollout only (not a success assertion): min base_z={min(heights):.3f}m; "
          f"final base_z={runner.data.qpos[2]:.3f}m; "
          f"final joint RMSE={np.sqrt(np.mean(error ** 2)):.4f}rad; max={abs(error).max():.4f}rad")


if __name__ == "__main__":
    main()
