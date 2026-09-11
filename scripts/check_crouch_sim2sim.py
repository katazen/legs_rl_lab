"""验证真实限位初姿态、按键等待、直接接管和真实导出策略的短回放。

python scripts/check_crouch_sim2sim.py [--run RUN]
需要该 run 的 params/deploy.yaml 和 exported/policy.pt；不创建窗口，不连接实机。
"""

import argparse
import importlib.util
from pathlib import Path
from unittest.mock import patch

import glfw
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/task/flat/sim2sim_crouch.py"
spec = importlib.util.spec_from_file_location("crouch_replay_check", path)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=replay.RUN)
    args = parser.parse_args()
    cfg = replay.flat.load_config(args.run)
    default = cfg.default_policy.copy()
    offset = cfg.action_offset.copy()
    runner = replay.CrouchRunner(cfg, show_viewer=False)
    q0 = runner.data.qpos.copy()
    assert runner.data.time == 0 and runner.episode_step == 0
    assert np.count_nonzero(runner.data.qvel) == 0
    assert not runner.policy_started and not runner.history.ready
    for name, side in replay.pose_generator.CROUCH_LIMIT_SIDE.items():
        joint = runner.model.joint(name)
        assert runner.data.qpos[joint.qposadr[0]] == joint.range[side], name
    for name in ("joint_L6", "joint_R6"):
        joint = runner.model.joint(name)
        assert joint.range[0] < runner.data.qpos[joint.qposadr[0]] < joint.range[1]
    assert np.array_equal(cfg.default_policy, default)
    assert np.array_equal(cfg.action_offset, offset)
    assert np.allclose(runner._target_sdk(np.zeros(12)), cfg.default_sdk)

    feet = [runner.model.body(f"Link_{side}6").id for side in ("L", "R")]
    normals = runner.data.xmat[feet].reshape(2, 3, 3)[:, :, 2]
    assert np.rad2deg(np.arccos(normals[:, 2])).max() < 0.5
    caps = np.flatnonzero(np.isin(runner.model.geom_bodyid, feet) & (runner.model.geom_contype != 0))
    axes = runner.data.geom_xmat[caps].reshape(-1, 3, 3)[:, :, 2]
    centers = runner.data.geom_xpos[caps]
    z = centers[:, 2, None] + axes[:, 2, None] * runner.model.geom_size[caps, 1, None] * [-1, 1]
    z -= runner.model.geom_size[caps, 0, None]
    assert np.isclose(z.min(), replay.pose_generator.CLEARANCE, atol=1e-7)
    assert np.ptp(z) <= 0.001

    policy = runner.policy
    observations = []

    def observe(obs):
        observations.append(obs.clone())
        return policy(obs)

    runner.policy = observe
    try:
        runner._wait_for_start()
    except ValueError as exc:
        assert "--auto-start" in str(exc)
    else:
        raise AssertionError("无窗口、未启动时应报错，而不是静默运行或永久等待")

    class WaitingViewer:
        frames = 0

        def is_running(self):
            return True

        def sync(self):
            self.frames += 1
            assert runner.data.time == 0 and runner.episode_step == 0
            assert np.array_equal(runner.data.qpos, q0)
            assert not runner.history.ready and not observations
            if self.frames == 3:
                runner._on_key(glfw.KEY_KP_5)

    runner.viewer = WaitingViewer()
    with patch.object(runner, "_update_markers"), patch.object(replay.time, "sleep"):
        assert runner._wait_for_start()
    assert runner.viewer.frames == 3
    runner.viewer = None
    assert not runner.policy_started

    for digit, (axis, increment) in replay.flat.KEY_BINDINGS.items():
        runner.command[:] = 0
        runner._on_key(glfw.KEY_KP_0 + int(digit))
        assert np.isclose(runner.command[axis], increment)
        runner.command[:] = 0
        runner._on_key(ord(digit))
        assert np.isclose(runner.command[axis], increment)
    runner.command[:] = 0
    runner._on_key(ord("5"))

    first_terms = runner._observation_terms()
    runner.run(duration=cfg.step_dt * 10, realtime=False)
    assert runner.policy_started and len(observations) == 10 and runner.episode_step == 10
    start = 0
    first_obs = observations[0].numpy()
    for name, term in cfg.observations.items():
        width = len(term["scale"])
        length = int(term["history_length"])
        block = first_obs[start:start + width * length].reshape(length, width)
        assert np.allclose(block, first_terms[name]), name
        start += width * length
    assert start == first_obs.size
    assert np.isfinite(runner.data.qpos).all() and np.isfinite(runner.last_torque).all()
    assert np.array_equal(cfg.default_policy, default) and np.array_equal(cfg.action_offset, offset)
    before = runner.episode_step
    runner._on_key(glfw.KEY_KP_5)
    assert runner.episode_step == before
    print("PASS: exact ten XML stops, solved feet/base, frozen wait, keypad/main digits, "
          "crouch first observation/history, unchanged standing action offset, real policy 10-step replay")


if __name__ == "__main__":
    main()
