"""CPU tests for Pi export order, full-frame histories and MuJoCo inference."""

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/task/flat_pi"
spec = importlib.util.spec_from_file_location("test_pi_sim2sim", TASK / "sim2sim.py")
replay = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = replay
spec.loader.exec_module(replay)


def fake_env():
    # Isaac's articulation order differs from Pi's explicit SDK policy order.
    names = [f"joint_{side}{i}" for i in range(1, 7) for side in ("R", "L")]
    sdk_names = [f"joint_{side}{i}" for side in ("R", "L") for i in range(1, 7)]
    ids = [names.index(name) for name in sdk_names]
    default = [-.1, -.1, 0., 0., 0., 0., .2, .2, -.1, -.1, 0., 0.]
    row = lambda value: torch.tensor([value], dtype=torch.float32)
    data = SimpleNamespace(
        joint_names=names, default_joint_pos=row(default),
        default_joint_stiffness=row([300.] * 10 + [50.] * 2),
        default_joint_damping=row([5.] * 10 + [1.] * 2),
        default_joint_armature=row([.0509] * 10 + [.00219] * 2),
        default_joint_friction_coeff=row([.5] * 10 + [0.] * 2),
        default_joint_dynamic_friction_coeff=row([.5] * 10 + [0.] * 2),
        default_joint_viscous_friction_coeff=row([0.] * 10 + [.54] * 2),
    )
    def actuator(indices, effort, velocity):
        return SimpleNamespace(joint_indices=indices,
            effort_limit=row([effort] * len(indices)), effort_limit_sim=row([effort] * len(indices)),
            velocity_limit=row([velocity] * len(indices)), velocity_limit_sim=row([velocity] * len(indices)),
            cfg=SimpleNamespace(min_delay=5, max_delay=25))
    legs = actuator(list(range(10)), 26., 6.)
    legs._saturation_effort = 26.
    roll = actuator([10, 11], 5.8, 29.)
    cfg = SimpleNamespace(
        scene=SimpleNamespace(
            robot=SimpleNamespace(joint_sdk_names=sdk_names, init_state=SimpleNamespace(pos=(0., 0., .555), rot=(1., 0., 0., 0.))),
            terrain=SimpleNamespace(physics_material=SimpleNamespace(dynamic_friction=.6))),
        command_ranges={"lin_vel_x": (-.3, .5), "lin_vel_y": (-.3, .3), "ang_vel_z": (-.5, .5)},
        sim=SimpleNamespace(dt=.001), decimation=20, action_scale=.25, clip_actions=18.,
        clip_observations=18., gait=SimpleNamespace(period=.4), actor_history=15, critic_history=3,
        action_randomization=True, dynamic_randomization=.02,
    )
    return SimpleNamespace(cfg=cfg, robot=SimpleNamespace(data=data, actuators={"legs": legs, "roll": roll}), joint_ids=ids)


class TestPiDeploy(unittest.TestCase):
    def test_cli_run_selection(self):
        cases = [
            ([], "default_run", None, None),
            (["--run", "other_run"], "other_run", None, None),
            (["--deploy", "deploy.yaml", "--policy", "policy.pt"], None, "deploy.yaml", "policy.pt"),
            (["--policy", "policy.pt"], "default_run", None, "policy.pt"),
        ]
        for args, expected_run, deploy, policy in cases:
            with self.subTest(args=args), mock.patch.object(replay, "RUN", "default_run"), \
                    mock.patch.object(sys, "argv", ["sim2sim.py", *args]), \
                    mock.patch.object(replay, "load_config", side_effect=ValueError("stop after parsing")) as load:
                with self.assertRaisesRegex(ValueError, "stop after parsing"):
                    replay.main()
                load.assert_called_once_with(expected_run, deploy=deploy, policy=policy, scene_xml=None)

    def test_actor_frame_order_and_scaling(self):
        # Known nonzero attitude; Euler sign and order differ from projected gravity.
        roll, pitch, yaw = .2, -.3, .4
        cr, sr = np.cos(roll/2), np.sin(roll/2)
        cp, sp = np.cos(pitch/2), np.sin(pitch/2)
        cy, sy = np.cos(yaw/2), np.sin(yaw/2)
        quat = [cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy]
        frame = replay.actor_frame(.25, [.2, -.1, .3], np.arange(12), np.arange(12)+20,
                                   -np.arange(12), [1., 2., 3.], quat)
        self.assertEqual(frame.shape, (47,))
        np.testing.assert_allclose(frame[:5], [1., 0., .4, -.2, .3], atol=1e-7)
        np.testing.assert_array_equal(frame[5:17], np.arange(12))
        np.testing.assert_allclose(frame[17:29], (np.arange(12)+20)*.05)
        np.testing.assert_array_equal(frame[29:41], -np.arange(12))
        np.testing.assert_array_equal(frame[41:44], [1., 2., 3.])
        np.testing.assert_allclose(frame[44:], [roll, pitch, yaw], atol=1e-7)

    def test_history_zero_padding_and_oldest_first(self):
        history = replay.FrameHistory(3)
        a, b, c, d = [np.arange(47, dtype=np.float32)+i*100 for i in range(4)]
        np.testing.assert_array_equal(history.update(a), np.concatenate((np.zeros(94), a)))
        history.update(b)
        np.testing.assert_array_equal(history.update(c), np.concatenate((a, b, c)))
        np.testing.assert_array_equal(history.update(d), np.concatenate((b, c, d)))

    def test_export_joint_and_actuator_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = replay.deploy_module.export_pi_deploy_cfg(fake_env(), directory)
            data = yaml.safe_load(Path(path).read_text())
            self.assertEqual(data["format_version"], 3)
            self.assertFalse(data["real_deployment_supported"])
            self.assertFalse(data["deployment_action_randomization"])
            self.assertTrue(data["training_action_randomization"]["enabled"])
            np.testing.assert_allclose(data["default_joint_pos"], [-.1, 0., 0., .2, -.1, 0.] * 2)
            self.assertEqual(data["actuators"]["roll"]["joint_ids"], [5, 11])
            self.assertEqual(data["actuators"]["legs"]["joint_ids"], [0, 6, 1, 7, 2, 8, 3, 9, 4, 10])
            self.assertEqual(data["actuators"]["legs"]["delay"], [5, 25])
            self.assertEqual(data["stiffness"], [300.] * 5 + [50.] + [300.] * 5 + [50.])

    def test_reject_legacy_observation_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = replay.deploy_module.export_pi_deploy_cfg(fake_env(), directory)
            data = yaml.safe_load(Path(path).read_text())
            data["format_version"] = 2
            Path(path).write_text(yaml.safe_dump(data))
            with self.assertRaisesRegex(ValueError, "incompatible"):
                replay.load_config(deploy=path, policy="unused.pt")

    def test_short_mujoco_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            path = replay.deploy_module.export_pi_deploy_cfg(fake_env(), directory)
            policy_path = Path(directory) / "policy.pt"
            policy = torch.nn.Linear(705, 12)
            with torch.no_grad():
                policy.weight.zero_()
                policy.bias.zero_()
            torch.jit.trace(policy.eval(), torch.zeros(1, 705)).save(str(policy_path))
            cfg = replay.load_config(deploy=path, policy=policy_path)
            np.random.seed(42)
            runner = replay.PiMujocoRunner(cfg, show_viewer=False)
            colliding = (runner.model.geom_contype != 0) | (runner.model.geom_conaffinity != 0)
            np.testing.assert_allclose(runner.model.geom_friction[colliding, 0], .6)
            runner.command[:] = [.2, 0., 0.]
            frame = runner._observation_frame()
            np.testing.assert_allclose(frame[:5], [0., 1., .4, 0., 0.], atol=1e-7)
            np.testing.assert_allclose(frame[5:], 0., atol=1e-6)
            runner.run(.1, realtime=False)
            self.assertTrue(np.isfinite(runner.data.qpos).all())
            self.assertTrue(np.isfinite(runner.data.qvel).all())
            self.assertTrue(np.isfinite(runner.last_torque).all())
            self.assertEqual(runner.history.buffer.shape, (15, 47))
            self.assertEqual(runner.episode_step, 5)
            np.testing.assert_array_equal(runner.history.buffer[:10], 0.)
            self.assertTrue(np.all((runner.latency.delay >= 5) & (runner.latency.delay <= 25)))


if __name__ == "__main__":
    unittest.main()
