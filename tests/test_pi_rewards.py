"""CPU parity checks against the pinned Mini_Pi rewards; no Isaac app required.

python tests/test_pi_rewards.py

Optional reference files under outputs/minipi_twist_review_20261028/reference
are from commit e8b787afa1e30da4f28f8cd7fefe542a3912ac27. When absent, source
parity reports SKIP; the independent numeric and state checks still run.
"""

import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/task/flat_pi/rewards.py"
REFERENCE = ROOT / "outputs/minipi_twist_review_20261028/reference/humanoid/envs/pai/pai_env.py"
REFERENCE_CONFIG = REFERENCE.with_name("pai_config.py")
DTYPE = torch.float64


def parse_required(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required Pi reward parity source is missing: {path}")
    return ast.parse(path.read_text())


def euler_xyz_from_quat(quat, wrap_to_2pi=False):
    # Independent XYZ extrinsic conversion, matching Isaac Lab's wxyz contract.
    w, x, y, z = quat.unbind(-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin((2 * (w * y - z * x)).clamp(-1, 1))
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    result = (roll, pitch, yaw)
    return tuple(angle % (2 * math.pi) for angle in result) if wrap_to_2pi else result


def wrap_to_pi(angles):
    wrapped = (angles + math.pi) % (2 * math.pi)
    return torch.where((wrapped == 0) & (angles > 0), math.pi, wrapped - math.pi)


def reference_euler_xyzw(quat):
    # The original helper wraps Isaac Gym's [0, 2pi) Euler output to [-pi, pi].
    angles = torch.stack(euler_xyz_from_quat(quat[:, [3, 0, 1, 2]], True), dim=1)
    return torch.where(angles > math.pi, angles - 2 * math.pi, angles)


def quat_from_euler(angles):
    half = angles / 2
    cr, cp, cy = torch.cos(half).unbind(-1)
    sr, sp, sy = torch.sin(half).unbind(-1)
    return torch.stack((cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                        cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy), dim=-1)


def load_rewards(path, class_name):
    source_class = next(node for node in parse_required(path).body
                        if isinstance(node, ast.ClassDef) and node.name == class_name)
    methods = [node for node in source_class.body if isinstance(node, ast.FunctionDef)
               and (node.name.startswith("_reward_") or node.name == "_neg_sqrd_exp")]
    isolated_class = ast.ClassDef(name=class_name, bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.Module(body=[isolated_class], type_ignores=[])
    namespace = {"torch": torch, "euler_xyz_from_quat": euler_xyz_from_quat,
                 "wrap_to_pi": wrap_to_pi, "get_euler_xyz_tensor": reference_euler_xyzw}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return type(class_name + "Fixture", (namespace[class_name],),
                {"_get_gait_phase": lambda self: self.stance_mask})


def reference_reward_config():
    pai_cfg = next(node for node in parse_required(REFERENCE_CONFIG).body
                   if isinstance(node, ast.ClassDef) and node.name == "PaiCfg")
    rewards = next(node for node in pai_cfg.body if isinstance(node, ast.ClassDef) and node.name == "rewards")
    scales = next(node for node in rewards.body if isinstance(node, ast.ClassDef) and node.name == "scales")
    literals = lambda nodes: {node.targets[0].id: ast.literal_eval(node.value)
                             for node in nodes if isinstance(node, ast.Assign)}
    return literals(rewards.body), literals(scales.body)


NEW_REWARDS = load_rewards(SOURCE, "PiRewardsMixin")
REFERENCE_AVAILABLE = REFERENCE.is_file() and REFERENCE_CONFIG.is_file()
ORIGINAL_REWARDS = load_rewards(REFERENCE, "PaiFreeEnv") if REFERENCE_AVAILABLE else None
REWARD_CONFIG = dict(base_height_target=0.3453, min_dist=0.15, max_dist=0.2,
                     target_feet_height=0.02, tracking_sigma_ang=0.1,
                     tracking_sigma_lin=0.1, max_contact_force=100)
ACTIVE_REWARDS = {"_reward_" + name for name in """
    joint_pos feet_clearance feet_contact_number feet_air_time foot_slip feet_distance
    knee_distance feet_contact_forces tracking_lin_vel tracking_ang_vel vel_mismatch_exp
    low_speed track_vel_hard default_hip_roll_joint_pos default_thigh_joint_pos
    default_ankle_roll_pos orientation base_height base_acc action_smoothness torques
    dof_vel dof_acc collision termination
""".split()}


def make_env(count=12, seed=7):
    env = NEW_REWARDS()
    generator = torch.Generator().manual_seed(seed)
    rand = lambda *shape: torch.randn(shape, generator=generator, dtype=DTYPE)
    env.num_envs, env.dt = count, 0.02
    env.cfg = SimpleNamespace(rewards=SimpleNamespace(
        **REWARD_CONFIG, contact_threshold=5.0, clearance_contact_threshold=0.1,
        collision_threshold=0.1, foot_height_offset=0.05, clearance_height_offset=0.05063))
    env.feet_indices = torch.tensor([3, 7])
    env.knee_indices = torch.tensor([2, 6])
    env.clearance_body_indices = env.feet_indices - 1
    env.penalised_contact_indices = torch.tensor([0, 4])
    env.pitch_joint_indices = [0, 3, 4, 6, 9, 10]
    env.roll_joint_indices = [1, 5, 7, 11]
    env.yaw_joint_indices = [2, 8]
    for name in ("dof_pos", "ref_dof_pos", "dof_vel", "last_dof_vel", "torques",
                 "actions", "last_actions", "last_last_actions"):
        setattr(env, name, rand(count, 12) * 0.1)
    env.rigid_state = rand(count, 9, 13) * 0.1
    env.rigid_state[:, :, 2] += 0.1
    env.rigid_state[:, :, 3:7] = quat_from_euler(rand(count, 9, 3) * 0.8)
    env.contact_forces = rand(count, 9, 3) * 20
    env.contact_forces[:, env.feet_indices, 2] = torch.tensor(
        [[0., 0.], [5., 0.1], [5.01, 20.], [200., 700.]], dtype=DTYPE).repeat((count + 3) // 4, 1)[:count]
    env.root_states = rand(count, 13) * 0.1
    env.root_states[:, 2] += 0.35
    env.last_root_vel = env.root_states[:, 7:13] + rand(count, 6) * 0.1
    env.base_euler_xyz = rand(count, 3) * 0.1
    env.projected_gravity = rand(count, 3) * 0.1
    env.projected_gravity[:, 2] = -1
    env.commands = rand(count, 4) * 0.3
    env.base_lin_vel = env.commands[:, :3] + rand(count, 3) * 0.05
    env.base_ang_vel = rand(count, 3) * 0.1
    env.base_ang_vel[:, 2] += env.commands[:, 2]
    env.stance_mask = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=DTYPE).repeat(
        (count + 2) // 3, 1)[:count]
    env.feet_air_time = rand(count, 2).abs() * 0.4
    env.last_contacts = rand(count, 2) > 0
    env.contact_filt = torch.zeros((count, 2), dtype=torch.bool)
    env.feet_height = rand(count, 2) * 0.02
    env.last_feet_z = (env.rigid_state[:, env.clearance_body_indices, 2] - 0.05063
                       + rand(count, 2) * 0.01)
    env.reset_buf = torch.arange(count) % 2 == 0
    env.time_out_buf = torch.arange(count) % 3 == 0
    return env


def original_from(env):
    if ORIGINAL_REWARDS is None:
        return None
    original = ORIGINAL_REWARDS()
    original.__dict__.update(copy.deepcopy(env.__dict__))
    original.rigid_state[:, :, 3:7] = env.rigid_state[:, :, [4, 5, 6, 3]]
    return original


def check(actual, expected):
    if not isinstance(expected, torch.Tensor):
        expected = torch.tensor(expected, dtype=actual.dtype)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


def test_all_25_active_rewards_and_state_match_reference():
    actual_names = {name for name in dir(NEW_REWARDS) if name.startswith("_reward_")}
    assert len(ACTIVE_REWARDS) == 25
    assert actual_names == ACTIVE_REWARDS
    if not REFERENCE_AVAILABLE:
        print(f"SKIP: pinned Mini_Pi source parity; requires {REFERENCE} and {REFERENCE_CONFIG}")
        return
    original_names = {name for name in dir(ORIGINAL_REWARDS) if name.startswith("_reward_")}
    original_config, original_scales = reference_reward_config()
    assert original_names == {"_reward_" + name for name, weight in original_scales.items() if weight != 0} == ACTIVE_REWARDS
    assert all(original_config[key] == value for key, value in REWARD_CONFIG.items())
    for seed in (0, 7, 42):
        for name in sorted(ACTIVE_REWARDS):
            env = make_env(seed=seed)
            original = original_from(env)
            result, expected = getattr(env, name)(), getattr(original, name)()
            assert result.shape == (env.num_envs,) and torch.isfinite(result).all(), name
            check(result, expected)
            for key, value in original.__dict__.items():
                if isinstance(value, torch.Tensor) and key != "rigid_state":
                    check(getattr(env, key), value)
            converted = original.rigid_state.clone()
            converted[:, :, 3:7] = original.rigid_state[:, :, [6, 3, 4, 5]]
            check(env.rigid_state, converted)


def test_airtime_multistep_planned_contact_history_and_cap():
    env = make_env(1)
    env.feet_air_time[:] = torch.tensor([0.48, 0.], dtype=DTYPE)
    env.last_contacts.zero_()
    env.contact_forces.zero_()
    original = original_from(env)
    # phase, actual right-foot force, reward, next air-time buffer
    steps = [([0, 1], 0., 0., [0.50, 0.]), ([0, 1], 0., 0., [0.52, 0.]),
             ([1, 0], 0., 0.5, [0., 0.02]), ([1, 0], 6., 0.04, [0., 0.]),
             ([1, 0], 0., 0., [0., 0.]), ([1, 0], 0., 0., [0., 0.02]),
             ([1, 0], 0., 0., [0., 0.04]), ([1, 0], 5., 0., [0., 0.06]),
             ([1, 0], 5.001, 0.08, [0., 0.])]
    for phase, force, reward, air_time in steps:
        for obj in (env, original) if original is not None else (env,):
            obj.stance_mask[:] = torch.tensor(phase)
            obj.contact_forces[:, obj.feet_indices[1], 2] = force
        check(env._reward_feet_air_time(), [reward])
        check(env.feet_air_time, [air_time])
        if original is not None:
            check(original._reward_feet_air_time(), [reward])
            for key in ("feet_air_time", "last_contacts", "contact_filt"):
                check(getattr(env, key), getattr(original, key))
        check(env.last_contacts, [[False, force > 5.]])


def test_clearance_multistep_descent_and_reward_before_contact_reset():
    env = make_env(1)
    env.contact_forces.zero_()
    env.commands.zero_()  # The original does not gate clearance on movement commands.
    env.stance_mask[:] = torch.tensor([0., 1.])
    env.feet_height.zero_()
    env.last_feet_z.zero_()
    env.rigid_state[:, env.clearance_body_indices, 2] = env.cfg.rewards.clearance_height_offset
    original = original_from(env)
    # absolute rise, force, expected reward, accumulated rise after contact reset
    steps = [(0.005, 0., 0., 0.005), (0.011, 0., 1., 0.011),
             (0.025, 0., 1., 0.025), (0.018, 0., 1., 0.018),
             (0.018, 0.1, 1., 0.018), (0.018, 0.100001, 1., 0.),
             (0.018, 0., 0., 0.), (0.036, 0., 1., 0.018)]
    for rise, force, reward, height in steps:
        for obj in (env, original) if original is not None else (env,):
            obj.rigid_state[:, obj.clearance_body_indices[0], 2] = rise + obj.cfg.rewards.clearance_height_offset
            obj.contact_forces[:, obj.feet_indices[0], 2] = force
        check(env._reward_feet_clearance(), [reward])
        check(env.feet_height, [[height, 0.]])
        if original is not None:
            check(original._reward_feet_clearance(), [reward])
            check(env.feet_height, original.feet_height)
            check(env.last_feet_z, original.last_feet_z)


def test_noncontiguous_joint_mapping_ignores_other_joints():
    env = make_env()
    error = (env.dof_pos - env.ref_dof_pos)[:, env.pitch_joint_indices].norm(dim=1)
    expected = [torch.exp(-2 * error) - 0.2 * error.clamp(0, 0.5),
                torch.exp(-env.dof_pos[:, env.roll_joint_indices].norm(dim=1).clamp(0, 50) / 0.1),
                torch.exp(-env.dof_pos[:, env.yaw_joint_indices].norm(dim=1).clamp(0, 50) / 0.1)]
    remapping = [14, 15, 8, 1, 9, 3, 4, 10, 16, 12, 6, 0]
    for key in ("dof_pos", "ref_dof_pos"):
        expanded = torch.full((env.num_envs, 17), 100., dtype=DTYPE)
        expanded[:, remapping] = getattr(env, key)
        setattr(env, key, expanded)
    env.pitch_joint_indices = [remapping[i] for i in env.pitch_joint_indices]
    env.roll_joint_indices = [remapping[i] for i in env.roll_joint_indices]
    env.yaw_joint_indices = [remapping[i] for i in env.yaw_joint_indices]
    names = ("_reward_joint_pos", "_reward_default_hip_roll_joint_pos", "_reward_default_thigh_joint_pos")
    for name, value in zip(names, expected):
        check(getattr(env, name)(), value)


def test_custom_contact_threshold_and_angular_xy_slip():
    env = make_env(4)
    env.feet_indices = torch.tensor([6, 1])
    env.cfg.rewards.contact_threshold = 2.
    env.contact_forces.zero_()
    env.contact_forces[:, 6, 2] = torch.tensor([0., 2., 2.000001, 10.])
    env.rigid_state[:, :, 7:13] = 0
    env.rigid_state[:, :, 7:10] = 1000.  # Linear foot motion is not this penalty.
    env.rigid_state[:, :, 12] = -1000.  # World yaw is not this penalty either.
    env.rigid_state[:, 6, 10:12] = torch.tensor([3., 4.])
    env.rigid_state[:, [0, 3, 7], 10:12] = 1000.  # Unselected bodies.
    check(env._reward_foot_slip(), [0., 0., math.sqrt(5), math.sqrt(5)])
    env.stance_mask[:] = torch.tensor([1., 0.])
    # This original torch.where produces float32, so compare at its native precision.
    torch.testing.assert_close(env._reward_feet_contact_number(), torch.tensor([0.35, 0.35, 1., 1.]))
    env.stance_mask[:] = torch.tensor([0., 1.])
    env.feet_air_time.fill_(0.1)
    env.last_contacts.zero_()
    check(env._reward_feet_air_time(), [0.12, 0.12, 0.24, 0.24])


def test_custom_clearance_bodies_offsets_and_contact_threshold():
    env = make_env(4)
    env.feet_indices = torch.tensor([6, 1])
    env.clearance_body_indices = torch.tensor([4, 0])
    env.cfg.rewards.clearance_height_offset = 0.2
    env.cfg.rewards.clearance_contact_threshold = 0.75
    env.contact_forces.zero_()
    env.contact_forces[:, 6, 2] = torch.tensor([0., 0.75, 0.750001, 2.])
    env.rigid_state[:, :, 2] = 10.
    env.rigid_state[:, env.clearance_body_indices, 2] = 0.31
    env.last_feet_z.fill_(0.09)
    env.feet_height.zero_()
    env.stance_mask[:] = torch.tensor([0., 1.])
    check(env._reward_feet_clearance(), [1., 1., 1., 1.])
    check(env.feet_height, [[0.02, 0.02], [0.02, 0.02], [0., 0.02], [0., 0.02]])
    check(env.last_feet_z, [[0.11, 0.11]] * 4)
    env.cfg.rewards.foot_height_offset = 0.13
    env.rigid_state[:, 6, 2] = 0.4
    env.rigid_state[:, 1, 2] = 0.6
    env.stance_mask[:] = torch.tensor([1., 0.])
    env.root_states[:, 2] = env.cfg.rewards.base_height_target + 0.4 - 0.13
    check(env._reward_base_height(), [1.] * 4)


def test_custom_collision_threshold_and_contact_force_clipping():
    env = make_env(4)
    env.penalised_contact_indices = torch.tensor([8, 5])
    env.cfg.rewards.collision_threshold = 0.75
    env.contact_forces.zero_()
    env.contact_forces[:, 8, 0] = torch.tensor([0., 0.75, 0.750001, 10.])
    env.contact_forces[:, [0, 4], :] = 1000.  # Old indices must not contribute.
    check(env._reward_collision(), [0., 0., 1., 1.])
    env.contact_forces[:, env.feet_indices[0], 2] = torch.tensor([100., 101., 500., 900.])
    check(env._reward_feet_contact_forces(), [0., 1., 400., 400.])


def test_euler_wxyz_and_original_xyzw_with_yaw_invariance():
    env = make_env(5)
    angles = torch.tensor([[0., 0., 0.], [0., 0., 2.4], [0.2, 0., -1.2],
                           [0., -0.3, 1.8], [-0.2, 0.3, -2.7]], dtype=DTYPE)
    quats = quat_from_euler(angles)
    check(torch.stack(euler_xyz_from_quat(quats), dim=1), angles)
    check(reference_euler_xyzw(quats[:, [1, 2, 3, 0]]), angles)
    env.rigid_state[:, env.feet_indices[0], 3:7] = quats
    env.rigid_state[:, env.feet_indices[1], 3:7] = -quats  # q and -q are equivalent.
    expected = (torch.exp(-angles[:, 0].abs() / 0.1) + torch.exp(-angles[:, 1].abs() / 0.1)) / 2
    check(env._reward_default_ankle_roll_pos(), expected)
    if REFERENCE_AVAILABLE:
        check(env._reward_default_ankle_roll_pos(), original_from(env)._reward_default_ankle_roll_pos())


def test_actual_exponential_uses_a_cubed_and_command_normalization():
    env = make_env(3)
    x = torch.tensor([0., 0.1, 0.2], dtype=DTYPE)
    a = 0.2
    check(env._neg_sqrd_exp(x, a), torch.exp(-x.square() / a**3))
    assert not torch.allclose(env._neg_sqrd_exp(x, a), torch.exp(-x.square() / a**2))
    env.cfg.rewards.tracking_sigma_lin = a
    env.cfg.rewards.tracking_sigma_ang = a
    env.commands[:, :3] = torch.tensor([[-1., 2., -2.], [0., 0., 0.], [1., -2., 2.]])
    env.base_lin_vel[:, :2] = env.commands[:, :2] + 0.1
    env.base_ang_vel[:, 2] = env.commands[:, 2] + 0.1
    error_xy = 0.1 / (1 + env.commands[:, :2].abs())
    error_yaw = 0.1 / (1 + env.commands[:, 2].abs())
    check(env._reward_tracking_lin_vel(), torch.exp(-error_xy.square() / a**3).mean(dim=1))
    check(env._reward_tracking_ang_vel(), torch.exp(-error_yaw.square() / a**3))


def test_low_speed_boundaries_sign_priority_and_zero_command():
    env = make_env(9)
    env.commands[:, 0] = torch.tensor([0., 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, -0.2, 0.2], dtype=DTYPE)
    env.base_lin_vel[:, 0] = torch.tensor([0., 0., 0.099, 0.1, 0.24, 0.240001, -0.1, -0.12, 0.], dtype=DTYPE)
    check(env._reward_low_speed(), [0., 0., -1., 1.2, 1.2, 0., -2., 1.2, -2.])


if __name__ == "__main__":
    tests = [value for name, value in globals().copy().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    detail = "all 25 active terms match the pinned reference" if REFERENCE_AVAILABLE else "source parity SKIPPED"
    print(f"PASS: {len(tests)} CPU Pi reward checks; {detail}")
