"""检查 W1196 训练配置、资产前置条件、短仿真、PPO 与 JIT/ONNX 导出。"""

import argparse
import ast
import copy
import math
import os
from pathlib import Path
import tempfile
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--plane-contact-only", action="store_true", help="只检查名义站姿几何和平地接触，不运行 PPO")
parser.add_argument("--asset-preflight-only", action="store_true", help="仅检查配置与用户资产的训练前置条件")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import yaml
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry
from pxr import Usd, UsdPhysics
from rsl_rl.runners import OnPolicyRunner

import legs_rl_lab.tasks  # noqa: F401
from legs_rl_lab.assets.nlegs_v2.nlegs_v2 import (
    W1196_CFG,
    W1196_FEET_WIDTH,
    W1196_FOOT_HEIGHT_OFFSET,
    W1196_NOMINAL_BASE_HEIGHT,
)
from legs_rl_lab.tasks.nlegs_task import mdp
from legs_rl_lab.tasks.nlegs_task.task.rough import body_env_cfg
from legs_rl_lab.utils.export_deploy_cfg import export_deploy_cfg


TASK = "nlegs_body_rough"
ROOT = Path(__file__).resolve().parents[1]
FEET = ["Link_R6", "Link_L6"]
TERRAINS = {"flat", "random_rough", "hf_pyramid_slope", "hf_pyramid_slope_inv"}
HISTORY = 15
POLICY_DIM = 47 * HISTORY
CRITIC_DIM = 249 * HISTORY
BASELINE = ROOT / "logs/rsl_rl/nlegs_rough/2026-08-26_20-37-55/params"
ASSET_DIR = ROOT / "source/legs_rl_lab/legs_rl_lab/assets/nlegs_v2"


def check_independence(cfg, agent):
    assert body_env_cfg.RoughBodyEnvCfg.__bases__ == (ManagerBasedRLEnvCfg,)
    assert body_env_cfg.RoughBodyPPORunnerCfg.__bases__ == (RslRlOnPolicyRunnerCfg,)
    for name in ("scene", "observations", "actions", "commands", "rewards", "terminations",
                 "events", "curriculum", "gait"):
        assert type(getattr(cfg, name)).__module__ == body_env_cfg.__name__, name
    for item in (cfg.observations.policy, cfg.observations.critic, agent.actor, agent.critic):
        assert type(item).__module__ == body_env_cfg.__name__
    # 只核对导入边界，不约束独立配置的源码排列方式。
    for node in ast.walk(ast.parse(Path(body_env_cfg.__file__).read_text())):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names]
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            for name in names:
                assert not name.endswith("rough_env_cfg"), name
                assert not name.startswith("legs_rl_lab.tasks.nlegs_task.agents"), name


def assert_same(actual, expected, path):
    """Compare saved configuration values, including callable strings and slices."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), (
            path, set(actual) ^ set(expected)
        )
        for key in expected:
            assert_same(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (tuple, list)):
        assert isinstance(actual, (tuple, list)) and len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_same(a, e, f"{path}[{i}]")
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def check_configs():
    cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    play = load_cfg_from_registry(TASK, "play_env_cfg_entry_point")
    agent = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    saved = yaml.unsafe_load((BASELINE / "env.yaml").read_text())
    saved_agent = yaml.safe_load((BASELINE / "agent.yaml").read_text())
    for group in ("policy", "critic"):
        saved["observations"][group]["history_length"] = HISTORY
    # 当前任务已关闭踝 PD 随机化。
    del saved["events"]["randomize_ankle_gains"]
    # 零偏采用当前编码器模型，且只施加到左右踝 pitch。
    saved["actions"]["JointPositionAction"]["class_type"] = (
        "legs_rl_lab.tasks.nlegs_task.mdp.actions:ZeroBiasJointPositionAction"
    )
    saved["events"]["joint_zero_bias"]["params"]["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=[".*5"]
    ).to_dict()
    # 保留 roll 约束，减弱 hip yaw，取消额外横移惩罚并分别约束脚和身体自转。
    rewards = saved["rewards"]
    del rewards["lateral_move"]
    rewards["base_yaw"] = {
        "func": "legs_rl_lab.tasks.nlegs_task.task.rough.body_rewards:base_yaw_no_command_l2",
        "weight": -1.0,
        "params": {},
    }
    rewards["joint_deviation_hip_yaw"] = copy.deepcopy(rewards["joint_deviation_legs"])
    rewards["joint_deviation_hip_yaw"]["weight"] = -0.1
    rewards["joint_deviation_hip_yaw"]["params"]["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=[".*3"]
    ).to_dict()
    rewards["joint_deviation_legs"]["params"]["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=[".*2", ".*6"]
    ).to_dict()
    rewards["feet_contact_yaw"] = {
        "func": "legs_rl_lab.tasks.nlegs_task.task.rough.body_rewards:feet_contact_yaw_l2",
        "weight": -0.1,
        "params": {
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET, preserve_order=True).to_dict(),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET, preserve_order=True).to_dict(),
            "min_force": 5.0,
            "full_force": 20.0,
        },
    }
    # 只覆盖本体迁移所需的几何、质量范围和实体顺序，其余继续与原基准核对。
    saved["scene"]["robot"]["init_state"]["pos"] = (0.0, 0.0, 0.555)
    saved["observations"]["critic"]["height_scan"]["params"]["offset"] = W1196_NOMINAL_BASE_HEIGHT
    rewards["base_height"]["params"]["target_height"] = W1196_NOMINAL_BASE_HEIGHT
    rewards["feet_y_distance"]["params"]["threshold"] = W1196_FEET_WIDTH
    rewards["feet_clearance"]["params"]["foot_height_offset"] = W1196_FOOT_HEIGHT_OFFSET
    saved["terminations"]["base_height"]["params"]["minimum_height"] = 0.14
    saved["events"]["add_base_mass"]["params"]["mass_distribution_params"] = (-0.9, 0.9)
    for term in [*rewards.values(), saved["terminations"]["base_height"]]:
        for selector_name in ("asset_cfg", "sensor_cfg"):
            selector = term["params"].get(selector_name)
            if isinstance(selector, dict) and selector.get("body_names") in (".*6", [".*6"], FEET):
                term["params"][selector_name] = SceneEntityCfg(
                    selector["name"], body_names=FEET, preserve_order=True
                ).to_dict()
    saved_agent["run_name"] = "w1196"
    check_independence(cfg, agent)
    check_independence(play, agent)
    for config in (cfg, play):
        assert Path(config.scene.robot.spawn.usd_path).resolve() == (
            ASSET_DIR / "w1196/w1196.usd"
        )
        assert config.scene.robot is not W1196_CFG
        assert set(config.scene.terrain.terrain_generator.sub_terrains) == TERRAINS
        terrain = config.scene.terrain
        assert not terrain.terrain_generator.curriculum
        assert terrain.max_init_terrain_level is None
        assert tuple(terrain.terrain_generator.difficulty_range) == (0.0, 1.0)
        assert getattr(config.curriculum, "terrain_levels", None) is None
        for name, proportion in zip(
            ("flat", "random_rough", "hf_pyramid_slope", "hf_pyramid_slope_inv"), (0.2, 0.2, 0.3, 0.3)
        ):
            assert math.isclose(terrain.terrain_generator.sub_terrains[name].proportion, proportion)
        assert math.isclose(config.gait.period, 0.6)
        assert math.isclose(config.gait.stance_ratio, 0.55)
        assert list(config.gait.feet_offset) == [0.0, 0.5]
        assert math.isclose(config.gait.period * config.gait.stance_ratio, 0.33)
        assert math.isclose(config.gait.period * (1 - config.gait.stance_ratio), 0.27)
        assert config.observations.policy.history_length == HISTORY
        assert config.observations.critic.history_length == HISTORY
        actual = config.to_dict()
        for section in ("actions", "observations", "commands", "rewards", "terminations", "events", "gait"):
            assert_same(actual[section], saved[section], section)
        assert_same(actual["scene"]["robot"]["init_state"], saved["scene"]["robot"]["init_state"], "robot.init_state")
        actuators = copy.deepcopy(saved["scene"]["robot"]["actuators"])
        # 保留用户当前执行器调整：1～5 同组、速度 6、踝 pitch 200/5、roll 速度 29。
        del actuators["ankle_pitch"]
        actuators["legs"]["joint_names_expr"].append(".*5")
        actuators["legs"]["velocity_limit"] = 6.0
        actuators["legs"]["effort_limit_sim"] = 26.0
        actuators["legs"]["velocity_limit_sim"] = 6.0
        actuators["legs"]["stiffness"][".*5"] = 200.0
        actuators["legs"]["damping"][".*5"] = 5.0
        actuators["ankle_roll"]["velocity_limit_sim"] = 29.0
        actuators["ankle_roll"]["effort_limit"] = 5.8
        actuators["ankle_roll"]["velocity_limit"] = 29.0
        assert_same(actual["scene"]["robot"]["actuators"], actuators, "robot.actuators")
        for field in ("decimation", "episode_length_s"):
            assert_same(actual[field], saved[field], field)
        assert config.actions.JointPositionAction.class_type is mdp.ZeroBiasJointPositionAction
        assert getattr(config.events, "base_com", None) is None
        assert getattr(config.events, "link_mass", None) is None
    assert cfg.scene.num_envs == 2048 and play.scene.num_envs == 32
    assert (cfg.scene.terrain.terrain_generator.num_rows, cfg.scene.terrain.terrain_generator.num_cols) == (10, 20)
    assert (play.scene.terrain.terrain_generator.num_rows, play.scene.terrain.terrain_generator.num_cols) == (5, 5)
    assert agent.experiment_name == TASK and not agent.resume
    assert agent.algorithm.num_mini_batches == 4
    assert agent.obs_groups == {"actor": ["policy"], "critic": ["critic"]}
    current_agent = agent.to_dict()
    for section in ("actor", "critic", "algorithm"):
        assert_same(current_agent[section], saved_agent[section], f"agent.{section}")
    for field in ("seed", "device", "num_steps_per_env", "max_iterations", "save_interval", "clip_actions",
                  "check_for_nan", "run_name", "logger", "resume", "load_run", "load_checkpoint", "class_name"):
        assert_same(current_agent[field], saved_agent[field], f"agent.{field}")
    print("PASS W1196 configs: standalone baseline, geometry, foot order, explicit actuator limits, 705/3735 observations", flush=True)
    return cfg, agent


def check_exports(env, agent, runner, wrapped, run_dir):
    import onnx
    from onnx.reference import ReferenceEvaluator

    robot = env.scene["robot"]
    export_deploy_cfg(env, run_dir, policy_action_clip=agent.clip_actions)
    deploy = yaml.safe_load((Path(run_dir) / "params/deploy.yaml").read_text())
    assert deploy["format_version"] == 2
    assert math.isclose(deploy["gait_period"], 0.6)
    assert deploy["joint_names"] == W1196_CFG.joint_sdk_names
    for side in "RL":
        sdk_id = deploy["joint_names"].index(f"joint_{side}5")
        assert deploy["stiffness"][sdk_id] == 200.0
        assert deploy["damping"][sdk_id] == 5.0
        groups = [group for group in deploy["actuators"].values() if sdk_id in group["joint_ids"]]
        assert len(groups) == 1
        assert groups[0]["torque_model"] == "dc_motor"
        assert groups[0]["delay"] == [4, 6]
    assert deploy["base_init_state"]["pos"] == list(env.cfg.scene.robot.init_state.pos)
    assert deploy["policy_action_clip"] == agent.clip_actions
    assert "gait_phase" in deploy["observations"]
    assert all(term["history_length"] == HISTORY for term in deploy["observations"].values())
    assert sum(len(term["scale"]) * term["history_length"]
               for term in deploy["observations"].values()) == POLICY_DIM
    torch.testing.assert_close(torch.tensor(deploy["default_joint_pos"]), robot.data.default_joint_pos[0].cpu())
    action = deploy["actions"]["JointPositionAction"]
    assert len(action["scale"]) == 12 and len(action["offset"]) == 12
    np.testing.assert_allclose(action["offset"], deploy["default_joint_pos"], atol=1e-7)

    model_dir = Path(run_dir) / "exported"
    runner.export_policy_to_jit(str(model_dir))
    runner.export_policy_to_onnx(str(model_dir))
    model = torch.jit.load(str(model_dir / "policy.pt"), map_location="cpu")
    current_obs = wrapped.get_observations()
    with torch.inference_mode():
        expected = runner.get_inference_policy()(current_obs).cpu()
        result = model(current_obs["policy"].cpu())
    assert result.shape == (env.num_envs, 12) and torch.isfinite(result).all()
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)
    graph = onnx.load(str(model_dir / "policy.onnx"))
    onnx.checker.check_model(graph)
    assert len(graph.graph.input) == 1 and len(graph.graph.output) == 1
    assert graph.graph.input[0].type.tensor_type.shape.dim[-1].dim_value == POLICY_DIM
    assert graph.graph.output[0].type.tensor_type.shape.dim[-1].dim_value == 12
    onnx_result, = ReferenceEvaluator(graph).run(
        None, {graph.graph.input[0].name: current_obs["policy"][:1].cpu().numpy()}
    )
    np.testing.assert_allclose(onnx_result, expected[:1].numpy(), rtol=1e-4, atol=1e-5)
    print("SKIP MuJoCo replay: matching collision/actuator XML is user-owned; old robot replay is not evidence", flush=True)


def check_randomization(env):
    robot = env.scene["robot"]
    base = robot.body_names.index("base")
    others = [i for i in range(robot.num_bodies) if i != base]
    zero_cfg = env.event_manager.get_term_cfg("joint_zero_bias").params["asset_cfg"]
    zero_ids = zero_cfg.joint_ids
    assert {robot.joint_names[i] for i in zero_ids} == {"joint_R5", "joint_L5"}
    unbiased_ids = [i for i in range(robot.num_joints) if i not in zero_ids]
    stage = Usd.Stage.Open(env.cfg.scene.robot.spawn.usd_path)
    assert stage is not None
    nominal_by_body = {
        prim.GetName(): tuple(UsdPhysics.MassAPI(prim).GetCenterOfMassAttr().Get())
        for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.MassAPI)
    }
    initial = robot.root_physx_view.get_coms().clone()
    nominal = torch.tensor([nominal_by_body[name] for name in robot.body_names],
                           dtype=initial.dtype, device=initial.device)
    assert torch.isfinite(nominal).all()
    torch.testing.assert_close(initial[..., :3], nominal.expand(env.num_envs, -1, -1), rtol=0, atol=1e-6)
    for _ in range(3):
        env.reset()
        bias = env._joint_zero_bias
        assert bias[:, zero_ids].ge(-0.05).all() and bias[:, zero_ids].le(0.05).all()
        assert bias[:, zero_ids].abs().sum() > 0
        assert torch.count_nonzero(bias[:, unbiased_ids]) == 0
        masses = robot.root_physx_view.get_masses()
        default_mass = robot.data.default_mass.to(masses.device)
        added = masses[:, base] - default_mass[:, base]
        assert added.ge(-0.9 - 1e-5).all() and added.le(0.9 + 1e-5).all()
        torch.testing.assert_close(masses[:, others], default_mass[:, others])
        for actuator in robot.actuators.values():
            ids = actuator.joint_indices
            for gains, nominal_gains in (
                (actuator.stiffness, robot.data.default_joint_stiffness[:, ids]),
                (actuator.damping, robot.data.default_joint_damping[:, ids]),
            ):
                torch.testing.assert_close(gains, nominal_gains)
        # 质量随机化可改变惯量主轴四元数，只核对质心位置。
        torch.testing.assert_close(robot.root_physx_view.get_coms()[..., :3],
                                   initial[..., :3], rtol=0, atol=0)

    saved_bias = env._joint_zero_bias.clone()
    reset_ids = torch.tensor([1, 3], device=env.device)
    untouched_ids = torch.tensor([0, 2], device=env.device)
    # 选中环境的旧全关节偏置必须清除，未 reset 环境必须原样保留。
    env._joint_zero_bias[reset_ids] = 0.04
    mdp.randomize_joint_zero_bias(env, reset_ids, (0.03, 0.03), asset_cfg=zero_cfg)
    torch.testing.assert_close(env._joint_zero_bias[untouched_ids], saved_bias[untouched_ids], rtol=0, atol=0)
    assert torch.count_nonzero(env._joint_zero_bias[reset_ids][:, unbiased_ids]) == 0
    torch.testing.assert_close(env._joint_zero_bias[reset_ids][:, zero_ids],
                               torch.full((2, 2), 0.03, device=env.device))
    # 不传 joint selector 的其他任务仍给全部关节采样；None 代表全部环境。
    mdp.randomize_joint_zero_bias(env, None, (0.025, 0.025))
    torch.testing.assert_close(env._joint_zero_bias, torch.full_like(saved_bias, 0.025))
    env._joint_zero_bias.copy_(saved_bias)

    action = env.action_manager.get_term("JointPositionAction")
    action.process_actions(torch.zeros((env.num_envs, 12), device=env.device))
    targets = action.processed_actions.clone()
    for bias in (-0.05, 0.0, 0.05):
        env._joint_zero_bias.zero_()
        env._joint_zero_bias[:, zero_ids] = bias
        physical = robot.data.joint_pos - robot.data.default_joint_pos
        torch.testing.assert_close(mdp.joint_pos_rel_biased(env),
                                   physical + env._joint_zero_bias)
        torch.testing.assert_close(mdp.joint_pos_rel(env), physical)
        for _ in range(4):
            action.apply_actions()
            torch.testing.assert_close(robot.data.joint_pos_target, targets - env._joint_zero_bias)
            torch.testing.assert_close(robot.data.joint_pos_target[:, unbiased_ids], targets[:, unbiased_ids])
            torch.testing.assert_close(action.processed_actions, targets, rtol=0, atol=0)
    env._joint_zero_bias.copy_(saved_bias)


def check_asset_geometry(cfg):
    """Inspect actual USD physics, including collision owners and source-MJCF limits."""
    import mujoco

    stage = Usd.Stage.Open(cfg.scene.robot.spawn.usd_path)
    assert stage is not None
    bodies = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert {prim.GetName() for prim in bodies} == {"base", *(f"Link_{s}{i}" for s in "RL" for i in range(1, 7))}
    total_mass = sum(float(UsdPhysics.MassAPI(prim).GetMassAttr().Get()) for prim in bodies)
    model = mujoco.MjModel.from_xml_path(str(ASSET_DIR / "w1196.xml"))
    source_mass = float(model.body_mass.sum())
    assert math.isclose(total_mass, source_mass, abs_tol=1e-5), (total_mass, source_mass)
    # Converted assets can instance the collision geometry beneath each rigid body.
    collisions = [prim for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
                  if prim.HasAPI(UsdPhysics.CollisionAPI)
                  and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()]
    owners = set()
    for prim in collisions:
        owner = prim.GetParent()
        while owner and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
            owner = owner.GetParent()
        assert owner, str(prim.GetPath())
        owners.add(owner.GetName())
    missing_feet = set(FEET) - owners
    if missing_feet:
        raise RuntimeError(
            f"用户 USD 缺少脚部碰撞体 {sorted(missing_feet)}；请在用户资产中配置足部碰撞并重新导出 USD，"
            "当前测试不修改 XML/USD，也不使用旧机器人代替验证。"
        )
    joints = [UsdPhysics.RevoluteJoint(prim) for prim in stage.Traverse()
              if prim.IsA(UsdPhysics.RevoluteJoint)]
    assert len(joints) == 12
    for joint in joints:
        name = joint.GetPrim().GetName()
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert joint_id >= 0, name
        limits = [joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()]
        np.testing.assert_allclose(np.deg2rad(limits), model.jnt_range[joint_id], atol=1e-6)
    print(f"PASS W1196 asset: {len(bodies)} bodies, 12 limits, {total_mass:.6f} kg, "
          f"{len(collisions)} collision shapes with both feet covered", flush=True)


def check_plane_contact(cfg):
    """A nominal PD drop proves contact support, not a trained walking policy."""
    cfg.scene.num_envs = 4
    cfg.sim.device = args.device
    cfg.scene.sky_light.spawn.texture_file = None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.terrain.visual_material = None
    cfg.commands.base_velocity.debug_vis = False
    cfg.observations.policy.enable_corruption = False
    for name in ("physics_material", "add_base_mass", "joint_zero_bias", "push_robot"):
        setattr(cfg.events, name, None)
    cfg.events.reset_base.params["pose_range"] = {}
    cfg.events.reset_base.params["velocity_range"] = {}
    cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    env = gym.make(TASK, cfg=cfg).unwrapped
    try:
        env.reset()
        robot = env.scene["robot"]
        feet_ids, _ = robot.find_bodies(FEET, preserve_order=True)
        feet_pos = robot.data.body_pos_w[:, feet_ids]
        width = feet_pos[:, 1, 1] - feet_pos[:, 0, 1]
        nominal_height = robot.data.root_pos_w[:, 2] - feet_pos[:, :, 2].mean(dim=1) + W1196_FOOT_HEIGHT_OFFSET
        torch.testing.assert_close(width, torch.full_like(width, W1196_FEET_WIDTH), rtol=0, atol=1e-4)
        torch.testing.assert_close(nominal_height, torch.full_like(nominal_height, W1196_NOMINAL_BASE_HEIGHT),
                                   rtol=0, atol=1e-4)
        contact = env.scene.sensors["contact_forces"]
        sensor_ids = [contact.body_names.index(name) for name in FEET]
        support_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        min_height = torch.full((env.num_envs,), float("inf"), device=env.device)
        for _ in range(round(1.0 / env.step_dt)):
            obs, reward, terminated, truncated, _ = env.step(torch.zeros((env.num_envs, 12), device=env.device))
            assert not (terminated | truncated).any(), "Nominal PD contact smoke reset or fell"
            assert torch.isfinite(reward).all() and all(torch.isfinite(value).all() for value in obs.values())
            height = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
            min_height = torch.minimum(min_height, height)
            support_seen |= contact.data.net_forces_w[:, sensor_ids, 2].sum(dim=1) > 5.0
        assert support_seen.all(), "No ground support detected: check foot collisions"
        assert min_height.min() > W1196_NOMINAL_BASE_HEIGHT - 0.15, min_height
        print(f"PASS W1196 plane contact: nominal width={width.mean().item():.6f} m, "
              f"base height={nominal_height.mean().item():.6f} m, "
              f"1 s PD support in all four environments, minimum base={min_height.min().item():.6f} m", flush=True)
    finally:
        env.close()


def main():
    cfg, agent = check_configs()
    check_asset_geometry(cfg)
    if args.asset_preflight_only:
        return
    if args.plane_contact_only:
        check_plane_contact(cfg)
        return
    cfg.seed = agent.seed
    cfg.scene.num_envs = 4
    cfg.sim.device = args.device
    cfg.scene.sky_light.spawn.texture_file = None
    cfg.scene.terrain.visual_material = None
    cfg.scene.terrain.terrain_generator.num_rows = 2
    cfg.scene.terrain.terrain_generator.num_cols = 4
    cfg.commands.base_velocity.debug_vis = False
    env = gym.make(TASK, cfg=cfg).unwrapped
    try:
        robot = env.scene["robot"]
        check_randomization(env)
        for index in range(1, 7):
            # 现有镜像增强按同一级左右关节成对交换。
            assert set(robot.joint_names[2 * index - 2:2 * index]) == {
                f"joint_R{index}", f"joint_L{index}"
            }
        for side in "RL":
            for index, angle in ((1, -0.1), (4, 0.2), (5, -0.1)):
                joint = robot.joint_names.index(f"joint_{side}{index}")
                torch.testing.assert_close(robot.data.default_joint_pos[:, joint],
                                           torch.full((4,), angle, device=env.device))
            for index, stiffness in ((1, 200.0), (2, 100.0), (3, 100.0), (4, 250.0), (5, 200.0), (6, 40.0)):
                joint = robot.joint_names.index(f"joint_{side}{index}")
                torch.testing.assert_close(
                    robot.data.default_joint_stiffness[:, joint],
                    torch.full((4,), stiffness, device=env.device),
                )
        ankle = robot.actuators["ankle_roll"]
        torch.testing.assert_close(ankle.velocity_limit, torch.full((4, 2), 29.0, device=env.device))
        # 显式设置模型与物理限幅，避免继承 USD 的默认驱动值。
        torch.testing.assert_close(ankle.effort_limit, torch.full((4, 2), 5.8, device=env.device))
        torch.testing.assert_close(ankle.effort_limit_sim, torch.full((4, 2), 5.8, device=env.device))
        obs, _ = env.reset()
        assert obs["policy"].shape == (4, POLICY_DIM)
        assert obs["critic"].shape == (4, CRITIC_DIM)
        for name in ("feet_slide", "feet_clearance"):
            params = env.reward_manager.get_term_cfg(name).params
            assert set(robot.body_names[i] for i in params["asset_cfg"].body_ids) == set(FEET)
        slide = env.reward_manager.get_term_cfg("feet_slide").params
        contact = env.scene.sensors["contact_forces"]
        feet_order = [robot.body_names[i] for i in slide["asset_cfg"].body_ids]
        assert [contact.body_names[i] for i in slide["sensor_cfg"].body_ids] == feet_order
        gait = env.reward_manager.get_term_cfg("gait").params
        assert [contact.body_names[i] for i in gait["sensor_cfg"].body_ids] == feet_order
        twist = env.reward_manager.get_term_cfg("feet_contact_yaw")
        base_yaw = env.reward_manager.get_term_cfg("base_yaw")
        assert [robot.body_names[i] for i in twist.params["asset_cfg"].body_ids] == FEET
        assert [contact.body_names[i] for i in twist.params["sensor_cfg"].body_ids] == FEET
        for name, indices in (("joint_deviation_legs", (2, 6)), ("joint_deviation_hip_yaw", (3,))):
            joint_ids = env.reward_manager.get_term_cfg(name).params["asset_cfg"].joint_ids
            assert {robot.joint_names[i] for i in joint_ids} == {
                f"joint_{side}{index}" for side in "RL" for index in indices
            }
        for _ in range(60):
            obs, reward, _, _, _ = env.step(torch.zeros((4, 12), device=env.device))
            assert torch.isfinite(reward).all()
            assert all(torch.isfinite(value).all() for value in obs.values())
            spin_cost = twist.func(env, **twist.params)
            assert spin_cost.shape == (4,) and torch.isfinite(spin_cost).all() and spin_cost.ge(0).all()
            yaw_cost = base_yaw.func(env, **base_yaw.params)
            assert yaw_cost.shape == (4,) and torch.isfinite(yaw_cost).all() and yaw_cost.ge(0).all()

        # 缩短更新只验证训练链路；正式任务的网络和镜像配置保持原值。
        agent.device = env.device
        agent.num_steps_per_env = 8
        agent.algorithm.num_learning_epochs = 1
        agent.algorithm.num_mini_batches = 2
        with tempfile.TemporaryDirectory(prefix="rough_body_check_") as run_dir:
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
            runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=run_dir, device=env.device)
            runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)
            check_exports(env, agent, runner, wrapped, run_dir)
        print("PASS nlegs_body_rough W1196: adapted geometry, explicit actuator limits, "
              "load-gated foot spin, weaker hip yaw, small-command base yaw penalty, no lateral penalty, "
              "ankle pitch Kp 200, retained body terrain, "
              "base mass randomization without accumulation, fixed PD, ankle-pitch-only paired zero bias, aligned feet, "
              "4-env/60-step finite rollout, 705/3735 obs, one PPO update, JIT/ONNX parity, "
              "deploy controller metadata; MuJoCo replay not validated", flush=True)
    finally:
        env.close()


try:
    main()
except Exception:
    traceback.print_exc()
    # Kit 关闭可能覆盖 Python 异常退出码，失败时直接返回非零。
    os._exit(1)
else:
    app.close()
