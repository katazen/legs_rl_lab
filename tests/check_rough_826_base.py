"""核对 rough 的 8.26 基准和当前域随机化；短仿真检查，不训练、不转换资产。"""

import argparse
import copy
import os
import subprocess
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import yaml
import gymnasium as gym
import torch
from pxr import Usd
import legs_rl_lab.tasks
from isaaclab_tasks.utils import load_cfg_from_registry
from legs_rl_lab.tasks.nlegs_task import mdp

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "logs/rsl_rl/nlegs_rough/2026-08-26_20-37-55/params"
ASSETS = ROOT / "source/legs_rl_lab/legs_rl_lab/assets"


def differences(actual, expected, path=""):
    if isinstance(expected, dict) and isinstance(actual, dict):
        result = [f"{path}.{key}: 字段增减" for key in actual.keys() ^ expected.keys()]
        for key in actual.keys() & expected.keys():
            result.extend(differences(actual[key], expected[key], f"{path}.{key}"))
        return result
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(actual) != len(expected):
            return [f"{path}: 长度 {len(actual)} != {len(expected)}"]
        return [error for i, (a, b) in enumerate(zip(actual, expected))
                for error in differences(a, b, f"{path}[{i}]")]
    if isinstance(expected, slice) and isinstance(actual, slice):
        actual, expected = (actual.start, actual.stop, actual.step), (expected.start, expected.stop, expected.step)
    return [] if actual == expected else [f"{path}: {actual!r} != {expected!r}"]


def check_usd():
    path = ASSETS / "nlegs/mjcf/nlegs/nlegs.usd"
    stage = Usd.Stage.Open(str(path))
    assert stage is not None, f"无法加载 {path}"
    joint_limits = [(prim.GetName(), prim.GetAttribute("physics:lowerLimit").Get(),
                     prim.GetAttribute("physics:upperLimit").Get())
                    for prim in stage.Traverse() if prim.HasAttribute("physics:lowerLimit")]
    assert len(joint_limits) == 12, f"原始 USD 关节数量异常：{len(joint_limits)}"
    # 8.26 的资产首次入库版本包含主 USD 和四个配套层。
    relative = str(path.parent.relative_to(ROOT))
    files = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", "7cb681d", relative], cwd=ROOT, text=True,
    ).splitlines()
    usd_files = [name for name in files if name.endswith(".usd")]
    assert len(usd_files) == 5
    for name in usd_files:
        archived = subprocess.check_output(["git", "show", f"7cb681d:{name}"], cwd=ROOT)
        assert (ROOT / name).read_bytes() == archived, f"USD 与 8.26 入库版本不同：{name}"
    print("PASS USD：主文件及四个配套层与 8.26 仓库版本字节一致，12 个关节可读取。", flush=True)
    print(f"USD limits (degrees): {joint_limits}", flush=True)


def check_config():
    saved = yaml.unsafe_load((RUN / "env.yaml").read_text())
    agent_saved = yaml.safe_load((RUN / "agent.yaml").read_text())
    cfg = load_cfg_from_registry("nlegs_rough", "env_cfg_entry_point")
    agent = load_cfg_from_registry("nlegs_rough", "rsl_rl_cfg_entry_point")
    cfg.seed = agent.seed
    expected = copy.deepcopy(saved)
    expected["scene"]["robot"]["spawn"]["usd_path"] = str(ASSETS / "nlegs/mjcf/nlegs/nlegs.usd")
    expected["actions"]["JointPositionAction"]["class_type"] = (
        "legs_rl_lab.tasks.nlegs_task.mdp.actions:ZeroBiasJointPositionAction"
    )
    expected["events"]["add_base_mass"]["params"]["mass_distribution_params"] = (5.0, 8.0)
    expected["events"]["randomize_ankle_gains"]["params"]["asset_cfg"]["joint_names"] = [".*[2356]"]
    expected["events"]["link_mass"] = copy.deepcopy(expected["events"]["add_base_mass"])
    expected["events"]["link_mass"]["params"]["asset_cfg"]["body_names"] = "Link_[LR][1-6]"
    expected["events"]["link_mass"]["params"]["mass_distribution_params"] = (0.9, 1.1)
    expected["events"]["link_mass"]["params"]["operation"] = "scale"
    actual = cfg.to_dict()
    # env.yaml 在场景创建后保存；这里只做框架同样的路径展开和地形字段继承，不加载 USD。
    for name in ("robot", "height_scanner", "contact_forces"):
        actual["scene"][name]["prim_path"] = actual["scene"][name]["prim_path"].format(
            ENV_REGEX_NS="/World/envs/env_.*")
    terrain = actual["scene"]["terrain"]
    terrain["num_envs"] = actual["scene"]["num_envs"]
    terrain["env_spacing"] = actual["scene"]["env_spacing"]
    generator = terrain["terrain_generator"]
    for sub in generator["sub_terrains"].values():
        sub["size"] = generator["size"]
        if "slope_threshold" in sub:
            sub["slope_threshold"] = generator["slope_threshold"]
    errors = [error for key in ("scene", "sim", "actions", "observations", "commands", "rewards",
                               "terminations", "events", "curriculum", "gait", "decimation",
                               "episode_length_s", "seed")
              for error in differences(actual[key], expected[key], key)]
    errors.extend(differences(agent.to_dict(), agent_saved, "agent"))
    assert not errors, "\n".join(errors)
    from legs_rl_lab.assets.nlegs.nlegs import NLEGS_CFG, NLEGS_PD_CFG
    assert cfg.scene.robot is not NLEGS_CFG and cfg.scene.robot is not NLEGS_PD_CFG
    assert NLEGS_CFG.actuators["legs"].stiffness[".*2"] == 200.0
    assert NLEGS_CFG.actuators["legs"].min_delay == 1
    print("PASS 配置：仅四项域随机化偏离 8.26；名义执行器、PPO、USD 与其他配置一致。", flush=True)


def check_randomization():
    cfg = load_cfg_from_registry("nlegs_rough", "env_cfg_entry_point")
    cfg.scene.num_envs = 4
    cfg.sim.device = args.device
    cfg.scene.sky_light.spawn.texture_file = None
    cfg.scene.terrain.visual_material = None
    cfg.scene.terrain.terrain_generator.num_rows = 2
    cfg.scene.terrain.terrain_generator.num_cols = 10
    cfg.scene.terrain.max_init_terrain_level = 1
    cfg.commands.base_velocity.debug_vis = False
    env = gym.make("nlegs_rough", cfg=cfg).unwrapped
    try:
        robot = env.scene["robot"]
        base = robot.body_names.index("base")
        links = env.event_manager.get_term_cfg("link_mass").params["asset_cfg"].body_ids
        assert len(links) == 12 and base not in links
        selected = env.event_manager.get_term_cfg("randomize_ankle_gains").params["asset_cfg"].joint_ids
        assert {robot.joint_names[i] for i in selected} == {
            f"joint_{side}{i}" for side in "LR" for i in (2, 3, 5, 6)
        }
        # 多次 reset 必须相对原始质量和增益抽样，不能逐次叠加。
        for _ in range(4):
            env.reset()
            masses = robot.root_physx_view.get_masses()
            base_delta = masses[:, base] - robot.data.default_mass[:, base].cpu()
            assert base_delta.ge(5.0 - 1e-5).all() and base_delta.le(8.0 + 1e-5).all()
            link_ratio = masses[:, links] / robot.data.default_mass[:, links].cpu()
            assert link_ratio.ge(0.9 - 1e-5).all() and link_ratio.le(1.1 + 1e-5).all()
            for actuator in robot.actuators.values():
                ids = list(actuator.joint_indices)
                for gains, nominal, bounds in (
                    (actuator.stiffness, robot.data.default_joint_stiffness[:, ids], (0.7, 1.1)),
                    (actuator.damping, robot.data.default_joint_damping[:, ids], (0.9, 1.5)),
                ):
                    ratio = gains / nominal
                    for column, joint in enumerate(ids):
                        if joint in selected:
                            assert ratio[:, column].ge(bounds[0] - 1e-5).all()
                            assert ratio[:, column].le(bounds[1] + 1e-5).all()
                        else:
                            torch.testing.assert_close(ratio[:, column], torch.ones_like(ratio[:, column]))
        action = env.action_manager.get_term("JointPositionAction")
        action.process_actions(torch.zeros((4, 12), device=env.device))
        for value in (-0.05, 0.0, 0.05):
            env._joint_zero_bias.fill_(value)
            physical = robot.data.joint_pos - robot.data.default_joint_pos
            torch.testing.assert_close(mdp.joint_pos_rel_biased(env), physical + env._joint_zero_bias)
            target = action.processed_actions.clone()
            for _ in range(3):
                action.apply_actions()
                torch.testing.assert_close(robot.data.joint_pos_target, target - env._joint_zero_bias)
                torch.testing.assert_close(action.processed_actions, target)
        for _ in range(10):
            obs, rewards, _, _, _ = env.step(torch.zeros((4, 12), device=env.device))
            assert torch.isfinite(rewards).all()
            assert all(torch.isfinite(value).all() for value in obs.values())
        print("PASS 仿真：4环境重复 reset，base/12连杆质量与指定关节PD范围正确；"
              "零偏成对补偿，10步观测与奖励有限。", flush=True)
    finally:
        env.close()


try:
    check_usd()
    check_config()
    check_randomization()
except Exception:
    traceback.print_exc()
    os._exit(1)
else:
    app.close()
