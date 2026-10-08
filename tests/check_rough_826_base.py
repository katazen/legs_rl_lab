"""配置与 XML 检查；不转换 USD、不启动训练。用 unitree_lab Python 加 --headless 运行。"""

import argparse
import copy
import traceback
from pathlib import Path
import xml.etree.ElementTree as ET

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import mujoco
import numpy as np
import yaml
import legs_rl_lab.tasks
from isaaclab_tasks.utils import load_cfg_from_registry

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


def structure(element):
    return element.tag, dict(element.attrib), [structure(child) for child in element]


def hip_midpoint(model, data):
    return (data.xanchor[model.joint("joint_L1").id]
            + data.xanchor[model.joint("joint_R1").id]) / 2


def check_xml():
    old_path = ASSETS / "nlegs/mjcf/nlegs.xml"
    new_path = ASSETS / "nlegs/mjcf/nlegs_limit.xml"
    body_path = ASSETS / "nlegs_body/mjcf/nlegs_body.xml"
    old, new = ET.parse(old_path).getroot(), ET.parse(new_path).getroot()
    new_base = new.find("worldbody/body")
    new_base.remove(new_base.find("inertial"))
    new_base.insert(2, copy.deepcopy(old.find("worldbody/body/inertial")))
    assert structure(new) == structure(old), "相对旧 nlegs.xml，base 惯性以外还有差异"
    models = [mujoco.MjModel.from_xml_path(str(path)) for path in (old_path, new_path, body_path)]
    data = [mujoco.MjData(model) for model in models]
    for model, state in zip(models, data):
        mujoco.mj_forward(model, state)
    _, hybrid, body = models
    h, b = hybrid.body("base").id, body.body("base").id
    np.testing.assert_allclose(hybrid.body_mass[h], body.body_mass[b], atol=1e-10)
    np.testing.assert_allclose(hybrid.body_inertia[h], body.body_inertia[b], atol=1e-10)
    np.testing.assert_allclose(hybrid.body_iquat[h], body.body_iquat[b], atol=1e-10)
    np.testing.assert_allclose(data[1].xipos[h] - hip_midpoint(hybrid, data[1]),
                               data[2].xipos[b] - hip_midpoint(body, data[2]), atol=1e-10)
    print(f"PASS XML：仅 base 惯性改变；质量 {hybrid.body_mass[h]:.5f} kg；"
          f"整机 {hybrid.body_mass.sum():.7f} kg", flush=True)


def check_config():
    saved = yaml.unsafe_load((RUN / "env.yaml").read_text())
    agent_saved = yaml.safe_load((RUN / "agent.yaml").read_text())
    cfg = load_cfg_from_registry("nlegs_rough", "env_cfg_entry_point")
    agent = load_cfg_from_registry("nlegs_rough", "rsl_rl_cfg_entry_point")
    cfg.seed = agent.seed
    expected = copy.deepcopy(saved)
    expected["scene"]["robot"]["spawn"]["usd_path"] = str(ASSETS / "nlegs/mjcf/nlegs_limit/nlegs_limit.usd")
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
    print("PASS 8.26 配置：场景、执行器、观测、奖励、事件、命令、终止、课程及 PPO 一致；"
          "只更换 USD 路径。未加载或转换 USD。", flush=True)


try:
    check_xml()
    check_config()
except Exception:
    traceback.print_exc()
    raise
finally:
    app.close()
