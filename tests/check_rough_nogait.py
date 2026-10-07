"""检查无相位奖励、任务注册、短 PPO 更新及部署导出：--headless --device cuda:0。"""

import argparse
import os
from pathlib import Path
import tempfile
import traceback
from types import SimpleNamespace

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
import yaml
from isaaclab.managers import SceneEntityCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner

import legs_rl_lab.tasks  # noqa: F401
from legs_rl_lab.tasks.nlegs_task import mdp
from legs_rl_lab.utils.export_deploy_cfg import export_deploy_cfg


def check_clearance():
    # 无 gait/episode 时钟的最小环境；覆盖站稳、跳跃、转向与不同地面高度。
    contact_time = torch.tensor([[1., 0.], [1., 1.], [0., 0.], [1., 0.], [1., 0.], [1., 0.]])
    feet = torch.tensor([[[0., 0., 0.0135], [1., 0., 0.1335]]]).repeat(6, 1, 1)
    hits = torch.tensor([[[0., 0., 0.], [1., 0., 0.]]]).repeat(6, 1, 1)
    feet[4, :, 2] += 2.0
    hits[4, :, 2] += 2.0
    hits[5] = float("inf")
    commands = torch.tensor([[0.2, 0., 0.], [0.2, 0., 0.], [0.2, 0., 0.],
                             [0., 0., 0.2], [0.2, 0., 0.], [0.2, 0., 0.]])
    scene = {"robot": SimpleNamespace(data=SimpleNamespace(body_pos_w=feet))}
    scene = type("Scene", (dict,), {})(scene)
    scene.sensors = {
        "contact_forces": SimpleNamespace(data=SimpleNamespace(current_contact_time=contact_time)),
        "height_scanner": SimpleNamespace(data=SimpleNamespace(ray_hits_w=hits)),
    }
    env = SimpleNamespace(scene=scene, command_manager=SimpleNamespace(get_command=lambda _: commands))
    params = dict(asset_cfg=SceneEntityCfg("robot", body_ids=[0, 1]),
                  sensor_cfg=SceneEntityCfg("height_scanner"),
                  contact_sensor_cfg=SceneEntityCfg("contact_forces", body_ids=[0, 1]))
    reward = mdp.feet_clearance_nogait(env, **params)
    torch.testing.assert_close(reward, torch.tensor([1., 0., 0., 1., 1., 0.]))
    commands.zero_()
    assert torch.count_nonzero(mdp.feet_clearance_nogait(env, **params)) == 0
    commands[:, 0] = 0.2
    feet[0, 1, 2] = 0.0235
    low = mdp.feet_clearance_nogait(env, **params)[0]
    feet[0, 1, 2] = 0.3735
    high = mdp.feet_clearance_nogait(env, **params)[0]
    assert low < reward[0] and high < reward[0]


def main():
    check_clearance()
    cfg = load_cfg_from_registry("rough_nogait", "env_cfg_entry_point")
    play = load_cfg_from_registry("rough_nogait", "play_env_cfg_entry_point")
    agent = load_cfg_from_registry("rough_nogait", "rsl_rl_cfg_entry_point")
    original = load_cfg_from_registry("nlegs_rough", "env_cfg_entry_point")
    for config in (cfg, play):
        assert config.gait is None and config.rewards.gait is None
        assert config.observations.policy.gait_phase is None
        assert config.observations.critic.gait_phase is None
    assert original.gait is not None and original.rewards.gait is not None
    assert original.observations.policy.gait_phase is not None
    assert original.scene.terrain.terrain_generator.to_dict() == cfg.scene.terrain.terrain_generator.to_dict()
    assert original.commands.to_dict() == cfg.commands.to_dict()
    assert play.events.randomize_leg_gains is None
    assert agent.experiment_name == "rough_nogait" and not agent.resume
    assert agent.algorithm.num_mini_batches == 16 and agent.algorithm.symmetry_cfg is None

    cfg.scene.num_envs = 16
    cfg.sim.device = args.device
    cfg.scene.sky_light.spawn.texture_file = None
    cfg.scene.terrain.visual_material = None
    cfg.scene.terrain.terrain_generator.num_rows = 2
    cfg.scene.terrain.terrain_generator.num_cols = 10
    cfg.scene.terrain.max_init_terrain_level = 1
    cfg.commands.base_velocity.debug_vis = False
    env = gym.make("rough_nogait", cfg=cfg).unwrapped
    try:
        obs, _ = env.reset()
        for name, sensor_key in (("feet_clearance", "contact_sensor_cfg"), ("feet_slide", "sensor_cfg")):
            params = env.reward_manager.get_term_cfg(name).params
            asset_names = [env.scene["robot"].body_names[i] for i in params["asset_cfg"].body_ids]
            sensor_names = [env.scene.sensors["contact_forces"].body_names[i] for i in params[sensor_key].body_ids]
            assert asset_names == sensor_names == ["Link_R6", "Link_L6"]
        assert obs["policy"].shape == (16, 450) and obs["critic"].shape == (16, 2470)
        for group in ("policy", "critic"):
            assert "gait_phase" not in env.observation_manager.active_terms[group]
        assert "gait" not in env.reward_manager.active_terms
        for _ in range(60):
            obs, reward, _, _, _ = env.step(torch.zeros((16, 12), device=env.device))
            assert torch.isfinite(reward).all()
            assert all(torch.isfinite(value).all() for value in obs.values())
        with tempfile.TemporaryDirectory(prefix="rough_nogait_check_") as run_dir:
            export_deploy_cfg(env, run_dir, policy_action_clip=agent.clip_actions)
            deploy = yaml.safe_load((Path(run_dir) / "params/deploy.yaml").read_text())
            assert "gait_period" not in deploy and "gait_phase" not in deploy["observations"]
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
            runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=run_dir, device=env.device)
            runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)
            model = torch.jit.script(runner.alg.actor.as_jit()).cpu()
            result = model(torch.zeros((1, 450)))
            assert result.shape == (1, 12) and torch.isfinite(result).all()
        print("PASS rough_nogait: reward boundaries, aligned feet, 16-env/60-step rollout, 450/2470 obs, 2 PPO iterations, JIT/deploy export", flush=True)
    finally:
        env.close()


try:
    main()
except Exception:
    traceback.print_exc()
    # Kit 的关闭可能覆盖 Python 异常退出码，失败时直接返回非零。
    os._exit(1)
else:
    app.close()
