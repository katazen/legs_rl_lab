"""有界 IsaacLab 冒烟测试：联合 reset、USD 几何/限位、部分 reset 隔离和首帧目标。

python scripts/check_crouch_reset.py --headless --device cuda:0
仅仿真，不加载策略、不访问机器人。另用 generate_crouch_pose_bank.py --check 复核整表。
"""

import argparse
import os
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply
from legs_rl_lab.assets.nlegs import nlegs
from legs_rl_lab.tasks.nlegs_task.task.flat.flat_env_cfg import FlatEnvCfg
from legs_rl_lab.tasks.nlegs_task.task.flat.crouch_env_cfg import (
    FlatCrouchEnvCfg, FlatCrouchPlayEnvCfg, NlegsFlatCrouchPPORunnerCfg,
)
from legs_rl_lab.tasks.nlegs_task.agents.rsl_rl_ppo_cfg import NlegsFlatPPORunnerCfg
from legs_rl_lab.tasks.nlegs_task.task.rough.rough_env_cfg import RoughEnvCfg
from legs_rl_lab.tasks.nlegs_task.task.rough.rough_info_env_cfg import RoughInfoEnvCfg
from legs_rl_lab.tasks.nlegs_task.task.rough.rough_step_env_cfg import RoughStepEnvCfg


def main():
    asset_dir = Path(nlegs.__file__).parent
    original_usd = str(asset_dir / "mjcf/nlegs/nlegs.usd")
    crouch_usd = str(asset_dir / "mjcf/nlegs_limit/nlegs_limit.usd")
    base_cfg = FlatEnvCfg()
    cfg = FlatCrouchEnvCfg()
    assert cfg.scene.robot.spawn.usd_path == crouch_usd
    assert FlatCrouchPlayEnvCfg().scene.robot.spawn.usd_path == crouch_usd
    assert base_cfg.scene.robot.spawn.usd_path == original_usd
    assert nlegs.NLEGS_CFG.spawn.usd_path == original_usd
    for env_cfg in (FlatEnvCfg, RoughEnvCfg, RoughInfoEnvCfg, RoughStepEnvCfg):
        assert env_cfg().scene.robot.spawn.usd_path == original_usd
    assert FlatEnvCfg().events.reset_base is not None
    assert NlegsFlatCrouchPPORunnerCfg().experiment_name == "nlegs_flat_crouch"
    assert NlegsFlatPPORunnerCfg().algorithm.symmetry_cfg is not None
    assert FlatCrouchPlayEnvCfg().events.reset_crouch.params == cfg.events.reset_crouch.params
    assert set(cfg.events.reset_crouch.params) == {"bank_path", "stand_probability", "crouch_probability"}
    assert cfg.commands.base_velocity.to_dict() == base_cfg.commands.base_velocity.to_dict()
    assert cfg.curriculum is None
    cfg.scene.num_envs = 16
    cfg.sim.device = args.device
    # 不依赖在线 HDR/材质下载；不改变几何和物理。
    cfg.scene.sky_light.spawn.texture_file = None
    cfg.scene.terrain.visual_material = None
    cfg.commands.base_velocity.debug_vis = False
    env = ManagerBasedRLEnv(cfg)
    try:
        robot = env.scene["robot"]
        default_q = robot.data.default_joint_pos.clone()
        term_cfg = env.event_manager.get_term_cfg("reset_crouch")
        term = term_cfg.func
        feet, _ = robot.find_bodies(["Link_L6", "Link_R6"], preserve_order=True)
        saved_limits = robot.data.joint_pos_limits.clone()
        try:
            robot.data.joint_pos_limits[0, 0, 1] += 0.1
            with patch.object(robot, "write_joint_position_limit_to_sim",
                              side_effect=AssertionError("reset 不应覆盖 USD 限位")):
                try:
                    type(term)(term_cfg, env)
                except ValueError as exc:
                    assert "关节限位与姿态表不一致" in str(exc)
                else:
                    raise AssertionError("USD 与姿态表限位不一致未被拒绝")
        finally:
            robot.data.joint_pos_limits.copy_(saved_limits)

        def sampled_groups():
            error, indices = (robot.data.joint_pos[:, None] - term.q).abs().amax(dim=-1).min(dim=-1)
            assert error.max() < 1e-5
            return term.groups[indices]

        def check_pose():
            assert torch.isfinite(robot.data.root_state_w).all()
            assert torch.equal(robot.data.default_joint_pos, default_q)
            assert robot.data.joint_vel.abs().max() < 1e-6
            assert robot.data.root_vel_w.abs().max() < 1e-6
            lim = robot.data.joint_pos_limits
            assert ((robot.data.joint_pos >= lim[..., 0]) & (robot.data.joint_pos <= lim[..., 1])).all()
            # 同时验证 table -> 名字重排 -> 实际 USD 运动学，而不只验证写入的张量。
            foot_q = robot.data.body_quat_w[:, feet]
            vertical = torch.zeros((*foot_q.shape[:-1], 3), device=env.device)
            vertical[..., 2] = 1
            normals = quat_apply(foot_q, vertical)
            assert normals[..., :2].abs().max() < 0.009, normals
            z = robot.data.body_pos_w[:, feet, 2] - env.scene.env_origins[:, 2, None]
            assert (z - 0.0155).abs().max() < 0.001, z
            assert not env.termination_manager.compute().any()

        obs, _ = env.reset(seed=42)
        assert (sampled_groups() == 2).any()
        check_pose()
        # 相同随机种子下，训练步数不能改变采样结果。
        torch.manual_seed(42)
        term(env, None, **term_cfg.params)
        first_q = robot.data.joint_pos.clone()
        env.common_step_counter = 1_000_000
        torch.manual_seed(42)
        term(env, None, **term_cfg.params)
        assert torch.equal(robot.data.joint_pos, first_q)
        env.common_step_counter = 0

        term_cfg.params.update(stand_probability=0.0, crouch_probability=0.0)
        env.event_manager.set_term_cfg("reset_crouch", term_cfg)
        env.reset()
        assert (sampled_groups() == 1).all()
        check_pose()

        term_cfg.params.update(stand_probability=0.0, crouch_probability=1.0)
        env.event_manager.set_term_cfg("reset_crouch", term_cfg)
        obs, _ = env.reset()
        assert (sampled_groups() == 2).all()
        # 下蹲出生立即获得行走指令，不能退化成所有环境零命令的起身任务。
        assert env.command_manager.get_command("base_velocity").abs().max() > 0.1
        check_pose()
        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(obs["critic"]).all()

        # 仅将 0、3 号重置为站立，其他环境继续保持下蹲。
        untouched = torch.tensor([1, 2, *range(4, env.num_envs)], device=env.device)
        before_q = robot.data.joint_pos[untouched].clone()
        before_root = robot.data.root_link_state_w[untouched].clone()
        env._prev_prev_action = torch.ones_like(default_q)
        term_cfg.params.update(stand_probability=1.0, crouch_probability=0.0)
        env.event_manager.set_term_cfg("reset_crouch", term_cfg)
        env.reset(env_ids=torch.tensor([0, 3], device=env.device))
        assert torch.equal(robot.data.joint_pos[untouched], before_q)
        assert torch.allclose(robot.data.root_link_state_w[untouched], before_root, atol=1e-6)
        assert (sampled_groups()[[0, 3]] == 0).all()
        assert (sampled_groups()[untouched] == 2).all()
        assert env._prev_prev_action[[0, 3]].abs().max() == 0
        assert (env._prev_prev_action[untouched] == 1).all()
        check_pose()
        for stand, crouch in ((-1.0, 0.0), (0.5, 0.6)):
            try:
                term(env, [0], **{**term_cfg.params, "stand_probability": stand, "crouch_probability": crouch})
            except ValueError:
                pass
            else:
                raise AssertionError("无效采样概率未被拒绝")

        # 零动作不是保持下蹲；显式用现有动作定义计算 hold，用于有界数值冒烟测试。
        hold = (robot.data.joint_pos - default_q) / 0.25
        assert hold.abs().max() < 5.0
        for _ in range(10):
            obs, reward, _, _, _ = env.step(hold)
            assert torch.isfinite(reward).all() and torch.isfinite(obs["policy"]).all()
        print("PASS: crouch-only USD override, unchanged flat/rough assets, USD limit mismatch rejection, "
              "fixed stand/intermediate/limit mixture, no curriculum, walking commands, partial-reset isolation, USD flat feet, hard limits, "
              "unchanged action offset, finite 10-step simulation", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # Kit fast shutdown 会吞掉异常并返回 0；失败时显式退出当前测试进程。
        os._exit(1)
    app.close()
