#!/usr/bin/env python3
"""Record one deterministic Isaac walk with exact run actuators and foot contacts."""
from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", required=True)
parser.add_argument("--out-dir", required=True)
parser.add_argument("--warmup", type=float, default=4.0)
parser.add_argument("--duration", type=float, default=8.0)
parser.add_argument("--command", default="0.3,0,0", help="vx,vy,yaw")
parser.add_argument("--ankle-delay", type=int, default=10, help="5 ms steps; run range is 4..16")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import csv
import json
import sys

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
import legs_rl_lab.tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from legs_rl_lab.utils.parser_cfg import parse_env_cfg

sys.path.append(os.path.dirname(__file__))
from run_actuator_cfg import configure_from_deploy, load_deploy


def main():
    deploy_path = os.path.join(args.run_dir, "params", "deploy.yaml")
    policy_path = os.path.join(args.run_dir, "exported", "policy.pt")
    deploy = load_deploy(deploy_path)
    command = [float(x) for x in args.command.split(",")]
    if len(command) != 3:
        raise ValueError("--command must contain vx,vy,yaw")
    os.makedirs(args.out_dir, exist_ok=True)

    env_cfg = parse_env_cfg("nlegs", device=args.device, num_envs=1,
                            use_fabric=True, entry_point_key="play_env_cfg_entry_point")
    env_cfg.scene.robot = configure_from_deploy(
        env_cfg.scene.robot, deploy,
        fixed_delays={"4310": 5, "4340": 5, "ankle": args.ankle_delay},
    )
    env_cfg.seed = 42
    env_cfg.episode_length_s = args.warmup + args.duration + 5.0
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    for ranges in (env_cfg.commands.base_velocity.ranges, env_cfg.commands.base_velocity.limit_ranges):
        ranges.lin_vel_x = (command[0], command[0])
        ranges.lin_vel_y = (command[1], command[1])
        ranges.ang_vel_z = (command[2], command[2])
    env_cfg.commands.base_velocity.debug_vis = False
    for name in ("physics_material", "add_base_mass", "reset_base", "reset_robot_joints",
                 "joint_zero_bias", "push_robot"):
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)
    for name in ("lin_vel_cmd_levels", "ang_vel_cmd_levels"):
        if hasattr(env_cfg.curriculum, name):
            setattr(env_cfg.curriculum, name, None)

    env = RslRlVecEnvWrapper(gym.make("nlegs", cfg=env_cfg), clip_actions=None)
    raw = env.unwrapped
    robot = raw.scene["robot"]
    contact = raw.scene.sensors["contact_forces"]
    foot_ids = {side: contact.find_bodies(f".*{side}6")[0][0] for side in ("R", "L")}
    joint_ids = [robot.find_joints(name)[0][0] for name in deploy["joint_names"]]
    action_term = raw.action_manager.get_term(raw.action_manager.active_terms[0])
    action_cols = [action_term._joint_names.index(name) for name in deploy["joint_names"]]
    policy = torch.jit.load(policy_path, map_location=raw.device).eval()
    obs, _ = env.reset()

    rows = []
    total = int(round((args.warmup + args.duration) / raw.step_dt))
    warmup_steps = int(round(args.warmup / raw.step_dt))
    for step in range(total):
        with torch.inference_mode():
            actions = policy(obs["policy"])
            obs, _, done, _ = env.step(actions)
        if bool(done[0]):
            raise RuntimeError(f"environment reset/fell at t={(step + 1) * raw.step_dt:.3f}s")
        if step < warmup_steps:
            continue
        target = action_term.processed_actions[0, action_cols].detach().cpu().numpy()
        q = robot.data.joint_pos[0, joint_ids].detach().cpu().numpy()
        v = robot.data.joint_vel[0, joint_ids].detach().cpu().numpy()
        tau = robot.data.applied_torque[0, joint_ids].detach().cpu().numpy()
        forces = {side: float(torch.linalg.vector_norm(contact.data.net_forces_w[0, idx]).item())
                  for side, idx in foot_ids.items()}
        root = robot.data.root_pos_w[0].detach().cpu().numpy()
        rows.append(((step - warmup_steps + 1) * raw.step_dt, target, q, v, tau,
                     forces["R"], forces["L"], root))

    walk_path = os.path.join(args.out_dir, "isaac_walk.csv")
    with open(walk_path, "w", newline="") as f:
        w = csv.writer(f)
        names = [n.removeprefix("joint_") for n in deploy["joint_names"]]
        w.writerow(["t", *(f"target_{n}" for n in names), *(f"q_{n}" for n in names),
                    *(f"v_{n}" for n in names), *(f"tau_{n}" for n in names),
                    "contact_R_N", "contact_L_N", "base_x", "base_y", "base_z"])
        for t, target, q, v, tau, fr, fl, root in rows:
            w.writerow([f"{t:.6f}", *target, *q, *v, *tau, fr, fl, *root])

    # fixed_base_replay expects real order L1..L6,R1..R6 and one moving joint per file.
    sdk_index = {name: i for i, name in enumerate(deploy["joint_names"])}
    for joint, name, suffix in ((4, "joint_L5", "left"), (10, "joint_R5", "right")):
        path = os.path.join(args.out_dir, f"j{joint}_kp40_kd2_replay_isaacwalk_{suffix}_cmd.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["t", "phase", *(f"qd{i}" for i in range(12))])
            for i, (_, target, *_rest) in enumerate(rows):
                qd = np.zeros(12); qd[joint] = target[sdk_index[name]]
                w.writerow([f"{i * raw.step_dt:.6f}", "excite", *qd])

    metadata = {
        "run_dir": os.path.abspath(args.run_dir), "deploy_yaml": os.path.abspath(deploy_path),
        "policy": os.path.abspath(policy_path), "command": command,
        "physics_dt": raw.physics_dt, "step_dt": raw.step_dt,
        "warmup_s": args.warmup, "record_s": args.duration,
        "fixed_delay_steps": {"4310": 5, "4340": 5, "ankle": args.ankle_delay},
        "observation_noise": False, "domain_randomization": False,
        "joint_order": deploy["joint_names"], "foot_contact_columns": foot_ids,
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"WALK_RECORD_DONE rows={len(rows)} q={walk_path} contacts={foot_ids}")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
