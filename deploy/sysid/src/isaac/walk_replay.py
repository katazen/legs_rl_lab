# -*- coding: utf-8 -*-
"""走路工况全 12 关节回放(吊起来)。

把实机 sim2real 记录里的 12 维目标命令(cmd0..11, 实机序 L1..L6,R1..R6)喂给
吊起来的机器人, 用 LEGS_CFG 的【原配置 actuator】(=已辨识的 DelayedPD + DelayedDCMotor
混合, 各关节 kp/kd/armature/vlim/sat/delay 都取自 legs.py), 记录仿真跟踪, 与实机对比。

与 replay.py 的区别: replay.py 只能给所有关节套单一 actuator 类型; 这里直接用 LEGS_CFG,
如实复现 J1/J4=DelayedDCMotor、其余=DelayedPD 的混合配置。

用法(unitree_lab + setup_conda_env.sh):
  python walk_replay.py --csv <sim2real.csv> --out <SIM_out.csv> --headless
"""
from __future__ import annotations
import argparse
import os
from isaaclab.app import AppLauncher

SIM_DT, DECIMATION = 0.005, 4
ASSET_SYSPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "source", "legs_rl_lab"))
ASSET_MODULE = "legs_rl_lab.assets.legs_URDF.legs"
ASSET_CFG = "LEGS_CFG"

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True, help="实机 sim2real 记录(含 t, cmd0..11 目标, q0..11 实测)")
parser.add_argument("--out", required=True, help="输出仿真跟踪 csv")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import sys, csv, copy, importlib
import numpy as np
import torch
sys.path.append(ASSET_SYSPATH)
ROBOT_CFG = getattr(importlib.import_module(ASSET_MODULE), ASSET_CFG)
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.assets import Articulation


def sim_joint(j):
    """实机 idx → USD 关节名: 0-5=joint_L1..L6, 6-11=joint_R1..R6。"""
    return f"joint_L{j + 1}" if j < 6 else f"joint_R{j - 5}"


def main():
    rows = list(csv.DictReader(open(args.csv)))
    t = np.array([float(r["t"]) for r in rows], dtype=np.float64)
    t -= t[0]
    cmd = np.array([[float(r[f"cmd{i}"]) for i in range(12)] for r in rows], dtype=np.float64)  # [n,12] 实机序

    sim = SimulationContext(SimulationCfg(dt=SIM_DT, device="cpu"))
    cfg = copy.deepcopy(ROBOT_CFG)
    cfg.prim_path = "/World/Robot"
    cfg.init_state.joint_pos = {".*": 0.0}
    robot = Articulation(cfg)         # 不覆盖 actuators -> 用 legs.py 原配置(混合 DelayedPD/DelayedDCMotor)
    sim.reset()
    for name, a in robot.actuators.items():
        print(f"[actuator] {name}: {type(a).__name__}  joints={a.joint_names}")

    real2col = [robot.find_joints(sim_joint(i))[0][0] for i in range(12)]
    root_pose = torch.tensor([[0.0, 0.0, 0.65, 1.0, 0.0, 0.0, 0.0]]).repeat(robot.num_instances, 1)
    root_vel = torch.zeros((robot.num_instances, 6))
    q_des = torch.zeros((robot.num_instances, robot.num_joints))
    v_des = torch.zeros_like(q_des)

    # 初始 = 首帧命令(避免从 0 起步瞬态)
    q_init = torch.zeros_like(q_des)
    for i in range(12):
        q_init[:, real2col[i]] = float(cmd[0, i])
    robot.write_joint_state_to_sim(q_init, torch.zeros_like(q_des))
    robot.reset()
    q_des.copy_(q_init)

    out_rows = []
    n = int(t[-1] / SIM_DT)
    for k in range(n + 1):
        tt = k * SIM_DT
        if k % DECIMATION == 0:                       # 50Hz 更新目标(与部署 step_dt 一致)
            for i in range(12):
                q_des[:, real2col[i]] = float(np.interp(tt, t, cmd[:, i]))
        robot.set_joint_position_target(q_des)
        robot.set_joint_velocity_target(v_des)
        robot.write_root_pose_to_sim(root_pose)       # 吊住: 每步锁 base 位姿
        robot.write_root_velocity_to_sim(root_vel)
        robot.write_data_to_sim()
        sim.step()
        robot.update(SIM_DT)
        q = robot.data.joint_pos[0].cpu().numpy()
        v = robot.data.joint_vel[0].cpu().numpy()
        tau = robot.data.applied_torque[0].cpu().numpy()
        out_rows.append((tt,
                         [q[real2col[i]] for i in range(12)],
                         [v[real2col[i]] for i in range(12)],
                         [tau[real2col[i]] for i in range(12)]))
        if not app.is_running():
            break

    with open(args.out, "w") as f:
        cols = [f"q{i}" for i in range(12)] + [f"v{i}" for i in range(12)] + [f"tau{i}" for i in range(12)]
        f.write("t," + ",".join(cols) + "\n")
        for tt, q, v, tau in out_rows:
            f.write(f"{tt:.6f}," + ",".join(f"{x:.6f}" for x in (q + v + tau)) + "\n")
    print(f"WALK_REPLAY_DONE saved {args.out} ({len(out_rows)} steps)")


if __name__ == "__main__":
    main()
    app.close()
