# -*- coding: utf-8 -*-
"""走路数据辨识 J2/J5 的 velocity_limit: 把目标关节配成 DelayedDCMotor, 一个 Isaac 会话内
运行时扫 vlim, 喂实机走路目标命令(全 12 维), 每个 vlim 存一份 sim state。
其余关节保持 legs.py 原配置(膝/髋pitch=DCMotor, 髋yaw/踝roll=PD)。

用法(unitree_lab + setup_conda_env):
  python walk_replay_sweep.py --csv <sim2real.csv> --out_dir <dir> --joints 2,5 --vlims 1.5,2,2.5,3,3.5,4 --headless
"""
from __future__ import annotations
import argparse, os
from isaaclab.app import AppLauncher

SIM_DT, DECIMATION = 0.005, 4
ASSET_SYSPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "source", "legs_rl_lab"))
ASSET_MODULE = "legs_rl_lab.assets.legs_URDF.legs"

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True)
parser.add_argument("--out_dir", required=True)
parser.add_argument("--joints", default="2,5", help="要辨识的关节号(1-6), 逗号分隔")
parser.add_argument("--vlims", default="1.5,2,2.5,3,3.5,4")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import sys, csv, copy, importlib
import numpy as np
import torch
sys.path.append(ASSET_SYSPATH)
LEGS_CFG = getattr(importlib.import_module(ASSET_MODULE), "LEGS_CFG")
from legs_rl_lab.actuators import DelayedDCMotorCfg
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.assets import Articulation


def sim_joint(j):  # 实机 idx -> USD 名
    return f"joint_L{j + 1}" if j < 6 else f"joint_R{j - 5}"


def main():
    os.makedirs(args.out_dir, exist_ok=True)
    jnums = [int(x) for x in args.joints.split(",")]        # 1-6
    vlims = [float(x) for x in args.vlims.split(",")]
    rows = list(csv.DictReader(open(args.csv)))
    t = np.array([float(r["t"]) for r in rows]); t -= t[0]
    cmd = np.array([[float(r[f"cmd{i}"]) for i in range(12)] for r in rows])

    sim = SimulationContext(SimulationCfg(dt=SIM_DT, device="cpu"))
    cfg = copy.deepcopy(LEGS_CFG); cfg.prim_path = "/World/Robot"
    cfg.init_state.joint_pos = {".*": 0.0}
    pd = cfg.actuators["delayed_pd"]
    test_groups = {}
    for jn in jnums:                                        # 把目标关节从 PD 组移出, 单开 DCMotor 组
        pat = f".*{jn}"
        kp = pd.stiffness[pat]; kd = pd.damping[pat]; arm = pd.armature[pat]
        pd.joint_names_expr = [e for e in pd.joint_names_expr if e != pat]
        for d in (pd.stiffness, pd.damping, pd.armature, pd.effort_limit_sim):
            d.pop(pat, None)
        gname = f"dc_j{jn}"
        cfg.actuators[gname] = DelayedDCMotorCfg(
            joint_names_expr=[pat], effort_limit=26.0, saturation_effort=26.0,
            velocity_limit=vlims[0], stiffness={pat: kp}, damping={pat: kd},
            armature={pat: arm}, min_delay=4, max_delay=6)
        test_groups[jn] = gname
        print(f"[cfg] J{jn} -> DelayedDCMotor kp{kp} kd{kd} arm{arm}")

    robot = Articulation(cfg); sim.reset()
    real2col = [robot.find_joints(sim_joint(i))[0][0] for i in range(12)]
    root_pose = torch.tensor([[0.,0.,0.65,1.,0.,0.,0.]]).repeat(robot.num_instances,1)
    root_vel = torch.zeros((robot.num_instances,6))
    q_des = torch.zeros((robot.num_instances, robot.num_joints)); v_des = torch.zeros_like(q_des)
    q_init = torch.zeros_like(q_des)
    for i in range(12): q_init[:, real2col[i]] = float(cmd[0,i])

    for vlim in vlims:
        robot.write_joint_state_to_sim(q_init, torch.zeros_like(q_des)); robot.reset()
        for jn, g in test_groups.items():
            act = robot.actuators[g]
            act.velocity_limit[:] = vlim
            act._vel_at_effort_lim = act.velocity_limit * (1.0 + act.effort_limit / act._saturation_effort)
        q_des.copy_(q_init)
        out_rows = []; n = int(t[-1]/SIM_DT)
        for k in range(n+1):
            tt = k*SIM_DT
            if k % DECIMATION == 0:
                for i in range(12): q_des[:, real2col[i]] = float(np.interp(tt, t, cmd[:,i]))
            robot.set_joint_position_target(q_des); robot.set_joint_velocity_target(v_des)
            robot.write_root_pose_to_sim(root_pose); robot.write_root_velocity_to_sim(root_vel)
            robot.write_data_to_sim(); sim.step(); robot.update(SIM_DT)
            q = robot.data.joint_pos[0].cpu().numpy(); v = robot.data.joint_vel[0].cpu().numpy()
            out_rows.append((tt, [q[real2col[i]] for i in range(12)], [v[real2col[i]] for i in range(12)]))
            if not app.is_running(): break
        out = f"{args.out_dir}/SIM_sweep_vlim{vlim:g}.csv"
        with open(out,"w") as f:
            cols=[f"q{i}" for i in range(12)]+[f"v{i}" for i in range(12)]
            f.write("t,"+",".join(cols)+"\n")
            for tt,q,v in out_rows: f.write(f"{tt:.6f},"+",".join(f"{x:.6f}" for x in (q+v))+"\n")
        print(f"SWEEP_DONE vlim={vlim:g} -> {os.path.basename(out)} ({len(out_rows)})")


if __name__ == "__main__":
    main(); app.close()
