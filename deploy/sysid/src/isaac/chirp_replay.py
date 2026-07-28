# -*- coding: utf-8 -*-
"""Isaac 单关节 chirp 扫频(吊起来), 用 LEGS_CFG 原配置 actuator。
对某关节的位置目标发 0.1->f1 Hz 对数扫频正弦, 其余关节保持 0, 记录 target/q。
用法(unitree_lab + setup_conda_env):
  python chirp_replay.py --joint 1 --f0 0.1 --f1 20 --amp 0.15 --dur 40 --out <csv> --headless
"""
from __future__ import annotations
import argparse, os, math
from isaaclab.app import AppLauncher

SIM_DT = 0.005                      # 200Hz 物理步; 目标每步更新(=实机 200Hz 下发)
ASSET_SYSPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "source", "legs_rl_lab"))

parser = argparse.ArgumentParser()
parser.add_argument("--joint", type=int, required=True, help="实机 idx 0-11 (L2=1,R2=7)")
parser.add_argument("--f0", type=float, default=0.1)
parser.add_argument("--f1", type=float, default=20.0)
parser.add_argument("--amp", type=float, default=0.15, help="扫频幅度 rad")
parser.add_argument("--dur", type=float, default=40.0)
parser.add_argument("--armature", type=float, default=None, help="覆盖被测关节 armature(不改 legs.py, 运行时设)")
parser.add_argument("--kd", type=float, default=None, help="覆盖被测关节 kd(阻尼)")
parser.add_argument("--viscous", type=float, default=None, help="额外粘滞摩擦(叠加到阻尼, 代表齿轮箱/反电动势物理阻尼; kd 保持不变)")
parser.add_argument("--replay_csv", type=str, default=None, help="给定则用该 cmd 文件的 qd{joint} 作为目标(替代 chirp), 用于用实机正弦命令驱动 sim")
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import sys, csv, copy, importlib
import numpy as np, torch
sys.path.append(ASSET_SYSPATH)
LEGS_CFG = getattr(importlib.import_module("legs_rl_lab.assets.legs_URDF.legs"), "LEGS_CFG")
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.assets import Articulation

def sim_joint(j): return f"joint_L{j + 1}" if j < 6 else f"joint_R{j - 5}"

def main():
    sim = SimulationContext(SimulationCfg(dt=SIM_DT, device="cpu"))
    cfg = copy.deepcopy(LEGS_CFG); cfg.prim_path = "/World/Robot"
    cfg.init_state.joint_pos = {".*": 0.0}
    robot = Articulation(cfg); sim.reset()
    for nm,a in robot.actuators.items(): print(f"[actuator] {nm}: {type(a).__name__} joints={a.joint_names}")
    jcol = robot.find_joints(sim_joint(args.joint))[0][0]
    if args.armature is not None:
        robot.write_joint_armature_to_sim(float(args.armature), joint_ids=[jcol])
        print(f"[override] joint {args.joint} armature -> {args.armature}")
    if args.kd is not None:
        for gname, act in robot.actuators.items():
            gi = list(np.array(act.joint_indices).ravel())
            if jcol in gi:
                loc = gi.index(jcol); act.damping[:, loc] = float(args.kd)
                print(f"[override] joint {args.joint} kd -> {args.kd} (group {gname})")
    if args.viscous is not None:
        for gname, act in robot.actuators.items():
            gi = list(np.array(act.joint_indices).ravel())
            if jcol in gi:
                loc = gi.index(jcol); act.damping[:, loc] += float(args.viscous)
                print(f"[override] joint {args.joint} +viscous {args.viscous} -> total damping {float(act.damping[0, loc]):.1f} (kd 保持配置值, 这是物理摩擦)")
    root_pose = torch.tensor([[0.,0.,0.65,1.,0.,0.,0.]]).repeat(robot.num_instances,1)
    root_vel = torch.zeros((robot.num_instances,6))
    q_des = torch.zeros((robot.num_instances, robot.num_joints)); v_des = torch.zeros_like(q_des)
    robot.write_joint_state_to_sim(q_des, torch.zeros_like(q_des)); robot.reset()

    # 对数扫频相位: f(t)=f0*k^(t/T), phase=2π f0 T (k^(t/T)-1)/ln k
    k = args.f1/args.f0; T = args.dur; lnk = math.log(k)
    rep_t = rep_q = None
    if args.replay_csv:                       # 用实机 cmd 文件的目标驱动(替代 chirp)
        rr = list(csv.DictReader(open(args.replay_csv)))
        rep_t = np.array([float(x["t"]) for x in rr]); rep_t -= rep_t[0]
        rep_q = np.array([float(x[f"qd{args.joint}"]) for x in rr])
        T = float(rep_t[-1]); print(f"[replay] {args.replay_csv} dur={T:.1f}s")
    n = int(T/SIM_DT); rows=[]
    for i in range(n+1):
        tt = i*SIM_DT
        if rep_t is not None:
            finst = 0.0; tgt = float(np.interp(tt, rep_t, rep_q))
        else:
            finst = args.f0*(k**(tt/T))
            phase = 2*math.pi*args.f0*T*(k**(tt/T)-1)/lnk
            tgt = args.amp*math.sin(phase)
        q_des[:, jcol] = tgt
        robot.set_joint_position_target(q_des); robot.set_joint_velocity_target(v_des)
        robot.write_root_pose_to_sim(root_pose); robot.write_root_velocity_to_sim(root_vel)
        robot.write_data_to_sim(); sim.step(); robot.update(SIM_DT)
        q = float(robot.data.joint_pos[0, jcol].cpu()); v=float(robot.data.joint_vel[0,jcol].cpu())
        rows.append((tt, finst, tgt, q, v))
        if not app.is_running(): break
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out,"w") as f:
        f.write("t,finst,target,q,v\n")
        for r in rows: f.write(",".join(f"{x:.6f}" for x in r)+"\n")
    print(f"CHIRP_DONE saved {args.out} ({len(rows)} steps)")

if __name__=="__main__":
    main(); app.close()
