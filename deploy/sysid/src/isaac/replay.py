# -*- coding: utf-8 -*-
"""Isaac Lab 单/多关节 sysid 回放 + 参数扫描(合并 replay_all/replay_sysid/replay_eff/
replay_fr2/sweep_friction/viscous_sweep/armature_sweep/delay_sweep 为一个脚本)。

复用训练资产(默认 legs_rl_lab LEGS_CFG), 一律在 DelayedPDActuator 上按训练时序回放实机命令,
输出与实机同格式 state CSV。PD 读 arm_control_node.yaml(与实机一致)。

约定/修正:
  1) 关节按名匹配: 实机 idx0-5 → joint_L1..L6, 6-11 → joint_R1..R6(与实机/部署一致)。
  2) effort_limit 固定=训练值(.*1-.*5=27, .*6=7)。
  3) 一律 DelayedPDActuator。
  4) 输出一律带标签(扫描: SIM_<tag>_<var><val>; 基线: SIM_<tag>[_suffix]), 不覆盖基线。
  5) 路径/资产可配置(--yaml/--asset-*)。

用法示例(env_isaaclab, GPU, --headless):
  # 基线全关节(friction 0)
  python replay.py --data_dir <dir> --joints all --headless
  # 摩擦扫描 j0,j6
  python replay.py --data_dir <dir> --joints 0,6 --sweep friction --values 0,0.5,1,2,4 --headless
  # 延迟扫描
  python replay.py --data_dir <dir> --joints 0,6 --sweep delay --values 0,1,2,4,6,8,10 --headless
  # armature 扫描(values=倍数×本关节基准惯量)
  python replay.py --data_dir <dir> --joints 0,6 --sweep armature --values 1,2,4,8,16 --headless
"""
from __future__ import annotations
import argparse
import os
from isaaclab.app import AppLauncher

SIM_DT, DECIMATION = 0.005, 4
ARM_DM4340, ARM_DM4310 = 2.193e-5 * 48.19 ** 2, 2.193e-5 * 10 ** 2   # 0.0509 / 0.00219 (实测: 转子惯量2.193e-5, 减速比48.19/10)
# 训练资产(legs_rl_lab LEGS_CFG, A1_legs_V2); 相对本文件定位(deploy/sysid/src/isaac/ 上溯4级到 legs_rl_lab)
ASSET_SYSPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "source", "legs_rl_lab"))
ASSET_MODULE = "legs_rl_lab.assets.legs_URDF.legs"
ASSET_CFG = "LEGS_CFG"
# 非被测关节的保持 PD(=训练默认 J1-4=200/5, J5-6=40/0.5); 被测关节用 --kp/--kd 覆盖
HOLD_KP = {j: (200.0 if (j % 6) < 4 else 40.0) for j in range(12)}
HOLD_KD = {j: (5.0 if (j % 6) < 4 else 0.5) for j in range(12)}

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True, help="实机 j*_cmd.csv 所在目录, 输出也写这里")
parser.add_argument("--joints", default="all", help="'all' 或 '0,3,4'(实机序 idx)")
parser.add_argument("--modes", default="step_fwd,step_rev,sine", help="要回放的激励模式(匹配 cmd 文件名)")
parser.add_argument("--kp", type=float, default=None, help="被测关节仿真 kp; 缺省=用每个 cmd 文件名里的 kp(sweep 目录逐 kp 回放)")
parser.add_argument("--kd", type=float, default=None, help="被测关节仿真 kd; 缺省=用文件名里的 kd")
parser.add_argument("--sweep", choices=["none", "friction", "viscous", "delay", "armature", "dcvel", "dcsat"], default="none")
parser.add_argument("--values", default="", help="扫描值(逗号); armature 为倍数×本关节基准惯量; dcvel/dcsat 为绝对值")
parser.add_argument("--actuator", choices=["pd", "dc", "ddc"], default="pd", help="pd=DelayedPD(带延迟); dc=DCMotor(转矩-转速滚降,无延迟); ddc=两者合体(延迟+滚降)")
parser.add_argument("--dc-sat", type=float, default=40.0, help="DCMotor 堵转力矩 saturation_effort [N·m]")
parser.add_argument("--dc-vlim", type=float, default=6.0, help="DCMotor 空载转速 velocity_limit [rad/s]")
parser.add_argument("--dc-effort", type=float, default=26.0, help="DCMotor 持续力矩平台 effort_limit [N·m]")
parser.add_argument("--friction", type=float, default=0.0, help="静摩擦(非 friction 扫描时的固定值)")
parser.add_argument("--dynamic-friction", type=float, default=0.0)
parser.add_argument("--viscous", type=float, default=0.0, help="黏性摩擦(非 viscous 扫描时的固定值)")
parser.add_argument("--delay", type=int, default=4, help="执行延迟物理步(非 delay 扫描时的固定值; 4=20ms)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import sys, csv, copy, glob, os, importlib
import numpy as np
import torch
sys.path.append(ASSET_SYSPATH)
ROBOT_CFG = getattr(importlib.import_module(ASSET_MODULE), ASSET_CFG)
from isaaclab.actuators import DelayedPDActuatorCfg, DCMotorCfg
from legs_rl_lab.actuators import DelayedDCMotorCfg   # 自定义: 延迟+转矩转速滚降
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.assets import Articulation


def sim_joint(j):
    """实机 idx → USD 关节名: idx0-5=joint_L1..L6, 6-11=joint_R1..R6(按名匹配, 与实机一致)。"""
    return f"joint_L{j + 1}" if j < 6 else f"joint_R{j - 5}"


def base_arm(j):
    return ARM_DM4310 if (j % 6) in (4, 5) else ARM_DM4340


def load_cmd(path):
    """读全 12 维命令(实机序): 仿真要复现真机发的整条 12 维指令(default 站姿 + 被测关节激励),
    否则非被测关节归 0 会让站姿/重力力矩与真机不符。返回 t, qd[n,12], phase。"""
    t, ph = [], []
    qd = [[] for _ in range(12)]
    with open(path) as f:
        for r in csv.DictReader(f):
            t.append(float(r["t"])); ph.append(r["phase"])
            for i in range(12):
                qd[i].append(float(r[f"qd{i}"]))
    return np.array(t), np.array(qd).T, np.array(ph, dtype=object)


def main():
    joints = list(range(12)) if args.joints == "all" else [int(x) for x in args.joints.split(",")]
    modes = args.modes.split(",")
    values = [float(x) for x in args.values.split(",")] if (args.sweep != "none" and args.values) else [None]
    if args.sweep != "none" and values == [None]:
        raise SystemExit(f"--sweep {args.sweep} 需要 --values")

    tested = set(joints)
    # 被测关节增益: 每份 cmd 文件回放前按文件名 kp/kd 在运行时设入(见下方循环);
    # 这里先用占位(HOLD), 非被测关节固定用 legs.py 保持 PD。
    stiff = {sim_joint(i): HOLD_KP[i] for i in range(12)}
    damp = {sim_joint(i): HOLD_KD[i] for i in range(12)}
    arm0 = {sim_joint(i): base_arm(i) for i in range(12)}

    max_delay = max([args.delay] + [int(v) for v in values if v is not None]) if args.sweep == "delay" else args.delay

    sim = SimulationContext(SimulationCfg(dt=SIM_DT, device="cpu"))
    cfg = copy.deepcopy(ROBOT_CFG)
    cfg.prim_path = "/World/Robot"
    cfg.init_state.joint_pos = {".*": 0.0}
    eff = {".*1": 26.0, ".*2": 26.0, ".*3": 26.0, ".*4": 26.0, ".*5": 26.0, ".*6": 5.8}  # =legs.py
    if args.actuator == "dc":
        # DCMotor: 转矩-转速滚降 τ_max(v)=sat·(1−v/vlim), 平台=effort_limit; 无延迟
        cfg.actuators = {"leg": DCMotorCfg(
            joint_names_expr=[".*"], effort_limit=eff,
            saturation_effort=args.dc_sat, velocity_limit=args.dc_vlim,
            stiffness=stiff, damping=damp, armature=arm0,
            friction=args.friction, dynamic_friction=args.dynamic_friction, viscous_friction=args.viscous)}
    elif args.actuator == "ddc":
        # DelayedDCMotor: 延迟(缓冲按扫描最大值分配, 运行时 set_time_lag) + 转矩-转速滚降
        cfg.actuators = {"leg": DelayedDCMotorCfg(
            joint_names_expr=[".*"], effort_limit=eff,
            saturation_effort=args.dc_sat, velocity_limit=args.dc_vlim,
            stiffness=stiff, damping=damp, armature=arm0,
            friction=args.friction, dynamic_friction=args.dynamic_friction, viscous_friction=args.viscous,
            min_delay=0, max_delay=int(max_delay))}
    else:
        cfg.actuators = {"leg": DelayedPDActuatorCfg(
            joint_names_expr=[".*"], effort_limit=eff,
            velocity_limit=14.0, stiffness=stiff, damping=damp, armature=arm0,
            friction=args.friction, dynamic_friction=args.dynamic_friction, viscous_friction=args.viscous,
            min_delay=0, max_delay=int(max_delay))}
    robot = Articulation(cfg)
    sim.reset()
    act = robot.actuators["leg"]
    real2col = [robot.find_joints(sim_joint(i))[0][0] for i in range(12)]
    root_pose = torch.tensor([[0.0, 0.0, 0.65, 1.0, 0.0, 0.0, 0.0]]).repeat(robot.num_instances, 1)
    root_vel = torch.zeros((robot.num_instances, 6))
    q_des = torch.zeros((robot.num_instances, robot.num_joints)); v_des = torch.zeros_like(q_des)

    def apply_cfg(jcol, val):
        """每次回放前把可运行时设置的参数设成本次的目标值(自包含, 避免跨次污染)。"""
        st, dy, vi, dly = args.friction, args.dynamic_friction, args.viscous, args.delay
        arm_j = base_arm_of_col.get(jcol, ARM_DM4340)
        if args.sweep == "friction": st = val
        elif args.sweep == "viscous": vi = val
        elif args.sweep == "delay": dly = int(val)
        elif args.sweep == "armature": arm_j = base_arm_of_col.get(jcol, ARM_DM4340) * val
        robot.write_joint_friction_coefficient_to_sim(st, dy, vi)
        robot.write_joint_armature_to_sim(float(arm_j), joint_ids=[jcol])
        if args.actuator in ("dc", "ddc"):        # 转矩-转速参数(可被 dcvel/dcsat 扫描覆盖)
            if args.sweep == "dcvel": act.velocity_limit[:] = val
            elif args.sweep == "dcsat": act._saturation_effort[:] = val
            act._vel_at_effort_lim = act.velocity_limit * (1.0 + act.effort_limit / act._saturation_effort)
        if args.actuator in ("pd", "ddc"):        # 有延迟缓冲的模型才设 time_lag
            for buf in (act.positions_delay_buffer, act.velocities_delay_buffer, act.efforts_delay_buffer):
                buf.set_time_lag(int(dly))

    base_arm_of_col = {real2col[i]: base_arm(i) for i in range(12)}

    def tag_of(val):
        if args.actuator in ("dc", "ddc"):       # dc/ddc 输出名总带 actuator+参数(+delay), 不覆盖 PD 基线
            vlim = val if args.sweep == "dcvel" else args.dc_vlim
            sat = val if args.sweep == "dcsat" else args.dc_sat
            dly = val if args.sweep == "delay" else args.delay   # 延迟扫描时 delay 也进 tag, 否则覆盖
            d = f"d{int(dly)}" if args.actuator == "ddc" else ""
            return f"_{args.actuator}v{vlim:g}s{sat:g}{d}"
        if args.sweep == "none":
            return ""
        short = {"friction": "fr", "viscous": "visc", "delay": "dly", "armature": "arm"}[args.sweep]
        return f"_{short}{val:g}"

    for J in joints:
        jcol = real2col[J]
        for mode in modes:
            fs = sorted(glob.glob(f"{args.data_dir}/j{J}_kp*_kd*_{mode}_cmd.csv"))
            if args.kp is not None:               # 给了 --kp: 只回放该 kp 的文件(文件选择过滤)
                fs = [f for f in fs if os.path.basename(f).split("_kp")[1].split("_kd")[0] == f"{args.kp:g}"]
            if not fs:
                print(f"[skip] j{J} {mode}: 无 cmd 文件"); continue
            for cf in fs:                       # sweep 目录: 同 (关节,模式) 的每个 kp 各回放一次
                base_tag = os.path.basename(cf)[:-len("_cmd.csv")]
                kp_f = base_tag.split("_kp")[1].split("_kd")[0]
                kd_f = base_tag.split("_kd")[1].split("_")[0]
                kp_use = args.kp if args.kp is not None else float(kp_f)
                kd_use = args.kd if args.kd is not None else float(kd_f)
                act.stiffness[:, jcol] = kp_use   # 被测关节增益 = 本文件辨识用的 kp/kd(其余关节保持 legs.py PD)
                act.damping[:, jcol] = kd_use
                t_cmd, qd_cmd, ph_cmd = load_cmd(cf)
                # 初始关节状态 = 首帧命令(实机 ramp 起点), 避免从全 0 起步的开场瞬态
                q_init = torch.zeros_like(q_des)
                for i in range(12):
                    q_init[:, real2col[i]] = float(qd_cmd[0, i])
                for val in values:
                    robot.write_joint_state_to_sim(q_init, torch.zeros_like(q_des)); robot.reset()
                    apply_cfg(jcol, val)
                    q_des.copy_(q_init)
                    rows = []; n = int(t_cmd[-1] / SIM_DT)
                    for k in range(n + 1):
                        tt = k * SIM_DT
                        if k % DECIMATION == 0:
                            for i in range(12):    # 复现真机整条 12 维命令(全身按 default, 被测关节激励)
                                q_des[:, real2col[i]] = float(np.interp(tt, t_cmd, qd_cmd[:, i]))
                        robot.set_joint_position_target(q_des); robot.set_joint_velocity_target(v_des)
                        robot.write_root_pose_to_sim(root_pose); robot.write_root_velocity_to_sim(root_vel)
                        robot.write_data_to_sim(); sim.step(); robot.update(SIM_DT)
                        q = robot.data.joint_pos[0].cpu().numpy()
                        v = robot.data.joint_vel[0].cpu().numpy()
                        tau = robot.data.applied_torque[0].cpu().numpy()
                        pp = ph_cmd[int(np.argmin(np.abs(t_cmd - tt)))]
                        rows.append((tt, pp, [q[real2col[i]] for i in range(12)],
                                     [v[real2col[i]] for i in range(12)], [tau[real2col[i]] for i in range(12)]))
                        if not app.is_running():
                            break
                    out = f"{args.data_dir}/SIM_{base_tag}{tag_of(val)}_state.csv"
                    with open(out, "w") as f:
                        cols = [f"q{i}" for i in range(12)] + [f"v{i}" for i in range(12)] + [f"tau{i}" for i in range(12)]
                        f.write("t,phase," + ",".join(cols) + "\n")
                        for tt, pp, q, v, tau in rows:
                            f.write(f"{tt:.6f},{pp}," + ",".join(f"{x:.6f}" for x in (q + v + tau)) + "\n")
                    print(f"[{args.sweep}] j{J} {mode} kp{kp_use:g} val={val} -> {os.path.basename(out)} ({len(rows)}步)")


if __name__ == "__main__":
    main()
    app.close()
