#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 MuJoCo 里运动学回放实机记录的姿态(纯可视化, 不跑动力学/控制)。

用途: 直接看实机当时每个关节(尤其踝)摆成什么样, 排查 sim/real 姿态不符。
输入: 实机数据 csv 路径 (rl_real 记录的 obs_*.csv, 含 q0..q11 与 qw..qz)。
行为: 把每帧的关节角写入 qpos, IMU 四元数写入 base 姿态, mj_forward + 渲染,
      按实机时间轴播放; 每轮循环开始前暂停 2s。

用法:
    python replay_real_in_mujoco.py <obs_xxx.csv>
    python replay_real_in_mujoco.py <obs_xxx.csv> --xml <scene.xml> --speed 1.0 --pause 2.0
"""
import sys
import os
import time
import argparse
import re
import numpy as np
import mujoco
import mujoco.viewer

DEFAULT_XML = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
              "source", "legs_rl_lab", "legs_rl_lab", "assets", "legs_URDF", "mjcf", "A1_legs_V2_mjcf_scene.xml"))
# 实机 csv 的 q0..q11 顺序(= /left_joint_states 实机序)
REAL_NAMES = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6']
BASE_HEIGHT = 0.55          # 固定 base 高度(纯可视化, 不受重力)
# 若某关节回放方向反了, 在这里对应名字填 -1(默认全 +1; /left_joint_states 一般已是策略约定)
SIGN = {n: 1.0 for n in REAL_NAMES}


def build_joint_map(model):
    """把模型里的 12 个腿铰关节按名字(含 R#/L#)映射到 REAL_NAMES 的列索引。
    返回 [(qpos_adr, real_col, sign), ...]; 失败则回退到 mjc 序 (R1..R6,L1..L6)。"""
    hinge = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            hinge.append((j, name, int(model.jnt_qposadr[j])))
    mapping = []
    matched = 0
    for j, name, adr in hinge:
        m = re.search(r'([RL])([1-6])', name)
        if m:
            token = m.group(1) + m.group(2)
            if token in REAL_NAMES:
                mapping.append((adr, REAL_NAMES.index(token), SIGN[token], token))
                matched += 1
                continue
        mapping.append((adr, None, 1.0, name))
    if matched >= 12:
        print("[map] 按关节名匹配成功:")
        for adr, col, sgn, tok in mapping:
            if col is not None:
                print(f"   qpos[{adr}] <- 实机 {REAL_NAMES[col]} (col q{col}) sign={sgn:+.0f}")
        return [(adr, col, sgn) for adr, col, sgn, _ in mapping if col is not None]
    # 回退: 假设 12 个铰关节按 mjc 序 R1..R6,L1..L6 排列
    print(f"[map] 名字匹配不全(matched={matched}), 回退到 mjc 序 R1..R6,L1..L6")
    mjc = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6']
    out = []
    for k, (j, name, adr) in enumerate(hinge[:12]):
        tok = mjc[k]
        out.append((adr, REAL_NAMES.index(tok), SIGN[tok]))
        print(f"   qpos[{adr}] <- 实机 {tok} (第{k}个铰关节 name={name})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--xml", default=DEFAULT_XML)
    ap.add_argument("--speed", type=float, default=1.0, help="播放速度倍率")
    ap.add_argument("--pause", type=float, default=2.0, help="每轮循环前暂停秒数")
    ap.add_argument("--no-imu", action="store_true", help="不用 IMU 姿态(base 保持竖直)")
    args = ap.parse_args()

    d = np.genfromtxt(args.data, delimiter=",", names=True, dtype=None, encoding=None)
    cols = d.dtype.names
    def c(k): return np.array(d[k], dtype=float)
    t = c('t'); t = t - t[0]
    q = np.stack([c(f'q{i}') for i in range(12)], axis=1)   # (N,12) 实机序
    has_quat = all(k in cols for k in ('qw', 'qx', 'qy', 'qz'))
    quat = (np.stack([c('qw'), c('qx'), c('qy'), c('qz')], axis=1)
            if (has_quat and not args.no_imu) else None)
    N = len(t)
    print(f"[data] {os.path.basename(args.data)}  帧={N} 时长={t[-1]:.1f}s  IMU姿态={'用' if quat is not None else '不用'}")

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    jmap = build_joint_map(model)

    # base free joint 的 qpos 起始(通常 0): 找 free 关节
    base_adr = None
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            base_adr = int(model.jnt_qposadr[j]); break

    def set_frame(k):
        mujoco.mj_resetData(model, data)
        if base_adr is not None:
            data.qpos[base_adr:base_adr + 3] = [0.0, 0.0, BASE_HEIGHT]
            data.qpos[base_adr + 3:base_adr + 7] = quat[k] if quat is not None else [1, 0, 0, 0]
        for adr, col, sgn in jmap:
            data.qpos[adr] = sgn * q[k, col]
        mujoco.mj_forward(model, data)

    dt = np.diff(t)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -20
        while viewer.is_running():
            # 每轮开始前: 摆到第0帧, 暂停 pause 秒
            set_frame(0)
            viewer.sync()
            print(f"[loop] 起始姿态, 暂停 {args.pause:.1f}s ...")
            t_end = time.time() + args.pause
            while viewer.is_running() and time.time() < t_end:
                viewer.sync(); time.sleep(0.02)
            if not viewer.is_running():
                break
            # 播放
            for k in range(N):
                if not viewer.is_running():
                    break
                set_frame(k)
                viewer.sync()
                if k < N - 1:
                    time.sleep(max(0.0, dt[k] / max(args.speed, 1e-6)))
    print("退出。")


if __name__ == "__main__":
    main()
