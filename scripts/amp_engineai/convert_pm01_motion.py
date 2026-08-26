"""把 EngineAI 的 PM01 运动数据 locomotion.npz 关节重映射到 nlegs, 存成 EngineAI 格式 npz。

用户选定「路线1: 关节重映射」——不走 GMR 的 IK, 直接把 PM01 的 12 条腿关节按语义角色
一一映射到 nlegs 的 12 关节, base 线/角速度取 PM01 root, 其余 body 用 nlegs 的 MuJoCo FK
正向算出。判别器实际只用 [joint_pos*9, base_lin_vel_b*7], 二者都从 PM01 精确迁移。

关键: locomotion.npz 的 joint_pos 是 EngineAI 用 robot.write_joint_state_to_sim 写入的,
即 **IsaacLab 内部关节序(PhysX 的 DOF/BFS 遍历序)**, 不是 URDF 里"先左腿后右腿"的 DFS 序!
对 PM01 URDF (LINK_BASE 出发 BFS), 23 列实测序为(已用 init 默认姿态 10/10 校验):
  0 J00_HIP_PITCH_L  1 J06_HIP_PITCH_R  2 J12_WAIST_YAW
  3 J01_HIP_ROLL_L   4 J07_HIP_ROLL_R   5 J13_SHOULDER_PITCH_L  6 J18_SHOULDER_PITCH_R
  7 J02_HIP_YAW_L    8 J08_HIP_YAW_R    9 J14_SHOULDER_ROLL_L  10 J19_SHOULDER_ROLL_R
 11 J03_KNEE_PITCH_L 12 J09_KNEE_PITCH_R 13 J15_SHOULDER_YAW_L  14 J20_SHOULDER_YAW_R
 15 J04_ANKLE_PITCH_L 16 J10_ANKLE_PITCH_R 17 J16_ELBOW_PITCH_L 18 J21_ELBOW_PITCH_R
 19 J05_ANKLE_ROLL_L 20 J11_ANKLE_ROLL_R 21 J17_ELBOW_YAW_L    22 J22_ELBOW_YAW_R
故 12 条腿关节散落在列 [0,1,3,4,7,8,11,12,15,16,19,20]。
nlegs 目标序(Isaac 交错, 与 replay ISAAC_JOINTS / env amp obs 一致):
  R1 L1 R2 L2 R3 L3 R4 L4 R5 L5 R6 L6
每条腿语义链两边一致: [hip_pitch(y), hip_roll(x), hip_yaw(z), knee(y), ankle_pitch(y), ankle_roll(x)]。

符号: PM01 与 nlegs 对应角色的关节轴向一致(pitch=+y, roll=+x, yaw=+z), 膝均为正屈曲,
默认全部 +1; 若回放发现某关节反了, 用 --flip joint_R4,joint_L4 翻转对应关节再重跑。

用法:
    python scripts/amp_engineai/convert_pm01_motion.py \
        [--src <locomotion.npz>] [--out <nlegs_locomotion.npz>] [--flip joint_R4,joint_L4]
"""

from __future__ import annotations

import argparse
import os

import mujoco
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_THIS))
_ASSETS = os.path.join(_REPO, "source", "legs_rl_lab", "legs_rl_lab", "assets")
NLEGS_XML = os.path.join(_ASSETS, "legs_narrow", "mjcf", "legs_narrow.xml")

DEFAULT_SRC = "/home/woan/workspace/engineai_amp/dataset/data/locomotion.npz"
DEFAULT_OUT = os.path.join(
    _REPO,
    "source/legs_rl_lab/legs_rl_lab/tasks/amp_task/datasets/motion_amp_engineai/nlegs_locomotion.npz",
)

# nlegs npz 关节列顺序 (Isaac 交错序); 必须与 env amp obs / replay 保持一致
NLEGS_AMP_JOINT_ORDER = [
    "joint_R1", "joint_L1", "joint_R2", "joint_L2", "joint_R3", "joint_L3",
    "joint_R4", "joint_L4", "joint_R5", "joint_L5", "joint_R6", "joint_L6",
]
# 每个 nlegs 关节对应的 PM01 列索引(索引进完整 23 列 IsaacLab-BFS 序, 按语义角色+左右)
# NLEGS_AMP_JOINT_ORDER = R1 L1 R2 L2 R3 L3 R4 L4 R5 L5 R6 L6
#   R_hip_pitch=1  L_hip_pitch=0 | R_hip_roll=4 L_hip_roll=3 | R_hip_yaw=8 L_hip_yaw=7
#   R_knee=12 L_knee=11 | R_ankle_pitch=16 L_ankle_pitch=15 | R_ankle_roll=20 L_ankle_roll=19
PM01_SRC_INDEX = [1, 0, 4, 3, 8, 7, 12, 11, 16, 15, 20, 19]

# nlegs body 顺序: base(锚点=body 0) + 两腿各 7 段
NLEGS_BODY_ORDER = [
    "base",
    "Link_R0", "Link_R1", "Link_R2", "Link_R3", "Link_R4", "Link_R5", "Link_R6",
    "Link_L0", "Link_L1", "Link_L2", "Link_L3", "Link_L4", "Link_L5", "Link_L6",
]


def _quat_finite_diff_ang_vel(quat: np.ndarray, dt: float) -> np.ndarray:
    """由 body 世界系四元数序列(wxyz)中心差分出世界系角速度 (N, 3)。"""
    n = quat.shape[0]
    ang = np.zeros((n, 3), dtype=np.float64)

    def _mul(a, b):  # 四元数乘 wxyz
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return np.array([
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ])

    for t in range(n):
        t0 = max(t - 1, 0)
        t1 = min(t + 1, n - 1)
        span = (t1 - t0) * dt
        if span <= 0:
            continue
        q0 = quat[t0].astype(np.float64)
        q1 = quat[t1].astype(np.float64)
        # q_rel = q1 * conj(q0)
        q_rel = _mul(q1, np.array([q0[0], -q0[1], -q0[2], -q0[3]]))
        w = np.clip(q_rel[0], -1.0, 1.0)
        angle = 2.0 * np.arccos(w)
        s = np.sqrt(max(1.0 - w * w, 1e-12))
        axis = q_rel[1:] / s if s > 1e-6 else np.zeros(3)
        if angle > np.pi:
            angle -= 2.0 * np.pi
        ang[t] = axis * (angle / span)
    return ang


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC, help="EngineAI PM01 locomotion.npz")
    ap.add_argument("--out", default=DEFAULT_OUT, help="输出 nlegs engineai 格式 npz")
    ap.add_argument("--flip", default="", help="逗号分隔的需翻符号关节名, 如 joint_R4,joint_L4")
    args = ap.parse_args()

    flip_set = {s.strip() for s in args.flip.split(",") if s.strip()}
    signs = np.array([-1.0 if jn in flip_set else 1.0 for jn in NLEGS_AMP_JOINT_ORDER])
    if flip_set:
        print(f"[flip] 翻转关节: {sorted(flip_set)}")

    src = np.load(args.src)
    fps = float(np.asarray(src["fps"]).reshape(-1)[0])
    dt = 1.0 / fps
    N = src["joint_pos"].shape[0]
    print(f"[load] {args.src}: {N} 帧, fps={fps}, joints={src['joint_pos'].shape[1]}, "
          f"bodies={src['body_pos_w'].shape[1]}")

    # --- 关节重映射(带符号) ---
    pm_jpos = src["joint_pos"][:, PM01_SRC_INDEX]      # (N,12)
    pm_jvel = src["joint_vel"][:, PM01_SRC_INDEX]      # (N,12)
    joint_pos = (pm_jpos * signs[None, :]).astype(np.float32)
    joint_vel = (pm_jvel * signs[None, :]).astype(np.float32)

    # PM01 root(body 0) 状态: 迁到 nlegs base, 保证 base_lin_vel_b 与 PM01 完全一致
    root_pos = src["body_pos_w"][:, 0, :].astype(np.float64)      # (N,3)
    root_quat = src["body_quat_w"][:, 0, :].astype(np.float64)    # (N,4) wxyz
    root_lin_w = src["body_lin_vel_w"][:, 0, :].astype(np.float64)
    root_ang_w = src["body_ang_vel_w"][:, 0, :].astype(np.float64)

    # --- nlegs MuJoCo FK: 逐帧算各 body 世界位姿 ---
    model = mujoco.MjModel.from_xml_path(NLEGS_XML)
    data = mujoco.MjData(model)
    qadr = {}
    for jn in NLEGS_AMP_JOINT_ORDER:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        assert jid >= 0, f"关节 {jn} 不在 mjcf"
        qadr[jn] = model.jnt_qposadr[jid]
    body_ids = []
    for bn in NLEGS_BODY_ORDER:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bn)
        assert bid >= 0, f"body {bn} 不在 mjcf"
        body_ids.append(bid)
    B = len(body_ids)

    body_pos_w = np.zeros((N, B, 3), dtype=np.float64)
    body_quat_w = np.zeros((N, B, 4), dtype=np.float64)

    for t in range(N):
        data.qpos[:] = 0.0
        data.qpos[0:3] = root_pos[t]
        data.qpos[3:7] = root_quat[t]  # wxyz
        for k, jn in enumerate(NLEGS_AMP_JOINT_ORDER):
            data.qpos[qadr[jn]] = joint_pos[t, k]
        mujoco.mj_kinematics(model, data)
        for j, bid in enumerate(body_ids):
            body_pos_w[t, j] = data.xpos[bid]
            body_quat_w[t, j] = data.xquat[bid]  # wxyz

    # --- body 速度: 位置中心差分 + 四元数差分角速度; body 0 用 PM01 精确值覆盖 ---
    body_lin_vel_w = np.zeros((N, B, 3), dtype=np.float64)
    body_ang_vel_w = np.zeros((N, B, 3), dtype=np.float64)
    body_lin_vel_w[1:-1] = (body_pos_w[2:] - body_pos_w[:-2]) / (2.0 * dt)
    body_lin_vel_w[0] = (body_pos_w[1] - body_pos_w[0]) / dt
    body_lin_vel_w[-1] = (body_pos_w[-1] - body_pos_w[-2]) / dt
    for j in range(B):
        body_ang_vel_w[:, j, :] = _quat_finite_diff_ang_vel(body_quat_w[:, j, :], dt)

    # 锚点(base)用 PM01 root 真值 -> AMP 的 base_lin_vel_b 与 PM01 完全对齐
    body_pos_w[:, 0, :] = root_pos
    body_quat_w[:, 0, :] = root_quat
    body_lin_vel_w[:, 0, :] = root_lin_w
    body_ang_vel_w[:, 0, :] = root_ang_w

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(
        args.out,
        fps=np.array([fps], dtype=np.float32),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w.astype(np.float32),
        body_quat_w=body_quat_w.astype(np.float32),
        body_lin_vel_w=body_lin_vel_w.astype(np.float32),
        body_ang_vel_w=body_ang_vel_w.astype(np.float32),
        joint_names=np.array(NLEGS_AMP_JOINT_ORDER),
        body_names=np.array(NLEGS_BODY_ORDER),
    )
    print(f"[save] {args.out}")
    print(f"       joint_pos{joint_pos.shape} joint_vel{joint_vel.shape} "
          f"body_pos_w{body_pos_w.shape} ({B} bodies)")
    # 快速自检: base 系线速度 = rot(inv(quat0)) * lin_w0
    print(f"[check] frame0 base_lin_vel_w={np.round(body_lin_vel_w[0,0],3)}  "
          f"joint_pos[:6]={np.round(joint_pos[0,:6],3)}")


if __name__ == "__main__":
    main()
