"""把 TienKung-Lab 的 AMP 走路数据 (walk.txt) 转成 nlegs 机器人的 30 维专家动作数据。

原始 walk.txt (DeepMimic 风格 JSON, Frames 形状 [T, 52]) 的列布局（由 TienKung
tienkung_env.get_amp_obs_for_expert_trans 的 torch.cat 顺序确定, 已数值核对）:
    0-3   右臂关节位置        4-7   左臂关节位置
    8-13  右腿关节位置        14-19 左腿关节位置
    20-23 右臂速度            24-27 左臂速度
    28-33 右腿速度            34-39 左腿速度
    40-42 左手位置(root系)     43-45 右手位置
    46-48 左脚位置             49-51 右脚位置
每条腿 6 个关节的顺序为 [hip_roll, hip_pitch, hip_yaw, knee, ankle_pitch, ankle_roll]。

nlegs 只有下肢 12 DoF, Isaac 交错序 [R1,L1,R2,L2,R3,L3,R4,L4,R5,L5,R6,L6],
物理含义 .*1=髋pitch .*2=髋roll .*3=髋yaw .*4=膝 .*5=踝pitch .*6=踝roll。

输出 nlegs_walk.txt 的每帧 30 维:
    [0:12]  关节位置 (Isaac 序, 绝对角, rad)
    [12:24] 关节速度 (Isaac 序, rad/s, 由位置中心差分重算, dt=FrameDuration)
    [24:27] 右脚位置 (base 系, m)   [27:30] 左脚位置 (base 系, m)
脚位置用 nlegs 自己的 MJCF 做前向运动学重算(不能直接抄 TienKung 的 —— 它腿更长,
脚在 root 下方 ~0.87m, 而 nlegs 只有 ~0.58m, 直接抄会让判别器索要够不到的脚位置)。

★ 为什么关节用【绝对角】而非【相对默认】: TienKung 走路绝对角恰好都落在 nlegs 关节
限位内(膝 [0.07,1.34]⊂[0,1.92] 等); 而"相对各自默认"映射会把 nlegs 膝推成负角
(超伸, nlegs 膝限位 [0,1.92] 不允许)。绝对角复制 + FK 重算脚位置是跨形态重定向的稳妥做法。

用法:
    python scripts/amp/convert_tienkung_motion.py \
        [--src <walk.txt>] [--out <nlegs_walk.txt>] [--motion_weight 0.5]
符号可疑时(髋roll/髋yaw 左右方向)先用 replay_amp_motion.py 可视校验, 再按需在
SIGN 里翻转对应关节。
"""

from __future__ import annotations

import argparse
import json
import os

import mujoco
import numpy as np

# ---- 路径 (相对本文件, 不写死绝对路径) ----
_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_THIS))
_ASSETS = os.path.join(_REPO, "source", "legs_rl_lab", "legs_rl_lab", "assets")
DEFAULT_SRC = os.path.join(
    os.path.expanduser("~"),
    "workspace/TienKung-Lab/legged_lab/envs/tienkung/datasets/motion_amp_expert/walk.txt",
)
DEFAULT_OUT = os.path.join(
    _REPO,
    "source/legs_rl_lab/legs_rl_lab/tasks/amp_task/datasets/motion_amp_expert/nlegs_walk.txt",
)
NLEGS_XML = os.path.join(_ASSETS, "legs_narrow", "mjcf", "legs_narrow.xml")

# ---- 源列映射: nlegs Isaac 序关节 -> walk.txt 的位置列号 ----
# 每条腿源顺序 [hip_roll, hip_pitch, hip_yaw, knee, ankle_pitch, ankle_roll]
#   右腿位置列 8-13, 左腿 14-19。物理含义对齐到 nlegs 的 .*1..*6。
# Isaac 序输出 [R1,L1,R2,L2,R3,L3,R4,L4,R5,L5,R6,L6]:
#   R1 髋pitch=col9  L1=col15 | R2 髋roll=col8  L2=col14 | R3 髋yaw=col10 L3=col16
#   R4 膝=col11      L4=col17 | R5 踝pitch=col12 L5=col18 | R6 踝roll=col13 L6=col19
POS_SRC_COLS = [9, 15, 8, 14, 10, 16, 11, 17, 12, 18, 13, 19]
# nlegs Isaac 序关节名(与 POS_SRC_COLS 一一对应)
ISAAC_JOINTS = [
    "joint_R1", "joint_L1", "joint_R2", "joint_L2", "joint_R3", "joint_L3",
    "joint_R4", "joint_L4", "joint_R5", "joint_L5", "joint_R6", "joint_L6",
]
# 每关节符号(默认全 +1)。若回放发现髋roll/髋yaw 左右方向反了, 把对应位改 -1。
SIGN = np.ones(12, dtype=np.float64)

# nlegs 关节限位(与 legs_narrow.xml range 一致), 绝对角复制后夹紧(仅踝roll 偶尔越界 ~0.02rad)
JOINT_LIMITS = {
    "joint_R1": (-3.14, 1.05), "joint_L1": (-3.14, 1.04),
    "joint_R2": (-3.14, 0.26), "joint_L2": (-0.26, 3.14),
    "joint_R3": (-2.75, 2.75), "joint_L3": (-2.75, 2.75),
    "joint_R4": (0.0, 1.92),   "joint_L4": (0.0, 1.92),
    "joint_R5": (-0.79, 0.79), "joint_L5": (-0.79, 0.79),
    "joint_R6": (-0.35, 0.35), "joint_L6": (-0.35, 0.35),
}


def central_diff(x: np.ndarray, dt: float, loop: bool) -> np.ndarray:
    """对 [T, D] 序列按时间中心差分求速度。loop=True 用循环边界(Wrap)。"""
    if loop:
        return (np.roll(x, -1, axis=0) - np.roll(x, 1, axis=0)) / (2.0 * dt)
    v = np.zeros_like(x)
    v[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
    v[0] = (x[1] - x[0]) / dt
    v[-1] = (x[-1] - x[-2]) / dt
    return v


def build_fk(xml_path: str):
    """返回 (model, data, foot_r_id, foot_l_id, qadr) 用于逐帧前向运动学求脚位置。"""
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    foot_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Link_R6")
    foot_l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Link_L6")
    assert foot_r >= 0 and foot_l >= 0, "找不到 Link_R6 / Link_L6"
    # 每个关节名 -> qpos 地址
    qadr = {}
    for jn in ISAAC_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        assert jid >= 0, f"MJCF 里找不到关节 {jn}"
        qadr[jn] = model.jnt_qposadr[jid]
    return model, data, foot_r, foot_l, qadr


def foot_positions(model, data, foot_r, foot_l, qadr, qpos_joints: np.ndarray) -> np.ndarray:
    """给定 12 个 Isaac 序关节角, 用 FK 求左右脚在 base 系的位置 (base 置于原点单位姿态)。

    返回 [6] = [right_xyz, left_xyz]。base freejoint 置零 => world 系即 base 系。
    """
    data.qpos[:] = 0.0
    data.qpos[3] = 1.0  # 四元数 w=1 (base 单位朝向)
    for jn, val in zip(ISAAC_JOINTS, qpos_joints):
        data.qpos[qadr[jn]] = val
    mujoco.mj_kinematics(model, data)
    return np.concatenate([data.xpos[foot_r].copy(), data.xpos[foot_l].copy()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--motion_weight", type=float, default=None,
                    help="默认沿用源文件的 MotionWeight")
    ap.add_argument("--vel_clip", type=float, default=10.0,
                    help="关节速度绝对值上限(rad/s), 削掉差分尖刺(膝在摆动/支撑切换处偶发 ~17rad/s)")
    args = ap.parse_args()

    with open(args.src) as f:
        src = json.load(f)
    frames = np.array(src["Frames"], dtype=np.float64)
    dt = float(src["FrameDuration"])
    loop = str(src.get("LoopMode", "Wrap")).lower() == "wrap"
    T = frames.shape[0]
    print(f"[load] {args.src}: {T} 帧, dt={dt}, LoopMode={src.get('LoopMode')}")

    # 1) 关节位置: 取源列 -> 符号 -> 夹紧限位
    jpos = frames[:, POS_SRC_COLS] * SIGN[None, :]
    n_clip = 0
    for k, jn in enumerate(ISAAC_JOINTS):
        lo, hi = JOINT_LIMITS[jn]
        before = jpos[:, k].copy()
        jpos[:, k] = np.clip(jpos[:, k], lo, hi)
        n_clip += int(np.sum(before != jpos[:, k]))
    print(f"[joint] 位置已重映射到 Isaac 序 + 夹紧限位 (被夹样本数={n_clip})")

    # 2) 关节速度: 由重映射后的位置中心差分重算(自洽, 避开源 vel 的尖刺/单位问题)
    jvel = central_diff(jpos, dt, loop)
    n_spike = int((np.abs(jvel) > args.vel_clip).sum())
    jvel = np.clip(jvel, -args.vel_clip, args.vel_clip)
    print(f"[joint] 速度由中心差分重算 + 夹紧 ±{args.vel_clip} rad/s "
          f"(削尖刺样本数={n_spike}), |vel| max={np.abs(jvel).max():.2f} rad/s")

    # 3) 脚位置: 用 nlegs MJCF 逐帧 FK 重算(base 系)
    model, data, fr, fl, qadr = build_fk(NLEGS_XML)
    feet = np.stack([foot_positions(model, data, fr, fl, qadr, jpos[t]) for t in range(T)], axis=0)
    print(f"[foot] FK 重算完成, 右脚 z 均值={feet[:,2].mean():+.3f}m 左脚 z 均值={feet[:,5].mean():+.3f}m")

    # 4) 拼 30 维帧
    out_frames = np.concatenate([jpos, jvel, feet], axis=1)  # [T, 30]
    assert out_frames.shape[1] == 30, out_frames.shape

    out = {
        "LoopMode": src.get("LoopMode", "Wrap"),
        "FrameDuration": dt,
        "EnableCycleOffsetPosition": src.get("EnableCycleOffsetPosition", True),
        "EnableCycleOffsetRotation": src.get("EnableCycleOffsetRotation", False),
        "MotionWeight": args.motion_weight if args.motion_weight is not None
        else float(src.get("MotionWeight", 1.0)),
        "Comment": "converted from TienKung walk.txt -> nlegs 30d [12 jpos(Isaac), "
                   "12 jvel(central-diff), 6 foot pos(base frame: R xyz, L xyz)]",
        "Frames": out_frames.tolist(),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"[write] {args.out}: {out_frames.shape[0]} 帧 x {out_frames.shape[1]} 维")


if __name__ == "__main__":
    main()
