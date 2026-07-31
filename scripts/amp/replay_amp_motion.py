"""在 nlegs MuJoCo 模型上回放转换后的 AMP 专家动作, 可视校验重定向是否正确。

主要用来肉眼确认:
  - 步态整体是否像"走路"(左右腿交替摆动/支撑)
  - 左右腿有没有搞反
  - 髋roll / 髋yaw 的符号方向对不对(这是转换里唯一的残余风险)
  - 脚位置(FK 重算的 base 系脚位置)与关节姿态是否自洽

回放只驱动关节角(base 悬浮固定在空中), 不做物理, 纯运动学展示。

用法:
    python scripts/amp/replay_amp_motion.py [--file <nlegs_walk.txt>] [--speed 1.0] [--loop]
无显示器(SSH 无 X)时加 --headless, 只打印首帧脚位置等数值不弹窗。
"""

from __future__ import annotations

import argparse
import json
import os
import time

import mujoco
import mujoco.viewer
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_THIS))
_ASSETS = os.path.join(_REPO, "source", "legs_rl_lab", "legs_rl_lab", "assets")
NLEGS_XML = os.path.join(_ASSETS, "legs_narrow", "mjcf", "legs_narrow.xml")
DEFAULT_FILE = os.path.join(
    _REPO,
    "source/legs_rl_lab/legs_rl_lab/tasks/amp_task/datasets/motion_amp_expert/nlegs_walk.txt",
)

# 30 维帧里关节位置的 Isaac 序(与转换脚本一致)
ISAAC_JOINTS = [
    "joint_R1", "joint_L1", "joint_R2", "joint_L2", "joint_R3", "joint_L3",
    "joint_R4", "joint_L4", "joint_R5", "joint_L5", "joint_R6", "joint_L6",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=DEFAULT_FILE)
    ap.add_argument("--speed", type=float, default=1.0, help="回放速度倍率")
    ap.add_argument("--loop", action="store_true", help="循环回放")
    ap.add_argument("--headless", action="store_true", help="不弹窗, 只打印数值自检")
    ap.add_argument("--base_z", type=float, default=1.0, help="base 悬浮高度(仅显示用)")
    args = ap.parse_args()

    d = json.load(open(args.file))
    F = np.array(d["Frames"], dtype=np.float64)
    dt = float(d["FrameDuration"])
    T = F.shape[0]
    jpos = F[:, 0:12]
    feet = F[:, 24:30]
    print(f"[load] {args.file}: {T} 帧, dt={dt}, dim={F.shape[1]}")

    model = mujoco.MjModel.from_xml_path(NLEGS_XML)
    data = mujoco.MjData(model)
    qadr = {}
    for jn in ISAAC_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        qadr[jn] = model.jnt_qposadr[jid]
    foot_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Link_R6")
    foot_l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Link_L6")

    def set_frame(t: int):
        data.qpos[:] = 0.0
        data.qpos[2] = args.base_z  # 悬浮
        data.qpos[3] = 1.0          # base 单位朝向
        for jn, k in zip(ISAAC_JOINTS, range(12)):
            data.qpos[qadr[jn]] = jpos[t, k]
        mujoco.mj_kinematics(model, data)

    # 数值自检: FK 复算脚位置 vs 文件里存的脚位置(应一致, 验证文件自洽)
    set_frame(0)
    fk_r = data.xpos[foot_r] - data.qpos[:3]
    fk_l = data.xpos[foot_l] - data.qpos[:3]
    print(f"[check] 首帧 FK 右脚={fk_r.round(3)} 文件存={feet[0,:3].round(3)}")
    print(f"[check] 首帧 FK 左脚={fk_l.round(3)} 文件存={feet[0,3:].round(3)}")
    err = np.abs(np.concatenate([fk_r, fk_l]) - feet[0]).max()
    print(f"[check] 脚位置最大误差={err:.4f}m (应 ~0, 验证 base 系一致)")

    if args.headless:
        print("[headless] 跳过可视化。")
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t = 0
        while viewer.is_running():
            step_start = time.time()
            set_frame(t)
            viewer.sync()
            t += 1
            if t >= T:
                if not args.loop:
                    break
                t = 0
            dt_wall = dt / max(args.speed, 1e-3)
            elapsed = time.time() - step_start
            if dt_wall - elapsed > 0:
                time.sleep(dt_wall - elapsed)


if __name__ == "__main__":
    main()
