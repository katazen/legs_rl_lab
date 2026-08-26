"""在 nlegs MuJoCo 上回放 EngineAI 格式 npz(convert_pm01_motion.py 的输出), 可视 + 数值校验重定向。

校验点(肉眼 + 数值):
  - 步态像不像"走路": 左右腿交替摆动/支撑(脚高度反相)
  - 左右腿有没有搞反
  - 膝/髋roll/髋yaw 符号方向对不对(转换里唯一残余风险)
  - FK 复算的 body 位姿是否与 npz 里存的 body_pos_w 自洽

回放只驱动关节角(base 悬浮), 不做物理, 纯运动学。SSH 无显示器时加 --headless 只跑数值自检。

用法:
    python scripts/amp_engineai/replay_npz_motion.py [--file <nlegs_locomotion.npz>] \
        [--speed 1.0] [--loop] [--headless]
"""

from __future__ import annotations

import argparse
import os
import time

import mujoco
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_THIS))
_ASSETS = os.path.join(_REPO, "source", "legs_rl_lab", "legs_rl_lab", "assets")
NLEGS_XML = os.path.join(_ASSETS, "legs_narrow", "mjcf", "legs_narrow.xml")
DEFAULT_FILE = os.path.join(
    _REPO,
    "source/legs_rl_lab/legs_rl_lab/tasks/amp_task/datasets/motion_amp_engineai/nlegs_locomotion.npz",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=DEFAULT_FILE)
    ap.add_argument("--speed", type=float, default=1.0, help="回放速度倍率")
    ap.add_argument("--loop", action="store_true", help="循环回放")
    ap.add_argument("--headless", action="store_true", help="不弹窗, 只跑数值自检")
    ap.add_argument("--base_z", type=float, default=1.0, help="base 悬浮高度(仅显示用)")
    args = ap.parse_args()

    d = np.load(args.file)
    joint_names = [str(x) for x in d["joint_names"]]
    body_names = [str(x) for x in d["body_names"]]
    jpos = d["joint_pos"].astype(np.float64)   # (N,12)
    body_pos_w = d["body_pos_w"].astype(np.float64)  # (N,B,3)
    fps = float(np.asarray(d["fps"]).reshape(-1)[0])
    dt = 1.0 / fps
    T = jpos.shape[0]
    print(f"[load] {args.file}: {T} 帧, fps={fps}, joints={len(joint_names)}, bodies={len(body_names)}")

    model = mujoco.MjModel.from_xml_path(NLEGS_XML)
    data = mujoco.MjData(model)
    qadr = {jn: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)] for jn in joint_names}
    foot_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Link_R6")
    foot_l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Link_L6")
    ir = body_names.index("Link_R6")
    il = body_names.index("Link_L6")

    def set_frame(t: int, floated: bool):
        data.qpos[:] = 0.0
        if floated:
            data.qpos[2] = args.base_z
            data.qpos[3] = 1.0
        for k, jn in enumerate(joint_names):
            data.qpos[qadr[jn]] = jpos[t, k]
        mujoco.mj_kinematics(model, data)

    # ---- 数值自检 ----
    # 1) FK 自洽: 用 npz 里存的 base 位姿驱动, FK 脚位置应与 npz 存的 body_pos_w 脚位置一致
    max_fk_err = 0.0
    foot_r_zrel = np.zeros(T)
    foot_l_zrel = np.zeros(T)
    for t in range(T):
        data.qpos[:] = 0.0
        data.qpos[0:3] = body_pos_w[t, 0]
        data.qpos[3:7] = d["body_quat_w"][t, 0].astype(np.float64)
        for k, jn in enumerate(joint_names):
            data.qpos[qadr[jn]] = jpos[t, k]
        mujoco.mj_kinematics(model, data)
        max_fk_err = max(max_fk_err,
                         np.abs(data.xpos[foot_r] - body_pos_w[t, ir]).max(),
                         np.abs(data.xpos[foot_l] - body_pos_w[t, il]).max())
        foot_r_zrel[t] = data.xpos[foot_r][2] - body_pos_w[t, 0][2]
        foot_l_zrel[t] = data.xpos[foot_l][2] - body_pos_w[t, 0][2]

    print(f"[check] FK 脚位置 vs npz body_pos_w 最大误差 = {max_fk_err:.5f} m (应 ~0)")
    print(f"[check] 右脚相对 base 高度: 均值={foot_r_zrel.mean():.3f} 最高={foot_r_zrel.max():.3f} "
          f"(应恒为负, 脚在 base 下方约 -0.5m)")
    print(f"[check] 左脚相对 base 高度: 均值={foot_l_zrel.mean():.3f} 最高={foot_l_zrel.max():.3f}")
    if foot_r_zrel.max() > 0 or foot_l_zrel.max() > 0:
        print("[warn] 有脚跑到 base 上方 -> 膝/髋符号很可能翻了, 检查 --flip")
    # 2) 步态交替: 左右脚高度应反相(负相关)
    corr = np.corrcoef(foot_r_zrel, foot_l_zrel)[0, 1]
    print(f"[check] 左右脚高度相关系数 = {corr:.3f} (走路应显著<0, 交替支撑)")
    # 3) 前进: base 水平位移
    disp = np.linalg.norm(body_pos_w[-1, 0, :2] - body_pos_w[0, 0, :2])
    print(f"[check] base 水平总位移 = {disp:.3f} m")

    if args.headless:
        print("[headless] 跳过可视化。")
        return

    import mujoco.viewer as mj_viewer  # 仅非 headless 才需要 GL
    with mj_viewer.launch_passive(model, data) as viewer:
        t = 0
        while viewer.is_running():
            step_start = time.time()
            set_frame(t, floated=True)
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
