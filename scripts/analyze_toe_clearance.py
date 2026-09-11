#!/usr/bin/env python3
"""对比 nlegs_rough 两次训练的抬脚高度（平地 MuJoCo headless 回放，一次性分析脚本）。

脚尖最低点 = Link_*6 体系 (0.14, 0, -0.0135)：脚部 collision capsule 前端 (x=0.14,
z=-0.0015) 的底缘 (半径 0.012)。脚跟最低点 = (-0.04, 0, -0.0135)。
平地 z=0，世界 z 即离地高度。按"脚尖+脚跟同时离地>5mm"切摆动段，报告每步脚尖峰值、
以及"脚最低点(min(尖,跟))峰值"——后者才是真正能跨过的台阶上限。

用法: python scripts/analyze_toe_clearance.py [--runs RUN1 RUN2] [--duration 20] [--vx 0.5]
"""

import argparse
import importlib.util
import os

import mujoco
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)


def _load_flat_sim2sim():
    path = os.path.join(_REPO, "source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/task/flat/sim2sim.py")
    spec = importlib.util.spec_from_file_location("nlegs_flat_sim2sim", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOE_LOCAL = np.array([0.14, 0.0, -0.0135])
HEEL_LOCAL = np.array([-0.04, 0.0, -0.0135])
AIR_THRESH = 0.005   # 离地判定 (m)
MIN_SWING_S = 0.08   # 短于此的"腾空"当抖动丢弃
SETTLE_S = 2.0       # 起步沉降期不统计


def rollout(flat, run, duration, vx):
    cfg = flat.load_config(run)
    runner = flat.MujocoRunner(cfg, show_viewer=False)
    runner.command[:] = [vx, 0.0, 0.0]
    feet = {}
    for side in ("L", "R"):
        bid = mujoco.mj_name2id(runner.model, mujoco.mjtObj.mjOBJ_BODY, f"Link_{side}6")
        assert bid >= 0, f"场景里找不到 Link_{side}6"
        feet[side] = bid
    steps = int(duration / cfg.step_dt)
    log = {s: {"toe": [], "heel": []} for s in feet}
    t_log, vx_log = [], []
    fell = False
    for _ in range(steps):
        action = runner._infer()
        target_sdk = runner._target_sdk(action)
        for _ in range(cfg.decimation):
            delayed = runner.latency.process(target_sdk)
            runner.data.ctrl[runner.actuator_ids] = runner._torque(delayed)
            mujoco.mj_step(runner.model, runner.data)
        runner.episode_step += 1
        t_log.append(runner.data.time)
        vx_log.append(runner.data.qvel[runner.base_dof_adr])
        for side, bid in feet.items():
            R = runner.data.xmat[bid].reshape(3, 3)
            p = runner.data.xpos[bid]
            log[side]["toe"].append((p + R @ TOE_LOCAL)[2])
            log[side]["heel"].append((p + R @ HEEL_LOCAL)[2])
        if runner.data.xpos[runner.base_body_id, 2] < 0.3:
            fell = True
            break
    return cfg, np.array(t_log), np.array(vx_log), {
        s: {k: np.array(v) for k, v in d.items()} for s, d in log.items()
    }, fell


def swing_stats(t, toe, heel, step_dt):
    """按双点离地切摆动段, 返回每步 (脚尖峰值, 脚最低点峰值, 摆动时长)。"""
    airborne = (toe > AIR_THRESH) & (heel > AIR_THRESH) & (t > SETTLE_S)
    stats = []
    i = 0
    n = len(t)
    while i < n:
        if not airborne[i]:
            i += 1
            continue
        j = i
        while j < n and airborne[j]:
            j += 1
        dur = (j - i) * step_dt
        if dur >= MIN_SWING_S:
            seg_toe = toe[i:j]
            seg_low = np.minimum(toe[i:j], heel[i:j])
            stats.append((seg_toe.max(), seg_low.max(), dur))
        i = j
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["2026-08-26_20-37-55", "2026-08-27_19-45-40"])
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--vx", type=float, default=0.5)
    args = parser.parse_args()

    flat = _load_flat_sim2sim()
    flat.LOGS_ROOT = os.path.join(_REPO, "logs", "rsl_rl", "nlegs_rough")

    for run in args.runs:
        cfg, t, vx, log, fell = rollout(flat, run, args.duration, args.vx)
        print(f"\n================ {run} (cmd vx={args.vx}, {t[-1]:.1f}s{', 中途摔倒!' if fell else ''}) ================")
        print(f"  平均前进速度(沉降后): {vx[t > SETTLE_S].mean():+.3f} m/s")
        for side in ("L", "R"):
            stats = swing_stats(t, log[side]["toe"], log[side]["heel"], cfg.step_dt)
            if not stats:
                print(f"  {side}: 无有效摆动段")
                continue
            toe_pk = np.array([s[0] for s in stats])
            low_pk = np.array([s[1] for s in stats])
            durs = np.array([s[2] for s in stats])
            print(f"  {side} 摆动 {len(stats)} 步, 平均腾空 {durs.mean()*1000:.0f}ms")
            print(f"     脚尖峰值      : 均值 {toe_pk.mean()*100:5.1f}cm  中位 {np.median(toe_pk)*100:5.1f}cm  "
                  f"p10 {np.percentile(toe_pk, 10)*100:5.1f}cm  max {toe_pk.max()*100:5.1f}cm")
            print(f"     脚最低点峰值  : 均值 {low_pk.mean()*100:5.1f}cm  中位 {np.median(low_pk)*100:5.1f}cm  "
                  f"p10 {np.percentile(low_pk, 10)*100:5.1f}cm  max {low_pk.max()*100:5.1f}cm")


if __name__ == "__main__":
    main()
