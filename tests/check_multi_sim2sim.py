"""CPU/MuJoCo 检查统一三任务回放，不打开窗口、不启动 ROS。

python tests/check_multi_sim2sim.py --crouch-run 新下蹲目录 [--interface-only]
"""

import argparse
import numpy as np
import glfw
import mujoco
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sim2sim import MultiTaskSim, MotionRunner, STABLE_TIME, HANDOVER_TIME, END_TIMEOUT, STOP_TIMEOUT, flat


def advance(sim, seconds):
    for _ in range(round(seconds / sim.dt)):
        sim.step()
        assert np.isfinite(sim.data.qpos).all() and np.isfinite(sim.robot.last_torque).all()
        assert np.all(sim.target >= sim.limits[:, 0] - 1e-6)
        assert np.all(sim.target <= sim.limits[:, 1] + 1e-6)


def until(sim, state, timeout=10.):
    for _ in range(round(timeout / sim.dt)):
        if sim.state == state:
            return
        assert sim.state != "stopped", sim.reason
        advance(sim, sim.dt)
    raise AssertionError(f"未到 {state}: {sim.state}, {sim.reason}; loads={sim.foot_loads()}")


def start(sim, task):
    before = sim.data.qpos.copy(), sim.data.qvel.copy(), sim.data.time
    delay = list(sim.robot.latency.buffer)
    assert sim.request(task)
    assert np.array_equal(sim.data.qpos, before[0])
    assert np.array_equal(sim.data.qvel, before[1]) and sim.data.time == before[2]
    assert all(np.array_equal(a, b) for a, b in zip(delay, sim.robot.latency.buffer))
    assert sim.runners[task].episode_step == 0
    assert sim.runners[task].history.ready
    for buffer in sim.runners[task].history.buffers.values():
        assert np.all(buffer == buffer[0])
    target = sim.target.copy()
    advance(sim, sim.dt)
    assert np.allclose(sim.target, target)  # 接管第一拍保持连续，参考暂不推进。
    advance(sim, HANDOVER_TIME - sim.dt)
    assert sim.runners[task].episode_step == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for task in ("walk", "crouch", "rise"):
        parser.add_argument(f"--{task}-run")
    parser.add_argument("--interface-only", action="store_true", help="只检查加载/数据衔接/按键接口，不宣称策略动作成功")
    args = parser.parse_args()
    runs = {task: getattr(args, task + "_run") for task in ("walk", "crouch", "rise")}
    np.random.seed(42)
    sim = MultiTaskSim(runs)
    assert len({id(r.data) for r in sim.runners.values()}) == 1
    assert len({id(r.model) for r in sim.runners.values()}) == 1
    assert len({id(r.latency) for r in sim.runners.values()}) == 1
    assert not sim.request("crouch")  # 尚未连续稳定
    until(sim, "stand_ready")
    assert sim.data.time >= STABLE_TIME
    assert not sim.request("rise")
    if args.interface_only:
        before = sim.data.qpos.copy()
        assert sim.request("crouch")
        assert np.array_equal(sim.data.qpos, before) and sim.state == "lowering"
        assert sim.request("halt")
        crouched = MultiTaskSim(runs, initial_pose="crouch")
        assert np.allclose(crouched.data.qpos[crouched.robot.qpos_adr], crouched.pose_q["crouch"])
        assert sim.pose_error("stand") is None
        print("PASS: three-policy loading, new endpoints/limits, stand/crouch initial poses, manual start/halt; no policy success claim")
        return
    start(sim, "crouch")
    assert not sim.request("crouch") and not sim.request("rise") and not sim.request("walk")
    until(sim, "crouch_ready")
    assert sim.active == "crouch" and sim.runners["crouch"].reference_frame == len(sim.runners["crouch"].ref_pos) - 1
    advance(sim, .5)
    assert not sim.request("walk")
    start(sim, "rise")
    until(sim, "stand_ready")
    assert sim.active == "rise" and sim.runners["rise"].reference_frame == len(sim.runners["rise"].ref_pos) - 1
    start(sim, "walk")
    assert not sim.request("crouch")
    sim.keyboard_command[:] = [.15, 0., .15]
    advance(sim, 2.)
    assert sim.request("stop")
    assert not sim.keyboard_command.any()
    until(sim, "stand_ready")
    assert any(t[2] == "standing_transition" for t in sim.transitions)
    actual_yaw = flat.yaw_from_quat(sim.data.qpos[3:7])
    start(sim, "crouch")
    motion = sim.runners["crouch"]
    assert isinstance(motion, MotionRunner)
    assert np.isclose(np.arctan2(motion.yaw_alignment[1, 0], motion.yaw_alignment[0, 0]), actual_yaw)
    until(sim, "crouch_ready")
    start(sim, "rise")
    until(sim, "stand_ready")
    print(f"PASS 连续真实策略循环: t={sim.data.time:.2f}s, base_z={sim.data.qpos[2]:.3f}m")

    # 停止最高优先级；长按组合键只产生一次请求，不与单键混用。
    buttons, axes = [0] * 15, [0.] * 6
    buttons[glfw.GAMEPAD_BUTTON_A] = buttons[glfw.GAMEPAD_BUTTON_LEFT_BUMPER] = 1
    sim.gamepad_input(buttons, axes)
    sim.process_keys()
    assert sim.state == "walking"
    advance(sim, .10)
    step = sim.runners["walk"].episode_step
    sim.gamepad_input(buttons, axes)
    sim.process_keys()
    assert sim.runners["walk"].episode_step == step
    sim.on_key(ord("2"))
    sim.on_key(ord("P"))
    sim.process_keys()
    assert sim.state == "stopped" and sim.active is None
    target, time_before = sim.target.copy(), sim.data.time
    advance(sim, .10)
    assert np.array_equal(sim.target, target) and sim.data.time > time_before
    assert not sim.request("walk")
    sim.gamepad_input(None, None)
    assert not sim.gamepad_command.any()

    np.random.seed(42)
    squatting = MultiTaskSim(runs, initial_pose="crouch")
    until(squatting, "crouch_ready")
    # 双脚悬空不可验收，单靠角度相同不够。
    original = squatting.data.qpos.copy()
    squatting.data.qpos[2] += .2
    mujoco.mj_forward(squatting.model, squatting.data)
    assert squatting.pose_error("crouch") == "双脚尚未同时承重"
    squatting.data.qpos[:] = original
    mujoco.mj_forward(squatting.model, squatting.data)
    squatting.request("halt")
    assert squatting.request("reset") and squatting.state == "checking"
    assert not squatting.request("rise")
    until(squatting, "crouch_ready")
    start(squatting, "rise")
    until(squatting, "stand_ready")
    old_task, old_target = squatting.active, squatting.target.copy()
    with patch.object(squatting, "_policy_target", return_value=squatting.limits[:, 1].copy()):
        assert not squatting.request("crouch")
    assert squatting.active == old_task and np.array_equal(squatting.target, old_target)
    # 参考到末帧，但实测不到位：不能宣布就绪。
    squatting.state = "rising"
    runner = squatting.runners["rise"]
    runner.episode_step = int(np.ceil((len(runner.ref_pos) - 1) / runner.frame_stride))
    with patch.object(squatting, "pose_error", return_value="测试：未到位"):
        squatting._update_state()
        assert squatting.state == "rising"
        runner.episode_step += round(END_TIMEOUT / squatting.dt) + 1
        squatting._update_state()
        assert squatting.state == "stopped"
    # 没有双脚承重窗口：停步超时锁存，不强行进入站姿。
    assert squatting.request("reset")
    until(squatting, "stand_ready")
    start(squatting, "walk")
    assert squatting.request("stop")
    squatting.state_since = squatting.data.time - STOP_TIMEOUT - squatting.dt
    with patch.object(squatting, "foot_loads", return_value=np.zeros(2)):
        squatting._update_state()
    assert squatting.state == "stopped"
    # 数字任务键不能同时切掉 MuJoCo 中机器人的可见几何体。
    sim.visible_geom_groups = np.array([1, 1, 1, 0, 0, 0])
    fake_viewer = SimpleNamespace(opt=SimpleNamespace(geomgroup=np.zeros(6)), lock=nullcontext)
    with patch.object(sim, "viewer", fake_viewer):
        sim.on_key(glfw.KEY_KP_1)
        sim.process_keys()
    assert np.array_equal(fake_viewer.opt.geomgroup, sim.visible_geom_groups)
    print("PASS 起点验收/无物理重置/独立观测历史/末帧保持/yaw 对齐/限位/停止锁存/手柄边沿")
    print("仅验证仿真；不表示实机已具备同样的接地判断和切换能力。")


if __name__ == "__main__":
    main()
