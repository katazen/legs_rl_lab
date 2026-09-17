"""三任务共用一个 MuJoCo 世界；不启动 ROS、不连接电机。

python scripts/sim2sim.py [--initial-pose stand|crouch] [--gamepad 0]
1/LB+A 走路，2/LB+X 下蹲，3/LB+Y 起身，0/Start 正常停步。
P/B 中断锁存，R/Back 重新验收（不重置物理、不自动运行）。
W/S 前后，A/D 左右，Q/E 转向；空格清零速度，不等于停步。
数字支持主键盘和小键盘；聚焦 MuJoCo 窗口操作。
--walk-run、--crouch-run、--rise-run 接受目录名或绝对路径。
"""

import argparse
from dataclasses import fields
import importlib.util
from pathlib import Path
from queue import SimpleQueue
import threading
import time

import glfw
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "multi_mimic", ROOT / "source/legs_rl_lab/legs_rl_lab/tasks/mimic_task/task/nlegs_crouch/sim2sim.py",
)
mimic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mimic)
flat = mimic.flat
RUNS = {"walk": ("nlegs_flat_static", "2026-09-16_11-41-05"),
        "crouch": ("nlegs_mimic_crouch", "2026-09-15_15-36-16"),
        "rise": ("nlegs_mimic_stand", "2026-09-15_18-50-29")}
STABLE_TIME = .30
STOP_BLEND_TIME = .50
STOP_TIMEOUT = 6.
END_TIMEOUT = 4.
HANDOVER_MAX_DELTA = .60
HANDOVER_TIME = .20
TASK_KEYS = {"1": "walk", "2": "crouch", "3": "rise", "0": "stop", "P": "halt", "R": "reset"}
VELOCITY_KEYS = {"W": (0, .05), "S": (0, -.05), "A": (1, .05),
                 "D": (1, -.05), "Q": (2, .05), "E": (2, -.05)}


MotionRunner = mimic.MimicCrouchRunner


class MultiTaskSim:
    def __init__(self, runs=None, initial_pose="stand", base_mass_add=2.5):
        if initial_pose not in ("stand", "crouch"):
            raise ValueError("initial_pose 必须是 stand 或 crouch")
        self.runners = {}
        for name, (task, default) in RUNS.items():
            path = Path((runs or {}).get(name) or default).expanduser()
            if not path.is_absolute():
                path = ROOT / "logs/rsl_rl" / task / path
            cfg = flat.load_config(str(path))
            if name == "walk":
                if set(cfg.commands) != {"base_velocity"}:
                    raise ValueError("走路模型必须使用 base_velocity 命令")
                runner = flat.MujocoRunner(cfg, show_viewer=False)
            else:
                runner = MotionRunner(cfg, show_viewer=False, base_mass_add=base_mass_add)
            self.runners[name] = runner
            print(f"[{name}] {cfg.model_path}")
        self.robot = self.runners["crouch"]
        self._check_compatibility()
        start = self.runners["rise"] if initial_pose == "crouch" else self.robot
        initial_q, initial_v = start.data.qpos.copy(), start.data.qvel.copy()
        self.model, self.data = self.robot.model, self.robot.data
        for runner in self.runners.values():
            runner.model, runner.data = self.model, self.data
            runner.latency = self.robot.latency
        self.data.qpos[:], self.data.qvel[:] = initial_q, initial_v
        mujoco.mj_forward(self.model, self.data)
        self.dt = self.robot.cfg.step_dt
        self.limits = self.model.jnt_range[self.robot.joint_ids].copy()
        self.feet = [self.model.body(f"Link_{side}6").id for side in ("L", "R")]
        self.pose_q = {"stand": self._reference_sdk("crouch", 0), "crouch": self._reference_sdk("rise", 0)}
        self.pose_rot = {"stand": mimic.rotation_matrix(self.robot.ref_quat[0]),
                         "crouch": mimic.rotation_matrix(self.runners["rise"].ref_quat[0])}
        self.target = self.data.qpos[self.robot.qpos_adr].astype(np.float32).copy()
        self.active = None
        self.pending_target = None
        self.state = "checking"
        self.state_since = self.data.time
        self.stable_since = None
        self.stable_pose = None
        self.reason = "等待落地并稳定"
        self.keyboard_command = np.zeros(3, np.float32)
        self.gamepad_command = np.zeros(3, np.float32)
        self.command = np.zeros(3, np.float32)
        self.events = SimpleQueue()
        self.prev_buttons = [0] * 15
        self.gamepad_connected = False
        self.viewer = None
        self.transitions = []
        print("[multi] 使用 nlegs_limit 场景；所有任务目标均受该场景限位约束。切换不重置机器人。")

    def _check_compatibility(self):
        cfg = self.robot.cfg
        for name, runner in self.runners.items():
            other = runner.cfg
            for field in ("joint_names", "physics_dt", "step_dt", "stiffness", "damping", "armature",
                          "friction", "dynamic_friction", "viscous_friction", "default_sdk"):
                if not np.array_equal(getattr(cfg, field), getattr(other, field)):
                    raise ValueError(f"{name}.{field} 不同；统一世界当前要求相同的物理参数与默认站姿")
            if len(cfg.actuators) != len(other.actuators):
                raise ValueError(f"{name} 执行器配置不同")
            groups = {tuple(group.joint_ids): group for group in other.actuators}
            for a in cfg.actuators:
                b = groups.get(tuple(a.joint_ids))
                if b is None:
                    raise ValueError(f"{name} 执行器关节分组不同")
                for field in fields(a):
                    if field.name != "name" and not np.array_equal(getattr(a, field.name), getattr(b, field.name)):
                        raise ValueError(f"{name} 执行器 {a.name}.{field.name} 不同")
        down, up = self.runners["crouch"], self.runners["rise"]
        for a, fa, b, fb in (("crouch", 0, "rise", -1), ("crouch", -1, "rise", 0)):
            if not np.allclose(self._reference_sdk(a, fa), self._reference_sdk(b, fb), atol=1e-5):
                raise ValueError("下蹲、起身参考的端点不衔接")
        if not np.allclose(self._reference_sdk("crouch", 0), cfg.default_sdk, atol=1e-5):
            raise ValueError("参考站姿与走路默认站姿不一致")
        for qa, qb in ((down.ref_quat[0], up.ref_quat[-1]), (down.ref_quat[-1], up.ref_quat[0])):
            if not np.allclose(mimic.rotation_matrix(qa), mimic.rotation_matrix(qb), atol=1e-5):
                raise ValueError("下蹲、起身参考的机身姿态不衔接")

    def _reference_sdk(self, task, frame):
        runner = self.runners[task]
        result = np.empty(12, np.float32)
        result[runner.cfg.policy_to_sdk] = runner.ref_pos[frame]
        return result

    def _change_state(self, state, reason):
        previous = self.state
        self.state, self.reason = state, reason
        self.state_since = self.data.time
        self.stable_since = self.stable_pose = None
        self.transitions.append((float(self.data.time), previous, state, reason))
        print(f"[multi {self.data.time:.2f}s] {previous} -> {state}: {reason}")

    def _clear_commands(self):
        self.keyboard_command[:] = self.gamepad_command[:] = self.command[:] = 0.
        for runner in self.runners.values():
            runner.command[:] = 0.

    def foot_loads(self):
        loads = np.zeros(2)
        force = np.empty(6)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            bodies = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if 0 not in bodies:
                continue
            mujoco.mj_contactForce(self.model, self.data, i, force)
            for side, body in enumerate(self.feet):
                if body in bodies:
                    loads[side] += max(0., force[0])
        return loads

    def pose_error(self, pose, stopping=False):
        q, qd = self.robot._joint_state()
        # 停步寻找动态落脚窗口；完成验收用更严格的静态阈值，不能混为一谈。
        tolerance = np.full(12, .45 if stopping else .12)
        tolerance[[self.robot.cfg.short_joint_names.index(n) for n in ("L6", "R6")]] = .20
        error = np.abs(q - self.pose_q[pose])
        if np.any(error > tolerance):
            i = int(np.argmax(error / tolerance))
            return f"{self.robot.cfg.short_joint_names[i]} 姿态误差 {error[i]:.3f}rad"
        rot = mimic.rotation_matrix(self.data.qpos[3:7])
        tilt = np.arccos(np.clip(np.dot(rot[2], self.pose_rot[pose][2]), -1., 1.))
        if tilt > (.20 if stopping else .15):
            return f"机身倾斜误差 {np.degrees(tilt):.1f}deg"
        if np.max(np.abs(qd)) > (5. if stopping else .35):
            return "关节尚未稳定"
        if np.linalg.norm(self.data.qvel[3:6]) > (.8 if stopping else .4):
            return "机身角速度过大"
        if np.linalg.norm(self.data.qvel[:3]) > (.30 if stopping else .10):
            return "机身仍在移动"
        if np.any(self.foot_loads() < 2.):
            return "双脚尚未同时承重"
        return None

    def _stable(self, pose):
        reason = self.pose_error(pose)
        if reason is not None:
            self.stable_since = self.stable_pose = None
            self.reason = reason
            return False
        if self.stable_pose != pose:
            self.stable_pose, self.stable_since = pose, self.data.time
        return self.data.time - self.stable_since >= STABLE_TIME

    def _halt(self, reason):
        self.pending_target = None
        self.active = None
        self._clear_commands()
        self._change_state("stopped", reason + "；保持最后目标，R 只重新验收，不复位物理")

    def request(self, task):
        if task == "halt":
            if self.state != "stopped":
                self._halt("操作员中断")
            return True
        if task == "reset":
            if self.state != "stopped":
                return self._reject("只有停止锁存后才需要 R；不会自动回站立")
            pose = next((p for p in self.pose_q if self.pose_error(p) is None), None)
            if pose is None:
                return self._reject("既不满足站姿也不满足蹲姿，需人工恢复或重开仿真")
            self._change_state("checking", "重新连续验收当前姿态；仍需手动启动任务")
            return True
        if task == "stop":
            if self.state != "walking":
                return self._reject("正常停步只用于 walking；中断动作请按 P/B")
            previous_command = self.command.copy()
            self._clear_commands()
            self.command[:] = previous_command
            self._change_state("stopping", "零速行走，等待双脚承重的站立接管窗口")
            return True
        required = {"walk": "stand_ready", "crouch": "stand_ready", "rise": "crouch_ready"}
        if task not in required or self.state != required[task]:
            return self._reject(f"{task} 不允许从 {self.state} 启动；不排队、不重播")
        if self.active == task:
            return self._reject("当前策略仍在保持，不重播；需要时先中断再重新验收")
        pose = "crouch" if task == "rise" else "stand"
        if (reason := self.pose_error(pose)) is not None:
            return self._reject(reason)
        runner = self.runners[task]
        runner.episode_step = 0
        runner.last_action[:] = 0.
        runner.history = flat.TermGroupedHistory(runner.cfg.observations)
        if task != "walk":
            yaw = flat.yaw_from_quat(self.data.qpos[3:7]) - flat.yaw_from_quat(runner.ref_quat[0])
            c, s = np.cos(yaw), np.sin(yaw)
            runner.yaw_alignment = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
            runner.end_announced = False
        runner.command[:] = 0.
        try:
            target = self._policy_target(runner)
        except Exception as exc:
            return self._reject(f"新策略预检失败: {exc}")
        delta = float(np.max(np.abs(target - self.target)))
        if delta > HANDOVER_MAX_DELTA:
            return self._reject(f"首拍目标跳变 {delta:.3f}rad，超过 {HANDOVER_MAX_DELTA}rad")
        self._clear_commands()
        self.active, self.pending_target = task, target
        self.handover_from = self.target.copy()
        self.handover_since = self.data.time
        self._change_state({"walk": "walking", "crouch": "lowering", "rise": "rising"}[task],
                           f"{HANDOVER_TIME:g}s 平滑接管；只重置策略历史/时钟，不重置机器人或延迟")
        return True

    def _reject(self, reason):
        print(f"[拒绝] {reason}")
        return False

    def _policy_target(self, runner):
        target = runner._target_sdk(runner._infer())
        return np.clip(target, self.limits[:, 0], self.limits[:, 1]).astype(np.float32)

    def _update_state(self):
        if self.state in ("checking", "stand_ready", "crouch_ready"):
            poses = [{"crouch": "crouch", "rise": "stand"}[self.active]] if self.active in ("crouch", "rise") else self.pose_q
            pose = next((p for p in poses if self.pose_error(p) is None), None)
            if pose is None:
                if self.state != "checking":
                    self._change_state("checking", "姿态不再满足就绪条件")
                self.stable_since = self.stable_pose = None
            elif self.state != pose + "_ready" and self._stable(pose):
                self._change_state(pose + "_ready", "实测姿态连续稳定；等待任务按键")
        elif self.state in ("lowering", "rising"):
            runner = self.runners[self.active]
            end_step = int(np.ceil((len(runner.ref_pos) - 1) / runner.frame_stride))
            if runner.episode_step >= end_step:
                pose = "crouch" if self.state == "lowering" else "stand"
                if self._stable(pose):
                    self._change_state(pose + "_ready", "参考结束且实测到位；继续末帧策略保持")
                elif (runner.episode_step - end_step) * self.dt > END_TIMEOUT:
                    self._halt("参考结束后仍未到位: " + self.reason)
        elif self.state == "stopping":
            reason = self.pose_error("stand", stopping=True)
            if reason is not None:
                self.reason = reason
            elif np.linalg.norm(self.command) < 1e-6:
                self.blend_start = self.target.copy()
                self.active = None
                self._change_state("standing_transition", "双脚承重；短程平滑接管站立目标")
            if self.state == "stopping" and self.data.time - self.state_since > STOP_TIMEOUT:
                self._halt("停步超时，未等到合适接管窗口: " + self.reason)
        elif self.state == "standing_transition":
            elapsed = self.data.time - self.state_since
            if elapsed >= STOP_BLEND_TIME and self._stable("stand"):
                self._change_state("stand_ready", "停步完成；等待走路/下蹲按键")
            elif elapsed > STOP_BLEND_TIME + END_TIMEOUT:
                self._halt("站立接管后未稳定: " + self.reason)

    def step(self):
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("仿真状态非有限数，终止仿真")
        rot = mimic.rotation_matrix(self.data.qpos[3:7])
        if self.state != "stopped" and (rot[2, 2] < np.cos(.8) or self.data.qpos[2] < .30):
            self._halt("机身倾斜/高度异常")
        self._update_state()
        runner = self.runners[self.active] if self.active else self.robot
        reference_running = self.active and self.data.time - self.handover_since >= HANDOVER_TIME - 1e-9
        if self.active:
            desired = self.keyboard_command + self.gamepad_command if self.state == "walking" else np.zeros(3)
            desired = np.clip(desired, runner.cfg.command_ranges[:, 0], runner.cfg.command_ranges[:, 1])
            self.command += np.clip(desired - self.command, -np.array([.5, .5, 1.]) * self.dt,
                                    np.array([.5, .5, 1.]) * self.dt)
            runner.command[:] = self.command
            try:
                if self.pending_target is not None:
                    target, self.pending_target = self.pending_target, None
                else:
                    target = self._policy_target(runner)
                a = np.clip((self.data.time - self.handover_since) / HANDOVER_TIME, 0., 1.)
                weight = a*a*(3-2*a)
                self.target = (1 - weight) * self.handover_from + weight * target
            except Exception as exc:
                self._halt(f"策略异常: {exc}")
        elif self.state == "standing_transition":
            a = np.clip((self.data.time - self.state_since) / STOP_BLEND_TIME, 0., 1.)
            self.target = (1 - a*a*(3-2*a)) * self.blend_start + a*a*(3-2*a) * self.pose_q["stand"]
        self.target = np.clip(self.target, self.limits[:, 0], self.limits[:, 1]).astype(np.float32)
        for _ in range(self.robot.cfg.decimation):
            delayed = self.robot.latency.process(self.target)
            self.robot.last_torque[:] = runner._torque(delayed)
            self.data.ctrl[self.robot.actuator_ids] = self.robot.last_torque
            mujoco.mj_step(self.model, self.data)
        if self.active and reference_running:
            runner.episode_step += 1

    def on_key(self, key):
        if glfw.KEY_KP_0 <= key <= glfw.KEY_KP_9:
            key = ord("0") + key - glfw.KEY_KP_0
        if 0 <= key < 128:
            self.events.put(chr(key).upper())

    def process_keys(self):
        keys = []
        while not self.events.empty():
            keys.append(self.events.get())
        if self.viewer is not None and any(key in ("0", "1", "2", "3") for key in keys):
            # MuJoCo 也用数字键切换几何体显示；任务按键不应让机器人消失。
            with self.viewer.lock():
                self.viewer.opt.geomgroup[:] = self.visible_geom_groups
        if "P" in keys:
            self.request("halt")
            return
        for key in keys:
            if key in TASK_KEYS:
                self.request(TASK_KEYS[key])
            elif key == " ":
                self._clear_commands()
            elif key in VELOCITY_KEYS and self.state == "walking":
                axis, delta = VELOCITY_KEYS[key]
                limits = self.runners["walk"].cfg.command_ranges
                self.keyboard_command[axis] = np.clip(self.keyboard_command[axis] + delta, *limits[axis])

    def gamepad_input(self, buttons, axes):
        if buttons is None:
            self.prev_buttons = [0] * 15
            self.gamepad_command[:] = 0.
            return
        pressed = lambda button: buttons[button] and not self.prev_buttons[button]
        if pressed(glfw.GAMEPAD_BUTTON_B):
            self.events.put("P")
        elif pressed(glfw.GAMEPAD_BUTTON_BACK):
            self.events.put("R")
        elif pressed(glfw.GAMEPAD_BUTTON_START):
            self.events.put("0")
        elif buttons[glfw.GAMEPAD_BUTTON_LEFT_BUMPER]:
            for button, key in ((glfw.GAMEPAD_BUTTON_A, "1"), (glfw.GAMEPAD_BUTTON_X, "2"),
                                (glfw.GAMEPAD_BUTTON_Y, "3")):
                if pressed(button):
                    self.events.put(key)
                    break
        self.prev_buttons = list(buttons)
        values = -np.asarray([axes[glfw.GAMEPAD_AXIS_LEFT_Y], axes[glfw.GAMEPAD_AXIS_LEFT_X],
                              axes[glfw.GAMEPAD_AXIS_RIGHT_X]], dtype=np.float32)
        values[np.abs(values) < .12] = 0.
        limits = self.runners["walk"].cfg.command_ranges
        self.gamepad_command[:] = values * np.where(values >= 0., limits[:, 1], -limits[:, 0])
        if self.state != "walking":
            self.gamepad_command[:] = 0.

    def run(self, duration, show_viewer=True, gamepad=None):
        viewer_threads = []
        if show_viewer:
            previous_threads = set(threading.enumerate())
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.on_key)
            viewer_threads = [t for t in threading.enumerate() if t not in previous_threads]
            self.visible_geom_groups = self.viewer.opt.geomgroup.copy()
            for name, value in flat.CAMERA.items():
                setattr(self.viewer.cam, name, value)
            if gamepad is not None:
                print(f"[手柄] 等待 GLFW 标准手柄 {gamepad}；未识别时仍可用键盘操作。")
        try:
            end = self.data.time + duration
            while self.data.time < end and (self.viewer is None or self.viewer.is_running()):
                began = time.monotonic()
                if gamepad is not None:
                    state = glfw.get_gamepad_state(gamepad)
                    connected = state is not None
                    if connected != self.gamepad_connected:
                        print("[手柄] " + ("已连接" if connected else "断开；摇杆速度归零"))
                        self.gamepad_connected = connected
                    self.gamepad_input(state.buttons, state.axes) if connected else self.gamepad_input(None, None)
                self.process_keys()
                self.step()
                if self.viewer is not None:
                    self.viewer.cam.lookat[:] = self.data.xpos[self.robot.base_body_id]
                    active = self.runners[self.active] if self.active else None
                    frame = active.reference_frame if isinstance(active, MotionRunner) else "-"
                    self.viewer.set_texts((None, None,
                        "State\nPolicy / frame\nVelocity\nTasks\nStop / halt / recheck\nMove / zero",
                        f"{self.state}\n{self.active or 'PD hold'} / {frame}\n{self.command.round(2)}\n"
                        "1 walk | 2 crouch | 3 rise\n0 / P / R\nWASD QE / Space"))
                    self.viewer.sync()
                    time.sleep(max(0., self.dt - (time.monotonic() - began)))
        finally:
            if self.viewer is not None:
                self.viewer.close()
                # close 只发退出请求，避免 GLFW 渲染线程尚未退出就结束 Python 进程。
                for thread in viewer_threads:
                    thread.join(timeout=5.)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for task in RUNS:
        parser.add_argument(f"--{task}-run")
    parser.add_argument("--initial-pose", choices=("stand", "crouch"), default="stand")
    parser.add_argument("--base-mass-add", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--gamepad", type=int, choices=range(16), help="GLFW 手柄编号，通常为 0；无需 ROS")
    args = parser.parse_args()
    duration = args.duration if args.duration is not None else (6. if args.headless else flat.SIM_DURATION)
    if not np.isfinite(duration) or duration <= 0. or (args.headless and args.gamepad is not None):
        parser.error("duration 必须为有限正数；无界面测试不读取手柄")
    np.random.seed(args.seed)
    sim = MultiTaskSim({task: getattr(args, task + "_run") for task in RUNS},
                       args.initial_pose, args.base_mass_add)
    sim.run(duration, show_viewer=not args.headless, gamepad=args.gamepad)
    print(f"[multi] 结束: t={sim.data.time:.2f}s state={sim.state} base_z={sim.data.qpos[2]:.3f}m")


if __name__ == "__main__":
    main()
