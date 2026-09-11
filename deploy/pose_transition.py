#!/usr/bin/env python3
"""当前姿态 -> STAND -> 本次平衡姿态的内侧目标；不运行 RL、不发使能/失能。

必须由操作者可靠支撑并准备硬件急停。fault_hold 只是保持最后指令，不是硬停。
ROS 时间新鲜仅代表消息到达，不能证明驱动缓存中每台电机都有新帧。
--dry-run / --selftest 不导入 ROS、不创建节点、不发布命令。
"""
import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from com_check import REAL, STAND, real_limits

# 与 pose_move 的纯定义一致；不导入旧脚本的 ROS 依赖/控制循环。
KP = np.array([200., 100., 100., 250., 40., 40.] * 2)
KD = np.array([5., 5., 5., 5., 2., .5] * 2)
EFF = np.array([26., 26., 26., 26., 26., 5.8] * 2)
ERR_LIMIT = np.minimum(.2, .8 * EFF / KP)
TIMEOUT = .2


def vector(value, label, limits=None):
    a = np.asarray(value, dtype=float)
    if a.shape != (12,):
        raise ValueError(f"{label}: 需要完整 12 维，实际 {a.shape}")
    bad = ~np.isfinite(a)
    if limits is not None:
        bad |= (a < limits[0]) | (a > limits[1])
    if bad.any():
        detail = ", ".join(f"{REAL[i]}={a[i]:.9g}" for i in np.flatnonzero(bad))
        raise ValueError(f"{label} 非有限或越限: {detail}；拒绝，不裁剪")
    return a.copy()


def read_snapshot(path):
    with open(path) as f:
        summary = json.load(f)["summary"]
    return vector([summary[j]["q_rad"] for j in REAL], str(path))


def interpolate(start, goal, elapsed, duration):
    u = min(max(elapsed / duration, 0.), 1.)
    s = u * u * (3. - 2. * u)
    return start + s * (goal - start)


def approach(start, goal, elapsed, duration):
    """前 2/3 时间到 90%，后 1/3 时间慢速靠记录；不是接触检测或硬限扭。"""
    middle = start + .9 * (goal - start)
    split = duration * 2. / 3.
    if elapsed <= split:
        return interpolate(start, middle, elapsed, split)
    return interpolate(middle, goal, elapsed - split, duration - split)


class Transition:
    """纯状态机：tick 返回待发目标；仅发布成功后调用 published。"""
    def __init__(self, balance, limits, move=60., fraction=.9, expected=None, stand_only=False, balance_only=False):
        if not math.isfinite(move) or move <= 0:
            raise ValueError("--move 必须为有限正数")
        if not math.isfinite(fraction) or not (0 < fraction < 1 or balance_only and fraction == 1):
            raise ValueError("--fraction 必须在 (0, 1) 内；仅 --balance-only 允许到记录的 1.0，不允许越过")
        if stand_only and balance_only:
            raise ValueError("stand-only 与 balance-only 不能同时使用")
        self.limits = limits
        self.stand = vector(STAND, "STAND", limits)
        balance = vector(balance, "平衡记录")
        self.goal = vector(STAND + fraction * (balance - STAND), "最终目标", limits)
        self.expected = None if expected is None else vector(expected, "预期起点", limits)
        self.move = move
        self.stand_only = stand_only
        self.balance_only = balance_only
        self.phase, self.error = "wait_state", ""
        self.q = self.dq = self.tau = self.last_sent = self.start = None
        self.last_valid = self.phase_at = self.began_at = self.stable_at = None
        self.valid_count = 0
        self.last_warn = ""

    def fault(self, reason):
        if self.phase != "fault_hold":
            self.phase, self.error = "fault_hold", reason

    def feedback(self, q, dq, tau, now):
        try:
            q = vector(q, "反馈 q", self.limits)
            dq = vector(dq, "反馈 dq")
            tau = vector(tau, "反馈 tau")
        except (TypeError, ValueError) as exc:
            self.fault(str(exc))
            return
        if self.last_valid is None or now - self.last_valid > TIMEOUT:
            if self.last_sent is not None and not (
                    self.phase == "init_hold" and now - self.phase_at < 2.):
                self.fault(f"有效反馈间隔超过 {TIMEOUT}s，拒绝继续轨迹")
            self.valid_count = 0
            self.stable_at = None
        self.q, self.dq, self.tau, self.last_valid = q, dq, tau, now
        self.valid_count += 1

    def warning(self, raw):
        self.last_warn = raw
        try:
            data = json.loads(raw)
            entries = data.get("errors", [])
            if not entries and data.get("collide_and_sync") == "无碰撞且从臂已经与主臂同步":
                return
            if not entries:
                raise ValueError("未知告警格式")
            for entry in entries:
                # 此驱动告警 ID 是数组槽位，不是 CAN ID：槽位 5=CAN6，6=CAN7。
                # 双腿模式每侧只有前 6 槽有效，后 6 槽是默认缓存。
                if int(entry["id"]) not in range(1, 7):
                    continue
                err = entry["err"]
                if err == "使能":
                    continue
                if err == "失能":
                    if self.phase in ("wait_state", "init_hold"):
                        continue
                self.fault("/motor_warn: " + raw)
        except (ValueError, KeyError, TypeError, AttributeError):
            self.fault("/motor_warn 无法确认安全: " + raw)

    def published(self, target):
        self.last_sent = vector(target, "已发目标", self.limits)

    def _enter(self, phase, now):
        self.phase, self.phase_at, self.stable_at = phase, now, None

    def tick(self, now, stop=False):
        if stop:
            self.fault("STOP 文件请求；请扶稳并使用硬件急停，不是断电急停")
        if self.phase == "fault_hold":
            return self.last_sent
        if self.last_valid is None:
            return None
        age = now - self.last_valid
        if age > TIMEOUT:
            self.stable_at = None
            # 第一次命令会触发驱动约 1.2 s 的重复 enable：仅启动原地保持有 2 s 宽限。
            if self.phase == "init_hold" and now - self.phase_at < 2.:
                return self.last_sent
            self.fault(f"关节反馈超时 {age:.3f}s > {TIMEOUT}s")
            return self.last_sent
        if self.phase == "wait_state":
            if self.valid_count < 3:
                return None
            if self.expected is not None and np.any(np.abs(self.q - self.expected) > .05):
                self.fault("首帧姿态与 --expected-start-file 相差超过 .05 rad，拒绝接管")
                return None
            if self.balance_only and np.any(np.abs(self.q - self.stand) > .05):
                self.fault("尚未到站立姿态 .05 rad 范围内，拒绝下蹲")
                return None
            # 下蹲接管原 stand_hold 的同一目标，不因实测静差而跳变指令。
            self.start = self.stand.copy() if self.balance_only else self.q.copy()
            self.began_at = now
            self._enter("init_hold", now)
        elapsed = now - self.phase_at
        if self.phase == "init_hold":
            candidate = self.start.copy()
        elif self.phase == "to_stand":
            candidate = interpolate(self.start, self.stand, elapsed, self.move)
        elif self.phase == "to_balance":
            candidate = (approach(self.start, self.goal, elapsed, self.move) if self.balance_only
                         else interpolate(self.stand, self.goal, elapsed, self.move))
        else:
            candidate = self.stand.copy() if self.phase in ("settle_stand", "stand_hold") else self.goal.copy()
        near_end = self.balance_only and (self.phase in ("settle_balance", "complete_hold") or
                                         self.phase == "to_balance" and elapsed >= self.move * 2. / 3.)
        # 仅收紧继续前进的门限；不是驱动硬限扭，fault_hold 仍保持最后目标。
        effort_limit = (.4 if near_end else .8) * EFF
        error_limit = np.minimum(ERR_LIMIT, effort_limit / KP)
        over = np.abs(candidate - self.q) > error_limit
        over_tau = np.abs(self.tau) > effort_limit
        if over.any() or over_tau.any():
            i = int(np.flatnonzero(over | over_tau)[0])
            self.fault(f"{REAL[i]} 保护: 误差={candidate[i]-self.q[i]:+.5f}rad "
                       f"(限{error_limit[i]:.5f}), 自报tau={self.tau[i]:+.3f}Nm (限{effort_limit[i]:.3f})")
            return self.last_sent
        if self.phase in ("init_hold", "settle_stand", "settle_balance"):
            if np.all(np.abs(candidate - self.q) <= .05):
                if self.stable_at is None:
                    self.stable_at = now
            else:
                self.stable_at = None
            settled = self.stable_at is not None and now - self.stable_at >= 1.
            if settled and (self.phase != "init_hold" or elapsed >= 2.):
                if self.phase == "init_hold" and not self.balance_only:
                    self.start = self.q.copy()
                    candidate = self.start.copy()
                following = {"init_hold": "to_balance" if self.balance_only else "to_stand", "settle_stand": "stand_hold" if self.stand_only else "to_balance",
                             "settle_balance": "complete_hold"}
                self._enter(following[self.phase], now)
            elif elapsed >= 10.:
                self.fault("到位等待超过 10 秒；所有关节需误差 <= .05 rad 连续 1 秒")
                return self.last_sent
        elif self.phase in ("to_stand", "to_balance") and elapsed >= self.move:
            self._enter("settle_stand" if self.phase == "to_stand" else "settle_balance", now)
        return candidate

    def status(self, now):
        values = lambda a: None if a is None else a.tolist()
        return {"wall_time": time.time(), "phase": self.phase, "error": self.error,
                "elapsed": 0. if self.began_at is None else now - self.began_at,
                "phase_elapsed": 0. if self.phase_at is None else now - self.phase_at,
                "valid_age": None if self.last_valid is None else now - self.last_valid,
                "q": values(self.q), "target": values(self.last_sent), "dq": values(self.dq),
                "tau": values(self.tau), "last_warn": self.last_warn}


def selftest():
    limits = (-np.ones(12) * 2, np.ones(12) * 2)
    balance = STAND + np.linspace(-.2, .2, 12)
    def ready(stand_only=False):
        c = Transition(balance, limits, move=.1, stand_only=stand_only)
        for t in (0., .01, .02):
            c.feedback(STAND, np.zeros(12), np.zeros(12), t)
        c.published(c.tick(.02))
        return c
    for t in (-1., 0., .25, .5, 1., 2.):
        q = interpolate(STAND, balance, t, 1.)
        assert np.all(q >= np.minimum(STAND, balance) - 1e-12)
        assert np.all(q <= np.maximum(STAND, balance) + 1e-12)
        assert np.ptp((q - STAND) / (balance - STAND)) < 1e-12
    assert np.allclose(interpolate(STAND, balance, 1., 1.), balance)
    for bad in (np.full(12, np.nan), np.ones(11), np.full(12, 100)):
        try:
            Transition(bad, limits)
        except ValueError:
            pass
        else:
            raise AssertionError("非法目标未拒绝")
    c = ready()
    assert np.array_equal(c.tick(1.), STAND) and c.phase == "init_hold"
    for t in np.arange(1.3, 2.51, .05):
        c.feedback(STAND, np.zeros(12), np.zeros(12), t)
        c.published(c.tick(t))
    assert c.phase in ("to_stand", "settle_stand")
    c.tick(2.9)
    assert c.phase == "fault_hold"
    c.feedback(STAND, np.zeros(12), np.zeros(12), 3.)
    assert np.array_equal(c.tick(3.), STAND) and c.phase == "fault_hold"
    for q, dq, tau in ((np.full(12, np.nan), np.zeros(12), np.zeros(12)),
                       (np.full(12, 3.), np.zeros(12), np.zeros(12)),
                       (STAND, [], np.zeros(12)), (STAND, np.zeros(12), [0.] * 11),
                       (STAND, np.zeros(12), np.full(12, np.inf))):
        c = ready(); c.feedback(q, dq, tau, .03)
        assert c.phase == "fault_hold" and np.array_equal(c.tick(.03), STAND)
    c = ready()
    for t in np.arange(.03, 6., .005):
        c.feedback(c.last_sent, np.zeros(12), np.zeros(12), t)
        c.published(c.tick(t))
    assert c.phase == "complete_hold" and np.allclose(c.last_sent, c.goal)
    c.feedback(c.last_sent, np.zeros(12), EFF, 6.)
    c.tick(6.)
    assert c.phase == "fault_hold"
    c = ready(stand_only=True)
    for t in np.arange(.03, 6., .005):
        c.feedback(c.last_sent, np.zeros(12), np.zeros(12), t)
        c.published(c.tick(t))
        assert c.phase not in ("to_balance", "settle_balance", "complete_hold")
    assert c.phase == "stand_hold" and np.array_equal(c.last_sent, STAND)
    c.feedback(c.last_sent, np.zeros(12), EFF, 6.)
    c.tick(6.)
    assert c.phase == "fault_hold"
    c = ready(); c.tick(.03, stop=True)
    assert c.phase == "fault_hold"
    c = ready(); c.phase = "complete_hold"; c.warning('{"errors":[{"id":1,"err":"失能"}]}')
    assert c.phase == "fault_hold"
    c = ready(); c.phase = "complete_hold"
    c.warning(json.dumps({"errors": [{"id": i, "err": "失能"} for i in range(7, 13)]}))
    assert c.phase == "complete_hold"
    c.warning('{"errors":[{"id":5,"err":"失能"}]}')
    assert c.phase == "fault_hold"
    c = ready(); c._enter("settle_stand", .02)
    for t in np.arange(.03, 10.1, .05):
        c.feedback(STAND + .06, np.zeros(12), np.zeros(12), t)
        c.published(c.tick(t))
    assert c.phase == "fault_hold" and "10 秒" in c.error
    c = ready(); c.move = 60.
    for t in np.arange(.03, 2.1, .01):
        c.feedback(STAND + .02, np.zeros(12), np.zeros(12), t)
        c.published(c.tick(t))
    assert c.phase == "to_stand" and np.allclose(c.start, STAND + .02)
    c.feedback(STAND + .02, np.zeros(12), np.zeros(12), 2.4)
    assert c.phase == "fault_hold"  # 新帧先到也不能抹去已发生的数据空窗。
    c = Transition(balance, limits, expected=STAND + .1)
    for t in (0., .01, .02):
        c.feedback(STAND, np.zeros(12), np.zeros(12), t)
    assert c.tick(.02) is None and c.phase == "fault_hold"
    for t in np.linspace(-1., 31., 100):
        q = approach(STAND, balance, t, 30.)
        assert np.all(q >= np.minimum(STAND, balance) - 1e-12)
        assert np.all(q <= np.maximum(STAND, balance) + 1e-12)
    assert np.allclose(approach(STAND, balance, 20., 30.), STAND + .9 * (balance - STAND))
    assert np.allclose(approach(STAND, balance, 30., 30.), balance)
    c = Transition(balance, limits, move=3., fraction=1., balance_only=True)
    for t in (0., .01, .02):
        c.feedback(STAND + .01, np.zeros(12), np.zeros(12), t)
    c.published(c.tick(.02))
    assert np.array_equal(c.last_sent, STAND)
    for t in np.arange(.03, 8., .005):
        c.feedback(c.last_sent, np.zeros(12), np.zeros(12), t)
        c.published(c.tick(t))
        assert c.phase not in ("to_stand", "settle_stand", "stand_hold", "fault_hold")
    assert c.phase == "complete_hold" and np.allclose(c.last_sent, balance)
    c.feedback(c.last_sent, np.zeros(12), .5 * EFF, 8.)
    c.tick(8.)
    assert c.phase == "fault_hold"
    print("selftest PASS: 同步/端点/无过冲/非法输入/启动空窗/超时锁存/到位/保持保护/STOP/告警")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--balance-file", type=Path)
    p.add_argument("--expected-start-file", type=Path)
    p.add_argument("--move", type=float, default=60.)
    p.add_argument("--fraction", type=float, default=.9)
    p.add_argument("--stand-only", action="store_true", help="只插值到站立并持续保持，不进入下蹲段")
    p.add_argument("--balance-only", action="store_true", help="从站立保持接管，分段慢速靠近记录下蹲角度")
    p.add_argument("--handoff-pid", type=int, help="旧控制进程退出前仅订阅，不发布，用于单写者交接")
    p.add_argument("--log-dir", type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.balance_file:
        p.error("必须显式指定本次 --balance-file，不使用旧 balance_pose.json")
    control = Transition(read_snapshot(args.balance_file), real_limits(), args.move, args.fraction,
                         None if args.expected_start_file is None else read_snapshot(args.expected_start_file),
                         stand_only=args.stand_only, balance_only=args.balance_only)
    if args.handoff_pid is not None and args.handoff_pid <= 1:
        p.error("--handoff-pid 必须是明确的旧控制进程 PID > 1")
    plan = {"joint_order": REAL, "balance_file": str(args.balance_file.resolve()),
            "expected_start_file": None if args.expected_start_file is None else str(args.expected_start_file.resolve()),
            "stand": STAND.tolist(), "goal": control.goal.tolist(), "segment_seconds": args.move,
            "fraction": args.fraction, "stand_only": args.stand_only, "balance_only": args.balance_only,
            "handoff_pid": args.handoff_pid, "kp": KP.tolist(), "kd": KD.tolist(),
            "error_limit": ERR_LIMIT.tolist(), "effort_limit": (.8 * EFF).tolist(),
            "approach_error_limit": np.minimum(ERR_LIMIT, .4 * EFF / KP).tolist() if args.balance_only else None,
            "approach_effort_limit": (.4 * EFF).tolist() if args.balance_only else None}
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return
    if args.log_dir is None:
        p.error("运行必须指定唯一的新 --log-dir")
    args.log_dir.mkdir(parents=True, exist_ok=False)
    (args.log_dir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    print("须可靠支撑并准备硬件急停。ROS 新鲜不证明逐电机缓存有效。STOP 只保持最后目标。", flush=True)

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray, String

    rclpy.init()
    node = Node("pose_transition")
    pub = node.create_publisher(Float64MultiArray, "/dog_joint_pos", 1)
    node.create_subscription(JointState, "/left_joint_states",
                             lambda m: control.feedback(m.position, m.velocity, m.effort, time.monotonic()), 1)
    node.create_subscription(String, "/motor_warn", lambda m: control.warning(m.data), 20)
    last_log, last_phase = -math.inf, ""
    with (args.log_dir / "trace.jsonl").open("x", buffering=1) as log:
        def tick():
            nonlocal last_log, last_phase
            now = time.monotonic()
            stop = (args.log_dir / "STOP").exists()
            subscribers = pub.get_subscription_count()
            handoff_pending = False
            if args.handoff_pid is not None:
                try:
                    os.kill(args.handoff_pid, 0)
                    handoff_pending = True  # 旧发布者仍在：仅记录反馈，不发竞争控制帧。
                except ProcessLookupError:
                    args.handoff_pid = None
            if not subscribers and control.last_sent is not None:
                control.fault("/dog_joint_pos 驱动订阅者消失")
            # 首个发布必须已有驱动订阅者，避免 init_hold 在命令未被接收时空走。
            target = control.tick(now, stop=stop) if not handoff_pending and (subscribers or control.last_sent is not None or stop) else None
            if target is not None:
                message = Float64MultiArray()
                message.data = target.tolist()
                pub.publish(message)
                control.published(target)
            if control.phase != last_phase:
                last_phase = control.phase
                print(f"phase={control.phase} {control.error}", flush=True)
                if control.phase == "fault_hold":
                    print("故障已锁存，只保持最后有效指令；请扶稳并使用硬件急停。不会自动恢复。", flush=True)
            if now - last_log >= .1:
                last_log = now
                record = json.dumps(control.status(now), ensure_ascii=False, allow_nan=False)
                try:
                    log.write(record + "\n")
                    temporary = args.log_dir / "status.json.tmp"
                    temporary.write_text(record + "\n")
                    os.replace(temporary, args.log_dir / "status.json")
                except OSError as exc:
                    control.fault("日志写入失败: " + str(exc))
        node.create_timer(.005, tick)
        if args.handoff_pid is not None:
            print(f"handoff_ready: 仅订阅，等待旧进程 {args.handoff_pid} 退出后接管站立目标", flush=True)
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            print("停止发布；这不是硬急停，机器人可能失去支撑，请人工扶稳。", flush=True)
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
