# -*- coding: utf-8 -*-
"""Replay real 200 Hz sysid commands in Isaac with a true world-fixed base."""
from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

SIM_DT = 0.005
ASSET_SYSPATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "source", "legs_rl_lab")
)

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", required=True, help="real *_cmd.csv input directory")
parser.add_argument("--out-dir", required=True, help="simulation state CSV output directory")
parser.add_argument("--asset", choices=("legs_narrow", "nlegs_body"), default="legs_narrow",
                    help="robot asset/config used for replay")
parser.add_argument("--deploy-yaml", help="rebuild actuator groups from this run description")
parser.add_argument("--joints", required=True, help="real-order indices, e.g. 0,3,4")
parser.add_argument("--contains", default="r1", help="only replay command files containing this text")
parser.add_argument("--delay", type=int, default=5, help="fixed actuator delay in 5 ms physics steps")
parser.add_argument("--base-quat", default="1,0,0,0", help="fixed-base quaternion w,x,y,z")
parser.add_argument("--friction", type=float, default=None, help="override tested-joint static friction")
parser.add_argument("--dynamic-friction", type=float, default=None)
parser.add_argument("--viscous", type=float, default=None)
parser.add_argument("--armature", type=float, default=None)
parser.add_argument("--vlim", type=float, default=None, help="override tested-joint DCMotor velocity limit")
parser.add_argument("--expand-limits", action="store_true",
                    help="temporarily expand the tested joint limit to cover the recorded command")
parser.add_argument("--zoh", action="store_true", help="zero-order-hold recorded commands instead of interpolation")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import copy
import csv
import glob
import importlib
import re
import sys

import numpy as np
import torch

sys.path.append(ASSET_SYSPATH)
asset_module, asset_cfg = {
    "legs_narrow": ("legs_rl_lab.assets.legs_narrow.nlegs", "NLEGS_FIX_CFG"),
    "nlegs_body": ("legs_rl_lab.assets.nlegs.nlegs_body", "NLEGS_BODY_CFG"),
}[args.asset]
ROBOT_CFG = getattr(importlib.import_module(asset_module), asset_cfg)
from run_actuator_cfg import configure_from_deploy, load_deploy

from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg, SimulationContext


def sim_joint(j: int) -> str:
    return f"joint_L{j + 1}" if j < 6 else f"joint_R{j - 5}"


def load_cmd(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty command CSV: {path}")
    needed = {"t", "phase", *(f"qd{i}" for i in range(12))}
    missing = needed - rows[0].keys()
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    t = np.array([float(r["t"]) for r in rows])
    qd = np.array([[float(r[f"qd{i}"]) for i in range(12)] for r in rows])
    phase = np.array([r["phase"] for r in rows], dtype=object)
    return t, qd, phase


def main() -> None:
    joints = [int(x) for x in args.joints.split(",")]
    if not joints or any(j < 0 or j >= 12 for j in joints):
        raise SystemExit("--joints must contain indices in [0, 11]")
    quat = np.array([float(x) for x in args.base_quat.split(",")])
    if len(quat) != 4 or np.linalg.norm(quat) == 0:
        raise SystemExit("--base-quat must be a non-zero w,x,y,z quaternion")
    quat /= np.linalg.norm(quat)
    os.makedirs(args.out_dir, exist_ok=True)

    sim = SimulationContext(SimulationCfg(dt=SIM_DT, device="cpu"))
    cfg = copy.deepcopy(ROBOT_CFG)
    cfg.prim_path = "/World/Robot"
    if args.deploy_yaml:
        cfg = configure_from_deploy(cfg, load_deploy(args.deploy_yaml))
        cfg.prim_path = "/World/Robot"
    else:
        # Legacy replay behavior; exact run comparisons should pass --deploy-yaml.
        cfg.actuators["4340"].velocity_limit_sim = 14.0
    robot = Articulation(cfg)

    import omni.physx.scripts.utils as pxutils
    import omni.usd
    from pxr import Gf

    stage = omni.usd.get_context().get_stage()
    base_prim = stage.GetPrimAtPath("/World/Robot/base/base")
    if not base_prim.IsValid():
        raise RuntimeError("missing rigid base prim /World/Robot/base/base")
    fixed = pxutils.createJoint(stage, "Fixed", None, base_prim)
    fixed.GetAttribute("physics:localRot0").Set(Gf.Quatf(*quat.tolist()))
    sim.reset()
    print(f"[fixed-base] world_joint={fixed.GetPath()} quat={quat.tolist()}")

    real2col = [robot.find_joints(sim_joint(i))[0][0] for i in range(12)]
    q_des = torch.zeros((robot.num_instances, robot.num_joints))
    v_des = torch.zeros_like(q_des)

    def actuator_local(jcol: int):
        for name, actuator in robot.actuators.items():
            indices = list(np.asarray(actuator.joint_indices).ravel())
            if jcol in indices:
                return name, actuator, indices.index(jcol)
        raise RuntimeError(f"joint column {jcol} has no actuator")

    def set_delay(tested_actuator) -> None:
        for actuator in robot.actuators.values():
            for name in ("positions_delay_buffer", "velocities_delay_buffer", "efforts_delay_buffer"):
                buffer = getattr(actuator, name, None)
                if buffer is not None:
                    # Untested joints hold a constant target, so their exact lag is irrelevant.
                    delay = args.delay if actuator is tested_actuator else min(args.delay, buffer.history_length)
                    buffer.set_time_lag(delay)

    files = []
    for joint in joints:
        files.extend(glob.glob(os.path.join(args.data_dir, f"j{joint}_kp*_kd*{args.contains}*_cmd.csv")))
    files = sorted(set(files))
    if not files:
        raise SystemExit(f"no command CSV matched --contains {args.contains!r}")

    for command_path in files:
        tag = os.path.basename(command_path)[:-len("_cmd.csv")]
        match = re.search(r"_kp([^_]+)_kd([^_]+)_", tag)
        if not match:
            raise ValueError(f"cannot read kp/kd from {tag}")
        joint = int(re.match(r"j(\d+)_", tag).group(1))
        jcol = real2col[joint]
        actuator_name, actuator, local = actuator_local(jcol)
        actuator.stiffness[:, local] = float(match.group(1))
        actuator.damping[:, local] = float(match.group(2))

        t_cmd, qd_cmd, phase_cmd = load_cmd(command_path)
        if args.expand_limits:
            limits = robot.data.joint_pos_limits[:, [jcol], :].clone()
            old_limits = limits.clone()
            limits[..., 0] = min(float(limits[0, 0, 0]), float(qd_cmd[:, joint].min()) - 0.01)
            limits[..., 1] = max(float(limits[0, 0, 1]), float(qd_cmd[:, joint].max()) + 0.01)
            if not torch.equal(limits, old_limits):
                robot.write_joint_position_limit_to_sim(limits, joint_ids=[jcol])
                print(f"[limit] j{joint} {old_limits[0, 0].tolist()} -> {limits[0, 0].tolist()}")
        q_init = torch.zeros_like(q_des)
        for i in range(12):
            q_init[:, real2col[i]] = float(qd_cmd[0, i])
        robot.write_joint_state_to_sim(q_init, torch.zeros_like(q_des))
        robot.reset()
        set_delay(actuator)
        if args.friction is not None:
            dynamic = args.friction if args.dynamic_friction is None else args.dynamic_friction
            viscous = 0.0 if args.viscous is None else args.viscous
            robot.write_joint_friction_coefficient_to_sim(
                args.friction, dynamic, viscous, joint_ids=[jcol]
            )
        elif args.viscous is not None:
            raise SystemExit("--viscous requires --friction so static/dynamic values are explicit")
        if args.armature is not None:
            robot.write_joint_armature_to_sim(args.armature, joint_ids=[jcol])
        if args.vlim is not None:
            if not hasattr(actuator, "_saturation_effort"):
                raise SystemExit(f"joint {joint} actuator {actuator_name} has no DCMotor velocity curve")
            actuator.velocity_limit[:, local] = args.vlim
            actuator._vel_at_effort_lim = actuator.velocity_limit * (
                1.0 + actuator.effort_limit / actuator._saturation_effort
            )

        q_des.copy_(q_init)
        records = []
        for k in range(int(np.ceil(t_cmd[-1] / SIM_DT))):
            t_apply = k * SIM_DT
            src = max(0, np.searchsorted(t_cmd, t_apply, side="right") - 1)
            for i in range(12):
                value = qd_cmd[src, i] if args.zoh else np.interp(t_apply, t_cmd, qd_cmd[:, i])
                q_des[:, real2col[i]] = float(value)
            robot.set_joint_position_target(q_des)
            robot.set_joint_velocity_target(v_des)
            robot.write_data_to_sim()
            sim.step()
            robot.update(SIM_DT)
            t_state = t_apply + SIM_DT
            phase = phase_cmd[src]
            q = robot.data.joint_pos[0].cpu().numpy()
            v = robot.data.joint_vel[0].cpu().numpy()
            tau = robot.data.applied_torque[0].cpu().numpy()
            records.append((t_state, phase,
                            [q[real2col[i]] for i in range(12)],
                            [v[real2col[i]] for i in range(12)],
                            [tau[real2col[i]] for i in range(12)]))
            if not app.is_running():
                break

        option_tag = f"fix_d{args.delay}"
        if args.expand_limits:
            option_tag += "_xl"
        if args.friction is not None:
            option_tag += f"_fc{args.friction:g}"
            if args.viscous is not None:
                option_tag += f"_cv{args.viscous:g}"
        if args.armature is not None:
            option_tag += f"_arm{args.armature:g}"
        if args.vlim is not None:
            option_tag += f"_vlim{args.vlim:g}"
        out = os.path.join(args.out_dir, f"SIM_{tag}_{option_tag}_state.csv")
        with open(out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "phase", *(f"q{i}" for i in range(12)),
                             *(f"v{i}" for i in range(12)), *(f"tau{i}" for i in range(12))])
            for t, phase, q, v, tau in records:
                writer.writerow([f"{t:.6f}", phase, *(f"{x:.6f}" for x in q + v + tau)])
        settle_tau = [r[4][joint] for r in records if r[1] == "settle"]
        mean_tau = float(np.mean(settle_tau)) if settle_tau else float("nan")
        print(f"[done] j{joint} {tag}: actuator={actuator_name} settle_tau={mean_tau:+.3f} "
              f"rows={len(records)} -> {out}")


if __name__ == "__main__":
    main()
    app.close()
