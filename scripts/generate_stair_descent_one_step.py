"""Plan one short swing per tread and per foot; offline kinematics, not a robot command."""

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from generate_crouch_pose_bank import NAMES, STAND
from generate_stair_descent_motion import ROOT, foot_points, render, solver_for, stair_model, terrain_height
from prepare_crouch_mimic import convert

OUTPUT = ROOT / "datasets/stair_descent_one_step_v4"
DT = .02
SWING_START, STEP_INTERVAL, SWING_TIME = .36, .46, .42
STEPS = [(1 if k % 2 == 0 else 0, 3 - k // 2) for k in range(8)]
END_TIME = SWING_START + 7 * STEP_INTERVAL + SWING_TIME + .2


def smooth(u):
    u = np.clip(u, 0., 1.)
    return u * u * u * (10. + u * (-15. + 6. * u))


def desired_feet(t):
    x = np.array([.2, .2])
    z = np.array([.2, .2])
    landing_weight = np.ones(2)
    airborne = np.zeros(2, dtype=bool)
    levels = np.full(2, 4, dtype=int)
    for k, (side, level) in enumerate(STEPS):
        start = SWING_START + k * STEP_INTERVAL
        if t < start:
            continue
        u = np.clip((t - start) / SWING_TIME, 0., 1.)
        p = smooth(u)
        x[side] += .2 * p
        z[side] -= .05 * p
        if u < 1.:
            z[side] += .08 * np.sin(np.pi * u)
            landing_weight[side] = smooth((u - .75) / .25)
            airborne[side] = True
        else:
            levels[side] = level
            landing_weight[side] = 1. - smooth((t - start - SWING_TIME - .06) / .08)
    return x, z, landing_weight, airborne, levels


def generate():
    model = stair_model()
    solver = solver_for(model)
    qadr = model.jnt_qposadr[[model.joint(name).id for name in NAMES]]
    lower = np.r_[0., -.06, .40, -.12, -.16, -.12, solver.limits[:, 0] + .01]
    upper = np.r_[1.25, .06, .84, .12, .16, .12, solver.limits[:, 1] - .01]

    def points(v):
        solver.data.qpos[:3] = v[:3]
        solver.data.qpos[3:7] = Rotation.from_euler("xyz", v[3:6]).as_quat(scalar_first=True)
        solver.data.qpos[qadr] = v[6:]
        mujoco.mj_forward(model, solver.data)
        return foot_points(solver)

    previous = np.r_[.15, 0., .785, 0., 0., 0., STAND].astype(float)
    foot_y = points(previous)[:, :, 1].mean(axis=1)
    timeline = np.arange(round(END_TIME / DT) + 1) * DT
    states, tracked, planned = [], [], []
    for frame, t in enumerate(timeline):
        foot_x, foot_z, flat, airborne, levels = desired_feet(t)
        base_x = .15 + (foot_x.mean() - .2)
        crouch = .04 * smooth(t / SWING_START)
        base_z = .785 + (foot_z.mean() - .2) - crouch

        def residual(v):
            p = points(v)
            center = p.mean(axis=1)
            sole_z = p[:, :, 2].min(axis=1)
            terrain = terrain_height(p[:, :, 0])
            x_low = np.where(levels == 4, 0., .9 - levels * .2)
            x_high = x_low + np.where(levels == 4, .3, .2)
            return np.r_[
                (center[:, 0] - foot_x) / .002,
                (center[:, 1] - foot_y) / .006,
                (sole_z - foot_z) / .001,
                flat * (p[:, :, 2].max(axis=1) - sole_z) / .001,
                np.maximum(terrain - p[:, :, 2], 0.).ravel() / .002,
                np.where(airborne, 0., np.maximum(x_low + .003 - p[:, :, 0].min(axis=1), 0.)) / .003,
                np.where(airborne, 0., np.maximum(p[:, :, 0].max(axis=1) - x_high + .003, 0.)) / .003,
                (v[:6] - [base_x, 0., base_z, 0., 0., 0.]) / [.05, .02, .05, .035, .08, .05],
                (v[6:] - previous[6:]) / .30,
            ]

        max_step = np.r_[[.02, .005, .012, .02, .02, .02], np.full(12, .16)]
        frame_lower = np.maximum(lower, previous - max_step)
        frame_upper = np.minimum(upper, previous + max_step)
        fit = least_squares(residual, previous, bounds=(frame_lower, frame_upper), max_nfev=80)
        previous = fit.x
        states.append(previous.copy())
        tracked.append(points(previous).copy())
        planned.append(~airborne)
        if frame % 25 == 0:
            print(f"t={t:.2f} s, max sole target error={np.max(np.abs(tracked[-1].mean(axis=1)[:, 0] - foot_x)):.3f} m", flush=True)
    states, tracked, planned = map(np.asarray, (states, tracked, planned))
    return model, timeline, states, tracked, planned


def validate(model, time, states, feet, planned):
    q = states[:, 6:]
    bounds = model.jnt_range[[model.joint(name).id for name in NAMES]]
    margin = np.minimum(q - bounds[:, 0], bounds[:, 1] - q).min()
    qd = np.gradient(q, DT, axis=0)
    landings = []
    for k, (side, level) in enumerate(STEPS):
        i = round((SWING_START + k * STEP_INTERVAL + SWING_TIME) / DT)
        p = feet[i, side]
        lo = .9 - level * .2
        hi = lo + .2
        landings.append(dict(time_s=round(float(time[i]), 2), foot="L" if side == 0 else "R",
                             height_cm=level * 5, x_min_m=round(float(p[:, 0].min()), 3),
                             x_max_m=round(float(p[:, 0].max()), 3),
                             z_min_m=round(float(p[:, 2].min()), 3),
                             z_max_m=round(float(p[:, 2].max()), 3)))
        if p[:, 0].min() < lo - .003 or p[:, 0].max() > hi + .003 or np.max(np.abs(p[:, 2] - level * .05)) > .005:
            raise ValueError(f"Foot is not fully on tread {level}: {landings[-1]}")
    if margin < 0 or np.max(np.abs(qd)) > 8.2:
        raise ValueError(f"Joint limit/speed check failed: margin={margin:.4f}, max qd={np.max(np.abs(qd)):.2f}")
    if np.max(np.abs(states[:, 1])) > .06 or np.max(np.abs(states[:, 3])) > .12:
        raise ValueError("Base moves too far laterally or rolls too much")
    return landings, qd, round(float(margin), 4)


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    model, time, states, feet, planned = generate()
    landings, qd, margin = validate(model, time, states, feet, planned)
    quat = Rotation.from_euler("xyz", states[:, 3:6]).as_quat(scalar_first=True)
    motion = dict(time=time, joint_names=np.asarray(NAMES), root_pos=states[:, :3],
                  root_quat_wxyz=quat, joint_pos=states[:, 6:], joint_vel=qd,
                  root_lin_vel_world=np.gradient(states[:, :3], DT, axis=0),
                  root_ang_vel_body=np.gradient(states[:, 3:6], DT, axis=0),
                  planned_contact=planned,
                  foot_sole_x=feet[:, :, :, 0].mean(axis=2),
                  foot_sole_min_z=feet[:, :, :, 2].min(axis=2))
    OUTPUT.mkdir(parents=True)
    np.savez_compressed(OUTPUT / "motion.npz", **motion)
    convert(OUTPUT / "motion.npz", OUTPUT / "mimic_motion.npz", ROOT / "source/legs_rl_lab/legs_rl_lab/assets/nlegs/mjcf/nlegs.xml")
    (OUTPUT / "metadata.json").write_text(json.dumps({
        "source": "contact-constrained IK; gait timing from rough sim2sim",
        "dynamic_check": "simple joint-PD replay fell; this is not a dynamically verified expert clip",
        "stair": "four 5 cm drops, 20 cm tread, 1 m wide",
        "swing_s": SWING_TIME, "joint_limit_margin_rad": margin,
        "landings": landings,
    }, indent=2) + "\n")
    render(motion, OUTPUT, landings)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
