"""Generate and render a standing -> two outward steps -> limit-crouch reference.

Offline kinematics only: this is not a torque-feasible demonstration or robot command.
Run with unitree_lab Python; --check validates an existing motion without rendering.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation, Slerp

from generate_crouch_pose_bank import CLEARANCE, CROUCH_LIMIT_SIDE, NAMES, STAND, XML, PoseSolver

OUTPUT = Path(__file__).resolve().parents[1] / "datasets/stand_to_crouch_v1"
FPS = 50
TIMES = np.array([0, .6, 1.6, 3.2, 4.4, 6.0, 7.0, 9.0, 10.0])
LABELS = ["Stand", "Load right foot", "Step left out", "Load left foot",
          "Step right out", "Center weight", "Crouch to limits", "Hold crouch"]


def smooth(u):
    return u**3 * (10 + u * (-15 + 6 * u))


def set_state(solver, x):
    solver.data.qpos[:3] = x[:3]
    solver.data.qpos[3:7] = Rotation.from_euler("xyz", x[3:6]).as_quat(scalar_first=True)
    solver.data.qpos[solver.qadr] = x[6:]
    mujoco.mj_forward(solver.model, solver.data)


def foot_points(solver):
    points = []
    for foot in solver.feet:
        ids = solver.caps[solver.model.geom_bodyid[solver.caps] == foot]
        axes = solver.data.geom_xmat[ids].reshape(-1, 3, 3)[:, :, 2]
        offset = axes * solver.model.geom_size[ids, 1, None]
        p = np.stack((solver.data.geom_xpos[ids] - offset,
                      solver.data.geom_xpos[ids] + offset), axis=1)
        p[:, :, 2] -= solver.model.geom_size[ids, 0, None]
        points.append(p.reshape(-1, 3))
    return np.asarray(points)


def com(solver):
    # ponytail: one nominal 2.5 kg payload; rerun dynamics/DR before using as an expert.
    return (solver.mass * solver.data.subtree_com[solver.base]
            + 2.5 * solver.data.xipos[solver.base]) / (solver.mass + 2.5)


def read_crouch_snapshot(path):
    snapshot = json.loads(path.read_text())
    return np.array(snapshot["L"]["q"] + snapshot["R"]["q"], dtype=float)


def motion_solver(measured_q=None):
    solver = PoseSolver()
    if measured_q is not None:
        measured_q = np.asarray(measured_q, dtype=float)
        if (measured_q.shape != (12,) or not np.isfinite(measured_q).all()
                or np.any(measured_q < solver.limits[:, 0]) or np.any(measured_q > solver.limits[:, 1])):
            raise ValueError("Measured crouch must contain 12 finite angles within the current XML bounds")
        solver.crouch[:] = measured_q
        for name, side in CROUCH_LIMIT_SIDE.items():
            i = NAMES.index(name)
            solver.limits[i, side] = measured_q[i]
        if np.any(STAND < solver.limits[:, 0]) or np.any(STAND > solver.limits[:, 1]):
            raise ValueError("Measured task stops exclude the standing pose")
    return solver


def endpoints(solver, measured=False):
    # Exact measured stops are mildly non-coplanar in the nominal geometry.
    q, state = solver.solve_limit(max_foot_tilt=1.0, max_height_error=.003) if measured else solver.solve_limit()
    end = np.r_[0, 0, state["height"] - CLEARANCE,
                Rotation.from_quat(state["quat"], scalar_first=True).as_euler("xyz"), q]
    state = solver.evaluate(STAND, 0, 0)
    start = np.r_[0, 0, state["height"] - CLEARANCE, 0, 0, 0, STAND]
    feet, rotations, centers = [], [], []
    for x in (start, end):
        set_state(solver, x)
        feet.append(solver.data.xpos[solver.feet].copy())
        rotations.append(solver.data.xmat[solver.feet].reshape(2, 3, 3).copy())
        centers.append(com(solver)[:2])
    return start, end, np.array(feet), np.array(rotations), np.array(centers)


def generate(solver, variant="v1", lift_height=None, time_scale=1.0, measured=False):
    if not np.isfinite(time_scale) or time_scale < 1:
        raise ValueError("time_scale must be finite and >= 1")
    start, end, feet, rots, centers = endpoints(solver, measured)
    quick = variant == "v2"
    times = np.array([0, .6, 1.0, 1.08, 1.48, 2.28, 3.4]) if quick else TIMES
    labels = (["Stand", "Low step left out", "Touch down", "Low step right out",
               "Settle to limits", "Hold crouch"] if quick else LABELS)
    fps = 100 if quick else FPS
    lift_times, swing_duration, default_lift = ((.6, 1.08), .4, .012) if quick else ((1.6, 4.4), 1.6, .05)
    lift_height = default_lift if lift_height is None else lift_height
    if not np.isfinite(lift_height) or not 0 < lift_height <= .05:
        raise ValueError("lift_height must be in (0, 0.05] m")
    times = times * time_scale
    if not np.allclose(times * fps, np.round(times * fps)):
        raise ValueError("Scaled segment boundaries must align with the frame interval")
    lift_times = np.array(lift_times) * time_scale
    swing_duration *= time_scale
    support = feet[:, :, :2] + np.einsum("sfij,j->sfi", rots, [.05, 0, 0])[:, :, :2]
    com_keys = np.array([centers[0], centers[0], support[0, 1], support[0, 1],
                         support[1, 0], support[1, 0], centers[1], centers[1], centers[1]])
    progress_keys = np.array([0, 0, 0, .35, .35, .65, .65, 1, 1])
    if quick:
        # Follow the outward step; do not preload by adducting the support hip.
        com_keys = np.array([centers[0], centers[0], centers[0] + [0, .015],
                             centers[0] + [0, .015], centers[1], centers[1], centers[1]])
        progress_keys = np.array([0, 0, .35, .35, .7, 1, 1])
    slerps = [Slerp([0, 1], Rotation.from_matrix(rots[:, side])) for side in range(2)]
    lower = np.r_[-.3, -.3, .35, -.7, -.15, -.4, solver.limits[:, 0]]
    upper = np.r_[.3, .3, .65, .7, .65, .4, solver.limits[:, 1]]
    roll_ids = np.array([NAMES.index("joint_L2"), NAMES.index("joint_R2")]) + 6
    free = np.arange(18)
    timestamps = np.arange(round(times[-1] * fps) + 1) / fps
    rows, targets, contacts, phases, com_targets = [], [], [], [], []
    x = start.copy()
    for frame, t in enumerate(timestamps):
        phase = min(int(np.searchsorted(times[1:], t, side="right")), len(labels) - 1)
        u = np.clip((t - times[phase]) / (times[phase + 1] - times[phase]), 0, 1)
        w = smooth(u)
        progress = (1 - w) * progress_keys[phase] + w * progress_keys[phase + 1]
        reference = (1 - progress) * start + progress * end
        target_com = (1 - w) * com_keys[phase] + w * com_keys[phase + 1]
        step = smooth(np.clip((t - np.array(lift_times)) / swing_duration, 0, 1))
        if quick:
            free = np.setdiff1d(np.arange(18), roll_ids[t < np.array(lift_times)])
            lower[roll_ids[0]] = min(upper[roll_ids[0]] - 1e-8, max(0, x[roll_ids[0]]))
            upper[roll_ids[1]] = max(lower[roll_ids[1]] + 1e-8, min(0, x[roll_ids[1]]))
        target = (1 - step[:, None]) * feet[0] + step[:, None] * feet[1]
        contact = np.ones(2, dtype=bool)
        for side, lift_time in enumerate(lift_times):
            swing = (t - lift_time) / swing_duration
            if lift_time + 1e-9 < t < lift_time + swing_duration - 1e-9:
                target[side, 2] += lift_height * 64 * swing**3 * (1 - swing)**3
                contact[side] = False
        target_rot = np.stack([slerps[side]([step[side]]).as_matrix()[0] for side in range(2)])

        def unpack(values):
            candidate = x.copy()
            candidate[free] = values
            return candidate

        def residual(values):
            candidate = unpack(values)
            set_state(solver, candidate)
            foot_error = solver.data.xpos[solver.feet] - target
            rotation_error = solver.data.xmat[solver.feet].reshape(2, 3, 3) - target_rot
            return np.r_[foot_error.ravel() / .0003, rotation_error.ravel() / .003,
                         (com(solver)[:2] - target_com) / .04,
                         (candidate[2:6] - reference[2:6]) * [5, 12, 12, 12],
                         candidate[6:] - reference[6:],
                         (candidate[roll_ids] - step * end[roll_ids]) * 20 if quick else []]

        if phase == 0:
            x = start.copy()
        elif phase == len(labels) - 1:
            x = end.copy()
        else:
            result = least_squares(residual, np.clip(x[free], lower[free] + 1e-10, upper[free] - 1e-10),
                                   bounds=(lower[free], upper[free]), max_nfev=90,
                                   ftol=1e-7, xtol=1e-7, gtol=1e-6)
            x = unpack(result.x)
        rows.append(x.copy())
        targets.append(target)
        contacts.append(contact)
        phases.append(phase)
        com_targets.append(target_com)
        if frame % fps == 0:
            print(f"{t:4.1f}s {labels[phase]}", flush=True)
    xs = np.array(rows)
    return {"time": timestamps, "joint_names": np.array(NAMES), "root_pos": xs[:, :3],
            "root_quat_wxyz": Rotation.from_euler("xyz", xs[:, 3:6]).as_quat(scalar_first=True),
            "joint_pos": xs[:, 6:], "planned_contact": np.array(contacts),
            "phase": np.array(phases), "foot_target_pos": np.array(targets),
            "com_target_xy": np.array(com_targets), "variant": np.array(variant),
            "segment_times": times, "segment_labels": np.array(labels),
            "swing_duration": np.array(swing_duration), "lift_height": np.array(lift_height),
            "time_scale": np.array(time_scale),
            **({"measured_crouch_joint_pos": solver.crouch.copy()} if measured else {})}


def validate_and_complete(motion, solver, reverse=False):
    n = len(motion["time"])
    dt = float(motion["time"][1] - motion["time"][0])
    assert motion["joint_names"].tolist() == NAMES
    assert dt > 0 and np.allclose(np.diff(motion["time"]), dt)
    q = motion["joint_pos"]
    assert np.isfinite(q).all() and q.shape == (n, 12)
    assert np.all(q >= solver.limits[:, 0] - 1e-9) and np.all(q <= solver.limits[:, 1] + 1e-9)
    stand_frame, crouch_frame = (-1, 0) if reverse else (0, -1)
    assert np.allclose(q[stand_frame], STAND, atol=1e-8)
    assert np.allclose(q[crouch_frame, solver.constrained], solver.crouch[solver.constrained], atol=1e-8)
    assert np.allclose(np.linalg.norm(motion["root_quat_wxyz"], axis=1), 1)
    qposes, foot_pos, foot_quat, points, margins, actual_com = [], [], [], [], [], []
    for i in range(n):
        solver.data.qpos[:3] = motion["root_pos"][i]
        solver.data.qpos[3:7] = motion["root_quat_wxyz"][i]
        solver.data.qpos[solver.qadr] = q[i]
        mujoco.mj_forward(solver.model, solver.data)
        qposes.append(solver.data.qpos.copy())
        foot_pos.append(solver.data.xpos[solver.feet].copy())
        foot_quat.append(solver.data.xquat[solver.feet].copy())
        p = foot_points(solver)
        points.append(p)
        actual_com.append(com(solver))
        hull = ConvexHull(p[motion["planned_contact"][i], :, :2].reshape(-1, 2))
        margins.append(-np.max(hull.equations[:, :2] @ com(solver)[:2] + hull.equations[:, 2]))
    qposes, foot_pos, points = np.array(qposes), np.array(foot_pos), np.array(points)
    velocities = np.zeros((n - 1, solver.model.nv))
    for i in range(n - 1):
        mujoco.mj_differentiatePos(solver.model, velocities[i], dt, qposes[i], qposes[i + 1])
    qvel = np.vstack((velocities[0], (velocities[:-1] + velocities[1:]) / 2, velocities[-1]))
    dofs = solver.model.jnt_dofadr[[solver.model.joint(name).id for name in NAMES]]
    planted = motion["planned_contact"][1:] & motion["planned_contact"][:-1]
    drift = np.linalg.norm(np.diff(points[:, :, :, :2], axis=0), axis=-1).max(axis=-1)
    slip = [float(drift[:, side][planted[:, side]].sum()) for side in range(2)]
    stats = {"max_foot_target_error_mm": float(np.linalg.norm(foot_pos - motion["foot_target_pos"], axis=-1).max() * 1000),
             "cumulative_planted_foot_drift_mm_LR": (np.array(slip) * 1000).tolist(),
             "min_foot_surface_height_mm": float(points[:, :, :, 2].min() * 1000),
             "min_nominal_com_support_margin_mm": float(min(margins) * 1000),
             "static_support_check_passed": bool(min(margins) >= 0),
             "max_joint_speed_rad_s": float(np.abs(qvel[:, dofs]).max()),
             "max_joint_acceleration_rad_s2": float(np.abs(np.gradient(qvel[:, dofs], dt, axis=0)).max()),
             "max_joint_frame_step_rad": float(np.abs(np.diff(q, axis=0)).max()),
             "crouch_sole_height_spread_mm": float(np.ptp(points[crouch_frame, :, :, 2]) * 1000),
             "crouch_rpy_deg": Rotation.from_quat(motion["root_quat_wxyz"][crouch_frame], scalar_first=True).as_euler("xyz", degrees=True).tolist(),
             "final_rpy_deg": Rotation.from_quat(motion["root_quat_wxyz"][-1], scalar_first=True).as_euler("xyz", degrees=True).tolist()}
    print(json.dumps(stats, indent=2), flush=True)
    assert stats["max_foot_target_error_mm"] < 2, "IK failed to preserve foot trajectory"
    assert stats["min_foot_surface_height_mm"] > -1, "Foot penetrates ground"
    assert max(slip) < .005, "Planted foot slides"
    assert stats["max_joint_frame_step_rad"] < .08, "Discontinuous joint motion"
    if reverse or str(motion.get("variant", "v1")) == "v2":
        outward = q[:, [1, 7]] * [1, -1]
        direction = -1 if reverse else 1
        assert np.all(outward >= -1e-8) and np.all(direction * np.diff(outward, axis=0) >= -1e-7), "Hip roll reverses direction"
        assert np.isclose(float(motion["swing_duration"]), .4 * float(motion.get("time_scale", 1)))
        assert 0 < float(motion["lift_height"]) <= .05
        stats["hip_roll_monotonic_inward" if reverse else "hip_roll_monotonic_outward"] = True
    motion.update(joint_vel=qvel[:, dofs], root_lin_vel_world=qvel[:, :3],
                  root_ang_vel_body=qvel[:, 3:6], foot_pos=foot_pos,
                  foot_quat_wxyz=np.array(foot_quat), com_pos=np.array(actual_com),
                  com_support_margin=np.array(margins))
    return stats


def render(motion, output):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    model = mujoco.MjModel.from_xml_path(str(XML.with_name("nlegs_limit_scene.xml")))
    data = mujoco.MjData(model)
    qadr = model.jnt_qposadr[[model.joint(name).id for name in NAMES]]
    model.vis.global_.offwidth, model.vis.global_.offheight = 640, 640
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [.02, 0, .29]
    camera.distance, camera.elevation = 1.45, -16
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    fps = round(1 / float(motion["time"][1] - motion["time"][0]))
    labels = motion.get("segment_labels", LABELS)
    keyframes = np.linspace(0, len(motion["time"]) - 1, 4).round().astype(int)
    with mujoco.Renderer(model, height=640, width=640) as renderer, imageio.get_writer(
            output / "preview.mp4", fps=fps, codec="libx264", quality=8,
            macro_block_size=1, ffmpeg_params=["-movflags", "+faststart"]) as writer:
        for i, t in enumerate(motion["time"]):
            data.qpos[:3] = motion["root_pos"][i]
            data.qpos[3:7] = motion["root_quat_wxyz"][i]
            data.qpos[qadr] = motion["joint_pos"][i]
            mujoco.mj_forward(model, data)
            views = []
            for azimuth in (25, 90):
                camera.azimuth = azimuth
                renderer.update_scene(data, camera=camera, scene_option=option)
                views.append(renderer.render().copy())
            img = Image.fromarray(np.concatenate(views, axis=1))
            draw = ImageDraw.Draw(img)
            draw.rectangle((0, 0, 1280, 70), fill=(18, 23, 31))
            draw.text((18, 8), f"{t:4.2f} s  |  {labels[motion['phase'][i]]}", font=font, fill="white")
            margin = motion["com_support_margin"][i] * 1000
            support_text = (f"Static COM outside support by {-margin:.0f} mm" if margin < 0
                            else f"Static COM margin {margin:.0f} mm")
            draw.text((18, 39), f"KINEMATIC ONLY | {support_text} | torques not checked", font=font, fill="#f9c869")
            draw.text((18, 606), "Front / three-quarter", font=font, fill="white")
            draw.text((660, 606), "Side", font=font, fill="white")
            writer.append_data(np.asarray(img))
            if i in keyframes:
                img.save(output / f"frame_{i:04d}.png")
    print(f"Video: {output / 'preview.mp4'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--variant", choices=("v1", "v2"), default="v1")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--crouch-snapshot", type=Path, help="Read ten task stops from an L/R q snapshot; solve ankles and base")
    parser.add_argument("--lift-height", type=float)
    parser.add_argument("--time-scale", type=float, default=1.)
    args = parser.parse_args()
    args.output = args.output or OUTPUT.with_name(f"stand_to_crouch_{args.variant}")
    path = args.output / "motion.npz"
    if args.check or args.render_only:
        motion = dict(np.load(path, allow_pickle=False))
        metadata = json.loads((args.output / "metadata.json").read_text())
        assert metadata["model_sha256"] == hashlib.sha256(XML.read_bytes()).hexdigest()
        solver = motion_solver(motion.get("measured_crouch_joint_pos"))
    else:
        if path.exists():
            parser.error(f"Refusing to overwrite {path}; choose a new --output directory")
        measured_q = read_crouch_snapshot(args.crouch_snapshot) if args.crouch_snapshot else None
        solver = motion_solver(measured_q)
        motion = generate(solver, args.variant, args.lift_height, args.time_scale, measured_q is not None)
    stats = validate_and_complete(motion, solver)
    if args.check:
        print("PASS: joint limits, endpoint, continuity, foot clearance and planted-foot drift")
        return
    if not args.render_only:
        args.output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **motion)
        metadata = {"version": 1, "variant": args.variant, "validation_level": "kinematic",
                    "fps": round(1 / float(motion["time"][1] - motion["time"][0])),
                    "duration_s": float(motion["time"][-1]), "model": str(XML),
                    "swing_duration_s": float(motion["swing_duration"]),
                    "lift_height_m": float(motion["lift_height"]),
                    "time_scale": float(motion["time_scale"]),
                    "crouch_snapshot": str(args.crouch_snapshot) if args.crouch_snapshot else None,
                    "crouch_snapshot_sha256": hashlib.sha256(args.crouch_snapshot.read_bytes()).hexdigest() if args.crouch_snapshot else None,
                    "measured_joint_pos": motion["measured_crouch_joint_pos"].tolist() if "measured_crouch_joint_pos" in motion else None,
                    "crouch_joint_pos": motion["joint_pos"][-1].tolist(),
                    "model_sha256": hashlib.sha256(XML.read_bytes()).hexdigest(),
                    "joint_order": NAMES, "foot_order": ["left", "right"],
                    "quaternion_order": "wxyz", "units": "m, rad, s",
                    "payload_for_com_kg": 2.5,
                    "segments": [{"name": str(label), "start_s": float(motion["segment_times"][i]),
                                  "end_s": float(motion["segment_times"][i+1])}
                                 for i, label in enumerate(motion["segment_labels"])],
                    "checks": stats,
                    "limitations": ["FK/IK replay, not a simulated policy or actuator rollout.",
                                    "No torque, friction-cone, contact-force or dynamic tracking validation.",
                                    "COM support check is nominal/static, not a dynamic stability certificate.",
                                    "Only feet have collision geometry; full-body self-collision is not certified.",
                                    "Exact ten-joint stops are preserved; see checks.crouch_sole_height_spread_mm for nominal sole non-coplanarity. Ankles/base are solved, not copied from the IMU snapshot.",
                                    "Reference data only; not directly executable on hardware or a ready-made Mimic format."]}
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Motion: {path}", flush=True)
    if not args.no_render:
        render(motion, args.output)


if __name__ == "__main__":
    main()
