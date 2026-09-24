"""Capture a four-step stair descent from the trained rough walking policy in MuJoCo.

The result is a simulated rollout for Mimic, not a verified real-robot command.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from generate_crouch_pose_bank import NAMES
from generate_stand_to_crouch_motion import foot_points
from prepare_crouch_mimic import convert

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "source/legs_rl_lab/legs_rl_lab/assets/nlegs/mjcf"
MODEL_XML = ASSET / "nlegs.xml"
SCENE_XML = ASSET / "nlegs_scene.xml"
ROUGH_SIM2SIM = ROOT / "source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/task/rough/sim2sim.py"
RUN = "2026-09-20_19-15-58"
OUTPUT = ROOT / "datasets/stair_descent_policy_v2"
STEP_H, STEP_D, WIDTH = .05, .20, 1.0
TOP_D = .30
START_X, START_Z, COMMAND_VX = .15, .785, .5
END_TIME = 2.12
LATENCY_SEED = 14


def terrain_height(x):
    x = np.asarray(x)
    return np.select([x < TOP_D, x < TOP_D + STEP_D,
                      x < TOP_D + 2 * STEP_D, x < TOP_D + 3 * STEP_D],
                     [.20, .15, .10, .05], default=0.)


def stair_model():
    spec = mujoco.MjSpec.from_file(str(SCENE_XML))
    for i, (start, depth, height) in enumerate(
            [(0., TOP_D, .20), (.30, STEP_D, .15), (.50, STEP_D, .10), (.70, STEP_D, .05)]):
        spec.worldbody.add_geom(name=f"stair_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                pos=[start + depth / 2, 0, height / 2],
                                size=[depth / 2, WIDTH / 2, height / 2],
                                rgba=[.75 - .12 * i] * 3 + [1.])
    return spec.compile()


def solver_for(model):
    feet = [model.body(f"Link_{side}6").id for side in ("L", "R")]
    return SimpleNamespace(model=model, data=mujoco.MjData(model), feet=feet,
                           limits=model.jnt_range[[model.joint(name).id for name in NAMES]],
                           caps=np.flatnonzero(np.isin(model.geom_bodyid, feet)
                                               & (model.geom_contype != 0)))


def load_rough_sim2sim():
    spec = importlib.util.spec_from_file_location("nlegs_stair_rough_sim2sim", ROUGH_SIM2SIM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_policy(run):
    rough = load_rough_sim2sim()

    class StairRunner(rough.flat.MujocoRunner):
        def _make_model(self, cfg):
            return stair_model()

        def _save_tracking(self):
            pass  # Capture into this task's output, not the training run's log directory.

    np.random.seed(LATENCY_SEED)  # Fixed actuator-delay sample for reproducibility.
    cfg = rough.flat.load_config(run)
    cfg.base_pos[:] = [START_X, 0., START_Z]
    runner = StairRunner(cfg, show_viewer=False, save_data=True)
    runner.command[:] = [COMMAND_VX, 0., 0.]
    runner.run(duration=END_TIME + .02, realtime=False)
    rows = [row for row in runner.track_log if row[0] <= END_TIME + 1e-6]
    if not rows or abs(rows[-1][0] - END_TIME) > .001:
        raise ValueError("Missing the final double-contact frame")
    return runner, rows


def contiguous_durations(mask, dt):
    ids = np.flatnonzero(mask)
    return [len(group) * dt for group in np.split(ids, np.where(np.diff(ids) > 1)[0] + 1) if len(group)]


def extract(runner, rows):
    solver = solver_for(runner.model)
    if len(solver.caps) != 6:
        raise ValueError("Expected six foot collision capsules")
    sdk_names = [f"joint_{name}" for name in runner.cfg.short_joint_names]
    order = [sdk_names.index(name) for name in NAMES]
    pose = np.array([row[7] for row in rows])
    q = np.array([row[2] for row in rows])[:, order]
    qd = np.array([row[3] for row in rows])[:, order]
    base_vel = np.array([row[8] for row in rows])
    force = np.abs(np.array([row[6][:2][::-1] for row in rows]))
    time = np.array([row[0] for row in rows])
    fps = 1 / np.median(np.diff(time))
    sole_x, sole_z, clearance = [], [], []
    for row in rows:
        solver.data.qpos[:3] = row[7][:3]
        solver.data.qpos[3:7] = row[7][3:]
        solver.data.qpos[runner.qpos_adr] = row[2]
        mujoco.mj_forward(runner.model, solver.data)
        points = foot_points(solver)
        sole_x.append(points[:, :, 0].mean(axis=1))
        sole_z.append(points[:, :, 2].min(axis=1))
        clearance.append((points[:, :, 2] - terrain_height(points[:, :, 0])).min(axis=1))
    sole_x, sole_z, clearance = map(np.asarray, (sole_x, sole_z, clearance))
    contact = force > 10
    events, seen = [], set()
    for i, t in enumerate(time):
        for side in range(2):
            if not contact[i, side]:
                continue
            level = int(round(float(terrain_height(sole_x[i, side])) / STEP_H))
            if level >= 4 or abs(sole_z[i, side] - level * STEP_H) > .025:
                continue
            key = (side, level)
            if key not in seen:
                events.append(dict(time_s=round(float(t - time[0]), 2), foot="L" if side == 0 else "R",
                                   height_cm=level * 5, x_m=round(float(sole_x[i, side]), 3)))
                seen.add(key)
    expected = [("R", 15), ("L", 10), ("R", 5), ("L", 0), ("R", 0)]
    if [(event["foot"], event["height_cm"]) for event in events] != expected:
        raise ValueError(f"Not one alternating landing per stair: {events}")

    angles = np.rad2deg(Rotation.from_quat(pose[:, 3:], scalar_first=True).as_euler("xyz"))
    swing = ~contact
    swing_durations = [contiguous_durations(swing[:, side], 1 / fps) for side in range(2)]
    stats = dict(fps=round(float(fps), 3), frames=len(rows), duration_s=round(float(time[-1] - time[0]), 3),
                 landings=events, swing_s=[[round(float(value), 3) for value in side] for side in swing_durations],
                 max_both_feet_airborne_s=round(max(contiguous_durations(swing.all(axis=1), 1 / fps), default=0), 3),
                 max_abs_base_y_m=round(float(np.abs(pose[:, 1]).max()), 4),
                 max_abs_roll_deg=round(float(np.abs(angles[:, 0]).max()), 2),
                 max_abs_pitch_deg=round(float(np.abs(angles[:, 1]).max()), 2),
                 max_abs_yaw_deg=round(float(np.abs(angles[:, 2]).max()), 2),
                 max_abs_knee_speed_rad_s=round(float(np.abs(qd[:, [3, 9]]).max()), 3),
                 min_sole_clearance_m=round(float(clearance.min()), 4))
    print(json.dumps(stats, indent=2), flush=True)
    if (stats["max_abs_base_y_m"] > .06 or stats["max_abs_roll_deg"] > 10
            or stats["max_abs_knee_speed_rad_s"] > 8.2 or stats["min_sole_clearance_m"] < -.025
            or not contact[-1].all()):
        raise ValueError("Descent rollout failed lateral/velocity/contact checks")
    if np.any(q < solver.limits[:, 0] - 1e-6) or np.any(q > solver.limits[:, 1] + 1e-6):
        raise ValueError("Rollout exceeds nlegs.xml joint ranges")
    motion = dict(time=time - time[0], joint_names=np.asarray(NAMES), root_pos=pose[:, :3],
                  root_quat_wxyz=pose[:, 3:], joint_pos=q, joint_vel=qd,
                  root_lin_vel_world=base_vel[:, :3], root_ang_vel_body=base_vel[:, 3:],
                  foot_sole_x=sole_x, foot_sole_min_z=sole_z, foot_clearance=clearance,
                  foot_force_z_magnitude=force, measured_contact=contact,
                  stair_width=np.array(WIDTH), stair_depth=np.array(STEP_D), stair_height=np.array(STEP_H),
                  command_vx=np.array(COMMAND_VX))
    return motion, stats


def render(motion, output, events):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    model = stair_model()
    data = mujoco.MjData(model)
    qadr = model.jnt_qposadr[[model.joint(name).id for name in NAMES]]
    model.vis.global_.offwidth = 640
    model.vis.global_.offheight = 480
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [.65, 0, .45]
    camera.distance, camera.elevation = 2.15, -14
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    picks = set(np.linspace(0, len(motion["time"]) - 1, 8, dtype=int))
    with mujoco.Renderer(model, height=480, width=640) as renderer, imageio.get_writer(
            output / "preview.mp4", fps=50, codec="libx264", quality=8,
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
            image = Image.fromarray(np.concatenate(views, axis=1))
            draw = ImageDraw.Draw(image)
            landed = sum(t >= event["time_s"] for event in events)
            draw.text((16, 12), f"{t:.2f}s | landings {landed}/{len(events)}",
                      fill="white", stroke_width=2, stroke_fill="black")
            writer.append_data(np.asarray(image))
            if i in picks:
                image.save(output / f"frame_{i:04d}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=RUN)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing {args.output}")
    runner, rows = run_policy(args.run)
    motion, stats = extract(runner, rows)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "motion.npz", **motion)
    convert(args.output / "motion.npz", args.output / "mimic_motion.npz", MODEL_XML)
    (args.output / "metadata.json").write_text(json.dumps({
        "source_run": args.run, "terrain": "four 5 cm drops, 20 cm tread, 1 m width",
        "checks": stats,
        "limitations": "MuJoCo policy rollout only; real-robot stability and Mimic tracking unverified.",
    }, indent=2) + "\n")
    if not args.no_render:
        render(motion, args.output, stats["landings"])
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
