"""Reverse the preserved low-step v2 into a separate crouch -> stand reference.

unitree_lab Python; --check verifies saved data without rendering or writing.
Kinematics only: no controller, motor I/O or claim of dynamic feasibility.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from generate_stand_to_crouch_motion import (
    XML, generate, motion_solver, read_crouch_snapshot, render, validate_and_complete,
)
from prepare_crouch_mimic import convert

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets/stand_to_crouch_v2/motion.npz"
VELOCITIES = ("joint_vel", "root_lin_vel_world", "root_ang_vel_body")
LABELS = ["Hold crouch", "Straighten torso", "Low step right in", "Touch down",
          "Low step left in", "Hold stand"]


def reverse_motion(source, initial_hold):
    assert str(source["variant"]) == "v2"
    dt = float(source["time"][1] - source["time"][0])
    end = float(source["segment_times"][-2]) + initial_hold
    assert np.isfinite(initial_hold) and dt <= initial_hold <= np.diff(source["segment_times"])[-1]
    assert np.isclose(end / dt, round(end / dt)), "Hold must align with the frame interval"
    indexes = np.arange(round(end / dt), -1, -1)
    motion = {key: value[indexes].copy() if value.ndim and value.shape[0] == len(source["time"])
              else value.copy() for key, value in source.items()}
    motion["time"] = np.arange(len(indexes)) * dt
    motion["segment_times"] = np.r_[0., end - source["segment_times"][-2::-1]]
    motion["segment_labels"] = np.asarray(LABELS)
    motion["phase"] = np.minimum(np.searchsorted(motion["segment_times"][1:], motion["time"], side="right"),
                                 len(LABELS) - 1)
    motion["variant"] = np.array("v2_reversed")
    motion["source_frame"] = indexes
    for key in VELOCITIES:
        motion[key] *= -1
    return motion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/crouch_to_stand_v1")
    parser.add_argument("--initial-hold", type=float, default=.4)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--crouch-snapshot", type=Path)
    parser.add_argument("--lift-height", type=float, default=.02)
    parser.add_argument("--time-scale", type=float, default=1.25)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    if not args.check and args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}; choose a new --output directory")
    metadata = json.loads(SOURCE.with_name("metadata.json").read_text())
    if args.check:
        saved_metadata = json.loads((args.output / "metadata.json").read_text())
        assert saved_metadata["model_sha256"] == hashlib.sha256(XML.read_bytes()).hexdigest(), "Saved model changed"
        if not args.crouch_snapshot and saved_metadata.get("crouch_snapshot"):
            args.crouch_snapshot = ROOT / saved_metadata["crouch_snapshot"]
            args.lift_height = saved_metadata["lift_height_m"]
            args.time_scale = saved_metadata["time_scale"]
            args.initial_hold = saved_metadata["initial_hold_s"] / args.time_scale
    measured_q = read_crouch_snapshot(args.crouch_snapshot) if args.crouch_snapshot else None
    solver = motion_solver(measured_q)
    if measured_q is not None:
        if args.check:
            assert saved_metadata["crouch_snapshot_sha256"] == hashlib.sha256(args.crouch_snapshot.read_bytes()).hexdigest()
        source = generate(solver, "v2", args.lift_height, args.time_scale, measured=True)
        validate_and_complete(source, solver)
    else:
        assert metadata["model_sha256"] == hashlib.sha256(XML.read_bytes()).hexdigest(), "Source model changed"
        with np.load(SOURCE, allow_pickle=False) as loaded:
            source = dict(loaded)
    initial_hold = args.initial_hold * args.time_scale if measured_q is not None else args.initial_hold
    motion = reverse_motion(source, initial_hold)
    reversed_velocities = {key: motion[key].copy() for key in VELOCITIES}
    stats = validate_and_complete(motion, solver, reverse=True)
    for key in VELOCITIES:
        np.testing.assert_allclose(motion[key], reversed_velocities[key], rtol=0., atol=1e-9)
        assert np.allclose(motion[key][[0, -1]], 0., atol=1e-8)
    contacts = motion["planned_contact"]
    right_swing = np.flatnonzero(~contacts[:, 1])
    left_swing = np.flatnonzero(~contacts[:, 0])
    assert len(right_swing) and len(left_swing) and right_swing[-1] < left_swing[0]
    assert contacts.any(axis=1).all()
    if args.check:
        with np.load(args.output / "motion.npz", allow_pickle=False) as saved:
            assert set(saved.files) == set(motion)
            for key, value in motion.items():
                np.testing.assert_array_equal(saved[key], value)
        with np.load(args.output / "mimic_motion.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["joint_pos"], motion["joint_pos"].astype(np.float32))
            np.testing.assert_array_equal(saved["joint_vel"], motion["joint_vel"].astype(np.float32))
            assert str(saved["source_sha256"]) == hashlib.sha256((args.output / "motion.npz").read_bytes()).hexdigest()
        print("PASS: saved reversal, endpoints, velocities, right-then-left contacts, limits and foot geometry")
        return
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "motion.npz", **motion)
    metadata.update(variant="v2_reversed", direction="crouch_to_stand", initial_hold_s=initial_hold,
                    source=str(SOURCE.relative_to(ROOT)), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    duration_s=float(motion["time"][-1]), checks=stats,
                    segments=[{"name": label, "start_s": float(motion["segment_times"][i]),
                               "end_s": float(motion["segment_times"][i + 1])} for i, label in enumerate(LABELS)])
    metadata.update(model_sha256=hashlib.sha256(XML.read_bytes()).hexdigest(),
                    swing_duration_s=float(motion["swing_duration"]), lift_height_m=float(motion["lift_height"]),
                    time_scale=float(motion.get("time_scale", 1)),
                    crouch_snapshot=str(args.crouch_snapshot) if args.crouch_snapshot else None,
                    crouch_snapshot_sha256=hashlib.sha256(args.crouch_snapshot.read_bytes()).hexdigest() if args.crouch_snapshot else None,
                    measured_joint_pos=measured_q.tolist() if measured_q is not None else None,
                    crouch_joint_pos=motion["joint_pos"][0].tolist(),
                    regenerated_from_v2_recipe=measured_q is not None)
    metadata["limitations"][-2] = "Ten task stops are exact; ankles/base are solved in nominal geometry. See checks.crouch_sole_height_spread_mm."
    metadata["limitations"][-1] = "Reference only; mimic_motion.npz is a format conversion, not a trained policy or hardware command."
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    convert(args.output / "motion.npz", args.output / "mimic_motion.npz")
    if not args.no_render:
        render(motion, args.output)


if __name__ == "__main__":
    main()
