"""Reverse the preserved low-step v2 into a separate crouch -> stand reference.

unitree_lab Python; --check verifies saved data without rendering or writing.
Kinematics only: no controller, motor I/O or claim of dynamic feasibility.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from generate_stand_to_crouch_motion import PoseSolver, XML, render, validate_and_complete
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
    args = parser.parse_args()
    if not args.check and args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}; choose a new --output directory")
    metadata = json.loads(SOURCE.with_name("metadata.json").read_text())
    assert metadata["model_sha256"] == hashlib.sha256(XML.read_bytes()).hexdigest(), "Source model changed"
    with np.load(SOURCE, allow_pickle=False) as loaded:
        source = dict(loaded)
    motion = reverse_motion(source, args.initial_hold)
    reversed_velocities = {key: motion[key].copy() for key in VELOCITIES}
    stats = validate_and_complete(motion, PoseSolver(), reverse=True)
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
    metadata.update(variant="v2_reversed", direction="crouch_to_stand", initial_hold_s=args.initial_hold,
                    source=str(SOURCE.relative_to(ROOT)), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    duration_s=float(motion["time"][-1]), checks=stats,
                    segments=[{"name": label, "start_s": float(motion["segment_times"][i]),
                               "end_s": float(motion["segment_times"][i + 1])} for i, label in enumerate(LABELS)])
    metadata["limitations"][-1] = "Reference only; mimic_motion.npz is a format conversion, not a trained policy or hardware command."
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    convert(args.output / "motion.npz", args.output / "mimic_motion.npz")
    render(motion, args.output)


if __name__ == "__main__":
    main()
