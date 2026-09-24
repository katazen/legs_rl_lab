"""Convert a kinematic reference into named full-body Mimic data, without Isaac Sim."""

import argparse
import hashlib
from pathlib import Path

import mujoco
import numpy as np

from generate_crouch_pose_bank import XML

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "source/legs_rl_lab/legs_rl_lab/tasks/mimic_task/task/nlegs_crouch/motions/stand_to_crouch_v3.npz"


def convert(source, output, model_file=XML):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    with np.load(source, allow_pickle=False) as loaded:
        motion = dict(loaded)
    model = mujoco.MjModel.from_xml_path(str(model_file))
    data = mujoco.MjData(model)
    names = motion["joint_names"].tolist()
    joints = [model.joint(name).id for name in names]
    dt = float(np.diff(motion["time"])[0])
    assert dt > 0 and np.allclose(np.diff(motion["time"]), dt)
    body_ids = list(range(1, model.nbody))
    log = {name: [] for name in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")}
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for i in range(len(motion["time"])):
        data.qpos[:3] = motion["root_pos"][i]
        data.qpos[3:7] = motion["root_quat_wxyz"][i]
        data.qpos[model.jnt_qposadr[joints]] = motion["joint_pos"][i]
        data.qvel[:3] = motion["root_lin_vel_world"][i]
        data.qvel[3:6] = motion["root_ang_vel_body"][i]
        data.qvel[model.jnt_dofadr[joints]] = motion["joint_vel"][i]
        mujoco.mj_forward(model, data)
        log["body_pos_w"].append(data.xpos[body_ids].copy())
        log["body_quat_w"].append(data.xquat[body_ids].copy())
        linear, angular = [], []
        for body in body_ids:
            mujoco.mj_jacBody(model, data, jp, jr, body)
            linear.append(jp @ data.qvel)
            angular.append(jr @ data.qvel)
        log["body_lin_vel_w"].append(linear)
        log["body_ang_vel_w"].append(angular)
    result = {key: np.asarray(value, dtype=np.float32) for key, value in log.items()}
    result.update(fps=np.array(1 / dt), joint_names=np.array(names),
                  body_names=np.array([model.body(i).name for i in body_ids]),
                  joint_pos=motion["joint_pos"].astype(np.float32),
                  joint_vel=motion["joint_vel"].astype(np.float32),
                  velocity_frame=np.array("world_link_origin"),
                  validation_level=np.array("kinematic"), source_sha256=np.array(hashlib.sha256(source.read_bytes()).hexdigest()),
                  model_sha256=np.array(hashlib.sha256(model_file.read_bytes()).hexdigest()))
    assert np.allclose(result["body_pos_w"][:, 0], motion["root_pos"], atol=1e-7)
    assert all(np.isfinite(v).all() for v in result.values() if v.dtype.kind == "f")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result)
    print(f"Saved {len(motion['time'])} frames at {1 / dt:g} Hz: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "datasets/stand_to_crouch_v3/motion.npz")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    convert(args.input, args.output)
