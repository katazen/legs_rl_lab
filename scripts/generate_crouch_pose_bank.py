"""离线生成平地 crouch reset 姿态表；只做 MuJoCo FK/优化，不连接实机。

用 unitree_lab 的 Python 运行。默认输出到 nlegs 资产目录；--check 复核已有表。
关节顺序显式存名字，四元数 WXYZ。碰撞检查仅覆盖模型已有碰撞体（目前只有脚）。
"""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ASSET = Path(__file__).resolve().parents[1] / "source/legs_rl_lab/legs_rl_lab/assets/nlegs"
XML = ASSET / "mjcf/nlegs_limit.xml"
BANK = ASSET / "crouch_pose_bank.json"
NAMES = [f"joint_{side}{i}" for side in ("L", "R") for i in range(1, 7)]
STAND = np.array([-0.1, 0, 0, 0.2, -0.1, 0] * 2, dtype=float)
CROUCH_LIMIT_SIDE = {
    "joint_L1": 0, "joint_L2": 1, "joint_L3": 1, "joint_L4": 1, "joint_L5": 0,
    "joint_R1": 0, "joint_R2": 0, "joint_R3": 0, "joint_R4": 1, "joint_R5": 0,
}  # 0=XML 下限，1=XML 上限；ankle roll 不作为限位姿态约束
CLEARANCE = 0.002
NEAR_LIMIT_DEG = 3.0
NEAR_STAND_DEG = 6.0


def model_hashes():
    usd_dir = ASSET / "mjcf/nlegs_limit"
    files = [XML, usd_dir / "nlegs_limit.usd", *sorted((usd_dir / "configuration").glob("*.usd"))]
    return {str(p.relative_to(ASSET)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


class PoseSolver:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(XML))
        self.data = mujoco.MjData(self.model)
        joint_ids = [self.model.joint(name).id for name in NAMES]
        self.qadr = self.model.jnt_qposadr[joint_ids]
        self.limits = self.model.jnt_range[joint_ids].copy()
        self.constrained = [NAMES.index(name) for name in CROUCH_LIMIT_SIDE]
        self.crouch = STAND.copy()
        for name, side in CROUCH_LIMIT_SIDE.items():
            i = NAMES.index(name)
            self.crouch[i] = self.limits[i, side]
        self.feet = [self.model.body(f"Link_{side}6").id for side in ("L", "R")]
        self.base = self.model.body("base").id
        self.caps = np.flatnonzero(
            np.isin(self.model.geom_bodyid, self.feet) & (self.model.geom_contype != 0)
        )
        assert len(self.caps) == 6
        self.mass = self.model.body_mass.sum()

    def classify(self, q):
        gap = np.rad2deg(np.abs(q - self.crouch)[self.constrained]).max()
        if gap <= NEAR_LIMIT_DEG:
            return "crouch", float(gap)
        stand_gap = np.rad2deg(np.abs(q - STAND)[self.constrained]).max()
        return ("stand" if stand_gap <= NEAR_STAND_DEG else "intermediate"), float(gap)

    def evaluate(self, q, pitch, roll):
        quat = Rotation.from_euler("xyz", [roll, pitch, 0]).as_quat(scalar_first=True)
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[:3] = 0
        self.data.qpos[3:7] = quat
        self.data.qpos[self.qadr] = q
        mujoco.mj_forward(self.model, self.data)
        # capsule 轴沿 geom 局部 z；两个半球最低点构成足底接触候选点。
        axes = self.data.geom_xmat[self.caps].reshape(-1, 3, 3)[:, :, 2]
        offsets = axes * self.model.geom_size[self.caps, 1, None]
        points = np.stack((self.data.geom_xpos[self.caps] - offsets,
                           self.data.geom_xpos[self.caps] + offsets), axis=1)
        points[:, :, 2] -= self.model.geom_size[self.caps, 0, None]
        points = points.reshape(2, 6, 3)
        normals = self.data.xmat[self.feet].reshape(2, 3, 3)[:, :, 2]
        hull = ConvexHull(points[:, :, :2].reshape(-1, 2))
        # 覆盖本任务继承的 base 额外质量 1~4kg；质心位置不随机。
        com = self.data.subtree_com[self.base]
        base_com = self.data.xipos[self.base]
        margins = []
        for added_mass in (1.0, 4.0):
            xy = ((com * self.mass + added_mass * base_com) / (self.mass + added_mass))[:2]
            margins.append(-np.max(hull.equations[:, :2] @ xy + hull.equations[:, 2]))
        return {
            "normals": normals, "points": points, "quat": quat,
            "height": CLEARANCE - points[:, :, 2].min(),
            "margin": min(margins),
            "height_error": float(np.ptp(points[:, :, 2])),
            "tilt_deg": float(np.rad2deg(np.arccos(np.clip(normals[:, 2], -1, 1))).max()),
            "self_penetration": max([0.0] + [-float(c.dist) for c in self.data.contact]),
        }

    def solve(self, pitch_deg, reference):
        pitch = np.deg2rad(pitch_deg)
        lower = np.r_[self.limits[:, 0] + 0.001, -np.deg2rad(5)]
        upper = np.r_[self.limits[:, 1] - 0.001, np.deg2rad(5)]
        weights = np.ones(12)
        weights[[5, 11]] = 0.1  # ankle roll 优先服从贴地条件，不钉住参考角

        def residual(x):
            state = self.evaluate(x[:12], pitch, x[12])
            z = state["points"][:, :, 2]
            return np.r_[
                state["normals"][:, :2].ravel() / 0.002,
                (z[0].mean() - z[1].mean()) / 0.0002,
                np.maximum(0, 0.008 - state["margin"]) / 0.002,
                weights * (x[:12] - reference), x[12] / 0.1,
            ]

        result = least_squares(residual, np.clip(np.r_[reference, 0], lower, upper),
                               bounds=(lower, upper), max_nfev=100)
        q, roll = result.x[:12], result.x[12]
        state = self.evaluate(q, pitch, roll)
        self.validate(q, state)
        posture, gap = self.classify(q)
        return {
            "posture": posture, "limit_distance_deg": gap,
            "pitch_deg": float(pitch_deg), "height": float(state["height"]),
            "quat_wxyz": state["quat"].tolist(), "joint_pos": q.tolist(),
            "foot_height_error_m": state["height_error"], "foot_tilt_deg": state["tilt_deg"],
            "com_margin_m": float(state["margin"]),
        }

    def solve_limit(self):
        """十个关节固定在 XML 端点，只求身体 roll/pitch 和双踝 roll。"""
        ankle_ids = [NAMES.index(f"joint_{side}6") for side in ("L", "R")]

        def evaluate(x):
            q = self.crouch.copy()
            q[ankle_ids] = x[2:]
            return q, self.evaluate(q, x[1], x[0])

        def residual(x):
            _, state = evaluate(x)
            z = state["points"][:, :, 2]
            return np.r_[state["normals"][:, :2].ravel() / 0.002,
                         (z[0].mean() - z[1].mean()) / 0.0002]

        lower = np.r_[np.deg2rad([-15, -10]), self.limits[ankle_ids, 0]]
        upper = np.r_[np.deg2rad([15, 45]), self.limits[ankle_ids, 1]]
        initial = np.clip([0.0, np.deg2rad(24), 0.0, 0.0], lower, upper)
        result = least_squares(residual, initial, bounds=(lower, upper), max_nfev=200)
        if not result.success:
            raise ValueError(f"精确限位姿态求解失败: {result.message}")
        q, state = evaluate(result.x)
        self.validate(q, state)
        return q, state

    def validate(self, q, state):
        if not (np.isfinite(q).all() and np.isfinite(state["height"])):
            raise ValueError("姿态包含非有限值")
        if np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1]):
            raise ValueError("关节越限")
        if (state["tilt_deg"] > 0.5 or state["height_error"] > 0.001
                or state["margin"] < 0.005 or state["self_penetration"] > 0.0001
                or -self.data.xpos[self.feet, 2].min() < 0.2):
            raise ValueError(f"未通过接触/支撑校验: tilt={state['tilt_deg']:.3f}deg, "
                             f"height_error={state['height_error']:.6f}m, margin={state['margin']:.4f}m")


def check_bank(bank, solver):
    assert bank["version"] == 3
    if bank["model_sha256"] != model_hashes() or bank["joint_names"] != NAMES:
        raise ValueError("模型/关节顺序已改变，请重新生成姿态表")
    counts = {name: 0 for name in ("stand", "intermediate", "crouch")}
    for row in bank["poses"]:
        quat = np.asarray(row["quat_wxyz"])
        assert np.isfinite(quat).all() and abs(np.linalg.norm(quat) - 1) < 1e-6
        roll, pitch, yaw = Rotation.from_quat(quat, scalar_first=True).as_euler("xyz")
        assert abs(yaw) < 1e-6 and abs(pitch - np.deg2rad(row["pitch_deg"])) < 1e-6
        assert 0 <= row["pitch_deg"] <= 24
        posture, gap = solver.classify(np.asarray(row["joint_pos"]))
        assert posture == row["posture"] and abs(gap - row["limit_distance_deg"]) < 1e-6
        state = solver.evaluate(np.asarray(row["joint_pos"]), pitch, roll)
        solver.validate(np.asarray(row["joint_pos"]), state)
        assert abs(state["height"] - row["height"]) < 1e-6
        counts[posture] += 1
    assert all(counts.values())
    assert min(r["pitch_deg"] for r in bank["poses"]) == 0
    assert max(r["pitch_deg"] for r in bank["poses"]) == 24
    assert np.allclose(bank["joint_limits"], solver.limits)
    assert np.allclose(bank["reference_crouch"], solver.crouch)
    print(f"PASS: {len(bank['poses'])} poses {counts}, "
          f"min COM margin={min(p['com_margin_m'] for p in bank['poses']):.4f}m, "
          f"max foot height error={max(p['foot_height_error_m'] for p in bank['poses']):.6f}m")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BANK)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", type=int, default=16)
    args = parser.parse_args()
    solver = PoseSolver()
    if args.check:
        check_bank(json.loads(args.output.read_text()), solver)
        return
    if args.variants < 1:
        parser.error("--variants 必须 >= 1")
    rng = np.random.default_rng(args.seed)
    poses = []
    for pitch_deg in np.linspace(0, 24, 25):
        for variant in range(args.variants):
            reference = STAND + pitch_deg / 24.0 * (solver.crouch - STAND)
            if variant:
                reference += rng.uniform(-0.03, 0.03, 12)
            poses.append(solver.solve(pitch_deg, reference))
        print(f"pitch={pitch_deg:4.1f}deg: {args.variants} valid poses", flush=True)
    bank = {"version": 3, "seed": args.seed, "joint_names": NAMES,
            "model_sha256": model_hashes(), "joint_limits": solver.limits.tolist(),
            "reference_crouch": solver.crouch.tolist(), "clearance_m": CLEARANCE,
            "near_limit_deg": NEAR_LIMIT_DEG, "near_stand_deg": NEAR_STAND_DEG, "poses": poses}
    check_bank(bank, solver)
    args.output.write_text(json.dumps(bank, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
