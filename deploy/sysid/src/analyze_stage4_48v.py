#!/usr/bin/env python3
"""Stage 4 (48 V): step limits/delay and ankle-pitch friction summary."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_ankle_friction import fit_line, hysteresis_points


ROOT = Path(__file__).resolve().parents[1] / "doc"
S4 = ROOT / "stage4_48v"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def step_metrics(data_dir: Path, stem: str, joint: int, amplitudes: tuple[float, ...]) -> dict:
    command = read(data_dir / f"{stem}_cmd.csv")
    state = read(data_dir / f"{stem}_state.csv")
    ct = np.array([float(row["t"]) for row in command])
    target = np.array([float(row[f"qd{joint}"]) for row in command])
    excite = np.array([row["phase"] == "excite" for row in command])
    st = np.array([float(row["t"]) for row in state])
    velocity = np.array([float(row[f"v{joint}"]) for row in state])
    torque = np.array([float(row[f"tau{joint}"]) for row in state])
    edges = np.where(excite[1:] & (np.abs(np.diff(target)) > 0.005))[0] + 1
    edges = [index for index in edges if abs(target[index - 1]) < 0.02 and abs(target[index]) > 0.02]
    assert edges, f"no step edges in {stem}"

    result = {"rate_hz": 1 / np.median(np.diff(st)), "amplitudes": {}, "delays_ms": []}
    for amplitude in amplitudes:
        peaks_v, peaks_tau, delays = [], [], []
        for index in edges:
            if abs(abs(target[index]) - amplitude) > 0.01:
                continue
            edge = ct[index]
            response = (st >= edge - 0.05) & (st <= edge + 0.60)
            torque_window = (st >= edge) & (st <= edge + 0.25)
            peak_v = np.max(np.abs(velocity[response]))
            peaks_v.append(peak_v)
            peaks_tau.append(np.max(np.abs(torque[torque_window])))
            if amplitude <= 0.10:
                threshold = max(0.02, 0.10 * peak_v)
                onset = np.where(response & (np.abs(velocity) > threshold))[0]
                if len(onset) and 0 <= st[onset[0]] - edge <= 0.2:
                    delays.append((st[onset[0]] - edge) * 1000)
        assert peaks_v, f"missing amplitude {amplitude:g} in {stem}"
        result["amplitudes"][amplitude] = {
            "peak_v_mean": float(np.mean(peaks_v)),
            "peak_v_std": float(np.std(peaks_v)),
            "peak_tau_mean": float(np.mean(peaks_tau)),
            "peak_tau_std": float(np.std(peaks_tau)),
            "repeats": len(peaks_v),
        }
        result["delays_ms"].extend(delays)
    return result


def write_step_summary() -> None:
    out = S4 / "4a"
    current = out / "data"
    cases = {
        "big_L_48V": (current, "j3_kp250_kd5_step_bigmotor48_r1", 3, (0.05, 0.10, 0.20, 0.35)),
        "big_R_48V": (current, "j9_kp250_kd5_step_bigmotor48_r1", 9, (0.05, 0.10, 0.20, 0.35)),
        "small_L_48V_low": (current, "j5_kp40_kd0.5_step_smallmotor48_r1", 5, (0.05, 0.10, 0.15, 0.20)),
        "small_L_48V_high": (current, "j5_kp40_kd0.5_step_smallmotor48_high_r1", 5, (0.25, 0.30)),
        "small_R_48V": (current, "j11_kp40_kd0.5_step_smallmotor48_r1", 11, (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)),
        "big_L_24V": (ROOT / "stage1/data", "j3_kp250_kd5_step_step_bi", 3, (0.05, 0.10, 0.20, 0.35)),
        "big_R_24V": (ROOT / "stage1/data", "j9_kp250_kd5_step_step_bi", 9, (0.05, 0.10, 0.20, 0.35)),
    }
    results = {name: step_metrics(*args) for name, args in cases.items()}

    with (current / "step_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("run", "rate_hz", "amplitude_rad", "peak_v_mean", "peak_v_std",
                         "peak_tau_mean", "peak_tau_std", "repeats", "onset_delay_mean_ms",
                         "onset_delay_std_ms"))
        for name, result in results.items():
            delays = result["delays_ms"]
            for amplitude, metrics in result["amplitudes"].items():
                writer.writerow((name, result["rate_hz"], amplitude, *metrics.values(),
                                 np.mean(delays) if delays else "", np.std(delays) if delays else ""))

    for metric, ylabel, filename in (
        ("peak_v_mean", "peak |velocity| [rad/s]", "step_peak_velocity.png"),
        ("peak_tau_mean", "peak |reported torque| [N m]", "step_peak_torque.png"),
    ):
        fig, axis = plt.subplots(figsize=(8, 5))
        for name, result in results.items():
            amplitudes = sorted(result["amplitudes"])
            values = [result["amplitudes"][amp][metric] for amp in amplitudes]
            axis.plot(amplitudes, values, "-o", label=name)
        axis.set(xlabel="step amplitude [rad]", ylabel=ylabel)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "png" / filename, dpi=140)
        plt.close(fig)


def write_friction_summary() -> None:
    data_dir = S4 / "4c/data"
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, joint in zip(axes, (4, 10)):
        source_points = {}
        for source in ("feedback", "pd"):
            groups = []
            for tag in ("amp003_48_r1", "amp005_48_r1", "amp010_48_r1"):
                groups.extend(result["points"] for result in hysteresis_points(
                    data_dir, joint, tag, (0.5, 0.8, 1.2, 2.0), 8, (0.5, 0.8), source
                ) if len(result["points"]))
            direction = np.sign(np.median(np.concatenate([points[:, 1] for points in groups])))
            groups = [np.c_[points[:, 0], direction * points[:, 1]] for points in groups]
            combined = np.concatenate(groups)
            fc, viscous = fit_line(combined)
            leave_one_out = [fit_line(np.concatenate([group for k, group in enumerate(groups) if k != i]))[0]
                             for i in range(len(groups))]
            rows.append((joint, source, fc, viscous, min(leave_one_out), max(leave_one_out), len(combined)))
            source_points[source] = (combined, fc, viscous)
        points, fc, viscous = source_points["feedback"]
        axis.scatter(points[:, 0], points[:, 1], s=10, alpha=0.35)
        speed = np.linspace(0, points[:, 0].max(), 100)
        axis.plot(speed, fc + viscous * speed, "k-", label=f"feedback Fc={fc:.3f} N m")
        pd_fc = source_points["pd"][1]
        axis.set(title=f"ankle pitch joint {joint} (PD Fc={pd_fc:.3f} N m)",
                 xlabel="paired |velocity| [rad/s]", ylabel="half torque separation [N m]")
        axis.grid(alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(S4 / "4c/png/friction_48v.png", dpi=140)
    plt.close(fig)

    with (data_dir / "friction_48v.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("joint", "torque_source", "fc_nm", "viscous_nms_rad",
                         "leave_one_condition_out_min", "leave_one_condition_out_max", "points"))
        writer.writerows(rows)


if __name__ == "__main__":
    synthetic = np.c_[np.linspace(0.01, 0.3, 20), 0.5 + 0.3 * np.linspace(0.01, 0.3, 20)]
    assert np.allclose(fit_line(synthetic), (0.5, 0.3))
    write_step_summary()
    write_friction_summary()
