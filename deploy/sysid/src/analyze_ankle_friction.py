#!/usr/bin/env python3
"""Estimate ankle-pitch Coulomb friction from low-speed torque hysteresis."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_stepsine import excite_start, frf_stepsine


RUNS = (
    ("A=0.05", "frf_anklepitch_fixedA005_r1", (0.2, 0.5, 0.8, 1.2, 2.0), 4, (0.2, 0.5)),
    ("A=0.03", "frf_ankle_small_r1", (0.5, 0.8, 1.2, 2, 3, 4, 6, 8, 10, 12, 16), 6, (0.5, 0.8)),
)


def read_csv(path: Path, columns: tuple[str, ...]) -> dict[str, np.ndarray]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    out = {name: np.array([float(row[name]) for row in rows]) for name in columns}
    out["phase"] = np.array([row["phase"] for row in rows])
    return out


def fit_line(points: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(np.c_[np.ones(len(points)), points[:, 0]], points[:, 1], rcond=None)[0]


def hysteresis_points(data_dir: Path, joint: int, tag: str, freqs, cycles: int, selected,
                      torque_source: str = "feedback") -> list[dict]:
    path = data_dir / f"j{joint}_kp40_kd2_sine_{tag}_state.csv"
    d = read_csv(path, ("t", f"q{joint}", f"v{joint}", f"tau{joint}"))
    t, q, vel, tau = d["t"], d[f"q{joint}"], d[f"v{joint}"], d[f"tau{joint}"]
    if torque_source == "pd":
        command = read_csv(data_dir / f"j{joint}_kp40_kd2_sine_{tag}_cmd.csv",
                           ("t", f"qd{joint}"))
        target = np.interp(t, command["t"], command[f"qd{joint}"])
        tau = 40.0 * (target - q) - 2.0 * vel
    t0 = t[d["phase"] == "excite"][0]
    bounds = np.r_[0.0, np.cumsum(cycles / np.asarray(freqs))]
    results = []

    for index, freq in enumerate(freqs):
        if freq not in selected:
            continue
        steady = (t >= t0 + bounds[index] + 2 / freq) & (t < t0 + bounds[index + 1])
        rising, falling = steady & (vel > 0.01), steady & (vel < -0.01)
        q_min = max(np.quantile(q[rising], 0.03), np.quantile(q[falling], 0.03))
        q_max = min(np.quantile(q[rising], 0.97), np.quantile(q[falling], 0.97))
        points = []
        for lo, hi in zip(np.linspace(q_min, q_max, 25)[:-1], np.linspace(q_min, q_max, 25)[1:]):
            up = rising & (q >= lo) & (q < hi)
            down = falling & (q >= lo) & (q < hi)
            if up.sum() < 3 or down.sum() < 3:
                continue
            speed = 0.5 * (np.median(vel[up]) - np.median(vel[down]))
            half_separation = 0.5 * (np.median(tau[up]) - np.median(tau[down]))
            points.append((speed, half_separation))
        results.append({"freq": freq, "points": np.asarray(points)})
    return results


def frf(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return (np.array([float(row["freq_hz"]) for row in rows]),
            np.array([float(row["gain"]) for row in rows]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("deploy/sysid/doc/stage3a/data"))
    parser.add_argument("--out-dir", type=Path, default=Path("deploy/sysid/doc/stage3c"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "png").mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = plt.get_cmap("tab10")
    summary = []
    replay_summary = []

    for column, joint in enumerate((4, 10)):
        small_f, small_g = frf(args.data_dir / f"j{joint}_frf_summary.csv")
        large_f, large_g = frf(args.data_dir / f"j{joint}_fixedA005_frf_summary.csv")
        axes[0, column].semilogx(small_f, small_g, "-o", label="small amplitude")
        axes[0, column].semilogx(large_f, large_g, "-o", label="A=0.05 rad")
        command = read_csv(
            args.data_dir / f"j{joint}_kp40_kd2_sine_frf_anklepitch_fixedA005_r1_cmd.csv",
            ("t", f"qd{joint}"),
        )
        replay_path = args.out_dir / "data" / (
            f"SIM_j{joint}_kp40_kd2_sine_frf_anklepitch_fixedA005_r1_fix_d2_fc0.5_state.csv"
        )
        if replay_path.exists():
            replay = read_csv(replay_path, ("t", f"q{joint}"))
            t0 = excite_start(command["t"], command["phase"])
            sim_f, sim_g, sim_lag = frf_stepsine(
                command["t"], command[f"qd{joint}"], replay["t"], replay[f"q{joint}"],
                (0.2, 0.5, 0.8, 1.2, 2.0), 4, 1, t0, t0,
            )
            axes[0, column].semilogx(sim_f, sim_g, "--s", label="nlegs_body, Fc=0.5")
            replay_summary.extend((joint, freq, gain, lag)
                                  for freq, gain, lag in zip(sim_f, sim_g, sim_lag))
        axes[0, column].set(title=f"joint {joint}: measured amplitude dependence",
                            xlabel="frequency [Hz]", ylabel="gain |q/target|")
        axes[0, column].grid(alpha=0.3, which="both")
        axes[0, column].legend()

        groups = []
        for run_index, (label, tag, freqs, cycles, selected) in enumerate(RUNS):
            for result in hysteresis_points(args.data_dir, joint, tag, freqs, cycles, selected):
                groups.append((label, result["freq"], result["points"]))

        direction = np.sign(np.median(np.concatenate([points[:, 1] for _, _, points in groups])))
        groups = [(label, freq, np.c_[points[:, 0], direction * points[:, 1]])
                  for label, freq, points in groups]
        combined = np.concatenate([points for _, _, points in groups])
        fc, viscous = fit_line(combined)
        loo = [fit_line(np.concatenate([g[2] for k, g in enumerate(groups) if k != omitted]))[0]
               for omitted in range(len(groups))]

        for index, (label, freq, points) in enumerate(groups):
            local_fc, local_b = fit_line(points)
            axes[1, column].scatter(points[:, 0], points[:, 1], s=18, alpha=0.7,
                                    color=colors(index), label=f"{label}, {freq:g} Hz")
            summary.append((joint, label, freq, local_fc, local_b,
                            np.median(points[:, 1]), len(points), "condition"))

        speed = np.linspace(0, combined[:, 0].max(), 100)
        axes[1, column].plot(speed, fc + viscous * speed, "k-", lw=2,
                             label=f"combined: Fc={fc:.3f} N m")
        axes[1, column].set(title=f"joint {joint}: direction-paired torque hysteresis",
                            xlabel="paired |velocity| [rad/s]",
                            ylabel="half torque separation [N m]")
        axes[1, column].grid(alpha=0.3)
        axes[1, column].legend(fontsize=8)
        summary.append((joint, "all", "", fc, viscous, np.median(combined[:, 1]),
                        len(combined), f"LOO Fc {min(loo):.3f}..{max(loo):.3f}"))

        pd_groups = []
        for _, tag, freqs, cycles, selected in RUNS:
            pd_groups.extend(result["points"] for result in hysteresis_points(
                args.data_dir, joint, tag, freqs, cycles, selected, torque_source="pd"
            ))
        pd_direction = np.sign(np.median(np.concatenate([points[:, 1] for points in pd_groups])))
        pd_points = np.concatenate([np.c_[points[:, 0], pd_direction * points[:, 1]]
                                    for points in pd_groups])
        pd_fc, pd_slope = fit_line(pd_points)
        summary.append((joint, "all_pd_reconstructed", "", pd_fc, pd_slope,
                        np.median(pd_points[:, 1]), len(pd_points), "cross-check"))

    fig.suptitle("Ankle-pitch friction evidence: amplitude dependence and torque hysteresis")
    fig.tight_layout()
    figure_path = args.out_dir / "png" / "ankle_pitch_friction_evidence.png"
    fig.savefig(figure_path, dpi=140)

    csv_path = args.out_dir / "ankle_pitch_friction_estimate.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("joint", "run", "freq_hz", "fc_intercept_nm", "slope_nms_rad",
                         "median_half_separation_nm", "points", "scope"))
        writer.writerows(summary)
    replay_csv = args.out_dir / "ankle_pitch_fc05_replay_frf.csv"
    with replay_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("joint", "freq_hz", "gain", "lag_deg"))
        writer.writerows(replay_summary)
    print(f"saved {csv_path}")
    print(f"saved {replay_csv}")
    print(f"saved {figure_path}")


if __name__ == "__main__":
    synthetic = np.c_[np.linspace(0.01, 0.3, 20), 0.5 + 0.3 * np.linspace(0.01, 0.3, 20)]
    assert np.allclose(fit_line(synthetic), (0.5, 0.3))
    main()
