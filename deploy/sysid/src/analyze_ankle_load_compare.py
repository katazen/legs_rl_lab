#!/usr/bin/env python3
"""Compare walking ankle-pitch response with suspended hardware replay."""
import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


SIDES = (("Left", 4, 0.5, "left"), ("Right", 10, 0.0, "right"))


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="loaded sim2real CSV")
    p.add_argument("--data", required=True, help="directory containing replay cmd/state CSVs")
    p.add_argument("--out", required=True)
    p.add_argument("--period", type=float, default=0.6)
    p.add_argument("--stance-ratio", type=float, default=0.55)
    p.add_argument("--moving-threshold", type=float, default=0.1)
    p.add_argument("--window", default="27.84,30.24", help="representative source-time window")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    loaded = read(args.source)
    t = np.array([float(r["t"]) for r in loaded]); t -= t[0]
    speed = np.sqrt(sum(np.array([float(r[k]) for r in loaded]) ** 2
                        for k in ("cmd_vx", "cmd_vy", "cmd_yaw")))
    moving = speed >= args.moving_threshold
    phase = (t / args.period) % 1.0
    results, paired = [], {}

    for side, joint, offset, suffix in SIDES:
        prefix = f"j{joint}_kp40_kd2_replay_walk48_{suffix}"
        cmd = [r for r in read(os.path.join(args.data, prefix + "_cmd.csv")) if r["phase"] == "excite"]
        state = [r for r in read(os.path.join(args.data, prefix + "_state.csv")) if r["phase"] == "excite"]
        tc = np.array([float(r["t"]) for r in cmd]); tc -= tc[0]
        ts = np.array([float(r["t"]) for r in state]) - float(cmd[0]["t"])
        target = np.interp(t, tc, np.array([float(r[f"qd{joint}"]) for r in cmd]))
        q_free = np.interp(t, ts, np.array([float(r[f"q{joint}"]) for r in state]))
        tau_free = np.interp(t, ts, np.array([float(r[f"tau{joint}"]) for r in state]))
        q_load = np.array([float(r[f"q{joint}"]) for r in loaded])
        tau_load = np.array([float(r[f"tau{joint}"]) for r in loaded])
        stance = ((phase + offset) % 1.0) < args.stance_ratio
        paired[side] = dict(target=target, q_load=q_load, q_free=q_free,
                            tau_load=tau_load, tau_free=tau_free, stance=stance)

        for label, mask in (("stance", moving & stance), ("swing", moving & ~stance)):
            delta = q_load[mask] - q_free[mask]
            results.append(dict(
                side=side.lower(), phase=label, samples=int(mask.sum()),
                response_delta_rmse=rms(delta), response_delta_mae=float(np.mean(abs(delta))),
                response_delta_p95=float(np.percentile(abs(delta), 95)),
                response_delta_bias=float(np.mean(delta)),
                walking_target_rmse=rms(target[mask] - q_load[mask]),
                suspended_target_rmse=rms(target[mask] - q_free[mask]),
                walking_tau_rms=rms(tau_load[mask]), suspended_tau_rms=rms(tau_free[mask]),
                walking_tau_p95=float(np.percentile(abs(tau_load[mask]), 95)),
                suspended_tau_p95=float(np.percentile(abs(tau_free[mask]), 95)),
                response_correlation=float(np.corrcoef(q_load[mask], q_free[mask])[0, 1]),
            ))

    cols = list(results[0])
    with open(os.path.join(args.out, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(results)

    # Representative cycles: same target, loaded and unloaded responses.
    lo, hi = map(float, args.window.split(",")); win = (t >= lo) & (t <= hi)
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    for row, (side, _, _, _) in enumerate(SIDES):
        d = paired[side]
        ax = axes[row, 0]
        ax.plot(t[win], d["target"][win], "k--", lw=1.2, label="200 Hz target")
        ax.plot(t[win], d["q_load"][win], lw=1.5, label="walking response")
        ax.plot(t[win], d["q_free"][win], lw=1.3, label="suspended replay")
        ax.fill_between(t[win], 0, 1, where=d["stance"][win], transform=ax.get_xaxis_transform(),
                        color="tab:orange", alpha=0.10, label="planned stance")
        ax.set_ylabel(f"{side} q (rad)"); ax.grid(alpha=.25)
        ax = axes[row, 1]
        ax.plot(t[win], d["tau_load"][win], lw=1.2, label="walking response")
        ax.plot(t[win], d["tau_free"][win], lw=1.1, label="suspended replay")
        ax.fill_between(t[win], 0, 1, where=d["stance"][win], transform=ax.get_xaxis_transform(),
                        color="tab:orange", alpha=0.10)
        ax.set_ylabel(f"{side} feedback tau (Nm)"); ax.grid(alpha=.25)
    axes[0, 0].legend(ncol=2, fontsize=8); axes[0, 1].legend(fontsize=8)
    axes[1, 0].set_xlabel("source time (s)"); axes[1, 1].set_xlabel("source time (s)")
    fig.suptitle("Ankle-pitch: identical target, walking vs suspended response")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "tracking_window.png"), dpi=180); plt.close(fig)

    # Phase summary: paired response mismatch and feedback-torque RMS.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(2); width = .34
    for i, phase_name in enumerate(("swing", "stance")):
        values = [next(r["response_delta_rmse"] for r in results
                       if r["side"] == side.lower() and r["phase"] == phase_name)
                  for side, *_ in SIDES]
        axes[0].bar(x + (i - .5) * width, values, width, label=phase_name)
    axes[0].set_xticks(x, ["Left", "Right"]); axes[0].set_ylabel("loaded-free response RMSE (rad)")
    axes[0].legend(); axes[0].grid(axis="y", alpha=.25)
    labels, values = [], []
    for side, *_ in SIDES:
        for phase_name in ("swing", "stance"):
            r = next(r for r in results if r["side"] == side.lower() and r["phase"] == phase_name)
            labels.append(f"{side[0]}-{'SW' if phase_name == 'swing' else 'ST'}")
            values.append((r["walking_tau_rms"], r["suspended_tau_rms"]))
    values = np.array(values); x = np.arange(len(labels))
    axes[1].bar(x - width / 2, values[:, 0], width, label="walking")
    axes[1].bar(x + width / 2, values[:, 1], width, label="suspended")
    axes[1].set_xticks(x, labels); axes[1].set_ylabel("feedback torque RMS (Nm)")
    axes[1].legend(); axes[1].grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "phase_metrics.png"), dpi=180); plt.close(fig)

    for r in results:
        print(r["side"], r["phase"],
              f"response_rmse={r['response_delta_rmse']:.4f}rad bias={r['response_delta_bias']:+.4f}rad",
              f"tau_rms walking/suspended={r['walking_tau_rms']:.2f}/{r['suspended_tau_rms']:.2f}Nm")


if __name__ == "__main__":
    main()
