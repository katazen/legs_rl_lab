#!/usr/bin/env python3
"""Compare Isaac walking/suspended ankle response and real-target suspended replay."""
import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np


SIDES = (("Left", "L", 4, "left"), ("Right", "R", 10, "right"))


def read(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    numeric = {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != "phase"}
    return numeric, np.array([r.get("phase", "") for r in rows])


def one(pattern):
    paths = glob.glob(pattern)
    if len(paths) != 1:
        raise ValueError(f"expected one file for {pattern}, got {paths}")
    return paths[0]


def stats(delta):
    return dict(rmse=float(np.sqrt(np.mean(delta**2))), mae=float(np.mean(abs(delta))),
                p95=float(np.percentile(abs(delta), 95)), bias=float(np.mean(delta)))


def write_rows(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="ankle_isaac_compare directory")
    p.add_argument("--hardware", required=True, help="ankle_load_compare/data directory")
    p.add_argument("--out", required=True)
    p.add_argument("--window", default="27.84,30.24", help="real source-time plot window")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    walk, _ = read(os.path.join(args.root, "isaac_walk", "isaac_walk.csv"))
    walk_t = walk["t"] - walk["t"][0]
    self_rows, self_data = [], {}
    for side, short, joint, suffix in SIDES:
        suspended, _ = read(one(os.path.join(
            args.root, "isaac_suspended_self", f"SIM_j{joint}_*isaacwalk*state.csv")))
        # Each walk row is state after a 20 ms policy step; command row i is timestamped i*20 ms.
        sample_t = walk_t + 0.02
        q_suspended = np.interp(sample_t, suspended["t"], suspended[f"q{joint}"])
        tau_suspended = np.interp(sample_t, suspended["t"], suspended[f"tau{joint}"])
        q_walk, tau_walk = walk[f"q_{short}5"], walk[f"tau_{short}5"]
        target, force = walk[f"target_{short}5"], walk[f"contact_{short}_N"]
        valid = (walk_t >= 0.5) & (sample_t <= suspended["t"][-1])
        self_data[side] = (target, q_walk, q_suspended, tau_walk, tau_suspended, force)
        for phase, phase_mask in (("swing", force < 5.0), ("contact", force >= 5.0)):
            mask = valid & phase_mask
            row = dict(side=side.lower(), phase=phase, samples=int(mask.sum()),
                       **{f"response_delta_{k}": v for k, v in stats(q_walk[mask] - q_suspended[mask]).items()})
            row.update(walking_target_rmse=float(np.sqrt(np.mean((target[mask] - q_walk[mask])**2))),
                       suspended_target_rmse=float(np.sqrt(np.mean((target[mask] - q_suspended[mask])**2))),
                       walking_tau_rms=float(np.sqrt(np.mean(tau_walk[mask]**2))),
                       suspended_tau_rms=float(np.sqrt(np.mean(tau_suspended[mask]**2))),
                       response_correlation=float(np.corrcoef(q_walk[mask], q_suspended[mask])[0, 1]))
            self_rows.append(row)
    write_rows(os.path.join(args.out, "isaac_walk_vs_suspended_metrics.csv"), self_rows)

    real_rows, real_data = [], {}
    for side, _short, joint, suffix in SIDES:
        prefix = f"j{joint}_kp40_kd2_replay_walk48_{suffix}"
        command, phase = read(os.path.join(args.hardware, prefix + "_cmd.csv"))
        hardware, _ = read(os.path.join(args.hardware, prefix + "_state.csv"))
        excite = phase == "excite"
        t0, t1 = command["t"][excite][[0, -1]]
        grid = command["t"][excite]
        grid = grid[(grid >= t0 + 0.5) & (grid <= t1 - 0.2)]
        target = np.interp(grid, command["t"], command[f"qd{joint}"])
        q_hardware = np.interp(grid, hardware["t"], hardware[f"q{joint}"])
        tau_hardware = np.interp(grid, hardware["t"], hardware[f"tau{joint}"])
        curves = {}
        for delay in (4, 10, 16):
            isaac, _ = read(one(os.path.join(
                args.root, "isaac_suspended_real", f"SIM_j{joint}_*fix_d{delay}_state.csv")))
            q_isaac = np.interp(grid, isaac["t"], isaac[f"q{joint}"])
            tau_isaac = np.interp(grid, isaac["t"], isaac[f"tau{joint}"])
            shifts = np.arange(-0.05, 0.0501, 0.005)
            shift = min(shifts, key=lambda x: np.mean(
                (np.interp(grid + x, isaac["t"], isaac[f"q{joint}"]) - q_hardware)**2))
            row = dict(side=side.lower(), delay_steps=delay, delay_ms=delay * 5,
                       **{f"isaac_minus_hardware_{k}": v for k, v in stats(q_isaac - q_hardware).items()})
            row.update(hardware_target_rmse=float(np.sqrt(np.mean((q_hardware - target)**2))),
                       isaac_target_rmse=float(np.sqrt(np.mean((q_isaac - target)**2))),
                       hardware_tau_rms=float(np.sqrt(np.mean(tau_hardware**2))),
                       isaac_tau_rms=float(np.sqrt(np.mean(tau_isaac**2))),
                       response_correlation=float(np.corrcoef(q_isaac, q_hardware)[0, 1]),
                       alignment_shift_ms=float(shift * 1000),
                       estimated_matching_delay_steps=float(delay - shift / 0.005))
            real_rows.append(row); curves[delay] = q_isaac
        real_data[side] = (grid - t0, target, q_hardware, curves)
    write_rows(os.path.join(args.out, "real_target_hardware_vs_isaac_metrics.csv"), real_rows)

    win = (walk_t >= 1.0) & (walk_t <= 4.0)
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    for row, (side, *_rest) in enumerate(SIDES):
        target, qw, qs, tw, ts, force = self_data[side]
        for ax in axes[row]:
            ax.fill_between(walk_t[win], 0, 1, where=force[win] >= 5.0,
                            transform=ax.get_xaxis_transform(), color="tab:orange", alpha=.12,
                            label="true foot contact" if row == 0 else None)
            ax.grid(alpha=.25)
        axes[row, 0].plot(walk_t[win], target[win], "k--", lw=1.1, label="target")
        axes[row, 0].plot(walk_t[win], qw[win], lw=1.4, label="walking")
        axes[row, 0].plot(walk_t[win], qs[win], lw=1.2, label="suspended")
        axes[row, 0].set_ylabel(f"{side} q (rad)")
        axes[row, 1].plot(walk_t[win], tw[win], lw=1.3, label="walking")
        axes[row, 1].plot(walk_t[win], ts[win], lw=1.1, label="suspended")
        axes[row, 1].set_ylabel(f"{side} applied tau (Nm)")
    axes[0, 0].legend(ncol=2, fontsize=8); axes[0, 1].legend(fontsize=8)
    axes[1, 0].set_xlabel("Isaac walk time (s)"); axes[1, 1].set_xlabel("Isaac walk time (s)")
    fig.suptitle("Isaac ankle pitch: closed-loop walking vs suspended replay of its target")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "isaac_walk_vs_suspended.png"), dpi=180); plt.close(fig)

    lo, hi = map(float, args.window.split(","))
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, (side, *_rest) in zip(axes, SIDES):
        t, target, hardware, curves = real_data[side]; mask = (t >= lo) & (t <= hi)
        ax.plot(t[mask], target[mask], "k--", lw=1.1, label="identical 200 Hz target")
        ax.plot(t[mask], hardware[mask], lw=1.7, label="hardware suspended")
        for delay in (4, 10, 16):
            ax.plot(t[mask], curves[delay][mask], lw=1.0, label=f"Isaac suspended d={delay}")
        ax.set_ylabel(f"{side} q (rad)"); ax.grid(alpha=.25)
    axes[0].legend(ncol=3, fontsize=8); axes[1].set_xlabel("source-walk target time (s)")
    fig.suptitle("Suspended response to identical real-walk ankle target")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "real_target_hardware_vs_isaac.png"), dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    width = .34; x = np.arange(2)
    for i, phase_name in enumerate(("swing", "contact")):
        vals = [next(r["response_delta_rmse"] for r in self_rows
                     if r["side"] == side.lower() and r["phase"] == phase_name) for side, *_ in SIDES]
        axes[0].bar(x + (i - .5) * width, vals, width, label=phase_name)
    axes[0].set_xticks(x, ["Left", "Right"]); axes[0].set_ylabel("walking-suspended RMSE (rad)")
    axes[0].legend(); axes[0].grid(axis="y", alpha=.25)
    for side, *_ in SIDES:
        rows = [r for r in real_rows if r["side"] == side.lower()]
        axes[1].plot([r["delay_ms"] for r in rows], [r["isaac_minus_hardware_rmse"] for r in rows],
                     "o-", label=side)
    axes[1].set_xlabel("Isaac explicit delay (ms)"); axes[1].set_ylabel("Isaac-hardware RMSE (rad)")
    axes[1].legend(); axes[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "summary_metrics.png"), dpi=180); plt.close(fig)

    for r in self_rows:
        print("self", r["side"], r["phase"], f"rmse={r['response_delta_rmse']:.4f}",
              f"bias={r['response_delta_bias']:+.4f}")
    for r in real_rows:
        print("real-target", r["side"], f"d={r['delay_steps']}",
              f"rmse={r['isaac_minus_hardware_rmse']:.4f}",
              f"tau_hw/sim={r['hardware_tau_rms']:.2f}/{r['isaac_tau_rms']:.2f}",
              f"match_delay={r['estimated_matching_delay_steps']:.1f}")


if __name__ == "__main__":
    main()
