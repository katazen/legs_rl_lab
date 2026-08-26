#!/usr/bin/env python3
"""Compare the original 100 Hz Stage 1 delay with the 200 Hz Stage 1B rerun."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DOC = os.path.join(os.path.dirname(__file__), "..", "doc")
OUT = os.path.join(DOC, "stage1b")
JOINTS = [
    ("hip L", 0, "j0_kp200_kd5_step_step_bi"),
    ("hip R", 6, "j6_kp200_kd5_step_step_bi"),
    ("knee L", 3, "j3_kp250_kd5_step_step_bi"),
    ("knee R", 9, "j9_kp250_kd5_step_step_bi"),
]


def read(stage, stem, j):
    root = os.path.join(DOC, stage, "data", stem)
    with open(root + "_cmd.csv") as f:
        cmd = list(csv.DictReader(f))
    with open(root + "_state.csv") as f:
        state = list(csv.DictReader(f))
    return {
        "ct": np.array([float(r["t"]) for r in cmd]),
        "cp": np.array([r["phase"] for r in cmd]),
        "qd": np.array([float(r[f"qd{j}"]) for r in cmd]),
        "st": np.array([float(r["t"]) for r in state]),
        "sp": np.array([r["phase"] for r in state]),
        "q": np.array([float(r[f"q{j}"]) for r in state]),
        "v": np.array([float(r[f"v{j}"]) for r in state]),
        "tau": np.array([float(r[f"tau{j}"]) for r in state]),
    }


def metrics(d):
    excite = d["cp"] == "excite"
    t, qd = d["ct"][excite], d["qd"][excite]
    k = np.where(np.abs(np.diff(qd)) > 0.005)[0]
    edges = [(t[i + 1], round(qd[i], 3), round(qd[i + 1], 3)) for i in k]
    edges = [e for e in edges if abs(e[1]) < 0.02 and 0.02 < abs(e[2]) < 0.15]
    raw, interp, gain = [], [], []
    for te, _, amp in edges:
        w = np.where((d["st"] >= te - 0.05) & (d["st"] <= te + 0.5))[0]
        av = np.abs(d["v"][w])
        threshold = max(0.02, 0.1 * av.max())
        i = np.where(av > threshold)[0][0]
        raw.append(d["st"][w[i]] - te)
        if i and av[i] > av[i - 1]:
            t0, t1 = d["st"][w[i - 1]], d["st"][w[i]]
            tc = t0 + (threshold - av[i - 1]) / (av[i] - av[i - 1]) * (t1 - t0)
        else:
            tc = d["st"][w[i]]
        interp.append(tc - te)
        pre = (d["st"] >= te - 0.15) & (d["st"] <= te - 0.02)
        ss = (d["st"] >= te + 1.0) & (d["st"] <= te + 1.4)
        gain.append((d["q"][ss].mean() - d["q"][pre].mean()) / amp)
    dc, ds = np.diff(d["ct"]), np.diff(d["st"])
    active = d["sp"] == "excite"
    return {
        "raw": np.array(raw), "interp": np.array(interp), "gain": np.array(gain),
        "cmd_hz": 1 / np.median(dc), "state_hz": 1 / np.median(ds),
        "state_p99_ms": np.percentile(ds, 99) * 1e3, "state_max_ms": ds.max() * 1e3,
        "qmin": d["q"][active].min(), "qmax": d["q"][active].max(),
        "vmax": np.abs(d["v"][active]).max(), "taumax": np.abs(d["tau"][active]).max(),
    }


rows = []
for label, j, stem in JOINTS:
    old, new = metrics(read("stage1", stem, j)), metrics(read("stage1b", stem, j))
    rows.append((label, old, new))

summary = os.path.join(OUT, "data", "summary.csv")
with open(summary, "w", newline="") as f:
    fields = ["joint", "old_raw_mean_ms", "old_raw_std_ms", "old_interp_mean_ms",
              "new_raw_mean_ms", "new_raw_std_ms", "new_interp_mean_ms", "new_interp_std_ms",
              "new_n", "cmd_hz", "state_hz", "state_p99_ms", "state_max_ms", "dc_gain",
              "qmin", "qmax", "vmax", "taumax"]
    out = csv.DictWriter(f, fieldnames=fields); out.writeheader()
    for label, old, new in rows:
        out.writerow({
            "joint": label,
            "old_raw_mean_ms": old["raw"].mean() * 1e3,
            "old_raw_std_ms": old["raw"].std() * 1e3,
            "old_interp_mean_ms": old["interp"].mean() * 1e3,
            "new_raw_mean_ms": new["raw"].mean() * 1e3,
            "new_raw_std_ms": new["raw"].std() * 1e3,
            "new_interp_mean_ms": new["interp"].mean() * 1e3,
            "new_interp_std_ms": new["interp"].std() * 1e3,
            "new_n": len(new["raw"]), "cmd_hz": new["cmd_hz"], "state_hz": new["state_hz"],
            "state_p99_ms": new["state_p99_ms"], "state_max_ms": new["state_max_ms"],
            "dc_gain": new["gain"].mean(), "qmin": new["qmin"], "qmax": new["qmax"],
            "vmax": new["vmax"], "taumax": new["taumax"],
        })

labels = [r[0] for r in rows]
x = np.arange(len(labels)); width = 0.34
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for a, key, title in [(ax[0], "raw", "First sample over velocity threshold"),
                      (ax[1], "interp", "Linearly interpolated threshold crossing")]:
    old_mean = np.array([r[1][key].mean() for r in rows]) * 1e3
    old_std = np.array([r[1][key].std() for r in rows]) * 1e3
    new_mean = np.array([r[2][key].mean() for r in rows]) * 1e3
    new_std = np.array([r[2][key].std() for r in rows]) * 1e3
    a.bar(x - width / 2, old_mean, width, yerr=old_std, capsize=3, label="Stage 1: 100 Hz")
    a.bar(x + width / 2, new_mean, width, yerr=new_std, capsize=3, label="Stage 1B: 200 Hz")
    a.set_xticks(x, labels); a.set_title(title); a.grid(axis="y", alpha=0.3); a.legend(fontsize=8)
ax[0].set_ylabel("observed onset delay [ms]")
fig.suptitle("Stage 1 vs Stage 1B: sampling-rate dependence of threshold delay")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "png", "exp1b_delay_100_vs_200.png"), dpi=130)

print("joint      old raw      new raw      new interp   cmd/state Hz   state p99/max ms")
for label, old, new in rows:
    print(f"{label:7s}  {old['raw'].mean()*1e3:6.2f}ms  {new['raw'].mean()*1e3:6.2f}ms  "
          f"{new['interp'].mean()*1e3:6.2f}ms  {new['cmd_hz']:6.2f}/{new['state_hz']:6.2f}  "
          f"{new['state_p99_ms']:5.2f}/{new['state_max_ms']:5.2f}")
print("saved", summary)
