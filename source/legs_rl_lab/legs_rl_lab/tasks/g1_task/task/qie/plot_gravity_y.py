"""Run sim2sim for 10s and plot gravity_y over time."""
import time
import numpy as np
import matplotlib.pyplot as plt
from sim2sim import SimToSimCfg, MujocoRunner
import mujoco
import torch

cfg = SimToSimCfg()
cfg.sim.sim_duration = 10.0
runner = MujocoRunner(cfg)

times = []
gravity_ys = []

while runner.data.time < cfg.sim.sim_duration:
    step_start = time.time()
    input_obs = runner.get_obs().flatten()
    gravity_y = runner.quat_rotate_inverse(
        runner.data.xquat[1][[1, 2, 3, 0]].astype(np.float32),
        np.array([0.0, 0.0, -1.0])
    )[1]
    times.append(runner.data.time)
    gravity_ys.append(gravity_y)

    raw_action = runner.policy(torch.tensor(input_obs, dtype=torch.float32)).detach().numpy()
    for _ in range(runner.decimation):
        runner.action[:] = runner.latency_sim.process_action(raw_action)
        runner.dof_pos = runner.data.qpos[7:].astype(np.float32)
        runner.dof_vel = runner.data.qvel[6:].astype(np.float32)
        runner.data.ctrl = runner.compute_torque()
        mujoco.mj_step(runner.model, runner.data)
    elapsed = time.time() - step_start
    time.sleep(max(0.0, runner.dt - elapsed))
    runner.viewer.sync()
    runner.episode_length_buf += 1
    runner.calculate_gait_para()

runner.viewer.close()

plt.figure(figsize=(14, 5))
plt.plot(times, gravity_ys, linewidth=0.8)
plt.xlabel("Time (s)")
plt.ylabel("Projected Gravity Y")
plt.title("Lateral Body Sway (gravity_y)")
plt.axhline(0, color='red', linestyle='--', linewidth=0.8)
plt.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.7)
plt.xticks(np.arange(0, max(times) + 0.5, 0.5))
plt.yticks(np.arange(-0.2, 0.2, 0.02))
plt.xlim(0, max(times))
plt.ylim(-0.2, 0.2)
plt.tight_layout()
plt.savefig("gravity_y_plot_25.png", dpi=150)
plt.show()
print("Saved.")
