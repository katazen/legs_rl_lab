import time
from collections import deque
import mujoco
import mujoco.viewer
import numpy as np
import torch
from pynput import keyboard



class SimToSimCfg:
    class path:
        # TODO: 窄版 scene xml 还没生成（需由 A1_legs_V2_narrow_mjcf.xml 包一层 scene）。
        pos_xml_path = '/home/woan/workspace/legs_rl_lab/source/legs_rl_lab/legs_rl_lab/assets/legs_URDF/mjcf/A1_legs_V2_narrow_mjcf_scene.xml'
        tau_xml_path = '/home/woan/workspace/legs_rl_lab/source/legs_rl_lab/legs_rl_lab/assets/legs_URDF/mjcf/A1_legs_V2_narrow_mjcf_scene.xml'
        # TODO: 换成 nlegs 训练导出的 policy.pt（logs/rsl_rl/nlegs/<run>/exported/policy.pt）。
        model_path = '/home/woan/workspace/legs_rl_lab/logs/rsl_rl/legs/2026-06-29_20-36-21/exported/policy.pt'

    class sim:
        sim_duration = 10000.0
        control_mode = "motor"  # "position" or "motor"
        action_dim = 12
        state_dim = 11 + 3 * action_dim
        dt = 0.005
        decimation = 4
        gait_cycle = 0.85  # 与训练 GaitCfg.period 一致
        action_delay_range = (3, 4)
        his_lens = 10
        collect_data = False       # 是否采集关节跟踪数据并出图 (不需要采时设 False)
        collect_duration = 10.0   # 采集时长 (秒)
        obs_slices = {
            "ang_vel": (0, 3),
            "gravity": (3, 6),
            "command": (6, 9),
            "dof_pos": (9, 9+action_dim),
            "dof_vel": (9+action_dim, 9+action_dim*2),
            "actions": (9+action_dim*2, 9+action_dim*3),
            "gait": (9+action_dim*3, 11+action_dim*3)
        }

    class robot:
        NATURAL_FREQ = 10 * 2.0 * np.pi  # 10Hz
        DAMPING_RATIO = 2.0
        # arrays below are in MuJoCo joint order: R1..R6, L1..L6
        default_dof_pos = np.array([-0.1, 0.0, 0.0, 0.2, -0.1, 0.0,
                                    -0.1, 0.0, 0.0, 0.2, -0.1, 0.0])
        reset_dof_pos = np.array([-0.1, 0.0, 0.0, 0.2, -0.1, 0.0,
                                  -0.1, 0.0, 0.0, 0.2, -0.1, 0.0])
        armature = np.array([0.032, 0.032, 0.032, 0.032, 0.0018, 0.0018,
                             0.032, 0.032, 0.032, 0.032, 0.0018, 0.0018])
        effort = np.array([27, 27, 27, 27, 7, 7,
                           27, 27, 27, 27, 7, 7])
        stiffness = np.array([200., 200, 200, 200, 40, 40,
                              200., 200, 200, 200, 40, 40])
        damping = np.array([5., 5, 5, 5, 0.5, 0.5,
                            5., 5, 5, 5, 0.5, 0.5])
        # stiffness = np.array([45., 45, 45, 45, 45, 45,
        #                       45., 45, 45, 45, 45, 45])
        # damping = np.array([2.5] * 12)

        # damping = np.array([1.5, 1.5, 1.5, 1.5, 0.8, 0.8,
        #                     1.5, 1.5, 1.5, 1.5, 0.8, 0.8])
        action_scale = 0.25
        cmd_range = [[-0.5, 1.0], [-0.5, 0.5], [-0.3, 0.3]]


class LatencySimulator:
    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        self.action_buffer = None

    def reset(self, action_delay_range: tuple):
        """
        在 Episode 重置时调用。
        :param action_delay_range: (min, max) 动作延迟步数范围
        :param obs_delay_ranges: dict { 'dof_pos': (min, max), ... } 观测延迟范围
        :param initial_obs: dict { 'dof_pos': initial_value, ... } 初始观测值，用于填充 Buffer
        """
        act_delay = np.random.randint(*action_delay_range)
        self.action_buffer = deque(
            [np.zeros(self.action_dim, dtype=np.float32)] * (act_delay + 1),
            maxlen=act_delay + 1
        )
        print(f"[Latency] Action Delay: {act_delay} steps")

    def process_action(self, new_action: np.ndarray) -> np.ndarray:
        """
        [在物理步调用]
        输入策略产生的最新动作，推入队列，并弹出当前应该执行的旧动作。
        """
        self.action_buffer.append(new_action.copy())
        return self.action_buffer.popleft()


class MujocoRunner:
    def __init__(self, cfg: SimToSimCfg):
        self.cfg = cfg
        used_xml = self.cfg.path.pos_xml_path if self.cfg.sim.control_mode=='position' else self.cfg.path.tau_xml_path
        self.model = mujoco.MjModel.from_xml_path(used_xml)
        self.model.opt.timestep = self.cfg.sim.dt
        self.policy = torch.jit.load(self.cfg.path.model_path, map_location="cpu")
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.distance = 4.0
        self.viewer.cam.elevation = -20
        self.viewer.cam.azimuth = 80
        self.latency_sim = LatencySimulator(action_dim=self.cfg.sim.action_dim)
        self.init_variables()

    def init_variables(self):
        self.action_dim = self.cfg.sim.action_dim
        self.action_scale = self.cfg.robot.action_scale
        # self.action_rate_limit = self.cfg.robot.action_rate_limit
        self.control_mode = self.cfg.sim.control_mode
        print(f"action_scale: {self.action_scale}, control_mode: {self.control_mode}")
        self.state_dim = self.cfg.sim.state_dim
        self.obs_his = np.zeros((self.cfg.sim.his_lens, self.state_dim))
        self.sim_duration = self.cfg.sim.sim_duration
        self.decimation = self.cfg.sim.decimation
        self.dt = self.decimation * self.cfg.sim.dt
        self.dof_pos = np.zeros(self.action_dim)
        self.dof_vel = np.zeros(self.action_dim)
        self.action = np.zeros(self.action_dim)
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.default_dof_pos = self.cfg.robot.default_dof_pos
        self.reset_dof_pos = self.cfg.robot.reset_dof_pos
        self.episode_length_buf = 0
        self.gait_phase = np.zeros(1)
        self.gait_cycle = self.cfg.sim.gait_cycle
        # Isaac/policy joint order for the A1_legs_V2 (MJCF) model: R before L per level,
        # from this run's deploy.yaml joint_ids_map [0,6,1,7,2,8,3,9,4,10,5,11]
        self.isaac_joint = ['R1', 'L1', 'R2', 'L2', 'R3', 'L3', 'R4', 'L4', 'R5', 'L5', 'R6', 'L6']
        self.mjc_joint = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'L1','L2', 'L3', 'L4', 'L5','L6']
        self.mjc_ctrl = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'L1','L2', 'L3', 'L4', 'L5','L6']

        self.isaac2mjc = np.array([self.isaac_joint.index(i) for i in self.mjc_joint])
        self.mjc2isaac = np.array([self.mjc_joint.index(i) for i in self.isaac_joint])
        self.command_vel = np.array([0.0, 0.0, 0.0])
        self.cmd_range = self.cfg.robot.cmd_range

        # ---- actuator config: position vs motor ----
        if self.control_mode == "position":
            # MuJoCo内部计算PD: force = kp*(ctrl - qpos), damping由dof_damping提供
            self.model.actuator_gainprm[:, 0] = self.cfg.robot.stiffness
            self.model.actuator_biasprm[:, 1] = -self.cfg.robot.stiffness
            self.model.dof_damping[-self.action_dim:] = self.cfg.robot.damping
        elif self.control_mode == "motor":
            # ctrl直接就是力矩，不经过任何增益
            self.model.actuator_gainprm[:, 0] = 1.0
            self.model.actuator_biasprm[:, 1] = 0.0
            self.model.dof_damping[-self.action_dim:] = 0.0
        self.model.dof_armature[-self.action_dim:] = self.cfg.robot.armature

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos = np.concatenate([np.array([0, 0, 0.55], dtype=np.float32),
                                         np.array([1, 0, 0, 0], dtype=np.float32), self.reset_dof_pos])
        mujoco.mj_forward(self.model, self.data)
        self.latency_sim.reset(self.cfg.sim.action_delay_range)
        self.start_time = time.time()
        self.his_data = []
        self.collect_data = self.cfg.sim.collect_data    # 是否采集
        self.collect_T = self.cfg.sim.collect_duration   # 采集时长 (秒)
        self.track_log = []          # 关节跟踪采集: (t, target_pos_mjc, qpos_mjc, cmd_vel)

    def update_obs_his(self, new_obs):
        self.obs_his[:-1] = self.obs_his[1:]
        self.obs_his[-1] = new_obs

    def get_inference_input(self):
        input_parts = []
        feature_names = ["ang_vel", "gravity", "command", "dof_pos", "dof_vel", "actions", "gait"]
        for name in feature_names:
            start, end = self.cfg.sim.obs_slices[name]
            feature_block = self.obs_his[:, start:end]
            flat_feature = feature_block.flatten()
            input_parts.append(flat_feature)
        return np.concatenate(input_parts)

    def get_obs(self):
        self.dof_pos = self.data.qpos[7:].astype(np.float32)
        self.dof_vel = self.data.qvel[6:].astype(np.float32)
        obs = np.zeros((self.state_dim,), dtype=np.float32)
        # Angular vel
        obs[0:3] = self.data.sensor("imu_gyro").data.astype(np.double) * 0.2
        # Projected gravity
        obs[3:6] = self.quat_rotate_inverse(self.data.xquat[1][[1, 2, 3, 0]].astype(np.float32),
                                            np.array([0.0, 0.0, -1.0]))
        # print(obs[3])
        # Command velocity
        obs[6:9] = self.command_vel
        # Dof pos
        obs[9:9 + self.action_dim] = (self.dof_pos - self.default_dof_pos)[self.mjc2isaac]
        # Dof vel
        obs[9 + self.action_dim:9 + 2 * self.action_dim] = self.dof_vel[self.mjc2isaac] * 0.05
        # Action
        obs[9 + 2 * self.action_dim:9 + 3 * self.action_dim] = self.action
        obs[9 + 3 * self.action_dim:10 + 3 * self.action_dim] = np.sin(2 * torch.pi * self.gait_phase)
        obs[10 + 3 * self.action_dim:11 + 3 * self.action_dim] = np.cos(2 * torch.pi * self.gait_phase)
        self.update_obs_his(obs)
        return self.get_inference_input()


    def compute_target_pos(self):
        actions_scaled = self.action * self.action_scale
        return actions_scaled[self.isaac2mjc] + self.default_dof_pos

    def compute_torque(self):
        target_pos = self.compute_target_pos()
        tau = self.cfg.robot.stiffness * (target_pos - self.dof_pos) - self.cfg.robot.damping * self.dof_vel
        return np.clip(tau, -self.cfg.robot.effort, self.cfg.robot.effort)

    def _draw_velocity_arrows(self):
        """在 viewer 里画两个实时箭头观察速度跟踪: 绿=指令速度, 蓝=实际速度 (世界系水平 vx,vy)。
        跟踪得好时两箭头重合。"""
        ARROW_SCALE = 1.0   # 1 m/s -> 1 m 长; 指令较小时可调大
        WIDTH = 0.02
        scn = self.viewer.user_scn
        base = self.data.xpos[1].copy()                 # base body (index 1)
        anchor = base + np.array([0.0, 0.0, 0.35])      # 抬到机器人上方, 便于观察
        # base 偏航 (从 wxyz 四元数取 yaw), 指令是 heading 系的 vx,vy -> 旋到世界系
        w, x, y, z = self.data.xquat[1]
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        c, s = np.cos(yaw), np.sin(yaw)
        cmd = self.command_vel
        cmd_world = np.array([c * cmd[0] - s * cmd[1], s * cmd[0] + c * cmd[1], 0.0])
        act_world = np.array([self.data.qvel[0], self.data.qvel[1], 0.0])  # 自由关节线速度=世界系
        n = 0
        for vec, rgba in [(cmd_world, np.array([0.1, 0.9, 0.1, 1.0], dtype=np.float32)),   # 指令=绿
                          (act_world, np.array([0.1, 0.4, 1.0, 1.0], dtype=np.float32))]:  # 实际=蓝
            if np.linalg.norm(vec) < 1e-3:   # 速度~0 不画 (避免零长箭头)
                continue
            g = scn.geoms[n]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW,
                                np.zeros(3), np.zeros(3), np.zeros(9), rgba)
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, WIDTH,
                                 anchor, anchor + vec * ARROW_SCALE)
            n += 1
        scn.ngeom = n

    def run(self):
        self.setup_keyboard_listener()
        self.listener.start()
        while self.data.time < self.sim_duration:
            input_obs = self.get_obs().flatten()
            raw_policy_action = self.policy(torch.tensor(input_obs, dtype=torch.float32)).detach().numpy()
            # 与训练 A1Env.step 对齐：对 policy 输出做每步 ±action_rate_limit 增量裁剪
            # delta = np.clip(raw_policy_action - self.prev_action, -self.action_rate_limit, self.action_rate_limit)
            # clamped_action = (self.prev_action + delta).astype(np.float32)
            # self.prev_action = clamped_action
            for _ in range(self.decimation):
                self.action[:] = self.latency_sim.process_action(raw_policy_action)
                if self.control_mode == "position":
                    self.data.ctrl = self.compute_target_pos()
                else:
                    self.dof_pos = self.data.qpos[7:].astype(np.float32)
                    self.dof_vel = self.data.qvel[6:].astype(np.float32)
                    self.data.ctrl = self.compute_torque()
                mujoco.mj_step(self.model, self.data)
            # --- 采集关节跟踪 (mjc 序 R1..R6,L1..L6) ---
            # target 取「策略决策时刻、延迟前」的目标(与实机发布 /dog_joint_pos 口径一致),
            # qpos 是经 action_delay 后的实际响应 -> 这样链路死区(action_delay)在图上才显现。
            if self.collect_data and self.data.time <= self.collect_T:
                raw_target = (raw_policy_action * self.action_scale)[self.isaac2mjc] + self.default_dof_pos
                self.track_log.append((float(self.data.time),
                                       raw_target.astype(np.float32).copy(),
                                       self.data.qpos[7:].astype(np.float32).copy(),
                                       self.command_vel.copy()))
            cost_time = time.time() - self.start_time
            time.sleep(max(0.0, 0.02 - cost_time))
            self.start_time = time.time()
            # print(self.data.qpos[2])
            # print(max(raw_policy_action))
            self.viewer.cam.lookat[:] = self.data.qpos.astype(np.float32)[0:3]
            self._draw_velocity_arrows()
            self.viewer.sync()
            self.episode_length_buf += 1
            self.calculate_gait_para()
            if self.collect_data and self.data.time >= self.collect_T:   # 采够就停下来出图
                break

        self.listener.stop()
        if self.collect_data:
            self._plot_tracking()
        self.viewer.close()

    def _plot_tracking(self):
        """画 12 关节位置跟踪 (target 黑虚 vs sim 实测), 并存 csv, 便于与实机对比。"""
        import os
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not self.track_log:
            print("[采集] 无数据"); return
        t = np.array([r[0] for r in self.track_log])
        tgt = np.array([r[1] for r in self.track_log])   # (N,12) mjc 序
        act = np.array([r[2] for r in self.track_log])
        cmd = np.array([r[3] for r in self.track_log])
        # 重排成 [L1..L6,R1..R6], 与实机图一致
        names = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6']
        order = [self.mjc_ctrl.index(n) for n in names]
        outdir = os.path.expanduser("~/rl_real_logs"); os.makedirs(outdir, exist_ok=True)
        # csv
        csv_path = os.path.join(outdir, "sim2sim_track.csv")
        with open(csv_path, "w") as f:
            f.write("t," + ",".join(f"target_{n}" for n in names) + "," +
                    ",".join(f"sim_{n}" for n in names) + ",cmd_vx,cmd_vy,cmd_yaw\n")
            for k in range(len(t)):
                row = [t[k]] + [tgt[k, j] for j in order] + [act[k, j] for j in order] + list(cmd[k])
                f.write(",".join(f"{v:.6f}" for v in row) + "\n")
        # 图
        fig, ax = plt.subplots(4, 3, figsize=(20, 12), sharex=True)
        for i, n in enumerate(names):
            j = order[i]; a = ax[i // 3, i % 3]
            a.plot(t, tgt[:, j], "k--", lw=1.0, label="target")
            a.plot(t, act[:, j], "-", color="tab:orange", lw=1.0, label="sim")
            rmse = np.sqrt(np.mean((act[:, j] - tgt[:, j]) ** 2)) * 1000
            a.set_title(f"{n}  RMSE={rmse:.0f}mrad", fontsize=11); a.grid(alpha=.3)
            if i == 0: a.legend(loc="upper right", fontsize=9)
        for c in range(3): ax[3, c].set_xlabel("t [s]")
        fig.suptitle(f"sim2sim dof following (0-{t[-1]:.0f}s, target=balck, sim=orange; obtain action_delay death range)", fontsize=12)
        plt.tight_layout()
        png = os.path.join(outdir, "sim2sim_track_0-10s.png")
        plt.savefig(png, dpi=100); plt.close()
        print(f"[采集] {len(t)} 帧 -> {csv_path}\n[采集] 图 -> {png}")

    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray):
        q_w = q[-1]
        q_vec = q[:3]
        a = v * (2.0 * q_w ** 2 - 1.0)
        b = np.cross(q_vec, v) * q_w * 2.0
        c = q_vec * np.dot(q_vec, v) * 2.0
        return a - b + c

    def calculate_gait_para(self):
        self.gait_phase = self.episode_length_buf * self.dt / self.gait_cycle % 1.0

    def adjust_command_vel(self, idx: int, increment: float):
        self.command_vel[idx] += increment
        self.command_vel[idx] = np.clip(self.command_vel[idx], self.cmd_range[idx][0], self.cmd_range[idx][1])
        print([round(float(i), 2) for i in self.command_vel])

    def setup_keyboard_listener(self):
        def on_press(key):
            try:
                if key.char == "8":
                    self.adjust_command_vel(0, 0.05)
                elif key.char == "2":
                    self.adjust_command_vel(0, -0.05)
                elif key.char == "4":
                    self.adjust_command_vel(1, -0.05)
                elif key.char == "6":
                    self.adjust_command_vel(1, 0.05)
                elif key.char == "7":
                    self.adjust_command_vel(2, 0.05)
                elif key.char == "9":
                    self.adjust_command_vel(2, -0.05)
            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press)


if __name__ == "__main__":
    sim_cfg = SimToSimCfg()
    runner = MujocoRunner(cfg=sim_cfg)
    runner.run()
