import time
from collections import deque
import mujoco
import mujoco.viewer
import numpy as np
import torch
from pynput import keyboard


class N7520_14:
    armature = 0.01
    effort = 88.


class N7520_22:
    armature = 0.01
    effort = 139.


class N5020_16:
    armature = 0.01
    effort = 25.


class N5020_16P:
    armature = 0.01
    effort = 35.


class SimToSimCfg:
    class path:
        pos_xml_path = '/home/woan/workspace/TienKung-Lab/legged_lab/assets/g1/g1_23dof_sim2sim.xml'
        tau_xml_path = '/home/woan/workspace/TienKung-Lab/legged_lab/assets/g1/g1_23dof_rev_1_0.xml'
        model_path = '/home/woan/workspace/legs_rl_lab/logs/rsl_rl/g1qie/2026-05-25_10-12-39/exported/policy.pt'

    class sim:
        sim_duration = 10000.0
        control_mode = "motor"  # "position" or "motor"
        action_dim = 23
        state_dim = 11 + 3 * action_dim
        dt = 0.005
        decimation = 4
        gait_cycle = 0.65
        his_lens = 10
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
        default_dof_pos = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0,
                                    -0.1, 0.0, 0.0, 0.3, -0.2, 0,
                                    0,
                                    0.0, 0.6, 0.0, 1.6, 0.0,
                                    0.0, -0.6, 0.0, 1.6, 0.0])
        reset_dof_pos = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0,
                                    -0.1, 0.0, 0.0, 0.3, -0.2, 0,
                                    0,
                                    0.0, 0.6, 0.0, 1.6, 0.0,
                                    0.0, -0.6, 0.0, 1.6, 0.0])
        articulation = [N7520_14, N7520_22, N7520_14, N7520_22, N5020_16P, N5020_16P,
                        N7520_14, N7520_22, N7520_14, N7520_22, N5020_16P, N5020_16P,
                        N5020_16,
                        N5020_16,N5020_16,N5020_16,N5020_16,N5020_16,
                        N5020_16,N5020_16,N5020_16,N5020_16,N5020_16,]
        armature = np.array([i.armature for i in articulation])
        effort = np.array([i.effort for i in articulation])
        stiffness = np.array([100., 100, 100, 150, 40,40,
                              100., 100, 100, 150, 40,40,
                              200,
                              40,40,40,40,40,
                              40,40,40,40,40,])
        damping = np.array([2., 2, 2, 4, 2, 2,
                            2, 2, 2, 4, 2, 2,
                            5,
                            1,1,1, 1, 1,
                            1,1,1, 1, 1,])
        action_scale = 0.25
        cmd_range = [[-0.3, 0.5], [-0.3, 0.3], [-0.3, 0.3]]


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
        self.viewer.cam.azimuth = 135
        self.latency_sim = LatencySimulator(action_dim=self.cfg.sim.action_dim)
        self.init_variables()

    def init_variables(self):
        self.action_dim = self.cfg.sim.action_dim
        self.action_scale = self.cfg.robot.action_scale
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
        self.default_dof_pos = self.cfg.robot.default_dof_pos
        self.reset_dof_pos = self.cfg.robot.reset_dof_pos
        self.episode_length_buf = 0
        self.gait_phase = np.zeros(1)
        self.gait_cycle = self.cfg.sim.gait_cycle
        self.isaac_joint = ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 'left_hip_roll_joint',
                            'right_hip_roll_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint',
                            'left_hip_yaw_joint', 'right_hip_yaw_joint', 'left_shoulder_roll_joint',
                            'right_shoulder_roll_joint', 'left_knee_joint', 'right_knee_joint',
                            'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_ankle_pitch_joint',
                            'right_ankle_pitch_joint', 'left_elbow_joint', 'right_elbow_joint', 'left_ankle_roll_joint',
                            'right_ankle_roll_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint']
        self.mjc_joint = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint',
                          'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint',
                          'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint',
                          'right_ankle_roll_joint', 'waist_yaw_joint', 'left_shoulder_pitch_joint',
                          'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint',
                          'left_wrist_roll_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
                          'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint']
        self.mjc_ctrl = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint',
                         'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint',
                         'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint',
                         'right_ankle_roll_joint', 'waist_yaw_joint', 'left_shoulder_pitch_joint',
                         'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint',
                         'left_wrist_roll_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
                         'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint']

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
        self.data.qpos = np.concatenate([np.array([0, 0, 0.75], dtype=np.float32),
                                         np.array([1, 0, 0, 0], dtype=np.float32), self.reset_dof_pos])
        mujoco.mj_forward(self.model, self.data)
        self.latency_sim.reset(action_delay_range=(3, 4))
        self.start_time = time.time()
        self.his_data = []

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
        obs[0:3] = self.data.sensor("imu-pelvis-angular-velocity").data.astype(np.double) * 0.2
        # Projected gravity
        obs[3:6] = self.quat_rotate_inverse(self.data.xquat[1][[1, 2, 3, 0]].astype(np.float32),
                                            np.array([0.0, 0.0, -1.0]))
        # Command velocity
        obs[6:9] = self.command_vel
        # Dof pos
        obs[9:9 + self.action_dim] = (self.dof_pos - self.default_dof_pos)[self.mjc2isaac]
        # Dof vel
        obs[9 + self.action_dim:9 + 2 * self.action_dim] = self.dof_vel[self.mjc2isaac] * 0.05
        # Action
        obs[9 + 2 * self.action_dim:9 + 3 * self.action_dim] = self.action
        # Gait parameters
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

    def run(self):
        self.setup_keyboard_listener()
        self.listener.start()
        while self.data.time < self.sim_duration:
            input_obs = self.get_obs().flatten()
            raw_policy_action = self.policy(torch.tensor(input_obs, dtype=torch.float32)).detach().numpy()
            for _ in range(self.decimation):
                self.action[:] = self.latency_sim.process_action(raw_policy_action)
                if self.control_mode == "position":
                    self.data.ctrl = self.compute_target_pos()
                else:
                    self.dof_pos = self.data.qpos[7:].astype(np.float32)
                    self.dof_vel = self.data.qvel[6:].astype(np.float32)
                    self.data.ctrl = self.compute_torque()
                mujoco.mj_step(self.model, self.data)
            cost_time = time.time() - self.start_time
            time.sleep(max(0.0, 0.02 - cost_time))
            self.start_time = time.time()
            self.viewer.cam.lookat[:] = self.data.qpos.astype(np.float32)[0:3]
            self.viewer.sync()
            self.episode_length_buf += 1
            self.calculate_gait_para()

        self.listener.stop()
        self.viewer.close()

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
