import os
import time
import types
from collections import deque
import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml
from pynput import keyboard


def _find_repo_root(start: str) -> str:
    """从本文件上溯找仓库根（含 source/ 与 scripts/），避免任何绝对路径。"""
    d = os.path.dirname(os.path.abspath(start))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "source")) and os.path.isdir(os.path.join(d, "scripts")):
            return d
        d = os.path.dirname(d)
    return d


_REPO_ROOT = _find_repo_root(__file__)
_ASSETS_DIR = os.path.join(_REPO_ROOT, "source", "legs_rl_lab", "legs_rl_lab", "assets")


# ============================================================================
#  唯一必填变量：训练 run 的日期文件夹名
#  sim2sim 会据此自动读取
#      logs/rsl_rl/<EXPERIMENT>/<RUN>/params/deploy.yaml   （所有模型参数）
#      logs/rsl_rl/<EXPERIMENT>/<RUN>/exported/policy.pt   （策略，需先 play 导出）
RUN = "2026-07-16_21-01-52"   # TODO: 换成你实际的 legs 训练 run（logs/rsl_rl/legs/<RUN>）
#  唯一可选变量：是否采集关节跟踪数据并出图
SAVE_DATA = False
# ============================================================================

# 与具体 run 无关的仿真侧设置（不是模型参数，故不放进 deploy.yaml）
EXPERIMENT = "legs"
LOGS_ROOT = os.path.join(_REPO_ROOT, "logs", "rsl_rl")
# MuJoCo 场景 xml（含 actuator/imu 传感器，与 USD 无关）
SCENE_XML = os.path.join(_ASSETS_DIR, "legs_URDF", "mjcf", "A1_legs_V2_mjcf_scene.xml")
PHYS_DT = 0.005            # MuJoCo 物理步长（仿真选择）
CONTROL_MODE = "motor"     # "motor" or "position"
SIM_DURATION = 10000.0
COLLECT_DURATION = 10.0    # 采集时长 (秒)

# 关节顺序：isaac(策略/yaml) vs mujoco
ISAAC_JOINT = ['R1', 'L1', 'R2', 'L2', 'R3', 'L3', 'R4', 'L4', 'R5', 'L5', 'R6', 'L6']
MJC_JOINT = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6']
_ISAAC2MJC = np.array([ISAAC_JOINT.index(j) for j in MJC_JOINT])

# ---- 标0偏置注入(sim2sim 验证用)----
# 模拟实机编码器零位标定误差: 机器人"以为"某关节在 x, 客观(物理)在 x+δ。
# 键=关节名(mjc/sdk 序: R1..R6,L1..L6; 1髋pitch 2髋roll 3髋yaw 4膝 5踝pitch 6踝roll),
# 值=物理偏置 δ(rad)。未列出的关节=0。留空 {} 关闭注入。
# 原理: 凡从 MuJoCo 读关节角喂给 obs / PD 的地方都用 (qpos - δ) 作"上报值",
#       物理 qpos 与施加力矩不变 -> 策略被瞒着、以为在 x, 实则在 x+δ。
ZERO_OFFSET = {}   # 例: {"L4": 0.05, "R2": -0.03}


# deploy.yaml 里 obs 术语名 -> get_obs 用的特征键
_OBS_NAME_MAP = {
    "base_ang_vel": "ang_vel",
    "projected_gravity": "gravity",
    "velocity_commands": "command",
    "joint_pos_rel": "dof_pos",
    "joint_vel_rel": "dof_vel",
    "last_action": "actions",
    "gait_phase": "gait",
}


def build_cfg(run: str, save_data: bool = False):
    """从某个训练 run 的 deploy.yaml 构造 sim2sim 配置（不依赖 mujoco，可单独测试）。"""
    run_dir = os.path.join(LOGS_ROOT, EXPERIMENT, run)
    deploy_path = os.path.join(run_dir, "params", "deploy.yaml")
    model_path = os.path.join(run_dir, "exported", "policy.pt")
    with open(deploy_path) as f:
        d = yaml.safe_load(f)

    def _req(key):
        """取 deploy.yaml 字段; 缺失直接报错, 绝不用默认兜底(避免静默用错配置)。"""
        if key not in d:
            raise KeyError(f"[sim2sim] deploy.yaml 缺字段 '{key}' ({deploy_path}); "
                           f"请重新导出 deploy.yaml, 不使用默认兜底")
        return d[key]

    action_dim = len(_req("default_joint_pos"))
    step_dt = float(_req("step_dt"))
    decimation = max(1, round(step_dt / PHYS_DT))

    # observations: 按 yaml 顺序推出切片 / state_dim / history_length
    obs = _req("observations")
    his_lens = int(next(iter(obs.values()))["history_length"])
    obs_slices, cursor = {}, 0
    for term_name, term in obs.items():
        dim = len(term["scale"])
        obs_slices[_OBS_NAME_MAP.get(term_name, term_name)] = (cursor, cursor + dim)
        cursor += dim
    state_dim = cursor

    action_scale = float(_req("actions")["JointPositionAction"]["scale"][0])
    r = _req("commands")["base_velocity"]["ranges"]
    cmd_range = [list(r["lin_vel_x"]), list(r["lin_vel_y"]), list(r["ang_vel_z"])]

    # default_joint_pos: yaml 为 isaac 顺序 -> 转 mjc 顺序
    default_mjc = np.array(_req("default_joint_pos"), dtype=np.float32)[_ISAAC2MJC]

    # stiffness/damping/armature/effort: yaml 已是 sdk(=mjc) 顺序, 直接用; 缺任一字段直接报错
    stiffness = np.array(_req("stiffness"), dtype=np.float32)
    damping = np.array(_req("damping"), dtype=np.float32)
    armature = np.array(_req("armature"), dtype=np.float32)
    effort = np.array(_req("effort"), dtype=np.float32)
    gait_cycle = float(_req("gait_period"))
    action_delay_range = tuple(int(x) for x in _req("action_delay"))

    path = types.SimpleNamespace(pos_xml_path=SCENE_XML, tau_xml_path=SCENE_XML, model_path=model_path)
    sim = types.SimpleNamespace(
        sim_duration=SIM_DURATION, control_mode=CONTROL_MODE, action_dim=action_dim,
        state_dim=state_dim, dt=PHYS_DT, decimation=decimation, gait_cycle=gait_cycle,
        action_delay_range=action_delay_range, his_lens=his_lens,
        collect_data=save_data, collect_duration=COLLECT_DURATION, obs_slices=obs_slices,
    )
    robot = types.SimpleNamespace(
        default_dof_pos=default_mjc, reset_dof_pos=default_mjc.copy(),
        armature=armature, effort=effort, stiffness=stiffness, damping=damping,
        action_scale=action_scale, cmd_range=cmd_range,
    )
    return types.SimpleNamespace(path=path, sim=sim, robot=robot)


class LatencySimulator:
    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        self.action_buffer = None
        self.action_delay_range = None

    def reset(self, action_delay_range: tuple):
        """
        在 Episode 重置时调用。
        :param action_delay_range: (min, max) 动作延迟步数范围（含端点，与 isaaclab 一致）
        """
        self.action_delay_range = tuple(int(x) for x in action_delay_range)
        self.action_buffer = deque(maxlen=self.action_delay_range[1] + 1)
        print(f"[Latency] Action Delay Range: {self.action_delay_range} steps")

    def process_action(self, new_action: np.ndarray) -> np.ndarray:
        """
        [在物理步调用]
        输入策略产生的最新动作，推入队列，并返回延迟后的动作。
        reset 后历史不足时，将 delay clamp 到已有历史，避免先执行一段全零动作。
        """
        self.action_buffer.append(new_action.astype(np.float32, copy=True))
        act_delay = np.random.randint(self.action_delay_range[0], self.action_delay_range[1] + 1)
        valid_delay = min(act_delay, len(self.action_buffer) - 1)
        return self.action_buffer[-1 - valid_delay].copy()


class MujocoRunner:
    def __init__(self, cfg):
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
        self.obs_his = np.zeros((self.cfg.sim.his_lens, self.state_dim), dtype=np.float32)
        self.sim_duration = self.cfg.sim.sim_duration
        self.decimation = self.cfg.sim.decimation
        self.dt = self.decimation * self.cfg.sim.dt
        self.dof_pos = np.zeros(self.action_dim)
        self.dof_vel = np.zeros(self.action_dim)
        self.action = np.zeros(self.action_dim, dtype=np.float32)
        self.raw_action = np.zeros(self.action_dim, dtype=np.float32)
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
        # 标0偏置(mjc 序): 由 ZERO_OFFSET 字典按关节名展开; 全 0 = 不注入
        self.zero_offset = np.array([ZERO_OFFSET.get(n, 0.0) for n in self.mjc_joint], dtype=np.float32)
        if np.any(self.zero_offset != 0.0):
            print(f"[sim2sim] 标0偏置注入(mjc序 rad): {dict(zip(self.mjc_joint, self.zero_offset))}")
        self.command_vel = np.array([0.0, 0.0, 0.0])
        self.cmd_range = self.cfg.robot.cmd_range
        self.joint_ids = np.array([self._joint_id(name) for name in self.mjc_joint], dtype=np.int32)
        joint_ranges = self.model.jnt_range[self.joint_ids].astype(np.float32)
        joint_limited = self.model.jnt_limited[self.joint_ids].astype(bool)
        self.joint_pos_min = np.where(joint_limited, joint_ranges[:, 0], -np.inf).astype(np.float32)
        self.joint_pos_max = np.where(joint_limited, joint_ranges[:, 1], np.inf).astype(np.float32)

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

        # ---- DCMotor 转矩-转速滚降参数(与 legs.py 一致, mjc 序 R1..R6,L1..L6) ----
        # 只有髋pitch(.*1)和膝(.*4)用 DCMotor: (velocity_limit, saturation_effort)
        _DC = {"R1": (2.2, 26.0), "L1": (2.2, 26.0), "R4": (3.0, 26.0), "L4": (2.3, 26.0)}
        _eff = np.asarray(self.cfg.robot.effort, dtype=np.float32)
        _vlim, _sat = [], []
        for j, name in enumerate(self.mjc_joint):
            if name in _DC:
                v, s = _DC[name]; _vlim.append(v); _sat.append(s)
            else:                                  # 非 DCMotor: vlim=inf → 退化成普通 ±effort clip
                _vlim.append(np.inf); _sat.append(float(_eff[j]))
        self._dc_vlim = np.asarray(_vlim, dtype=np.float32)
        self._dc_sat = np.asarray(_sat, dtype=np.float32)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos = np.concatenate([np.array([0, 0, 0.55], dtype=np.float32),
                                         np.array([1, 0, 0, 0], dtype=np.float32), self.reset_dof_pos])
        mujoco.mj_forward(self.model, self.data)
        self.latency_sim.reset(self.cfg.sim.action_delay_range)
        self.reset_obs_history()
        self.start_time = time.time()
        self.his_data = []
        self.collect_data = self.cfg.sim.collect_data    # 是否采集
        self.collect_T = self.cfg.sim.collect_duration   # 采集时长 (秒)
        self.track_log = []          # 关节跟踪采集: (t, target_pos_mjc, qpos_mjc, cmd_vel, tau_mjc)
        self._tau = np.zeros(self.action_dim, dtype=np.float32)  # 最近一步应用力矩(mjc序, 已 clip effort)

    def _joint_id(self, short_name: str) -> int:
        joint_name = short_name if short_name.startswith("joint_") else f"joint_{short_name}"
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo XML 缺少关节: {joint_name}")
        return joint_id

    def reset_obs_history(self):
        self.obs_his[:] = self.compute_obs()

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

    def compute_obs(self):
        # qpos 是物理真值; 减标0偏置 -> "上报值"(策略以为的位置)。offset=0 时无影响。
        self.dof_pos = self.data.qpos[7:].astype(np.float32) - self.zero_offset
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
        obs[9 + 2 * self.action_dim:9 + 3 * self.action_dim] = self.raw_action
        obs[9 + 3 * self.action_dim:10 + 3 * self.action_dim] = np.sin(2 * torch.pi * self.gait_phase)
        obs[10 + 3 * self.action_dim:11 + 3 * self.action_dim] = np.cos(2 * torch.pi * self.gait_phase)
        return obs

    def get_obs(self):
        obs = self.compute_obs()
        self.update_obs_his(obs)
        return self.get_inference_input()


    def compute_target_pos(self, action: np.ndarray | None = None):
        if action is None:
            action = self.action
        actions_scaled = action * self.action_scale
        target_pos = actions_scaled[self.isaac2mjc] + self.default_dof_pos
        return np.clip(target_pos, self.joint_pos_min, self.joint_pos_max)

    def compute_torque(self):
        target_pos = self.compute_target_pos()
        tau = self.cfg.robot.stiffness * (target_pos - self.dof_pos) - self.cfg.robot.damping * self.dof_vel
        # DCMotor 转矩-转速限幅(与 Isaac DelayedDCMotor._clip_effort 一致):
        #   τ_max(v)=clip(sat·(1−v/vlim), max=+effort);  τ_min(v)=clip(sat·(−1−v/vlim), min=−effort)
        # 非 DCMotor 关节 vlim=inf → 退化成普通 ±effort clip。缺这条会导致 sim2sim 飞车(执行器与训练不一致)。
        eff = np.asarray(self.cfg.robot.effort, dtype=np.float32)
        v = self.dof_vel
        vel_at_lim = self._dc_vlim * (1.0 + eff / self._dc_sat)   # 速度预夹(inf 关节不夹)
        vv = np.clip(v, -vel_at_lim, vel_at_lim)
        tmax = np.minimum(self._dc_sat * (1.0 - vv / self._dc_vlim), eff)
        tmin = np.maximum(self._dc_sat * (-1.0 - vv / self._dc_vlim), -eff)
        return np.clip(tau, tmin, tmax)

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
        if self.listener is not None:
            self.listener.start()
        while self.data.time < self.sim_duration:
            input_obs = self.get_obs().flatten()
            raw_policy_action = self.policy(torch.tensor(input_obs, dtype=torch.float32)).detach().numpy()
            # 与训练 A1Env.step 对齐：对 policy 输出做每步 ±action_rate_limit 增量裁剪
            # delta = np.clip(raw_policy_action - self.prev_action, -self.action_rate_limit, self.action_rate_limit)
            # clamped_action = (self.prev_action + delta).astype(np.float32)
            # self.prev_action = clamped_action
            self.raw_action[:] = raw_policy_action
            for _ in range(self.decimation):
                self.action[:] = self.latency_sim.process_action(raw_policy_action)
                if self.control_mode == "position":
                    # MuJoCo 内部 PD 用物理 qpos, 故 ctrl 加回偏置: kp*(target+δ - qpos)=kp*(target - 上报)
                    self.data.ctrl = self.compute_target_pos() + self.zero_offset
                else:
                    self.dof_pos = self.data.qpos[7:].astype(np.float32) - self.zero_offset
                    self.dof_vel = self.data.qvel[6:].astype(np.float32)
                    self._tau = self.compute_torque()
                    self.data.ctrl = self._tau
                mujoco.mj_step(self.model, self.data)
            # --- 采集关节跟踪 (mjc 序 R1..R6,L1..L6) ---
            # target 取「策略决策时刻、延迟前」的目标(与实机发布 /dog_joint_pos 口径一致),
            # qpos 是经 action_delay 后的实际响应 -> 这样链路死区(action_delay)在图上才显现。
            if self.collect_data and self.data.time <= self.collect_T:
                raw_target = self.compute_target_pos(raw_policy_action)
                self.track_log.append((float(self.data.time),
                                       raw_target.astype(np.float32).copy(),
                                       self.data.qpos[7:].astype(np.float32).copy(),
                                       self.command_vel.copy(),
                                       np.asarray(self._tau, np.float32).copy()))
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

        if self.listener is not None:
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
        tau = np.array([r[4] for r in self.track_log])        # (N,12) mjc 序, 已 clip effort
        eff = np.asarray(self.cfg.robot.effort, dtype=float)  # mjc 序 effort 上限
        # 重排成 [L1..L6,R1..R6], 与实机图一致
        names = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6']
        order = [self.mjc_ctrl.index(n) for n in names]
        import datetime
        run_dir = os.path.dirname(os.path.dirname(self.cfg.path.model_path))  # <训练run>/(exported 的上级)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")             # 保存时间
        outdir = os.path.join(run_dir, "sim2sim"); os.makedirs(outdir, exist_ok=True)
        fig_dir = os.path.join(outdir, stamp); os.makedirs(fig_dir, exist_ok=True)
        # csv
        csv_path = os.path.join(outdir, f"{stamp}.csv")
        with open(csv_path, "w") as f:
            f.write("t," + ",".join(f"target_{n}" for n in names) + "," +
                    ",".join(f"sim_{n}" for n in names) + "," +
                    ",".join(f"tau_{n}" for n in names) + ",cmd_vx,cmd_vy,cmd_yaw\n")
            for k in range(len(t)):
                row = [t[k]] + [tgt[k, j] for j in order] + [act[k, j] for j in order] + \
                      [tau[k, j] for j in order] + list(cmd[k])
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
        png = os.path.join(fig_dir, "track.png")
        plt.savefig(png, dpi=100); plt.close()
        # 力矩图: 每关节 tau 与 ±effort 上限, 标注饱和%(踝 L5/R5 重点)
        fig2, ax2 = plt.subplots(4, 3, figsize=(20, 12), sharex=True)
        for i, n in enumerate(names):
            j = order[i]; a = ax2[i // 3, i % 3]
            a.plot(t, tau[:, j], "-", color="tab:purple", lw=1.0)
            a.axhline(eff[j], color="r", ls=":", lw=1.0); a.axhline(-eff[j], color="r", ls=":", lw=1.0)
            sat = np.mean(np.abs(tau[:, j]) >= eff[j] - 1e-3) * 100
            a.set_title(f"{n} tau (limit +-{eff[j]:.0f}Nm, sat {sat:.0f}%)", fontsize=11); a.grid(alpha=.3)
        for c in range(3): ax2[3, c].set_xlabel("t [s]")
        fig2.suptitle("sim2sim joint torque (red dotted = effort limit)", fontsize=12)
        plt.tight_layout(); png_t = os.path.join(fig_dir, "torque.png")
        plt.savefig(png_t, dpi=100); plt.close()
        print(f"[采集] 力矩图 -> {png_t}")
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
        if keyboard is None:
            self.listener = None
            print("[sim2sim] 警告: 未安装 pynput，键盘速度控制不可用")
            return

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
    sim_cfg = build_cfg(RUN, SAVE_DATA)
    runner = MujocoRunner(cfg=sim_cfg)
    runner.run()
