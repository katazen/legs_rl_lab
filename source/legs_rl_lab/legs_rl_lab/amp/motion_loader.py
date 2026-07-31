"""专家动作数据加载器。简化自 TienKung rsl_rl 的 AMPLoader:

  - 只保留 nlegs 需要的功能: 读 DeepMimic 风格 JSON, 按时间插值取帧, 预加载
    大量 (s, s') 转移, feed_forward_generator 产出判别器正样本。
  - 去掉了 vel-conditioned / vel-label / 多末端等 TienKung 专用分支。

amp_obs 帧布局 (30 维, 与 env 的 mdp.amp_obs 及 convert_tienkung_motion.py 一致):
    [0:12]  关节位置 (Isaac 序)   [12:24] 关节速度   [24:30] 左右脚 base 系位置
末端(脚)位置维度 END_EFFECTOR_POS_SIZE = 6, 关节数由帧宽推断 J=(W-6)//2。
"""

from __future__ import annotations

import glob
import json

import numpy as np
import torch


class AMPLoader:
    END_EFFECTOR_POS_SIZE = 6  # 2 只脚 x xyz

    def __init__(
        self,
        device,
        time_between_frames: float,
        motion_files=None,
        data_dir: str = "",
        preload_transitions: bool = True,
        num_preload_transitions: int = 200000,
    ):
        """time_between_frames: 一次策略步的时间 (env step_dt), 决定 (s, s') 的时间间隔。"""
        self.device = device
        self.time_between_frames = time_between_frames
        if motion_files is None:
            motion_files = glob.glob(f"{data_dir}/*")
        assert len(motion_files) > 0, f"没有找到 motion 文件: data_dir={data_dir}, files={motion_files}"

        self.trajectories = []          # 每条轨迹 [T, obs_dim]
        self.trajectory_idxs = []
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_lens = []       # 秒
        self.trajectory_num_frames = []

        # 由第一条轨迹推断维度布局
        with open(motion_files[0]) as f:
            width = np.array(json.load(f)["Frames"]).shape[1]
        j = (width - AMPLoader.END_EFFECTOR_POS_SIZE) // 2
        self.joint_pose_start_idx, self.joint_pose_end_idx = 0, j
        self.joint_vel_start_idx, self.joint_vel_end_idx = j, 2 * j
        self.end_pos_start_idx = 2 * j
        self.end_pos_end_idx = 2 * j + AMPLoader.END_EFFECTOR_POS_SIZE
        self._obs_dim = self.end_pos_end_idx

        for i, motion_file in enumerate(motion_files):
            with open(motion_file) as f:
                mj = json.load(f)
            data = np.array(mj["Frames"], dtype=np.float32)[:, : self.end_pos_end_idx]
            self.trajectories.append(torch.tensor(data, dtype=torch.float32, device=device))
            self.trajectory_idxs.append(i)
            self.trajectory_weights.append(float(mj.get("MotionWeight", 1.0)))
            fd = float(mj["FrameDuration"])
            self.trajectory_frame_durations.append(fd)
            self.trajectory_lens.append((data.shape[0] - 1) * fd)
            self.trajectory_num_frames.append(float(data.shape[0]))
            print(f"[AMPLoader] {motion_file}: {data.shape[0]} 帧, {self.trajectory_lens[-1]:.2f}s")

        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        self.preload_transitions = preload_transitions
        if preload_transitions:
            print(f"[AMPLoader] 预加载 {num_preload_transitions} 条转移 ...")
            idxs = self._weighted_traj_idx_batch(num_preload_transitions)
            times = self._traj_time_batch(idxs)
            self.preloaded_s = self._frame_at_time_batch(idxs, times)
            self.preloaded_s_next = self._frame_at_time_batch(idxs, times + self.time_between_frames)
            print("[AMPLoader] 预加载完成")

    # ---- 采样工具 ----
    def _weighted_traj_idx_batch(self, size):
        return np.random.choice(self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True)

    def _traj_time_batch(self, traj_idxs):
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
        t = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(0.0, t)

    @staticmethod
    def _slerp(a, b, blend):
        return (1.0 - blend) * a + blend * b

    def _frame_at_time_batch(self, traj_idxs, times):
        """按时间线性插值取整帧, 批量。返回 [N, obs_dim]。"""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low = np.floor(p * n).astype(np.int64)
        idx_high = np.ceil(p * n).astype(np.int64)
        starts = torch.zeros(len(traj_idxs), self._obs_dim, device=self.device)
        ends = torch.zeros(len(traj_idxs), self._obs_dim, device=self.device)
        for ti in set(traj_idxs):
            traj = self.trajectories[ti]
            m = traj_idxs == ti
            hi = np.clip(idx_high[m], 0, traj.shape[0] - 1)
            lo = np.clip(idx_low[m], 0, traj.shape[0] - 1)
            starts[m] = traj[lo]
            ends[m] = traj[hi]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self._slerp(starts, ends, blend)

    def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
        """产出判别器专家正样本 (s, s'), 各 [mini_batch_size, obs_dim]。"""
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size)
                yield self.preloaded_s[idxs], self.preloaded_s_next[idxs]
            else:
                traj_idxs = self._weighted_traj_idx_batch(mini_batch_size)
                times = self._traj_time_batch(traj_idxs)
                yield (
                    self._frame_at_time_batch(traj_idxs, times),
                    self._frame_at_time_batch(traj_idxs, times + self.time_between_frames),
                )

    @property
    def observation_dim(self) -> int:
        return self._obs_dim
