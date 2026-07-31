"""AMPPPO: 在 rsl_rl 5.0.1 官方 PPO 上最小化叠加 AMP (对抗式动作先验)。

设计要点(见 amp/__init__.py 的说明):
  - obs 是 TensorDict, amp_obs 走一个名为 "amp" 的观测组, 训练时用 obs["amp"] 取到,
    无需自定义 runner / 侧信道。
  - 只重写 6 个钩子:
      construct_algorithm  —— 弹出 amp_* 配置, 构造判别器/专家数据/归一化器/回放缓存,
                              复用 PPO.construct_algorithm 造 actor/critic/storage;
      act                  —— 记录当前 amp_obs 作为转移的 s;
      process_env_step     —— 用判别器把 style reward 与任务 reward 混合成最终 reward,
                              并把非终止的 (s, s') 存进策略回放缓存;
      update               —— 先训判别器(独立 Adam), 再调用 PPO.update 训策略, 合并日志;
      save / load          —— 额外存取判别器 + 判别器优化器 + 归一化器。
  - 判别器用独立优化器, 不掺进 PPO 的 actor/critic 优化器, 从而 PPO 的自适应学习率
    (KL 调整)不会误伤判别器。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms import PPO

from .discriminator import Discriminator
from .motion_loader import AMPLoader
from .normalizer import Normalizer
from .replay_buffer import ReplayBuffer


class AMPPPO(PPO):
    # ---- 构造 ----
    @staticmethod
    def construct_algorithm(obs, env, cfg, device) -> "AMPPPO":
        alg_cfg = cfg["algorithm"]
        # 弹出 amp_* 专属配置, 使 PPO.__init__ 只收到它认识的参数
        amp_reward_coef = alg_cfg.pop("amp_reward_coef", 2.0)
        amp_reward_lerp = alg_cfg.pop("amp_reward_lerp", 0.3)
        amp_discr_hidden_dims = alg_cfg.pop("amp_discr_hidden_dims", [1024, 512])
        amp_replay_buffer_size = alg_cfg.pop("amp_replay_buffer_size", 100000)
        amp_learning_rate = alg_cfg.pop("amp_learning_rate", 1e-4)
        amp_grad_pen_lambda = alg_cfg.pop("amp_grad_pen_lambda", 10.0)
        amp_num_preload_transitions = alg_cfg.pop("amp_num_preload_transitions", 200000)
        amp_motion_files = alg_cfg.pop("amp_motion_files", None)
        amp_data_dir = alg_cfg.pop("amp_data_dir", "")

        # amp_obs 单状态维度
        amp_dim = int(obs["amp"].shape[-1])
        # (s, s') 时间间隔 = 一次策略步时长
        time_between_frames = float(getattr(env.unwrapped, "step_dt", 0.02))

        # 复用官方流程造 actor/critic/storage/alg(此时 alg_cfg 已只剩 PPO 参数)
        alg: AMPPPO = PPO.construct_algorithm(obs, env, cfg, device)

        # 专家数据 / 判别器 / 归一化器 / 策略回放缓存
        amp_data = AMPLoader(
            device=device,
            time_between_frames=time_between_frames,
            motion_files=amp_motion_files,
            data_dir=amp_data_dir,
            preload_transitions=True,
            num_preload_transitions=amp_num_preload_transitions,
        )
        discriminator = Discriminator(
            input_dim=amp_dim,
            amp_reward_coef=amp_reward_coef,
            hidden_layer_sizes=amp_discr_hidden_dims,
            device=device,
            task_reward_lerp=amp_reward_lerp,
        ).to(device)
        amp_normalizer = Normalizer(amp_dim)
        amp_storage = ReplayBuffer(amp_dim, amp_replay_buffer_size, device)

        alg._setup_amp(discriminator, amp_data, amp_normalizer, amp_storage,
                       amp_learning_rate, amp_grad_pen_lambda)
        return alg

    def _setup_amp(self, discriminator, amp_data, amp_normalizer, amp_storage,
                   amp_learning_rate, amp_grad_pen_lambda):
        self.discriminator = discriminator
        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer
        self.amp_storage = amp_storage
        self.amp_grad_pen_lambda = amp_grad_pen_lambda
        # 判别器独立优化器: trunk / head 用不同 weight_decay (对齐 TienKung)
        self.amp_optimizer = optim.Adam(
            [
                {"params": self.discriminator.trunk.parameters(), "weight_decay": 1e-3},
                {"params": self.discriminator.amp_linear.parameters(), "weight_decay": 1e-2},
            ],
            lr=amp_learning_rate,
        )
        self._cur_amp = None  # rollout 中暂存当前 amp_obs(转移的 s)

    # ---- rollout ----
    def act(self, obs):
        self._cur_amp = obs["amp"].clone()
        return super().act(obs)

    def process_env_step(self, obs, rewards, dones, extras):
        amp_state = self._cur_amp
        amp_next = obs["amp"]
        # style reward(内部按 task_reward_lerp 与任务 reward 混合)
        style_reward, _ = self.discriminator.predict_amp_reward(
            amp_state, amp_next, task_reward=rewards, normalizer=self.amp_normalizer
        )
        style_reward = style_reward.view_as(rewards)
        # 只把非终止转移存入策略回放缓存(终止步的 s' 跨了 reset, 不是真实动力学转移)
        keep = (dones == 0).flatten()
        self.amp_storage.insert(amp_state[keep], amp_next[keep])
        # 交给官方 PPO 处理 reward 克隆 / timeout bootstrap / storage
        super().process_env_step(obs, style_reward, dones, extras)

    # ---- 更新 ----
    def _train_discriminator(self):
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mini_batch_size = self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        policy_gen = self.amp_storage.feed_forward_generator(num_updates, mini_batch_size)
        expert_gen = self.amp_data.feed_forward_generator(num_updates, mini_batch_size)

        mse = nn.MSELoss()
        sums = dict(amp=0.0, grad_pen=0.0, policy_pred=0.0, expert_pred=0.0)
        for (pol_s, pol_ns), (exp_s, exp_ns) in zip(policy_gen, expert_gen):
            if self.amp_normalizer is not None:
                with torch.no_grad():
                    pol_s_n = self.amp_normalizer.normalize_torch(pol_s, self.device)
                    pol_ns_n = self.amp_normalizer.normalize_torch(pol_ns, self.device)
                    exp_s_n = self.amp_normalizer.normalize_torch(exp_s, self.device)
                    exp_ns_n = self.amp_normalizer.normalize_torch(exp_ns, self.device)
            else:
                pol_s_n, pol_ns_n, exp_s_n, exp_ns_n = pol_s, pol_ns, exp_s, exp_ns

            policy_d = self.discriminator(torch.cat([pol_s_n, pol_ns_n], dim=-1))
            expert_d = self.discriminator(torch.cat([exp_s_n, exp_ns_n], dim=-1))
            expert_loss = mse(expert_d, torch.ones_like(expert_d))
            policy_loss = mse(policy_d, -torch.ones_like(policy_d))
            amp_loss = 0.5 * (expert_loss + policy_loss)
            grad_pen = self.discriminator.compute_grad_pen(exp_s_n, exp_ns_n, lambda_=self.amp_grad_pen_lambda)

            self.amp_optimizer.zero_grad()
            (amp_loss + grad_pen).backward()
            self.amp_optimizer.step()

            # 用原始(未归一化)样本更新归一化统计
            if self.amp_normalizer is not None:
                self.amp_normalizer.update(pol_s.cpu().numpy())
                self.amp_normalizer.update(exp_s.cpu().numpy())

            sums["amp"] += amp_loss.item()
            sums["grad_pen"] += grad_pen.item()
            sums["policy_pred"] += policy_d.mean().item()
            sums["expert_pred"] += expert_d.mean().item()

        return {f"amp_{k}": v / max(num_updates, 1) for k, v in sums.items()}

    def update(self):
        amp_losses = self._train_discriminator()  # 先训判别器(会消费 self.storage 的计数, 但不清空)
        loss_dict = super().update()              # 再训策略(内部会清空 storage)
        loss_dict.update(amp_losses)
        return loss_dict

    # ---- checkpoint ----
    def save(self):
        d = super().save()
        d["discriminator_state_dict"] = self.discriminator.state_dict()
        d["amp_optimizer_state_dict"] = self.amp_optimizer.state_dict()
        d["amp_normalizer"] = self.amp_normalizer.state_dict() if self.amp_normalizer else None
        return d

    def load(self, loaded_dict, load_cfg, strict):
        out = super().load(loaded_dict, load_cfg, strict)
        if "discriminator_state_dict" in loaded_dict:
            self.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"], strict=strict)
            if load_cfg is None or load_cfg.get("optimizer", True):
                self.amp_optimizer.load_state_dict(loaded_dict["amp_optimizer_state_dict"])
            if loaded_dict.get("amp_normalizer") is not None and self.amp_normalizer is not None:
                self.amp_normalizer.load_state_dict(loaded_dict["amp_normalizer"])
        return out
