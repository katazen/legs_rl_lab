from isaaclab.envs.mdp.actions import JointPositionAction


class ZeroBiasJointPositionAction(JointPositionAction):
    """编码器角 q_enc = q_phys + b；将编码器目标换算为物理目标。"""

    def apply_actions(self):
        # processed_actions 保留原有的编码器坐标限幅与导出语义。
        target = self.processed_actions
        bias = getattr(self._env, "_joint_zero_bias", None)
        if bias is not None:
            target = target - bias[:, self._joint_ids]
        # 不在物理坐标重新裁剪目标：零偏可能使电机顶住限位，应由 PhysX 模拟。
        # 不原地修改 processed_actions，避免每个物理子步重复减去偏置。
        self._asset.set_joint_position_target(target, joint_ids=self._joint_ids)
