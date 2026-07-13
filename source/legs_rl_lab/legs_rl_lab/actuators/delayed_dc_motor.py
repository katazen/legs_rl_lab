# -*- coding: utf-8 -*-
"""带延迟的 DCMotor 执行器 = DelayedPDActuator(命令延迟) + DCMotor(转矩-转速滚降)。

IsaacLab 内置里 DelayedPDActuator 有延迟无转矩-转速曲线, DCMotor 有曲线无延迟, 二者不可兼得。
本类继承 DelayedPDActuator 拿到命令延迟(min_delay..max_delay 随机), 并用 DCMotor 的
_clip_effort(转矩-转速直线, 反电动势)替换掉恒定 effort 限幅, 从而同时建模:
  - 传输/处理延迟(相位)
  - 电机随速度下降的可用力矩(高速跟踪幅值滚降) —— 见实机膝辨识: 1.5Hz 幅值仅 ~0.68

力矩上下限(与 isaaclab.actuators.DCMotor 完全一致, 关节侧量):
  τ_max(q̇) = clip( sat·(1 − q̇/vlim),  max=+effort_limit )
  τ_min(q̇) = clip( sat·(−1 − q̇/vlim), min=−effort_limit )
  τ_applied = clip(τ_PD, τ_min, τ_max)
参数: saturation_effort(堵转力矩), velocity_limit(空载转速), effort_limit(持续力矩平台)。
辨识拟合值(膝, 关节侧): sat=26, velocity_limit≈2.3, effort_limit=26。
"""
from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.actuators import DelayedPDActuator
from isaaclab.actuators.actuator_pd_cfg import DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class DelayedDCMotor(DelayedPDActuator):
    """DelayedPDActuator + DCMotor 转矩-转速限幅。"""

    cfg: DelayedDCMotorCfg

    def __init__(self, cfg: DelayedDCMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)   # DelayedPD: 延迟缓冲 + IdealPD/base 初始化
        if cfg.saturation_effort is None:
            raise ValueError("DelayedDCMotor 需要 saturation_effort。")
        self._saturation_effort = cfg.saturation_effort
        # 力矩-转速线与持续力矩的交点速度(与 DCMotor 一致), clip 用
        self._vel_at_effort_lim = self.velocity_limit * (1 + self.effort_limit / self._saturation_effort)
        self._joint_vel = torch.zeros_like(self.computed_effort)
        self._zeros_effort = torch.zeros_like(self.computed_effort)
        if cfg.velocity_limit is None:
            raise ValueError("DelayedDCMotor 需要 velocity_limit(空载转速)。")

    def compute(self, control_action: ArticulationActions, joint_pos: torch.Tensor,
                joint_vel: torch.Tensor) -> ArticulationActions:
        # DCMotor 的限幅要用当前关节速度, 先存下来; 再走 DelayedPD.compute(延迟)->IdealPD.compute->_clip_effort
        self._joint_vel[:] = joint_vel
        return super().compute(control_action, joint_pos, joint_vel)

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        # 与 isaaclab.actuators.DCMotor._clip_effort 相同
        self._joint_vel[:] = torch.clip(self._joint_vel, min=-self._vel_at_effort_lim, max=self._vel_at_effort_lim)
        torque_speed_top = self._saturation_effort * (1.0 - self._joint_vel / self.velocity_limit)
        torque_speed_bottom = self._saturation_effort * (-1.0 - self._joint_vel / self.velocity_limit)
        max_effort = torch.clip(torque_speed_top, max=self.effort_limit)
        min_effort = torch.clip(torque_speed_bottom, min=-self.effort_limit)
        return torch.clip(effort, min=min_effort, max=max_effort)


@configclass
class DelayedDCMotorCfg(DelayedPDActuatorCfg):
    """DelayedDCMotor 配置: DelayedPD(min_delay/max_delay) + DCMotor(saturation_effort)。

    注意: velocity_limit 用【模型量】(不是 velocity_limit_sim), 否则不进转矩-转速曲线。
    """

    class_type: type = DelayedDCMotor

    saturation_effort: float = MISSING
    """堵转力矩 τ_stall [N·m](关节侧), 转矩-转速线在 v=0 的截距。"""


__all__ = ["DelayedDCMotor", "DelayedDCMotorCfg"]
