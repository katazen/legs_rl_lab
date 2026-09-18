"""nlegs_mimic_stand 的 MuJoCo 策略回放，不启动 Isaac Sim、不连接实机。

conda activate unitree_lab
python source/legs_rl_lab/legs_rl_lab/tasks/mimic_task/task/nlegs_stand/sim2sim.py
--run RUN 可选择其他已导出策略的训练目录；也接受 run 的绝对路径。
默认使用 2026-09-17_15-40-44；先用 play 导出该 run 的策略。
从训练参考动作首帧的限位蹲姿开始，机身姿态、高度和关节角均读取动作文件。
v2 参考共 3.35s，初始前倾约 28.28°；以所选 run 记录的动作文件为准。
聚焦窗口按小键盘/主键盘 5 开始起身；等待期间物理和参考时钟暂停。
参考结束后保持最后的站立目标，策略继续运行；再次按 5 不会重置。
离线测试：--headless --auto-start --duration 6；--save-data 保存关节跟踪数据。
"""

import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "nlegs_stand_mimic_replay", Path(__file__).resolve().parents[1] / "nlegs_crouch/sim2sim.py",
)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
replay.flat.LOGS_ROOT = str(Path(replay.flat._REPO_ROOT) / "logs/rsl_rl/nlegs_mimic_stand")
RUN = "2026-09-17_15-40-44"


def main():
    replay.main(default_run=RUN, description=__doc__)


if __name__ == "__main__":
    main()
