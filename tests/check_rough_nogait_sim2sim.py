"""在 MuJoCo 环境运行：检查无相位策略加载、拒绝相位模型和有限值短回放。"""

import importlib.util
from pathlib import Path
import tempfile

import numpy as np
import yaml


script = Path(__file__).resolve().parents[1] / "source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/task/rough/sim2sim_nogait.py"
spec = importlib.util.spec_from_file_location("nogait_sim2sim_check", script)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
cfg = replay.load_config(replay.RUN)
replay.rough._check_terrain()
with tempfile.TemporaryDirectory() as folder:
    params = Path(folder) / "params"
    params.mkdir()
    deploy = yaml.safe_load((Path(cfg.run_dir) / "params/deploy.yaml").read_text())
    deploy["observations"]["gait_phase"] = {"scale": [1., 1.], "history_length": 10}
    deploy["gait_period"] = 0.6
    (params / "deploy.yaml").write_text(yaml.safe_dump(deploy))
    try:
        replay.load_config(folder)
    except ValueError as error:
        assert "无相位" in str(error)
    else:
        raise AssertionError("带相位的模型应被拒绝")

cfg.base_pos[2] = replay.DROP_HEIGHT
runner = replay.rough.RoughRunner(cfg, show_viewer=False)
assert runner.history.update(runner._observation_terms()).shape == (450,)
for command in ((0., 0., 0.), (0.2, 0., 0.)):
    runner.command[:] = command
    runner.run(duration=1.0, realtime=False)
    assert np.isfinite(runner.data.qpos).all() and np.isfinite(runner.data.qvel).all()
    assert np.isfinite(runner.last_action).all() and np.isfinite(runner.last_torque).all()
    if cfg.policy_action_clip is not None:
        assert np.abs(runner.last_action).max() <= cfg.policy_action_clip
print("PASS rough_nogait sim2sim: terrain, phase rejection, 450 inputs, zero/forward command rollout, finite states and action clip")
