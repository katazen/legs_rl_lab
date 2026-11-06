"""W1196 nlegs_body_flat 的纯平地 MuJoCo 回放，不连接实机。

按文件路径运行，避免包导入启动 Isaac Lab：
    python source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/task/flat/sim2sim_body.py \
        --run RUN [--headless --duration 10] [--save-data]
    python source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/task/flat/sim2sim_body.py --check-terrain

RUN 是 logs/rsl_rl/nlegs_body_flat 下的目录名或绝对路径，须先导出 policy.pt。
策略、观测和执行器参数均读取该 run 的 params/deploy.yaml；小键盘操作与 flat 一致。
"""

import argparse
import importlib.util
from pathlib import Path
import sys

import mujoco
import numpy as np


spec = importlib.util.spec_from_file_location("nlegs_body_flat_replay", Path(__file__).with_name("sim2sim.py"))
flat = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = flat
spec.loader.exec_module(flat)
LOGS_ROOT = Path(flat._REPO_ROOT) / "logs/rsl_rl/nlegs_body_flat"
SCENE_XML = Path(flat._REPO_ROOT) / "source/legs_rl_lab/legs_rl_lab/assets/nlegs_v2/w1196_scene.xml"
flat.LOGS_ROOT = str(LOGS_ROOT)
flat.SCENE_XML = str(SCENE_XML)


def _check_ground(model):
    """固定环境的碰撞体只能是 z=0、法向向上的水平面，排除坡道和台阶。"""
    collision = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    ground_ids = np.flatnonzero(collision & (model.body_weldid[model.geom_bodyid] == 0))
    if ground_ids.size != 1 or model.geom_type[ground_ids[0]] != mujoco.mjtGeom.mjGEOM_PLANE:
        raise ValueError("nlegs_body_flat 场景必须只有一个固定的水平地面碰撞体")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ground = ground_ids[0]
    if (not np.isclose(data.geom_xpos[ground, 2], 0.0, atol=1e-8)
            or not np.allclose(data.geom_xmat[ground].reshape(3, 3)[:, 2], [0, 0, 1], atol=1e-8)):
        raise ValueError("nlegs_body_flat 地面必须位于 z=0 且法向向上")


class BodyFlatRunner(flat.MujocoRunner):
    def _make_model(self, cfg):
        model = super()._make_model(cfg)
        _check_ground(model)
        return model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="新任务导出的 run 目录名或绝对路径")
    parser.add_argument("--xml", "--scene-xml", dest="scene_xml", default=str(SCENE_XML))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--check-terrain", action="store_true", help="仅检查纯平地场景，不加载策略")
    args = parser.parse_args()
    scene_xml = str(Path(args.scene_xml).expanduser().resolve())
    if args.check_terrain:
        _check_ground(mujoco.MjModel.from_xml_path(scene_xml))
        print(f"[body_flat] 地面检查通过：z=0 水平面，无坡道或台阶。XML={scene_xml}")
        return
    if not args.run:
        parser.error("请用 --run 指定已导出的 nlegs_body_flat run")
    duration = args.duration if args.duration is not None else (10.0 if args.headless else flat.SIM_DURATION)
    if not np.isfinite(duration) or duration <= 0:
        parser.error("--duration 必须是有限正数")
    cfg = flat.load_config(str(Path(args.run).expanduser()))
    cfg.scene_xml = scene_xml
    if not Path(cfg.model_path).is_file():
        parser.error(f"没有 {cfg.model_path}；请先用 play.py --task nlegs_body_flat 导出策略")
    flat._self_check()
    print(f"[body_flat] run={cfg.run_dir}, XML={cfg.scene_xml}, action_clip=±{cfg.policy_action_clip}")
    runner = BodyFlatRunner(cfg, show_viewer=not args.headless, save_data=args.save_data)
    runner.run(duration=duration, realtime=not args.headless)
    print(f"[body_flat] 完成 {runner.data.time:.2f}s，base_z={runner.data.xpos[runner.base_body_id, 2]:.3f}m")


if __name__ == "__main__":
    main()
