"""CPU regression for the existing 10-frame layouts and body_rough's 15 frames."""

import importlib.util
from pathlib import Path

import torch
from tensordict import TensorDict


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/mdp/symmetry.py"
spec = importlib.util.spec_from_file_location("observation_symmetry", PATH)
symmetry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(symmetry)


def test_all_layouts_are_involutions():
    for group, dims in {"policy": [470, 646, 705], "critic": [620, 2490, 807, 796, 3735]}.items():
        for dim in dims:
            values = torch.arange(dim, dtype=torch.float32)[None, :]
            perm, sign = symmetry._mirror_maps(group, dim, torch.device("cpu"))
            mirrored = values[:, perm] * sign
            torch.testing.assert_close(mirrored[:, perm] * sign, values)


def test_body_frames_and_grid():
    policy = torch.arange(705, dtype=torch.float32).reshape(1, -1)
    critic = torch.arange(3735, dtype=torch.float32).reshape(1, -1)
    actions = torch.arange(12, dtype=torch.float32).reshape(1, -1)
    obs = TensorDict({"policy": policy, "critic": critic}, batch_size=[1])
    augmented, augmented_actions = symmetry.compute_symmetric_states(obs=obs, actions=actions)
    assert augmented.batch_size == torch.Size([2])
    torch.testing.assert_close(augmented["policy"][0], policy[0])
    torch.testing.assert_close(augmented["critic"][0], critic[0])

    # Each 3-vector frame stays in its own time slot; only roll/yaw signs flip.
    torch.testing.assert_close(
        augmented["policy"][1, :45].reshape(15, 3),
        policy[0, :45].reshape(15, 3) * torch.tensor([-1.0, 1.0, -1.0]),
    )
    # The first joint block starts after angular velocity, gravity and commands.
    joint_frames = policy[0, 135:315].reshape(15, 6, 2)
    expected_joints = joint_frames.flip(-1) * torch.tensor([1., -1., -1., 1., 1., -1.])[None, :, None]
    torch.testing.assert_close(augmented["policy"][1, 135:315], expected_joints.flatten())
    # Mirror shifts phase by half a period without reversing history.
    torch.testing.assert_close(augmented["policy"][1, 675:705], -policy[0, 675:705])
    torch.testing.assert_close(augmented["critic"][1, 900:930], -critic[0, 900:930])
    # Every one of 15 height grids flips y, retaining x and time order.
    heights = critic[0, 930:].reshape(15, 11, 17)
    torch.testing.assert_close(augmented["critic"][1, 930:], heights.flip(1).flatten())
    expected_actions = torch.tensor([[1., 0., -3., -2., -5., -4., 7., 6., 9., 8., -11., -10.]])
    torch.testing.assert_close(augmented_actions[1:], expected_actions)


if __name__ == "__main__":
    test_all_layouts_are_involutions()
    test_body_frames_and_grid()
    print("PASS: 10/15-frame symmetry, joint/phase signs, time ordering, height grids and actions")
