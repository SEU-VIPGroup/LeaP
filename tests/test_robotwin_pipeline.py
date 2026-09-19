"""CPU checks of data conversion, training checkpoints and the deployment interface."""
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np
import torch
import yaml
import zarr

from leap.robotwin.data import RobotDataset, convert_episodes
from leap.robotwin.checkpoint import load_policy
from leap.robotwin.deploy import Model, eval as deploy_eval, reset_model
from leap.robotwin.train import update_ema

ROOT = Path(__file__).resolve().parents[1]


class RobotwinPipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="leap-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(4)
        self.states = []
        for i in range(2):
            state = rng.normal(size=(12, 14)).astype(np.float32)
            pc = rng.normal(size=(12, 32, 6)).astype(np.float32)
            with h5py.File(self.root / f"episode{i}.hdf5", "w") as episode:
                episode.create_dataset("joint_action/vector", data=state)
                episode.create_dataset("pointcloud", data=pc)
            self.states.append(state)
        self.zarr_path = self.root / "demo.zarr"
        convert_episodes(self.root, self.zarr_path, 2)

    def test_alignment_padding_and_no_overwrite(self):
        root = zarr.open_group(str(self.zarr_path), mode="r")
        np.testing.assert_array_equal(root["data/state"][:11], self.states[0][:-1])
        np.testing.assert_array_equal(root["data/action"][:11], self.states[0][1:])
        np.testing.assert_array_equal(root["meta/episode_ends"][:], [11, 22])
        dataset = RobotDataset(self.zarr_path)
        self.assertEqual(len(dataset), 22)
        np.testing.assert_array_equal(dataset[0]["obs"]["agent_pos"][:3],
                                      np.repeat(self.states[0][:1], 3, axis=0))
        np.testing.assert_array_equal(dataset[10]["action"][-6:],
                                      np.repeat(self.states[0][-1:], 6, axis=0))
        with self.assertRaises(FileExistsError):
            convert_episodes(self.root, self.zarr_path, 2)

    def test_training_checkpoint_reload_and_deployment(self):
        config = yaml.safe_load((ROOT / "configs/leap.yaml").read_text())
        config["policy"].update(encoder_output_dim=16, encoder_hidden_dims=[16, 16],
                                 flow_net_hidden_dim=32, flow_net_num_layers=1,
                                 flow_net_time_embed_dim=16)
        config["training"]["warmup_steps"] = 0
        config_path = self.root / "config.yaml"
        config_path.write_text(yaml.safe_dump(config))
        subprocess.run([sys.executable, "-m", "leap.robotwin.train", "--config", str(config_path),
                        "--data", str(self.zarr_path), "--output", str(self.root / "run"),
                        "--device", "cpu", "--batch-size", "2", "--epochs", "1", "--max-steps", "2"],
                       check=True, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        policy = load_policy(self.root / "run/checkpoints/latest.ckpt")
        wrapper = Model(policy)

        class Environment:
            def __init__(self):
                self.actions = []

            def get_obs(self):
                return {"joint_action": {"vector": np.zeros(14, dtype=np.float32)},
                        "pointcloud": np.zeros((32, 6), dtype=np.float32)}

            def take_action(self, action, action_type):
                assert action_type == "qpos"
                assert action.shape == (14,)
                assert np.isfinite(action).all()
                self.actions.append(action)

        env = Environment()
        deploy_eval(env, wrapper, env.get_obs())
        self.assertEqual(len(env.actions), 6)
        self.assertEqual(len(wrapper.obs), 3)
        reset_model(wrapper)
        self.assertEqual(len(wrapper.obs), 0)
        # EMA must not change the online model or silently alias its parameters.
        ema = copy.deepcopy(policy).requires_grad_(False)
        policy.requires_grad_(True)
        with torch.no_grad():
            next(policy.parameters()).add_(1)
        update_ema(ema, policy, 0)
        for left, right in zip(ema.parameters(), policy.parameters()):
            torch.testing.assert_close(left, right)


if __name__ == "__main__":
    unittest.main()
