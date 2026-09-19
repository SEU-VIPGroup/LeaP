"""RoboTwin joint-position demos and boundary-padded action windows."""
from pathlib import Path
import copy
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import zarr
from leap.normalizer import LinearNormalizer


def convert_episodes(input_dir, output, episodes):
    """Align observation[t] with joint_position[t+1], as RoboTwin DP3 does."""
    if episodes < 1:
        raise ValueError("episodes must be positive")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {output}")
    arrays = {key: [] for key in ("point_cloud", "state", "action")}
    ends = []
    for i in range(episodes):
        path = Path(input_dir) / f"episode{i}.hdf5"
        with h5py.File(path, "r") as episode:
            state = np.asarray(episode["joint_action/vector"], dtype=np.float32)
            pc = np.asarray(episode["pointcloud"], dtype=np.float32)
        if state.ndim != 2 or pc.ndim != 3 or len(state) != len(pc) or len(state) < 2:
            raise ValueError(f"Invalid episode shapes in {path}: {state.shape}, {pc.shape}")
        if not np.isfinite(state).all() or not np.isfinite(pc).all():
            raise ValueError(f"Non-finite data in {path}")
        arrays["state"].append(state[:-1])
        arrays["action"].append(state[1:])
        arrays["point_cloud"].append(pc[:-1])
        ends.append((ends[-1] if ends else 0) + len(state) - 1)
    arrays = {key: np.concatenate(value) for key, value in arrays.items()}
    root = zarr.open_group(str(output), mode="w-")
    data = root.create_group("data")
    for key, value in arrays.items():
        data.create_dataset(key, data=value, chunks=(min(100, len(value)), *value.shape[1:]),
                            compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=1))
    root.create_group("meta").create_dataset("episode_ends", data=np.asarray(ends, dtype=np.int64))
    return {"episodes": episodes, "frames": ends[-1], "output": str(output)}


class RobotDataset(Dataset):
    """Matches the research SequenceSampler's edge padding without its framework."""
    def __init__(self, path, horizon=8, n_obs_steps=3, n_action_steps=6, episodes=None,
                 val_ratio=0.0, seed=0):
        if not (horizon >= 1 and 1 <= n_obs_steps <= horizon
                and 1 <= n_action_steps <= horizon - n_obs_steps + 1):
            raise ValueError("Invalid horizon, observation steps or action steps")
        root = zarr.open_group(str(path), mode="r")
        ends = np.asarray(root["meta/episode_ends"])
        if episodes is not None:
            if not 1 <= episodes <= len(ends):
                raise ValueError("episodes exceeds available episodes or is non-positive")
            ends = ends[:episodes]
        if not len(ends) or ends[0] <= 0 or np.any(np.diff(ends) <= 0):
            raise ValueError("episode_ends must be positive and strictly increasing")
        self.arrays = {key: np.asarray(root[f"data/{key}"][:int(ends[-1])], dtype=np.float32)
                       for key in ("state", "action", "point_cloud")}
        if any(len(value) != ends[-1] for value in self.arrays.values()):
            raise ValueError("Dataset lengths do not match episode_ends")
        self.horizon, self.ends = horizon, ends
        if not 0 <= val_ratio < 1:
            raise ValueError("val_ratio must be in [0, 1)")
        val_mask = np.zeros(len(ends), dtype=bool)
        if val_ratio > 0:
            count = min(max(1, round(len(ends) * val_ratio)), len(ends) - 1)
            val_mask[np.random.default_rng(seed).choice(len(ends), size=count, replace=False)] = True
        self.indices = []
        self.val_indices = []
        begin = 0
        for episode, end in enumerate(ends):
            for offset in range(-(n_obs_steps - 1), int(end) - begin - horizon + n_action_steps):
                target = self.val_indices if val_mask[episode] else self.indices
                target.append((begin, int(end), offset))
            begin = int(end)
        if not self.indices:
            raise ValueError("No training windows; episodes are too short")

    def __len__(self):
        return len(self.indices)

    def get_validation_dataset(self):
        dataset = copy.copy(self)
        dataset.indices = self.val_indices
        return dataset

    def __getitem__(self, index):
        begin, end, offset = self.indices[index]
        rows = np.clip(begin + offset + np.arange(self.horizon), begin, end - 1)
        sample = {key: torch.from_numpy(value[rows]) for key, value in self.arrays.items()}
        return {"obs": {"agent_pos": sample["state"], "point_cloud": sample["point_cloud"]},
                "action": sample["action"]}

    def get_normalizer(self):
        normalizer = LinearNormalizer()
        normalizer.fit({"agent_pos": self.arrays["state"], "action": self.arrays["action"],
                        "point_cloud": self.arrays["point_cloud"]}, last_n_dims=1, mode="limits")
        return normalizer

    @property
    def shape_meta(self):
        return {"obs": {"agent_pos": {"shape": list(self.arrays["state"].shape[1:])},
                        "point_cloud": {"shape": list(self.arrays["point_cloud"].shape[1:])}},
                "action": {"shape": list(self.arrays["action"].shape[1:])}}
