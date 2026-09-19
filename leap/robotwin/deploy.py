"""Official RoboTwin policy interface: get_model, eval and reset_model."""
from collections import deque
import numpy as np
import torch
from leap.robotwin.checkpoint import load_policy


def encode_obs(observation):
    return {"agent_pos": np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
            "point_cloud": np.asarray(observation["pointcloud"], dtype=np.float32)}


class Model:
    def __init__(self, policy):
        self.policy = policy
        self.obs = deque(maxlen=policy.n_obs_steps)

    def get_action(self):
        history = list(self.obs)
        history = [history[0]] * (self.policy.n_obs_steps - len(history)) + history
        tensors = {key: torch.as_tensor(np.stack([obs[key] for obs in history]),
                                        device=self.policy.device).unsqueeze(0)
                   for key in history[0]}
        pc = tensors["point_cloud"]
        channels = self.policy.normalizer["point_cloud"].params_dict["scale"].numel()
        if pc.shape[-1] == 3 and channels == 6 and not self.policy.use_pc_color:
            tensors["point_cloud"] = torch.cat((pc, torch.zeros_like(pc)), dim=-1)
        with torch.no_grad():
            return self.policy.predict_action(tensors)["action"][0].cpu().numpy()


def get_model(usr_args):
    return Model(load_policy(usr_args["ckpt_path"], device=usr_args.get("device", "cuda:0")))


def eval(TASK_ENV, model, observation):
    if not model.obs:
        model.obs.append(encode_obs(observation))
    for action in model.get_action():
        TASK_ENV.take_action(action, action_type="qpos")
        model.obs.append(encode_obs(TASK_ENV.get_obs()))


def reset_model(model):
    model.obs.clear()
    model.policy.reset()
