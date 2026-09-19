"""Compare release inference, training losses and gradients with a research checkout."""
import argparse
import copy
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def rename(key):
    if key.startswith("state_encoder."):
        return "prior." + key
    if key.startswith("obs_to_dist."):
        return key.replace("obs_to_dist.", "prior.head.", 1)
    return key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", type=Path, required=True, help="Research FlowPolicy-3D directory")
    parser.add_argument("--data", type=Path, help="Optional real RoboTwin Zarr for dataset comparison")
    args = parser.parse_args()
    sys.path.insert(0, str(args.framework.resolve()))
    from flow_policy_3d.policy.structureflowpolicy_fixedattn_film_state_a2afm import (
        StructureFlowPolicyFixedAttnFiLMStateA2AFM as Original)
    from leap.policy import LeaPPolicy

    torch.set_num_threads(1)
    torch.manual_seed(10)
    config = dict(shape_meta={"obs": {"point_cloud": {"shape": [32, 6]},
                                     "agent_pos": {"shape": [14]}}, "action": {"shape": [14]}},
                  horizon=8, n_obs_steps=3, n_action_steps=6)
    # The original constructs then deletes a U-Net. Reduce that unused allocation.
    original = Original(**config, down_dims=(32, 64), diffusion_step_embed_dim=16)
    release = LeaPPolicy(**config)
    batch = {"obs": {"point_cloud": torch.randn(2, 8, 32, 6),
                     "agent_pos": torch.randn(2, 8, 14)}, "action": torch.randn(2, 8, 14)}
    original.normalizer.fit(dict(batch["obs"], action=batch["action"]))
    # The unused parent mask generator carries only an empty device marker.
    state = original.state_dict()
    marker = state.pop("mask_generator._dummy_variable")
    assert marker.numel() == 0
    release.load_state_dict({rename(k): v for k, v in state.items()}, strict=True)

    for training in (False, True):
        original.train(training)
        release.train(training)
        original.zero_grad(set_to_none=True)
        release.zero_grad(set_to_none=True)
        torch.manual_seed(123)
        left, left_metrics = original.compute_loss(batch)
        left.backward()
        torch.manual_seed(123)
        right, right_metrics = release.compute_loss(batch)
        right.backward()
        torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-6)
        for key in ("flow_loss", "nll_loss", "align_loss"):
            assert abs(left_metrics[key] - right_metrics[key]) < 1e-6, key
        targets = dict(release.named_parameters())
        for key, value in original.named_parameters():
            if value.grad is not None:
                torch.testing.assert_close(value.grad, targets[rename(key)].grad, rtol=1e-5, atol=1e-6)
        print(f"PASS losses and parameter gradients (training={training})")
    # Check an optimizer update and EMA warmup against the actual research helper.
    from flow_policy_3d.model.flow.ema_model import EMAModel
    from leap.robotwin.train import update_ema
    left_ema, right_ema = copy.deepcopy(original), copy.deepcopy(release).requires_grad_(False)
    reference_ema = EMAModel(left_ema, power=0.75)
    left_opt = torch.optim.AdamW(original.parameters(), lr=1e-4, betas=(0.95, 0.999), weight_decay=1e-6)
    right_opt = torch.optim.AdamW(release.parameters(), lr=1e-4, betas=(0.95, 0.999), weight_decay=1e-6)
    for step in range(4):
        left_opt.step()
        right_opt.step()
        reference_ema.step(original)
        update_ema(right_ema, release, step)
        for key, value in original.state_dict().items():
            if key != "mask_generator._dummy_variable":
                torch.testing.assert_close(value, release.state_dict()[rename(key)], rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(left_ema.state_dict()[key], right_ema.state_dict()[rename(key)],
                                           rtol=1e-5, atol=1e-6)
    print("PASS AdamW updates and EMA warmup")
    original.eval()
    release.eval()
    obs = {key: value[:, :3] for key, value in batch["obs"].items()}
    torch.manual_seed(456)
    with torch.no_grad():
        left = original.predict_action(obs)
        torch.manual_seed(456)
        right = release.predict_action(obs)
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=1e-6, atol=1e-6)
    print("PASS stochastic inference with identical weights and RNG state")
    if args.data:
        from leap.robotwin.data import RobotDataset
        from flow_policy_3d.dataset.robot_dataset import RobotDataset as ResearchDataset
        a = RobotDataset(args.data, val_ratio=0.02, seed=0)
        b = ResearchDataset(str(args.data.resolve()), horizon=8, pad_before=2, pad_after=5,
                            val_ratio=0.02, seed=0)
        assert len(a) == len(b)
        assert len(a.get_validation_dataset()) == len(b.get_validation_dataset())
        for i in sorted(set([0, 1, len(a) // 2, len(a) - 1])):
            for key in ("agent_pos", "point_cloud"):
                torch.testing.assert_close(a[i]["obs"][key], b[i]["obs"][key], rtol=0, atol=0)
            torch.testing.assert_close(a[i]["action"], b[i]["action"], rtol=0, atol=0)
        an, bn = a.get_normalizer().state_dict(), b.get_normalizer().state_dict()
        assert an.keys() == bn.keys()
        for key in an:
            torch.testing.assert_close(an[key], bn[key], rtol=1e-6, atol=1e-6)
        print("PASS real-data window alignment and normalizer parity")


if __name__ == "__main__":
    main()
