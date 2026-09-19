"""CPU regression checks for the learned source used at inference."""
import math
import unittest
from unittest.mock import patch

import torch

from leap import LeaPPolicy


class PolicySamplingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.policy = LeaPPolicy(
            shape_meta={
                "obs": {"point_cloud": {"shape": [16, 3]},
                        "agent_pos": {"shape": [4]}},
                "action": {"shape": [2]},
            },
            horizon=8, n_obs_steps=3, n_action_steps=6,
            encoder_output_dim=16, encoder_hidden_dims=(16, 16),
            flow_net_hidden_dim=32, flow_net_num_layers=1,
            flow_net_time_embed_dim=16,
        )
        self.obs = {"point_cloud": torch.randn(2, 3, 16, 3),
                    "agent_pos": torch.randn(2, 3, 4)}
        self.actions = torch.randn(2, 8, 2)
        self.policy.normalizer.fit(dict(self.obs, action=self.actions))
        self.policy.eval()

    def predict_with_source(self, seed):
        torch.manual_seed(seed)
        # Execute the actual Euler sampler while observing its source input.
        with patch.object(self.policy.flow_matcher, "sample",
                          wraps=self.policy.flow_matcher.sample) as sampler:
            result = self.policy.predict_action(self.obs)
        return result, sampler.call_args.kwargs["start"].reshape(2, 8, 2)

    def test_inference_uses_learned_mean_and_standard_deviation(self):
        with torch.no_grad():
            mu, log_var, _ = self.policy._encode_obs(
                self.policy.normalizer.normalize(self.obs))
        torch.manual_seed(123)
        expected = mu[:, None] + (0.5 * log_var).exp()[:, None] * torch.randn(2, 8, 2)
        result, source = self.predict_with_source(123)
        torch.testing.assert_close(source, expected)
        self.assertEqual(result["action"].shape, (2, 6, 2))
        self.assertEqual(result["action_pred"].shape, (2, 8, 2))
        torch.testing.assert_close(result["action"], result["action_pred"][:, 2:8])
        self.assertTrue(torch.isfinite(result["action_pred"]).all())

    def test_eval_remains_stochastic_but_seed_is_reproducible(self):
        first, source1 = self.predict_with_source(123)
        repeat, source2 = self.predict_with_source(123)
        changed, source3 = self.predict_with_source(456)
        torch.testing.assert_close(source1, source2, rtol=0, atol=0)
        torch.testing.assert_close(first["action_pred"], repeat["action_pred"], rtol=0, atol=0)
        self.assertFalse(torch.equal(source1, source3))
        self.assertFalse(torch.equal(first["action_pred"], changed["action_pred"]))

    def test_full_training_window_returns_same_executable_actions(self):
        # Future frames in a dataset window must not shift the action chunk.
        full_obs = {key: torch.cat([value, torch.randn(
            value.shape[0], 5, *value.shape[2:])], dim=1)
            for key, value in self.obs.items()}
        torch.manual_seed(123)
        short = self.policy.predict_action(self.obs)
        torch.manual_seed(123)
        full = self.policy.predict_action(full_obs)
        self.assertEqual(full["action"].shape, (2, 6, 2))
        torch.testing.assert_close(full["action"], short["action"], rtol=0, atol=0)
        torch.testing.assert_close(full["action_pred"], short["action_pred"], rtol=0, atol=0)

    def test_sigma_changes_inference_source(self):
        with torch.no_grad():
            self.policy.prior.head[-1].weight.zero_()
            self.policy.prior.head[-1].bias.zero_()
        _, source1 = self.predict_with_source(123)
        with torch.no_grad():
            self.policy.prior.head[-1].bias[2:] = math.log(4.0)
        _, source2 = self.predict_with_source(123)
        self.assertGreater(source1.abs().max().item(), 0)
        torch.testing.assert_close(source2, 2 * source1)

    def test_training_backward_reaches_both_prior_outputs(self):
        self.policy.train()
        loss, _ = self.policy.compute_loss({"obs": self.obs, "action": self.actions})
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        grad = self.policy.prior.head[-1].weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad[:2].abs().sum().item(), 0)
        self.assertGreater(grad[2:].abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
