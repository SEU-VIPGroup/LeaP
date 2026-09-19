# Adapted through the research A2A port; see THIRD_PARTY_NOTICES.md.
# Modified to accept LeaP's sampled source without benchmark dependencies.
"""Conditional flow matcher with a learnable source prior.

Standard conditional flow matching (Lipman et al., 2023; Tong et al., 2024)
draws the source x0 ~ N(0, I). LeaP instead lets x0 come from a *learned*
state-conditioned Gaussian prior (see leap.policy.LeaPPrior); this module only
implements the linear-interpolation OT path and Euler sampler, and is agnostic
to where x0 comes from — the caller passes `start=x0`.

Linear OT path (sigma=0):
    x_t = (1 - t) * x0 + t * x1,    u_t = x1 - x0
Training target is the constant velocity u_t; inference integrates v_theta with
forward Euler from t=0 (x = x0) to t=1.
"""
import torch


class SimpleConditionalFlowMatcher:
    def __init__(self, num_sampling_steps=6, sigma=0.0):
        self.num_sampling_steps = num_sampling_steps
        self.sigma = sigma

    def sample_location_and_conditional_flow(self, x0, x1):
        t = torch.rand(x0.shape[0], device=x0.device, dtype=x0.dtype)
        t_b = t.view(-1, *([1] * (x0.dim() - 1)))
        xt = (1.0 - t_b) * x0 + t_b * x1
        if self.sigma > 0:
            xt = xt + self.sigma * torch.randn_like(xt)
        ut = x1 - x0
        return t, xt, ut

    def compute_loss(self, model, target, start=None, **kwargs):
        """Flow-matching MSE loss. `start` = learned source sample x0."""
        x0 = torch.randn_like(target) if start is None else start
        t, xt, ut = self.sample_location_and_conditional_flow(x0, target)
        vt = model(xt, t, **kwargs)
        loss = torch.mean((vt - ut) ** 2)
        return loss, {"loss": loss.item()}

    def sample(self, model, shape, device, num_steps=None,
               return_traces=False, start=None, **kwargs):
        """Forward-Euler integration from x0 (=start) to t=1."""
        if num_steps is None:
            num_steps = self.num_sampling_steps
        x = torch.randn(shape, device=device) if start is None else start
        dt = 1.0 / num_steps

        traj_history = [x] if return_traces else None
        vel_history = [torch.zeros_like(x)] if return_traces else None

        for step in range(num_steps):
            t = torch.full((x.shape[0],), step / num_steps, device=x.device, dtype=x.dtype)
            vt = model(x, t, **kwargs)
            x = x + vt * dt
            if return_traces:
                traj_history.append(x.detach().clone().cpu())
                vel_history.append(vt.detach().clone().cpu())

        if return_traces:
            return x, (traj_history, vel_history)
        return x
