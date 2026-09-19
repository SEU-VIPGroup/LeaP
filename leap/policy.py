"""LeaP — Learnable source Prior for generative robot policies.

A generative visuomotor policy that replaces the uninformative N(0, I) source
of flow matching with a *learned*, state-conditioned diagonal Gaussian prior:

    (mu, log sigma^2) = LeaPPrior(state)              # per action dim
    x0 = mu + sigma . eps,   eps ~ N(0, I)            # reparameterized source
    x1 = FlowMatch(v_theta, start=x0, cond=z_obs)     # linear-OT flow to action

The prior mu/sigma are produced from the proprioceptive state only; the point
cloud is used to condition the flow velocity network (z_obs). Training combines
three terms:

    L = L_flow  +  w_nll . L_nll  +  w_align . L_align

  * L_flow : conditional flow-matching MSE on the linear OT path
  * L_nll  : Gaussian NLL of the ground-truth action chunk under (mu, sigma),
             which is what makes the prior *informative* (pulls x0 near x1)
  * L_align: symmetric InfoNCE between the source sample and the action chunk,
             a CLIP-style alignment that sharpens the state->action coupling

This single file collapses the original 4-level research inheritance into one
self-contained class; behavior is identical to the paper's main model.
"""
from typing import Dict, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint

from .base_policy import BasePolicy
from .normalizer import LinearNormalizer
from .pointnet import PointNetEncoderXYZ, PointNetEncoderXYZRGB, create_mlp
from .flow_net import SimpleFlowNet
from .flow_matcher import SimpleConditionalFlowMatcher
from .model_util import print_params


def _make_mlp(in_dim: int, hidden_dims: Tuple[int, ...], out_dim: int) -> nn.Sequential:
    if len(hidden_dims) == 0:
        raise RuntimeError("MLP hidden_dims is empty")
    net_arch = [] if len(hidden_dims) == 1 else list(hidden_dims[:-1])
    output_dim = hidden_dims[-1]
    layers = create_mlp(in_dim, output_dim, net_arch, nn.ReLU)
    if output_dim != out_dim:
        layers.append(nn.Linear(output_dim, out_dim))
    return nn.Sequential(*layers)


class LeaPPrior(nn.Module):
    """Learnable diagonal-Gaussian source prior conditioned on robot state.

    state (B, To, state_dim) -> state encoder -> MLP head -> (mu, log_var),
    each of size action_dim. A single (mu, sigma) is broadcast across the H
    prediction steps (shared-across-chunk prior).
    """

    def __init__(self, state_dim, action_dim, n_obs_steps,
                 encoder_output_dim=128, encoder_hidden_dims=(128, 128)):
        super().__init__()
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.state_encoder = _make_mlp(state_dim, encoder_hidden_dims, encoder_output_dim)
        dist_feature_dim = encoder_output_dim * n_obs_steps
        self.head = nn.Sequential(
            nn.LayerNorm(dist_feature_dim),
            nn.Linear(dist_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, action_dim * 2),     # -> (mu, log_var)
        )

    def forward(self, state):
        """state: (B, To, state_dim) -> mu, log_var each (B, action_dim)."""
        B, To = state.shape[:2]
        z_s = self.state_encoder(state.reshape(-1, state.shape[-1])).reshape(B, To, -1)
        dist_params = self.head(z_s.reshape(B, -1))
        mu, log_var = dist_params.chunk(2, dim=-1)
        log_var = torch.clamp(log_var, min=-10.0, max=5.0)
        return mu, log_var, z_s

    def sample(self, mu, log_var, horizon):
        """Reparameterized source x0 = mu + sigma . eps, broadcast over H."""
        eps = torch.randn((mu.shape[0], horizon, mu.shape[-1]), device=mu.device, dtype=mu.dtype)
        return mu.unsqueeze(1) + torch.exp(0.5 * log_var).unsqueeze(1) * eps


class LeaPPolicy(BasePolicy):
    def __init__(self,
                 shape_meta: dict,
                 horizon: int,
                 n_action_steps: int,
                 n_obs_steps: int,
                 encoder_output_dim: int = 128,
                 encoder_hidden_dims=(128, 128),
                 use_pc_color: bool = False,
                 pointcloud_encoder_cfg=None,
                 # ---- losses ----
                 nll_loss_weight: float = 1.0,
                 align_loss_weight: float = 0.3,
                 align_temperature: float = 0.1,
                 # ---- flow matcher / net ----
                 num_sampling_steps: int = 6,
                 flow_matcher_sigma: float = 0.0,
                 flow_net_hidden_dim: int = 512,
                 flow_net_num_layers: int = 4,
                 flow_net_mlp_ratio: float = 4.0,
                 flow_net_dropout: float = 0.0,
                 flow_net_time_embed_dim: int = 256,
                 **kwargs):
        super().__init__()

        action_shape = shape_meta['action']['shape']
        action_dim = action_shape[0] if len(action_shape) == 1 else action_shape[0] * action_shape[1]
        obs_meta = shape_meta['obs']
        pc_in_dim = obs_meta['point_cloud']['shape'][-1]
        state_dim = obs_meta['agent_pos']['shape'][-1]
        if not use_pc_color:
            pc_in_dim = min(pc_in_dim, 3)

        # ---- point-cloud encoder (DP3 PointNet) ----
        if pointcloud_encoder_cfg is None:
            pointcloud_encoder_cfg = dict(
                in_channels=pc_in_dim, out_channels=encoder_output_dim,
                use_layernorm=True, final_norm="layernorm")
        else:
            pointcloud_encoder_cfg = dict(pointcloud_encoder_cfg)
            pointcloud_encoder_cfg["in_channels"] = pc_in_dim
            pointcloud_encoder_cfg["out_channels"] = encoder_output_dim
        self.pc_encoder = (PointNetEncoderXYZRGB if use_pc_color else PointNetEncoderXYZ)(**pointcloud_encoder_cfg)

        # ---- LeaP learnable source prior (state-only) ----
        self.prior = LeaPPrior(state_dim, action_dim, n_obs_steps,
                               encoder_output_dim, encoder_hidden_dims)

        # ---- alignment projection heads (InfoNCE) ----
        self.align_action_proj = _make_mlp(horizon * action_dim, encoder_hidden_dims, encoder_output_dim)
        self.align_obs_proj = _make_mlp(horizon * action_dim, encoder_hidden_dims, encoder_output_dim)

        # ---- flow velocity net + matcher (operate on flattened H*A chunk) ----
        flow_dim = horizon * action_dim
        # obs condition = cat(z_pc, z_s) flattened over To
        self.obs_feature_dim = (encoder_output_dim * 2) * n_obs_steps
        self.flow_net = SimpleFlowNet(
            input_dim=flow_dim, hidden_dim=flow_net_hidden_dim, output_dim=flow_dim,
            num_layers=flow_net_num_layers, mlp_ratio=flow_net_mlp_ratio,
            dropout=flow_net_dropout, time_embed_dim=flow_net_time_embed_dim,
            condition_dim=self.obs_feature_dim)
        self.flow_matcher = SimpleConditionalFlowMatcher(
            num_sampling_steps=num_sampling_steps, sigma=flow_matcher_sigma)

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.encoder_output_dim = encoder_output_dim
        self.use_pc_color = use_pc_color
        self.num_sampling_steps = num_sampling_steps
        self.nll_loss_weight = nll_loss_weight
        self.align_loss_weight = align_loss_weight
        self.align_temperature = align_temperature

        cprint("[LeaP] learnable source prior (state -> mu/sigma) + linear-OT flow matcher", "green")
        cprint(f"[LeaP] w_nll={nll_loss_weight}  w_align={align_loss_weight}  tau={align_temperature}  "
               f"NFE={num_sampling_steps}", "green")
        print_params(self)

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    # ------------------------------------------------------------------
    def _encode_obs(self, nobs):
        """Returns (mu, log_var, z_obs) where z_obs is the flow condition."""
        pc = nobs["point_cloud"]
        if not self.use_pc_color:
            pc = pc[..., :3]
        state = nobs["agent_pos"]
        pc = pc[:, :self.n_obs_steps]
        state = state[:, :self.n_obs_steps]
        B, To, npts, cdim = pc.shape

        pc_feat = self.pc_encoder(pc.reshape(B * To, npts, cdim)).reshape(B, To, -1)
        mu, log_var, z_s = self.prior(state)                       # prior from state only
        z_obs = torch.cat([pc_feat, z_s], dim=-1).reshape(B, -1)   # flow condition uses pc + state
        return mu, log_var, z_obs

    def _flow_velocity(self, x_flat, t, global_cond):
        return self.flow_net(x_flat, t, global_cond=global_cond)

    # ---- losses ----
    def _gaussian_nll(self, actions, mu, log_var):
        mu = mu.unsqueeze(1)
        log_var = log_var.unsqueeze(1)
        var = torch.exp(log_var)
        nll = 0.5 * (log_var + (actions - mu) ** 2 / var + math.log(2 * math.pi))
        return nll.mean()

    def _contrastive(self, x0, actions):
        z = F.normalize(self.align_obs_proj(x0.reshape(x0.shape[0], -1)), dim=-1)
        a = F.normalize(self.align_action_proj(actions.reshape(actions.shape[0], -1)), dim=-1)
        logits = torch.matmul(z, a.transpose(0, 1)) / self.align_temperature
        labels = torch.arange(z.shape[0], device=z.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))

    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        if nactions.ndim == 4:
            nactions = nactions.reshape(nactions.shape[0], nactions.shape[1], -1)
        B, T, A = nactions.shape

        mu, log_var, z_obs = self._encode_obs(nobs)
        x0 = self.prior.sample(mu, log_var, T)                    # learned source (B, T, A)

        loss_flow, _ = self.flow_matcher.compute_loss(
            self._flow_velocity, target=nactions.reshape(B, T * A),
            start=x0.reshape(B, T * A), global_cond=z_obs)
        loss_nll = self._gaussian_nll(nactions, mu, log_var)
        loss_align = self._contrastive(x0, nactions)

        loss = loss_flow + self.nll_loss_weight * loss_nll + self.align_loss_weight * loss_align
        return loss, {"flow_loss": loss_flow.item(),
                      "nll_loss": loss_nll.item(),
                      "align_loss": loss_align.item()}

    # ---- inference (learned Gaussian source: x0 = mu + sigma * eps) ----
    @torch.no_grad()
    def predict_action(self, obs_dict):
        """Generate actions from a sampled learned prior, including in eval mode.

        Call ``eval()`` before inference to disable dropout. The source remains
        stochastic; seed PyTorch externally for reproducible evaluations.
        Only the first ``n_obs_steps`` frames condition the prediction, even
        when a complete training window is supplied.
        """
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        T, A = self.horizon, self.action_dim

        mu, log_var, z_obs = self._encode_obs(nobs)
        x0 = self.prior.sample(mu, log_var, T)                  # same source distribution as training
        x = self.flow_matcher.sample(
            self._flow_velocity, shape=(B, T * A), device=x0.device,
            num_steps=self.num_sampling_steps, start=x0.reshape(B, T * A), global_cond=z_obs)

        naction_pred = x.reshape(B, T, A)
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        start = self.n_obs_steps - 1
        action = action_pred[:, start:start + self.n_action_steps]
        return {"action": action, "action_pred": action_pred}
