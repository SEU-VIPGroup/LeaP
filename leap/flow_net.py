# Adapted from A2A / RoboVerse via the research port; Apache-2.0.
# Modified for the standalone LeaP package; see THIRD_PARTY_NOTICES.md.
"""Velocity network for LeaP's flow matcher.

A DiT-style modulated MLP that predicts the conditional flow velocity
v_theta(x_t, t, c) over the flattened action chunk. Time and observation
condition are injected through affine modulation. To match the research
implementation, the enclosing network applies Xavier initialization to all
linear layers, including the modulation layers.
"""
import torch
import torch.nn as nn

from .pos_embed import SinusoidalPosEmb


class _Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class _FlowNetLayer(nn.Module):
    """Modulated residual MLP block; the parent network sets final initialization."""

    def __init__(self, dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()

        def approx_gelu():
            return nn.GELU(approximate="tanh")

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = _Mlp(in_features=dim,
                        hidden_features=int(dim * mlp_ratio),
                        act_layer=approx_gelu,
                        drop=dropout)
        # time/condition modulation → (gate, scale, shift)
        self.time_modulator = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim))
        self.dim = dim
        self._init_weights()

    def _init_weights(self):
        # Local block default; SimpleFlowNet._init_weights overwrites this with
        # Xavier weights to preserve the original research initialization.
        nn.init.constant_(self.time_modulator[-1].weight, 0)
        nn.init.constant_(self.time_modulator[-1].bias, 0)

    def forward(self, x, t):
        B = x.shape[0]
        gamma, scale, shift = self.time_modulator(t).view(B, 3, self.dim).unbind(1)
        x_norm = self.norm(x).mul(scale.add(1)).add_(shift)
        x = x + self.mlp(x_norm).mul_(gamma)
        return x


class SimpleFlowNet(nn.Module):
    """Flow velocity network: (x_flat, t, global_cond) -> v_flat.

    Args:
        input_dim:     flattened action-chunk dim (H * A)
        hidden_dim:    transformer width
        output_dim:    == input_dim
        num_layers:    number of adaLN-Zero blocks
        time_embed_dim: sinusoidal time-embedding dim
        condition_dim: observation condition dim (None disables conditioning)
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 mlp_ratio=4.0, dropout=0.0, time_embed_dim=256,
                 condition_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(time_embed_dim * 4, hidden_dim),
        )
        self.cond_embed = nn.Linear(condition_dim, hidden_dim) if condition_dim is not None else None

        self.layers = nn.ModuleList([
            _FlowNetLayer(dim=hidden_dim, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        self._init_weights()

    def _init_weights(self):
        def _basic(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic)
        nn.init.normal_(self.time_embed[1].weight, std=0.02)
        nn.init.normal_(self.time_embed[3].weight, std=0.02)

    def forward(self, x, t, global_cond=None):
        x = self.input_proj(x)
        t = self.time_embed(t)
        if global_cond is not None and self.cond_embed is not None:
            t = t + self.cond_embed(global_cond)
        for blk in self.layers:
            x = blk(x, t)
        x = self.norm(x)
        return self.out_proj(x)
