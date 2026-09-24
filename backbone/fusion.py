import math
import torch
import torch.nn as nn


class AdaptiveGranularityFusion(nn.Module):
    """Residual adaptive granularity-gated concatenation.

    Given three granularity representations:
        h_atomic, h_molecular, h_material

    It learns instance-wise residual weights:
        w_i = 1 + rho * tanh(gate(LayerNorm([h_atomic || h_molecular || h_material])))

    and returns:
        h_i = [w_i^a * h_i^a || w_i^b * h_i^b || w_i^g * h_i^g]

    The final layer is zero-initialized so that the module is exactly
    equivalent to fixed concat at initialization.
    """

    def __init__(
        self,
        hidden_dim,
        gate_hidden=None,
        dropout=0.0,
        gate_scale=0.5,
        use_layernorm=True,
    ):
        super().__init__()
        gate_hidden = gate_hidden or hidden_dim
        self.gate_scale = gate_scale
        self.use_layernorm = use_layernorm

        if use_layernorm:
            self.norm = nn.LayerNorm(hidden_dim * 3)
        else:
            self.norm = nn.Identity()

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 3),
        )

        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, h_atomic, h_molecular, h_material, return_weights=False):
        gate_input = torch.cat([h_atomic, h_molecular, h_material], dim=-1)
        gate_input = self.norm(gate_input)

        weights = 1.0 + self.gate_scale * torch.tanh(self.gate(gate_input))

        fused = torch.cat(
            [
                weights[:, 0:1] * h_atomic,
                weights[:, 1:2] * h_molecular,
                weights[:, 2:3] * h_material,
            ],
            dim=-1,
        )

        if return_weights:
            return fused, weights
        return fused


class GlobalLearnableFusion(nn.Module):
    """Dataset-level learnable residual weights — ablation baseline.

    Unlike AdaptiveGranularityFusion, this learns a single set of three
    weights shared across ALL samples, proving that instance-wise
    adaptation matters beyond dataset-level scaling.

    Also zero-initialized so that at the start weights ≈ 1 (concat).
    """

    def __init__(self, hidden_dim, gate_scale=0.5):
        super().__init__()
        self.gate_scale = gate_scale

        self.raw_weights = nn.Parameter(torch.zeros(3))

    def forward(self, h_atomic, h_molecular, h_material, return_weights=False):
        weights_single = 1.0 + self.gate_scale * torch.tanh(self.raw_weights)
        weights = weights_single.unsqueeze(0).expand(h_atomic.shape[0], -1)

        fused = torch.cat(
            [
                weights[:, 0:1] * h_atomic,
                weights[:, 1:2] * h_molecular,
                weights[:, 2:3] * h_material,
            ],
            dim=-1,
        )

        if return_weights:
            return fused, weights
        return fused


class HybridGranularityFusion(nn.Module):
    """Hybrid residual granularity-gated concatenation.

    This module interpolates between:
      1) global task-level granularity calibration;
      2) instance-wise granularity gating.

    raw_i = eta * raw_global + (1 - eta) * raw_instance_i
    w_i   = 1 + rho * tanh(raw_i)
    h_i   = [w_i^a h_i^a || w_i^b h_i^b || w_i^g h_i^g]

    When eta = 1, it degenerates to global learnable fusion.
    When eta = 0, it degenerates to instance-wise adaptive fusion.
    With zero initialization, the module starts exactly from fixed concat.
    """

    def __init__(
        self,
        hidden_dim,
        gate_hidden=None,
        dropout=0.0,
        gate_scale=0.5,
        use_layernorm=True,
        hybrid_eta=0.5,
        learnable_eta=True,
        eta_init=0.5,
    ):
        super().__init__()
        gate_hidden = gate_hidden or hidden_dim
        self.gate_scale = gate_scale
        self.learnable_eta = learnable_eta

        if use_layernorm:
            self.norm = nn.LayerNorm(hidden_dim * 3)
        else:
            self.norm = nn.Identity()

        self.instance_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 3),
        )

        self.global_raw = nn.Parameter(torch.zeros(3))

        nn.init.zeros_(self.instance_gate[-1].weight)
        nn.init.zeros_(self.instance_gate[-1].bias)

        if learnable_eta:
            eta_init = float(min(max(eta_init, 1e-4), 1.0 - 1e-4))
            eta_logit = math.log(eta_init / (1.0 - eta_init))
            self.eta_logit = nn.Parameter(torch.tensor(eta_logit, dtype=torch.float32))
        else:
            self.register_buffer(
                "fixed_eta", torch.tensor(float(hybrid_eta), dtype=torch.float32)
            )

    def get_eta(self):
        if self.learnable_eta:
            return torch.sigmoid(self.eta_logit)
        return self.fixed_eta

    def forward(self, h_atomic, h_molecular, h_material, return_weights=False):
        gate_input = torch.cat([h_atomic, h_molecular, h_material], dim=-1)
        gate_input = self.norm(gate_input)

        raw_instance = self.instance_gate(gate_input)

        raw_global = self.global_raw.unsqueeze(0).expand_as(raw_instance)

        eta = self.get_eta().to(raw_instance.device)
        raw_hybrid = eta * raw_global + (1.0 - eta) * raw_instance

        weights = 1.0 + self.gate_scale * torch.tanh(raw_hybrid)

        fused = torch.cat(
            [
                weights[:, 0:1] * h_atomic,
                weights[:, 1:2] * h_molecular,
                weights[:, 2:3] * h_material,
            ],
            dim=-1,
        )

        if return_weights:
            return fused, weights
        return fused
