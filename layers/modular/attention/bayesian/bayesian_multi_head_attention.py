"""Separate BayesianMultiHeadAttention implementation."""
from __future__ import annotations
import torch
from ..base import BaseAttention
from .bayesian_attention import BayesianAttention
from utils.logger import logger

class BayesianMultiHeadAttention(BaseAttention):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, prior_std: float = 1.0,
                 n_samples: int = 5, output_attention: bool = False) -> None:
        super().__init__()
        logger.info("Initializing BayesianMultiHeadAttention (split module)")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_samples = n_samples
        self.output_attention = output_attention
        self.bayesian_attention = BayesianAttention(d_model, n_heads, dropout=dropout, prior_std=prior_std, output_attention=True)
        self.output_dim_multiplier = 1

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):  # type: ignore[override]
        if self.training:
            output, attn_weights, uncertainty = self.bayesian_attention(queries, keys, values, attn_mask)
            if self.output_attention:
                return output, attn_weights, uncertainty
            return output, None
        outputs = []
        attns = []
        uncerts = []
        self.bayesian_attention.train()
        with torch.no_grad():
            for _ in range(self.n_samples):
                out, attn_w, uncert = self.bayesian_attention(queries, keys, values, attn_mask)
                outputs.append(out)
                if attn_w is not None:
                    attns.append(attn_w)
                if uncert is not None:
                    uncerts.append(uncert)
        self.bayesian_attention.eval()
        outputs_tensor = torch.stack(outputs, dim=0)
        mean_output = outputs_tensor.mean(dim=0)
        if attns:
            mean_attn = torch.stack(attns, dim=0).mean(dim=0)
        else:
            mean_attn = None
        epistemic_uncertainty = outputs_tensor.var(dim=0)
        aleatoric_uncertainty = torch.stack(uncerts, dim=0).mean(dim=0) if uncerts else None
        if self.output_attention:
            return mean_output, mean_attn, epistemic_uncertainty, aleatoric_uncertainty
        return mean_output, None

# --- Registration ---
from ...core.registry import component_registry, ComponentFamily  # noqa: E402

component_registry.register(
    name="BayesianMultiHeadAttention",
    component_class=BayesianMultiHeadAttention,
    component_type=ComponentFamily.ATTENTION,
    test_config={
        "d_model": 32,
        "n_heads": 4,
        "dropout": 0.1,
        "prior_std": 1.0,
        "n_samples": 3,
        "output_attention": False,
    },
)
