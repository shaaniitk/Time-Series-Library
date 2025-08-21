"""Meta-learning adapter (MAML-style) split from adaptive_components."""
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseAttention
from utils.logger import logger


class MetaLearningAdapter(BaseAttention):
    """MAML-style meta-learning adapter with inner-loop adaptation."""

    def __init__(
        self,
        d_model: int,
        n_heads: int | None = None,
        adaptation_steps: int = 5,
        meta_lr: float = 0.01,
        inner_lr: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        logger.info("Initializing MAML MetaLearningAdapter: adaptation_steps=%s", adaptation_steps)
        self.adaptation_steps = adaptation_steps
        self.d_model = d_model
        self.inner_lr = inner_lr
        self.output_dim_multiplier = 1
        self.fast_weights = nn.ParameterDict(
            {
                "layer1_weight": nn.Parameter(torch.randn(d_model, d_model) * 0.01),
                "layer1_bias": nn.Parameter(torch.zeros(d_model)),
                "layer2_weight": nn.Parameter(torch.randn(d_model, d_model) * 0.01),
                "layer2_bias": nn.Parameter(torch.zeros(d_model)),
                "output_weight": nn.Parameter(torch.randn(d_model, d_model) * 0.01),
                "output_bias": nn.Parameter(torch.zeros(d_model)),
            }
        )
        self.meta_lr_param = nn.Parameter(torch.tensor(meta_lr))
        self.support_encoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model),
        )
        self.task_adapter = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        support_set: torch.Tensor | None = None,
        support_labels: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, None]:
        if support_set is not None and self.training:
            out = self._maml_forward(query, support_set, support_labels)
        else:
            out = self._standard_forward(query)
        return out, None

    def _maml_forward(
        self, query: torch.Tensor, support_set: torch.Tensor, support_labels: torch.Tensor | None
    ) -> torch.Tensor:
        B, L, D = query.shape
        _, S, D_s = support_set.shape
        assert D == D_s, "Support set feature dimension mismatch"
        support_context = self.support_encoder(support_set.mean(dim=1))
        adapted_params: dict[str, torch.Tensor] = {k: v.clone() for k, v in self.fast_weights.items()}
        for _ in range(self.adaptation_steps):
            support_pred = self._forward_with_params(support_set, adapted_params)
            loss = (
                F.mse_loss(support_pred, support_labels)
                if support_labels is not None
                else F.mse_loss(support_pred, support_set)
            )
            grads = torch.autograd.grad(loss, adapted_params.values(), create_graph=True, retain_graph=True, allow_unused=True)
            for (name, param), grad in zip(adapted_params.items(), grads):
                if grad is not None:
                    adapted_params[name] = param - self.inner_lr * grad
        adapted_query = self._forward_with_params(query, adapted_params)
        task_adaptation = self.task_adapter(support_context).unsqueeze(1)
        return self.dropout(adapted_query + task_adaptation)

    def _forward_with_params(self, x: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
        h1 = F.relu(F.linear(x, params["layer1_weight"], params["layer1_bias"]))
        h2 = F.relu(F.linear(h1, params["layer2_weight"], params["layer2_bias"]))
        return F.linear(h2, params["output_weight"], params["output_bias"])

    def _standard_forward(self, query: torch.Tensor) -> torch.Tensor:
        return self._forward_with_params(query, self.fast_weights)

    def extract_task_context(self, support_set: torch.Tensor) -> torch.Tensor:
        support_mean = torch.mean(support_set, dim=1)
        return self.support_encoder(support_mean)


__all__ = ["MetaLearningAdapter"]
