"""Bayesian linear layer reused across Bayesian attention variants."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianLinear(nn.Module):
    """Bayesian linear layer with variational inference for uncertainty quantification.

    Maintains distributions over weights and biases using mean + log variance
    parameterization and performs reparameterized sampling during training.
    """
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.weight_log_var = nn.Parameter(torch.randn(out_features, in_features) * 0.01 - 3)
        self.bias_log_var = nn.Parameter(torch.zeros(out_features) - 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.training:
            weight_std = torch.exp(0.5 * self.weight_log_var)
            bias_std = torch.exp(0.5 * self.bias_log_var)
            weight = self.weight_mu + weight_std * torch.randn_like(self.weight_mu)
            bias = self.bias_mu + bias_std * torch.randn_like(self.bias_mu)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        weight_var = torch.exp(self.weight_log_var)
        bias_var = torch.exp(self.bias_log_var)
        weight_kl = 0.5 * torch.sum(
            (self.weight_mu ** 2 + weight_var) / (self.prior_std ** 2)
            - 1 - self.weight_log_var + 2 * math.log(self.prior_std)
        )
        bias_kl = 0.5 * torch.sum(
            (self.bias_mu ** 2 + bias_var) / (self.prior_std ** 2)
            - 1 - self.bias_log_var + 2 * math.log(self.prior_std)
        )
        return weight_kl + bias_kl
