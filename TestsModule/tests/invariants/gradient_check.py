"""Finite difference gradient check utility (selective use).

Keep shapes tiny; intended for diagnosing silent autograd issues.
"""
from __future__ import annotations

from typing import Callable, Tuple
import torch

Tensor = torch.Tensor


def finite_difference_check(
    module: torch.nn.Module,
    input: Tensor,
    proj: Callable[[Tensor], Tensor] | None = None,
    eps: float = 1e-3,
    atol: float = 1e-2,
) -> Tuple[float, float]:
    """Return (relative_diff, atol) for a scalar projection gradient.

    Projection defaults to sum of outputs.
    """
    module.eval()
    input = input.clone().detach().requires_grad_(True)
    out = module(input)
    if proj is None:
        scalar = out.sum()
    else:
        scalar = proj(out)
    scalar.backward()
    analytic = input.grad.detach().clone()
    fd = torch.zeros_like(analytic)
    # Manual multi-index iteration (tensor flattened)
    flat_fd = fd.view(-1)
    base_input = input.detach()
    for i in range(base_input.numel()):  # pragma: no cover
        orig = base_input.view(-1)[i].item()
        plus_inp = base_input.clone()
        plus_inp.view(-1)[i] = orig + eps
        minus_inp = base_input.clone()
        minus_inp.view(-1)[i] = orig - eps
        plus_out = module(plus_inp)
        plus = plus_out.sum() if proj is None else proj(plus_out)
        minus_out = module(minus_inp)
        minus = minus_out.sum() if proj is None else proj(minus_out)
        flat_fd[i] = (plus - minus) / (2 * eps)
    rel_diff = torch.linalg.norm(analytic - fd) / (torch.linalg.norm(analytic) + 1e-12)
    return float(rel_diff), float(atol)


__all__ = ["finite_difference_check"]
