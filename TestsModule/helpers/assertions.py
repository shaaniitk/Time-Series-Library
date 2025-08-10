"""Reusable assertion helpers for TestsModule."""
from __future__ import annotations
import torch

def assert_tensor_close(a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-4, atol: float = 1e-5, msg: str | None = None) -> None:
    if a.shape != b.shape:
        raise AssertionError(f"Shape mismatch {a.shape} != {b.shape}: {msg}")
    if not torch.allclose(a, b, rtol=rtol, atol=atol):
        diff = (a - b).abs().max().item()
        raise AssertionError(f"Tensors not close (max diff {diff}, rtol={rtol}, atol={atol}): {msg}")
