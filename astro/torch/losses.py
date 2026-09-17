"""Soft astrology training terms, applied in the experiment layer only.

Hard astronomical constraints (circular encodings, ayanamsha-rotation
invariance of relative rules, bounded ranges) are enforced by construction in
the operators and compiler.  What remains here is deliberately soft:

- a prior penalty that pulls each rule gate toward its declared importance and
  applies declared L1 sparsity; data can always overcome it;
- a response-regularity penalty that discourages sharp forecast changes under
  small perturbations of astrology inputs, so the model cannot memorize dates
  through high-frequency channels.

Neither term encodes a claim about how markets respond to the sky.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch

from astro.compile.compiler import ChannelMeta


def rule_prior_tensors(
    channel_meta: Sequence[ChannelMeta], rule_ids: Sequence[str], device, dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    by_rule: dict[str, ChannelMeta] = {}
    for meta in channel_meta:
        by_rule.setdefault(meta.rule_id, meta)
    missing = [rule_id for rule_id in rule_ids if rule_id not in by_rule]
    if missing:
        raise ValueError(f"No prior metadata for gated rules {missing}.")
    ordered = [by_rule[rule_id] for rule_id in rule_ids]

    def tensor(values):
        return torch.tensor(values, device=device, dtype=dtype)

    return (
        tensor([meta.prior_importance for meta in ordered]),
        tensor([meta.l2_anchor for meta in ordered]),
        tensor([meta.l1_weight for meta in ordered]),
    )


def prior_penalty(
    gates: torch.Tensor, importance: torch.Tensor, l2_anchor: torch.Tensor, l1_weight: torch.Tensor
) -> torch.Tensor:
    return (l2_anchor * (gates - importance) ** 2).sum() + (l1_weight * gates.abs()).sum()


def response_regularity(
    predict: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_mark_enc: torch.Tensor,
    x_mark_dec: torch.Tensor,
    columns: torch.Tensor,
    epsilon: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Squared symmetric finite-difference slope along a random astro direction.

    The direction is shared across the history and future marks of each sample
    and normalized per sample, so the penalty measures sensitivity to the
    astrology block as a whole.  Training mode is kept (recurrent backends may
    reject eval-mode backward), with dropout neutralized by RNG replay.
    """

    batch = x_mark_enc.shape[0]
    width = int(columns.numel())
    direction = torch.randn(
        (batch, 1, width), generator=generator, device=generator.device, dtype=torch.float32
    ).to(device=x_mark_enc.device, dtype=x_mark_enc.dtype)
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def shifted(marks: torch.Tensor, sign: float) -> torch.Tensor:
        delta = torch.zeros_like(marks)
        delta[:, :, columns] = sign * epsilon * direction.expand(-1, marks.shape[1], -1)
        return marks + delta

    # Replay one RNG state for both passes so dropout masks match and the
    # difference reflects only the astro perturbation; fork_rng then restores
    # the global stream so training randomness is unaffected.
    device = x_mark_enc.device
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        cpu_state = torch.get_rng_state()
        cuda_states = [torch.cuda.get_rng_state(index) for index in cuda_devices]
        plus = predict(shifted(x_mark_enc, 1.0), shifted(x_mark_dec, 1.0))
        torch.set_rng_state(cpu_state)
        for index, state in zip(cuda_devices, cuda_states):
            torch.cuda.set_rng_state(state, index)
        minus = predict(shifted(x_mark_enc, -1.0), shifted(x_mark_dec, -1.0))
    return ((plus - minus) ** 2).mean() / (2.0 * epsilon) ** 2
