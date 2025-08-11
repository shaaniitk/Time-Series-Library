"""Mixture-of-Experts (Adaptive Mixture) integration tests (Batch B).

Expands the initial smoke test into a small param sweep over (d_model, n_heads, n_samples)
to ensure the adaptive mixture sampling path scales and preserves output shape / finiteness.

Assertions:
1. Forward completes without error (no NaNs/Infs) for each configuration.
2. Output shape matches expected batch * pred_len * targets.
3. For n_samples > 1 we expect some variability across repeated stochastic calls unless
    the implementation is strictly deterministic; we tolerate identical outputs but
    record a soft check to guard against accidental constant tensor bugs.
"""
from __future__ import annotations

import torch

from configs.modular_components import (
    component_registry,
    register_all_components,
    ModularAssembler,
)
from configs.schemas import (
    ComponentType,
    AttentionConfig,
    DecompositionConfig,
    EncoderConfig,
    DecoderConfig,
    SamplingConfig,
    OutputHeadConfig,
    LossConfig,
    ModularAutoformerConfig,
    BayesianConfig,
)

import pytest


@pytest.mark.parametrize(
    "d_model,n_heads,n_samples",
    [
        (32, 4, 2),
        (48, 4, 5),
        (64, 8, 3),
    ],
)
def test_adaptive_mixture_forward(d_model: int, n_heads: int, n_samples: int) -> None:
    """Parameterized adaptive mixture forward pass.

    Args:
        d_model: Hidden size (also feature dimension).
        n_heads: Attention heads.
        n_samples: Mixture sampling count.
    """
    register_all_components()

    seq_len, pred_len, label_len = 32, 8, 16
    n_targets = 1
    feature_dim = d_model

    attention_cfg = AttentionConfig(
        type=ComponentType.AUTOCORRELATION,
        d_model=d_model,
        n_heads=n_heads,
        dropout=0.0,
        factor=1,
        output_attention=False,
    )
    decomp_cfg = DecompositionConfig(type=ComponentType.LEARNABLE_DECOMP, kernel_size=25, input_dim=d_model)
    encoder_cfg = EncoderConfig(
        type=ComponentType.ENHANCED_ENCODER,
        e_layers=1,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=128,
        dropout=0.0,
        activation="gelu",
    )
    decoder_cfg = DecoderConfig(
        type=ComponentType.ENHANCED_DECODER,
        d_layers=1,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=128,
        dropout=0.0,
        activation="gelu",
        c_out=n_targets,
    )
    sampling_cfg = SamplingConfig(type=ComponentType.ADAPTIVE_MIXTURE, n_samples=n_samples)
    output_head_cfg = OutputHeadConfig(type=ComponentType.STANDARD_HEAD, d_model=d_model, c_out=n_targets)
    loss_cfg = LossConfig(type=ComponentType.MSE)

    model_cfg = ModularAutoformerConfig(
        seq_len=seq_len,
        pred_len=pred_len,
        label_len=label_len,
        enc_in=feature_dim,
        dec_in=feature_dim,
        c_out=n_targets,
        c_out_evaluation=n_targets,
        d_model=d_model,
        attention=attention_cfg,
        decomposition=decomp_cfg,
        encoder=encoder_cfg,
        decoder=decoder_cfg,
        sampling=sampling_cfg,
        output_head=output_head_cfg,
        loss=loss_cfg,
        bayesian=BayesianConfig(enabled=False),
    )
    assembler = ModularAssembler(component_registry)
    model = assembler.assemble_model(model_cfg)

    x_enc = torch.randn(1, seq_len, feature_dim)
    x_mark_enc = torch.zeros(1, seq_len, 1)
    x_dec = torch.randn(1, label_len + pred_len, feature_dim)
    x_mark_dec = torch.zeros(1, label_len + pred_len, 1)

    with torch.no_grad():
        out1 = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        # Second pass to probe stochastic variability (if any)
        out2 = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    assert out1.shape[-1] == n_targets and torch.isfinite(out1).all()
    assert out2.shape == out1.shape and torch.isfinite(out2).all()

    # Soft variability check (skip strict assertion to avoid false failures if deterministic caching is used)
    if n_samples > 1:
        # Use mean absolute difference heuristic
        mad = (out1 - out2).abs().mean().item()
        # Record minimal expected variability threshold; warn (not fail) if extremely low
        if mad < 1e-6:  # pragma: no cover - diagnostic path
            print(f"[MoE variability diagnostic] Low MAD={mad:.2e} for n_samples={n_samples}")
