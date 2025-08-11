"""Core component integration smoke tests (Batch B migration).

Covers registry creation + minimal forward for encoder/decoder/sampling/output head combinations
that represent the canonical deterministic path. Ensures shapes align and components interoperate.
"""
from __future__ import annotations

import torch
import pytest

from configs.modular_components import component_registry, register_all_components, ModularAssembler
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


BASIC_ATTENTION_TYPES = [ComponentType.AUTOCORRELATION, ComponentType.ADAPTIVE_AUTOCORRELATION]
BASIC_DECOMP_TYPES = [ComponentType.MOVING_AVG, ComponentType.LEARNABLE_DECOMP]

@pytest.mark.parametrize("attn_type", BASIC_ATTENTION_TYPES)
@pytest.mark.parametrize("decomp_type", BASIC_DECOMP_TYPES)
def test_core_encoder_decoder_pipeline(attn_type, decomp_type) -> None:
    """End-to-end assembly of encoder+decoder with basic attention & decomposition variants."""
    register_all_components()

    seq_len, pred_len, label_len = 48, 12, 24
    d_model = 64
    n_heads = 4
    n_targets = 2  # output dimension
    feature_dim = d_model  # align input feature dimension with d_model to satisfy attention projections

    attention_cfg = AttentionConfig(
        type=attn_type,
        d_model=d_model,
        n_heads=n_heads,
        dropout=0.0,
        factor=1,
        output_attention=False,
    )
    decomp_cfg = DecompositionConfig(type=decomp_type, kernel_size=25, input_dim=d_model)
    encoder_cfg = EncoderConfig(
        type=ComponentType.STANDARD_ENCODER,
        e_layers=1,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=128,
        dropout=0.0,
        activation="gelu",
    )
    decoder_cfg = DecoderConfig(
        type=ComponentType.STANDARD_DECODER,
        d_layers=1,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=128,
        dropout=0.0,
        activation="gelu",
        c_out=n_targets,
    )
    sampling_cfg = SamplingConfig(type=ComponentType.DETERMINISTIC)
    output_head_cfg = OutputHeadConfig(
        type=ComponentType.STANDARD_HEAD,
        d_model=d_model,
        c_out=n_targets,
    )
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

    batch = 2
    x_enc = torch.randn(batch, seq_len, feature_dim)
    x_mark_enc = torch.zeros(batch, seq_len, 1)
    x_dec = torch.randn(batch, label_len + pred_len, feature_dim)
    x_mark_dec = torch.zeros(batch, label_len + pred_len, 1)

    with torch.no_grad():
        out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    assert out.shape[0] == batch and out.shape[-1] == n_targets
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "sampling_type", [ComponentType.DETERMINISTIC, ComponentType.BAYESIAN]
)
def test_sampling_component_interop(sampling_type) -> None:
    """Validate that switching sampling type keeps forward path functional (Bayesian treated as deterministic if uncertainty disabled)."""
    register_all_components()

    seq_len, pred_len, label_len = 32, 8, 16
    d_model = 32
    n_heads = 4
    n_targets = 1  # output dim
    feature_dim = d_model

    attention_cfg = AttentionConfig(
        type=ComponentType.AUTOCORRELATION,
        d_model=d_model,
        n_heads=n_heads,
        dropout=0.0,
        factor=1,
        output_attention=False,
    )
    decomp_cfg = DecompositionConfig(type=ComponentType.MOVING_AVG, kernel_size=25)
    encoder_cfg = EncoderConfig(
        type=ComponentType.STANDARD_ENCODER,
        e_layers=1,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=128,
        dropout=0.0,
        activation="gelu",
    )
    decoder_cfg = DecoderConfig(
        type=ComponentType.STANDARD_DECODER,
        d_layers=1,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=128,
        dropout=0.0,
        activation="gelu",
        c_out=n_targets,
    )
    sampling_cfg = SamplingConfig(type=sampling_type, n_samples=10 if sampling_type == ComponentType.BAYESIAN else 1)
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
        out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    assert out.shape[-1] == n_targets and torch.isfinite(out).all()
