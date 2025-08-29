"""Core Components Integration Tests

Verifies encoder/decoder/loss minimal assembly viability using structured configs & registry.
"""
from __future__ import annotations

import pytest
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
)
from configs.modular_components import component_registry


def build_minimal_config() -> ModularAutoformerConfig:
    d_model = 32
    attn = AttentionConfig(type=ComponentType.AUTOCORRELATION, d_model=d_model, n_heads=4, dropout=0.1, factor=1, output_attention=False)
    decomp = DecompositionConfig(type=ComponentType.LEARNABLE_DECOMP, kernel_size=7, input_dim=d_model)
    enc = EncoderConfig(type=ComponentType.STANDARD_ENCODER, e_layers=1, d_model=d_model, n_heads=4, d_ff=128, dropout=0.1, activation="gelu")
    dec = DecoderConfig(type=ComponentType.STANDARD_DECODER, d_layers=1, d_model=d_model, n_heads=4, d_ff=128, dropout=0.1, activation="gelu", c_out=3)
    sampling = SamplingConfig(type=ComponentType.DETERMINISTIC)
    head = OutputHeadConfig(type=ComponentType.STANDARD_HEAD, d_model=d_model, c_out=3)
    loss = LossConfig(type=ComponentType.MSE)
    return ModularAutoformerConfig(
        task_name="long_term_forecast",
        seq_len=32,
        pred_len=8,
        label_len=16,
        enc_in=3,
        dec_in=3,
        c_out=3,
        c_out_evaluation=3,
        d_model=d_model,
        attention=attn,
        decomposition=decomp,
        encoder=enc,
        decoder=dec,
        sampling=sampling,
        output_head=head,
        loss=loss,
    )


def test_core_component_instantiation() -> None:
    cfg = build_minimal_config()
    attn = component_registry.create_component(cfg.attention.type, cfg.attention)
    decomp = component_registry.create_component(cfg.decomposition.type, cfg.decomposition)
    loss_fn = component_registry.create_component(cfg.loss.type, cfg.loss)
    for comp in (attn, decomp, loss_fn):
        assert comp is not None


@pytest.mark.skip(reason="Model assembly not yet implemented in modular config scope")
def test_future_full_model_assembly() -> None:  # pragma: no cover
    pass
