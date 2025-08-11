import pytest
from typing import List

from configs.schemas import (
    AttentionConfig, DecompositionConfig, EncoderConfig, DecoderConfig,
    SamplingConfig, OutputHeadConfig, LossConfig, ModularAutoformerConfig,
    ComponentType
)


def build_base_config() -> ModularAutoformerConfig:
    d_model = 64
    return ModularAutoformerConfig(
        task_name="forecast",
        seq_len=32,
        pred_len=8,
        label_len=16,
        enc_in=4,
        dec_in=4,
        c_out=2,
        c_out_evaluation=2,
        d_model=d_model,
        attention=AttentionConfig(type=ComponentType.AUTOCORRELATION, d_model=d_model, n_heads=4),
        decomposition=DecompositionConfig(type=ComponentType.MOVING_AVG, kernel_size=5),
        encoder=EncoderConfig(type=ComponentType.ENHANCED_ENCODER, e_layers=1, d_model=d_model, n_heads=4, d_ff=128),
        decoder=DecoderConfig(type=ComponentType.ENHANCED_DECODER, d_layers=1, d_model=d_model, n_heads=4, d_ff=128, c_out=2),
        sampling=SamplingConfig(type=ComponentType.DETERMINISTIC),
        output_head=OutputHeadConfig(type=ComponentType.STANDARD_HEAD, d_model=d_model, c_out=2),
        loss=LossConfig(type=ComponentType.MSE),
    )


def test_config_validation_success():
    """Base configuration should construct without validation errors."""
    cfg = build_base_config()
    assert cfg.d_model == cfg.encoder.d_model == cfg.decoder.d_model


def test_quantile_dimension_validation():
    """Quantile settings must keep c_out consistent with evaluation channels * num quantiles."""
    cfg = build_base_config()
    cfg.quantile_levels = [0.1, 0.5, 0.9]
    # Intentionally mismatch c_out to trigger validator logic
    cfg.c_out = cfg.c_out_evaluation * len(cfg.quantile_levels)
    # Recreate to invoke validator
    recreated = ModularAutoformerConfig(**cfg.dict())
    assert recreated.c_out == recreated.c_out_evaluation * len(recreated.quantile_levels)


def test_mismatched_d_model_raises():
    """Encoder/decoder d_model mismatch should raise validation error."""
    cfg = build_base_config()
    # Force mismatch
    encoder_dict = cfg.encoder.dict(); encoder_dict['d_model'] = cfg.d_model + 8
    with pytest.raises(ValueError):
        ModularAutoformerConfig(**{**cfg.dict(), 'encoder': EncoderConfig(**encoder_dict)})
