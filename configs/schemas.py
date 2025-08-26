"""Structured configuration schemas for the Modular Autoformer (clean, deduplicated).

This module defines a single authoritative set of Pydantic v2 models and
helper factory functions. It keeps backward compatibility with legacy flat
arguments via aliases and the `from_legacy` constructor.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt, model_validator


# ---------------------------------------------------------------------------
# Component Enumeration
# ---------------------------------------------------------------------------


class ComponentType(str, Enum):
    """Enumeration of registry keys used across components and configs.

    This list unifies all component identifiers referenced by the unified
    registry/factory and tests. Values are lower-case snake_case strings.
    """

    # Attention families
    AUTOCORRELATION = "autocorrelation"
    ADAPTIVE_AUTOCORRELATION = "adaptive_autocorrelation"
    CROSS_RESOLUTION = "cross_resolution_attention"
    FOURIER_ATTENTION = "fourier_attention"
    FOURIER_BLOCK = "fourier_block"
    FOURIER_CROSS_ATTENTION = "fourier_cross_attention"
    WAVELET_ATTENTION = "wavelet_attention"
    ADAPTIVE_WAVELET_ATTENTION = "adaptive_wavelet_attention"
    MULTI_SCALE_WAVELET_ATTENTION = "multi_scale_wavelet_attention"
    VARIATIONAL_ATTENTION = "variational_attention"
    BAYESIAN_ATTENTION = "bayesian_attention"
    BAYESIAN_MULTI_HEAD_ATTENTION = "bayesian_multi_head_attention"
    BAYESIAN_CROSS_ATTENTION = "bayesian_cross_attention"
    ADAPTIVE_MIXTURE_ATTN = "adaptive_mixture_attention"
    CONVOLUTIONAL_ATTENTION = "convolutional_attention"

    # Decomposition
    MOVING_AVG = "moving_avg"
    LEARNABLE_DECOMP = "learnable_decomp"
    STABLE_DECOMP = "stable_decomp"
    WAVELET_DECOMP = "wavelet_decomp"
    WAVELET_DECOMPOSITION = "wavelet_decomposition"  # alias variant used in registrations
    ADVANCED_WAVELET = "advanced_wavelet"

    # Encoders / Decoders
    STANDARD_ENCODER = "standard_encoder"
    ENHANCED_ENCODER = "enhanced_encoder"
    HIERARCHICAL_ENCODER = "hierarchical_encoder"
    TEMPORAL_CONV_ENCODER = "temporal_conv_encoder"
    META_LEARNING_ADAPTER = "meta_learning_adapter"

    STANDARD_DECODER = "standard_decoder"
    ENHANCED_DECODER = "enhanced_decoder"
    HIERARCHICAL_DECODER = "hierarchical_decoder"

    # Sampling / Heads / Losses
    DETERMINISTIC = "deterministic"
    BAYESIAN = "bayesian"
    MONTE_CARLO = "monte_carlo"

    STANDARD_HEAD = "standard"
    QUANTILE = "quantile"
    BAYESIAN_HEAD = "bayesian_head"

    MSE = "mse"
    MAE = "mae"
    QUANTILE_LOSS = "quantile_loss"
    BAYESIAN_MSE = "bayesian_mse"
    BAYESIAN_QUANTILE = "bayesian_quantile"
    FOCAL_LOSS = "focal_loss"
    ADAPTIVE_AUTOFORMER_LOSS = "adaptive_autoformer_loss"
    ADAPTIVE_LOSS_WEIGHTING = "adaptive_loss_weighting"
    MAPE_LOSS = "mape_loss"
    SMAPE_LOSS = "smape_loss"
    MASE_LOSS = "mase_loss"
    PS_LOSS = "ps_loss"
    FREQUENCY_AWARE_LOSS = "frequency_aware_loss"
    MULTI_QUANTILE_LOSS = "multi_quantile_loss"
    UNCERTAINTY_CALIBRATION_LOSS = "uncertainty_calibration_loss"

    # Normalization
    LAYER_NORM = "layer_norm"
    RMS_NORM = "rms_norm"

    # Embeddings
    POSITIONAL_EMBEDDING = "positional_embedding"
    TOKEN_EMBEDDING = "token_embedding"
    FIXED_EMBEDDING = "fixed_embedding"
    TEMPORAL_EMBEDDING = "temporal_embedding"
    TIME_FEATURE_EMBEDDING = "time_feature_embedding"
    DATA_EMBEDDING = "data_embedding"
    DATA_EMBEDDING_INVERTED = "data_embedding_inverted"
    DATA_EMBEDDING_WO_POS = "data_embedding_wo_pos"
    PATCH_EMBEDDING = "patch_embedding"

    # Misc building blocks
    FOURIER = "fourier"
    FOURIER_RESIDUAL = "fourier_residual"
    CAUSAL_CONVOLUTION = "causal_convolution"
    TEMPORAL_CONV_NET = "temporal_conv_net"


# ---------------------------------------------------------------------------
# Component configs
# ---------------------------------------------------------------------------


class AttentionConfig(BaseModel):
    type: ComponentType
    d_model: Optional[int] = None
    n_heads: Optional[int] = None
    num_heads: Optional[int] = None  # compatibility mirror
    head_dim: Optional[int] = None
    dropout: float = 0.1
    factor: int = 1
    output_attention: bool = False
    seq_len: Optional[int] = None
    local_window: Optional[int] = None
    global_window: Optional[int] = None

    model_config = {"extra": "allow", "populate_by_name": True}

    @model_validator(mode="after")
    def _normalize_heads(self):  # type: ignore[override]
        if self.n_heads is None and self.num_heads is not None:
            self.n_heads = self.num_heads
        elif self.num_heads is None and self.n_heads is not None:
            self.num_heads = self.n_heads
        if self.n_heads is not None and self.num_heads is not None and self.n_heads != self.num_heads:
            raise ValueError("n_heads and num_heads must be equal if both are provided")
        if self.d_model is not None and self.n_heads is not None and self.head_dim is None:
            if self.d_model % self.n_heads == 0:
                self.head_dim = self.d_model // self.n_heads
        return self


class DecompositionConfig(BaseModel):
    type: ComponentType
    kernel_size: int = 25
    input_dim: Optional[int] = None
    wavelet_type: Optional[str] = "db4"
    levels: Optional[int] = 3

    model_config = {"extra": "allow"}


class HierarchicalConfig(BaseModel):
    n_levels: int = 3
    level_configs: Optional[List[Dict[str, Any]]] = None


class EncoderConfig(BaseModel):
    type: ComponentType
    num_encoder_layers: int = Field(..., alias="e_layers")
    d_ff: int
    dropout: float = 0.1
    activation: str = "gelu"
    d_model: Optional[int] = None
    n_heads: Optional[int] = None

    attention_comp: Optional[AttentionConfig] = None
    decomp_comp: Optional[DecompositionConfig] = None
    hierarchical: Optional[HierarchicalConfig] = None

    model_config = {"extra": "allow", "populate_by_name": True}

    # Legacy compatibility: allow accessing encoder layers via `e_layers`
    @property
    def e_layers(self) -> int:  # pragma: no cover - simple alias
        return self.num_encoder_layers

    @e_layers.setter
    def e_layers(self, value: int) -> None:  # pragma: no cover - simple alias
        self.num_encoder_layers = value


class DecoderConfig(BaseModel):
    type: ComponentType
    num_decoder_layers: int = Field(..., alias="d_layers")
    d_ff: int
    dropout: float = 0.1
    activation: str = "gelu"
    c_out: int
    d_model: Optional[int] = None
    n_heads: Optional[int] = None

    attention_comp: Optional[AttentionConfig] = None
    decomp_comp: Optional[DecompositionConfig] = None

    model_config = {"extra": "allow", "populate_by_name": True}

    # Legacy compatibility: allow accessing decoder layers via `d_layers`
    @property
    def d_layers(self) -> int:  # pragma: no cover - simple alias
        return self.num_decoder_layers

    @d_layers.setter
    def d_layers(self, value: int) -> None:  # pragma: no cover - simple alias
        self.num_decoder_layers = value


class SamplingConfig(BaseModel):
    type: ComponentType
    n_samples: int = 50
    quantile_levels: Optional[List[float]] = None
    dropout_rate: Optional[float] = 0.1
    temperature: Optional[float] = 1.0
    kl_weight: Optional[float] = 1e-5

    model_config = {"extra": "allow"}


class OutputHeadConfig(BaseModel):
    type: ComponentType
    d_model: int
    c_out: int
    num_quantiles: Optional[int] = None

    model_config = {"extra": "allow"}


class LossConfig(BaseModel):
    type: ComponentType
    quantiles: Optional[List[float]] = None
    prior_scale: Optional[float] = 1.0
    kl_weight: Optional[float] = 1.0

    model_config = {"extra": "allow"}


class BayesianConfig(BaseModel):
    enabled: bool = False
    layers_to_convert: List[str] = Field(default_factory=lambda: ["projection"])
    prior_scale: float = 1.0
    posterior_scale_init: float = -3.0
    kl_weight: float = 1.0


class BackboneConfig(BaseModel):
    type: Optional[ComponentType] = None
    model_name: Optional[str] = None
    use_backbone: bool = False
    prediction_length: Optional[int] = None
    context_length: Optional[int] = None

class EmbeddingConfig(BaseModel):
    # Make fields lenient for lightweight tests that only set d_model/dropout
    type: Optional[ComponentType] = None
    c_in: int = 1
    d_model: int
    max_len: int = 5000
    dropout: float = 0.1
    time_features: bool = True
    freq: str = "h"
    model_config = {"extra": "allow"}

class NormalizationConfig(BaseModel):
    type: ComponentType
    normalized_shape: int
    eps: float = 1e-5
    affine: bool = True
    model_config = {"extra": "allow"}

## NOTE: An earlier duplicated ModularAutoformerConfig and helper block existed here.
## It has been removed to keep a single authoritative definition below.


# Additional configuration factory helpers (extending coverage)
# ---------------------------------------------------------------------------


class ModularAutoformerConfig(BaseModel):
    task_name: str = "long_term_forecast"
    seq_len: int
    pred_len: int
    label_len: int
    enc_in: int
    dec_in: int
    c_out: int
    c_out_evaluation: int
    d_model: int = 512

    attention: AttentionConfig
    decomposition: DecompositionConfig
    encoder: EncoderConfig
    decoder: DecoderConfig
    sampling: SamplingConfig
    output_head: OutputHeadConfig
    loss: LossConfig

    bayesian: BayesianConfig = Field(default_factory=BayesianConfig)
    backbone: BackboneConfig = Field(default_factory=BackboneConfig)

    quantile_levels: Optional[List[float]] = None
    embed: str = "timeF"
    freq: str = "h"
    dropout: float = 0.1

    model_config = {"extra": "allow", "populate_by_name": True}

    @model_validator(mode="after")
    def _validate_consistency(self):  # type: ignore[override]
        if self.quantile_levels:
            expected_c_out = self.c_out_evaluation * len(self.quantile_levels)
            if self.c_out != expected_c_out:
                raise ValueError(
                    f"c_out ({self.c_out}) should equal c_out_evaluation ({self.c_out_evaluation}) * num_quantiles ({len(self.quantile_levels)})"
                )
        if self.encoder and self.encoder.d_model is not None and self.encoder.d_model != self.d_model:
            raise ValueError("Encoder d_model must match top-level d_model")
        if self.decoder and self.decoder.d_model is not None and self.decoder.d_model != self.d_model:
            raise ValueError("Decoder d_model must match top-level d_model")
        return self

    def to_namespace(self):
        from argparse import Namespace

        flat: Dict[str, Any] = {
            "task_name": self.task_name,
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "label_len": self.label_len,
            "enc_in": self.enc_in,
            "dec_in": self.dec_in,
            "c_out": self.c_out,
            "c_out_evaluation": self.c_out_evaluation,
            "d_model": self.d_model,
            "embed": self.embed,
            "freq": self.freq,
            "dropout": self.dropout,
            # Component types
            "attention_type": self.attention.type.value,
            "decomposition_type": self.decomposition.type.value,
            "encoder_type": self.encoder.type.value,
            "decoder_type": self.decoder.type.value,
            "sampling_type": self.sampling.type.value,
            "output_head_type": self.output_head.type.value,
            "loss_function_type": self.loss.type.value,
            # Component params
            "attention_params": self.attention.model_dump(exclude={"type"}),
            "decomposition_params": self.decomposition.model_dump(exclude={"type"}),
            "encoder_params": self.encoder.model_dump(exclude={"type"}),
            "decoder_params": self.decoder.model_dump(exclude={"type"}),
            "sampling_params": self.sampling.model_dump(exclude={"type"}),
            "output_head_params": self.output_head.model_dump(exclude={"type"}),
            "loss_params": self.loss.model_dump(exclude={"type"}),
            # Specials
            "quantile_levels": self.quantile_levels,
            "bayesian_layers": self.bayesian.layers_to_convert if self.bayesian.enabled else [],
            "use_backbone_component": self.backbone.use_backbone,
            "backbone_type": self.backbone.type.value if self.backbone.type else None,
        }
        return Namespace(**flat)

    @classmethod
    def from_legacy(cls, legacy_config):
        """Construct from a legacy Namespace or dict with best-effort mappings."""
        cfg = legacy_config if isinstance(legacy_config, dict) else vars(legacy_config)

        def _map_type(val: Optional[str], default: ComponentType, domain: str) -> ComponentType:
            if val is None:
                return default
            v = str(val).lower()
            if domain == "encoder":
                if v in {"enhanced", "enhanced_encoder"}:
                    return ComponentType.ENHANCED_ENCODER
                if v in {"standard", "standard_encoder"}:
                    return ComponentType.STANDARD_ENCODER
                if v in {"hierarchical", "hierarchical_encoder"}:
                    return ComponentType.HIERARCHICAL_ENCODER
            if domain == "decoder":
                if v in {"enhanced", "enhanced_decoder"}:
                    return ComponentType.ENHANCED_DECODER
                if v in {"standard", "standard_decoder"}:
                    return ComponentType.STANDARD_DECODER
                if v in {"hierarchical", "hierarchical_decoder"}:
                    return ComponentType.HIERARCHICAL_DECODER
            if domain == "attention":
                if v in {"adaptive_autocorrelation", "adaptive_autocorrelation_layer"}:
                    return ComponentType.ADAPTIVE_AUTOCORRELATION
                if v in {"autocorrelation", "autocorrelation_layer"}:
                    return ComponentType.AUTOCORRELATION_LAYER
            if domain == "decomposition":
                if v in {"learnable", "learnable_decomp"}:
                    return ComponentType.LEARNABLE_DECOMP
            if domain == "sampling":
                if v in {"deterministic"}:
                    return ComponentType.DETERMINISTIC
                if v in {"bayesian"}:
                    return ComponentType.BAYESIAN
            if domain == "head":
                if v in {"standard", "linear", "head"}:
                    return ComponentType.STANDARD_HEAD
                if v in {"quantile", "quantile_head"}:
                    return ComponentType.QUANTILE
            if domain == "loss":
                if v in {"mse"}:
                    return ComponentType.MSE
                if v in {"quantile", "quantile_loss"}:
                    return ComponentType.QUANTILE_LOSS
                if v in {"bayesian_mse", "bayesian_mse_loss"}:
                    return ComponentType.BAYESIAN_MSE
                if v in {"bayesian_quantile", "bayesian_quantile_loss"}:
                    return ComponentType.BAYESIAN_QUANTILE
            return ComponentType(v)

        def _clean_params(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            if not isinstance(d, dict):
                return {}
            cleaned = dict(d)
            cleaned.pop("type", None)
            return cleaned

        base_params = {
            "task_name": cfg.get("task_name", "long_term_forecast"),
            "seq_len": cfg["seq_len"],
            "pred_len": cfg["pred_len"],
            "label_len": cfg["label_len"],
            "enc_in": cfg["enc_in"],
            "dec_in": cfg["dec_in"],
            "c_out": cfg["c_out"],
            "c_out_evaluation": cfg.get("c_out_evaluation", cfg["c_out"]),
            "d_model": cfg.get("d_model", 512),
            "embed": cfg.get("embed", "timeF"),
            "freq": cfg.get("freq", "h"),
            "dropout": cfg.get("dropout", 0.1),
            "quantile_levels": cfg.get("quantile_levels"),
        }

        attention = AttentionConfig(
            type=_map_type(cfg.get("attention_type"), ComponentType.ADAPTIVE_AUTOCORRELATION, "attention"),
            **_clean_params(cfg.get("attention_params")),
        )
        decomposition = DecompositionConfig(
            type=_map_type(cfg.get("decomposition_type"), ComponentType.LEARNABLE_DECOMP, "decomposition"),
            **_clean_params(cfg.get("decomposition_params")),
        )
        encoder = EncoderConfig(
            type=_map_type(cfg.get("encoder_type"), ComponentType.ENHANCED_ENCODER, "encoder"),
            **_clean_params(cfg.get("encoder_params")),
        )
        decoder = DecoderConfig(
            type=_map_type(cfg.get("decoder_type"), ComponentType.ENHANCED_DECODER, "decoder"),
            **_clean_params(cfg.get("decoder_params")),
        )
        sampling = SamplingConfig(
            type=_map_type(cfg.get("sampling_type"), ComponentType.DETERMINISTIC, "sampling"),
            **_clean_params(cfg.get("sampling_params")),
        )
        output_head = OutputHeadConfig(
            type=_map_type(cfg.get("output_head_type"), ComponentType.STANDARD_HEAD, "head"),
            **_clean_params(cfg.get("output_head_params")),
        )
        loss = LossConfig(
            type=_map_type(cfg.get("loss_function_type"), ComponentType.MSE, "loss"),
            **_clean_params(cfg.get("loss_params")),
        )

        bayesian = BayesianConfig(
            enabled=bool(cfg.get("bayesian_layers")),
            layers_to_convert=cfg.get("bayesian_layers", []),
        )
        backbone = BackboneConfig(
            use_backbone=cfg.get("use_backbone_component", False),
            type=ComponentType(cfg.get("backbone_type")) if cfg.get("backbone_type") else None,
        )

        return cls(
            **base_params,
            attention=attention,
            decomposition=decomposition,
            encoder=encoder,
            decoder=decoder,
            sampling=sampling,
            output_head=output_head,
            loss=loss,
            bayesian=bayesian,
            backbone=backbone,
        )


## NOTE: Duplicate helper factory functions removed. A clean set is defined earlier in this module.


# ---------------------------------------------------------------------------
# Backward-compatible helper factories used by tests and examples
# These small builders return sensible preset configurations without
# duplicating schema definitions. Keep them thin shims around
# ModularAutoformerConfig to avoid drift.
# ---------------------------------------------------------------------------

def _make_common_configs(
    *,
    seq_len: int = 96,
    pred_len: int = 24,
    label_len: int = 48,
    enc_in: int = 7,
    dec_in: int = 7,
    c_out_evaluation: int = 7,
    d_model: int = 512,
    n_heads: int = 8,
    d_ff: int = 2048,
    dropout: float = 0.1,
    attention_type: ComponentType = ComponentType.AUTOCORRELATION,
    decomp_type: ComponentType = ComponentType.LEARNABLE_DECOMP,
    encoder_type: ComponentType = ComponentType.ENHANCED_ENCODER,
    decoder_type: ComponentType = ComponentType.ENHANCED_DECODER,
    sampling_type: ComponentType = ComponentType.DETERMINISTIC,
    head_type: ComponentType = ComponentType.STANDARD_HEAD,
    loss_type: ComponentType = ComponentType.MSE,
    quantile_levels: Optional[List[float]] = None,
    bayesian_enabled: bool = False,
    bayesian_layers: Optional[List[str]] = None,
) -> ModularAutoformerConfig:
    """Internal helper to assemble a ModularAutoformerConfig with sane defaults."""
    num_q = len(quantile_levels) if quantile_levels else 0
    c_out = c_out_evaluation * num_q if num_q > 0 else c_out_evaluation

    attention = AttentionConfig(
        type=attention_type,
        d_model=d_model,
        n_heads=n_heads,
        dropout=dropout,
        factor=1,
        seq_len=seq_len,
    )
    decomposition = DecompositionConfig(
        type=decomp_type,
        kernel_size=25,
        input_dim=d_model,
    )
    encoder = EncoderConfig(
        type=encoder_type,
        e_layers=2,
        d_ff=d_ff,
        dropout=dropout,
        activation="gelu",
        d_model=d_model,
        n_heads=n_heads,
        attention_comp=attention,
        decomp_comp=decomposition,
    )
    decoder = DecoderConfig(
        type=decoder_type,
        d_layers=1,
        d_ff=d_ff,
        dropout=dropout,
        activation="gelu",
        c_out=c_out,
        d_model=d_model,
        n_heads=n_heads,
        attention_comp=attention,
        decomp_comp=decomposition,
    )
    sampling = SamplingConfig(
        type=sampling_type,
        n_samples=50 if bayesian_enabled else 0,
        quantile_levels=quantile_levels,
    )
    output_head = OutputHeadConfig(
        type=head_type,
        d_model=d_model,
        c_out=c_out_evaluation,
        num_quantiles=num_q or None,
    )
    loss = LossConfig(
        type=loss_type,
        quantiles=quantile_levels,
    )
    bayesian = BayesianConfig(
        enabled=bayesian_enabled,
        layers_to_convert=bayesian_layers or ["projection"],
    )
    backbone = BackboneConfig(use_backbone=False, type=None)

    return ModularAutoformerConfig(
        task_name="long_term_forecast",
        seq_len=seq_len,
        pred_len=pred_len,
        label_len=label_len,
        enc_in=enc_in,
        dec_in=dec_in,
        c_out=c_out,
        c_out_evaluation=c_out_evaluation,
        d_model=d_model,
        attention=attention,
        decomposition=decomposition,
        encoder=encoder,
        decoder=decoder,
        sampling=sampling,
        output_head=output_head,
        loss=loss,
        bayesian=bayesian,
        backbone=backbone,
        quantile_levels=quantile_levels,
        embed="timeF",
        freq="h",
        dropout=dropout,
    )


def _apply_builder_overrides(
    cfg: ModularAutoformerConfig,
    *,
    num_targets: Optional[int] = None,
    num_covariates: Optional[int] = None,
    seq_len: Optional[int] = None,
    label_len: Optional[int] = None,
    pred_len: Optional[int] = None,
    d_model: Optional[int] = None,
) -> ModularAutoformerConfig:
    """Apply lightweight overrides expected by tests without duplicating schema logic.

    - num_targets maps to c_out_evaluation (and updates c_out if quantiles present)
    - num_covariates contributes to enc_in/dec_in along with targets
    - seq/label/pred lengths override top-level dims
    - d_model propagates to encoder/decoder/attention dims
    """
    # Dimensions
    if seq_len is not None:
        cfg.seq_len = seq_len
    if label_len is not None:
        cfg.label_len = label_len
    if pred_len is not None:
        cfg.pred_len = pred_len

    # Channel layout
    if num_targets is not None:
        cfg.c_out_evaluation = int(num_targets)
    if num_targets is not None or num_covariates is not None:
        t = int(num_targets or cfg.c_out_evaluation)
        c = int(num_covariates or 0)
        total = t + c
        cfg.enc_in = total
        cfg.dec_in = total
        # If quantiles are set, expand c_out accordingly else keep equal to targets
        if cfg.quantile_levels:
            cfg.c_out = cfg.c_out_evaluation * len(cfg.quantile_levels)
        else:
            cfg.c_out = cfg.c_out_evaluation

    # Model width
    if d_model is not None:
        cfg.d_model = int(d_model)
        if cfg.attention:
            cfg.attention.d_model = cfg.d_model
        if cfg.encoder:
            cfg.encoder.d_model = cfg.d_model
        if cfg.decoder:
            cfg.decoder.d_model = cfg.d_model
        if cfg.output_head:
            cfg.output_head.d_model = cfg.d_model
    return cfg


def create_enhanced_config(**overrides) -> ModularAutoformerConfig:
    """Preset for enhanced encoder/decoder with standard head and MSE loss.

    Accepts optional overrides such as num_targets, num_covariates, seq_len,
    label_len, pred_len, d_model as used by tests.
    """
    cfg = _make_common_configs(
        encoder_type=ComponentType.ENHANCED_ENCODER,
        decoder_type=ComponentType.ENHANCED_DECODER,
        head_type=ComponentType.STANDARD_HEAD,
        loss_type=ComponentType.MSE,
    )
    return _apply_builder_overrides(cfg, **{k: v for k, v in overrides.items() if k in {"num_targets","num_covariates","seq_len","label_len","pred_len","d_model"}})


def create_hierarchical_config(**overrides) -> ModularAutoformerConfig:
    """Preset for hierarchical encoder/decoder.

    Supports the same overrides as create_enhanced_config.
    """
    cfg = _make_common_configs(
        encoder_type=ComponentType.HIERARCHICAL_ENCODER,
        decoder_type=ComponentType.HIERARCHICAL_DECODER,
        head_type=ComponentType.STANDARD_HEAD,
        loss_type=ComponentType.MSE,
    )
    return _apply_builder_overrides(cfg, **{k: v for k, v in overrides.items() if k in {"num_targets","num_covariates","seq_len","label_len","pred_len","d_model"}})


def create_bayesian_enhanced_config(**overrides) -> ModularAutoformerConfig:
    """Preset for enhanced encoder/decoder with Bayesian sampling and loss.

    Supports overrides like num_targets/num_covariates/seq_len/label_len/pred_len/d_model.
    """
    cfg = _make_common_configs(
        encoder_type=ComponentType.ENHANCED_ENCODER,
        decoder_type=ComponentType.ENHANCED_DECODER,
        sampling_type=ComponentType.BAYESIAN,
        head_type=ComponentType.STANDARD_HEAD,
        loss_type=ComponentType.BAYESIAN_MSE,
        bayesian_enabled=True,
        bayesian_layers=["projection", "ffn"],
    )
    return _apply_builder_overrides(cfg, **{k: v for k, v in overrides.items() if k in {"num_targets","num_covariates","seq_len","label_len","pred_len","d_model"}})


def create_quantile_bayesian_config(**overrides) -> ModularAutoformerConfig:
    """Preset for quantile forecasting with Bayesian training.

    Uses three quantiles by default and expands c_out accordingly while keeping
    c_out_evaluation equal to the target dimension for metrics.
    """
    quantiles = [0.1, 0.5, 0.9]
    cfg = _make_common_configs(
        encoder_type=ComponentType.ENHANCED_ENCODER,
        decoder_type=ComponentType.ENHANCED_DECODER,
        sampling_type=ComponentType.BAYESIAN,
        head_type=ComponentType.QUANTILE,
        loss_type=ComponentType.BAYESIAN_QUANTILE,
        quantile_levels=quantiles,
        bayesian_enabled=True,
        bayesian_layers=["projection", "ffn"],
    )
    return _apply_builder_overrides(cfg, **{k: v for k, v in overrides.items() if k in {"num_targets","num_covariates","seq_len","label_len","pred_len","d_model"}})
