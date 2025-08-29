"""
Decoder wrappers that adapt legacy layers.modular.decoders into the utils registry
as 'processor' components with a uniform process_sequence API.

We expose decoders as processors that take (decoder_input, encoder_output, trend)
and return the seasonal output (and update trend internally).

This file also registers the wrapped decoders into the global ComponentRegistry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ...modular_components.base_interfaces import BaseProcessor
from ...modular_components.config_schemas import ProcessorConfig
from ...modular_components.registry import register_component


@dataclass
class DecoderProcessorConfig(ProcessorConfig):
    """Configuration for wrapped decoders exposed as processors.
    
    Attributes:
        c_out: output channels for projection.
        use_projection: whether to add final projection conv.
    """
    c_out: int = 1
    use_projection: bool = True


class _BaseDecoderProcessor(BaseProcessor):
    def __init__(self, config: DecoderProcessorConfig):
        super().__init__(config)
        self.config = config
        self._d_model = config.d_model

    def get_output_dim(self) -> int:
        return self._d_model

    def forward(self, decoder_input: torch.Tensor, encoder_output: Optional[torch.Tensor] = None, target_length: Optional[int] = None, **kwargs) -> torch.Tensor:
        # Delegate to process_sequence; target_length defaults to decoder_input length
        return self.process_sequence(decoder_input, encoder_output, target_length or decoder_input.size(1), **kwargs)


class _LegacyDecompAdapter(nn.Module):
    """Adapter to make utils decomposition processors look like legacy decomp returning (seasonal, trend)."""
    def __init__(self, decomp_processor: BaseProcessor, kernel_size: int = 25):
        super().__init__()
        self._proc = decomp_processor
        self._legacy = getattr(decomp_processor, 'decomp', None)
        if self._legacy is None:
            from layers.Autoformer_EncDec import series_decomp
            self._legacy = series_decomp(kernel_size)

    def forward(self, x: torch.Tensor):
        try:
            return self._legacy(x)
        except Exception:
            with torch.no_grad():
                seasonal = self._proc.process_sequence(x, None, x.size(1))
            trend = x - seasonal
            return seasonal, trend


class StandardDecoderProcessor(_BaseDecoderProcessor):
    """Wraps Standard decoder semantics using modular decoder layers."""

    def __init__(self, config: DecoderProcessorConfig):
        super().__init__(config)
        from layers.Autoformer_EncDec import Decoder as AutoformerDecoder
        from layers.modular.layers.standard_layers import StandardDecoderLayer
        # Components expected via custom_params
        self_attn = config.custom_params.get('self_attention_component')
        cross_attn = config.custom_params.get('cross_attention_component')
        decomp = config.custom_params.get('decomposition_component')
        if hasattr(decomp, 'process_sequence'):
            k = getattr(getattr(decomp, 'config', object()), 'kernel_size', 25)
            decomp = _LegacyDecompAdapter(decomp, kernel_size=k)
        d_model = config.d_model
        c_out = config.custom_params.get('c_out', config.c_out)
        d_layers = config.custom_params.get('d_layers', 2)
        d_ff = config.custom_params.get('d_ff', 4 * d_model)
        activation = config.custom_params.get('activation', 'relu')
        dropout = config.dropout
        projection = None
        if config.use_projection:
            # Project last dimension [B, L, D] -> [B, L, c_out]
            projection = nn.Linear(d_model, c_out, bias=False)
        self.decoder = AutoformerDecoder(
            [
                StandardDecoderLayer(
                    self_attn, cross_attn, decomp, d_model, c_out, d_ff, dropout=dropout, activation=activation
                ) for _ in range(d_layers)
            ],
            norm_layer=config.custom_params.get('norm_layer'),
            projection=projection
        )

    def process_sequence(self, decoder_input: torch.Tensor, encoder_output: Optional[torch.Tensor], target_length: int, **kwargs) -> torch.Tensor:
        x_mask = kwargs.get('x_mask')
        cross_mask = kwargs.get('cross_mask')
        trend = kwargs.get('trend')
        out, trend_out = self.decoder(decoder_input, encoder_output, x_mask=x_mask, cross_mask=cross_mask, trend=trend)
        if out.size(1) != target_length:
            out = nn.functional.interpolate(out.transpose(1, 2), size=target_length, mode='linear', align_corners=False).transpose(1, 2)
        return out

    def get_processor_type(self) -> str:
        return "decoder_standard"


class EnhancedDecoderProcessor(_BaseDecoderProcessor):
    """Enhanced decoder assembled locally without importing models package."""

    def __init__(self, config: DecoderProcessorConfig):
        super().__init__(config)
        from layers.Autoformer_EncDec import Decoder as AutoformerDecoder
        from layers.modular.layers.enhanced_layers import EnhancedDecoderLayer
        self_attn = config.custom_params.get('self_attention_component')
        cross_attn = config.custom_params.get('cross_attention_component')
        decomp = config.custom_params.get('decomposition_component')
        if hasattr(decomp, 'process_sequence'):
            k = getattr(getattr(decomp, 'config', object()), 'kernel_size', 25)
            decomp = _LegacyDecompAdapter(decomp, kernel_size=k)
        d_model = config.d_model
        c_out = config.custom_params.get('c_out', config.c_out)
        d_layers = config.custom_params.get('d_layers', 2)
        d_ff = config.custom_params.get('d_ff', 4 * d_model)
        activation = config.custom_params.get('activation', 'relu')
        dropout = config.dropout
        projection = None
        if config.use_projection:
            # Enhanced projection operating on feature dim
            projection = nn.Sequential(
                nn.Linear(d_model, d_model, bias=False),
                nn.ReLU(),
                nn.Linear(d_model, c_out, bias=False),
            )
        self.decoder = AutoformerDecoder(
            [
                EnhancedDecoderLayer(
                    self_attn, cross_attn, decomp, d_model, c_out, d_ff, dropout=dropout, activation=activation
                ) for _ in range(d_layers)
            ],
            norm_layer=config.custom_params.get('norm_layer'),
            projection=projection
        )

    def process_sequence(self, decoder_input: torch.Tensor, encoder_output: Optional[torch.Tensor], target_length: int, **kwargs) -> torch.Tensor:
        x_mask = kwargs.get('x_mask')
        cross_mask = kwargs.get('cross_mask')
        trend = kwargs.get('trend')
        out, trend_out = self.decoder(decoder_input, encoder_output, x_mask=x_mask, cross_mask=cross_mask, trend=trend)
        if out.size(1) != target_length:
            out = nn.functional.interpolate(out.transpose(1, 2), size=target_length, mode='linear', align_corners=False).transpose(1, 2)
        return out

    def get_processor_type(self) -> str:
        return "decoder_enhanced"


class StableDecoderProcessor(_BaseDecoderProcessor):
    """Stable decoder using enhanced layers, fixed-weights variant is deprecated."""

    def __init__(self, config: DecoderProcessorConfig):
        super().__init__(config)
        from layers.Autoformer_EncDec import Decoder as AutoformerDecoder
        from layers.modular.layers.enhanced_layers import EnhancedDecoderLayer
        self_attn = config.custom_params.get('self_attention_component')
        cross_attn = config.custom_params.get('cross_attention_component')
        decomp = config.custom_params.get('decomposition_component')
        if hasattr(decomp, 'process_sequence'):
            k = getattr(getattr(decomp, 'config', object()), 'kernel_size', 25)
            decomp = _LegacyDecompAdapter(decomp, kernel_size=k)
        d_model = config.d_model
        c_out = config.custom_params.get('c_out', config.c_out)
        d_layers = config.custom_params.get('d_layers', 2)
        d_ff = config.custom_params.get('d_ff', 4 * d_model)
        activation = config.custom_params.get('activation', 'relu')
        dropout = config.dropout
        projection = None
        if config.use_projection:
            projection = nn.Linear(d_model, c_out, bias=False)
        self.decoder = AutoformerDecoder(
            [
                EnhancedDecoderLayer(
                    self_attn, cross_attn, decomp, d_model, c_out, d_ff, dropout=dropout, activation=activation
                ) for _ in range(d_layers)
            ],
            norm_layer=config.custom_params.get('norm_layer'),
            projection=projection
        )

    def process_sequence(self, decoder_input: torch.Tensor, encoder_output: Optional[torch.Tensor], target_length: int, **kwargs) -> torch.Tensor:
        x_mask = kwargs.get('x_mask')
        cross_mask = kwargs.get('cross_mask')
        trend = kwargs.get('trend')
        out, trend_out = self.decoder(decoder_input, encoder_output, x_mask=x_mask, cross_mask=cross_mask, trend=trend)
        if out.size(1) != target_length:
            out = nn.functional.interpolate(out.transpose(1, 2), size=target_length, mode='linear', align_corners=False).transpose(1, 2)
        return out

    def get_processor_type(self) -> str:
        return "decoder_stable"


def register_layers_decoders() -> None:
    """Register wrapped decoder processors into the utils registry."""
    register_component(
        "processor",
        "decoder_standard_processor",
        StandardDecoderProcessor,
        metadata={
            "source": "layers.modular.decoder.standard_decoder",
            "features": ["transformer", "autoformer_layers"],
            "domain": "decoder",
            "sophistication_level": "medium",
        },
    )
    register_component(
        "processor",
        "decoder_enhanced_processor",
        EnhancedDecoderProcessor,
        metadata={
            "source": "layers.modular.decoder.enhanced_decoder",
            "features": ["transformer", "gated_ffn", "scaled_attn"],
            "domain": "decoder",
            "sophistication_level": "high",
        },
    )
    register_component(
        "processor",
        "decoder_stable_processor",
        StableDecoderProcessor,
        metadata={
            "source": "layers.modular.decoder.stable_decoder",
            "features": ["transformer", "fixed_enhanced"],
            "domain": "decoder",
            "sophistication_level": "medium",
            "deprecated": True,
            "note": "Use decoder_enhanced_processor unless fixed-weight variant is required.",
        },
    )
