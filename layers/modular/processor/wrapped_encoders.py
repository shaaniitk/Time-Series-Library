"""Wrapped encoder processors registered under unified registry (no utils imports)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch
import torch.nn as nn

from ..base_interfaces import BaseProcessor
from ..core.registry import register_component


@dataclass
class ProcessorConfig:
    d_model: int = 16
    dropout: float = 0.0
    seq_len: int = 8
    pred_len: int = 8
    label_len: int = 0
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EncoderProcessorConfig(ProcessorConfig):
    return_attention: bool = False
    n_levels: int = 3
    share_weights: bool = False


class _BaseEncoderProcessor(BaseProcessor):
    def __init__(self, config: EncoderProcessorConfig):
        super().__init__(config)
        self.config = config
        self._d_model = config.d_model

    def get_output_dim(self) -> int:
        return self._d_model

    def forward(self, embedded_input: torch.Tensor, backbone_output: Optional[torch.Tensor] = None, target_length: Optional[int] = None, **kwargs) -> torch.Tensor:
        return self.process_sequence(embedded_input, backbone_output, target_length or embedded_input.size(1), **kwargs)


class _LegacyDecompAdapter(nn.Module):
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


class StandardEncoderProcessor(_BaseEncoderProcessor):
    def __init__(self, config: EncoderProcessorConfig):
        super().__init__(config)
        from layers.Autoformer_EncDec import Encoder as AutoformerEncoder
        from layers.modular.layers.standard_layers import StandardEncoderLayer
        attention_comp = config.custom_params.get('attention_component')
        decomp_comp = config.custom_params.get('decomposition_component')
        if isinstance(decomp_comp, BaseProcessor) or hasattr(decomp_comp, 'process_sequence'):
            k = getattr(getattr(decomp_comp, 'config', object()), 'kernel_size', 25)
            decomp_comp = _LegacyDecompAdapter(decomp_comp, kernel_size=k)
        e_layers = config.custom_params.get('e_layers', 2)
        d_ff = config.custom_params.get('d_ff', 4 * config.d_model)
        activation = config.custom_params.get('activation', 'relu')
        dropout = config.dropout
        layers = [
            StandardEncoderLayer(attention_comp, decomp_comp, config.d_model, d_ff, dropout=dropout, activation=activation)
            for _ in range(e_layers)
        ]
        self.encoder = AutoformerEncoder(layers, conv_layers=config.custom_params.get('conv_layers'), norm_layer=config.custom_params.get('norm_layer'))

    def process_sequence(self, embedded_input: torch.Tensor, backbone_output: Optional[torch.Tensor], target_length: int, **kwargs) -> torch.Tensor:
        out, _ = self.encoder(embedded_input, attn_mask=kwargs.get('attn_mask'))
        if out.size(1) != target_length:
            out = nn.functional.interpolate(out.transpose(1, 2), size=target_length, mode='linear', align_corners=False).transpose(1, 2)
        return out

    def get_processor_type(self) -> str:
        return "encoder_standard"


class EnhancedEncoderProcessor(_BaseEncoderProcessor):
    def __init__(self, config: EncoderProcessorConfig):
        super().__init__(config)
        from layers.Autoformer_EncDec import Encoder as AutoformerEncoder
        from layers.modular.layers.enhanced_layers import EnhancedEncoderLayer
        attention_comp = config.custom_params.get('attention_component')
        decomp_comp = config.custom_params.get('decomposition_component')
        if isinstance(decomp_comp, BaseProcessor) or hasattr(decomp_comp, 'process_sequence'):
            k = getattr(getattr(decomp_comp, 'config', object()), 'kernel_size', 25)
            decomp_comp = _LegacyDecompAdapter(decomp_comp, kernel_size=k)
        e_layers = config.custom_params.get('e_layers', 2)
        d_ff = config.custom_params.get('d_ff', 4 * config.d_model)
        activation = config.custom_params.get('activation', 'relu')
        dropout = config.dropout
        layers = [
            EnhancedEncoderLayer(attention_comp, decomp_comp, config.d_model, d_ff, dropout=dropout, activation=activation)
            for _ in range(e_layers)
        ]
        self.encoder = AutoformerEncoder(layers, conv_layers=config.custom_params.get('conv_layers'), norm_layer=config.custom_params.get('norm_layer'))

    def process_sequence(self, embedded_input: torch.Tensor, backbone_output: Optional[torch.Tensor], target_length: int, **kwargs) -> torch.Tensor:
        out, _ = self.encoder(embedded_input, attn_mask=kwargs.get('attn_mask'))
        if out.size(1) != target_length:
            out = nn.functional.interpolate(out.transpose(1, 2), size=target_length, mode='linear', align_corners=False).transpose(1, 2)
        return out

    def get_processor_type(self) -> str:
        return "encoder_enhanced"


class StableEncoderProcessor(_BaseEncoderProcessor):
    def __init__(self, config: EncoderProcessorConfig):
        super().__init__(config)
        from layers.Autoformer_EncDec import Encoder as AutoformerEncoder
        from layers.modular.layers.enhanced_layers import EnhancedEncoderLayer
        attention_comp = config.custom_params.get('attention_component')
        decomp_comp = config.custom_params.get('decomposition_component')
        if isinstance(decomp_comp, BaseProcessor) or hasattr(decomp_comp, 'process_sequence'):
            k = getattr(getattr(decomp_comp, 'config', object()), 'kernel_size', 25)
            decomp_comp = _LegacyDecompAdapter(decomp_comp, kernel_size=k)
        e_layers = config.custom_params.get('e_layers', 2)
        d_ff = config.custom_params.get('d_ff', 4 * config.d_model)
        activation = config.custom_params.get('activation', 'relu')
        dropout = config.dropout
        layers = [
            EnhancedEncoderLayer(attention_comp, decomp_comp, config.d_model, d_ff, dropout=dropout, activation=activation)
            for _ in range(e_layers)
        ]
        self.encoder = AutoformerEncoder(layers, conv_layers=config.custom_params.get('conv_layers'), norm_layer=config.custom_params.get('norm_layer'))

    def process_sequence(self, embedded_input: torch.Tensor, backbone_output: Optional[torch.Tensor], target_length: int, **kwargs) -> torch.Tensor:
        out, _ = self.encoder(embedded_input, attn_mask=kwargs.get('attn_mask'))
        if out.size(1) != target_length:
            out = nn.functional.interpolate(out.transpose(1, 2), size=target_length, mode='linear', align_corners=False).transpose(1, 2)
        return out

    def get_processor_type(self) -> str:
        return "encoder_stable"


class HierarchicalEncoderProcessor(_BaseEncoderProcessor):
    def __init__(self, config: EncoderProcessorConfig):
        super().__init__(config)
        from layers.modular.encoder.hierarchical_encoder import HierarchicalEncoder as LegacyHierarchicalEncoder
        self.encoder = LegacyHierarchicalEncoder(
            e_layers=config.custom_params.get('e_layers', 2),
            d_model=config.d_model,
            n_heads=config.custom_params.get('n_heads', 8),
            d_ff=config.custom_params.get('d_ff', 4 * config.d_model),
            dropout=config.dropout,
            activation=config.custom_params.get('activation', 'relu'),
            attention_type=config.custom_params.get('attention_type', 'autocorrelation_layer'),
            decomp_type=config.custom_params.get('decomp_type', 'series_decomp'),
            decomp_params=config.custom_params.get('decomp_params', {}),
            n_levels=config.n_levels,
            share_weights=config.share_weights,
        )

    def process_sequence(self, embedded_input: torch.Tensor, backbone_output: Optional[torch.Tensor], target_length: int, **kwargs) -> torch.Tensor:
        out, _ = self.encoder(embedded_input, attn_mask=kwargs.get('attn_mask'))
        if isinstance(out, list) and len(out) > 0:
            out = out[-1]
        if out.size(1) != target_length:
            out = nn.functional.interpolate(out.transpose(1, 2), size=target_length, mode='linear', align_corners=False).transpose(1, 2)
        return out

    def get_processor_type(self) -> str:
        return "encoder_hierarchical"


def register_layers_encoders() -> None:
    register_component("processor", "encoder_standard_processor", StandardEncoderProcessor, metadata={
        "source": "layers.modular.encoder.standard_encoder",
        "features": ["transformer", "autoformer_layers"],
        "domain": "encoder",
        "sophistication_level": "medium",
    })
    register_component("processor", "encoder_enhanced_processor", EnhancedEncoderProcessor, metadata={
        "source": "layers.modular.encoder.enhanced_encoder",
        "features": ["transformer", "gated_ffn", "scaled_attn"],
        "domain": "encoder",
        "sophistication_level": "high",
    })
    register_component("processor", "encoder_stable_processor", StableEncoderProcessor, metadata={
        "source": "layers.modular.encoder.stable_encoder",
        "features": ["transformer", "fixed_enhanced"],
        "domain": "encoder",
        "sophistication_level": "medium",
        "deprecated": True,
        "note": "Use encoder_enhanced_processor unless fixed-weight variant is required.",
    })
    register_component("processor", "encoder_hierarchical_processor", HierarchicalEncoderProcessor, metadata={
        "source": "layers.modular.encoder.hierarchical_encoder",
        "features": ["hierarchical", "multi_resolution", "cross_scale"],
        "domain": "encoder",
        "sophistication_level": "high",
    })
