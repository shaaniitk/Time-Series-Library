"""
UNIFIED ENCODER COMPONENTS
All encoder mechanisms in one place - no duplicates, clean modular structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Dict, Any, List
import logging

# Import base interfaces and unified components
from ..base_interfaces import BaseComponent
from ..config_schemas import ComponentConfig
from .attentions_unified import get_attention_component
from .decomposition_unified import get_decomposition_component

logger = logging.getLogger(__name__)


# =============================================================================
# BASE ENCODER
# =============================================================================

class BaseEncoder(BaseComponent):
    """Base class for all encoder components"""
    
    def __init__(self, config: ComponentConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.dropout = config.dropout
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [batch_size, seq_len, d_model]
        attn_mask: [batch_size, seq_len, seq_len] or None
        """
        raise NotImplementedError


# =============================================================================
# TRANSFORMER ENCODER
# =============================================================================

class TransformerEncoder(BaseEncoder):
    """Standard Transformer encoder layer"""
    
    def __init__(self, config: ComponentConfig):
        super().__init__(config)
        self.n_heads = config.custom_params.get('n_heads', 8)
        self.d_ff = config.custom_params.get('d_ff', self.d_model * 4)
        
        # Attention mechanism
        attention_config = ComponentConfig(
            component_name='multihead',
            d_model=self.d_model,
            dropout=self.dropout,
            custom_params={'n_heads': self.n_heads}
        )
        self.self_attention = get_attention_component('multihead', attention_config)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_ff, self.d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_out, _ = self.self_attention(x, x, x, attn_mask)
        x = self.norm1(x + self.dropout_layer(attn_out))
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout_layer(ff_out))
        
        return x


# =============================================================================
# AUTOFORMER ENCODER
# =============================================================================

class AutoformerEncoder(BaseEncoder):
    """Autoformer encoder with decomposition and autocorrelation"""
    
    def __init__(self, config: ComponentConfig):
        super().__init__(config)
        self.n_heads = config.custom_params.get('n_heads', 8)
        self.d_ff = config.custom_params.get('d_ff', self.d_model * 4)
        self.moving_avg_window = config.custom_params.get('moving_avg_window', 25)
        
        # Decomposition
        decomp_config = ComponentConfig(
            component_name='series',
            d_model=self.d_model,
            custom_params={'kernel_size': self.moving_avg_window}
        )
        self.decomposition = get_decomposition_component('series', decomp_config)
        
        # Auto-correlation attention
        attention_config = ComponentConfig(
            component_name='autocorrelation',
            d_model=self.d_model,
            dropout=self.dropout,
            custom_params={'n_heads': self.n_heads}
        )
        self.attention = get_attention_component('autocorrelation', attention_config)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_ff, self.d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Decomposition
        trend, seasonal = self.decomposition(x)
        
        # Auto-correlation attention on seasonal component
        seasonal_attn, _ = self.attention(seasonal, seasonal, seasonal, attn_mask)
        seasonal = self.norm1(seasonal + self.dropout_layer(seasonal_attn))
        
        # Feed-forward
        seasonal_ff = self.feed_forward(seasonal)
        seasonal = self.norm2(seasonal + self.dropout_layer(seasonal_ff))
        
        # Combine trend and processed seasonal
        output = trend + seasonal
        
        return output


# =============================================================================
# FOURIER ENCODER
# =============================================================================

class FourierEncoder(BaseEncoder):
    """Encoder with Fourier-based processing"""
    
    def __init__(self, config: ComponentConfig):
        super().__init__(config)
        self.seq_len = config.custom_params.get('seq_len', 96)
        self.modes = config.custom_params.get('modes', 32)
        
        # Fourier attention
        attention_config = ComponentConfig(
            component_name='fourier',
            d_model=self.d_model,
            dropout=self.dropout,
            custom_params={'seq_len': self.seq_len}
        )
        self.fourier_attention = get_attention_component('fourier', attention_config)
        
        # Fourier block for direct frequency processing
        fourier_config = ComponentConfig(
            component_name='fourier_block',
            d_model=self.d_model,
            custom_params={'seq_len': self.seq_len, 'modes': self.modes}
        )
        self.fourier_block = get_attention_component('fourier_block', fourier_config)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Fourier attention
        fourier_attn, _ = self.fourier_attention(x, x, x, attn_mask)
        x = self.norm1(x + self.dropout_layer(fourier_attn))
        
        # Direct Fourier processing
        fourier_out, _ = self.fourier_block(x, x, x)
        x = self.norm2(x + self.dropout_layer(fourier_out))
        
        return x


# =============================================================================
# WAVELET ENCODER
# =============================================================================

class WaveletEncoder(BaseEncoder):
    """Encoder with wavelet-based multi-scale processing"""
    
    def __init__(self, config: ComponentConfig):
        super().__init__(config)
        self.levels = config.custom_params.get('levels', 3)
        
        # Wavelet decomposition
        decomp_config = ComponentConfig(
            component_name='wavelet',
            d_model=self.d_model,
            custom_params={'levels': self.levels}
        )
        self.wavelet_decomp = get_decomposition_component('wavelet', decomp_config)
        
        # Wavelet attention
        attention_config = ComponentConfig(
            component_name='wavelet',
            d_model=self.d_model,
            dropout=self.dropout,
            custom_params={'levels': self.levels}
        )
        self.wavelet_attention = get_attention_component('wavelet', attention_config)
        
        # Processing layers
        self.trend_processor = nn.Linear(self.d_model, self.d_model)
        self.seasonal_processor = nn.Linear(self.d_model, self.d_model)
        
        # Layer normalization
        self.norm = nn.LayerNorm(self.d_model)
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Wavelet decomposition
        trend, seasonal = self.wavelet_decomp(x)
        
        # Process components separately
        trend_processed = self.trend_processor(trend)
        
        # Apply wavelet attention to seasonal component
        seasonal_attn, _ = self.wavelet_attention(seasonal, seasonal, seasonal, attn_mask)
        seasonal_processed = self.seasonal_processor(seasonal_attn)
        
        # Combine processed components
        output = trend_processed + seasonal_processed
        output = self.norm(output)
        
        return output


# =============================================================================
# ADAPTIVE ENCODER
# =============================================================================

class AdaptiveEncoder(BaseEncoder):
    """Adaptive encoder that selects best processing method"""
    
    def __init__(self, config: ComponentConfig):
        super().__init__(config)
        
        # Multiple encoder types
        self.encoders = nn.ModuleList([
            TransformerEncoder(config),
            AutoformerEncoder(config),
            FourierEncoder(config)
        ])
        
        # Encoder selection network
        self.encoder_selector = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.encoders)),
            nn.Softmax(dim=-1)
        )
        
        # Final processing
        self.output_projection = nn.Linear(self.d_model, self.d_model)
        
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Select encoder based on input characteristics
        x_summary = x.mean(dim=1)  # [batch_size, d_model]
        encoder_weights = self.encoder_selector(x_summary)  # [batch_size, num_encoders]
        
        # Apply each encoder and combine
        output = torch.zeros_like(x)
        
        for i, encoder in enumerate(self.encoders):
            encoder_out = encoder(x, attn_mask)
            weight = encoder_weights[:, i:i+1, None]  # [batch_size, 1, 1]
            output += weight * encoder_out
        
        # Final projection
        output = self.output_projection(output)
        
        return output


# =============================================================================
# HIERARCHICAL ENCODER
# =============================================================================

class HierarchicalEncoder(BaseEncoder):
    """Hierarchical encoder processing at multiple time scales"""
    
    def __init__(self, config: ComponentConfig):
        super().__init__(config)
        self.scales = config.custom_params.get('scales', [1, 2, 4])
        
        # Encoder for each scale
        self.scale_encoders = nn.ModuleList([
            TransformerEncoder(config) for _ in self.scales
        ])
        
        # Scale fusion
        self.scale_fusion = nn.Linear(len(self.scales) * self.d_model, self.d_model)
        
        # Layer normalization
        self.norm = nn.LayerNorm(self.d_model)
        
    def _downsample(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        """Downsample input by given scale"""
        if scale == 1:
            return x
        
        # Simple average pooling for downsampling
        batch_size, seq_len, d_model = x.shape
        new_len = seq_len // scale
        
        if new_len > 0:
            x_reshaped = x[:, :new_len * scale].view(batch_size, new_len, scale, d_model)
            return x_reshaped.mean(dim=2)
        else:
            return x[:, :1]  # Fallback for very short sequences
    
    def _upsample(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """Upsample to target length"""
        if x.shape[1] == target_len:
            return x
        
        return F.interpolate(
            x.transpose(1, 2), 
            size=target_len, 
            mode='linear', 
            align_corners=False
        ).transpose(1, 2)
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        scale_outputs = []
        
        # Process at each scale
        for i, scale in enumerate(self.scales):
            # Downsample input
            x_downsampled = self._downsample(x, scale)
            
            # Create appropriate attention mask
            scale_mask = None
            if attn_mask is not None and scale > 1:
                mask_len = x_downsampled.shape[1]
                scale_mask = attn_mask[:, :mask_len, :mask_len]
            elif attn_mask is not None:
                scale_mask = attn_mask
            
            # Process with encoder
            processed = self.scale_encoders[i](x_downsampled, scale_mask)
            
            # Upsample back to original length
            processed_upsampled = self._upsample(processed, seq_len)
            scale_outputs.append(processed_upsampled)
        
        # Concatenate and fuse scale outputs
        fused_input = torch.cat(scale_outputs, dim=-1)
        output = self.scale_fusion(fused_input)
        output = self.norm(output)
        
        return output


# =============================================================================
# COMPONENT REGISTRY
# =============================================================================

class EncoderRegistry:
    """Registry for all encoder components"""
    
    _components = {
        'transformer': TransformerEncoder,
        'autoformer': AutoformerEncoder,
        'fourier': FourierEncoder,
        'wavelet': WaveletEncoder,
        'adaptive': AdaptiveEncoder,
        'hierarchical': HierarchicalEncoder,
    }
    
    @classmethod
    def register(cls, name: str, component_class):
        """Register a new encoder component"""
        cls._components[name] = component_class
        logger.info(f"Registered encoder component: {name}")
    
    @classmethod
    def create(cls, name: str, config: ComponentConfig):
        """Create an encoder component by name"""
        if name not in cls._components:
            raise ValueError(f"Unknown encoder component: {name}")
        
        component_class = cls._components[name]
        return component_class(config)
    
    @classmethod
    def list_components(cls):
        """List all available encoder components"""
        return list(cls._components.keys())


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_encoder_component(name: str, config: ComponentConfig = None, **kwargs):
    """Factory function to create encoder components"""
    if config is None:
        # Create config from kwargs for backward compatibility
        config = ComponentConfig(
            component_name=name,
            d_model=kwargs.get('d_model', 512),
            dropout=kwargs.get('dropout', 0.1),
            custom_params=kwargs
        )
    
    return EncoderRegistry.create(name, config)


def register_encoder_components(registry):
    """Register all encoder components with the main registry"""
    for name, component_class in EncoderRegistry._components.items():
        registry.register('encoder', name, component_class)
    
    logger.info(f"Registered {len(EncoderRegistry._components)} encoder components")
