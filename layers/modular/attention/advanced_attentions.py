"""
Advanced Attention Mechanisms Integration

Integrates existing optimized attention implementations into the modular framework.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass

from ..base_interfaces import BaseAttention
from .attentions import AttentionConfig

logger = logging.getLogger(__name__)


@dataclass
class OptimizedAttentionConfig(AttentionConfig):
    """Configuration for optimized attention mechanisms"""
    max_seq_len: int = 1024
    use_mixed_precision: bool = True
    chunk_size: Optional[int] = None
    memory_efficient: bool = True


@dataclass
class AdaptiveAttentionConfig(AttentionConfig):
    """Configuration for adaptive attention mechanisms"""
    adaptive_k: bool = True
    multi_scale: bool = True
    scales: list = None
    min_scale: int = 1
    max_scale: int = 4


class OptimizedAutoCorrelationAttention(BaseAttention):
    """Wraps existing OptimizedAutoCorrelation for modular framework"""
    
    def __init__(self, config: OptimizedAttentionConfig):
        super().__init__(config)
        
        try:
            from layers.AutoCorrelation_Optimized import OptimizedAutoCorrelation
            
            self.attention = OptimizedAutoCorrelation(
                mask_flag=getattr(config, 'mask_flag', True),
                factor=getattr(config, 'factor', 1),
                attention_dropout=getattr(config, 'dropout', 0.1),
                output_attention=getattr(config, 'output_attention', False),
                max_seq_len=getattr(config, 'max_seq_len', 1024)
            )
            
            self.d_model = config.d_model
            self.num_heads = config.num_heads
            
            logger.info(f"OptimizedAutoCorrelationAttention initialized: d_model={self.d_model}, heads={self.num_heads}")
            
        except ImportError as e:
            logger.warning(f"Could not import OptimizedAutoCorrelation: {e}")
            # Fallback to standard attention
            from .attentions import MultiHeadAttention
            self.attention = MultiHeadAttention(config)
    
    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.attention(queries, keys, values, attn_mask)
    
    def get_attention_type(self) -> str:
        return "optimized_autocorrelation"
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            'type': 'optimized_autocorrelation',
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'supports_long_sequences': True,
            'memory_optimized': True,
            'mixed_precision': True,
            'chunked_processing': True
        }


class AdaptiveAutoCorrelationAttention(BaseAttention):
    """Wraps existing AdaptiveAutoCorrelation for modular framework"""
    
    def __init__(self, config: AdaptiveAttentionConfig):
        super().__init__(config)
        
        try:
            from layers.EnhancedAutoCorrelation import AdaptiveAutoCorrelation
            
            scales = config.scales if config.scales else [1, 2, 4]
            
            self.attention = AdaptiveAutoCorrelation(
                mask_flag=getattr(config, 'mask_flag', True),
                factor=getattr(config, 'factor', 1),
                attention_dropout=getattr(config, 'dropout', 0.1),
                output_attention=getattr(config, 'output_attention', False),
                adaptive_k=getattr(config, 'adaptive_k', True),
                multi_scale=getattr(config, 'multi_scale', True),
                scales=scales
            )
            
            self.d_model = config.d_model
            self.num_heads = config.num_heads
            
            logger.info(f"AdaptiveAutoCorrelationAttention initialized: scales={scales}")
            
        except ImportError as e:
            logger.warning(f"Could not import AdaptiveAutoCorrelation: {e}")
            # Fallback to optimized version
            self.attention = OptimizedAutoCorrelationAttention(config)
    
    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.attention(queries, keys, values, attn_mask)
    
    def get_attention_type(self) -> str:
        return "adaptive_autocorrelation"
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            'type': 'adaptive_autocorrelation',
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'adaptive_k': True,
            'multi_scale': True,
            'scales': getattr(self.attention, 'scales', [1, 2, 4])
        }


class EnhancedAutoCorrelationLayer(BaseAttention):
    """Complete attention layer wrapper with projections"""
    
    def __init__(self, config: AdaptiveAttentionConfig):
        super().__init__(config)
        
        try:
            from layers.EnhancedAutoCorrelation import AdaptiveAutoCorrelationLayer
            
            # Create the adaptive autocorrelation
            from layers.EnhancedAutoCorrelation import AdaptiveAutoCorrelation
            
            scales = config.scales if config.scales else [1, 2, 4]
            correlation = AdaptiveAutoCorrelation(
                mask_flag=getattr(config, 'mask_flag', True),
                factor=getattr(config, 'factor', 1),
                attention_dropout=getattr(config, 'dropout', 0.1),
                output_attention=getattr(config, 'output_attention', False),
                adaptive_k=getattr(config, 'adaptive_k', True),
                multi_scale=getattr(config, 'multi_scale', True),
                scales=scales
            )
            
            # Create the complete layer
            self.attention_layer = AdaptiveAutoCorrelationLayer(
                correlation=correlation,
                d_model=config.d_model,
                n_heads=config.num_heads
            )
            
            self.d_model = config.d_model
            self.num_heads = config.num_heads
            
            logger.info(f"EnhancedAutoCorrelationLayer initialized")
            
        except ImportError as e:
            logger.warning(f"Could not import AdaptiveAutoCorrelationLayer: {e}")
            # Fallback to standard attention
            from .attentions import MultiHeadAttention
            self.attention_layer = MultiHeadAttention(config)
    
    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        # The enhanced layer expects 3D inputs [B, L, D] 
        if len(queries.shape) == 4:  # [B, L, H, E]
            B, L, H, E = queries.shape
            queries = queries.view(B, L, H * E)
            keys = keys.view(B, keys.size(1), H * E) 
            values = values.view(B, values.size(1), H * E)
        
        return self.attention_layer(queries, keys, values, attn_mask)
    
    def get_attention_type(self) -> str:
        return "enhanced_autocorrelation_layer"
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            'type': 'enhanced_autocorrelation_layer',
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'has_projections': True,
            'adaptive_k': True,
            'multi_scale': True
        }


class MemoryEfficientAttention(BaseAttention):
    """Memory-efficient attention with gradient checkpointing"""
    
    def __init__(self, config: OptimizedAttentionConfig):
        super().__init__(config)
        
        # Choose best available implementation
        try:
            from layers.AutoCorrelation_Optimized import OptimizedAutoCorrelation
            self.attention = OptimizedAutoCorrelation(
                mask_flag=getattr(config, 'mask_flag', True),
                factor=getattr(config, 'factor', 1),
                attention_dropout=getattr(config, 'dropout', 0.1),
                output_attention=getattr(config, 'output_attention', False),
                max_seq_len=getattr(config, 'max_seq_len', 1024)
            )
            self.implementation = "optimized"
        except ImportError:
            try:
                from layers.AutoCorrelation import AutoCorrelation
                self.attention = AutoCorrelation(
                    mask_flag=getattr(config, 'mask_flag', True),
                    factor=getattr(config, 'factor', 1),
                    attention_dropout=getattr(config, 'dropout', 0.1),
                    output_attention=getattr(config, 'output_attention', False)
                )
                self.implementation = "standard"
            except ImportError:
                # Ultimate fallback
                self.attention = nn.MultiheadAttention(
                    embed_dim=config.d_model,
                    num_heads=config.num_heads,
                    dropout=getattr(config, 'dropout', 0.1),
                    batch_first=True
                )
                self.implementation = "pytorch"
        
        self.use_checkpointing = getattr(config, 'memory_efficient', True)
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        
        logger.info(f"MemoryEfficientAttention initialized with {self.implementation} implementation")
    
    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        if self.use_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, queries, keys, values, attn_mask, use_reentrant=False
            )
        else:
            return self._forward_impl(queries, keys, values, attn_mask)
    
    def _forward_impl(self, queries, keys, values, attn_mask):
        if self.implementation == "pytorch":
            # PyTorch multihead attention expects [L, B, D]
            if len(queries.shape) == 4:  # [B, L, H, E] -> [B, L, D]
                B, L, H, E = queries.shape
                queries = queries.view(B, L, H * E)
                keys = keys.view(B, keys.size(1), H * E)
                values = values.view(B, values.size(1), H * E)
            
            # Transpose for PyTorch attention
            q = queries.transpose(0, 1)  # [L, B, D]
            k = keys.transpose(0, 1)
            v = values.transpose(0, 1)
            
            out, attn_weights = self.attention(q, k, v, attn_mask=attn_mask)
            out = out.transpose(0, 1)  # Back to [B, L, D]
            return out, attn_weights
        else:
            return self.attention(queries, keys, values, attn_mask)
    
    def get_attention_type(self) -> str:
        return f"memory_efficient_{self.implementation}"
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            'type': f'memory_efficient_{self.implementation}',
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'memory_efficient': True,
            'gradient_checkpointing': self.use_checkpointing,
            'implementation': self.implementation
        }


def get_attention_layer_from_config(attention_type: str, config: Dict[str, Any]) -> BaseAttention:
    """Factory function to create attention layers from existing implementations"""
    
    if attention_type == 'optimized_autocorrelation':
        return OptimizedAutoCorrelationAttention(OptimizedAttentionConfig(**config))
    
    elif attention_type == 'adaptive_autocorrelation':
        return AdaptiveAutoCorrelationAttention(AdaptiveAttentionConfig(**config))
    
    elif attention_type == 'enhanced_autocorrelation':
        return EnhancedAutoCorrelationLayer(AdaptiveAttentionConfig(**config))
    
    elif attention_type == 'memory_efficient':
        return MemoryEfficientAttention(OptimizedAttentionConfig(**config))
    
    else:
        # Fallback to standard attention
        logger.warning(f"Unknown attention type {attention_type}, falling back to standard")
        from .attentions import MultiHeadAttention, AttentionConfig
        return MultiHeadAttention(AttentionConfig(**config))
