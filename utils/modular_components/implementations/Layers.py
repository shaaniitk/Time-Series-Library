"""
UNIFIED LAYER COMPONENTS
Complete layer implementations including normalization, residuals, and feed-forward
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
import logging

# Import base interfaces from new framework
from ..base_interfaces import BaseComponent
from ..config_schemas import ComponentConfig

# Import attention components
from .attention_clean import AutoCorrelationAttention, MultiHeadAttention

logger = logging.getLogger(__name__)


class AutoCorrelationLayer(BaseComponent):
    """
    Complete AutoCorrelation layer with normalization, residuals, and feed-forward
    
    This is a complete transformer layer that uses AutoCorrelation attention
    instead of standard multi-head attention. Includes all the standard
    transformer layer components.
    """
    
    def __init__(self, d_model=512, n_heads=8, factor=1, d_ff=2048, dropout=0.1, activation='relu'):
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
                self.factor = factor
                self.d_ff = d_ff
                self.dropout = dropout
                self.activation = activation
        super().__init__(Config())
        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor
        self.d_ff = d_ff
        self.dropout = dropout
        
        # AutoCorrelation attention mechanism
        self.autocorr_attention = AutoCorrelationAttention(
            d_model=d_model, 
            n_heads=n_heads, 
            factor=factor, 
            dropout=dropout
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        logger.info(f"AutoCorrelationLayer initialized: d_model={d_model}, factor={factor}")
    
    def forward(self, x, attn_mask=None):
        """
        Forward pass for AutoCorrelation layer
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            attn_mask: Attention mask (optional)
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # AutoCorrelation attention with residual connection
        residual = x
        x = self.norm1(x)
        
        # For autocorrelation, we typically use x as queries, keys, and values
        attn_output, _ = self.autocorr_attention(x, x, x, attn_mask)
        x = residual + self.dropout_layer(attn_output)
        
        # Feed-forward with residual connection
        residual = x
        x = self.norm2(x)
        ff_output = self.feed_forward(x)
        x = residual + ff_output
        
        return x
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get layer capabilities"""
        return {
            'type': 'autocorrelation_layer',
            'supports_self_attention': True,
            'supports_temporal_patterns': True,
            'supports_seasonal_decomposition': True,
            'factor': self.factor
        }
    def get_output_dim(self) -> int:
        """Return the output dimension of this layer"""
        return self.d_model
    def get_output_dim(self) -> int:
        """Return the output dimension of this layer"""
        return self.d_model


class MultiHeadAttentionLayer(BaseComponent):
    """
    Complete Multi-Head attention layer with normalization, residuals, and feed-forward
    
    Standard transformer layer using multi-head attention.
    """
    
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1, activation='relu'):
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
                self.d_ff = d_ff
                self.dropout = dropout
                self.activation = activation
        super().__init__(Config())
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        
        # Multi-head attention mechanism
        self.multihead_attention = MultiHeadAttention(
            d_model=d_model, 
            n_heads=n_heads, 
            dropout=dropout
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        logger.info(f"MultiHeadAttentionLayer initialized: d_model={d_model}, n_heads={n_heads}")
    
    def forward(self, x, attn_mask=None):
        """
        Forward pass for Multi-Head attention layer
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            attn_mask: Attention mask (optional)
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Multi-head attention with residual connection
        residual = x
        x = self.norm1(x)
        
        attn_output, _ = self.multihead_attention(x, x, x, attn_mask)
        x = residual + self.dropout_layer(attn_output)
        
        # Feed-forward with residual connection
        residual = x
        x = self.norm2(x)
        ff_output = self.feed_forward(x)
        x = residual + ff_output
        
        return x
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get layer capabilities"""
        return {
            'type': 'multihead_attention_layer',
            'supports_self_attention': True,
            'supports_cross_attention': True,
            'n_heads': self.n_heads
        }
    def get_output_dim(self) -> int:
        """Return the output dimension of this layer"""
        return self.d_model

class FeedForwardLayer(BaseComponent):
    """
    Stand-alone feed-forward layer with normalization and residual connection
    """
    
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1, activation='relu'):
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.d_ff = d_ff
                self.dropout = dropout
                self.activation = activation
        super().__init__(Config())
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        logger.info(f"FeedForwardLayer initialized: d_model={d_model}, d_ff={d_ff}")
    
    def forward(self, x):
        """
        Forward pass for feed-forward layer
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        residual = x
        x = self.norm(x)
        ff_output = self.feed_forward(x)
        x = residual + ff_output
        
        return x
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get layer capabilities"""
        return {
            'type': 'feedforward_layer',
            'supports_nonlinearity': True,
            'd_ff': self.d_ff
        }
    def get_output_dim(self) -> int:
        """Return the output dimension of this layer"""
        return self.d_model

class PositionalEncodingLayer(BaseComponent):
    """
    Positional encoding layer for transformer models
    """
    
    def __init__(self, d_model=512, max_seq_len=5000, dropout=0.1):
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.max_seq_len = max_seq_len
                self.dropout = dropout
        super().__init__(Config())
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
        logger.info(f"PositionalEncodingLayer initialized: d_model={d_model}, max_len={max_seq_len}")
    
    def forward(self, x):
        """
        Add positional encoding to input
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor with positional encoding [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].transpose(0, 1)
        return self.dropout(x)
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get layer capabilities"""
        return {
            'type': 'positional_encoding',
            'max_seq_len': self.max_seq_len,
            'supports_sinusoidal_encoding': True
        }
    def get_output_dim(self) -> int:
        """Return the output dimension of this layer"""
        return self.d_model

# Registry for layer components
LAYER_REGISTRY = {
    'autocorrelation_layer': AutoCorrelationLayer,
    'multihead_attention_layer': MultiHeadAttentionLayer,
    'feedforward_layer': FeedForwardLayer,
    'positional_encoding': PositionalEncodingLayer,
}


def get_layer_component(name: str, config: ComponentConfig = None, **kwargs):
    """Get layer component by name"""
    if name not in LAYER_REGISTRY:
        raise ValueError(f"Unknown layer component: {name}")
    
    component_class = LAYER_REGISTRY[name]
    
    if config is not None:
        # Use config parameters
        params = {
            'd_model': getattr(config, 'd_model', 512),
            'dropout': getattr(config, 'dropout', 0.1),
            **getattr(config, 'custom_params', {}),
            **kwargs
        }
    else:
        params = kwargs
    
    return component_class(**params)


def register_layer_components(registry):
    """Register all layer components with the registry"""
    for name, component_class in LAYER_REGISTRY.items():
        registry.register('layer', name, component_class)
    
    logger.info(f"Registered {len(LAYER_REGISTRY)} layer components")


def list_layer_components():
    """List all available layer components"""
    return list(LAYER_REGISTRY.keys())
