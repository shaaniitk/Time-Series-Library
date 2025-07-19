"""
Concrete Component Implementations for GCLI Architecture

This module implements concrete components that inherit from the ModularComponent
base class and can be used with the "dumb assembler" pattern.
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Any, Optional, Tuple

from configs.modular_components import (
    ModularComponent, AttentionComponent, DecompositionComponent,
    EncoderComponent, DecoderComponent, SamplingComponent,
    OutputHeadComponent, LossComponent, ComponentMetadata,
    component_registry
)
from configs.schemas import (
    ComponentType, AttentionConfig, DecompositionConfig,
    EncoderConfig, DecoderConfig, SamplingConfig,
    OutputHeadConfig, LossConfig
)


class AutoCorrelationAttention(AttentionComponent):
    """AutoCorrelation attention component"""
    
    def __init__(self, config: AttentionConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="AutoCorrelation",
            component_type=ComponentType.AUTOCORRELATION,
            required_params=['d_model', 'n_heads'],
            optional_params=['dropout', 'factor'],
            description="AutoCorrelation mechanism for time series"
        )
    
    def _initialize_component(self, **kwargs):
        self.d_model = self.config.d_model
        self.n_heads = self.config.n_heads
        self.dropout = nn.Dropout(self.config.dropout)
        self.factor = self.config.factor
        
        # Linear projections
        self.query_projection = nn.Linear(self.d_model, self.d_model)
        self.key_projection = nn.Linear(self.d_model, self.d_model)
        self.value_projection = nn.Linear(self.d_model, self.d_model)
        self.out_projection = nn.Linear(self.d_model, self.d_model)
    
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        
        # Simplified autocorrelation (for demonstration)
        scale = 1. / math.sqrt(queries.shape[-1])
        scores = torch.einsum("blhd,bshd->bhls", queries, keys) * scale
        
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.einsum("bhls,bshd->blhd", attn, values)
        out = out.contiguous().view(B, L, -1)
        
        return self.out_projection(out), attn


class LearnableDecomposition(DecompositionComponent):
    """Learnable decomposition component"""
    
    def __init__(self, config: DecompositionConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="LearnableDecomposition",
            component_type=ComponentType.LEARNABLE_DECOMP,
            required_params=['input_dim'],
            optional_params=['kernel_size'],
            description="Learnable decomposition for trend-seasonal separation"
        )
    
    def _initialize_component(self, **kwargs):
        # Get input dimension from multiple sources
        decomp_params = getattr(self.config, 'decomposition_params', {})
        self.input_dim = (
            decomp_params.get('input_dim') or 
            getattr(self.config, 'input_dim', None) or
            getattr(self.config, 'd_model', 512)  # Use d_model as fallback
        )
        self.kernel_size = decomp_params.get('kernel_size', getattr(self.config, 'kernel_size', 25))
        
        # Simple learnable trend extraction
        self.trend_conv = nn.Conv1d(self.input_dim, self.input_dim, self.kernel_size, padding=self.kernel_size//2)
        
    def forward(self, x):
        # x: [B, L, D]
        x_permuted = x.permute(0, 2, 1)  # [B, D, L]
        x_trend = self.trend_conv(x_permuted).permute(0, 2, 1)  # [B, L, D]
        x_seasonal = x - x_trend
        return x_seasonal, x_trend


class MovingAverageDecomposition(DecompositionComponent):
    """Moving average decomposition component"""
    
    def __init__(self, config: DecompositionConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="MovingAverage",
            component_type=ComponentType.MOVING_AVG,
            required_params=['kernel_size'],
            optional_params=[],
            description="Moving average decomposition for trend-seasonal separation"
        )
    
    def _initialize_component(self, **kwargs):
        self.kernel_size = self.config.kernel_size
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=0)
    
    def forward(self, x):
        # x: [B, L, D]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        x_trend = self.avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1)
        x_seasonal = x - x_trend
        return x_seasonal, x_trend


class StandardEncoder(EncoderComponent):
    """Standard encoder component"""
    
    def __init__(self, config: EncoderConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="StandardEncoder",
            component_type=ComponentType.STANDARD_ENCODER,
            required_params=['e_layers', 'd_model', 'n_heads', 'd_ff'],
            optional_params=['dropout', 'activation'],
            description="Standard transformer encoder"
        )
        
        # Store sub-components
        self.attention_comp = kwargs.get('attention_comp')
        self.decomp_comp = kwargs.get('decomp_comp')
    
    def _initialize_component(self, **kwargs):
        # Store configuration parameters as instance variables for validation
        self.e_layers = self.config.e_layers
        self.d_model = self.config.d_model
        self.n_heads = self.config.n_heads
        self.d_ff = self.config.d_ff
        self.dropout = getattr(self.config, 'dropout', 0.1)
        self.activation = getattr(self.config, 'activation', 'gelu')
        
        # Store sub-components as instance variables
        self.attention_comp = kwargs.get('attention_comp')
        self.decomp_comp = kwargs.get('decomp_comp')
        
        self.layers = nn.ModuleList([
            EncoderLayer(
                self.config.d_model,
                self.config.n_heads,
                self.config.d_ff,
                self.config.dropout,
                self.attention_comp,
                self.decomp_comp
            ) for _ in range(self.config.e_layers)
        ])
        self.norm = nn.LayerNorm(self.config.d_model)
    
    def forward(self, x, attn_mask=None):
        attns = []
        for layer in self.layers:
            x, attn = layer(x, attn_mask)
            attns.append(attn)
        x = self.norm(x)
        return x, attns


class EncoderLayer(nn.Module):
    """Individual encoder layer"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout, attention_comp, decomp_comp):
        super().__init__()
        self.attention = attention_comp
        self.decomp = decomp_comp
        
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask)
        x = x + self.dropout(new_x)
        
        if self.decomp is not None:
            x, _ = self.decomp(self.norm1(x))
        else:
            x = self.norm1(x)
        
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm2(x + y), attn


class StandardDecoder(DecoderComponent):
    """Standard decoder component"""
    
    def __init__(self, config: DecoderConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="StandardDecoder",
            component_type=ComponentType.STANDARD_DECODER,
            required_params=['d_layers', 'd_model', 'n_heads', 'd_ff', 'c_out'],
            optional_params=['dropout', 'activation'],
            description="Standard transformer decoder"
        )
        
        # Store sub-components
        self.attention_comp = kwargs.get('attention_comp')
        self.decomp_comp = kwargs.get('decomp_comp')
    
    def _initialize_component(self, **kwargs):
        # Store configuration parameters as instance variables for validation
        self.d_layers = self.config.d_layers
        self.d_model = self.config.d_model
        self.n_heads = self.config.n_heads
        self.d_ff = self.config.d_ff
        self.c_out = self.config.c_out
        self.dropout = getattr(self.config, 'dropout', 0.1)
        self.activation = getattr(self.config, 'activation', 'gelu')
        
        # Store sub-components as instance variables  
        self.attention_comp = kwargs.get('attention_comp')
        self.decomp_comp = kwargs.get('decomp_comp')
        
        self.layers = nn.ModuleList([
            DecoderLayer(
                self.config.d_model,
                self.config.n_heads,
                self.config.d_ff,
                self.config.dropout,
                self.attention_comp,
                self.decomp_comp
            ) for _ in range(self.config.d_layers)
        ])
        self.norm = nn.LayerNorm(self.config.d_model)
        # Don't project here - let the output head handle the final projection
        # self.projection = nn.Linear(self.config.d_model, self.config.c_out)
    
    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask, cross_mask)
        x = self.norm(x)
        # Return raw decoder output in d_model dimension for output head
        return x, trend


class DecoderLayer(nn.Module):
    """Individual decoder layer"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout, attention_comp, decomp_comp):
        super().__init__()
        self.self_attention = attention_comp
        self.cross_attention = attention_comp  # Could be different instance
        self.decomp = decomp_comp
        
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(x, x, x, x_mask)[0])
        x = self.norm1(x)
        
        x = x + self.dropout(self.cross_attention(x, cross, cross, cross_mask)[0])
        
        if self.decomp is not None:
            x, _ = self.decomp(self.norm2(x))
        else:
            x = self.norm2(x)
        
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm3(x + y)


class DeterministicSampling(SamplingComponent):
    """Deterministic sampling (no sampling, just pass-through)"""
    
    def __init__(self, config: SamplingConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="Deterministic",
            component_type=ComponentType.DETERMINISTIC,
            required_params=[],
            optional_params=[],
            description="Pass-through sampling (no uncertainty)"
        )
    
    def _initialize_component(self, **kwargs):
        pass  # No parameters needed
    
    def forward(self, forward_fn, *args, **kwargs):
        """Simply call the forward function without sampling"""
        return forward_fn(*args, **kwargs)


class StandardOutputHead(OutputHeadComponent):
    """Standard output head component"""
    
    def __init__(self, config: OutputHeadConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="StandardHead",
            component_type=ComponentType.STANDARD_HEAD,
            required_params=['d_model', 'c_out'],
            optional_params=[],
            description="Standard linear output projection"
        )
    
    def _initialize_component(self, **kwargs):
        # Store required parameters for validation
        self.d_model = self.config.d_model
        self.c_out = self.config.c_out
        
        self.projection = nn.Linear(self.config.d_model, self.config.c_out)
    
    def forward(self, x):
        return self.projection(x)


class QuantileOutputHead(OutputHeadComponent):
    """Quantile output head for multi-quantile predictions"""
    
    def __init__(self, config: OutputHeadConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="QuantileHead",
            component_type=ComponentType.QUANTILE,
            required_params=['d_model', 'c_out', 'num_quantiles'],
            optional_params=[],
            description="Quantile output head for multi-quantile predictions"
        )
    
    def _initialize_component(self, **kwargs):
        # Store required parameters for validation
        self.d_model = self.config.d_model
        self.c_out = self.config.c_out
        self.num_quantiles = self.config.num_quantiles
        
        self.projection = nn.Linear(self.config.d_model, self.config.c_out * self.config.num_quantiles)
    
    def forward(self, x):
        """Forward pass for quantile predictions"""
        # x shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        # Project to quantile predictions
        output = self.projection(x)  # (batch_size, seq_len, c_out * num_quantiles)
        
        # Reshape to separate quantiles: (batch_size, seq_len, c_out, num_quantiles)
        output = output.view(batch_size, seq_len, self.config.c_out, self.config.num_quantiles)
        
        # For consistency with other heads, return flattened: (batch_size, seq_len, c_out * num_quantiles)
        return output.view(batch_size, seq_len, -1)


class MSELoss(LossComponent):
    """MSE loss component"""
    
    def __init__(self, config: LossConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="MSE",
            component_type=ComponentType.MSE,
            required_params=[],
            optional_params=[],
            description="Mean squared error loss"
        )
    
    def _initialize_component(self, **kwargs):
        self.loss_fn = nn.MSELoss()
    
    def forward(self, predictions, targets, **kwargs):
        return self.loss_fn(predictions, targets)


def register_concrete_components():
    """Register all concrete components with the global registry"""
    
    # Register attention components
    component_registry.register_component(
        ComponentType.AUTOCORRELATION,
        AutoCorrelationAttention,
        ComponentMetadata(
            name="AutoCorrelation",
            component_type=ComponentType.AUTOCORRELATION,
            required_params=['d_model', 'n_heads'],
            optional_params=['dropout', 'factor'],
            description="AutoCorrelation mechanism for time series"
        )
    )
    
    # For now, use AutoCorrelationAttention for adaptive as well
    component_registry.register_component(
        ComponentType.ADAPTIVE_AUTOCORRELATION,
        AutoCorrelationAttention,
        ComponentMetadata(
            name="AdaptiveAutoCorrelation",
            component_type=ComponentType.ADAPTIVE_AUTOCORRELATION,
            required_params=['d_model', 'n_heads'],
            optional_params=['dropout', 'factor'],
            description="Adaptive AutoCorrelation mechanism for time series"
        )
    )
    
    # Register decomposition components
    component_registry.register_component(
        ComponentType.MOVING_AVG,
        MovingAverageDecomposition,
        ComponentMetadata(
            name="MovingAverage",
            component_type=ComponentType.MOVING_AVG,
            required_params=['kernel_size'],
            optional_params=[],
            description="Moving average decomposition"
        )
    )
    
    component_registry.register_component(
        ComponentType.LEARNABLE_DECOMP,
        LearnableDecomposition,
        ComponentMetadata(
            name="LearnableDecomposition",
            component_type=ComponentType.LEARNABLE_DECOMP,
            required_params=['input_dim'],
            optional_params=['kernel_size'],
            description="Learnable decomposition"
        )
    )
    
    # Register encoder components
    component_registry.register_component(
        ComponentType.STANDARD_ENCODER,
        StandardEncoder,
        ComponentMetadata(
            name="StandardEncoder",
            component_type=ComponentType.STANDARD_ENCODER,
            required_params=['e_layers', 'd_model', 'n_heads', 'd_ff'],
            optional_params=['dropout', 'activation'],
            description="Standard transformer encoder"
        )
    )
    
    # For now, use StandardEncoder for enhanced as well
    component_registry.register_component(
        ComponentType.ENHANCED_ENCODER,
        StandardEncoder,
        ComponentMetadata(
            name="EnhancedEncoder",
            component_type=ComponentType.ENHANCED_ENCODER,
            required_params=['e_layers', 'd_model', 'n_heads', 'd_ff'],
            optional_params=['dropout', 'activation'],
            description="Enhanced transformer encoder"
        )
    )
    
    # Register decoder components
    component_registry.register_component(
        ComponentType.STANDARD_DECODER,
        StandardDecoder,
        ComponentMetadata(
            name="StandardDecoder",
            component_type=ComponentType.STANDARD_DECODER,
            required_params=['d_layers', 'd_model', 'n_heads', 'd_ff', 'c_out'],
            optional_params=['dropout', 'activation'],
            description="Standard transformer decoder"
        )
    )
    
    # For now, use StandardDecoder for enhanced as well  
    component_registry.register_component(
        ComponentType.ENHANCED_DECODER,
        StandardDecoder,
        ComponentMetadata(
            name="EnhancedDecoder",
            component_type=ComponentType.ENHANCED_DECODER,
            required_params=['d_layers', 'd_model', 'n_heads', 'd_ff', 'c_out'],
            optional_params=['dropout', 'activation'],
            description="Enhanced transformer decoder"
        )
    )
    
    # Register sampling components
    component_registry.register_component(
        ComponentType.DETERMINISTIC,
        DeterministicSampling,
        ComponentMetadata(
            name="Deterministic",
            component_type=ComponentType.DETERMINISTIC,
            required_params=[],
            optional_params=[],
            description="Deterministic sampling"
        )
    )
    
    # Register output head components
    component_registry.register_component(
        ComponentType.STANDARD_HEAD,
        StandardOutputHead,
        ComponentMetadata(
            name="StandardHead",
            component_type=ComponentType.STANDARD_HEAD,
            required_params=['d_model', 'c_out'],
            optional_params=[],
            description="Standard output head"
        )
    )
    
    # Register quantile output head
    component_registry.register_component(
        ComponentType.QUANTILE,
        QuantileOutputHead,
        ComponentMetadata(
            name="QuantileHead",
            component_type=ComponentType.QUANTILE,
            required_params=['d_model', 'c_out', 'num_quantiles'],
            optional_params=[],
            description="Quantile output head for multi-quantile predictions"
        )
    )
    
    # Register loss components
    component_registry.register_component(
        ComponentType.MSE,
        MSELoss,
        ComponentMetadata(
            name="MSE",
            component_type=ComponentType.MSE,
            required_params=[],
            optional_params=[],
            description="MSE loss function"
        )
    )

class BayesianSampling(SamplingComponent):
    """Bayesian sampling component for uncertainty quantification"""
    
    def __init__(self, config: SamplingConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="BayesianSampling",
            component_type=ComponentType.BAYESIAN,
            required_params=['n_samples', 'quantile_levels'],
            optional_params=['dropout_rate', 'temperature'],
            description="Bayesian sampling for uncertainty quantification"
        )
    
    def _initialize_component(self, **kwargs):
        self.n_samples = self.config.n_samples
        self.quantile_levels = self.config.quantile_levels
        self.dropout_rate = self.config.dropout_rate
        self.temperature = self.config.temperature
        
    def forward(self, model_fn, *args, **kwargs):
        """Perform Bayesian sampling"""
        # This is a simplified implementation - for now just call the model function
        # In practice, this would involve multiple forward passes with dropout
        return model_fn()

# Register sampling components
component_registry.register_component(
    ComponentType.BAYESIAN,
    BayesianSampling,
    ComponentMetadata(
        name="BayesianSampling",
        component_type=ComponentType.BAYESIAN,
        required_params=['n_samples', 'quantile_levels'],
        optional_params=['dropout_rate', 'temperature'],
        description="Bayesian sampling for uncertainty quantification"
    )
)

class HierarchicalEncoder(EncoderComponent):
    """Hierarchical encoder component"""
    
    def __init__(self, config: EncoderConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="HierarchicalEncoder",
            component_type=ComponentType.HIERARCHICAL_ENCODER,
            required_params=['e_layers', 'd_model', 'n_heads', 'd_ff'],
            optional_params=['dropout', 'activation', 'n_levels'],
            description="Hierarchical encoder with multi-level processing"
        )
    
    def _initialize_component(self, **kwargs):
        # For now, delegate to standard encoder functionality
        # In practice, this would implement hierarchical processing
        self.attention_comp = kwargs.get('attention_comp')
        self.decomp_comp = kwargs.get('decomp_comp')
        
        # Set required parameters
        self.e_layers = self.config.e_layers
        self.d_model = self.config.d_model
        self.n_heads = self.config.n_heads
        self.d_ff = self.config.d_ff
        
        # Initialize as a standard encoder for now
        from layers.Autoformer_EncDec import Encoder, EncoderLayer
        from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
        from layers.Embed import DataEmbedding_wo_pos
        
        # Create layers similar to standard encoder
        self.layers = nn.ModuleList([
            EncoderLayer(
                AutoCorrelationLayer(
                    AutoCorrelation(False, 1, attention_dropout=self.config.dropout,
                                    output_attention=False),
                    self.config.d_model, self.config.n_heads
                ),
                self.config.d_model,
                self.config.d_ff,
                dropout=self.config.dropout,
                activation=self.config.activation
            ) for _ in range(self.config.e_layers)
        ])
        self.norm = nn.LayerNorm(self.config.d_model)
    
    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x: [B, L, D]
        attns = []
        for layer in self.layers:
            x, attn = layer(x, attn_mask=attn_mask)  # Remove tau and delta
            attns.append(attn)
        
        if self.norm is not None:
            x = self.norm(x)
            
        return x, attns

# Register hierarchical encoder
component_registry.register_component(
    ComponentType.HIERARCHICAL_ENCODER,
    HierarchicalEncoder,
    ComponentMetadata(
        name="HierarchicalEncoder",
        component_type=ComponentType.HIERARCHICAL_ENCODER,
        required_params=['e_layers', 'd_model', 'n_heads', 'd_ff'],
        optional_params=['dropout', 'activation', 'n_levels'],
        description="Hierarchical encoder with multi-level processing"
    )
)

class CrossResolutionAttention(AttentionComponent):
    """Cross-resolution attention component"""
    
    def __init__(self, config: AttentionConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="CrossResolutionAttention",
            component_type=ComponentType.CROSS_RESOLUTION,
            required_params=['d_model', 'n_heads'],
            optional_params=['dropout', 'factor', 'output_attention', 'n_levels'],
            description="Cross-resolution attention for hierarchical processing"
        )
    
    def _initialize_component(self, **kwargs):
        # Store required parameters
        self.d_model = self.config.d_model
        self.n_heads = self.config.n_heads
        
        # Use AutoCorrelationLayer which handles proper tensor reshaping
        from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
        
        autocorr = AutoCorrelation(
            mask_flag=False, 
            factor=self.config.factor, 
            attention_dropout=self.config.dropout,
            output_attention=self.config.output_attention
        )
        
        self.attention = AutoCorrelationLayer(
            correlation=autocorr,
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            d_keys=None,
            d_values=None
        )
        self.dropout = nn.Dropout(self.config.dropout)
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        # AutoCorrelation only expects 4 parameters (no tau, delta)
        return self.attention(queries, keys, values, attn_mask)

# Register cross-resolution attention
component_registry.register_component(
    ComponentType.CROSS_RESOLUTION,
    CrossResolutionAttention,
    ComponentMetadata(
        name="CrossResolutionAttention",
        component_type=ComponentType.CROSS_RESOLUTION,
        required_params=['d_model', 'n_heads'],
        optional_params=['dropout', 'factor', 'output_attention', 'n_levels'],
        description="Cross-resolution attention for hierarchical processing"
    )
)

class WaveletDecomposition(DecompositionComponent):
    """Wavelet decomposition component"""
    
    def __init__(self, config: DecompositionConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="WaveletDecomposition",
            component_type=ComponentType.WAVELET_DECOMP,
            required_params=['wavelet_type', 'levels'],
            optional_params=[],
            description="Wavelet decomposition for hierarchical analysis"
        )
    
    def _initialize_component(self, **kwargs):
        decomp_params = getattr(self.config, 'decomposition_params', {})
        self.wavelet_type = decomp_params.get('wavelet_type', 'db4')
        self.levels = decomp_params.get('levels', 3)
        
        # For now, use simple moving average as placeholder
        self.kernel_size = 25
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=0)
    
    def forward(self, x):
        # x: [B, L, D] - for now, use simple decomposition
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        x_trend = self.avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1)
        x_seasonal = x - x_trend
        return x_seasonal, x_trend

# Register wavelet decomposition
component_registry.register_component(
    ComponentType.WAVELET_DECOMP,
    WaveletDecomposition,
    ComponentMetadata(
        name="WaveletDecomposition",
        component_type=ComponentType.WAVELET_DECOMP,
        required_params=['wavelet_type', 'levels'],
        optional_params=[],
        description="Wavelet decomposition for hierarchical analysis"
    )
)

