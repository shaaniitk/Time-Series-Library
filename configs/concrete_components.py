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
        # Get input dimension from multiple sources (kwargs override config)
        decomp_params = getattr(self.config, 'decomposition_params', {})
        self.input_dim = (
            kwargs.get('input_dim') or
            kwargs.get('d_model') or
            decomp_params.get('input_dim') or 
            getattr(self.config, 'input_dim', None) or
            getattr(self.config, 'd_model', 512)  # final fallback
        )
        self.kernel_size = (
            kwargs.get('kernel_size') or
            decomp_params.get('kernel_size') or 
            getattr(self.config, 'kernel_size', 25)
        )
        
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
        # quantile_levels should only be required when used alongside a quantile head;
        # treat it as optional for generic Bayesian sampling.
        self.metadata = ComponentMetadata(
            name="BayesianSampling",
            component_type=ComponentType.BAYESIAN,
            required_params=['n_samples'],
            optional_params=['quantile_levels', 'dropout_rate', 'temperature'],
            description="Bayesian sampling for uncertainty quantification"
        )
    
    def _initialize_component(self, **kwargs):
        self.n_samples = self.config.n_samples
        self.quantile_levels = self.config.quantile_levels
        self.dropout_rate = self.config.dropout_rate
        self.temperature = self.config.temperature
        # Some downstream utilities expect a kl_weight attribute on the sampling component;
        # SamplingConfig does not define it, so provide a safe fallback sourced from:
        # 1. explicit kwarg, 2. parent structured config.loss.kl_weight, 3. bayesian config, 4. default 1e-5.
        parent_config = kwargs.get('parent_config') or getattr(self, 'parent_config', None)
        kl_fallback = 1e-5
        if 'kl_weight' in kwargs:
            kl_fallback = kwargs['kl_weight']
        elif parent_config is not None:
            try:
                kl_fallback = getattr(getattr(parent_config, 'loss', object()), 'kl_weight', kl_fallback)
                if kl_fallback is None:
                    kl_fallback = getattr(getattr(parent_config, 'bayesian', object()), 'kl_weight', kl_fallback)
            except Exception:
                pass
        self.kl_weight = kl_fallback
        
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


# Additional Loss Components

class MAELoss(LossComponent):
    """MAE loss component"""
    
    def __init__(self, config: LossConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="MAE",
            component_type=ComponentType.MAE,
            required_params=[],
            optional_params=[],
            description="Mean absolute error loss function"
        )
    
    def _initialize_component(self, **kwargs):
        self.loss_fn = nn.L1Loss()
    
    def forward(self, predictions, targets, **kwargs):
        return self.loss_fn(predictions, targets)


class QuantileLoss(LossComponent):
    """Quantile loss component for single quantile prediction"""
    
    def __init__(self, config: LossConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="QuantileLoss",
            component_type=ComponentType.QUANTILE_LOSS,
            required_params=['quantiles'],
            optional_params=[],
            description="Quantile loss for probabilistic forecasting"
        )
    
    def _initialize_component(self, **kwargs):
        self.quantiles = torch.tensor(self.config.quantiles, dtype=torch.float32)
    
    def forward(self, predictions, targets, **kwargs):
        """Compute quantile loss for multiple quantiles"""
        # Predictions shape: [B, T, C*Q] where Q is number of quantiles
        batch_size, seq_len, combined_dim = predictions.shape
        num_quantiles = len(self.quantiles)
        num_features = combined_dim // num_quantiles
        
        # Reshape predictions: [B, T, C, Q]
        pred_quantiles = predictions.view(batch_size, seq_len, num_features, num_quantiles)
        
        # Compute quantile loss
        total_loss = 0.0
        for i, tau in enumerate(self.quantiles):
            pred_q = pred_quantiles[:, :, :, i]  # [B, T, C]
            residual = targets - pred_q
            loss_q = torch.where(residual >= 0, 
                               tau * residual, 
                               (tau - 1) * residual)
            total_loss += loss_q.mean()
        
        return total_loss


class BayesianMSELoss(LossComponent):
    """Bayesian MSE loss with KL divergence"""
    
    def __init__(self, config: LossConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="BayesianMSE",
            component_type=ComponentType.BAYESIAN_MSE,
            required_params=['kl_weight'],
            optional_params=['prior_scale'],
            description="Bayesian MSE loss with KL divergence regularization"
        )
    
    def _initialize_component(self, **kwargs):
        self.mse_loss = nn.MSELoss()
        self.kl_weight = self.config.kl_weight
        self.prior_scale = getattr(self.config, 'prior_scale', 1.0)
    
    def forward(self, predictions, targets, model=None, **kwargs):
        """Compute MSE + KL divergence loss"""
        mse_loss = self.mse_loss(predictions, targets)
        
        # Add KL divergence if model has Bayesian layers
        kl_loss = 0.0
        if model is not None and hasattr(model, 'get_kl_divergence'):
            kl_loss = model.get_kl_divergence()
        
        return mse_loss + self.kl_weight * kl_loss


class BayesianQuantileLoss(LossComponent):
    """Bayesian quantile loss with KL divergence"""
    
    def __init__(self, config: LossConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="BayesianQuantileLoss",
            component_type=ComponentType.BAYESIAN_QUANTILE,
            required_params=['quantiles', 'kl_weight'],
            optional_params=['prior_scale'],
            description="Bayesian quantile loss for uncertainty quantification"
        )
    
    def _initialize_component(self, **kwargs):
        self.quantiles = torch.tensor(self.config.quantiles, dtype=torch.float32)
        self.kl_weight = self.config.kl_weight
        self.prior_scale = getattr(self.config, 'prior_scale', 1.0)
    
    def forward(self, predictions, targets, model=None, **kwargs):
        """Compute combined quantile and KL divergence loss"""
        # Quantile loss computation (same as QuantileLoss)
        batch_size, seq_len, combined_dim = predictions.shape
        num_quantiles = len(self.quantiles)
        num_features = combined_dim // num_quantiles
        
        pred_quantiles = predictions.view(batch_size, seq_len, num_features, num_quantiles)
        
        quantile_loss = 0.0
        for i, tau in enumerate(self.quantiles):
            pred_q = pred_quantiles[:, :, :, i]
            residual = targets - pred_q
            loss_q = torch.where(residual >= 0, 
                               tau * residual, 
                               (tau - 1) * residual)
            quantile_loss += loss_q.mean()
        
        # Add KL divergence
        kl_loss = 0.0
        if model is not None and hasattr(model, 'get_kl_divergence'):
            kl_loss = model.get_kl_divergence()
        
        return quantile_loss + self.kl_weight * kl_loss


# Register additional loss components
component_registry.register_component(
    ComponentType.MAE,
    MAELoss,
    ComponentMetadata(
        name="MAE",
        component_type=ComponentType.MAE,
        required_params=[],
        optional_params=[],
        description="Mean absolute error loss function"
    )
)

component_registry.register_component(
    ComponentType.QUANTILE_LOSS,
    QuantileLoss,
    ComponentMetadata(
        name="QuantileLoss",
        component_type=ComponentType.QUANTILE_LOSS,
        required_params=['quantiles'],
        optional_params=[],
        description="Quantile loss for probabilistic forecasting"
    )
)

component_registry.register_component(
    ComponentType.BAYESIAN_MSE,
    BayesianMSELoss,
    ComponentMetadata(
        name="BayesianMSE",
        component_type=ComponentType.BAYESIAN_MSE,
        required_params=['kl_weight'],
        optional_params=['prior_scale'],
        description="Bayesian MSE loss with KL divergence regularization"
    )
)

component_registry.register_component(
    ComponentType.BAYESIAN_QUANTILE,
    BayesianQuantileLoss,
    ComponentMetadata(
        name="BayesianQuantileLoss",
        component_type=ComponentType.BAYESIAN_QUANTILE,
        required_params=['quantiles', 'kl_weight'],
        optional_params=['prior_scale'],
        description="Bayesian quantile loss for uncertainty quantification"
    )
)


# ========================================================================================
# ADVANCED COMPONENTS - INTEGRATED FROM LAYERS AND UTILS
# ========================================================================================

# Advanced Attention Components
# ========================================================================================

class FourierAttention(AttentionComponent):
    """Fourier-based attention for capturing periodic patterns"""
    
    def __init__(self, config: AttentionConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="FourierAttention",
            component_type=ComponentType.FOURIER_ATTENTION,
            required_params=['d_model', 'n_heads', 'seq_len'],
            optional_params=['dropout'],
            description="Fourier-based attention for periodic pattern capture"
        )
    
    def _initialize_component(self, **kwargs):
        import torch.nn.functional as F
        
        self.d_model = self.config.d_model
        self.n_heads = self.config.n_heads
        self.seq_len = getattr(self.config, 'seq_len', 96)
        
        # Learnable frequency components
        self.freq_weights = nn.Parameter(torch.randn(self.seq_len // 2 + 1, self.n_heads))
        self.phase_weights = nn.Parameter(torch.zeros(self.seq_len // 2 + 1, self.n_heads))
        
        self.qkv = nn.Linear(self.d_model, self.d_model * 3)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(getattr(self.config, 'dropout', 0.1))
        
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, D = queries.shape
        
        # Transform to frequency domain
        x_freq = torch.fft.rfft(queries, dim=1)
        
        # Apply learnable frequency filtering
        freq_filter = torch.complex(
            torch.cos(self.phase_weights) * self.freq_weights,
            torch.sin(self.phase_weights) * self.freq_weights
        )
        
        x_freq = x_freq.unsqueeze(-1) * freq_filter.unsqueeze(0).unsqueeze(2)
        x_filtered = torch.fft.irfft(x_freq.mean(-1), n=L, dim=1)
        
        # Standard attention on filtered signal
        qkv = self.qkv(x_filtered).reshape(B, L, 3, self.n_heads, D // self.n_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(D // self.n_heads)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out), attn


class AdaptiveAutoCorrelationAttention(AttentionComponent):
    """Enhanced AutoCorrelation with adaptive window selection"""
    
    def __init__(self, config: AttentionConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="AdaptiveAutoCorrelation",
            component_type=ComponentType.ADAPTIVE_AUTOCORRELATION,
            required_params=['d_model', 'n_heads'],
            optional_params=['factor', 'dropout', 'adaptive_k', 'scales'],
            description="Enhanced AutoCorrelation with adaptive features"
        )
    
    def _initialize_component(self, **kwargs):
        self.d_model = self.config.d_model
        self.n_heads = self.config.n_heads
        self.factor = getattr(self.config, 'factor', 1)
        self.adaptive_k = getattr(self.config, 'adaptive_k', True)
        self.scales = getattr(self.config, 'scales', [1, 2, 4])
        
        # Learnable components
        self.scale_weights = nn.Parameter(torch.ones(len(self.scales)) / len(self.scales))
        self.frequency_filter = nn.Parameter(torch.ones(1))
        
        # Projections
        self.query_projection = nn.Linear(self.d_model, self.d_model)
        self.key_projection = nn.Linear(self.d_model, self.d_model)
        self.value_projection = nn.Linear(self.d_model, self.d_model)
        self.out_projection = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(getattr(self.config, 'dropout', 0.1))
        
    def select_adaptive_k(self, corr_energy, length):
        """Intelligently select the number of correlation peaks"""
        # Simple adaptive selection based on energy distribution
        sorted_energy, _ = torch.sort(corr_energy, descending=True)
        cumsum = torch.cumsum(sorted_energy, dim=-1)
        total_energy = cumsum[:, -1:] + 1e-8
        
        # Select k such that we capture 80% of energy
        energy_ratio = cumsum / total_energy
        k = torch.sum(energy_ratio < 0.8, dim=-1) + 1
        k = torch.clamp(k, 1, length // self.factor)
        
        return k
        
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        H = self.n_heads
        
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, L, H, -1)
        values = self.value_projection(values).view(B, L, H, -1)
        
        # Multi-scale correlation
        correlations = []
        for scale in self.scales:
            if scale > 1:
                # Downsample for multi-scale analysis
                q_scaled = torch.nn.functional.avg_pool1d(
                    queries.permute(0, 2, 3, 1).reshape(B*H, -1, L), 
                    scale, stride=scale
                ).reshape(B, H, -1, L//scale).permute(0, 3, 1, 2)
                k_scaled = torch.nn.functional.avg_pool1d(
                    keys.permute(0, 2, 3, 1).reshape(B*H, -1, L), 
                    scale, stride=scale
                ).reshape(B, H, -1, L//scale).permute(0, 3, 1, 2)
            else:
                q_scaled, k_scaled = queries, keys
                
            # Compute correlation
            corr = torch.einsum("blhd,bshd->bhls", q_scaled, k_scaled)
            correlations.append(corr)
        
        # Weighted combination of scales
        scale_weights = torch.softmax(self.scale_weights, dim=0)
        correlation = sum(w * corr for w, corr in zip(scale_weights, correlations))
        
        # Apply frequency filtering
        correlation = correlation * self.frequency_filter
        
        # Adaptive top-k selection
        if self.adaptive_k:
            corr_energy = torch.norm(correlation, dim=-1)
            k = self.select_adaptive_k(corr_energy, L)
        else:
            k = L // self.factor
            
        # Apply attention
        attn = torch.softmax(correlation / math.sqrt(queries.shape[-1]), dim=-1)
        attn = self.dropout(attn)
        
        out = torch.einsum("bhls,bshd->blhd", attn, values)
        out = out.contiguous().view(B, L, -1)
        
        return self.out_projection(out), attn


# Advanced Decomposition Components  
# ========================================================================================

class AdvancedWaveletDecomposition(DecompositionComponent):
    """Advanced learnable wavelet decomposition for multi-resolution analysis"""
    
    def __init__(self, config: DecompositionConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="AdvancedWaveletDecomposition",
            component_type=ComponentType.ADVANCED_WAVELET,
            required_params=['input_dim'],
            optional_params=['levels'],
            description="Advanced learnable wavelet decomposition"
        )
    
    def _initialize_component(self, **kwargs):
        decomp_params = getattr(self.config, 'decomposition_params', {})
        self.input_dim = (
            decomp_params.get('input_dim') or 
            getattr(self.config, 'input_dim', None) or
            getattr(self.config, 'd_model', 512)
        )
        self.levels = decomp_params.get('levels', getattr(self.config, 'levels', 3))
        
        # Learnable wavelet filters
        self.low_pass = nn.ModuleList([
            nn.Conv1d(self.input_dim, self.input_dim, 4, stride=2, padding=1, groups=self.input_dim)
            for _ in range(self.levels)
        ])
        
        self.high_pass = nn.ModuleList([
            nn.Conv1d(self.input_dim, self.input_dim, 4, stride=2, padding=1, groups=self.input_dim)
            for _ in range(self.levels)
        ])
        
        # Reconstruction weights
        self.recon_weights = nn.Parameter(torch.ones(self.levels + 1) / (self.levels + 1))
        
    def forward(self, x):
        B, L, D = x.shape
        x_transpose = x.transpose(1, 2)  # [B, D, L]
        
        components = []
        current = x_transpose
        
        # Decomposition
        for i in range(self.levels):
            low = self.low_pass[i](current)
            high = self.high_pass[i](current)
            components.append(high)
            current = low
            
        components.append(current)  # Final low-frequency component
        
        # Weighted reconstruction
        weights = torch.softmax(self.recon_weights, dim=0)
        
        # Upsample and combine
        reconstructed = torch.zeros_like(x_transpose)
        for i, (comp, weight) in enumerate(zip(components, weights)):
            if comp.size(-1) < L:
                comp = torch.nn.functional.interpolate(comp, size=L, mode='linear', align_corners=False)
            reconstructed += comp * weight
            
        seasonal = reconstructed.transpose(1, 2)
        trend = x - seasonal  # Residual as trend
        
        return seasonal, trend


# Advanced Encoder Components
# ========================================================================================

class TemporalConvEncoder(EncoderComponent):
    """Temporal Convolutional Network encoder for sequence modeling"""
    
    def __init__(self, config: EncoderConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="TemporalConvEncoder",
            component_type=ComponentType.TEMPORAL_CONV_ENCODER,
            required_params=['d_model'],
            optional_params=['num_channels', 'kernel_size', 'dropout'],
            description="Temporal Convolutional Network for sequence modeling"
        )
    
    def _initialize_component(self, **kwargs):
        self.d_model = self.config.d_model
        self.num_channels = getattr(self.config, 'num_channels', [64, 64, 64])
        self.kernel_size = getattr(self.config, 'kernel_size', 2)
        self.dropout_rate = getattr(self.config, 'dropout', 0.2)
        
        # Causal convolutions
        layers = []
        num_levels = len(self.num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = self.d_model if i == 0 else self.num_channels[i-1]
            out_channels = self.num_channels[i]
            
            # Causal convolution
            padding = (self.kernel_size - 1) * dilation_size
            conv = nn.Conv1d(in_channels, out_channels, self.kernel_size,
                           padding=padding, dilation=dilation_size)
            
            layers.extend([
                conv,
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            
        self.tcn = nn.Sequential(*layers)
        
        # Output projection
        self.output_projection = nn.Linear(self.num_channels[-1], self.d_model)
        
    def forward(self, x, attn_mask=None):
        # x: [B, L, D]
        B, L, D = x.shape
        
        # TCN expects [B, D, L]
        x_conv = x.transpose(1, 2)
        
        # Apply TCN
        out_conv = self.tcn(x_conv)
        
        # Remove future information (causal padding)
        for layer in self.tcn:
            if isinstance(layer, nn.Conv1d):
                padding = layer.padding[0]
                if padding > 0:
                    out_conv = out_conv[:, :, :-padding]
                    
        # Back to [B, L, D]
        out = out_conv.transpose(1, 2)
        
        # Project to d_model
        out = self.output_projection(out)
        
        return out, None  # No attention weights for TCN


class MetaLearningAdapter(EncoderComponent):
    """Meta-learning adapter for quick adaptation to new patterns"""
    
    def __init__(self, config: EncoderConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="MetaLearningAdapter",
            component_type=ComponentType.META_LEARNING_ADAPTER,
            required_params=['d_model'],
            optional_params=['adaptation_steps'],
            description="Meta-learning adapter for quick pattern adaptation"
        )
    
    def _initialize_component(self, **kwargs):
        self.d_model = self.config.d_model
        self.adaptation_steps = getattr(self.config, 'adaptation_steps', 5)
        
        # Fast adaptation parameters
        self.fast_weights = nn.ParameterList([
            nn.Parameter(torch.randn(self.d_model, self.d_model) * 0.01)
            for _ in range(self.adaptation_steps)
        ])
        
        self.fast_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(self.d_model))
            for _ in range(self.adaptation_steps)
        ])
        
        # Meta-learning rate
        self.meta_lr = nn.Parameter(torch.ones(1) * 0.01)
        
    def forward(self, x, support_set=None, attn_mask=None):
        if support_set is not None and self.training:
            # Fast adaptation using support set
            adapted_weights = []
            adapted_biases = []
            
            for i in range(self.adaptation_steps):
                # Compute gradients on support set
                loss = torch.nn.functional.mse_loss(
                    torch.nn.functional.linear(support_set, self.fast_weights[i], self.fast_biases[i]),
                    support_set
                )
                
                # Update fast weights
                grad_w = torch.autograd.grad(loss, self.fast_weights[i], create_graph=True)[0]
                grad_b = torch.autograd.grad(loss, self.fast_biases[i], create_graph=True)[0]
                
                adapted_w = self.fast_weights[i] - self.meta_lr * grad_w
                adapted_b = self.fast_biases[i] - self.meta_lr * grad_b
                
                adapted_weights.append(adapted_w)
                adapted_biases.append(adapted_b)
            
            # Use adapted weights
            for w, b in zip(adapted_weights, adapted_biases):
                x = torch.nn.functional.linear(x, w, b)
                x = torch.nn.functional.relu(x)
        else:
            # Standard forward pass
            for w, b in zip(self.fast_weights, self.fast_biases):
                x = torch.nn.functional.linear(x, w, b)
                x = torch.nn.functional.relu(x)
                
        return x, None


# Advanced Sampling Components
# ========================================================================================

class AdaptiveMixtureSampling(SamplingComponent):
    """Adaptive mixture of experts for different time series patterns"""
    
    def __init__(self, config: SamplingConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="AdaptiveMixture",
            component_type=ComponentType.ADAPTIVE_MIXTURE,
            required_params=['d_model'],
            optional_params=['num_experts'],
            description="Adaptive mixture of experts for pattern-specific sampling"
        )
    
    def _initialize_component(self, **kwargs):
        self.d_model = getattr(self.config, 'd_model', 512)
        self.num_experts = getattr(self.config, 'num_experts', 4)
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, self.d_model * 2),
                nn.ReLU(),
                nn.Linear(self.d_model * 2, self.d_model)
            ) for _ in range(self.num_experts)
        ])
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, self.num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, model_fn, **kwargs):
        # Get base prediction
        x = model_fn()
        
        # Compute gating weights
        gate_weights = self.gate(x)  # [B, L, num_experts]
        
        # Compute expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        
        expert_outputs = torch.stack(expert_outputs, dim=-1)  # [B, L, d_model, num_experts]
        
        # Weighted combination
        output = torch.sum(expert_outputs * gate_weights.unsqueeze(-2), dim=-1)
        
        return {
            'prediction': output,
            'uncertainty': torch.std(expert_outputs, dim=-1),
            'expert_weights': gate_weights
        }


# Advanced Loss Components
# ========================================================================================

class FocalLoss(LossComponent):
    """Focal loss for handling imbalanced data"""
    
    def __init__(self, config: LossConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="FocalLoss",
            component_type=ComponentType.FOCAL_LOSS,
            required_params=[],
            optional_params=['alpha', 'gamma'],
            description="Focal loss for imbalanced data handling"
        )
    
    def _initialize_component(self, **kwargs):
        self.alpha = getattr(self.config, 'alpha', 1.0)
        self.gamma = getattr(self.config, 'gamma', 2.0)
        
    def forward(self, predictions, targets, **kwargs):
        ce_loss = torch.nn.functional.mse_loss(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class AdaptiveAutoformerLoss(LossComponent):
    """Adaptive loss with trend/seasonal decomposition weighting"""
    
    def __init__(self, config: LossConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="AdaptiveAutoformerLoss",
            component_type=ComponentType.ADAPTIVE_AUTOFORMER_LOSS,
            required_params=[],
            optional_params=['base_loss', 'moving_avg', 'adaptive_weights'],
            description="Adaptive loss with trend/seasonal component weighting"
        )
    
    def _initialize_component(self, **kwargs):
        from layers.Autoformer_EncDec import series_decomp
        
        self.base_loss = getattr(self.config, 'base_loss', 'mse')
        self.moving_avg = getattr(self.config, 'moving_avg', 25)
        self.adaptive_weights = getattr(self.config, 'adaptive_weights', True)
        
        # Decomposition for loss calculation
        self.decomp = series_decomp(kernel_size=self.moving_avg)
        
        # Learnable weight parameters
        if self.adaptive_weights:
            self.trend_weight = nn.Parameter(torch.tensor(1.0))
            self.seasonal_weight = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer('trend_weight', torch.tensor(1.0))
            self.register_buffer('seasonal_weight', torch.tensor(1.0))
        
        # Base loss function
        if self.base_loss == 'mse':
            self.loss_fn = torch.nn.functional.mse_loss
        elif self.base_loss == 'mae':
            self.loss_fn = torch.nn.functional.l1_loss
        else:
            self.loss_fn = torch.nn.functional.mse_loss
        
    def forward(self, predictions, targets, **kwargs):
        # Decompose both predictions and ground truth
        pred_seasonal, pred_trend = self.decomp(predictions)
        true_seasonal, true_trend = self.decomp(targets)
        
        # Compute component-wise losses
        trend_loss = self.loss_fn(pred_trend, true_trend, reduction='mean')
        seasonal_loss = self.loss_fn(pred_seasonal, true_seasonal, reduction='mean')
        
        # Apply adaptive weighting with softplus for positivity
        if self.adaptive_weights:
            trend_w = torch.nn.functional.softplus(self.trend_weight)
            seasonal_w = torch.nn.functional.softplus(self.seasonal_weight)
        else:
            trend_w = self.trend_weight
            seasonal_w = self.seasonal_weight
        
        # Total adaptive loss
        total_loss = trend_w * trend_loss + seasonal_w * seasonal_loss
        
        return total_loss


class AdaptiveLossWeighting(LossComponent):
    """Adaptive loss weighting for multi-task learning"""
    
    def __init__(self, config: LossConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="AdaptiveLossWeighting",
            component_type=ComponentType.ADAPTIVE_LOSS_WEIGHTING,
            required_params=['num_tasks'],
            optional_params=[],
            description="Adaptive weighting for multi-task loss optimization"
        )
    
    def _initialize_component(self, **kwargs):
        self.num_tasks = getattr(self.config, 'num_tasks', 3)
        self.log_vars = nn.Parameter(torch.zeros(self.num_tasks))
        
    def forward(self, losses, **kwargs):
        """
        Args:
            losses: List of individual task losses
        """
        if not isinstance(losses, (list, tuple)):
            losses = [losses]
            
        weighted_losses = []
        for i, loss in enumerate(losses[:self.num_tasks]):
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * loss + self.log_vars[i]
            weighted_losses.append(weighted_loss)
        
        return sum(weighted_losses)


# Register Advanced Components
# ========================================================================================

# Register Fourier Attention
component_registry.register_component(
    ComponentType.FOURIER_ATTENTION,
    FourierAttention,
    ComponentMetadata(
        name="FourierAttention",
        component_type=ComponentType.FOURIER_ATTENTION,
        required_params=['d_model', 'n_heads', 'seq_len'],
        optional_params=['dropout'],
        description="Fourier-based attention for periodic pattern capture"
    )
)

# Register Adaptive AutoCorrelation
component_registry.register_component(
    ComponentType.ADAPTIVE_AUTOCORRELATION,
    AdaptiveAutoCorrelationAttention,
    ComponentMetadata(
        name="AdaptiveAutoCorrelation",
        component_type=ComponentType.ADAPTIVE_AUTOCORRELATION,
        required_params=['d_model', 'n_heads'],
        optional_params=['factor', 'dropout', 'adaptive_k', 'scales'],
        description="Enhanced AutoCorrelation with adaptive features"
    )
)

# Register Advanced Wavelet Decomposition
component_registry.register_component(
    ComponentType.ADVANCED_WAVELET,
    AdvancedWaveletDecomposition,
    ComponentMetadata(
        name="AdvancedWaveletDecomposition",
        component_type=ComponentType.ADVANCED_WAVELET,
        required_params=['input_dim'],
        optional_params=['levels'],
        description="Advanced learnable wavelet decomposition"
    )
)

# Register Temporal Conv Encoder
component_registry.register_component(
    ComponentType.TEMPORAL_CONV_ENCODER,
    TemporalConvEncoder,
    ComponentMetadata(
        name="TemporalConvEncoder",
        component_type=ComponentType.TEMPORAL_CONV_ENCODER,
        required_params=['d_model'],
        optional_params=['num_channels', 'kernel_size', 'dropout'],
        description="Temporal Convolutional Network for sequence modeling"
    )
)

# Register Meta Learning Adapter
component_registry.register_component(
    ComponentType.META_LEARNING_ADAPTER,
    MetaLearningAdapter,
    ComponentMetadata(
        name="MetaLearningAdapter",
        component_type=ComponentType.META_LEARNING_ADAPTER,
        required_params=['d_model'],
        optional_params=['adaptation_steps'],
        description="Meta-learning adapter for quick pattern adaptation"
    )
)

# Register Adaptive Mixture Sampling
component_registry.register_component(
    ComponentType.ADAPTIVE_MIXTURE,
    AdaptiveMixtureSampling,
    ComponentMetadata(
        name="AdaptiveMixture",
        component_type=ComponentType.ADAPTIVE_MIXTURE,
        required_params=['d_model'],
        optional_params=['num_experts'],
        description="Adaptive mixture of experts for pattern-specific sampling"
    )
)

# Register Focal Loss
component_registry.register_component(
    ComponentType.FOCAL_LOSS,
    FocalLoss,
    ComponentMetadata(
        name="FocalLoss",
        component_type=ComponentType.FOCAL_LOSS,
        required_params=[],
        optional_params=['alpha', 'gamma'],
        description="Focal loss for imbalanced data handling"
    )
)

# Register Adaptive Autoformer Loss
component_registry.register_component(
    ComponentType.ADAPTIVE_AUTOFORMER_LOSS,
    AdaptiveAutoformerLoss,
    ComponentMetadata(
        name="AdaptiveAutoformerLoss",
        component_type=ComponentType.ADAPTIVE_AUTOFORMER_LOSS,
        required_params=[],
        optional_params=['base_loss', 'moving_avg', 'adaptive_weights'],
        description="Adaptive loss with trend/seasonal component weighting"
    )
)

# Register Adaptive Loss Weighting
component_registry.register_component(
    ComponentType.ADAPTIVE_LOSS_WEIGHTING,
    AdaptiveLossWeighting,
    ComponentMetadata(
        name="AdaptiveLossWeighting",
        component_type=ComponentType.ADAPTIVE_LOSS_WEIGHTING,
        required_params=['num_tasks'],
        optional_params=[],
        description="Adaptive weighting for multi-task loss optimization"
    )
)


# ========================================================================================
# BAYESIAN COMPONENTS - INTEGRATED FROM BAYESIANLAYERS.PY
# ========================================================================================

class BayesianLinearHead(OutputHeadComponent):
    """Bayesian linear output head with weight uncertainty"""
    
    def __init__(self, config: OutputHeadConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="BayesianLinearHead",
            component_type=ComponentType.BAYESIAN_HEAD,
            required_params=['d_model', 'c_out'],
            optional_params=['prior_std', 'samples'],
            description="Bayesian linear head with weight uncertainty quantification"
        )
    
    def _initialize_component(self, **kwargs):
        self.d_model = self.config.d_model
        self.c_out = self.config.c_out
        self.prior_std = getattr(self.config, 'prior_std', 1.0)
        self.samples = getattr(self.config, 'samples', 10)
        
        # Weight parameters (mean and log variance)
        self.weight_mean = nn.Parameter(torch.randn(self.c_out, self.d_model) * 0.1)
        self.weight_logvar = nn.Parameter(torch.full((self.c_out, self.d_model), -5.0))
        
        # Bias parameters
        self.bias_mean = nn.Parameter(torch.randn(self.c_out) * 0.1)
        self.bias_logvar = nn.Parameter(torch.full((self.c_out,), -5.0))
    
    def sample_weights(self):
        """Sample weights from learned distributions"""
        weight_std = torch.exp(0.5 * self.weight_logvar)
        bias_std = torch.exp(0.5 * self.bias_logvar)
        
        weight_eps = torch.randn_like(self.weight_mean)
        bias_eps = torch.randn_like(self.bias_mean)
        
        weight = self.weight_mean + weight_std * weight_eps
        bias = self.bias_mean + bias_std * bias_eps
        
        return weight, bias
    
    def get_kl_divergence(self):
        """Compute KL divergence from prior"""
        weight_var = torch.exp(self.weight_logvar)
        bias_var = torch.exp(self.bias_logvar)
        
        kl_weight = 0.5 * torch.sum(
            (self.weight_mean**2 + weight_var) / (self.prior_std**2) - 
            torch.log(weight_var / (self.prior_std**2)) - 1
        )
        
        kl_bias = 0.5 * torch.sum(
            (self.bias_mean**2 + bias_var) / (self.prior_std**2) - 
            torch.log(bias_var / (self.prior_std**2)) - 1
        )
        
        return kl_weight + kl_bias
    
    def forward(self, x):
        if self.training:
            # Single forward pass during training
            weight, bias = self.sample_weights()
            return torch.nn.functional.linear(x, weight, bias)
        else:
            # Multiple samples during inference for uncertainty
            predictions = []
            for _ in range(self.samples):
                weight, bias = self.sample_weights()
                pred = torch.nn.functional.linear(x, weight, bias)
                predictions.append(pred)
            
            predictions = torch.stack(predictions, dim=0)
            mean_pred = torch.mean(predictions, dim=0)
            uncertainty = torch.std(predictions, dim=0)
            
            return {
                'prediction': mean_pred,
                'uncertainty': uncertainty,
                'samples': predictions
            }


# Register Bayesian Head
component_registry.register_component(
    ComponentType.BAYESIAN_HEAD,
    BayesianLinearHead,
    ComponentMetadata(
        name="BayesianLinearHead",
        component_type=ComponentType.BAYESIAN_HEAD,
        required_params=['d_model', 'c_out'],
        optional_params=['prior_std', 'samples'],
        description="Bayesian linear head with weight uncertainty quantification"
    )
)


# ========================================================================================
# FOURIER BLOCK COMPONENT - INTEGRATED FROM FOURIERCORRELATION.PY
# ========================================================================================

class FourierBlockAttention(AttentionComponent):
    """Fourier block for frequency domain representation learning"""
    
    def __init__(self, config: AttentionConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="FourierBlock",
            component_type=ComponentType.FOURIER_BLOCK,
            required_params=['d_model', 'n_heads', 'seq_len'],
            optional_params=['modes', 'mode_select_method'],
            description="Fourier block for frequency domain representation learning"
        )
    
    def _initialize_component(self, **kwargs):
        self.d_model = self.config.d_model
        self.n_heads = self.config.n_heads
        self.seq_len = getattr(self.config, 'seq_len', 96)
        self.modes = getattr(self.config, 'modes', 64)
        self.mode_select_method = getattr(self.config, 'mode_select_method', 'random')
        
        # Get frequency modes
        self.index = self._get_frequency_modes()
        
        self.scale = (1 / (self.d_model * self.d_model))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(
                self.n_heads, self.d_model // self.n_heads, 
                self.d_model // self.n_heads, len(self.index), dtype=torch.float
            )
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(
                self.n_heads, self.d_model // self.n_heads, 
                self.d_model // self.n_heads, len(self.index), dtype=torch.float
            )
        )
    
    def _get_frequency_modes(self):
        """Get modes on frequency domain"""
        import numpy as np
        
        modes = min(self.modes, self.seq_len // 2)
        if self.mode_select_method == 'random':
            index = list(range(0, self.seq_len // 2))
            np.random.shuffle(index)
            index = index[:modes]
        else:
            index = list(range(0, modes))
        index.sort()
        return index
    
    def _compl_mul1d(self, order, x, weights):
        """Complex multiplication in 1D"""
        return torch.einsum(order, x, weights)
    
    def forward(self, queries, keys, values, attn_mask=None):
        # queries: [B, L, H, D//H]
        B, L, H, E = queries.shape
        
        # FFT
        x_ft = torch.fft.rfft(queries, dim=1)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(B, L // 2 + 1, H, E, device=queries.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index):
            if j < x_ft.size(1):
                out_ft[:, j, :, :] = self._compl_mul1d(
                    "bhi,hio->bho", x_ft[:, j, :, :], 
                    torch.complex(self.weights1[:, :, :, i], self.weights2[:, :, :, i])
                )
        
        # IFFT
        output = torch.fft.irfft(out_ft, n=L, dim=1)
        
        return output.view(B, L, -1), None


# Register Fourier Block
component_registry.register_component(
    ComponentType.FOURIER_BLOCK,
    FourierBlockAttention,
    ComponentMetadata(
        name="FourierBlock",
        component_type=ComponentType.FOURIER_BLOCK,
        required_params=['d_model', 'n_heads', 'seq_len'],
        optional_params=['modes', 'mode_select_method'],
        description="Fourier block for frequency domain representation learning"
    )
)


# =============================
# ADVANCED COMPONENTS FROM LAYERS/UTILS
# =============================

class AdvancedWaveletDecomposition(DecompositionComponent):
    """Advanced learnable wavelet decomposition for multi-resolution analysis"""
    
    def __init__(self, config: DecompositionConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="AdvancedWaveletDecomposition",
            component_type=ComponentType.ADVANCED_WAVELET,
            required_params=['input_dim'],
            optional_params=['levels'],
            description="Advanced learnable wavelet decomposition for multi-resolution analysis"
        )
    
    def _initialize_component(self, **kwargs):
        decomp_params = getattr(self.config, 'decomposition_params', {})
        self.input_dim = (
            decomp_params.get('input_dim') or 
            getattr(self.config, 'input_dim', None) or
            getattr(self.config, 'd_model', 512)
        )
        self.levels = decomp_params.get('levels', 3)
        
        # Learnable wavelet filters
        self.low_pass = nn.ModuleList([
            nn.Conv1d(self.input_dim, self.input_dim, 4, stride=2, padding=1, groups=self.input_dim)
            for _ in range(self.levels)
        ])
        
        self.high_pass = nn.ModuleList([
            nn.Conv1d(self.input_dim, self.input_dim, 4, stride=2, padding=1, groups=self.input_dim)
            for _ in range(self.levels)
        ])
        
        # Reconstruction weights
        self.recon_weights = nn.Parameter(torch.ones(self.levels + 1) / (self.levels + 1))
        
    def forward(self, x):
        B, L, D = x.shape
        x = x.transpose(1, 2)  # [B, D, L]
        
        components = []
        current = x
        
        # Decomposition
        for i in range(self.levels):
            low = self.low_pass[i](current)
            high = self.high_pass[i](current)
            components.append(high)
            current = low
            
        components.append(current)  # Final low-frequency component
        
        # Weighted reconstruction
        weights = torch.softmax(self.recon_weights, dim=0)
        
        # Upsample and combine
        reconstructed = torch.zeros_like(x)
        for i, (comp, weight) in enumerate(zip(components, weights)):
            if comp.size(-1) < L:
                comp = torch.nn.functional.interpolate(comp, size=L, mode='linear', align_corners=False)
            reconstructed += comp * weight
            
        return reconstructed.transpose(1, 2), components


class MetaLearningAdapter(SamplingComponent):
    """Meta-learning adapter for quick adaptation to new patterns"""
    
    def __init__(self, config: SamplingConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="MetaLearningAdapter",
            component_type=ComponentType.META_LEARNING_ADAPTER,
            required_params=['d_model'],
            optional_params=['adaptation_steps'],
            description="Meta-learning adapter for quick adaptation to new time series patterns"
        )
    
    def _initialize_component(self, **kwargs):
        self.d_model = getattr(self.config, 'd_model', 512)
        self.adaptation_steps = getattr(self.config, 'adaptation_steps', 5)
        
        # Fast adaptation parameters
        self.fast_weights = nn.ParameterList([
            nn.Parameter(torch.randn(self.d_model, self.d_model) * 0.01)
            for _ in range(self.adaptation_steps)
        ])
        
        self.fast_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(self.d_model))
            for _ in range(self.adaptation_steps)
        ])
        
        # Meta-learning rate
        self.meta_lr = nn.Parameter(torch.ones(1) * 0.01)
        
    def forward(self, model_fn, x=None, support_set=None, **kwargs):
        if support_set is not None and self.training:
            # Fast adaptation using support set
            adapted_weights = []
            adapted_biases = []
            
            for i in range(self.adaptation_steps):
                # Compute gradients on support set
                loss = torch.nn.functional.mse_loss(
                    torch.nn.functional.linear(support_set, self.fast_weights[i], self.fast_biases[i]),
                    support_set
                )
                
                # Update fast weights
                grad_w = torch.autograd.grad(loss, self.fast_weights[i], create_graph=True)[0]
                grad_b = torch.autograd.grad(loss, self.fast_biases[i], create_graph=True)[0]
                
                adapted_w = self.fast_weights[i] - self.meta_lr * grad_w
                adapted_b = self.fast_biases[i] - self.meta_lr * grad_b
                
                adapted_weights.append(adapted_w)
                adapted_biases.append(adapted_b)
            
            # Use adapted weights for prediction
            prediction = model_fn()
            for w, b in zip(adapted_weights, adapted_biases):
                prediction = torch.nn.functional.linear(prediction, w, b)
                prediction = torch.nn.functional.relu(prediction)
            
            return {
                'prediction': prediction,
                'uncertainty': None,
                'adapted_weights': adapted_weights
            }
        else:
            # Standard forward pass
            prediction = model_fn()
            for w, b in zip(self.fast_weights, self.fast_biases):
                prediction = torch.nn.functional.linear(prediction, w, b)
                prediction = torch.nn.functional.relu(prediction)
                
            return {
                'prediction': prediction,
                'uncertainty': None
            }


class TemporalConvEncoder(EncoderComponent):
    """Temporal Convolutional Network encoder for sequence modeling"""
    
    def __init__(self, config: EncoderConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="TemporalConvEncoder",
            component_type=ComponentType.TEMPORAL_CONV_ENCODER,
            required_params=['input_size', 'num_channels'],
            optional_params=['kernel_size', 'dropout'],
            description="Temporal Convolutional Network for sequence modeling"
        )
    
    def _initialize_component(self, **kwargs):
        self.input_size = getattr(self.config, 'input_size', getattr(self.config, 'd_model', 512))
        self.num_channels = getattr(self.config, 'num_channels', [64, 128, 256])
        self.kernel_size = getattr(self.config, 'kernel_size', 2)
        self.dropout = getattr(self.config, 'dropout', 0.2)
        
        layers = []
        num_levels = len(self.num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = self.input_size if i == 0 else self.num_channels[i-1]
            out_channels = self.num_channels[i]
            
            # Causal convolution
            padding = (self.kernel_size - 1) * dilation_size
            layers.append(nn.Conv1d(
                in_channels, out_channels, self.kernel_size,
                padding=padding, dilation=dilation_size
            ))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            
        self.network = nn.Sequential(*layers)
        
    def forward(self, x, attn_mask=None):
        # x: [B, L, D] -> [B, D, L] for conv1d
        x = x.transpose(1, 2)
        x = self.network(x)
        
        # Remove future information (causal)
        if x.size(-1) > x.size(-1):
            x = x[:, :, :-(x.size(-1) - x.size(-1))]
            
        # [B, D, L] -> [B, L, D]
        x = x.transpose(1, 2)
        
        return x, None


class AdaptiveMixtureSampling(SamplingComponent):
    """Adaptive mixture of experts for different time series patterns"""
    
    def __init__(self, config: SamplingConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="AdaptiveMixtureSampling",
            component_type=ComponentType.ADAPTIVE_MIXTURE,
            required_params=['d_model'],
            optional_params=['num_experts'],
            description="Adaptive mixture of experts for different time series patterns"
        )
    
    def _initialize_component(self, **kwargs):
        self.d_model = getattr(self.config, 'd_model', 512)
        self.num_experts = getattr(self.config, 'num_experts', 4)
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, self.d_model * 2),
                nn.ReLU(),
                nn.Linear(self.d_model * 2, self.d_model)
            ) for _ in range(self.num_experts)
        ])
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, self.num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, model_fn, x=None, **kwargs):
        # Get base prediction
        prediction = model_fn()
        
        # Compute gating weights
        gate_weights = self.gate(prediction)  # [B, L, num_experts]
        
        # Compute expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(prediction))
        
        expert_outputs = torch.stack(expert_outputs, dim=-1)  # [B, L, d_model, num_experts]
        
        # Weighted combination
        output = torch.sum(expert_outputs * gate_weights.unsqueeze(-2), dim=-1)
        
        return {
            'prediction': output,
            'uncertainty': torch.std(expert_outputs, dim=-1),
            'expert_weights': gate_weights
        }


class FocalLossComponent(LossComponent):
    """Focal loss for addressing class imbalance in time series forecasting"""
    
    def __init__(self, config: LossConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="FocalLoss",
            component_type=ComponentType.FOCAL_LOSS,
            required_params=[],
            optional_params=['alpha', 'gamma'],
            description="Focal loss for handling imbalanced forecasting scenarios"
        )
    
    def _initialize_component(self, **kwargs):
        self.alpha = getattr(self.config, 'alpha', 1.0)
        self.gamma = getattr(self.config, 'gamma', 2.0)
        
    def forward(self, predictions, targets, **kwargs):
        mse = torch.nn.functional.mse_loss(predictions, targets, reduction='none')
        p_t = torch.exp(-mse)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        return torch.mean(focal_weight * mse)


class AdaptiveAutoformerLossComponent(LossComponent):
    """Adaptive loss with trend/seasonal decomposition weighting"""
    
    def __init__(self, config: LossConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="AdaptiveAutoformerLoss",
            component_type=ComponentType.ADAPTIVE_AUTOFORMER_LOSS,
            required_params=[],
            optional_params=['moving_avg', 'initial_trend_weight', 'initial_seasonal_weight'],
            description="Adaptive loss with learnable trend/seasonal weighting"
        )
    
    def _initialize_component(self, **kwargs):
        self.moving_avg = getattr(self.config, 'moving_avg', 25)
        self.initial_trend_weight = getattr(self.config, 'initial_trend_weight', 1.0)
        self.initial_seasonal_weight = getattr(self.config, 'initial_seasonal_weight', 1.0)
        
        # Import decomposition
        try:
            from layers.Autoformer_EncDec import series_decomp
            self.decomp = series_decomp(kernel_size=self.moving_avg)
        except ImportError:
            # Fallback simple decomposition
            self.decomp = nn.AvgPool1d(kernel_size=self.moving_avg, stride=1, padding=self.moving_avg//2)
        
        # Learnable weights
        self.trend_weight = nn.Parameter(torch.tensor(self.initial_trend_weight))
        self.seasonal_weight = nn.Parameter(torch.tensor(self.initial_seasonal_weight))
        
    def forward(self, predictions, targets, **kwargs):
        # Decompose both predictions and targets
        if hasattr(self.decomp, '__call__') and hasattr(self.decomp, 'kernel_size'):
            # Using series_decomp
            pred_seasonal, pred_trend = self.decomp(predictions)
            true_seasonal, true_trend = self.decomp(targets)
        else:
            # Fallback decomposition
            pred_trend = self.decomp(predictions.transpose(1, 2)).transpose(1, 2)
            true_trend = self.decomp(targets.transpose(1, 2)).transpose(1, 2)
            pred_seasonal = predictions - pred_trend
            true_seasonal = targets - true_trend
        
        # Component-wise losses
        trend_loss = torch.nn.functional.mse_loss(pred_trend, true_trend)
        seasonal_loss = torch.nn.functional.mse_loss(pred_seasonal, true_seasonal)
        
        # Apply adaptive weighting with softplus for positivity
        trend_w = torch.nn.functional.softplus(self.trend_weight)
        seasonal_w = torch.nn.functional.softplus(self.seasonal_weight)
        
        # Total adaptive loss
        total_loss = trend_w * trend_loss + seasonal_w * seasonal_loss
        
        return total_loss


# =============================
# ADVANCED COMPONENT REGISTRATION
# =============================

component_registry.register_component(
    ComponentType.ADVANCED_WAVELET,
    AdvancedWaveletDecomposition,
    ComponentMetadata(
        name="AdvancedWaveletDecomposition",
        component_type=ComponentType.ADVANCED_WAVELET,
        required_params=['input_dim'],
        optional_params=['levels'],
        description="Advanced learnable wavelet decomposition for multi-resolution analysis"
    )
)

component_registry.register_component(
    ComponentType.META_LEARNING_ADAPTER,
    MetaLearningAdapter,
    ComponentMetadata(
        name="MetaLearningAdapter",
        component_type=ComponentType.META_LEARNING_ADAPTER,
        required_params=['d_model'],
        optional_params=['adaptation_steps'],
        description="Meta-learning adapter for quick adaptation to new time series patterns"
    )
)

component_registry.register_component(
    ComponentType.TEMPORAL_CONV_ENCODER,
    TemporalConvEncoder,
    ComponentMetadata(
        name="TemporalConvEncoder",
        component_type=ComponentType.TEMPORAL_CONV_ENCODER,
        required_params=['input_size', 'num_channels'],
        optional_params=['kernel_size', 'dropout'],
        description="Temporal Convolutional Network for sequence modeling"
    )
)

component_registry.register_component(
    ComponentType.ADAPTIVE_MIXTURE,
    AdaptiveMixtureSampling,
    ComponentMetadata(
        name="AdaptiveMixtureSampling",
        component_type=ComponentType.ADAPTIVE_MIXTURE,
        required_params=['d_model'],
        optional_params=['num_experts'],
        description="Adaptive mixture of experts for different time series patterns"
    )
)

component_registry.register_component(
    ComponentType.FOCAL_LOSS,
    FocalLossComponent,
    ComponentMetadata(
        name="FocalLoss",
        component_type=ComponentType.FOCAL_LOSS,
        required_params=[],
        optional_params=['alpha', 'gamma'],
        description="Focal loss for handling imbalanced forecasting scenarios"
    )
)

component_registry.register_component(
    ComponentType.ADAPTIVE_AUTOFORMER_LOSS,
    AdaptiveAutoformerLossComponent,
    ComponentMetadata(
        name="AdaptiveAutoformerLoss",
        component_type=ComponentType.ADAPTIVE_AUTOFORMER_LOSS,
        required_params=[],
        optional_params=['moving_avg', 'initial_trend_weight', 'initial_seasonal_weight'],
        description="Adaptive loss with learnable trend/seasonal weighting"
    )
)


# ============================================================================
# Advanced Loss Components (Phase 1 Integration)
# ============================================================================

class MAPELossComponent(LossComponent):
    """Mean Absolute Percentage Error loss component"""
    
    def __init__(self, config: LossConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="MAPELoss",
            component_type=ComponentType.MAPE_LOSS,
            required_params=[],
            optional_params=[],
            description="Mean Absolute Percentage Error for percentage-based forecasting"
        )
    
    def _initialize_component(self, **kwargs):
        from layers.modular.losses.advanced_losses import MAPELoss
        self.loss_fn = MAPELoss()
    
    def forward(self, predictions, targets, **kwargs):
        return self.loss_fn.forward(predictions, targets, **kwargs)


class SMAPELossComponent(LossComponent):
    """Symmetric Mean Absolute Percentage Error loss component"""
    
    def __init__(self, config: LossConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="SMAPELoss",
            component_type=ComponentType.SMAPE_LOSS,
            required_params=[],
            optional_params=[],
            description="Symmetric Mean Absolute Percentage Error for robust percentage forecasting"
        )
    
    def _initialize_component(self, **kwargs):
        from layers.modular.losses.advanced_losses import SMAPELoss
        self.loss_fn = SMAPELoss()
    
    def forward(self, predictions, targets, **kwargs):
        return self.loss_fn.forward(predictions, targets, **kwargs)


class MASELossComponent(LossComponent):
    """Mean Absolute Scaled Error loss component"""
    
    def __init__(self, config: LossConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="MASELoss",
            component_type=ComponentType.MASE_LOSS,
            required_params=[],
            optional_params=['freq'],
            description="Mean Absolute Scaled Error for scale-invariant forecasting evaluation"
        )
    
    def _initialize_component(self, **kwargs):
        from layers.modular.losses.advanced_losses import MASELoss
        self.freq = getattr(self.config, 'freq', 1)
        self.loss_fn = MASELoss(freq=self.freq)
    
    def forward(self, predictions, targets, **kwargs):
        return self.loss_fn.forward(predictions, targets, **kwargs)


class PSLossComponent(LossComponent):
    """Patch-wise Structural Loss component"""
    
    def __init__(self, config: LossConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="PSLoss",
            component_type=ComponentType.PS_LOSS,
            required_params=['pred_len'],
            optional_params=['mse_weight', 'w_corr', 'w_var', 'w_mean', 'k_dominant_freqs'],
            description="Patch-wise Structural Loss for capturing temporal patterns and correlations"
        )
    
    def _initialize_component(self, **kwargs):
        from layers.modular.losses.advanced_losses import PSLoss
        self.pred_len = self.config.pred_len
        self.mse_weight = getattr(self.config, 'mse_weight', 0.5)
        self.w_corr = getattr(self.config, 'w_corr', 1.0)
        self.w_var = getattr(self.config, 'w_var', 1.0)
        self.w_mean = getattr(self.config, 'w_mean', 1.0)
        self.k_dominant_freqs = getattr(self.config, 'k_dominant_freqs', 3)
        
        self.loss_fn = PSLoss(
            pred_len=self.pred_len,
            mse_weight=self.mse_weight,
            w_corr=self.w_corr,
            w_var=self.w_var,
            w_mean=self.w_mean,
            k_dominant_freqs=self.k_dominant_freqs
        )
    
    def forward(self, predictions, targets, **kwargs):
        return self.loss_fn.forward(predictions, targets)


class FrequencyAwareLossComponent(LossComponent):
    """Frequency-aware loss component"""
    
    def __init__(self, config: LossConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="FrequencyAwareLoss",
            component_type=ComponentType.FREQUENCY_AWARE_LOSS,
            required_params=[],
            optional_params=['freq_bands', 'band_weights', 'base_loss'],
            description="Frequency-aware loss that emphasizes different frequency components"
        )
    
    def _initialize_component(self, **kwargs):
        from layers.modular.losses.adaptive_bayesian_losses import FrequencyAwareLoss
        self.freq_bands = getattr(self.config, 'freq_bands', None)
        self.band_weights = getattr(self.config, 'band_weights', None)
        self.base_loss = getattr(self.config, 'base_loss', 'mse')
        
        self.loss_fn = FrequencyAwareLoss(
            freq_bands=self.freq_bands,
            band_weights=self.band_weights,
            base_loss=self.base_loss
        )
    
    def forward(self, predictions, targets, **kwargs):
        return self.loss_fn.forward(predictions, targets)


class MultiQuantileLossComponent(LossComponent):
    """Multi-quantile loss component"""
    
    def __init__(self, config: LossConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="MultiQuantileLoss",
            component_type=ComponentType.MULTI_QUANTILE_LOSS,
            required_params=[],
            optional_params=['quantiles'],
            description="Multi-quantile loss for probabilistic forecasting with uncertainty bounds"
        )
    
    def _initialize_component(self, **kwargs):
        from layers.modular.losses.adaptive_bayesian_losses import QuantileLoss
        self.quantiles = getattr(self.config, 'quantiles', [0.1, 0.5, 0.9])
        self.loss_fn = QuantileLoss(quantiles=self.quantiles)
    
    def forward(self, predictions, targets, **kwargs):
        return self.loss_fn.forward(predictions, targets)


class FocalLossComponent(LossComponent):
    """Focal loss component for imbalanced forecasting"""
    
    def __init__(self, config: LossConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="FocalLoss",
            component_type=ComponentType.FOCAL_LOSS,
            required_params=[],
            optional_params=['alpha', 'gamma'],
            description="Focal loss for addressing imbalanced forecasting scenarios"
        )
    
    def _initialize_component(self, **kwargs):
        from layers.modular.losses.advanced_losses import FocalLoss
        self.alpha = getattr(self.config, 'alpha', 1.0)
        self.gamma = getattr(self.config, 'gamma', 2.0)
        self.loss_fn = FocalLoss(alpha=self.alpha, gamma=self.gamma)
    
    def forward(self, predictions, targets, **kwargs):
        return self.loss_fn.forward(predictions, targets)


class UncertaintyCalibrationLossComponent(LossComponent):
    """Uncertainty calibration loss component"""
    
    def __init__(self, config: LossConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="UncertaintyCalibrationLoss",
            component_type=ComponentType.UNCERTAINTY_CALIBRATION_LOSS,
            required_params=[],
            optional_params=['calibration_weight'],
            description="Loss for calibrating uncertainty estimates with actual prediction errors"
        )
    
    def _initialize_component(self, **kwargs):
        from layers.modular.losses.adaptive_bayesian_losses import UncertaintyCalibrationLoss
        self.calibration_weight = getattr(self.config, 'calibration_weight', 1.0)
        self.loss_fn = UncertaintyCalibrationLoss(calibration_weight=self.calibration_weight)
    
    def forward(self, predictions, targets, uncertainties=None, **kwargs):
        if uncertainties is None:
            raise ValueError("UncertaintyCalibrationLoss requires 'uncertainties' parameter")
        return self.loss_fn.forward(predictions, targets, uncertainties)


# Register advanced loss components
component_registry.register_component(
    ComponentType.MAPE_LOSS,
    MAPELossComponent,
    ComponentMetadata(
        name="MAPELoss",
        component_type=ComponentType.MAPE_LOSS,
        required_params=[],
        optional_params=[],
        description="Mean Absolute Percentage Error for percentage-based forecasting"
    )
)

component_registry.register_component(
    ComponentType.SMAPE_LOSS,
    SMAPELossComponent,
    ComponentMetadata(
        name="SMAPELoss",
        component_type=ComponentType.SMAPE_LOSS,
        required_params=[],
        optional_params=[],
        description="Symmetric Mean Absolute Percentage Error for robust percentage forecasting"
    )
)

component_registry.register_component(
    ComponentType.MASE_LOSS,
    MASELossComponent,
    ComponentMetadata(
        name="MASELoss",
        component_type=ComponentType.MASE_LOSS,
        required_params=[],
        optional_params=['freq'],
        description="Mean Absolute Scaled Error for scale-invariant forecasting evaluation"
    )
)

component_registry.register_component(
    ComponentType.PS_LOSS,
    PSLossComponent,
    ComponentMetadata(
        name="PSLoss",
        component_type=ComponentType.PS_LOSS,
        required_params=['pred_len'],
        optional_params=['mse_weight', 'w_corr', 'w_var', 'w_mean', 'k_dominant_freqs'],
        description="Patch-wise Structural Loss for capturing temporal patterns and correlations"
    )
)

component_registry.register_component(
    ComponentType.FOCAL_LOSS,
    FocalLossComponent,
    ComponentMetadata(
        name="FocalLoss",
        component_type=ComponentType.FOCAL_LOSS,
        required_params=[],
        optional_params=['alpha', 'gamma'],
        description="Focal loss for addressing imbalanced forecasting scenarios"
    )
)

component_registry.register_component(
    ComponentType.FREQUENCY_AWARE_LOSS,
    FrequencyAwareLossComponent,
    ComponentMetadata(
        name="FrequencyAwareLoss",
        component_type=ComponentType.FREQUENCY_AWARE_LOSS,
        required_params=[],
        optional_params=['freq_bands', 'band_weights', 'base_loss'],
        description="Frequency-aware loss that emphasizes different frequency components"
    )
)

component_registry.register_component(
    ComponentType.MULTI_QUANTILE_LOSS,
    MultiQuantileLossComponent,
    ComponentMetadata(
        name="MultiQuantileLoss",
        component_type=ComponentType.MULTI_QUANTILE_LOSS,
        required_params=[],
        optional_params=['quantiles'],
        description="Multi-quantile loss for probabilistic forecasting with uncertainty bounds"
    )
)

component_registry.register_component(
    ComponentType.UNCERTAINTY_CALIBRATION_LOSS,
    UncertaintyCalibrationLossComponent,
    ComponentMetadata(
        name="UncertaintyCalibrationLoss",
        component_type=ComponentType.UNCERTAINTY_CALIBRATION_LOSS,
        required_params=[],
        optional_params=['calibration_weight'],
        description="Loss for calibrating uncertainty estimates with actual prediction errors"
    )
)

