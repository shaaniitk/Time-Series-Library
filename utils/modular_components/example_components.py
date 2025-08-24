"""
Example Component Implementations with Cross-Functionality Dependencies

This module provides concrete implementations of modular components
that demonstrate the dependency metadata system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging
logger = logging.getLogger(__name__)

from utils.modular_components.base_interfaces import (
    BaseBackbone, BaseProcessor, BaseAttention, BaseLoss, BaseEmbedding, BaseOutput
)
from configs.schemas import ComponentConfig


class ChronosBackbone(BaseBackbone):
    """Chronos-based backbone implementation"""
    
    def __init__(self, config):
        super().__init__(config)
        self.d_model = getattr(config, 'd_model', 512)
        self.seq_len = getattr(config, 'seq_len', 96)
        
        # Simplified transformer layers
        self.encoder_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=3
        )
        
        self.set_info('backbone_type', 'chronos')
        self.set_info('supports_seq2seq', True)
    
    def forward(self, x, mask=None):
        # x shape: [batch, seq_len, d_model]
        return self.encoder_layers(x, src_key_padding_mask=mask)
    
    def get_d_model(self) -> int:
        return self.d_model
    
    def supports_seq2seq(self) -> bool:
        return True
    
    def get_backbone_type(self) -> str:
        return 'chronos'
    
    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ['time_domain', 'frequency_domain', 'transformer_based', 'seq2seq']
    
    @classmethod
    def get_compatibility_tags(cls) -> List[str]:
        return ['huggingface_compatible', 'attention_based', 'autoregressive']


class TimeDomainProcessor(BaseProcessor):
    """Time domain processing strategy"""
    
    def __init__(self, config):
        super().__init__(config)
        self.d_model = getattr(config, 'd_model', 512)
        self.pred_len = getattr(config, 'pred_len', 24)
        
        # Simple projection for demonstration
        self.projection = nn.Linear(self.d_model, self.d_model)
        self.set_info('processor_type', 'time_domain')
    
    def forward(self, embedded_input, backbone_output, target_length, **kwargs):
        # Simple time domain processing
        # backbone_output shape: [batch, seq_len, d_model]
        
        # Pool to target length
        if backbone_output.size(1) != target_length:
            # Simple average pooling and repeat
            pooled = backbone_output.mean(dim=1, keepdim=True)  # [batch, 1, d_model]
            output = pooled.repeat(1, target_length, 1)  # [batch, target_length, d_model]
        else:
            output = backbone_output
        
        return self.projection(output)
    
    def process_sequence(self, embedded_input, backbone_output, target_length, **kwargs):
        return self.forward(embedded_input, backbone_output, target_length, **kwargs)
    
    def get_processor_type(self) -> str:
        return 'time_domain'
    
    def get_output_dim(self) -> int:
        return self.d_model
    
    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ['time_domain', 'causal', 'autoregressive']
    
    @classmethod
    def get_requirements(cls) -> Dict[str, str]:
        return {
            'attention': 'time_domain_compatible'
        }


class FrequencyDomainProcessor(BaseProcessor):
    """Frequency domain processing strategy"""
    
    def __init__(self, config):
        super().__init__(config)
        self.d_model = getattr(config, 'd_model', 512)
        self.pred_len = getattr(config, 'pred_len', 24)
        
        # FFT-based processing
        self.frequency_projection = nn.Linear(self.d_model, self.d_model)
        self.inverse_projection = nn.Linear(self.d_model, self.d_model)
        
        self.set_info('processor_type', 'frequency_domain')
    
    def forward(self, embedded_input, backbone_output, target_length, **kwargs):
        # Frequency domain processing
        batch_size, seq_len, d_model = backbone_output.shape
        
        # Apply FFT (simplified)
        # In reality, we'd do proper frequency domain processing
        freq_features = self.frequency_projection(backbone_output)
        
        # Process in frequency domain
        freq_processed = F.relu(freq_features)
        
        # Convert back to time domain
        time_features = self.inverse_projection(freq_processed)
        
        # Adapt to target length
        if seq_len != target_length:
            # Simple interpolation
            time_features = F.interpolate(
                time_features.transpose(1, 2), 
                size=target_length, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
        
        return time_features
    
    def process_sequence(self, embedded_input, backbone_output, target_length, **kwargs):
        return self.forward(embedded_input, backbone_output, target_length, **kwargs)
    
    def get_processor_type(self) -> str:
        return 'frequency_domain'
    
    def get_output_dim(self) -> int:
        return self.d_model
    
    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ['frequency_domain', 'fourier_based', 'spectral_analysis']
    
    @classmethod
    def get_requirements(cls) -> Dict[str, str]:
        return {
            'attention': 'frequency_domain_compatible'
        }


class MultiHeadAttention(BaseAttention):
    """Standard multi-head attention mechanism"""
    
    def __init__(self, config):
        super().__init__(config)
        self.d_model = getattr(config, 'd_model', 512)
        self.n_heads = getattr(config, 'n_heads', 8)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.n_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.set_info('attention_type', 'multi_head')
    
    def forward(self, queries, keys, values, attn_mask=None):
        return self.apply_attention(queries, keys, values, attn_mask)
    
    def apply_attention(self, queries, keys, values, attn_mask=None):
        attn_output, attn_weights = self.attention(
            queries, keys, values, 
            key_padding_mask=attn_mask
        )
        return attn_output, attn_weights
    
    def get_attention_type(self) -> str:
        return 'multi_head'
    
    def get_output_dim(self) -> int:
        return self.d_model
    
    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ['time_domain_compatible', 'standard_attention', 'parallel']
    
    @classmethod
    def get_compatibility_tags(cls) -> List[str]:
        return ['transformer_standard', 'pytorch_native']


class AutoCorrAttention(BaseAttention):
    """Auto-correlation attention mechanism (frequency domain compatible)"""
    
    def __init__(self, config):
        super().__init__(config)
        self.d_model = getattr(config, 'd_model', 512)
        self.factor = getattr(config, 'factor', 3)
        
        # Auto-correlation specific layers
        self.query_projection = nn.Linear(self.d_model, self.d_model)
        self.key_projection = nn.Linear(self.d_model, self.d_model)
        self.value_projection = nn.Linear(self.d_model, self.d_model)
        self.out_projection = nn.Linear(self.d_model, self.d_model)
        
        self.set_info('attention_type', 'autocorr')
    
    def forward(self, queries, keys, values, attn_mask=None):
        return self.apply_attention(queries, keys, values, attn_mask)
    
    def apply_attention(self, queries, keys, values, attn_mask=None):
        # Simplified auto-correlation
        batch_size, seq_len, d_model = queries.shape
        
        Q = self.query_projection(queries)
        K = self.key_projection(keys)
        V = self.value_projection(values)
        
        # Auto-correlation computation (simplified)
        # In reality, this would involve FFT-based correlation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_model)
        
        if attn_mask is not None:
            scores.masked_fill_(attn_mask.unsqueeze(1), -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        return self.out_projection(attn_output), attn_weights
    
    def get_attention_type(self) -> str:
        return 'autocorr'
    
    def get_output_dim(self) -> int:
        return self.d_model
    
    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ['frequency_domain_compatible', 'autocorr_based', 'time_series_optimized']
    
    @classmethod
    def get_compatibility_tags(cls) -> List[str]:
        return ['autoformer_style', 'correlation_based']


class BayesianMSELoss(BaseLoss):
    """Bayesian MSE loss with KL divergence"""
    
    def __init__(self, config):
        super().__init__(config)
        self.kl_weight = getattr(config, 'kl_weight', 1e-5)
        self.base_loss = nn.MSELoss()
        
        self.set_info('loss_type', 'bayesian_mse')
    
    def forward(self, predictions, targets, model=None):
        # Standard MSE loss
        mse_loss = self.base_loss(predictions, targets)
        
        # Add KL divergence if model supports it
        kl_loss = 0.0
        if model and hasattr(model, 'get_kl_loss'):
            kl_loss = model.get_kl_loss()
        
        return mse_loss + self.kl_weight * kl_loss
    
    def get_loss_type(self) -> str:
        return 'bayesian_mse'
    
    def get_output_dim(self) -> int:
        return 1  # Scalar loss
    
    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ['bayesian', 'mse_based', 'uncertainty_aware']
    
    @classmethod
    def get_requirements(cls) -> Dict[str, str]:
        return {
            'backbone': 'bayesian_compatible'
        }


class StandardEmbedding(BaseEmbedding):
    """Standard embedding with positional encoding"""
    
    def __init__(self, config):
        super().__init__(config)
        self.enc_in = getattr(config, 'enc_in', 7)
        self.d_model = getattr(config, 'd_model', 512)
        
        self.value_embedding = nn.Linear(self.enc_in, self.d_model)
        self.position_embedding = nn.Parameter(torch.randn(1000, self.d_model))
        self.dropout = nn.Dropout(0.1)
        
        self.set_info('embedding_type', 'standard')
    
    def forward(self, x, x_mark=None):
        return self.embed_sequence(x, x_mark)
    
    def embed_sequence(self, x, x_mark=None):
        batch_size, seq_len, _ = x.shape
        
        # Value embedding
        value_emb = self.value_embedding(x)
        
        # Position embedding
        pos_emb = self.position_embedding[:seq_len, :].unsqueeze(0)
        pos_emb = pos_emb.expand(batch_size, -1, -1)
        
        # Combine embeddings
        embedded = value_emb + pos_emb
        
        return self.dropout(embedded)
    
    def get_embedding_type(self) -> str:
        return 'standard'
    
    def get_output_dim(self) -> int:
        return self.d_model
    
    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ['positional_encoding', 'linear_projection', 'standard']
    
    @classmethod
    def get_compatibility_tags(cls) -> List[str]:
        return ['transformer_compatible', 'standard_embedding']


# Import ChronosX components
from .chronos_backbone import (
    ChronosXBackbone, 
    ChronosXTinyBackbone, 
    ChronosXLargeBackbone, 
    ChronosXUncertaintyBackbone
)

# Registry helper function
class LinearOutput(BaseOutput):
    """Simple linear output projection component"""
    
    def __init__(self, config: ComponentConfig):
        super().__init__(config)
        self.d_model = getattr(config, 'd_model', 256)
        self.output_dim = getattr(config, 'output_dim', 1)
        
        self.projection = nn.Linear(self.d_model, self.output_dim)
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Project features to output dimension"""
        return self.projection(x)
    
    def get_output_type(self) -> str:
        return 'linear'
    
    def get_output_dim(self) -> int:
        return self.output_dim
    
    @classmethod
    def get_capabilities(cls) -> List[str]:
        return [
            'linear_projection',
            'dimensionality_reduction',
            'regression_output'
        ]
    
    @classmethod
    def get_requirements(cls) -> List[str]:
        return [
            'tensor_input',
            'feature_dimension_match'
        ]


class MultiOutputHead(BaseOutput):
    """Multi-task output head with uncertainty"""
    
    def __init__(self, config: ComponentConfig):
        super().__init__(config)
        self.d_model = getattr(config, 'd_model', 256)
        self.num_outputs = getattr(config, 'num_outputs', 3)  # value, lower, upper
        
        self.output_projection = nn.Linear(self.d_model, self.num_outputs)
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Multi-output projection with uncertainty bounds"""
        return self.output_projection(x)
    
    def get_output_type(self) -> str:
        return 'multi_output'
    
    def get_output_dim(self) -> int:
        return self.num_outputs
    
    @classmethod
    def get_capabilities(cls) -> List[str]:
        return [
            'multi_task_output',
            'uncertainty_bounds',
            'quantile_prediction'
        ]
    
    @classmethod
    def get_requirements(cls) -> List[str]:
        return [
            'tensor_input',
            'uncertainty_compatible'
        ]


def register_example_components(registry):
    """Register example components with the registry"""
    
    # Register backbones
    registry.register('backbone', 'chronos', ChronosBackbone, {
        'description': 'Chronos-based transformer backbone',
        'supports_seq2seq': True,
        'typical_d_model': 512
    })
    
    # Register ChronosX variants
    registry.register('backbone', 'chronos_x', ChronosXBackbone, {
        'description': 'ChronosX-based backbone with HF Transformers',
        'supports_seq2seq': True,
        'supports_uncertainty': True,
        'supports_quantiles': True,
        'model_family': 'chronos',
        'typical_d_model': 512
    })
    
    registry.register('backbone', 'chronos_x_tiny', ChronosXTinyBackbone, {
        'description': 'Tiny ChronosX model for fast experimentation',
        'supports_seq2seq': True,
        'supports_uncertainty': True,
        'model_family': 'chronos',
        'typical_d_model': 256,
        'speed': 'fast'
    })
    
    registry.register('backbone', 'chronos_x_large', ChronosXLargeBackbone, {
        'description': 'Large ChronosX model for maximum performance',
        'supports_seq2seq': True,
        'supports_uncertainty': True,
        'model_family': 'chronos',
        'typical_d_model': 1024,
        'performance': 'high'
    })
    
    registry.register('backbone', 'chronos_x_uncertainty', ChronosXUncertaintyBackbone, {
        'description': 'ChronosX optimized for uncertainty quantification',
        'supports_seq2seq': True,
        'supports_uncertainty': True,
        'supports_epistemic': True,
        'supports_aleatoric': True,
        'model_family': 'chronos',
        'typical_d_model': 512,
        'specialty': 'uncertainty'
    })
    
    # Register processors
    registry.register('processor', 'time_domain', TimeDomainProcessor, {
        'description': 'Time domain processing strategy',
        'compatible_attention': ['multi_head', 'sparse']
    })
    
    registry.register('processor', 'frequency_domain', FrequencyDomainProcessor, {
        'description': 'Frequency domain processing strategy',
        'compatible_attention': ['autocorr', 'fourier']
    })
    
    # Register attention mechanisms
    registry.register('attention', 'multi_head', MultiHeadAttention, {
        'description': 'Standard multi-head attention',
        'compatible_processors': ['time_domain']
    })
    
    registry.register('attention', 'autocorr', AutoCorrAttention, {
        'description': 'Auto-correlation attention mechanism',
        'compatible_processors': ['frequency_domain', 'time_domain']
    })
    
    # Register loss functions
    registry.register('loss', 'bayesian_mse', BayesianMSELoss, {
        'description': 'Bayesian MSE loss with KL divergence',
        'requires_bayesian_model': True
    })
    
    # Register embeddings
    registry.register('embedding', 'standard', StandardEmbedding, {
        'description': 'Standard embedding with positional encoding'
    })
    
    # Register output components
    registry.register('output', 'linear', LinearOutput, {
        'description': 'Simple linear output projection',
        'supports_regression': True
    })
    
    registry.register('output', 'multi_output', MultiOutputHead, {
        'description': 'Multi-task output head with uncertainty bounds',
        'supports_uncertainty': True
    })
    
    logger.info("Registered example components with dependency metadata")
