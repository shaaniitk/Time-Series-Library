"""
Flexible Autoformer for Custom Time Series Processing Paths
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class SeriesProcessingConfig:
    """Configuration for different series processing paths"""
    target_indices: List[int]  # Which series are targets (get wavelet + self-attention)
    covariate_indices: List[int]  # Which series are covariates (direct self-attention)
    wavelet_levels: int = 3
    d_model: int = 512
    n_heads: int = 8

class FlexibleAutoformer(nn.Module):
    """
    Autoformer with flexible processing paths for different time series
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Series routing configuration
        self.series_config = SeriesProcessingConfig(
            target_indices=getattr(config, 'target_indices', list(range(7))),
            covariate_indices=getattr(config, 'covariate_indices', []),
            wavelet_levels=getattr(config, 'wavelet_levels', 3),
            d_model=config.d_model,
            n_heads=config.n_heads
        )
        
        self._init_components()
    
    def _init_components(self):
        """Initialize processing components using existing modular components"""
        from layers.Embed import DataEmbedding_wo_pos
        from layers.modular.attention.registry import get_attention_component
        from layers.modular.decomposition.registry import get_decomposition_component
        
        # Embeddings
        self.enc_embedding = DataEmbedding_wo_pos(
            self.config.enc_in, self.config.d_model, 'timeF', 'h', self.config.dropout
        )
        
        # Wavelet decomposition for targets using existing component
        self.wavelet_decomp = get_decomposition_component(
            'wavelet_decomposition',
            d_model=len(self.series_config.target_indices),
            n_levels=self.series_config.wavelet_levels,
            wavelet_type='learnable'
        )
        
        # Self-attention components using existing registry
        self.target_self_attention = get_attention_component(
            'enhanced_autocorrelation',
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            dropout=self.config.dropout
        )
        
        self.covariate_self_attention = get_attention_component(
            'autocorrelation_layer',
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            dropout=self.config.dropout
        )
        
        # Cross-attention components
        self.target_covariate_cross_attention = get_attention_component(
            'autocorrelation_layer',
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            dropout=self.config.dropout
        )
        
        self.consolidated_cross_attention = get_attention_component(
            'enhanced_autocorrelation',
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            dropout=self.config.dropout
        )
        
        # Output projection
        self.projection = nn.Linear(self.config.d_model, len(self.series_config.target_indices))
        
        # Final decomposition using existing component
        self.final_decomp = get_decomposition_component(
            'series_decomposition',
            kernel_size=self.config.moving_avg
        )
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Forward pass with flexible routing
        """
        B, L, _ = x_enc.shape
        
        # Embed input
        x_embedded = self.enc_embedding(x_enc, x_mark_enc)  # [B, L, d_model]
        
        # Split series based on configuration
        target_series = x_enc[:, :, self.series_config.target_indices]  # [B, L, n_targets]
        covariate_series = x_enc[:, :, self.series_config.covariate_indices] if self.series_config.covariate_indices else None
        
        # Process targets through wavelet decomposition
        target_processed = self._process_targets_with_wavelet(target_series)  # [B, L, n_targets]
        
        # Embed processed targets
        target_embedded = self._embed_targets(target_processed, x_mark_enc)  # [B, L, d_model]
        
        # Self-attention on wavelet-processed targets
        target_self_attended, _ = self.target_self_attention(
            target_embedded, target_embedded, target_embedded
        )
        
        # Process covariates if they exist
        if covariate_series is not None:
            # Embed covariates
            covariate_embedded = self._embed_covariates(covariate_series, x_mark_enc)
            
            # Self-attention on covariates
            covariate_self_attended, _ = self.covariate_self_attention(
                covariate_embedded, covariate_embedded, covariate_embedded
            )
            
            # Cross-attention: targets attend to covariates
            target_cross_attended, _ = self.target_covariate_cross_attention(
                target_self_attended, covariate_self_attended, covariate_self_attended
            )
            
            # Consolidated cross-attention
            consolidated = torch.cat([target_cross_attended, covariate_self_attended], dim=-1)
            consolidated_attended, _ = self.consolidated_cross_attention(
                consolidated, consolidated, consolidated
            )
        else:
            consolidated_attended = target_self_attended
        
        # Final processing with decomposition
        seasonal, trend = self.final_decomp(consolidated_attended)
        final_output = seasonal + trend
        
        # Project to target dimensions
        output = self.projection(final_output)
        
        return output[:, -self.config.pred_len:, :]
    
    def _process_targets_with_wavelet(self, target_series):
        """Apply wavelet decomposition using existing component"""
        # Use existing wavelet decomposition component
        seasonal, trend = self.wavelet_decomp(target_series)
        # Combine seasonal and trend components
        return seasonal + trend
    
    def _embed_targets(self, target_processed, x_mark_enc):
        """Embed processed targets to d_model dimension"""
        B, L, n_targets = target_processed.shape
        
        # Simple linear projection to d_model
        target_proj = nn.Linear(n_targets, self.config.d_model).to(target_processed.device)
        return target_proj(target_processed)
    
    def _embed_covariates(self, covariate_series, x_mark_enc):
        """Embed covariates to d_model dimension"""
        B, L, n_covariates = covariate_series.shape
        
        # Simple linear projection to d_model
        covariate_proj = nn.Linear(n_covariates, self.config.d_model).to(covariate_series.device)
        return covariate_proj(covariate_series)

# Factory function for easy creation
def create_flexible_autoformer(
    target_indices: List[int],
    covariate_indices: List[int],
    seq_len: int = 96,
    pred_len: int = 24,
    total_features: int = 40,
    **kwargs
):
    """
    Create flexible autoformer with custom processing paths
    
    Args:
        target_indices: Indices of target series (get wavelet + self-attention)
        covariate_indices: Indices of covariate series (direct self-attention)
        seq_len: Input sequence length
        pred_len: Prediction length
        total_features: Total number of input features
    """
    from argparse import Namespace
    
    config = Namespace(
        seq_len=seq_len,
        pred_len=pred_len,
        label_len=seq_len // 2,
        enc_in=total_features,
        dec_in=total_features,
        c_out=len(target_indices),
        d_model=kwargs.get('d_model', 512),
        n_heads=kwargs.get('n_heads', 8),
        dropout=kwargs.get('dropout', 0.1),
        moving_avg=kwargs.get('moving_avg', 25),
        target_indices=target_indices,
        covariate_indices=covariate_indices,
        wavelet_levels=kwargs.get('wavelet_levels', 3)
    )
    
    return FlexibleAutoformer(config)

# Usage example
if __name__ == "__main__":
    # Example: 40 time series, first 10 are targets, rest are covariates
    model = create_flexible_autoformer(
        target_indices=list(range(10)),      # First 10 series as targets
        covariate_indices=list(range(10, 40)), # Remaining 30 as covariates
        seq_len=96,
        pred_len=24,
        total_features=40
    )
    
    # Test forward pass
    batch_size = 32
    x_enc = torch.randn(batch_size, 96, 40)
    x_mark_enc = torch.randn(batch_size, 96, 4)
    x_dec = torch.randn(batch_size, 48, 40)
    x_mark_dec = torch.randn(batch_size, 48, 4)
    
    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    print(f"Output shape: {output.shape}")  # Should be [32, 24, 10]