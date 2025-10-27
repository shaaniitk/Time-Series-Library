"""
Target Autocorrelation Module for Log Returns

This module captures target-specific patterns, autocorrelations, and cross-correlations
for financial log returns data (ln(Ot/Oprev_t)). It models:
1. Log return temporal dependencies (momentum, mean reversion around 0)
2. Log return cross-correlations (OHLC log return relationships)
3. Volatility clustering (GARCH-like effects)
4. Return distribution properties (fat tails, skewness)
5. Calendar effects integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class LogReturnConstraintLayer(nn.Module):
    """
    Learns and models log return relationships and constraints:
    - Log return correlations between OHLC
    - Volatility clustering (GARCH-like effects)
    - Return distribution properties (fat tails, skewness)
    - Mean reversion around zero for log returns
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Learn log return correlation matrix (4x4 for OHLC log returns)
        self.log_return_correlation = nn.Parameter(torch.eye(4) * 0.1)  # Start with small correlations
        
        # Volatility clustering modeling (GARCH-like)
        self.volatility_clustering = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus()  # Ensure positive volatility
        )
        
        # Mean reversion strength (log returns should revert to 0)
        self.mean_reversion_strength = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()  # Reversion strength between 0 and 1
        )
        
        # Fat tail modeling (capture extreme return events)
        self.fat_tail_encoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus()  # Positive tail parameter
        )
        
        # Skewness modeling (asymmetric return distributions)
        self.skewness_encoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Tanh()  # Skewness can be positive or negative
        )
        
    def forward(self, target_features: torch.Tensor) -> torch.Tensor:
        """
        Apply log return constraints and relationships
        
        Args:
            target_features: [batch, seq_len, d_model]
            
        Returns:
            Log return constraint-aware target features
        """
        batch_size, seq_len, _ = target_features.shape
        
        # Extract log return properties
        volatility_clustering = self.volatility_clustering(target_features)  # [batch, seq_len, 1]
        mean_reversion = self.mean_reversion_strength(target_features)  # [batch, seq_len, 1]
        fat_tail_param = self.fat_tail_encoder(target_features)  # [batch, seq_len, 1]
        skewness_param = self.skewness_encoder(target_features)  # [batch, seq_len, 1]
        
        # Apply learned log return correlations
        reshaped_features = target_features.reshape(-1, self.d_model)
        
        # Apply log return correlation matrix
        correlation_weight = torch.sigmoid(self.log_return_correlation.sum())
        enhanced_features = reshaped_features * correlation_weight
        
        # Incorporate log return properties
        # Volatility clustering effect
        enhanced_features = enhanced_features * (1 + volatility_clustering.view(-1, 1) * 0.1)
        
        # Mean reversion toward zero (log returns should center around 0)
        enhanced_features = enhanced_features * (1 - mean_reversion.view(-1, 1) * 0.05)
        
        # Fat tail and skewness effects
        enhanced_features = enhanced_features + fat_tail_param.view(-1, 1) * 0.02
        enhanced_features = enhanced_features + skewness_param.view(-1, 1) * 0.02
        
        return enhanced_features.reshape(batch_size, seq_len, self.d_model)


class TargetAutocorrelationModule(nn.Module):
    """
    Captures target-specific autocorrelations and patterns for financial time series.
    
    This module specifically models:
    1. Price momentum and mean reversion
    2. OHLC interdependencies
    3. Multi-timeframe patterns
    4. Volatility clustering
    """
    
    def __init__(
        self,
        d_model: int,
        num_targets: int = 4,  # OHLC
        hidden_dim: int = None,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_targets = num_targets
        self.hidden_dim = hidden_dim or d_model
        
        # Target-specific feature extraction
        self.target_projection = nn.Linear(d_model, self.hidden_dim)
        
        # Autocorrelation modeling with LSTM
        self.autocorr_lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Log return constraint and correlation modeling
        self.log_return_constraint_layer = LogReturnConstraintLayer(self.hidden_dim)
        
        # Multi-scale temporal attention
        # Choose a number of attention heads compatible with hidden_dim to
        # avoid PyTorch assertion "embed_dim must be divisible by num_heads".
        # Prefer up to 8 heads but fall back to the largest divisor of hidden_dim.
        max_preferred_heads = min(8, self.hidden_dim)
        attn_heads = next((h for h in range(max_preferred_heads, 0, -1) if self.hidden_dim % h == 0), 1)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Price momentum modeling
        self.momentum_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim),
            nn.Tanh()  # Bounded momentum
        )
        
        # Mean reversion modeling
        self.mean_reversion_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim),
            nn.Sigmoid()  # Reversion strength
        )
        
        # Output projection back to d_model
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Residual connection weight
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(
        self, 
        target_features: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process target features to capture autocorrelations and patterns
        
        Args:
            target_features: [batch, seq_len, d_model] - decoder features
            target_mask: Optional mask for target positions
            
        Returns:
            Enhanced target features with autocorrelation modeling
        """
        batch_size, seq_len, _ = target_features.shape
        
        # Project to target-specific space
        target_proj = self.target_projection(target_features)  # [batch, seq_len, hidden_dim]
        
        # 1. Autocorrelation modeling with LSTM
        lstm_out, (hidden, cell) = self.autocorr_lstm(target_proj)  # [batch, seq_len, hidden_dim]
        
        # 2. Apply log return constraints and correlations
        constrained_features = self.log_return_constraint_layer(lstm_out)
        
        # 3. Multi-scale temporal attention (self-attention on target sequence)
        attended_features, attention_weights = self.temporal_attention(
            constrained_features, constrained_features, constrained_features,
            key_padding_mask=target_mask
        )
        
        # 4. Price momentum modeling
        momentum_features = self.momentum_encoder(attended_features)
        
        # 5. Mean reversion modeling
        reversion_strength = self.mean_reversion_encoder(attended_features)
        
        # Combine momentum and mean reversion
        combined_features = attended_features + momentum_features * reversion_strength
        
        # 6. Project back to d_model space
        enhanced_features = self.output_projection(combined_features)
        
        # 7. Residual connection with original features
        output = (
            torch.sigmoid(self.residual_weight) * enhanced_features +
            (1 - torch.sigmoid(self.residual_weight)) * target_features
        )
        
        return output
    
    def get_autocorr_diagnostics(self) -> dict:
        """Return diagnostic information about learned log return autocorrelations"""
        return {
            'log_return_correlation_matrix': self.log_return_constraint_layer.log_return_correlation.detach().cpu(),
            'residual_weight': torch.sigmoid(self.residual_weight).item(),
            'momentum_params': sum(p.numel() for p in self.momentum_encoder.parameters()),
            'reversion_params': sum(p.numel() for p in self.mean_reversion_encoder.parameters()),
            'volatility_clustering_params': sum(p.numel() for p in self.log_return_constraint_layer.volatility_clustering.parameters()),
            'fat_tail_params': sum(p.numel() for p in self.log_return_constraint_layer.fat_tail_encoder.parameters())
        }


class DualStreamDecoder(nn.Module):
    """
    Dual-stream decoder that processes celestial and target features separately
    before fusion for final prediction.
    """
    
    def __init__(
        self,
        d_model: int,
        num_targets: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_targets = num_targets
        
        # Target autocorrelation module
        self.target_autocorr = TargetAutocorrelationModule(
            d_model=d_model,
            num_targets=num_targets,
            dropout=dropout
        )
        
        # Celestial-target fusion attention
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
    def forward(
        self,
        decoder_features: torch.Tensor,
        celestial_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Process decoder features with dual-stream approach
        
        Args:
            decoder_features: [batch, seq_len, d_model] - standard decoder output
            celestial_features: [batch, seq_len, d_model] - celestial graph features
            
        Returns:
            Enhanced decoder features with target autocorrelation modeling
        """
        # Stream 1: Target autocorrelation processing
        target_enhanced = self.target_autocorr(decoder_features)
        
        # Stream 2: Celestial-target cross-attention
        celestial_attended, _ = self.fusion_attention(
            target_enhanced, celestial_features, celestial_features
        )
        
        # Fusion of both streams
        combined = torch.cat([target_enhanced, celestial_attended], dim=-1)
        fused_features = self.fusion_layer(combined)
        
        return fused_features