"""
EnhancedTargetProcessor - Improved target processing with explicit trend-seasonal decomposition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
from utils.logger import logger

from layers.MambaBlock import TargetMambaBlock
from layers.modular.decomposition.wavelet_decomposition import WaveletDecomposition


class TrendSeasonalDecomposer(nn.Module):
    """Explicit trend-seasonal decomposition for targets."""
    
    def __init__(self, input_dim: int, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # Learnable decomposition weights
        self.trend_conv = nn.Conv1d(input_dim, input_dim, kernel_size, padding=self.padding, groups=input_dim)
        self.seasonal_norm = nn.LayerNorm(input_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, L, D]
        Returns:
            trend, seasonal: Both [B, L, D]
        """
        # Extract trend using convolution
        x_transposed = x.transpose(1, 2)  # [B, D, L]
        trend_transposed = self.trend_conv(x_transposed)  # [B, D, L]
        trend = trend_transposed.transpose(1, 2)  # [B, L, D]
        
        # Seasonal = Original - Trend
        seasonal = x - trend
        seasonal = self.seasonal_norm(seasonal)
        
        return trend, seasonal


class EnhancedTargetProcessor(nn.Module):
    """
    Enhanced target processor with explicit trend-seasonal decomposition.
    
    Pipeline: Input → Trend-Seasonal Decomp → Wavelet Decomp → Mamba → Attention → Context
    """
    
    def __init__(
        self,
        num_targets: int,
        seq_len: int,
        d_model: int,
        wavelet_type: str = 'db4',
        wavelet_levels: int = 3,
        mamba_d_state: int = 64,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        attention_heads: int = 8,
        dropout: float = 0.1,
        trend_kernel_size: int = 25
    ):
        super().__init__()
        
        self.num_targets = num_targets
        self.seq_len = seq_len
        self.d_model = d_model
        self.wavelet_levels = wavelet_levels
        
        # Input projection
        self.input_projection = nn.Linear(num_targets, d_model)
        
        # Trend-Seasonal decomposition
        self.trend_seasonal_decomp = TrendSeasonalDecomposer(d_model, trend_kernel_size)
        
        # Wavelet decomposition for both trend and seasonal
        try:
            from layers.modular.decomposition.wavelet_decomposition import WaveletDecomposition
            self.trend_wavelet_decomp = WaveletDecomposition(seq_len, d_model, wavelet_type, wavelet_levels)
            self.seasonal_wavelet_decomp = WaveletDecomposition(seq_len, d_model, wavelet_type, wavelet_levels)
            logger.info(f"Using modular WaveletDecomposition for trend and seasonal")
        except ImportError:
            logger.warning("Using basic wavelet decomposition")
            self.trend_wavelet_decomp = self._create_basic_wavelet_decomp(wavelet_type, wavelet_levels)
            self.seasonal_wavelet_decomp = self._create_basic_wavelet_decomp(wavelet_type, wavelet_levels)
        
        # Mamba blocks for trend components
        self.trend_mamba_blocks = nn.ModuleList([
            TargetMambaBlock(d_model, d_model, mamba_d_state, mamba_d_conv, mamba_expand, dropout)
            for _ in range(wavelet_levels + 1)
        ])
        
        # Mamba blocks for seasonal components
        self.seasonal_mamba_blocks = nn.ModuleList([
            TargetMambaBlock(d_model, d_model, mamba_d_state, mamba_d_conv, mamba_expand, dropout)
            for _ in range(wavelet_levels + 1)
        ])
        
        # Gated fusion for combining trend and seasonal contexts
        self.context_fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Sigmoid()
        )
        
        logger.info(f"EnhancedTargetProcessor initialized with explicit trend-seasonal decomposition")
    
    def _create_basic_wavelet_decomp(self, wavelet_type: str, levels: int):
        """Basic wavelet decomposition fallback."""
        class BasicWaveletDecomp(nn.Module):
            def __init__(self, wavelet_type, levels):
                super().__init__()
                self.levels = levels
                
            def forward(self, x):
                batch_size, seq_len, features = x.shape
                level_len = seq_len // (self.levels + 1)
                components = []
                for i in range(self.levels + 1):
                    start_idx = i * level_len
                    end_idx = min((i + 1) * level_len, seq_len)
                    components.append(x[:, start_idx:end_idx, :])
                return components
        
        return BasicWaveletDecomp(wavelet_type, levels)
    
    def forward(
        self, 
        targets: torch.Tensor, 
        target_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass with explicit trend-seasonal processing.
        
        Args:
            targets: [B, L, num_targets]
            target_mask: Optional mask
            
        Returns:
            Dictionary with trend_context, seasonal_context, and fused_context
        """
        batch_size, seq_len, _ = targets.shape
        
        # Step 1: Project to d_model space
        projected_targets = self.input_projection(targets)  # [B, L, d_model]
        
        # Step 2: Trend-Seasonal decomposition
        trend_component, seasonal_component = self.trend_seasonal_decomp(projected_targets)
        
        logger.debug(f"Trend-seasonal decomp: trend {trend_component.shape}, seasonal {seasonal_component.shape}")
        
        # Step 3: Wavelet decomposition for both components
        try:
            trend_wavelet_components = self.trend_wavelet_decomp(trend_component)
            seasonal_wavelet_components = self.seasonal_wavelet_decomp(seasonal_component)
        except Exception as e:
            logger.error(f"Wavelet decomposition failed: {e}")
            trend_wavelet_components = [trend_component]
            seasonal_wavelet_components = [seasonal_component]
        
        # Step 4: Process trend components through Mamba
        trend_outputs = []
        for i, component in enumerate(trend_wavelet_components):
            if i < len(self.trend_mamba_blocks):
                mamba_out = self.trend_mamba_blocks[i](component, target_mask)
                trend_outputs.append(mamba_out)
        
        # Step 5: Process seasonal components through Mamba
        seasonal_outputs = []
        for i, component in enumerate(seasonal_wavelet_components):
            if i < len(self.seasonal_mamba_blocks):
                mamba_out = self.seasonal_mamba_blocks[i](component, target_mask)
                seasonal_outputs.append(mamba_out)
        
        # Step 6: Aggregate trend and seasonal contexts
        if trend_outputs:
            # Stack and pool trend outputs
            stacked_trend = torch.stack(trend_outputs, dim=2)  # [B, L, num_components, D]
            trend_context = stacked_trend.mean(dim=(1, 2))  # [B, D]
        else:
            trend_context = trend_component.mean(dim=1)
        
        if seasonal_outputs:
            # Stack and pool seasonal outputs
            stacked_seasonal = torch.stack(seasonal_outputs, dim=2)  # [B, L, num_components, D]
            seasonal_context = stacked_seasonal.mean(dim=(1, 2))  # [B, D]
        else:
            seasonal_context = seasonal_component.mean(dim=1)
        
        # Step 7: Gated fusion of trend and seasonal contexts
        combined_context = torch.cat([trend_context, seasonal_context], dim=-1)  # [B, 2*D]
        gate = self.context_fusion_gate(combined_context)
        
        # The gate decides the weight of the trend context.
        # (1 - gate) will be the weight for the seasonal context.
        fused_context = gate * trend_context + (1 - gate) * seasonal_context
        
        # Return detailed outputs
        outputs = {
            'trend_context': trend_context,
            'seasonal_context': seasonal_context,
            'fused_context': fused_context,
            'trend_component': trend_component,
            'seasonal_component': seasonal_component
        }
        
        logger.debug(f"Enhanced target processing complete: fused_context {fused_context.shape}")
        
        return outputs