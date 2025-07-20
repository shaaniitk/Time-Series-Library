"""
Wavelet-based Attention Components for Modular Autoformer

This module implements wavelet decomposition-based attention mechanisms
for multi-resolution time series analysis and hierarchical pattern capture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base import BaseAttention
from utils.logger import logger


class WaveletAttention(BaseAttention):
    """
    Wavelet-based attention using learnable wavelet decomposition.
    
    This component performs multi-resolution analysis by decomposing
    time series into different frequency bands using wavelet transforms.
    """
    
    def __init__(self, d_model, n_heads, levels=3, n_levels=None, wavelet_type='learnable', 
                 dropout=0.1):
        super(WaveletAttention, self).__init__()
        
        # Handle both levels and n_levels parameters for compatibility
        if n_levels is not None:
            levels = n_levels
        
        logger.info(f"Initializing WaveletAttention: d_model={d_model}, levels={levels}")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.levels = levels
        self.head_dim = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # Wavelet decomposition
        self.wavelet_decomp = WaveletDecomposition(d_model, levels)
        
        # Multi-head attention for each decomposition level
        self.level_attentions = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            for _ in range(levels + 1)  # +1 for final low-frequency component
        ])
        
        # Level fusion
        self.level_weights = nn.Parameter(torch.ones(levels + 1) / (levels + 1))
        self.fusion_proj = nn.Linear(d_model * (levels + 1), d_model)
        
        self.output_dim_multiplier = 1
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Forward pass using wavelet-based multi-resolution attention.
        
        Args:
            queries: [B, L, D] query tensor
            keys: [B, L, D] key tensor  
            values: [B, L, D] value tensor
            attn_mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        B, L, D = queries.shape
        
        # Wavelet decomposition of queries, keys, values
        _, q_components = self.wavelet_decomp(queries)
        _, k_components = self.wavelet_decomp(keys)
        _, v_components = self.wavelet_decomp(values)
        
        # Apply attention at each decomposition level
        level_outputs = []
        level_attentions = []
        
        for i, (q_comp, k_comp, v_comp, attention_layer) in enumerate(
            zip(q_components, k_components, v_components, self.level_attentions)
        ):
            # Ensure components have the right shape
            if q_comp.size(1) != L:
                q_comp = F.interpolate(q_comp.transpose(1, 2), size=L, mode='linear', align_corners=False).transpose(1, 2)
                k_comp = F.interpolate(k_comp.transpose(1, 2), size=L, mode='linear', align_corners=False).transpose(1, 2)
                v_comp = F.interpolate(v_comp.transpose(1, 2), size=L, mode='linear', align_corners=False).transpose(1, 2)
            
            # Apply attention
            level_out, level_attn = attention_layer(q_comp, k_comp, v_comp, attn_mask=attn_mask)
            level_outputs.append(level_out)
            level_attentions.append(level_attn)
        
        # Weighted fusion of multi-resolution outputs
        weights = F.softmax(self.level_weights, dim=0)
        
        # Concatenate and fuse
        concatenated = torch.cat(level_outputs, dim=-1)  # [B, L, D*(levels+1)]
        fused_output = self.fusion_proj(concatenated)     # [B, L, D]
        
        # Aggregate attention weights (simple average)
        avg_attention = torch.stack(level_attentions, dim=0).mean(dim=0)
        
        return fused_output, avg_attention


class WaveletDecomposition(nn.Module):
    """
    Learnable wavelet decomposition for multi-resolution analysis.
    
    This module learns wavelet-like filters to decompose time series
    into multiple frequency bands for hierarchical processing.
    """
    
    def __init__(self, input_dim, levels=3, kernel_size=4):
        super(WaveletDecomposition, self).__init__()
        logger.info(f"Initializing WaveletDecomposition: input_dim={input_dim}, levels={levels}")
        
        self.levels = levels
        self.kernel_size = kernel_size
        
        # Learnable wavelet filters (low-pass and high-pass)
        self.low_pass = nn.ModuleList([
            nn.Conv1d(input_dim, input_dim, kernel_size, stride=2, padding=kernel_size//2, groups=input_dim)
            for _ in range(levels)
        ])
        
        self.high_pass = nn.ModuleList([
            nn.Conv1d(input_dim, input_dim, kernel_size, stride=2, padding=kernel_size//2, groups=input_dim)
            for _ in range(levels)
        ])
        
        # Reconstruction weights for combining components
        self.recon_weights = nn.Parameter(torch.ones(levels + 1) / (levels + 1))
        
        self.output_dim_multiplier = 1
    
    def forward(self, x):
        """
        Perform wavelet decomposition.
        
        Args:
            x: [B, L, D] input tensor
            
        Returns:
            Tuple of (reconstructed, components_list)
        """
        B, L, D = x.shape
        x_conv = x.transpose(1, 2)  # [B, D, L] for conv1d
        
        components = []
        current = x_conv
        
        # Multi-level decomposition
        for level in range(self.levels):
            # Apply low-pass and high-pass filters
            low = self.low_pass[level](current)
            high = self.high_pass[level](current)
            
            # Store high-frequency component
            components.append(high.transpose(1, 2))  # Convert back to [B, L', D]
            
            # Continue with low-frequency component
            current = low
            
        # Store final low-frequency component
        components.append(current.transpose(1, 2))
        
        # Weighted reconstruction
        weights = F.softmax(self.recon_weights, dim=0)
        
        # Upsample and combine all components
        reconstructed = torch.zeros_like(x)
        for i, (comp, weight) in enumerate(zip(components, weights)):
            if comp.size(1) < L:
                # Upsample to original length
                comp_upsampled = F.interpolate(
                    comp.transpose(1, 2), size=L, mode='linear', align_corners=False
                ).transpose(1, 2)
            else:
                comp_upsampled = comp
                
            reconstructed += comp_upsampled * weight
            
        return reconstructed, components


class AdaptiveWaveletAttention(BaseAttention):
    """
    Adaptive wavelet attention with learnable decomposition levels.
    
    This component adaptively selects the optimal number of decomposition
    levels based on the input characteristics.
    """
    
    def __init__(self, d_model, n_heads, max_levels=5):
        super(AdaptiveWaveletAttention, self).__init__()
        logger.info(f"Initializing AdaptiveWaveletAttention: d_model={d_model}, max_levels={max_levels}")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_levels = max_levels
        
        # Level selection network
        self.level_selector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, max_levels),
            nn.Softmax(dim=-1)
        )
        
        # Multiple wavelet attention modules for different levels
        self.wavelet_attentions = nn.ModuleList([
            WaveletAttention(d_model, n_heads, levels=i+1)
            for i in range(max_levels)
        ])
        
        self.output_dim_multiplier = 1
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Forward pass with adaptive level selection.
        
        Args:
            queries: [B, L, D] query tensor
            keys: [B, L, D] key tensor
            values: [B, L, D] value tensor
            attn_mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        B, L, D = queries.shape
        
        # Select optimal decomposition levels based on input characteristics
        input_summary = queries.mean(dim=1)  # [B, D]
        level_weights = self.level_selector(input_summary)  # [B, max_levels]
        
        # Compute outputs for all levels
        level_outputs = []
        level_attentions = []
        
        for i, wavelet_attn in enumerate(self.wavelet_attentions):
            output, attention = wavelet_attn(queries, keys, values, attn_mask)
            level_outputs.append(output)
            level_attentions.append(attention)
        
        # Weighted combination based on adaptive selection
        level_outputs = torch.stack(level_outputs, dim=-1)  # [B, L, D, max_levels]
        level_attentions = torch.stack(level_attentions, dim=-1)  # [B, H, L, L, max_levels]
        
        # Apply level weights
        final_output = torch.sum(level_outputs * level_weights.view(B, 1, 1, -1), dim=-1)
        final_attention = torch.sum(level_attentions * level_weights.view(B, 1, 1, 1, -1), dim=-1)
        
        return final_output, final_attention


class MultiScaleWaveletAttention(BaseAttention):
    """
    Multi-scale wavelet attention for capturing patterns at different time scales.
    
    This component applies wavelet attention at multiple predefined scales
    and combines the results for comprehensive temporal modeling.
    """
    
    def __init__(self, d_model, n_heads, scales=[1, 2, 4, 8]):
        super(MultiScaleWaveletAttention, self).__init__()
        logger.info(f"Initializing MultiScaleWaveletAttention: scales={scales}")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.scales = scales
        
        # Wavelet attention for each scale
        self.scale_attentions = nn.ModuleList([
            WaveletAttention(d_model, n_heads, levels=3)
            for _ in scales
        ])
        
        # Scale fusion
        self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))
        self.scale_proj = nn.Linear(d_model * len(scales), d_model)
        
        self.output_dim_multiplier = 1
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Forward pass with multi-scale wavelet processing.
        
        Args:
            queries: [B, L, D] query tensor
            keys: [B, L, D] key tensor
            values: [B, L, D] value tensor
            attn_mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        B, L, D = queries.shape
        
        scale_outputs = []
        scale_attentions = []
        
        for scale, scale_attn in zip(self.scales, self.scale_attentions):
            if scale == 1:
                # Original scale
                q_scaled, k_scaled, v_scaled = queries, keys, values
            else:
                # Downsample for larger scales
                target_len = max(L // scale, 1)
                q_scaled = F.interpolate(queries.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(1, 2)
                k_scaled = F.interpolate(keys.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(1, 2)
                v_scaled = F.interpolate(values.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(1, 2)
            
            # Apply wavelet attention at this scale
            output, attention = scale_attn(q_scaled, k_scaled, v_scaled, attn_mask)
            
            # Upsample back to original length if needed
            if output.size(1) != L:
                output = F.interpolate(output.transpose(1, 2), size=L, mode='linear', align_corners=False).transpose(1, 2)
            
            scale_outputs.append(output)
            scale_attentions.append(attention)
        
        # Weighted fusion of multi-scale outputs
        weights = F.softmax(self.scale_weights, dim=0)
        
        # Concatenate and project
        concatenated = torch.cat(scale_outputs, dim=-1)  # [B, L, D*len(scales)]
        fused_output = self.scale_proj(concatenated)      # [B, L, D]
        
        # Average attention weights
        avg_attention = torch.stack(scale_attentions, dim=0).mean(dim=0)
        
        return fused_output, avg_attention
