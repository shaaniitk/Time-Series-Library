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
        super().__init__()
        
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

class TwoStageAttention(BaseAttention):
    """
    Modular implementation of Two-Stage Attention (TSA).
    This combines cross-time and cross-dimension attention for segment merging and multi-variate time series.
    Algorithm adapted from TwoStageAttentionLayer in SelfAttention_Family.py.
    """
    def __init__(self, d_model: int, n_heads: int, seg_num: int, factor: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.dim_sender = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.dim_receiver = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x: torch.Tensor, attn_mask=None, tau=None, delta=None):
        # x: [batch, ts_d, seg_num, d_model]
        batch = x.shape[0]
        ts_d = x.shape[1]
        seg_num = x.shape[2]
        d_model = x.shape[3]
        # Cross Time Stage
        time_in = x.reshape(-1, seg_num, d_model)
        time_enc, _ = self.time_attention(time_in, time_in, time_in, attn_mask=attn_mask)
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)
        # Cross Dimension Stage
        dim_send = dim_in.reshape(batch * seg_num, ts_d, d_model)
        batch_router = self.router.repeat(batch, 1, 1)
        dim_buffer, _ = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=attn_mask)
        dim_receive, _ = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=attn_mask)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)
        final_out = dim_enc.reshape(batch, ts_d, seg_num, d_model)
        return final_out, None


class ExponentialSmoothingAttention(BaseAttention):
    """
    Modular implementation of Exponential Smoothing Attention.
    Implements ETS-style attention using learnable smoothing weights.
    Algorithm adapted from ExponentialSmoothing in ETSformer_EncDec.py.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self._smoothing_weight = nn.Parameter(torch.randn(n_heads, 1))
        self.v0 = nn.Parameter(torch.randn(1, 1, n_heads, d_model // n_heads))
        self.dropout = nn.Dropout(dropout)

    @property
    def weight(self):
        return torch.sigmoid(self._smoothing_weight)

    def get_exponential_weight(self, T):
        powers = torch.arange(T, dtype=torch.float, device=self.weight.device)
        weight = (1 - self.weight) * (self.weight ** torch.flip(powers, dims=(0,)))
        init_weight = self.weight ** (powers + 1)
        return init_weight.unsqueeze(0).unsqueeze(-1), weight.unsqueeze(0).unsqueeze(-1)

    def forward(self, values: torch.Tensor, attn_mask=None, tau=None, delta=None):
        # values: [B, L, D]
        B, L, D = values.shape
        H = self.n_heads
        d_head = D // H
        values = values.view(B, L, H, d_head)
        init_weight, weight = self.get_exponential_weight(L)
        output = F.conv1d(self.dropout(values.permute(0, 2, 3, 1)), weight.squeeze(-1), groups=H)
        output = init_weight * self.v0 + output
        output = output.permute(0, 3, 1, 2).reshape(B, L, D)
        return output, None


class MultiWaveletCrossAttention(BaseAttention):
    """
    Modular implementation of Multi-Wavelet Cross Attention.
    Applies cross-attention using multi-resolution wavelet transforms.
    Algorithm adapted from MultiWaveletCorrelation and wavelet attention patterns.
    """
    def __init__(self, d_model: int, n_heads: int, levels: int = 3, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.levels = levels
        self.dropout = nn.Dropout(dropout)
        self.wavelet_decomp = WaveletDecomposition(d_model, levels)
        self.cross_attentions = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            for _ in range(levels + 1)
        ])
        self.level_weights = nn.Parameter(torch.ones(levels + 1) / (levels + 1))
        self.fusion_proj = nn.Linear(d_model * (levels + 1), d_model)

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, D = queries.shape
        _, q_components = self.wavelet_decomp(queries)
        _, k_components = self.wavelet_decomp(keys)
        _, v_components = self.wavelet_decomp(values)
        level_outputs = []
        level_attentions = []
        for i, (q_comp, k_comp, v_comp, attention_layer) in enumerate(
            zip(q_components, k_components, v_components, self.cross_attentions)
        ):
            if q_comp.size(1) != L:
                q_comp = F.interpolate(q_comp.transpose(1, 2), size=L, mode='linear', align_corners=False).transpose(1, 2)
                k_comp = F.interpolate(k_comp.transpose(1, 2), size=L, mode='linear', align_corners=False).transpose(1, 2)
                v_comp = F.interpolate(v_comp.transpose(1, 2), size=L, mode='linear', align_corners=False).transpose(1, 2)
            level_out, level_attn = attention_layer(q_comp, k_comp, v_comp, attn_mask=attn_mask)
            level_outputs.append(level_out)
            level_attentions.append(level_attn)
        weights = F.softmax(self.level_weights, dim=0)
        concatenated = torch.cat(level_outputs, dim=-1)
        fused_output = self.fusion_proj(concatenated)
        avg_attention = torch.stack(level_attentions, dim=0).mean(dim=0)
        return fused_output, avg_attention
    
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
