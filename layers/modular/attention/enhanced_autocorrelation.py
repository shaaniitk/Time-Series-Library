"""
Enhanced AutoCorrelation Components for Modular Autoformer

This module implements enhanced autocorrelation mechanisms with adaptive
features, multi-scale analysis, and improved numerical stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base import BaseAttention
from utils.logger import logger


class EnhancedAutoCorrelation(BaseAttention):
    """
    Enhanced AutoCorrelation with adaptive window selection and multi-scale analysis.
    
    Key improvements over standard AutoCorrelation:
    1. Adaptive top-k selection based on correlation energy
    2. Multi-scale correlation analysis
    3. Learnable frequency filtering
    4. Numerical stability enhancements
    """

    def __init__(self, d_model, n_heads, factor=1, scale=None, attention_dropout=0.1, 
                 output_attention=False, adaptive_k=True, multi_scale=True, 
                 scales=[1, 2, 4], eps=1e-8):
        super(EnhancedAutoCorrelation, self).__init__()
        logger.info(f"Initializing EnhancedAutoCorrelation with enhanced features: d_model={d_model}")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
        # Adaptive parameters
        self.adaptive_k = adaptive_k
        self.multi_scale = multi_scale
        self.scales = scales
        self.eps = eps
        
        # Learnable components
        if self.multi_scale:
            self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))
            
        # Frequency filter for noise reduction
        self.frequency_filter = nn.Parameter(torch.ones(1))
        
        # Projection layers
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        
        self.output_dim_multiplier = 1

    def _select_adaptive_k(self, corr_energy, length):
        """
        Intelligently select the number of correlation peaks to use.
        
        Args:
            corr_energy: [B, L] correlation energy per time lag
            length: sequence length
            
        Returns:
            optimal k value
        """
        # Sort correlation energies in descending order
        sorted_energies, _ = torch.sort(corr_energy, dim=-1, descending=True)
        
        # Find the elbow point using second derivative
        if length > 10:  # Only for reasonable sequence lengths
            # Compute first and second derivatives
            first_diff = sorted_energies[:, :-1] - sorted_energies[:, 1:]
            second_diff = first_diff[:, :-1] - first_diff[:, 1:]
            
            # Find elbow as point of maximum curvature
            elbow_candidates = torch.argmax(second_diff, dim=-1) + 2  # +2 due to double differencing
            
            # Ensure reasonable bounds
            min_k = max(2, int(0.1 * math.log(length)))
            max_k = min(int(0.3 * length), int(self.factor * math.log(length) * 2))
            
            if min_k > max_k:
                max_k = min_k
            
            adaptive_k = torch.clamp(elbow_candidates, min_k, max_k)
            
            # Use median across batch for stability
            return int(torch.median(adaptive_k.float()).item())
        else:
            return max(2, int(self.factor * math.log(length)))

    def _multi_scale_correlation(self, queries, keys, length):
        """
        Compute correlations at multiple scales to capture different periodicities.
        
        Args:
            queries, keys: input tensors [B, H, L, E]
            length: sequence length
            
        Returns:
            aggregated multi-scale correlations
        """
        correlations = []
        B, H, L, E = queries.shape
        
        for i, scale in enumerate(self.scales):
            if scale == 1:
                # Original scale - ensure contiguous tensors
                q_input = queries.contiguous()
                k_input = keys.contiguous()
                q_fft = torch.fft.rfft(q_input, dim=-2)  # FFT along sequence dimension
                k_fft = torch.fft.rfft(k_input, dim=-2)
                
            else:
                # Downsample for multi-scale analysis - handle 4D properly
                target_len = max(length // scale, 1)
                q_downsampled = torch.zeros(B, H, target_len, E, device=queries.device)
                k_downsampled = torch.zeros(B, H, target_len, E, device=keys.device)
                
                # Process each head separately for F.interpolate compatibility
                for h in range(H):
                    for e in range(E):
                        q_downsampled[:, h, :, e] = F.interpolate(
                            queries[:, h, :, e].unsqueeze(1), size=target_len, 
                            mode='linear', align_corners=False
                        ).squeeze(1)
                        k_downsampled[:, h, :, e] = F.interpolate(
                            keys[:, h, :, e].unsqueeze(1), size=target_len, 
                            mode='linear', align_corners=False
                        ).squeeze(1)
                
                q_fft = torch.fft.rfft(q_downsampled, dim=-2)
                k_fft = torch.fft.rfft(k_downsampled, dim=-2)
            
            # Compute correlation in frequency domain
            correlation = q_fft * torch.conj(k_fft)
            
            # Apply learnable frequency filtering
            if hasattr(self, 'frequency_filter'):
                correlation = correlation * self.frequency_filter
            
            # Transform back to time domain
            correlation_time = torch.fft.irfft(correlation, n=length if scale == 1 else target_len, dim=-2)
            
            # Upsample back to original length if needed
            if scale != 1:
                correlation_upsampled = torch.zeros(B, H, length, E, device=correlation_time.device)
                for h in range(H):
                    for e in range(E):
                        correlation_upsampled[:, h, :, e] = F.interpolate(
                            correlation_time[:, h, :, e].unsqueeze(1), size=length,
                            mode='linear', align_corners=False
                        ).squeeze(1)
                correlation_time = correlation_upsampled
            
            correlations.append(correlation_time)
        
        # Weighted combination of multi-scale correlations
        if self.multi_scale and len(correlations) > 1:
            weights = F.softmax(self.scale_weights, dim=0)
            correlation_combined = sum(w * corr for w, corr in zip(weights, correlations))
        else:
            correlation_combined = correlations[0]
        
        return correlation_combined

    def _correlation_based_attention(self, queries, keys, values, length):
        """
        Compute attention weights based on autocorrelation.
        
        Args:
            queries: [B, H, L, E] query tensor
            keys: [B, H, L, E] key tensor  
            values: [B, H, L, E] value tensor
            length: sequence length
            
        Returns:
            Tuple of (output, attention_weights)
        """
        B, H, L, E = queries.shape
        
        # Compute multi-scale correlations
        correlation = self._multi_scale_correlation(queries, keys, length)
        
        # Compute correlation energy for adaptive k selection
        correlation_energy = torch.mean(torch.abs(correlation), dim=[1, 3])  # [B, L]
        
        # Select adaptive k
        if self.adaptive_k:
            k = self._select_adaptive_k(correlation_energy, length)
        else:
            k = int(self.factor * math.log(length))
        
        k = max(k, 1)  # Ensure k is at least 1
        
        # Find top-k correlations
        mean_correlation = torch.mean(correlation, dim=1)  # [B, L, E]
        _, top_k_indices = torch.topk(torch.mean(torch.abs(mean_correlation), dim=-1), k, dim=-1)
        
        # Initialize output
        output = torch.zeros_like(values)
        weights = torch.zeros(B, H, L, L, device=queries.device)
        
        # Apply correlation-based attention
        for b in range(B):
            for h in range(H):
                for i, lag in enumerate(top_k_indices[b]):
                    lag = lag.item()
                    
                    # Circular shift for autocorrelation
                    if lag > 0:
                        shifted_values = torch.cat([
                            values[b, h, -lag:, :], 
                            values[b, h, :-lag, :]
                        ], dim=0)
                    else:
                        shifted_values = values[b, h, :, :]
                    
                    # Compute attention weight based on correlation strength
                    corr_strength = torch.abs(mean_correlation[b, lag, :]).mean()
                    weight = F.softmax(torch.tensor([corr_strength]), dim=0)[0]
                    
                    output[b, h, :, :] += weight * shifted_values
                    
                    # Store attention weights for visualization
                    if lag > 0:
                        weights[b, h, :, :] += weight * torch.eye(L, device=queries.device).roll(-lag, dims=0)
                    else:
                        weights[b, h, :, :] += weight * torch.eye(L, device=queries.device)
        
        return output, weights

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Forward pass of enhanced autocorrelation attention.
        
        Args:
            queries: [B, L, D] or [B, H, L, E] query tensor
            keys: [B, L, D] or [B, H, L, E] key tensor
            values: [B, L, D] or [B, H, L, E] value tensor
            attn_mask: Optional attention mask
            tau: Temperature parameter (optional)
            delta: Delta parameter (optional)
            
        Returns:
            Tuple of (output, attention_weights)
        """
        
        # Handle different input shapes
        if queries.dim() == 4:
            # Input is [B, H, L, E] - reshape to [B, L, D]
            B, H, L, E = queries.shape
            D = H * E
            queries = queries.transpose(1, 2).contiguous().view(B, L, D)
            keys = keys.transpose(1, 2).contiguous().view(B, L, D)
            values = values.transpose(1, 2).contiguous().view(B, L, D)
        else:
            # Input is [B, L, D]
            B, L, D = queries.shape
            H = self.n_heads
            E = D // H
        
        # Apply projections
        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)
        
    # Reshape for multi-head processing
        queries = queries.view(B, L, H, E).transpose(1, 2)  # [B, H, Lq, E]
        # For cross-attention, keys/values may have different sequence length
        Lk = keys.size(1)
        Lv = values.size(1)
        keys = keys.view(B, Lk, H, E).transpose(1, 2)        # [B, H, Lk, E]
        values = values.view(B, Lv, H, E).transpose(1, 2)    # [B, H, Lv, E]
            
        # Apply correlation-based attention
    # Use query length for output shape
        output, attn_weights = self._correlation_based_attention(queries, keys, values, L)
        
        # Apply dropout
        output = self.dropout(output)
        
        # Reshape back and apply output projection
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.out_projection(output)
        
        if self.output_attention:
            return output, attn_weights
        else:
            return output, None


class AdaptiveAutoCorrelationLayer(BaseAttention):
    """
    Layer wrapper for Enhanced AutoCorrelation with additional processing.
    
    This provides a complete layer interface compatible with transformer architectures.
    """
    
    def __init__(self, d_model, n_heads, factor=1, attention_dropout=0.1, 
                 output_attention=False, adaptive_k=True, multi_scale=True):
        super(AdaptiveAutoCorrelationLayer, self).__init__()
        logger.info(f"Initializing AdaptiveAutoCorrelationLayer: d_model={d_model}")
        
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Core autocorrelation mechanism
        self.autocorrelation = EnhancedAutoCorrelation(
            d_model=d_model,
            n_heads=n_heads,
            factor=factor,
            attention_dropout=attention_dropout,
            output_attention=output_attention,
            adaptive_k=adaptive_k,
            multi_scale=multi_scale
        )
        
        # Additional processing layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(attention_dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(attention_dropout)
        )
        
        self.output_dim_multiplier = 1
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Forward pass with residual connections and normalization.
        
        Args:
            queries: [B, L, D] input tensor
            keys: [B, L, D] key tensor (often same as queries)
            values: [B, L, D] value tensor (often same as queries)
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Self-attention with residual connection
        attn_output, attn_weights = self.autocorrelation(queries, keys, values, attn_mask, tau, delta)
        queries = self.norm1(queries + attn_output)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(queries)
        output = self.norm2(queries + ffn_output)
        
        return output, attn_weights


class HierarchicalAutoCorrelation(BaseAttention):
    """
    Hierarchical autocorrelation for capturing patterns at multiple time horizons.
    
    This component applies autocorrelation at different temporal resolutions
    and combines the results hierarchically.
    """
    
    def __init__(self, d_model, n_heads, hierarchy_levels=[1, 4, 16], factor=1):
        super(HierarchicalAutoCorrelation, self).__init__()
        logger.info(f"Initializing HierarchicalAutoCorrelation: levels={hierarchy_levels}")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.hierarchy_levels = hierarchy_levels
        
        # Autocorrelation for each hierarchy level
        self.level_correlations = nn.ModuleList([
            EnhancedAutoCorrelation(
                d_model=d_model,
                n_heads=n_heads,
                factor=factor,
                scales=[level]
            )
            for level in hierarchy_levels
        ])
        
        # Hierarchical fusion
        self.level_fusion = nn.Linear(d_model * len(hierarchy_levels), d_model)
        self.level_weights = nn.Parameter(torch.ones(len(hierarchy_levels)) / len(hierarchy_levels))
        
        self.output_dim_multiplier = 1
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Forward pass with hierarchical processing.
        
        Args:
            queries: [B, L, D] query tensor
            keys: [B, L, D] key tensor
            values: [B, L, D] value tensor
            
        Returns:
            Tuple of (output, attention_weights)
        """
        B, L, D = queries.shape
        
        level_outputs = []
        level_attentions = []
        
        # Process at each hierarchy level
        for level, correlation_layer in zip(self.hierarchy_levels, self.level_correlations):
            if level == 1:
                # Original resolution
                level_out, level_attn = correlation_layer(queries, keys, values, attn_mask)
            else:
                # Downsample for higher levels
                target_len = max(L // level, 1)
                q_down = F.interpolate(queries.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(1, 2)
                k_down = F.interpolate(keys.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(1, 2)
                v_down = F.interpolate(values.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(1, 2)
                
                level_out, level_attn = correlation_layer(q_down, k_down, v_down)
                
                # Upsample back to original resolution
                level_out = F.interpolate(level_out.transpose(1, 2), size=L, mode='linear', align_corners=False).transpose(1, 2)
            
            level_outputs.append(level_out)
            level_attentions.append(level_attn)
        
        # Hierarchical fusion
        weights = F.softmax(self.level_weights, dim=0)
        
        # Weighted combination
        weighted_outputs = [w * out for w, out in zip(weights, level_outputs)]
        combined_output = sum(weighted_outputs)
        
        # Aggregate attention weights
        if level_attentions[0] is not None:
            avg_attention = torch.stack([attn for attn in level_attentions if attn is not None], dim=0).mean(dim=0)
        else:
            avg_attention = None
        
        return combined_output, avg_attention
