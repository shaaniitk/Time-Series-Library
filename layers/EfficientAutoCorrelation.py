import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class EfficientAutoCorrelation(nn.Module):
    """Enhanced AutoCorrelation with multi-scale analysis and adaptive features"""
    
    def __init__(self, mask_flag=True, factor=1, attention_dropout=0.1, 
                 max_seq_len=1024, use_checkpoint=True, adaptive_k=True, 
                 multi_scale=True, scales=[1, 2, 4], eps=1e-8):
        super().__init__()
        self.factor = factor
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)
        self.max_seq_len = max_seq_len
        self.use_checkpoint = use_checkpoint
        
        # Enhanced features
        self.adaptive_k = adaptive_k
        self.multi_scale = multi_scale
        self.scales = scales
        self.eps = eps
        
        # Learnable scale weights for multi-scale analysis
        if self.multi_scale:
            self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))
            
        # Adaptive k-selection parameters
        if self.adaptive_k:
            # Learnable k-predictor that adapts to correlation patterns
            self.k_predictor = nn.Sequential(
                nn.Linear(3, 32),  # 3 features: energy stats, variance, peak count
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()  # Output k ratio in [0, 1]
            )
        
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        
        # Use gradient checkpointing for large sequences
        if self.use_checkpoint and L > self.max_seq_len:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, queries, keys, values, attn_mask
            )
        else:
            return self._forward_impl(queries, keys, values, attn_mask)
    
    def _forward_impl(self, queries, keys, values, attn_mask):
        """Enhanced implementation with multi-scale analysis and adaptive k-selection"""
        B, L, H, E = queries.shape
        
        if self.multi_scale:
            # Multi-scale correlation analysis
            scale_outputs = []
            for i, scale in enumerate(self.scales):
                scale_output = self._multi_scale_correlation(queries, keys, values, scale, attn_mask)
                scale_outputs.append(scale_output * self.scale_weights[i])
            
            # Combine multi-scale outputs
            combined_output = torch.stack(scale_outputs, dim=0).sum(dim=0)
            return combined_output, None
        else:
            # Standard single-scale processing with chunking
            chunk_size = min(512, queries.size(1))
            outputs = []
            
            for i in range(0, queries.size(1), chunk_size):
                end_idx = min(i + chunk_size, queries.size(1))
                chunk_q = queries[:, i:end_idx]
                chunk_out = self._process_chunk(chunk_q, keys, values, attn_mask)
                outputs.append(chunk_out)
            
            return torch.cat(outputs, dim=1), None
    
    def _multi_scale_correlation(self, queries, keys, values, scale, attn_mask):
        """Perform correlation analysis at a specific scale"""
        B, L, H, E = queries.shape
        
        if scale > 1:
            # Downsample for multi-scale analysis
            scaled_L = L // scale
            if scaled_L < 1:
                scaled_L = 1
                
            # Reshape and pool to achieve downsampling
            queries_scaled = F.adaptive_avg_pool1d(
                queries.permute(0, 2, 3, 1).reshape(B * H * E, L).unsqueeze(1), 
                scaled_L
            ).squeeze(1).reshape(B, H, E, scaled_L).permute(0, 3, 1, 2)
            
            keys_scaled = F.adaptive_avg_pool1d(
                keys.permute(0, 2, 3, 1).reshape(B * H * E, L).unsqueeze(1), 
                scaled_L
            ).squeeze(1).reshape(B, H, E, scaled_L).permute(0, 3, 1, 2)
            
            values_scaled = F.adaptive_avg_pool1d(
                values.permute(0, 2, 3, 1).reshape(B * H * E, L).unsqueeze(1), 
                scaled_L
            ).squeeze(1).reshape(B, H, E, scaled_L).permute(0, 3, 1, 2)
        else:
            queries_scaled, keys_scaled, values_scaled = queries, keys, values
            scaled_L = L
        
        # Perform correlation at this scale
        corr_output = self._correlation_computation(queries_scaled, keys_scaled, values_scaled, attn_mask)
        
        # Upsample back to original resolution if needed
        if scale > 1 and corr_output.shape[1] != L:
            corr_output = F.interpolate(
                corr_output.permute(0, 2, 3, 1).reshape(B * H * E, scaled_L).unsqueeze(1),
                size=L, mode='linear', align_corners=False
            ).squeeze(1).reshape(B, H, E, L).permute(0, 3, 1, 2)
        
        return corr_output
    
    def _correlation_computation(self, queries, keys, values, attn_mask):
        """Core correlation computation with adaptive k-selection"""
        B, L, H, E = queries.shape
        
        # FFT-based correlation
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        
        # Correlation in frequency domain
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1).float()
        
        # Adaptive k-selection with learned predictor
        if self.adaptive_k:
            # Extract sophisticated correlation features for learned k-prediction
            B, H, E, L = corr.shape
            
            # Compute intelligent correlation statistics
            corr_energy_mean = torch.mean(torch.sum(corr ** 2, dim=(1, 2)))  # Scalar - avg energy
            corr_variance = torch.var(corr.view(B, -1), dim=1)  # [B] - variance per batch
            corr_peaks = self._detect_correlation_peaks_simple(corr)  # [B] - number of peaks
            
            # Create compact feature vector [B, 3] for k-prediction
            correlation_features = torch.stack([
                corr_energy_mean.expand(B),  # [B] - energy feature
                corr_variance,                # [B] - variance feature
                corr_peaks                    # [B] - peak count feature
            ], dim=1)  # [B, 3]
            
            # Predict optimal k ratio using learned network
            k_ratio = self.k_predictor(correlation_features)  # [B, 1]
            k_values = (k_ratio.squeeze(-1) * L * self.factor).int().clamp(min=1, max=L)
        else:
            k_values = torch.full((B,), int(L * self.factor), device=corr.device)
        
        # Apply adaptive k-selection to correlation
        return self._adaptive_time_delay_agg(values, corr, k_values)
    
    def _process_chunk(self, queries, keys, values, attn_mask):
        # Efficient FFT-based correlation computation
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        
        # Use half precision for intermediate computations if available
        if queries.device.type == 'cuda':
            q_fft = q_fft.half()
            k_fft = k_fft.half()
        
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1).float()
        
        # Simplified aggregation
        return self._efficient_time_delay_agg(values, corr)
    
    def _adaptive_time_delay_agg(self, values: torch.Tensor, corr: torch.Tensor, k_values: torch.Tensor) -> torch.Tensor:
        """
        Advanced time delay aggregation with adaptive k-selection and intelligent peak detection.
        
        Args:
            values: [B, L, H, E] input values
            corr: [B, H, E, L] correlation weights
            k_values: [B] adaptive k values per batch
            
        Returns:
            torch.Tensor: Aggregated values [B, L, H, E]
        """
        B, L, H, E = values.shape
        
        # Convert correlation to proper format for attention
        corr = corr.permute(0, 3, 1, 2)  # [B, L, H, E]
        
        outputs = []
        for b in range(B):
            k = k_values[b].item()
            
            # Find top-k correlations with peak detection
            corr_b = corr[b]  # [L, H, E]
            values_b = values[b]  # [L, H, E]
            
            # Apply intelligent peak detection
            corr_peaks = self._detect_correlation_peaks(corr_b, k)
            
            # Create attention weights from selected peaks
            attn_weights = torch.zeros_like(corr_b)
            attn_weights[corr_peaks] = corr_b[corr_peaks]
            
            # Normalize attention weights
            attn_weights = F.softmax(attn_weights + self.eps, dim=0)
            
            # Weighted aggregation
            output_b = torch.einsum('lhe,lhe->lhe', attn_weights, values_b)
            outputs.append(output_b)
        
        return torch.stack(outputs, dim=0)
    
    def _detect_correlation_peaks(self, corr: torch.Tensor, k: int) -> torch.Tensor:
        """
        Intelligent peak detection in correlation patterns.
        
        Args:
            corr: [L, H, E] correlation tensor
            k: number of peaks to select
            
        Returns:
            torch.Tensor: Boolean mask of selected peaks
        """
        L, H, E = corr.shape
        
        # Flatten for easier processing
        corr_flat = corr.view(L, -1)  # [L, H*E]
        
        # Find local maxima (simple peak detection)
        peaks = torch.zeros_like(corr_flat, dtype=torch.bool)
        
        for i in range(1, L - 1):
            local_max = (corr_flat[i] > corr_flat[i-1]) & (corr_flat[i] > corr_flat[i+1])
            peaks[i] = local_max
        
        # Select top-k peaks based on correlation magnitude
        if peaks.sum() > k:
            peak_values = corr_flat[peaks]
            _, top_indices = torch.topk(peak_values, k)
            
            # Create new peak mask with only top-k
            new_peaks = torch.zeros_like(peaks)
            peak_positions = torch.nonzero(peaks, as_tuple=False)
            selected_positions = peak_positions[top_indices]
            new_peaks[selected_positions[:, 0], selected_positions[:, 1]] = True
            peaks = new_peaks
        
        # Reshape back to original dimensions
        return peaks.view(L, H, E)
    
    def _detect_correlation_peaks_simple(self, corr: torch.Tensor) -> torch.Tensor:
        """
        Simple peak detection for k-prediction features.
        
        Args:
            corr: [B, H, E, L] correlation tensor
            
        Returns:
            torch.Tensor: [B] number of peaks per batch
        """
        B, H, E, L = corr.shape
        
        # Flatten spatial dimensions for peak detection
        corr_flat = corr.view(B, -1, L)  # [B, H*E, L]
        
        # Simple peak detection: count local maxima
        peak_counts = []
        for b in range(B):
            corr_b = corr_flat[b].mean(dim=0)  # [L] - average across spatial dims
            
            # Find local maxima
            peaks = 0
            for i in range(1, L - 1):
                if corr_b[i] > corr_b[i-1] and corr_b[i] > corr_b[i+1]:
                    peaks += 1
            
            peak_counts.append(peaks)
        
        return torch.tensor(peak_counts, device=corr.device, dtype=torch.float32)


class EfficientAutoCorrelationLayer(nn.Module):
    """
    Efficient AutoCorrelation layer with memory optimization.
    """
    
    def __init__(self, correlation: nn.Module, d_model: int, n_heads: int, 
                 d_keys: Optional[int] = None, d_values: Optional[int] = None):
        super(EfficientAutoCorrelationLayer, self).__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        
        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        
    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, _ = queries.shape
        S = keys.size(1)
        H = self.n_heads
        
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        
        out, attn = self.inner_correlation(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)
        
        return self.out_projection(out), attn