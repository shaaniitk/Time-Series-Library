import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.logger import logger


class AdaptiveAutoCorrelation(nn.Module):
    """
    Enhanced AutoCorrelation with adaptive window selection and multi-scale analysis.
    
    Key improvements:
    1. Adaptive top-k selection based on correlation energy
    2. Multi-scale correlation analysis
    3. Learnable frequency filtering
    4. Numerical stability enhancements
    """

    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, 
                 output_attention=False, adaptive_k=True, multi_scale=True, 
                 scales=[1, 2, 4], eps=1e-8):
        super(AdaptiveAutoCorrelation, self).__init__()
        logger.info("Initializing AdaptiveAutoCorrelation with enhanced features")
        
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
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
        
    def select_adaptive_k(self, corr_energy, length):
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

    def multi_scale_correlation(self, queries, keys, length):
        """
        Compute correlations at multiple scales to capture different periodicities.
        
        Args:
            queries, keys: input tensors
            length: sequence length
            
        Returns:
            aggregated multi-scale correlations
        """
        correlations = []
        
        for i, scale in enumerate(self.scales):
            if scale == 1:
                # Original scale
                q_fft = torch.fft.rfft(queries, dim=-1)
                k_fft = torch.fft.rfft(keys, dim=-1)
            else:
                # Downsampled scale - reshape for pooling
                if length >= scale * 2:  # Ensure enough points for downsampling
                    # Reshape to [B*H*E, L] for pooling
                    B, H, E, L = queries.shape
                    q_reshaped = queries.reshape(B * H * E, L).unsqueeze(1)  # [B*H*E, 1, L]
                    k_reshaped = keys.reshape(B * H * E, L).unsqueeze(1)     # [B*H*E, 1, L]
                    
                    q_down = F.avg_pool1d(q_reshaped, kernel_size=scale, stride=scale)
                    k_down = F.avg_pool1d(k_reshaped, kernel_size=scale, stride=scale)
                    
                    # Reshape back to original dimensions
                    downsampled_L = q_down.size(-1)
                    q_down = q_down.squeeze(1).reshape(B, H, E, downsampled_L)
                    k_down = k_down.squeeze(1).reshape(B, H, E, downsampled_L)
                    
                    q_fft = torch.fft.rfft(q_down, dim=-1)
                    k_fft = torch.fft.rfft(k_down, dim=-1)
                else:
                    # Skip scales that are too large
                    continue
            
            # Apply frequency filtering for noise reduction
            freq_filter = torch.sigmoid(self.frequency_filter)
            q_fft = q_fft * freq_filter
            k_fft = k_fft * freq_filter
            
            # Compute correlation with numerical stability
            k_fft_conj = torch.conj(k_fft)
            k_magnitude = torch.abs(k_fft) + self.eps
            k_fft_normalized = k_fft_conj / k_magnitude
            
            res = q_fft * k_fft_normalized
            corr = torch.fft.irfft(res, dim=-1)
            
            # ALWAYS ensure correlation matches target length exactly
            B, H, E, curr_L = corr.shape
            if curr_L != length:
                # Reshape to [B*H*E, curr_L] for interpolation
                corr_reshaped = corr.reshape(B * H * E, curr_L).unsqueeze(1)  # [B*H*E, 1, curr_L]
                corr_upsampled = F.interpolate(corr_reshaped, size=length, mode='linear', align_corners=False)
                corr = corr_upsampled.squeeze(1).reshape(B, H, E, length)  # Back to [B, H, E, length]
            
            # Double check the length is correct
            assert corr.size(-1) == length, f"Correlation length {corr.size(-1)} != target length {length}"
            correlations.append(corr)
        
        if len(correlations) > 1:
            # Weighted combination of multi-scale correlations
            scale_weights = F.softmax(self.scale_weights[:len(correlations)], dim=0)
            multi_scale_corr = sum(w * corr for w, corr in zip(scale_weights, correlations))
        else:
            multi_scale_corr = correlations[0]
            
        return multi_scale_corr

    def enhanced_time_delay_agg(self, values, corr):
        """
        Enhanced time delay aggregation with better numerical stability and efficiency.
        
        Args:
            values: [B, H, C, L] value tensor
            corr: [B, H, C, L] correlation tensor
            
        Returns:
            aggregated values based on time delays
        """
        batch, head, channel, length = values.shape
        
        # Compute correlation energy for adaptive k selection
        corr_energy = torch.sum(corr ** 2, dim=(1, 2))  # [B, L]
        
        if self.adaptive_k:
            top_k = self.select_adaptive_k(corr_energy, length)
        else:
            top_k = int(self.factor * math.log(length))
            
        logger.debug(f"Using top_k={top_k} for sequence length {length}")
        
        # Select top correlations more efficiently
        mean_corr = torch.mean(torch.mean(corr, dim=1), dim=1)  # [B, L]
        weights, delays = torch.topk(mean_corr, top_k, dim=-1)  # [B, top_k]
        
        # Normalize weights with temperature scaling for better control
        temperature = 1.0  # Could be learnable
        normalized_weights = F.softmax(weights / temperature, dim=-1)
        
        # Efficient vectorized aggregation
        delays_agg = torch.zeros_like(values)
        values_padded = torch.cat([values, values], dim=-1)  # Circular padding
        
        for k in range(top_k):
            # Create delay indices efficiently
            delay = delays[:, k].unsqueeze(1).unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 1]
            
            # Advanced indexing for delayed values
            time_indices = torch.arange(length, device=values.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            delayed_indices = (time_indices + delay) % (2 * length)
            delayed_indices = delayed_indices.expand(batch, head, channel, length)
            
            # Gather delayed values
            delayed_values = torch.gather(values_padded, dim=-1, index=delayed_indices)
            
            # Apply weights
            weight = normalized_weights[:, k].view(batch, 1, 1, 1)
            delays_agg += delayed_values * weight
            
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        """
        Enhanced forward pass with adaptive correlation and multi-scale analysis.
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        
        # Handle sequence length mismatch
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # Input normalization for numerical stability
        queries = F.layer_norm(queries, queries.shape[-1:])
        keys = F.layer_norm(keys, keys.shape[-1:])
        
        # Reshape for correlation computation
        queries_reshaped = queries.permute(0, 2, 3, 1).contiguous()  # [B, H, E, L]
        keys_reshaped = keys.permute(0, 2, 3, 1).contiguous()        # [B, H, E, L]
        
        # Multi-scale correlation computation
        if self.multi_scale and len(self.scales) > 1:
            corr = self.multi_scale_correlation(queries_reshaped, keys_reshaped, L)
        else:
            # Standard single-scale correlation
            q_fft = torch.fft.rfft(queries_reshaped, dim=-1)
            k_fft = torch.fft.rfft(keys_reshaped, dim=-1)
            
            # Numerical stability improvements
            k_fft_conj = torch.conj(k_fft)
            k_magnitude = torch.abs(k_fft) + self.eps
            k_fft_normalized = k_fft_conj / k_magnitude
            
            res = q_fft * k_fft_normalized
            corr = torch.fft.irfft(res, dim=-1)
        
        # Clamp correlation values to prevent extreme values
        corr = torch.clamp(corr, -10.0, 10.0)
        
        # Enhanced time delay aggregation
        values_reshaped = values.permute(0, 2, 3, 1).contiguous()  # [B, H, D, L]
        V = self.enhanced_time_delay_agg(values_reshaped, corr)
        V = V.permute(0, 3, 1, 2)  # Back to [B, L, H, D]

        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)


class AdaptiveAutoCorrelationLayer(nn.Module):
    """
    Enhanced AutoCorrelation layer with adaptive features.
    """
    
    def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None):
        super(AdaptiveAutoCorrelationLayer, self).__init__()
        logger.info("Initializing AdaptiveAutoCorrelationLayer")

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        
        # Layer scaling for better gradient flow
        self.layer_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project inputs
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # Apply enhanced correlation
        out, attn = self.inner_correlation(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)

        # Apply output projection with layer scaling
        output = self.out_projection(out)
        output = self.layer_scale * output
        
        return output, attn


# Example usage and integration
class EnhancedAutoformerModel(nn.Module):
    """
    Example of how to integrate the enhanced AutoCorrelation into Autoformer.
    """
    
    def __init__(self, configs):
        super().__init__()
        logger.info("Initializing EnhancedAutoformerModel")
        
        # Enhanced AutoCorrelation with adaptive features
        self.adaptive_autocorr = AdaptiveAutoCorrelation(
            factor=configs.factor,
            adaptive_k=True,
            multi_scale=True,
            scales=[1, 2, 4],  # Multi-scale analysis
            attention_dropout=configs.dropout
        )
        
        # Enhanced AutoCorrelation layer
        self.autocorr_layer = AdaptiveAutoCorrelationLayer(
            self.adaptive_autocorr,
            configs.d_model,
            configs.n_heads
        )
        
        # Rest of the model architecture...
        
    def forward(self, x, x_mark):
        # Use enhanced autocorrelation
        enhanced_out, attention_weights = self.autocorr_layer(x, x, x, None)
        
        return enhanced_out, attention_weights


if __name__ == "__main__":
    # Example configuration for testing
    class TestConfig:
        factor = 1
        d_model = 64
        n_heads = 8
        dropout = 0.1
    
    config = TestConfig()
    
    # Create enhanced model
    enhanced_autocorr = AdaptiveAutoCorrelation(
        factor=config.factor,
        adaptive_k=True,
        multi_scale=True,
        scales=[1, 2, 4]
    )
    
    # Test with sample data
    B, L, H, E = 4, 96, 8, 8
    queries = torch.randn(B, L, H, E)
    keys = torch.randn(B, L, H, E)
    values = torch.randn(B, L, H, E)
    
    output, attention = enhanced_autocorr(queries, keys, values, None)
    
    print(f"Input shape: {queries.shape}")
    print(f"Output shape: {output.shape}")
    print("Enhanced AutoCorrelation test completed successfully!")
