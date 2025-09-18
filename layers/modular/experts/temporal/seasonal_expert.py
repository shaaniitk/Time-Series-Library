"""
Seasonal Pattern Expert

Specialized expert for detecting and modeling seasonal patterns in time series.
Handles multiple seasonality types: daily, weekly, monthly, yearly cycles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List
import math

from ..base_expert import TemporalExpert, ExpertOutput


class SeasonalDecomposition(nn.Module):
    """Learnable seasonal decomposition module."""
    
    def __init__(self, d_model: int, seasonal_periods: List[int], decomp_kernel_size: int = 25):
        super().__init__()
        self.d_model = d_model
        self.seasonal_periods = seasonal_periods
        self.decomp_kernel_size = decomp_kernel_size
        
        # Moving average for trend extraction
        self.moving_avg = nn.AvgPool1d(
            kernel_size=decomp_kernel_size, 
            stride=1, 
            padding=decomp_kernel_size // 2
        )
        
        # Seasonal extractors for different periods
        self.seasonal_extractors = nn.ModuleList()
        for period in seasonal_periods:
            extractor = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=period, padding=period//2, groups=d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Conv1d(d_model, d_model, kernel_size=1)
            )
            self.seasonal_extractors.append(extractor)
        
        # Seasonal fusion
        self.seasonal_fusion = nn.Sequential(
            nn.Linear(d_model * len(seasonal_periods), d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decompose input into trend and seasonal components.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Dictionary with trend, seasonal, and residual components
        """
        batch_size, seq_len, d_model = x.shape
        
        # Extract trend using moving average
        x_transposed = x.transpose(1, 2)  # [batch_size, d_model, seq_len]
        trend = self.moving_avg(x_transposed).transpose(1, 2)  # [batch_size, seq_len, d_model]
        
        # Extract seasonal components for different periods
        seasonal_components = []
        for extractor in self.seasonal_extractors:
            seasonal_comp = extractor(x_transposed).transpose(1, 2)
            seasonal_components.append(seasonal_comp)
        
        # Fuse seasonal components
        if len(seasonal_components) > 1:
            seasonal_concat = torch.cat(seasonal_components, dim=-1)
            seasonal = self.seasonal_fusion(seasonal_concat)
        else:
            seasonal = seasonal_components[0]
        
        # Compute residual
        residual = x - trend - seasonal
        
        return {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual,
            'seasonal_components': seasonal_components
        }


class FourierSeasonalEncoder(nn.Module):
    """Fourier-based seasonal pattern encoder."""
    
    def __init__(self, d_model: int, max_seq_len: int, num_harmonics: int = 10):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_harmonics = num_harmonics
        
        # Learnable frequency weights
        self.frequency_weights = nn.Parameter(torch.randn(num_harmonics, d_model) * 0.1)
        self.phase_shifts = nn.Parameter(torch.randn(num_harmonics, d_model) * 0.1)
        
        # Amplitude modulation
        self.amplitude_modulator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_harmonics),
            nn.Softplus()  # Ensure positive amplitudes
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode seasonal patterns using Fourier basis.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Fourier-encoded seasonal features
        """
        batch_size, seq_len, d_model = x.shape
        device = x.device
        
        # Create time indices
        time_indices = torch.arange(seq_len, device=device, dtype=torch.float32)
        time_indices = time_indices.unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
        
        # Compute amplitudes based on input
        amplitudes = self.amplitude_modulator(x.mean(dim=1))  # [batch_size, num_harmonics]
        amplitudes = amplitudes.unsqueeze(1)  # [batch_size, 1, num_harmonics]
        
        # Generate Fourier components
        fourier_features = torch.zeros(batch_size, seq_len, d_model, device=device)
        
        for h in range(self.num_harmonics):
            # Fundamental frequency and harmonics
            frequency = (h + 1) * 2 * math.pi / seq_len
            
            # Sine and cosine components
            sin_component = torch.sin(frequency * time_indices + self.phase_shifts[h])
            cos_component = torch.cos(frequency * time_indices + self.phase_shifts[h])
            
            # Weight by learnable parameters and amplitudes
            weighted_sin = sin_component * self.frequency_weights[h] * amplitudes[:, :, h:h+1]
            weighted_cos = cos_component * self.frequency_weights[h] * amplitudes[:, :, h:h+1]
            
            fourier_features += weighted_sin + weighted_cos
        
        return fourier_features


class AdaptiveSeasonalAttention(nn.Module):
    """Attention mechanism that adapts to seasonal patterns."""
    
    def __init__(self, d_model: int, num_heads: int = 8, seasonal_periods: List[int] = [7, 30, 365]):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.seasonal_periods = seasonal_periods
        self.head_dim = d_model // num_heads
        
        # Multi-head attention components
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Seasonal position encodings
        self.seasonal_pos_encodings = nn.ModuleList()
        for period in seasonal_periods:
            pos_encoding = nn.Parameter(torch.randn(period, d_model) * 0.1)
            self.seasonal_pos_encodings.append(pos_encoding)
        
        # Period selection mechanism
        self.period_selector = nn.Sequential(
            nn.Linear(d_model, len(seasonal_periods)),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply seasonal-aware attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Attention output with seasonal awareness
        """
        batch_size, seq_len, d_model = x.shape
        
        # Select dominant seasonal period
        period_weights = self.period_selector(x.mean(dim=1))  # [batch_size, num_periods]
        
        # Apply seasonal position encodings
        seasonal_x = x.clone()
        for i, (period, pos_encoding) in enumerate(zip(self.seasonal_periods, self.seasonal_pos_encodings)):
            # Create seasonal position indices
            pos_indices = torch.arange(seq_len, device=x.device) % period
            seasonal_pos = pos_encoding[pos_indices]  # [seq_len, d_model]
            
            # Weight by period selection
            weight = period_weights[:, i:i+1].unsqueeze(1)  # [batch_size, 1, 1]
            seasonal_x += weight * seasonal_pos.unsqueeze(0)
        
        # Multi-head attention
        q = self.q_proj(seasonal_x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(seasonal_x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(seasonal_x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output


class SeasonalPatternExpert(TemporalExpert):
    """
    Expert specialized in seasonal pattern detection and modeling.
    
    Combines multiple techniques:
    - Seasonal decomposition
    - Fourier-based encoding
    - Seasonal-aware attention
    """
    
    def __init__(self, config, seasonal_periods: Optional[List[int]] = None):
        super().__init__(config, 'seasonal_expert')
        
        # Default seasonal periods (daily, weekly, monthly patterns)
        if seasonal_periods is None:
            seasonal_periods = [7, 30, 365]  # Weekly, monthly, yearly
        self.seasonal_periods = seasonal_periods
        
        # Seasonal decomposition
        self.seasonal_decomp = SeasonalDecomposition(
            self.d_model, 
            seasonal_periods,
            decomp_kernel_size=getattr(config, 'seasonal_decomp_kernel', 25)
        )
        
        # Fourier seasonal encoder
        self.fourier_encoder = FourierSeasonalEncoder(
            self.d_model,
            getattr(config, 'seq_len', 96),
            num_harmonics=getattr(config, 'seasonal_harmonics', 10)
        )
        
        # Seasonal attention
        self.seasonal_attention = AdaptiveSeasonalAttention(
            self.d_model,
            getattr(config, 'seasonal_heads', 8),
            seasonal_periods
        )
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.d_model * 3, self.d_model),  # decomp + fourier + attention
            nn.ReLU(),
            nn.LayerNorm(self.d_model),
            nn.Dropout(self.dropout)
        )
        
        # Seasonal strength estimator
        self.seasonal_strength = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 4),
            nn.ReLU(),
            nn.Linear(self.d_model // 4, len(seasonal_periods)),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Linear(self.d_model, self.expert_dim)
        
    def forward(self, x: torch.Tensor, **kwargs) -> ExpertOutput:
        """
        Process input through seasonal pattern expert.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            ExpertOutput with seasonal pattern analysis
        """
        batch_size, seq_len, d_model = x.shape
        
        # 1. Seasonal decomposition
        decomp_results = self.seasonal_decomp(x)
        seasonal_component = decomp_results['seasonal']
        
        # 2. Fourier-based seasonal encoding
        fourier_features = self.fourier_encoder(x)
        
        # 3. Seasonal-aware attention
        attention_features = self.seasonal_attention(x)
        
        # 4. Feature fusion
        combined_features = torch.cat([
            seasonal_component,
            fourier_features,
            attention_features
        ], dim=-1)
        
        fused_features = self.feature_fusion(combined_features)
        
        # 5. Apply temporal convolution and normalization
        temporal_features = self.temporal_conv(fused_features.transpose(1, 2)).transpose(1, 2)
        temporal_features = self.temporal_norm(temporal_features)
        
        # 6. Output projection
        output = self.output_proj(temporal_features)
        
        # 7. Compute confidence based on seasonal strength
        seasonal_strengths = self.seasonal_strength(fused_features.mean(dim=1))
        confidence = seasonal_strengths.max(dim=-1)[0].unsqueeze(1).unsqueeze(1)
        confidence = confidence.expand(-1, seq_len, -1)
        
        # 8. Compile metadata
        metadata = {
            'seasonal_periods': self.seasonal_periods,
            'seasonal_strengths': seasonal_strengths,
            'decomposition': {
                'trend_magnitude': decomp_results['trend'].abs().mean().item(),
                'seasonal_magnitude': decomp_results['seasonal'].abs().mean().item(),
                'residual_magnitude': decomp_results['residual'].abs().mean().item()
            },
            'fourier_energy': fourier_features.pow(2).mean().item(),
            'dominant_period': self.seasonal_periods[seasonal_strengths.mean(dim=0).argmax().item()]
        }
        
        return ExpertOutput(
            output=output,
            confidence=confidence,
            metadata=metadata,
            attention_weights=None,  # Could add attention weights if needed
            uncertainty=None  # Could add uncertainty estimation
        )
    
    def detect_seasonality(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Detect seasonal patterns in the input.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Dictionary with seasonality detection results
        """
        with torch.no_grad():
            # Get seasonal decomposition
            decomp_results = self.seasonal_decomp(x)
            
            # Compute seasonal strength for each period
            seasonal_strengths = self.seasonal_strength(x.mean(dim=1))
            
            # Analyze frequency content
            x_freq = torch.fft.fft(x.mean(dim=-1))  # [batch_size, seq_len]
            power_spectrum = torch.abs(x_freq) ** 2
            
            # Find dominant frequencies
            dominant_freqs = torch.topk(power_spectrum, k=min(5, x.size(1)//2), dim=-1)[1]
            
            return {
                'seasonal_strengths': seasonal_strengths,
                'dominant_frequencies': dominant_freqs,
                'seasonal_magnitude': decomp_results['seasonal'].abs().mean().item(),
                'trend_magnitude': decomp_results['trend'].abs().mean().item(),
                'seasonality_score': seasonal_strengths.max(dim=-1)[0].mean().item()
            }
    
    def get_seasonal_forecast(self, x: torch.Tensor, forecast_steps: int) -> torch.Tensor:
        """
        Generate seasonal forecast for future steps.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            forecast_steps: Number of steps to forecast
            
        Returns:
            Seasonal forecast [batch_size, forecast_steps, d_model]
        """
        with torch.no_grad():
            batch_size, seq_len, d_model = x.shape
            
            # Extract seasonal component
            decomp_results = self.seasonal_decomp(x)
            seasonal_component = decomp_results['seasonal']
            
            # Project seasonal pattern forward
            forecasts = []
            for step in range(forecast_steps):
                # Use periodicity to project forward
                forecast_step_features = []
                
                for period in self.seasonal_periods:
                    # Find corresponding historical point
                    hist_idx = (seq_len + step) % period
                    if hist_idx < seq_len:
                        period_forecast = seasonal_component[:, hist_idx:hist_idx+1, :]
                    else:
                        # Use average if no direct historical match
                        period_forecast = seasonal_component.mean(dim=1, keepdim=True)
                    
                    forecast_step_features.append(period_forecast)
                
                # Average across periods
                step_forecast = torch.stack(forecast_step_features).mean(dim=0)
                forecasts.append(step_forecast)
            
            return torch.cat(forecasts, dim=1)