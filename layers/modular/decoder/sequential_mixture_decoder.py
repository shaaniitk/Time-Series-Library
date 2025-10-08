"""
Sequential Mixture Density Decoder

This module implements a sequence-to-sequence mixture density decoder that preserves
temporal structure throughout the decoding process, addressing the critical information
bottleneck in the original implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional


class SequentialMixtureDensityDecoder(nn.Module):
    """
    Sequence-to-Sequence Mixture Density Network decoder.
    
    This decoder preserves temporal structure by using a transformer decoder
    architecture instead of collapsing the sequence to a single vector.
    
    Key improvements over the original:
    1. Preserves temporal structure throughout decoding
    2. Enables time-varying mixture parameters
    3. Better uncertainty quantification over time
    4. Supports cross-attention with encoder features
    """
    
    def __init__(self, d_model: int, pred_len: int, num_components: int = 3, 
                 num_targets: int = 4, num_decoder_layers: int = 3, 
                 num_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            pred_len: Prediction horizon length
            num_components: Number of Gaussian mixture components
            num_targets: Number of target features (e.g., 4 for OHLC)
            num_decoder_layers: Number of transformer decoder layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.pred_len = pred_len
        self.num_components = num_components
        self.num_targets = num_targets
        self.num_heads = num_heads
        
        # Positional encoding for decoder sequence
        self.pos_encoding = PositionalEncoding(d_model, max_len=pred_len * 2)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=num_decoder_layers
        )
        
        # Mixture parameter heads - separate for each timestep
        self.mixture_heads = nn.ModuleDict({
            'means': nn.Linear(d_model, num_targets * num_components),
            'log_stds': nn.Linear(d_model, num_targets * num_components),
            'log_weights': nn.Linear(d_model, num_components)
        })
        
        # Layer normalization before mixture heads
        self.pre_mixture_norm = nn.LayerNorm(d_model)
        
        # Initialize mixture heads with appropriate scales
        self._initialize_mixture_heads()
    
    def _initialize_mixture_heads(self):
        """Initialize mixture parameter heads with appropriate scales."""
        # Initialize means with small random values
        nn.init.normal_(self.mixture_heads['means'].weight, std=0.01)
        nn.init.zeros_(self.mixture_heads['means'].bias)
        
        # Initialize log_stds to reasonable values (std around 0.1-1.0)
        nn.init.normal_(self.mixture_heads['log_stds'].weight, std=0.01)
        nn.init.constant_(self.mixture_heads['log_stds'].bias, -2.0)  # log(0.135) ≈ -2
        
        # Initialize log_weights to be roughly uniform
        nn.init.normal_(self.mixture_heads['log_weights'].weight, std=0.01)
        nn.init.zeros_(self.mixture_heads['log_weights'].bias)
    
    def forward(self, encoder_output: torch.Tensor, 
                decoder_input: torch.Tensor,
                encoder_mask: Optional[torch.Tensor] = None,
                decoder_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with sequence-to-sequence decoding.
        
        Args:
            encoder_output: [batch, enc_seq_len, d_model] Encoder output
            decoder_input: [batch, dec_seq_len, d_model] Decoder input (embedded)
            encoder_mask: Optional encoder padding mask
            decoder_mask: Optional decoder causal mask
            
        Returns:
            Tuple of:
            - means: [batch, dec_seq_len, num_targets, num_components]
            - log_stds: [batch, dec_seq_len, num_targets, num_components]  
            - log_weights: [batch, dec_seq_len, num_components]
        """
        batch_size, dec_seq_len, _ = decoder_input.shape
        
        # Add positional encoding to decoder input
        decoder_input_pos = self.pos_encoding(decoder_input)
        
        # Create causal mask for decoder if not provided
        if decoder_mask is None:
            decoder_mask = self._generate_causal_mask(dec_seq_len, decoder_input.device)
        
        # Transformer decoder with cross-attention to encoder
        decoder_output = self.transformer_decoder(
            tgt=decoder_input_pos,
            memory=encoder_output,
            tgt_mask=decoder_mask,
            memory_mask=encoder_mask
        )  # [batch, dec_seq_len, d_model]
        
        # Apply layer normalization before mixture heads
        normalized_output = self.pre_mixture_norm(decoder_output)
        
        # Generate mixture parameters for each timestep
        means_flat = self.mixture_heads['means'](normalized_output)
        log_stds_flat = self.mixture_heads['log_stds'](normalized_output)
        log_weights_flat = self.mixture_heads['log_weights'](normalized_output)
        
        # Reshape to proper mixture dimensions
        means = means_flat.view(batch_size, dec_seq_len, self.num_targets, self.num_components)
        log_stds = log_stds_flat.view(batch_size, dec_seq_len, self.num_targets, self.num_components)
        log_weights = log_weights_flat.view(batch_size, dec_seq_len, self.num_components)
        
        # Clamp log_stds to prevent numerical issues
        log_stds = torch.clamp(log_stds, min=-10, max=2)  # std between ~0.00005 and ~7.4
        
        return means, log_stds, log_weights
    
    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for decoder self-attention."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()
    
    def sample(self, mixture_params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
               num_samples: int = 1) -> torch.Tensor:
        """
        Sample from the mixture distribution.
        
        Args:
            mixture_params: Tuple (means, log_stds, log_weights)
            num_samples: Number of samples to draw
            
        Returns:
            samples: [num_samples, batch, seq_len, num_targets]
        """
        means, log_stds, log_weights = mixture_params
        batch_size, seq_len, num_targets, num_components = means.shape
        
        # Convert to usable parameters
        stds = torch.exp(log_stds).clamp_min(1e-6)
        weights = F.softmax(log_weights, dim=-1)
        
        samples = []
        for _ in range(num_samples):
            # Sample mixture component for each (batch, time, target) position
            component_probs = weights.unsqueeze(2).expand(-1, -1, num_targets, -1)  # [B, T, num_targets, K]
            component_probs_flat = component_probs.reshape(-1, num_components)
            
            # Sample component indices
            component_indices = torch.multinomial(component_probs_flat, 1).squeeze(-1)  # [B*T*num_targets]
            component_indices = component_indices.view(batch_size, seq_len, num_targets)
            
            # Gather selected means and stds
            batch_indices = torch.arange(batch_size, device=means.device).view(-1, 1, 1)
            time_indices = torch.arange(seq_len, device=means.device).view(1, -1, 1)
            target_indices = torch.arange(num_targets, device=means.device).view(1, 1, -1)
            
            selected_means = means[batch_indices, time_indices, target_indices, component_indices]
            selected_stds = stds[batch_indices, time_indices, target_indices, component_indices]
            
            # Sample from selected Gaussians
            noise = torch.randn_like(selected_means)
            sample = selected_means + selected_stds * noise
            samples.append(sample)
        
        return torch.stack(samples, dim=0)  # [num_samples, batch, seq_len, num_targets]
    
    def get_point_prediction(self, mixture_params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Get point prediction as weighted mean of mixture components.
        
        Args:
            mixture_params: Tuple (means, log_stds, log_weights)
            
        Returns:
            predictions: [batch, seq_len, num_targets]
        """
        means, log_stds, log_weights = mixture_params
        
        # Convert weights to probabilities
        weights = F.softmax(log_weights, dim=-1)  # [batch, seq_len, num_components]
        
        # Expand weights to match means dimensions
        weights_expanded = weights.unsqueeze(2).expand_as(means)  # [batch, seq_len, num_targets, num_components]
        
        # Weighted average of component means
        predictions = (means * weights_expanded).sum(dim=-1)  # [batch, seq_len, num_targets]
        
        return predictions
    
    def get_uncertainty_estimates(self, mixture_params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> dict:
        """
        Get various uncertainty estimates from the mixture distribution.
        
        Args:
            mixture_params: Tuple (means, log_stds, log_weights)
            
        Returns:
            Dictionary with uncertainty estimates
        """
        means, log_stds, log_weights = mixture_params
        
        # Convert parameters
        stds = torch.exp(log_stds).clamp_min(1e-6)
        weights = F.softmax(log_weights, dim=-1)
        weights_expanded = weights.unsqueeze(2).expand_as(means)
        
        # Mixture mean and variance
        mixture_mean = (means * weights_expanded).sum(dim=-1)  # [batch, seq_len, num_targets]
        
        # Mixture variance: E[X²] - E[X]²
        second_moment = (weights_expanded * (stds**2 + means**2)).sum(dim=-1)
        mixture_variance = second_moment - mixture_mean**2
        mixture_std = torch.sqrt(mixture_variance.clamp_min(1e-8))
        
        # Epistemic uncertainty (variance of means)
        mean_of_means = mixture_mean
        epistemic_variance = (weights_expanded * (means - mean_of_means.unsqueeze(-1))**2).sum(dim=-1)
        epistemic_std = torch.sqrt(epistemic_variance.clamp_min(1e-8))
        
        # Aleatoric uncertainty (expected variance)
        aleatoric_variance = (weights_expanded * stds**2).sum(dim=-1)
        aleatoric_std = torch.sqrt(aleatoric_variance.clamp_min(1e-8))
        
        return {
            'total_std': mixture_std,
            'epistemic_std': epistemic_std,
            'aleatoric_std': aleatoric_std,
            'mixture_weights': weights
        }


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer decoder."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:x.size(1), :].transpose(0, 1)


class SequentialMixtureNLLLoss(nn.Module):
    """
    Negative Log-Likelihood Loss for Sequential Mixture Density Networks.
    
    Optimized for the sequential output format with proper handling of
    multivariate targets and temporal structure.
    """
    
    def __init__(self, eps: float = 1e-8, reduction: str = 'mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
    
    def forward(self, mixture_params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log-likelihood loss.
        
        Args:
            mixture_params: Tuple (means, log_stds, log_weights)
                - means: [batch, seq_len, num_targets, num_components]
                - log_stds: [batch, seq_len, num_targets, num_components]
                - log_weights: [batch, seq_len, num_components]
            targets: [batch, seq_len, num_targets]
            
        Returns:
            loss: Scalar tensor
        """
        means, log_stds, log_weights = mixture_params
        
        # Ensure targets have the right shape
        if targets.dim() == 3:
            batch_size, seq_len, num_targets = targets.shape
        else:
            raise ValueError(f"Expected targets to be 3D [batch, seq_len, num_targets], got {targets.shape}")
        
        # Convert parameters
        stds = torch.exp(log_stds).clamp_min(self.eps)
        log_weights_normalized = F.log_softmax(log_weights, dim=-1)
        
        # Expand targets to match mixture dimensions
        targets_expanded = targets.unsqueeze(-1).expand(-1, -1, -1, means.size(-1))  # [B, T, num_targets, K]
        
        # Compute log probabilities for each component
        # log N(x|μ,σ) = -0.5*((x-μ)/σ)² - log(σ) - 0.5*log(2π)
        log_probs = (
            -0.5 * ((targets_expanded - means) / stds) ** 2
            - log_stds
            - 0.5 * math.log(2 * math.pi)
        )  # [B, T, num_targets, K]
        
        # Sum log probabilities across targets (assuming independence)
        log_probs_per_component = log_probs.sum(dim=2)  # [B, T, K]
        
        # Add log weights and compute log-sum-exp
        log_weighted_probs = log_weights_normalized + log_probs_per_component  # [B, T, K]
        
        # Stable log-sum-exp
        max_log_prob = torch.max(log_weighted_probs, dim=-1, keepdim=True)[0]
        log_sum_exp = max_log_prob + torch.log(
            torch.sum(torch.exp(log_weighted_probs - max_log_prob), dim=-1, keepdim=True) + self.eps
        )
        log_likelihood = log_sum_exp.squeeze(-1)  # [B, T]
        
        # Negative log-likelihood
        nll = -log_likelihood
        
        # Apply reduction
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll