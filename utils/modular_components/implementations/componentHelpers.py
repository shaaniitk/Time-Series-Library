"""
COMPONENT HELPERS - UTILITY CLASSES
Helper classes and utility modules used by attention and other components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# BAYESIAN HELPERS
# =============================================================================

class BayesianLinear(nn.Module):
    """
    Bayesian linear layer with variational inference for uncertainty quantification.
    
    This layer maintains distributions over weights rather than point estimates,
    enabling uncertainty-aware predictions.
    """
    
    def __init__(self, in_features, out_features, prior_std=1.0):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # Mean parameters for weights and biases
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        
        # Log variance parameters (to ensure positivity)
        self.weight_log_var = nn.Parameter(torch.randn(out_features, in_features) * 0.01 - 3)
        self.bias_log_var = nn.Parameter(torch.zeros(out_features) - 3)
        
    def forward(self, x):
        """
        Forward pass with weight sampling from posterior distribution.
        
        Args:
            x: [B, *, in_features] input tensor
            
        Returns:
            output: [B, *, out_features] output tensor
        """
        if self.training:
            # Sample weights from posterior during training
            weight_std = torch.exp(0.5 * self.weight_log_var)
            bias_std = torch.exp(0.5 * self.bias_log_var)
            
            # Reparameterization trick
            weight_eps = torch.randn_like(self.weight_mu)
            bias_eps = torch.randn_like(self.bias_mu)
            
            weight = self.weight_mu + weight_std * weight_eps
            bias = self.bias_mu + bias_std * bias_eps
        else:
            # Use mean parameters during inference
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self):
        """
        Compute KL divergence between posterior and prior distributions.
        
        Returns:
            KL divergence scalar
        """
        # Prior distribution: N(0, prior_std^2)
        # Posterior distribution: N(mu, exp(log_var))
        
        weight_var = torch.exp(self.weight_log_var)
        bias_var = torch.exp(self.bias_log_var)
        
        # KL for weights
        weight_kl = 0.5 * torch.sum(
            (self.weight_mu ** 2 + weight_var) / (self.prior_std ** 2) 
            - 1 - self.weight_log_var + 2 * math.log(self.prior_std)
        )
        
        # KL for biases
        bias_kl = 0.5 * torch.sum(
            (self.bias_mu ** 2 + bias_var) / (self.prior_std ** 2) 
            - 1 - self.bias_log_var + 2 * math.log(self.prior_std)
        )
        
        return weight_kl + bias_kl


# =============================================================================
# TEMPORAL CONVOLUTION HELPERS
# =============================================================================

class TemporalBlock(nn.Module):
    """
    Temporal block for TCN with residual connections and causal convolutions.
    """
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # First convolution layer
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        
        # Apply causal masking by chopping off the future
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second convolution layer
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Combine layers
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # Downsample for residual connection if needed
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize convolution weights."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        """Forward pass through temporal block."""
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """
    Remove rightmost elements to ensure causality in convolutions.
    """
    
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        """Remove future information from convolution output."""
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size].contiguous()
        return x


# =============================================================================
# WAVELET HELPERS
# =============================================================================

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
                comp_upsampled = comp[:, :L, :]  # Truncate if too long
                
            reconstructed += comp_upsampled * weight
        
        return reconstructed, components


# =============================================================================
# FOURIER HELPERS
# =============================================================================

class FourierModeSelector:
    """Helper class for selecting frequency modes in Fourier-based components"""
    
    @staticmethod
    def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
        """
        Select frequency modes for Fourier transforms.
        
        Args:
            seq_len: Sequence length
            modes: Number of modes to select
            mode_select_method: Method for mode selection ('random', 'low', 'high')
            
        Returns:
            Selected mode indices
        """
        if mode_select_method == 'random':
            modes = min(modes, seq_len // 2)
            index = list(range(0, seq_len // 2))
            np.random.shuffle(index)
            index = index[:modes]
        elif mode_select_method == 'low':
            modes = min(modes, seq_len // 2)
            index = list(range(0, modes))
        elif mode_select_method == 'high':
            modes = min(modes, seq_len // 2)
            index = list(range(seq_len // 2 - modes, seq_len // 2))
        else:
            raise ValueError(f"Unsupported mode selection method: {mode_select_method}")
        
        return index


class ComplexMultiply1D:
    """Helper for complex multiplication in frequency domain"""
    
    @staticmethod
    def complex_mul1d(x, weights_real, weights_imag):
        """
        Complex multiplication for 1D Fourier operations.
        
        Args:
            x: Complex input tensor
            weights_real: Real part of weights
            weights_imag: Imaginary part of weights
            
        Returns:
            Complex multiplication result
        """
        real_part = x.real * weights_real - x.imag * weights_imag
        imag_part = x.real * weights_imag + x.imag * weights_real
        return torch.complex(real_part, imag_part)


# =============================================================================
# ATTENTION MASK HELPERS
# =============================================================================

class AttentionMaskGenerator:
    """Helper class for generating various attention masks"""
    
    @staticmethod
    def create_causal_mask(seq_len, device=None):
        """Create a causal (lower triangular) mask"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.bool()
    
    @staticmethod
    def create_sparse_mask(seq_len, sparsity_factor=0.1, device=None):
        """Create a sparse attention mask"""
        mask = torch.zeros(seq_len, seq_len, device=device)
        k = int(seq_len * sparsity_factor)
        if k > 0:
            indices = torch.randperm(seq_len, device=device)[:k]
            mask[:, indices] = 1
        return mask.bool()
    
    @staticmethod
    def create_local_mask(seq_len, window_size, device=None):
        """Create a local attention mask with sliding window"""
        mask = torch.zeros(seq_len, seq_len, device=device)
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = 1
        return mask.bool()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def apply_attention_dropout(attention_weights, dropout_rate=0.1, training=True):
    """Apply dropout to attention weights during training"""
    if training and dropout_rate > 0:
        return F.dropout(attention_weights, p=dropout_rate, training=training)
    return attention_weights


def normalize_attention_weights(attention_scores, dim=-1, method='softmax'):
    """Normalize attention scores to weights"""
    if method == 'softmax':
        return F.softmax(attention_scores, dim=dim)
    elif method == 'sigmoid':
        return torch.sigmoid(attention_scores)
    elif method == 'tanh':
        return torch.tanh(attention_scores)
    else:
        raise ValueError(f"Unsupported normalization method: {method}")


def compute_attention_entropy(attention_weights, dim=-1):
    """Compute entropy of attention weights for analysis"""
    log_weights = torch.log(attention_weights + 1e-8)
    entropy = -torch.sum(attention_weights * log_weights, dim=dim)
    return entropy


logger.info("✅ Component helpers loaded successfully")
