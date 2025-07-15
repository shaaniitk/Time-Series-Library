import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FourierAttention(nn.Module):
    """Fourier-based attention for capturing periodic patterns"""
    
    def __init__(self, d_model, n_heads, seq_len):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.seq_len = seq_len
        
        # Learnable frequency components
        self.freq_weights = nn.Parameter(torch.randn(seq_len // 2 + 1, n_heads))
        self.phase_weights = nn.Parameter(torch.zeros(seq_len // 2 + 1, n_heads))
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, attn_mask=None):
        B, L, D = x.shape
        
        # Transform to frequency domain
        x_freq = torch.fft.rfft(x, dim=1)
        
        # Apply learnable frequency filtering
        freq_filter = torch.complex(
            torch.cos(self.phase_weights) * self.freq_weights,
            torch.sin(self.phase_weights) * self.freq_weights
        )
        
        x_freq = x_freq.unsqueeze(-1) * freq_filter.unsqueeze(0).unsqueeze(2)
        x_filtered = torch.fft.irfft(x_freq.mean(-1), n=L, dim=1)
        
        # Standard attention on filtered signal
        qkv = self.qkv(x_filtered).reshape(B, L, 3, self.n_heads, D // self.n_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(D // self.n_heads)
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out), attn


class WaveletDecomposition(nn.Module):
    """Learnable wavelet decomposition for multi-resolution analysis"""
    
    def __init__(self, input_dim, levels=3):
        super().__init__()
        self.levels = levels
        
        # Learnable wavelet filters
        self.low_pass = nn.ModuleList([
            nn.Conv1d(input_dim, input_dim, 4, stride=2, padding=1, groups=input_dim)
            for _ in range(levels)
        ])
        
        self.high_pass = nn.ModuleList([
            nn.Conv1d(input_dim, input_dim, 4, stride=2, padding=1, groups=input_dim)
            for _ in range(levels)
        ])
        
        # Reconstruction weights
        self.recon_weights = nn.Parameter(torch.ones(levels + 1) / (levels + 1))
        
    def forward(self, x):
        B, L, D = x.shape
        x = x.transpose(1, 2)  # [B, D, L]
        
        components = []
        current = x
        
        # Decomposition
        for i in range(self.levels):
            low = self.low_pass[i](current)
            high = self.high_pass[i](current)
            components.append(high)
            current = low
            
        components.append(current)  # Final low-frequency component
        
        # Weighted reconstruction
        weights = F.softmax(self.recon_weights, dim=0)
        
        # Upsample and combine
        reconstructed = torch.zeros_like(x)
        for i, (comp, weight) in enumerate(zip(components, weights)):
            if comp.size(-1) < L:
                comp = F.interpolate(comp, size=L, mode='linear', align_corners=False)
            reconstructed += comp * weight
            
        return reconstructed.transpose(1, 2), components


class MetaLearningAdapter(nn.Module):
    """Meta-learning adapter for quick adaptation to new patterns"""
    
    def __init__(self, d_model, adaptation_steps=5):
        super().__init__()
        self.adaptation_steps = adaptation_steps
        
        # Fast adaptation parameters
        self.fast_weights = nn.ParameterList([
            nn.Parameter(torch.randn(d_model, d_model) * 0.01)
            for _ in range(adaptation_steps)
        ])
        
        self.fast_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(d_model))
            for _ in range(adaptation_steps)
        ])
        
        # Meta-learning rate
        self.meta_lr = nn.Parameter(torch.ones(1) * 0.01)
        
    def forward(self, x, support_set=None):
        if support_set is not None and self.training:
            # Fast adaptation using support set
            adapted_weights = []
            adapted_biases = []
            
            for i in range(self.adaptation_steps):
                # Compute gradients on support set
                loss = F.mse_loss(
                    F.linear(support_set, self.fast_weights[i], self.fast_biases[i]),
                    support_set
                )
                
                # Update fast weights
                grad_w = torch.autograd.grad(loss, self.fast_weights[i], create_graph=True)[0]
                grad_b = torch.autograd.grad(loss, self.fast_biases[i], create_graph=True)[0]
                
                adapted_w = self.fast_weights[i] - self.meta_lr * grad_w
                adapted_b = self.fast_biases[i] - self.meta_lr * grad_b
                
                adapted_weights.append(adapted_w)
                adapted_biases.append(adapted_b)
            
            # Use adapted weights
            for w, b in zip(adapted_weights, adapted_biases):
                x = F.linear(x, w, b)
                x = F.relu(x)
        else:
            # Standard forward pass
            for w, b in zip(self.fast_weights, self.fast_biases):
                x = F.linear(x, w, b)
                x = F.relu(x)
                
        return x


class CausalConvolution(nn.Module):
    """Causal convolution for autoregressive modeling"""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
        
    def forward(self, x):
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]  # Remove future information
        return x


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network for sequence modeling"""
    
    def __init__(self, input_size, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(CausalConvolution(
                in_channels, out_channels, kernel_size, dilation=dilation_size
            ))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x.transpose(1, 2)).transpose(1, 2)


class AdaptiveMixture(nn.Module):
    """Adaptive mixture of experts for different time series patterns"""
    
    def __init__(self, d_model, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
                nn.Linear(d_model * 2, d_model)
            ) for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        # Compute gating weights
        gate_weights = self.gate(x)  # [B, L, num_experts]
        
        # Compute expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        
        expert_outputs = torch.stack(expert_outputs, dim=-1)  # [B, L, d_model, num_experts]
        
        # Weighted combination
        output = torch.sum(expert_outputs * gate_weights.unsqueeze(-2), dim=-1)
        
        return output