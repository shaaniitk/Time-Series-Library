"""
Temporal Convolution Attention Components for Modular Autoformer

This module implements temporal convolution-based attention mechanisms
for capturing local temporal patterns and causal relationships.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base import BaseAttention
from utils.logger import logger


class CausalConvolution(BaseAttention):
    """
    Causal convolution attention mechanism for temporal sequence modeling.
    
    This component uses dilated causal convolutions to capture temporal
    dependencies while maintaining causality constraints.
    """
    
    def __init__(self, d_model, n_heads, kernel_sizes=[3, 5, 7], dilation_rates=[1, 2, 4],
                 dropout=0.1, activation='gelu'):
        super(CausalConvolution, self).__init__()
        logger.info(f"Initializing CausalConvolution: d_model={d_model}, kernels={kernel_sizes}")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.kernel_sizes = kernel_sizes
        self.dilation_rates = dilation_rates
        
        # Multi-scale causal convolutions
        self.causal_convs = nn.ModuleList()
        for kernel_size in kernel_sizes:
            conv_layers = nn.ModuleList()
            for dilation in dilation_rates:
                # Causal padding calculation
                padding = (kernel_size - 1) * dilation
                conv = nn.Conv1d(
                    d_model, d_model, 
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding
                )
                conv_layers.append(conv)
            self.causal_convs.append(conv_layers)
        
        # Attention projection layers
        self.query_conv = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.key_conv = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.value_conv = nn.Conv1d(d_model, d_model, kernel_size=1)
        
        # Output projection
        self.output_projection = nn.Linear(d_model * len(kernel_sizes), d_model)
        
        # Normalization and activation
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()
        
        # Positional encoding for temporal awareness
        self.positional_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        
        self.output_dim_multiplier = 1
    
    def apply_causal_mask(self, x, kernel_size, dilation):
        """
        Apply causal masking by removing future information.
        
        Args:
            x: [B, D, L] convolution output
            kernel_size: Convolution kernel size
            dilation: Dilation rate
            
        Returns:
            masked output with causality preserved
        """
        # Calculate how much to trim from the end
        trim_amount = (kernel_size - 1) * dilation
        if trim_amount > 0:
            x = x[:, :, :-trim_amount]
        return x
    
    def multi_scale_causal_conv(self, x):
        """
        Apply multi-scale causal convolutions.
        
        Args:
            x: [B, L, D] input tensor
            
        Returns:
            List of convolution outputs at different scales
        """
        B, L, D = x.shape
        x_conv = x.transpose(1, 2)  # [B, D, L] for convolution
        
        scale_outputs = []
        
        for kernel_idx, conv_layers in enumerate(self.causal_convs):
            kernel_size = self.kernel_sizes[kernel_idx]
            
            # Apply dilated convolutions at current kernel size
            dilated_outputs = []
            for dilation_idx, conv_layer in enumerate(conv_layers):
                dilation = self.dilation_rates[dilation_idx]
                
                # Apply convolution
                conv_out = conv_layer(x_conv)
                
                # Apply causal masking
                conv_out = self.apply_causal_mask(conv_out, kernel_size, dilation)
                
                # Pad to original length if needed
                if conv_out.size(-1) < L:
                    padding_needed = L - conv_out.size(-1)
                    conv_out = F.pad(conv_out, (0, padding_needed))
                
                # Apply activation
                conv_out = self.activation(conv_out)
                dilated_outputs.append(conv_out)
            
            # Combine dilated outputs (average)
            combined_output = torch.stack(dilated_outputs, dim=0).mean(dim=0)
            scale_outputs.append(combined_output)
        
        return scale_outputs
    
    def temporal_attention(self, queries, keys, values):
        """
        Compute temporal attention with convolution features.
        
        Args:
            queries: [B, D, L] query features
            keys: [B, D, L] key features  
            values: [B, D, L] value features
            
        Returns:
            attention output and weights
        """
        B, D, L = queries.shape
        H = self.n_heads
        d_k = D // H
        
        # Reshape for multi-head processing
        q = queries.view(B, H, d_k, L).transpose(2, 3)  # [B, H, L, d_k]
        k = keys.view(B, H, d_k, L).transpose(2, 3)     # [B, H, L, d_k]
        v = values.view(B, H, d_k, L).transpose(2, 3)   # [B, H, L, d_k]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
        
        # Apply causal mask (future positions set to -inf)
        causal_mask = torch.triu(torch.ones(L, L, device=queries.device), diagonal=1).bool()
        scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)  # [B, H, L, d_k]
        
        # Reshape back
        context = context.transpose(2, 3).contiguous().view(B, D, L)
        
        return context, attn_weights
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Forward pass with causal convolution attention.
        
        Args:
            queries: [B, L, D] query tensor
            keys: [B, L, D] key tensor
            values: [B, L, D] value tensor
            attn_mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        B, L, D = queries.shape
        
        residual = queries
        
        # Apply positional encoding
        pos_encoded = queries + self.positional_conv(queries.transpose(1, 2)).transpose(1, 2)
        
        # Multi-scale causal convolutions
        conv_outputs = self.multi_scale_causal_conv(pos_encoded)
        
        # Convert to attention format
        q_conv = self.query_conv(queries.transpose(1, 2))    # [B, D, L]
        k_conv = self.key_conv(keys.transpose(1, 2))         # [B, D, L]
        v_conv = self.value_conv(values.transpose(1, 2))     # [B, D, L]
        
        # Add convolution features to keys and values
        enhanced_keys = k_conv
        enhanced_values = v_conv
        
        for conv_out in conv_outputs:
            enhanced_keys = enhanced_keys + conv_out
            enhanced_values = enhanced_values + conv_out
        
        # Apply temporal attention
        attn_output, attn_weights = self.temporal_attention(q_conv, enhanced_keys, enhanced_values)
        
        # Combine multi-scale outputs
        combined_conv = torch.cat(conv_outputs, dim=1)  # [B, D*len(kernels), L]
        combined_conv = combined_conv.transpose(1, 2)   # [B, L, D*len(kernels)]
        
        # Project to original dimension
        conv_features = self.output_projection(combined_conv)
        
        # Combine attention and convolution features
        final_output = attn_output.transpose(1, 2) + conv_features
        
        # Residual connection and normalization
        output = self.layer_norm(final_output + residual)
        
        return output, attn_weights


class TemporalConvNet(BaseAttention):
    """
    Temporal Convolution Network (TCN) based attention mechanism.
    
    This implements a full TCN architecture for sequence modeling
    with attention-like interfaces.
    """
    
    def __init__(self, d_model, n_heads, num_levels=4, kernel_size=3, dropout=0.1):
        super(TemporalConvNet, self).__init__()
        logger.info(f"Initializing TemporalConvNet: levels={num_levels}, kernel={kernel_size}")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_levels = num_levels
        self.kernel_size = kernel_size
        
        # Build TCN layers with exponentially increasing dilation
        self.tcn_layers = nn.ModuleList()
        dilation_size = 1
        
        for i in range(num_levels):
            # Residual block with dilated convolution
            layer = TemporalBlock(
                d_model, d_model, kernel_size, 
                stride=1, dilation=dilation_size, 
                padding=(kernel_size-1) * dilation_size, 
                dropout=dropout
            )
            self.tcn_layers.append(layer)
            dilation_size *= 2  # Exponential dilation
        
        # Attention mechanism for combining temporal features
        self.temporal_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # Output processing
        self.output_norm = nn.LayerNorm(d_model)
        self.output_dropout = nn.Dropout(dropout)
        
        self.output_dim_multiplier = 1
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Forward pass through TCN with attention combination.
        
        Args:
            queries: [B, L, D] query tensor
            keys: [B, L, D] key tensor (used as input)
            values: [B, L, D] value tensor (used as input)
            attn_mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        B, L, D = queries.shape
        
        # Use keys as primary input for TCN processing
        x = keys
        residual = x
        
        # Apply TCN layers sequentially
        temporal_features = []
        current_input = x.transpose(1, 2)  # [B, D, L] for conv
        
        for tcn_layer in self.tcn_layers:
            current_input = tcn_layer(current_input)
            # Store features from each level
            temporal_features.append(current_input.transpose(1, 2))  # Back to [B, L, D]
        
        # Final TCN output
        tcn_output = current_input.transpose(1, 2)  # [B, L, D]
        
        # Apply attention to combine with queries
        attn_output, attn_weights = self.temporal_attention(
            queries, tcn_output, values, attn_mask=attn_mask
        )
        
        # Apply output processing
        output = self.output_norm(attn_output + residual)
        output = self.output_dropout(output)
        
        return output, attn_weights


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


class ConvolutionalAttention(BaseAttention):
    """
    Convolutional attention mechanism combining spatial and temporal convolutions.
    
    This component uses 2D convolutions to capture both spatial (feature)
    and temporal relationships in attention computation.
    """
    
    def __init__(self, d_model, n_heads, conv_kernel_size=3, pool_size=2, dropout=0.1):
        super(ConvolutionalAttention, self).__init__()
        logger.info(f"Initializing ConvolutionalAttention: d_model={d_model}")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.conv_kernel_size = conv_kernel_size
        
        # 2D convolutions for attention feature extraction
        self.spatial_conv = nn.Conv2d(
            1, n_heads, 
            kernel_size=(conv_kernel_size, conv_kernel_size),
            padding=(conv_kernel_size//2, conv_kernel_size//2)
        )
        
        # Temporal convolutions
        self.temporal_conv_q = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.temporal_conv_k = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.temporal_conv_v = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        
        # Standard attention projections
        self.w_qs = nn.Linear(d_model, d_model)
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Pooling for dimension reduction
        self.adaptive_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.output_dim_multiplier = 1
    
    def compute_conv_attention_weights(self, queries, keys):
        """
        Compute attention weights using 2D convolutions.
        
        Args:
            queries: [B, L, D] query tensor
            keys: [B, L, D] key tensor
            
        Returns:
            conv_attention_weights: [B, H, L, L] attention weights
        """
        B, L, D = queries.shape
        H = self.n_heads
        
        # Compute pairwise interactions
        q_expanded = queries.unsqueeze(2).expand(B, L, L, D)  # [B, L, L, D]
        k_expanded = keys.unsqueeze(1).expand(B, L, L, D)     # [B, L, L, D]
        
        # Element-wise product for interaction
        interactions = q_expanded * k_expanded  # [B, L, L, D]
        
        # Average across feature dimension and add channel dimension
        interaction_map = interactions.mean(dim=-1, keepdim=True)  # [B, L, L, 1]
        interaction_map = interaction_map.permute(0, 3, 1, 2)     # [B, 1, L, L]
        
        # Apply 2D convolution to capture spatial patterns
        conv_weights = self.spatial_conv(interaction_map)  # [B, H, L, L]
        
        # Apply softmax to get attention weights
        conv_attention_weights = F.softmax(conv_weights, dim=-1)
        
        return conv_attention_weights
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Forward pass with convolutional attention.
        
        Args:
            queries: [B, L, D] query tensor
            keys: [B, L, D] key tensor
            values: [B, L, D] value tensor
            attn_mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        B, L, D = queries.shape
        H = self.n_heads
        
        residual = queries
        
        # Apply temporal convolutions
        q_temp = self.temporal_conv_q(queries.transpose(1, 2)).transpose(1, 2)
        k_temp = self.temporal_conv_k(keys.transpose(1, 2)).transpose(1, 2)
        v_temp = self.temporal_conv_v(values.transpose(1, 2)).transpose(1, 2)
        
        # Apply standard projections
        q = self.w_qs(q_temp).view(B, L, H, self.d_k).transpose(1, 2)
        k = self.w_ks(k_temp).view(B, L, H, self.d_k).transpose(1, 2)
        v = self.w_vs(v_temp).view(B, L, H, self.d_k).transpose(1, 2)
        
        # Compute standard attention scores
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)
        
        # Compute convolutional attention weights
        conv_attn_weights = self.compute_conv_attention_weights(q_temp, k_temp)
        
        # Combine standard and convolutional attention
        combined_scores = scores + conv_attn_weights
        
        # Apply attention mask if provided
        if attn_mask is not None:
            combined_scores.masked_fill_(attn_mask == 0, -1e9)
        
        # Compute final attention weights
        attn_weights = F.softmax(combined_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        
        # Apply output projection
        output = self.w_o(context)
        
        # Residual connection and normalization
        output = self.layer_norm(output + residual)
        
        return output, attn_weights
