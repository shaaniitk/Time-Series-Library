"""
MambaBlock - Wrapper for Mamba with proper input/output handling for time series
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from layers.Normalization import get_norm_layer
from utils.logger import logger

# Safe Mamba import with fallback
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    logger.warning("mamba_ssm not found. This is normal on non-GPU systems (like macOS). "
                   "Using a standard LSTM as a fallback. Performance will differ.")
    
    class MambaFallback(nn.Module):
        """A fallback implementation using LSTM when mamba_ssm is not available."""
        def __init__(self, d_model, d_state=64, d_conv=4, expand=2):
            super().__init__()
            logger.info(f"Initializing MambaFallback (LSTM) with d_model={d_model}")
            self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
            
        def forward(self, x):
            output, _ = self.lstm(x)
            return output
    
    Mamba = MambaFallback


class MambaBlock(nn.Module):
    """
    Wrapper around Mamba SSM for time series processing.
    Handles input normalization, embedding, and output projection.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        normalize_input: bool = True,
        use_projection: bool = True,
        norm_type: str = 'layernorm'
    ):
        super(MambaBlock, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.normalize_input = normalize_input
        self.use_projection = use_projection
        
        # Input projection if needed
        if input_dim != d_model:
            self.input_projection = nn.Linear(input_dim, d_model)
        else:
            self.input_projection = nn.Identity()
            
        # Input normalization
        if normalize_input:
            self.input_norm = get_norm_layer(norm_type, d_model)
        
        # Core Mamba block
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # Output components
        self.dropout = nn.Dropout(dropout)
        
        if use_projection:
            self.output_projection = nn.Linear(d_model, d_model)
        else:
            self.output_projection = nn.Identity()
            
        # Residual connection
        self.residual_connection = input_dim == d_model
        
        logger.debug(f"MambaBlock initialized: input_dim={input_dim}, d_model={d_model}, "
                    f"d_state={d_state}, d_conv={d_conv}, expand={expand}")
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through MambaBlock.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            mask: Optional attention mask (not used by Mamba but kept for compatibility)
            
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Store original input for residual connection
        residual = x
        
        # Input projection
        x = self.input_projection(x)
        
        # Input normalization
        if self.normalize_input:
            x = self.input_norm(x)
        
        # Mamba processing
        x = self.mamba(x)
        
        # Dropout
        x = self.dropout(x)
        
        # Output projection
        x = self.output_projection(x)
        
        # Residual connection with proper dimension handling
        if self.residual_connection:
            if self.input_dim == self.d_model:
                x = x + residual
            else:
                x = x + self.input_projection(residual)
        
        return x


class TargetMambaBlock(MambaBlock):
    """
    Specialized MambaBlock for target variables processing.
    Includes additional features for target-specific processing.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_trend_decomposition: bool = False
    ):
        super(TargetMambaBlock, self).__init__(
            input_dim=input_dim,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            normalize_input=True,
            use_projection=True
        )
        
        self.use_trend_decomposition = use_trend_decomposition
        
        if use_trend_decomposition:
            # Simple moving average for trend extraction
            self.trend_kernel_size = 25
            self.trend_padding = self.trend_kernel_size // 2
            
        logger.debug(f"TargetMambaBlock initialized with trend_decomposition={use_trend_decomposition}")
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional trend decomposition.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        if self.use_trend_decomposition:
            # Simple trend extraction using moving average
            trend = F.avg_pool1d(
                x.transpose(1, 2),
                kernel_size=self.trend_kernel_size,
                stride=1,
                padding=self.trend_padding
            ).transpose(1, 2)
            
            # Process detrended signal
            detrended = x - trend
            output = super().forward(detrended, mask)
            
            # Add trend back (optional - can be controlled by config)
            # output = output + self.input_projection(trend)
            
            return output
        else:
            return super().forward(x, mask)


class CovariateMambaBlock(MambaBlock):
    """
    Specialized MambaBlock for covariate family processing.
    Optimized for processing groups of related covariates.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        family_size: int = 4,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_family_attention: bool = True
    ):
        super(CovariateMambaBlock, self).__init__(
            input_dim=input_dim,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            normalize_input=True,
            use_projection=True
        )
        
        self.family_size = family_size
        self.use_family_attention = use_family_attention
        
        if use_family_attention and family_size > 1:
            # Simple attention mechanism for family members
            self.family_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=min(8, d_model // 64),
                dropout=dropout,
                batch_first=True
            )
            
        logger.debug(f"CovariateMambaBlock initialized: family_size={family_size}, "
                    f"use_family_attention={use_family_attention}")
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional family attention.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        # Standard Mamba processing
        output = super().forward(x, mask)
        
        # Apply family attention if enabled
        if self.use_family_attention and hasattr(self, 'family_attention'):
            # Apply self-attention across the sequence dimension
            attended_output, _ = self.family_attention(output, output, output, key_padding_mask=mask)
            output = output + attended_output  # Residual connection
        
        return output