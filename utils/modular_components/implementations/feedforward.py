"""
FeedForward Network Implementations for Modular Framework

This module provides concrete implementations of the BaseFeedForward interface
for different feed-forward network architectures used in transformer models.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass

from ..base_interfaces import BaseFeedForward

logger = logging.getLogger(__name__)


@dataclass
class FFNConfig:
    """Configuration for feed-forward networks"""
    d_model: int = 512
    d_ff: int = 2048
    dropout: float = 0.1
    activation: str = 'relu'
    use_bias: bool = True
    layer_norm: bool = False


class StandardFFN(BaseFeedForward):
    """
    Standard feed-forward network used in vanilla Transformers
    
    Architecture: Linear -> Activation -> Dropout -> Linear -> Dropout
    """
    
    def __init__(self, config: FFNConfig):
        from dataclasses import dataclass
        @dataclass
        class MockConfig:
            pass
        
        super().__init__(MockConfig())
        
        self.config = config
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.dropout_rate = config.dropout
        
        # Build feed-forward layers
        self.linear1 = nn.Linear(config.d_model, config.d_ff, bias=config.use_bias)
        self.linear2 = nn.Linear(config.d_ff, config.d_model, bias=config.use_bias)
        
        # Activation function
        self.activation = self._get_activation(config.activation)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        
        # Optional layer normalization
        self.layer_norm = nn.LayerNorm(config.d_model) if config.layer_norm else None
        
        logger.info(f"StandardFFN initialized: d_model={config.d_model}, d_ff={config.d_ff}")
    
    def _get_activation(self, activation_name: str) -> nn.Module:
        """Get activation function by name"""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
        }
        
        if activation_name.lower() in activations:
            return activations[activation_name.lower()]
        else:
            logger.warning(f"Unknown activation {activation_name}, defaulting to ReLU")
            return nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through standard FFN
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # First linear layer + activation + dropout
        hidden = self.linear1(x)
        hidden = self.activation(hidden)
        hidden = self.dropout1(hidden)
        
        # Second linear layer + dropout
        output = self.linear2(hidden)
        output = self.dropout2(output)
        
        # Optional layer normalization
        if self.layer_norm is not None:
            output = self.layer_norm(output)
        
        return output
    
    def apply_feedforward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward transformation (alias for forward)"""
        return self.forward(x)
    
    def get_ffn_type(self) -> str:
        """Return the type identifier of this feed-forward network"""
        return "standard_ffn"
    
    def get_output_dim(self) -> int:
        """Return the output dimension"""
        return self.d_model
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get FFN capabilities"""
        return {
            'type': 'standard_ffn',
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'activation': self.config.activation,
            'supports_layer_norm': self.config.layer_norm,
            'dropout': self.dropout_rate
        }


class GatedFFN(BaseFeedForward):
    """
    Gated feed-forward network (GLU-style)
    
    Uses gating mechanism for better control over information flow.
    Architecture: (Linear_gate, Linear_value) -> Gate * Value -> Linear_out
    """
    
    def __init__(self, config: FFNConfig):
        from dataclasses import dataclass
        @dataclass
        class MockConfig:
            pass
        
        super().__init__(MockConfig())
        
        self.config = config
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.dropout_rate = config.dropout
        
        # Gated architecture: separate gate and value projections
        self.gate_projection = nn.Linear(config.d_model, config.d_ff, bias=config.use_bias)
        self.value_projection = nn.Linear(config.d_model, config.d_ff, bias=config.use_bias)
        self.output_projection = nn.Linear(config.d_ff, config.d_model, bias=config.use_bias)
        
        # Activation functions
        self.gate_activation = nn.Sigmoid()  # Gate uses sigmoid
        self.value_activation = self._get_activation(config.activation)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Optional layer normalization
        self.layer_norm = nn.LayerNorm(config.d_model) if config.layer_norm else None
        
        logger.info(f"GatedFFN initialized: d_model={config.d_model}, d_ff={config.d_ff}")
    
    def _get_activation(self, activation_name: str) -> nn.Module:
        """Get activation function by name"""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
        }
        
        if activation_name.lower() in activations:
            return activations[activation_name.lower()]
        else:
            logger.warning(f"Unknown activation {activation_name}, defaulting to GELU")
            return nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through gated FFN
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Compute gate and value
        gate = self.gate_activation(self.gate_projection(x))
        value = self.value_activation(self.value_projection(x))
        
        # Apply gating
        gated = gate * value
        gated = self.dropout(gated)
        
        # Output projection
        output = self.output_projection(gated)
        output = self.dropout(output)
        
        # Optional layer normalization
        if self.layer_norm is not None:
            output = self.layer_norm(output)
        
        return output
    
    def apply_feedforward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward transformation (alias for forward)"""
        return self.forward(x)
    
    def get_ffn_type(self) -> str:
        """Return the type identifier of this feed-forward network"""
        return "gated_ffn"
    
    def get_output_dim(self) -> int:
        """Return the output dimension"""
        return self.d_model
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get FFN capabilities"""
        return {
            'type': 'gated_ffn',
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'gated_architecture': True,
            'activation': self.config.activation,
            'supports_layer_norm': self.config.layer_norm,
            'dropout': self.dropout_rate
        }


class MoEFFN(BaseFeedForward):
    """
    Mixture of Experts Feed-Forward Network
    
    Routes inputs to different expert networks based on learned gating.
    """
    
    def __init__(self, config: FFNConfig, num_experts: int = 8, num_selected: int = 2):
        from dataclasses import dataclass
        @dataclass
        class MockConfig:
            pass
        
        super().__init__(MockConfig())
        
        self.config = config
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.num_experts = num_experts
        self.num_selected = num_selected
        self.dropout_rate = config.dropout
        
        # Gating network
        self.gate = nn.Linear(config.d_model, num_experts, bias=False)
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.d_model, config.d_ff, bias=config.use_bias),
                self._get_activation(config.activation),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_ff, config.d_model, bias=config.use_bias),
                nn.Dropout(config.dropout)
            )
            for _ in range(num_experts)
        ])
        
        # Optional layer normalization
        self.layer_norm = nn.LayerNorm(config.d_model) if config.layer_norm else None
        
        logger.info(f"MoEFFN initialized: {num_experts} experts, select top-{num_selected}")
    
    def _get_activation(self, activation_name: str) -> nn.Module:
        """Get activation function by name"""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
        }
        
        if activation_name.lower() in activations:
            return activations[activation_name.lower()]
        else:
            logger.warning(f"Unknown activation {activation_name}, defaulting to GELU")
            return nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MoE FFN
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Flatten for easier processing
        x_flat = x.view(-1, d_model)  # [batch_size * seq_len, d_model]
        
        # Compute gating weights
        gate_logits = self.gate(x_flat)  # [batch_size * seq_len, num_experts]
        gate_weights = F.softmax(gate_logits, dim=-1)
        
        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.num_selected, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1)  # Renormalize
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Route to selected experts
        for i in range(self.num_selected):
            expert_idx = top_k_indices[:, i]
            expert_weight = top_k_weights[:, i].unsqueeze(-1)
            
            # Apply experts
            for expert_id in range(self.num_experts):
                mask = (expert_idx == expert_id)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    output[mask] += expert_weight[mask] * expert_output
        
        # Reshape back
        output = output.view(batch_size, seq_len, d_model)
        
        # Optional layer normalization
        if self.layer_norm is not None:
            output = self.layer_norm(output)
        
        return output
    
    def apply_feedforward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward transformation (alias for forward)"""
        return self.forward(x)
    
    def get_ffn_type(self) -> str:
        """Return the type identifier of this feed-forward network"""
        return "moe_ffn"
    
    def get_output_dim(self) -> int:
        """Return the output dimension"""
        return self.d_model
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get FFN capabilities"""
        return {
            'type': 'moe_ffn',
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'num_experts': self.num_experts,
            'num_selected': self.num_selected,
            'mixture_of_experts': True,
            'activation': self.config.activation,
            'supports_layer_norm': self.config.layer_norm,
            'dropout': self.dropout_rate
        }


class ConvFFN(BaseFeedForward):
    """
    Convolutional Feed-Forward Network
    
    Uses 1D convolutions instead of linear layers for better inductive bias
    in sequential data processing.
    """
    
    def __init__(self, config: FFNConfig, kernel_size: int = 3):
        from dataclasses import dataclass
        @dataclass
        class MockConfig:
            pass
        
        super().__init__(MockConfig())
        
        self.config = config
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.kernel_size = kernel_size
        self.dropout_rate = config.dropout
        
        # Convolutional layers
        padding = kernel_size // 2  # Same padding
        self.conv1 = nn.Conv1d(config.d_model, config.d_ff, kernel_size, padding=padding, bias=config.use_bias)
        self.conv2 = nn.Conv1d(config.d_ff, config.d_model, kernel_size, padding=padding, bias=config.use_bias)
        
        # Activation function
        self.activation = self._get_activation(config.activation)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        
        # Optional layer normalization
        self.layer_norm = nn.LayerNorm(config.d_model) if config.layer_norm else None
        
        logger.info(f"ConvFFN initialized: d_model={config.d_model}, d_ff={config.d_ff}, kernel_size={kernel_size}")
    
    def _get_activation(self, activation_name: str) -> nn.Module:
        """Get activation function by name"""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
        }
        
        if activation_name.lower() in activations:
            return activations[activation_name.lower()]
        else:
            logger.warning(f"Unknown activation {activation_name}, defaulting to GELU")
            return nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through convolutional FFN
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Conv1d expects [batch_size, d_model, seq_len]
        x_transposed = x.transpose(1, 2)
        
        # First conv layer + activation + dropout
        hidden = self.conv1(x_transposed)
        hidden = self.activation(hidden)
        hidden = self.dropout1(hidden)
        
        # Second conv layer + dropout
        output = self.conv2(hidden)
        output = self.dropout2(output)
        
        # Transpose back to [batch_size, seq_len, d_model]
        output = output.transpose(1, 2)
        
        # Optional layer normalization
        if self.layer_norm is not None:
            output = self.layer_norm(output)
        
        return output
    
    def apply_feedforward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward transformation (alias for forward)"""
        return self.forward(x)
    
    def get_ffn_type(self) -> str:
        """Return the type identifier of this feed-forward network"""
        return "conv_ffn"
    
    def get_output_dim(self) -> int:
        """Return the output dimension"""
        return self.d_model
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get FFN capabilities"""
        return {
            'type': 'conv_ffn',
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'kernel_size': self.kernel_size,
            'convolutional': True,
            'activation': self.config.activation,
            'supports_layer_norm': self.config.layer_norm,
            'dropout': self.dropout_rate
        }
