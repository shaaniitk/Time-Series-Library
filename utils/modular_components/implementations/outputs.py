"""
Output Head Implementations for Modular Framework

This module provides concrete implementations of the BaseOutput interface
for different output tasks like forecasting, regression, classification, etc.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass

from ..base_interfaces import BaseOutput

logger = logging.getLogger(__name__)


@dataclass
class OutputConfig:
    """Configuration for output heads"""
    d_model: int = 512
    output_dim: int = 1
    horizon: int = 1
    dropout: float = 0.1
    use_bias: bool = True
    activation: Optional[str] = None


class ForecastingHead(BaseOutput):
    """
    Output head for time series forecasting tasks
    
    Transforms model hidden states into future predictions.
    Supports both single-step and multi-step forecasting.
    """
    
    def __init__(self, config: OutputConfig):
        from dataclasses import dataclass
        @dataclass
        class MockConfig:
            pass
        
        super().__init__(MockConfig())
        
        self.config = config
        self.d_model = config.d_model
        self.output_dim = config.output_dim
        self.horizon = config.horizon
        self.dropout_rate = config.dropout
        
        # Build forecasting layers
        self.projection = nn.Linear(config.d_model, config.output_dim * config.horizon, bias=config.use_bias)
        self.dropout = nn.Dropout(config.dropout)
        
        # Optional activation
        self.activation = self._get_activation(config.activation) if config.activation else None
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(config.d_model)
        
        logger.info(f"ForecastingHead initialized: horizon={config.horizon}, output_dim={config.output_dim}")
    
    def _get_activation(self, activation_name: str) -> nn.Module:
        """Get activation function by name"""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'softplus': nn.Softplus(),
        }
        
        if activation_name.lower() in activations:
            return activations[activation_name.lower()]
        else:
            logger.warning(f"Unknown activation {activation_name}, using linear output")
            return None
    
    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass for forecasting
        
        Args:
            hidden_states: Model hidden states [batch_size, seq_len, d_model]
            
        Returns:
            Forecasts [batch_size, horizon, output_dim]
        """
        # Use the last hidden state for forecasting
        last_hidden = hidden_states[:, -1, :]  # [batch_size, d_model]
        
        # Layer normalization
        last_hidden = self.layer_norm(last_hidden)
        last_hidden = self.dropout(last_hidden)
        
        # Project to forecast space
        forecast = self.projection(last_hidden)  # [batch_size, horizon * output_dim]
        
        # Reshape to proper forecast format
        batch_size = forecast.size(0)
        forecast = forecast.view(batch_size, self.horizon, self.output_dim)
        
        # Apply activation if specified
        if self.activation is not None:
            forecast = self.activation(forecast)
        
        return forecast
    
    def get_output_type(self) -> str:
        """Return the type identifier of this output head"""
        return "forecasting"
    
    def get_output_dim(self) -> int:
        """Return the output dimension"""
        return self.output_dim
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get output head capabilities"""
        return {
            'type': 'forecasting',
            'horizon': self.horizon,
            'output_dim': self.output_dim,
            'd_model': self.d_model,
            'supports_multistep': True,
            'activation': self.config.activation,
            'dropout': self.dropout_rate
        }


class RegressionHead(BaseOutput):
    """
    Output head for regression tasks
    
    Transforms model hidden states into continuous value predictions.
    """
    
    def __init__(self, config: OutputConfig):
        from dataclasses import dataclass
        @dataclass
        class MockConfig:
            pass
        
        super().__init__(MockConfig())
        
        self.config = config
        self.d_model = config.d_model
        self.output_dim = config.output_dim
        self.dropout_rate = config.dropout
        
        # Build regression layers
        self.hidden_layer = nn.Linear(config.d_model, config.d_model // 2, bias=config.use_bias)
        self.output_layer = nn.Linear(config.d_model // 2, config.output_dim, bias=config.use_bias)
        
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()  # GELU for hidden layer
        
        # Optional output activation
        self.output_activation = self._get_activation(config.activation) if config.activation else None
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.d_model)
        
        logger.info(f"RegressionHead initialized: output_dim={config.output_dim}")
    
    def _get_activation(self, activation_name: str) -> nn.Module:
        """Get activation function by name"""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'softplus': nn.Softplus(),
        }
        
        if activation_name.lower() in activations:
            return activations[activation_name.lower()]
        else:
            logger.warning(f"Unknown activation {activation_name}, using linear output")
            return None
    
    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass for regression
        
        Args:
            hidden_states: Model hidden states [batch_size, seq_len, d_model]
            
        Returns:
            Regression outputs [batch_size, seq_len, output_dim] or [batch_size, output_dim]
        """
        # Option 1: Use all hidden states (sequence-to-sequence)
        # Option 2: Use only last hidden state (sequence-to-one)
        use_sequence = kwargs.get('use_sequence', False)
        
        if use_sequence:
            # Process all positions
            x = hidden_states  # [batch_size, seq_len, d_model]
        else:
            # Use only last position
            x = hidden_states[:, -1, :]  # [batch_size, d_model]
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Hidden layer with activation
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Output layer
        output = self.output_layer(x)
        
        # Apply output activation if specified
        if self.output_activation is not None:
            output = self.output_activation(output)
        
        return output
    
    def get_output_type(self) -> str:
        """Return the type identifier of this output head"""
        return "regression"
    
    def get_output_dim(self) -> int:
        """Return the output dimension"""
        return self.output_dim
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get output head capabilities"""
        return {
            'type': 'regression',
            'output_dim': self.output_dim,
            'd_model': self.d_model,
            'supports_sequence': True,
            'supports_single': True,
            'activation': self.config.activation,
            'dropout': self.dropout_rate
        }


class ClassificationHead(BaseOutput):
    """
    Output head for classification tasks
    
    Transforms model hidden states into class probabilities.
    """
    
    def __init__(self, config: OutputConfig, num_classes: int):
        from dataclasses import dataclass
        @dataclass
        class MockConfig:
            pass
        
        super().__init__(MockConfig())
        
        self.config = config
        self.d_model = config.d_model
        self.num_classes = num_classes
        self.dropout_rate = config.dropout
        
        # Build classification layers
        self.hidden_layer = nn.Linear(config.d_model, config.d_model // 2, bias=config.use_bias)
        self.classifier = nn.Linear(config.d_model // 2, num_classes, bias=config.use_bias)
        
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()  # GELU for hidden layer
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.d_model)
        
        logger.info(f"ClassificationHead initialized: num_classes={num_classes}")
    
    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass for classification
        
        Args:
            hidden_states: Model hidden states [batch_size, seq_len, d_model]
            
        Returns:
            Class logits [batch_size, num_classes] or [batch_size, seq_len, num_classes]
        """
        # Option 1: Use all hidden states (sequence classification)
        # Option 2: Use only last hidden state (single classification)
        use_sequence = kwargs.get('use_sequence', False)
        
        if use_sequence:
            # Process all positions
            x = hidden_states  # [batch_size, seq_len, d_model]
        else:
            # Use only last position or mean pooling
            pooling_method = kwargs.get('pooling', 'last')
            if pooling_method == 'mean':
                x = hidden_states.mean(dim=1)  # [batch_size, d_model]
            else:  # 'last'
                x = hidden_states[:, -1, :]  # [batch_size, d_model]
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Hidden layer with activation
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Classification layer (no activation - raw logits)
        logits = self.classifier(x)
        
        return logits
    
    def get_output_type(self) -> str:
        """Return the type identifier of this output head"""
        return "classification"
    
    def get_output_dim(self) -> int:
        """Return the output dimension"""
        return self.num_classes
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get output head capabilities"""
        return {
            'type': 'classification',
            'num_classes': self.num_classes,
            'd_model': self.d_model,
            'supports_sequence': True,
            'supports_pooling': True,
            'pooling_methods': ['last', 'mean'],
            'dropout': self.dropout_rate
        }


class ProbabilisticForecastingHead(BaseOutput):
    """
    Output head for probabilistic forecasting
    
    Predicts both mean and uncertainty (variance) for forecasts.
    """
    
    def __init__(self, config: OutputConfig):
        from dataclasses import dataclass
        @dataclass
        class MockConfig:
            pass
        
        super().__init__(MockConfig())
        
        self.config = config
        self.d_model = config.d_model
        self.output_dim = config.output_dim
        self.horizon = config.horizon
        self.dropout_rate = config.dropout
        
        # Separate heads for mean and variance
        self.mean_head = nn.Linear(config.d_model, config.output_dim * config.horizon, bias=config.use_bias)
        self.logvar_head = nn.Linear(config.d_model, config.output_dim * config.horizon, bias=config.use_bias)
        
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.d_model)
        
        logger.info(f"ProbabilisticForecastingHead initialized: horizon={config.horizon}, output_dim={config.output_dim}")
    
    def forward(self, hidden_states: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass for probabilistic forecasting
        
        Args:
            hidden_states: Model hidden states [batch_size, seq_len, d_model]
            
        Returns:
            Dictionary with 'mean' and 'logvar' tensors [batch_size, horizon, output_dim]
        """
        # Use the last hidden state for forecasting
        last_hidden = hidden_states[:, -1, :]  # [batch_size, d_model]
        
        # Layer normalization
        last_hidden = self.layer_norm(last_hidden)
        last_hidden = self.dropout(last_hidden)
        
        # Predict mean and log variance
        mean_flat = self.mean_head(last_hidden)  # [batch_size, horizon * output_dim]
        logvar_flat = self.logvar_head(last_hidden)  # [batch_size, horizon * output_dim]
        
        # Reshape to proper forecast format
        batch_size = mean_flat.size(0)
        mean = mean_flat.view(batch_size, self.horizon, self.output_dim)
        logvar = logvar_flat.view(batch_size, self.horizon, self.output_dim)
        
        # Ensure log variance is reasonable (not too negative)
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        return {
            'mean': mean,
            'logvar': logvar,
            'std': torch.exp(0.5 * logvar)
        }
    
    def get_output_type(self) -> str:
        """Return the type identifier of this output head"""
        return "probabilistic_forecasting"
    
    def get_output_dim(self) -> int:
        """Return the output dimension"""
        return self.output_dim
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get output head capabilities"""
        return {
            'type': 'probabilistic_forecasting',
            'horizon': self.horizon,
            'output_dim': self.output_dim,
            'd_model': self.d_model,
            'supports_uncertainty': True,
            'supports_multistep': True,
            'outputs': ['mean', 'logvar', 'std'],
            'dropout': self.dropout_rate
        }


class QuantileForecastingHead(BaseOutput):
    """Quantile forecasting head producing monotonic quantile predictions.

    Given a list of quantiles q1 < q2 < ... < qk, outputs forecasts with shape
    [batch, horizon, output_dim, k]. Monotonicity is enforced by cumulative
    summation over softplus-transformed unconstrained deltas.
    """

    def __init__(self, config: OutputConfig, quantiles: List[float]):
        from dataclasses import dataclass
        @dataclass
        class MockConfig:
            pass
        super().__init__(MockConfig())
        assert all(0 < q < 1 for q in quantiles), "Quantiles must be in (0,1)"
        assert all(x < y for x, y in zip(quantiles, quantiles[1:])), "Quantiles must be strictly increasing"
        self.config = config
        self.quantiles = quantiles
        self.d_model = config.d_model
        self.output_dim = config.output_dim
        self.horizon = config.horizon
        self.num_q = len(quantiles)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.d_model)
        # Base location prediction (median proxy)
        self.base_head = nn.Linear(config.d_model, config.output_dim * config.horizon, bias=config.use_bias)
        # Positive deltas for remaining quantiles (k-1)
        if self.num_q > 1:
            self.delta_head = nn.Linear(config.d_model, (self.num_q - 1) * config.output_dim * config.horizon, bias=config.use_bias)
        else:  # degenerate single quantile
            self.delta_head = None
        self.softplus = nn.Softplus()
        logger.info(f"QuantileForecastingHead initialized: horizon={self.horizon}, output_dim={self.output_dim}, quantiles={quantiles}")

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:  # noqa: D401
        last_hidden = hidden_states[:, -1, :]
        last_hidden = self.layer_norm(last_hidden)
        last_hidden = self.dropout(last_hidden)
        base = self.base_head(last_hidden).view(-1, self.horizon, self.output_dim)  # baseline (approx median)
        if self.num_q == 1:
            q_preds = base.unsqueeze(-1)
        else:
            deltas_raw = self.delta_head(last_hidden)
            deltas = self.softplus(deltas_raw).view(-1, self.horizon, self.output_dim, self.num_q - 1)
            # Cumulative sum to ensure monotonic increments above base
            increments = torch.cumsum(deltas, dim=-1)
            q_stack = torch.cat([base.unsqueeze(-1), base.unsqueeze(-1) + increments], dim=-1)
            q_preds = q_stack
        return {"quantiles": q_preds, "q_levels": torch.tensor(self.quantiles, device=q_preds.device)}

    def get_output_type(self) -> str:
        return "quantile_forecasting"

    def get_output_dim(self) -> int:
        return self.output_dim

    def get_capabilities(self) -> Dict[str, Any]:  # noqa: D401
        return {
            'type': 'quantile_forecasting',
            'horizon': self.horizon,
            'output_dim': self.output_dim,
            'quantiles': self.quantiles,
            'supports_multistep': True,
            'supports_monotonic_quantiles': True,
        }


class MultiTaskHead(BaseOutput):
    """
    Output head for multi-task learning
    
    Supports multiple output tasks simultaneously.
    """
    
    def __init__(self, config: OutputConfig, task_configs: Dict[str, Dict[str, Any]]):
        from dataclasses import dataclass
        @dataclass
        class MockConfig:
            pass
        
        super().__init__(MockConfig())
        
        self.config = config
        self.d_model = config.d_model
        self.task_configs = task_configs
        self.dropout_rate = config.dropout
        
        # Build task-specific heads
        self.task_heads = nn.ModuleDict()
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        for task_name, task_config in task_configs.items():
            task_type = task_config.get('type', 'regression')
            task_output_dim = task_config.get('output_dim', 1)
            
            if task_type == 'regression':
                head = nn.Sequential(
                    nn.Linear(config.d_model, config.d_model // 2),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.d_model // 2, task_output_dim)
                )
            elif task_type == 'classification':
                num_classes = task_config.get('num_classes', 2)
                head = nn.Sequential(
                    nn.Linear(config.d_model, config.d_model // 2),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.d_model // 2, num_classes)
                )
            elif task_type == 'forecasting':
                horizon = task_config.get('horizon', 1)
                head = nn.Linear(config.d_model, task_output_dim * horizon)
            else:
                # Default to linear head
                head = nn.Linear(config.d_model, task_output_dim)
            
            self.task_heads[task_name] = head
        
        logger.info(f"MultiTaskHead initialized with tasks: {list(task_configs.keys())}")
    
    def forward(self, hidden_states: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-task learning
        
        Args:
            hidden_states: Model hidden states [batch_size, seq_len, d_model]
            
        Returns:
            Dictionary with outputs for each task
        """
        # Use the last hidden state
        last_hidden = hidden_states[:, -1, :]  # [batch_size, d_model]
        
        # Layer normalization
        last_hidden = self.layer_norm(last_hidden)
        last_hidden = self.dropout(last_hidden)
        
        # Generate outputs for each task
        outputs = {}
        for task_name, head in self.task_heads.items():
            task_output = head(last_hidden)
            
            # Reshape forecasting outputs
            task_config = self.task_configs[task_name]
            if task_config.get('type') == 'forecasting':
                horizon = task_config.get('horizon', 1)
                output_dim = task_config.get('output_dim', 1)
                batch_size = task_output.size(0)
                task_output = task_output.view(batch_size, horizon, output_dim)
            
            outputs[task_name] = task_output
        
        return outputs
    
    def get_output_type(self) -> str:
        """Return the type identifier of this output head"""
        return "multi_task"
    
    def get_output_dim(self) -> int:
        """Return the total output dimension across all tasks"""
        total_dim = 0
        for task_config in self.task_configs.values():
            task_output_dim = task_config.get('output_dim', 1)
            if task_config.get('type') == 'forecasting':
                horizon = task_config.get('horizon', 1)
                total_dim += task_output_dim * horizon
            else:
                total_dim += task_output_dim
        return total_dim
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get output head capabilities"""
        return {
            'type': 'multi_task',
            'tasks': list(self.task_configs.keys()),
            'task_configs': self.task_configs,
            'd_model': self.d_model,
            'supports_multiple_outputs': True,
            'dropout': self.dropout_rate
        }
