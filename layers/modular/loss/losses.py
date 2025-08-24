"""
Loss Function Implementations for Modular Framework

This module provides concrete implementations of the BaseLoss interface
for different loss functions used in time series and ML tasks.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass

from ..base_interfaces import BaseLoss

logger = logging.getLogger(__name__)


@dataclass
class LossConfig:
    """Configuration for loss functions"""
    reduction: str = 'mean'  # 'mean', 'sum', 'none'
    weight: Optional[torch.Tensor] = None
    ignore_index: int = -100


class MSELoss(BaseLoss):
    """
    Mean Squared Error loss for regression tasks
    """
    
    def __init__(self, config: LossConfig):
        from dataclasses import dataclass
        @dataclass
        class MockConfig:
            pass
        
        super().__init__(MockConfig())
        
        self.config = config
        self.mse_loss = nn.MSELoss(reduction=config.reduction)
        
        logger.info(f"MSELoss initialized with reduction={config.reduction}")
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute MSE loss
        
        Args:
            predictions: Model predictions [batch_size, ..., output_dim]
            targets: Ground truth targets [batch_size, ..., output_dim]
            
        Returns:
            MSE loss tensor
        """
        return self.mse_loss(predictions, targets)
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute loss between predictions and targets"""
        return self.forward(predictions, targets, **kwargs)
    
    def get_loss_type(self) -> str:
        """Return the type identifier of this loss function"""
        return "mse"
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get loss function capabilities"""
        return {
            'type': 'mse',
            'task_types': ['regression', 'forecasting'],
            'reduction': self.config.reduction,
            'differentiable': True
        }


class MAELoss(BaseLoss):
    """
    Mean Absolute Error loss for regression tasks
    """
    
    def __init__(self, config: LossConfig):
        from dataclasses import dataclass
        @dataclass
        class MockConfig:
            pass
        
        super().__init__(MockConfig())
        
        self.config = config
        self.mae_loss = nn.L1Loss(reduction=config.reduction)
        
        logger.info(f"MAELoss initialized with reduction={config.reduction}")
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute MAE loss
        
        Args:
            predictions: Model predictions [batch_size, ..., output_dim]
            targets: Ground truth targets [batch_size, ..., output_dim]
            
        Returns:
            MAE loss tensor
        """
        return self.mae_loss(predictions, targets)
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute loss between predictions and targets"""
        return self.forward(predictions, targets, **kwargs)
    
    def get_loss_type(self) -> str:
        """Return the type identifier of this loss function"""
        return "mae"
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get loss function capabilities"""
        return {
            'type': 'mae',
            'task_types': ['regression', 'forecasting'],
            'reduction': self.config.reduction,
            'differentiable': True,
            'robust_to_outliers': True
        }


class CrossEntropyLoss(BaseLoss):
    """
    Cross-entropy loss for classification tasks
    """
    
    def __init__(self, config: LossConfig, num_classes: Optional[int] = None):
        from dataclasses import dataclass
        @dataclass
        class MockConfig:
            pass
        
        super().__init__(MockConfig())
        
        self.config = config
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss(
            weight=config.weight,
            ignore_index=config.ignore_index,
            reduction=config.reduction
        )
        
        logger.info(f"CrossEntropyLoss initialized with {num_classes} classes")
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute cross-entropy loss
        
        Args:
            predictions: Model logits [batch_size, num_classes] or [batch_size, seq_len, num_classes]
            targets: Ground truth class indices [batch_size] or [batch_size, seq_len]
            
        Returns:
            Cross-entropy loss tensor
        """
        return self.ce_loss(predictions, targets)
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute loss between predictions and targets"""
        return self.forward(predictions, targets, **kwargs)
    
    def get_loss_type(self) -> str:
        """Return the type identifier of this loss function"""
        return "cross_entropy"
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get loss function capabilities"""
        return {
            'type': 'cross_entropy',
            'task_types': ['classification'],
            'num_classes': self.num_classes,
            'supports_class_weights': True,
            'supports_ignore_index': True,
            'reduction': self.config.reduction,
            'differentiable': True
        }


class HuberLoss(BaseLoss):
    """
    Huber loss for robust regression
    """
    
    def __init__(self, config: LossConfig, delta: float = 1.0):
        from dataclasses import dataclass
        @dataclass
        class MockConfig:
            pass
        
        super().__init__(MockConfig())
        
        self.config = config
        self.delta = delta
        self.huber_loss = nn.HuberLoss(reduction=config.reduction, delta=delta)
        
        logger.info(f"HuberLoss initialized with delta={delta}")
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute Huber loss
        
        Args:
            predictions: Model predictions [batch_size, ..., output_dim]
            targets: Ground truth targets [batch_size, ..., output_dim]
            
        Returns:
            Huber loss tensor
        """
        return self.huber_loss(predictions, targets)
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute loss between predictions and targets"""
        return self.forward(predictions, targets, **kwargs)
    
    def get_loss_type(self) -> str:
        """Return the type identifier of this loss function"""
        return "huber"
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get loss function capabilities"""
        return {
            'type': 'huber',
            'task_types': ['regression', 'forecasting'],
            'delta': self.delta,
            'reduction': self.config.reduction,
            'differentiable': True,
            'robust_to_outliers': True
        }


class NegativeLogLikelihood(BaseLoss):
    """
    Negative log-likelihood loss for probabilistic models
    """
    
    def __init__(self, config: LossConfig):
        from dataclasses import dataclass
        @dataclass
        class MockConfig:
            pass
        
        super().__init__(MockConfig())
        
        self.config = config
        
        logger.info("NegativeLogLikelihood initialized")
    
    def forward(self, predictions: Dict[str, torch.Tensor], targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute negative log-likelihood loss
        
        Args:
            predictions: Dictionary with 'mean' and 'logvar' tensors
            targets: Ground truth targets [batch_size, ..., output_dim]
            
        Returns:
            NLL loss tensor
        """
        mean = predictions['mean']
        logvar = predictions['logvar']
        
        # Compute negative log-likelihood for Gaussian
        # NLL = 0.5 * (log(2π) + logvar + (targets - mean)^2 / exp(logvar))
        mse = (targets - mean) ** 2
        var = torch.exp(logvar)
        nll = 0.5 * (torch.log(2 * torch.pi * var) + mse / var)
        
        if self.config.reduction == 'mean':
            return nll.mean()
        elif self.config.reduction == 'sum':
            return nll.sum()
        else:
            return nll
    
    def compute_loss(self, predictions: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                    targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute loss between predictions and targets"""
        if isinstance(predictions, dict):
            return self.forward(predictions, targets, **kwargs)
        else:
            raise ValueError("NLL loss requires dictionary input with 'mean' and 'logvar'")
    
    def get_loss_type(self) -> str:
        """Return the type identifier of this loss function"""
        return "negative_log_likelihood"
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get loss function capabilities"""
        return {
            'type': 'negative_log_likelihood',
            'task_types': ['probabilistic_forecasting', 'probabilistic_regression'],
            'requires_uncertainty': True,
            'reduction': self.config.reduction,
            'differentiable': True
        }


class QuantileLoss(BaseLoss):
    """
    Quantile loss for quantile regression
    """
    
    def __init__(self, config: LossConfig, quantiles: List[float] = [0.1, 0.5, 0.9]):
        from dataclasses import dataclass
        @dataclass
        class MockConfig:
            pass
        
        super().__init__(MockConfig())
        
        self.config = config
        self.quantiles = torch.tensor(quantiles, dtype=torch.float32)
        
        logger.info(f"QuantileLoss initialized with quantiles={quantiles}")
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute quantile loss
        
        Args:
            predictions: Quantile predictions [batch_size, ..., num_quantiles]
            targets: Ground truth targets [batch_size, ..., 1]
            
        Returns:
            Quantile loss tensor
        """
        device = predictions.device
        quantiles = self.quantiles.to(device)
        
        # Expand targets to match quantile predictions
        if targets.dim() == predictions.dim() - 1:
            targets = targets.unsqueeze(-1)
        
        targets = targets.expand_as(predictions)
        
        # Compute quantile loss
        errors = targets - predictions
        losses = torch.max(quantiles * errors, (quantiles - 1) * errors)
        
        if self.config.reduction == 'mean':
            return losses.mean()
        elif self.config.reduction == 'sum':
            return losses.sum()
        else:
            return losses
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute loss between predictions and targets"""
        return self.forward(predictions, targets, **kwargs)
    
    def get_loss_type(self) -> str:
        """Return the type identifier of this loss function"""
        return "quantile"
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get loss function capabilities"""
        return {
            'type': 'quantile',
            'task_types': ['quantile_regression', 'probabilistic_forecasting'],
            'quantiles': self.quantiles.tolist(),
            'reduction': self.config.reduction,
            'differentiable': True,
            'provides_uncertainty': True
        }


class MultiTaskLoss(BaseLoss):
    """
    Multi-task loss combining multiple loss functions
    """
    
    def __init__(self, config: LossConfig, task_losses: Dict[str, BaseLoss], task_weights: Optional[Dict[str, float]] = None):
        from dataclasses import dataclass
        @dataclass
        class MockConfig:
            pass
        
        super().__init__(MockConfig())
        
        self.config = config
        self.task_losses = nn.ModuleDict(task_losses)
        self.task_weights = task_weights or {task: 1.0 for task in task_losses.keys()}
        
        logger.info(f"MultiTaskLoss initialized with tasks: {list(task_losses.keys())}")
    
    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss
        
        Args:
            predictions: Dictionary of task predictions
            targets: Dictionary of task targets
            
        Returns:
            Dictionary of task losses plus total loss
        """
        losses = {}
        total_loss = 0.0
        
        for task_name, loss_fn in self.task_losses.items():
            if task_name in predictions and task_name in targets:
                task_loss = loss_fn.compute_loss(predictions[task_name], targets[task_name], **kwargs)
                weight = self.task_weights.get(task_name, 1.0)
                weighted_loss = weight * task_loss
                
                losses[task_name] = task_loss
                total_loss = total_loss + weighted_loss
            else:
                logger.warning(f"Missing predictions or targets for task: {task_name}")
        
        losses['total'] = total_loss
        return losses
    
    def compute_loss(self, predictions: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                    targets: Union[torch.Tensor, Dict[str, torch.Tensor]], **kwargs) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute loss between predictions and targets"""
        if isinstance(predictions, dict) and isinstance(targets, dict):
            return self.forward(predictions, targets, **kwargs)
        else:
            raise ValueError("MultiTaskLoss requires dictionary inputs for predictions and targets")
    
    def get_loss_type(self) -> str:
        """Return the type identifier of this loss function"""
        return "multi_task"
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get loss function capabilities"""
        return {
            'type': 'multi_task',
            'task_types': ['multi_task'],
            'tasks': list(self.task_losses.keys()),
            'task_weights': self.task_weights,
            'reduction': self.config.reduction,
            'differentiable': True
        }


class FocalLoss(BaseLoss):
    """
    Focal loss for addressing class imbalance in classification
    """
    
    def __init__(self, config: LossConfig, alpha: float = 1.0, gamma: float = 2.0):
        from dataclasses import dataclass
        @dataclass
        class MockConfig:
            pass
        
        super().__init__(MockConfig())
        
        self.config = config
        self.alpha = alpha
        self.gamma = gamma
        
        logger.info(f"FocalLoss initialized with alpha={alpha}, gamma={gamma}")
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute focal loss
        
        Args:
            predictions: Model logits [batch_size, num_classes]
            targets: Ground truth class indices [batch_size]
            
        Returns:
            Focal loss tensor
        """
        # Compute cross-entropy
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        
        # Compute probabilities
        probs = F.softmax(predictions, dim=1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal weight
        focal_weight = (1 - target_probs) ** self.gamma
        
        # Compute focal loss
        focal_loss = self.alpha * focal_weight * ce_loss
        
        if self.config.reduction == 'mean':
            return focal_loss.mean()
        elif self.config.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute loss between predictions and targets"""
        return self.forward(predictions, targets, **kwargs)
    
    def get_loss_type(self) -> str:
        """Return the type identifier of this loss function"""
        return "focal"
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get loss function capabilities"""
        return {
            'type': 'focal',
            'task_types': ['classification'],
            'alpha': self.alpha,
            'gamma': self.gamma,
            'handles_class_imbalance': True,
            'reduction': self.config.reduction,
            'differentiable': True
        }