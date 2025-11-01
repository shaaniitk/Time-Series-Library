#!/usr/bin/env python3
"""
Modular Loss Handler - Clean separation of loss computation logic
Integrates with existing modular loss components to replace fallback mechanisms
"""

from typing import Optional, Tuple, Dict, Any, Union
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

# Import from existing modular components
from ..losses.directional_trend_loss import HybridMDNDirectionalLoss, DirectionalTrendLoss
from ..decoder.mdn_decoder import mdn_nll_loss
from ..decoder.mixture_density_decoder import MixtureNLLLoss
from ..decoder.sequential_mixture_decoder import SequentialMixtureNLLLoss


class BaseLossHandler(ABC):
    """Base class for loss handlers - enforces clean interface"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.loss_fn = self._create_loss_function()
    
    @abstractmethod
    def _create_loss_function(self) -> nn.Module:
        """Create the appropriate loss function based on config"""
        pass
    
    @abstractmethod
    def compute_loss(self, 
                    model_outputs: Any, 
                    targets: torch.Tensor,
                    **kwargs) -> torch.Tensor:
        """Compute loss with proper input validation and format conversion"""
        pass
    
    @abstractmethod
    def validate_compatibility(self, model_config: Dict[str, Any]) -> None:
        """Validate that this loss is compatible with the model configuration"""
        pass


class HybridMDNDirectionalLossHandler(BaseLossHandler):
    """Handler for Hybrid MDN + Directional Loss"""
    
    def _create_loss_function(self) -> HybridMDNDirectionalLoss:
        return HybridMDNDirectionalLoss(
            nll_weight=self.config.get('nll_weight', 0.15),
            direction_weight=self.config.get('direction_weight', 8.0),
            trend_weight=self.config.get('trend_weight', 0.5),
            magnitude_weight=self.config.get('magnitude_weight', 0.1)
        )
    
    def compute_loss(self, 
                    model_outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                    targets: torch.Tensor,
                    **kwargs) -> torch.Tensor:
        """
        Compute hybrid loss from MDN outputs
        
        Args:
            model_outputs: (pi, mu, sigma) from MDN decoder
            targets: Ground truth targets
            
        Returns:
            Scalar loss tensor
        """
        pi, mu, sigma = model_outputs
        
        # Convert MDN format (pi, mu, sigma) to hybrid loss format (mu, log_sigma, log_pi)
        log_stds = torch.log(sigma.clamp(min=1e-6))
        log_weights = torch.log(pi.clamp(min=1e-8))
        mdn_params_hybrid = (mu, log_stds, log_weights)
        
        return self.loss_fn(mdn_params_hybrid, targets)
    
    def validate_compatibility(self, model_config: Dict[str, Any]) -> None:
        """Validate compatibility with model configuration"""
        if not model_config.get('enable_mdn_decoder', False):
            raise ValueError(
                "HybridMDNDirectionalLoss requires enable_mdn_decoder=True. "
                "Current config has enable_mdn_decoder=False."
            )
        
        if not model_config.get('use_mixture_decoder', False):
            raise ValueError(
                "HybridMDNDirectionalLoss requires use_mixture_decoder=True. "
                "Current config has use_mixture_decoder=False."
            )


class DirectionalTrendLossHandler(BaseLossHandler):
    """Handler for pure Directional Trend Loss (no MDN)"""
    
    def _create_loss_function(self) -> DirectionalTrendLoss:
        return DirectionalTrendLoss(
            direction_weight=self.config.get('direction_weight', 5.0),
            trend_weight=self.config.get('trend_weight', 2.0),
            magnitude_weight=self.config.get('magnitude_weight', 0.1),
            use_mdn_mean=False  # Pure directional, no MDN
        )
    
    def compute_loss(self, 
                    model_outputs: torch.Tensor, 
                    targets: torch.Tensor,
                    **kwargs) -> torch.Tensor:
        """
        Compute directional loss from deterministic outputs
        
        Args:
            model_outputs: Direct model predictions [batch, seq, features]
            targets: Ground truth targets
            
        Returns:
            Scalar loss tensor
        """
        return self.loss_fn(model_outputs, targets)
    
    def validate_compatibility(self, model_config: Dict[str, Any]) -> None:
        """Validate compatibility with model configuration"""
        if model_config.get('enable_mdn_decoder', False):
            raise ValueError(
                "DirectionalTrendLoss (pure) is incompatible with enable_mdn_decoder=True. "
                "Use HybridMDNDirectionalLoss instead or disable MDN decoder."
            )


class MDNNLLLossHandler(BaseLossHandler):
    """Handler for pure MDN NLL Loss"""
    
    def _create_loss_function(self) -> None:
        # MDN NLL uses functional approach, no class needed
        return None
    
    def compute_loss(self, 
                    model_outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                    targets: torch.Tensor,
                    **kwargs) -> torch.Tensor:
        """
        Compute MDN NLL loss
        
        Args:
            model_outputs: (pi, mu, sigma) from MDN decoder
            targets: Ground truth targets
            
        Returns:
            Scalar loss tensor
        """
        pi, mu, sigma = model_outputs
        return mdn_nll_loss(pi, mu, sigma, targets, reduce='mean')
    
    def validate_compatibility(self, model_config: Dict[str, Any]) -> None:
        """Validate compatibility with model configuration"""
        if not model_config.get('enable_mdn_decoder', False):
            raise ValueError(
                "MDNNLLLoss requires enable_mdn_decoder=True. "
                "Current config has enable_mdn_decoder=False."
            )


class MixtureNLLLossHandler(BaseLossHandler):
    """Handler for Mixture NLL Loss (sequential mixture decoder)"""
    
    def _create_loss_function(self) -> Union[MixtureNLLLoss, SequentialMixtureNLLLoss]:
        if self.config.get('use_sequential_mixture', False):
            return SequentialMixtureNLLLoss(
                eps=self.config.get('eps', 1e-8),
                reduction=self.config.get('reduction', 'mean')
            )
        else:
            return MixtureNLLLoss(
                eps=self.config.get('eps', 1e-8),
                multivariate_mode=self.config.get('multivariate_mode', 'independent')
            )
    
    def compute_loss(self, 
                    model_outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                    targets: torch.Tensor,
                    **kwargs) -> torch.Tensor:
        """
        Compute mixture NLL loss
        
        Args:
            model_outputs: (means, log_stds, log_weights) from mixture decoder
            targets: Ground truth targets
            
        Returns:
            Scalar loss tensor
        """
        return self.loss_fn(model_outputs, targets)
    
    def validate_compatibility(self, model_config: Dict[str, Any]) -> None:
        """Validate compatibility with model configuration"""
        use_mixture = model_config.get('use_mixture_decoder', False)
        use_sequential = model_config.get('use_sequential_mixture_decoder', False)
        
        if not (use_mixture or use_sequential):
            raise ValueError(
                "MixtureNLLLoss requires use_mixture_decoder=True or "
                "use_sequential_mixture_decoder=True. Both are currently False."
            )


class MSELossHandler(BaseLossHandler):
    """Handler for standard MSE Loss"""
    
    def _create_loss_function(self) -> nn.MSELoss:
        return nn.MSELoss()
    
    def compute_loss(self, 
                    model_outputs: torch.Tensor, 
                    targets: torch.Tensor,
                    **kwargs) -> torch.Tensor:
        """
        Compute MSE loss
        
        Args:
            model_outputs: Direct model predictions [batch, seq, features]
            targets: Ground truth targets
            
        Returns:
            Scalar loss tensor
        """
        return self.loss_fn(model_outputs, targets)
    
    def validate_compatibility(self, model_config: Dict[str, Any]) -> None:
        """MSE is compatible with any deterministic model"""
        pass  # MSE is always compatible


class LossHandlerFactory:
    """Factory for creating appropriate loss handlers based on configuration"""
    
    _handlers = {
        'hybrid_mdn_directional': HybridMDNDirectionalLossHandler,
        'directional_trend': DirectionalTrendLossHandler,
        'mdn_nll': MDNNLLLossHandler,
        'mixture_nll': MixtureNLLLossHandler,
        'sequential_mixture_nll': MixtureNLLLossHandler,
        'mse': MSELossHandler,
    }
    
    @classmethod
    def create_handler(cls, loss_config: Dict[str, Any], model_config: Dict[str, Any]) -> BaseLossHandler:
        """
        Create appropriate loss handler based on configuration
        
        Args:
            loss_config: Loss configuration dictionary
            model_config: Model configuration dictionary
            
        Returns:
            Configured loss handler
            
        Raises:
            ValueError: If loss type is unsupported or incompatible
        """
        loss_type = loss_config.get('type', 'auto')
        
        # Auto-detect loss type based on model configuration
        if loss_type == 'auto':
            loss_type = cls._auto_detect_loss_type(model_config)
        
        if loss_type not in cls._handlers:
            available_types = list(cls._handlers.keys())
            raise ValueError(
                f"Unsupported loss type: '{loss_type}'. "
                f"Available types: {available_types}"
            )
        
        # Create handler
        handler_class = cls._handlers[loss_type]
        handler = handler_class(loss_config)
        
        # Validate compatibility
        try:
            handler.validate_compatibility(model_config)
        except ValueError as e:
            raise ValueError(
                f"Loss type '{loss_type}' is incompatible with current model configuration. "
                f"Error: {str(e)}"
            ) from e
        
        return handler
    
    @classmethod
    def _auto_detect_loss_type(cls, model_config: Dict[str, Any]) -> str:
        """Auto-detect appropriate loss type based on model configuration"""
        
        enable_mdn = model_config.get('enable_mdn_decoder', False)
        use_mixture = model_config.get('use_mixture_decoder', False)
        use_sequential = model_config.get('use_sequential_mixture_decoder', False)
        
        if enable_mdn and use_mixture:
            # MDN + mixture decoder -> default to hybrid directional
            return 'hybrid_mdn_directional'
        elif enable_mdn:
            # Pure MDN decoder
            return 'mdn_nll'
        elif use_mixture or use_sequential:
            # Mixture decoder without MDN
            return 'mixture_nll'
        else:
            # Deterministic model
            return 'mse'
    
    @classmethod
    def list_available_handlers(cls) -> Dict[str, str]:
        """List all available loss handlers with descriptions"""
        return {
            'hybrid_mdn_directional': 'Hybrid MDN + Directional Loss (for probabilistic + directional forecasting)',
            'directional_trend': 'Pure Directional Trend Loss (for deterministic directional forecasting)',
            'mdn_nll': 'Pure MDN NLL Loss (for probabilistic forecasting)',
            'mixture_nll': 'Mixture NLL Loss (for mixture density models)',
            'mse': 'Mean Squared Error Loss (for deterministic forecasting)',
        }


def create_loss_handler(loss_config: Dict[str, Any], model_config: Dict[str, Any]) -> BaseLossHandler:
    """
    Convenience function to create loss handler
    
    Args:
        loss_config: Loss configuration
        model_config: Model configuration
        
    Returns:
        Configured loss handler
    """
    return LossHandlerFactory.create_handler(loss_config, model_config)