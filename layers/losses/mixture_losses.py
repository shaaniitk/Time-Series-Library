"""
Mixture density losses and loss configuration for Enhanced SOTA PGAT
Contains MDN loss handling and loss function configuration
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Any


class LossConfigurator:
    """Configures appropriate loss functions for the model"""
    
    @staticmethod
    def configure_optimizer_loss(decoder, mixture_loss: Optional[nn.Module], 
                               base_criterion: nn.Module, verbose: bool = False):
        """Configure the appropriate loss function for the model."""
        # Check if decoder is MixtureDensityDecoder (duck typing)
        is_mixture_decoder = hasattr(decoder, 'num_components') and hasattr(decoder, 'num_targets')
        
        if is_mixture_decoder:
            if mixture_loss is None:
                raise RuntimeError("Mixture loss requested but not initialized")
            if verbose:
                print("Enhanced PGAT using MixtureNLLLoss for MDN outputs")
            return mixture_loss
        else:
            if verbose:
                print("Enhanced PGAT using standard loss function")
            return base_criterion
    
    @staticmethod
    def compute_loss(forward_output: Any, targets: torch.Tensor, 
                    decoder, mixture_loss: Optional[nn.Module]) -> torch.Tensor:
        """Compute loss based on decoder type"""
        # Check if decoder is MixtureDensityDecoder (duck typing)
        is_mixture_decoder = hasattr(decoder, 'num_components') and hasattr(decoder, 'num_targets')
        
        if is_mixture_decoder:
            # Fixed: Use correct MDN loss computation
            if mixture_loss is None:
                raise RuntimeError("Mixture loss requested but not initialized")
            return mixture_loss(forward_output, targets)
        else:
            # Standard MSE loss for the base decoder
            loss_fn = nn.MSELoss()
            return loss_fn(forward_output, targets)


class MixtureLossWrapper:
    """Wrapper for mixture density loss functionality"""
    
    def __init__(self, multivariate_mode: str = 'independent'):
        self.multivariate_mode = multivariate_mode
        self.mixture_loss = None
        
    def initialize_mixture_loss(self):
        """Initialize mixture loss with multivariate support"""
        try:
            # This would import the actual MixtureNLLLoss class
            # For now, we'll create a placeholder
            self.mixture_loss = self._create_mixture_loss_placeholder()
        except ImportError:
            print("Warning: MixtureNLLLoss not available, using MSE fallback")
            self.mixture_loss = None
    
    def _create_mixture_loss_placeholder(self):
        """Placeholder for MixtureNLLLoss creation"""
        # This would be replaced with actual MixtureNLLLoss import and initialization
        class PlaceholderMixtureLoss(nn.Module):
            def __init__(self, multivariate_mode):
                super().__init__()
                self.multivariate_mode = multivariate_mode
            
            def forward(self, predictions, targets):
                # Placeholder implementation
                if isinstance(predictions, tuple) and len(predictions) == 3:
                    means, log_stds, log_weights = predictions
                    # Simple MSE fallback for now
                    return nn.MSELoss()(means.mean(dim=-1), targets)
                else:
                    return nn.MSELoss()(predictions, targets)
        
        return PlaceholderMixtureLoss(self.multivariate_mode)
    
    def get_loss(self):
        """Get the mixture loss instance"""
        return self.mixture_loss


class DecoderManager:
    """Manages decoder initialization and configuration"""
    
    @staticmethod
    def create_decoder(config, d_model: int):
        """Create appropriate decoder based on configuration"""
        use_mixture_decoder = getattr(config, 'use_mixture_decoder', True)
        
        if use_mixture_decoder:
            try:
                # This would import the actual MixtureDensityDecoder
                # For now, we'll create a placeholder
                return DecoderManager._create_mixture_decoder_placeholder(config, d_model)
            except ImportError:
                print("Warning: MixtureDensityDecoder not available, using linear decoder")
                return nn.Linear(d_model, getattr(config, 'c_out', 3))
        else:
            return nn.Linear(d_model, getattr(config, 'c_out', 3))
    
    @staticmethod
    def _create_mixture_decoder_placeholder(config, d_model: int):
        """Placeholder for MixtureDensityDecoder creation"""
        # This would be replaced with actual MixtureDensityDecoder import
        class PlaceholderMixtureDensityDecoder(nn.Module):
            def __init__(self, d_model, pred_len, num_components, num_targets):
                super().__init__()
                self.d_model = d_model
                self.pred_len = pred_len
                self.num_components = num_components
                self.num_targets = num_targets
                
                # Simple linear layers as placeholder
                self.means_layer = nn.Linear(d_model, pred_len * num_targets * num_components)
                self.stds_layer = nn.Linear(d_model, pred_len * num_targets * num_components)
                self.weights_layer = nn.Linear(d_model, pred_len * num_components)
            
            def forward(self, x):
                batch_size = x.size(0)
                
                means = self.means_layer(x).view(batch_size, self.pred_len, self.num_targets, self.num_components)
                log_stds = self.stds_layer(x).view(batch_size, self.pred_len, self.num_targets, self.num_components)
                log_weights = self.weights_layer(x).view(batch_size, self.pred_len, self.num_components)
                
                return means, log_stds, log_weights
        
        return PlaceholderMixtureDensityDecoder(
            d_model=d_model,
            pred_len=getattr(config, 'pred_len', 24),
            num_components=getattr(config, 'mdn_components', 3),
            num_targets=getattr(config, 'c_out', 3)
        )


class RegularizationManager:
    """Manages regularization losses"""
    
    @staticmethod
    def get_regularization_loss(model) -> torch.Tensor:
        """Get regularization loss from model components"""
        loss = 0
        if hasattr(model, 'latest_stochastic_loss'):
            loss += model.latest_stochastic_loss
        return loss