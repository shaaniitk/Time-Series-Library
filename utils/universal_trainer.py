#!/usr/bin/env python3
"""
Universal Enhanced Training Framework

This module provides a model-agnostic training framework that can apply
all the enhancements (curriculum learning, adaptive losses, Bayesian training, etc.)
to ANY time series model, not just Autoformer variants.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable
from utils.enhanced_losses import CurriculumLossScheduler, AdaptiveAutoformerLoss
from utils.bayesian_losses import BayesianLoss
from utils.logger import logger


class UniversalEnhancedTrainer:
    """
    Universal trainer that can enhance ANY time series model with:
    - Curriculum learning
    - Adaptive losses
    - Bayesian uncertainty
    - Dynamic configuration
    - Advanced metrics
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: Dict[str, Any],
                 enhancements: Optional[Dict[str, bool]] = None):
        """
        Initialize universal trainer.
        
        Args:
            model: ANY PyTorch model (Autoformer, TimesNet, Transformer, etc.)
            config: Training configuration dictionary
            enhancements: Dict of enhancement flags
        """
        self.model = model
        self.config = config
        self.device = self._setup_device()
        
        # Default enhancements
        self.enhancements = {
            'curriculum_learning': False,
            'adaptive_loss': False,
            'bayesian_training': False,
            'uncertainty_quantification': False,
            'frequency_aware_loss': False,
            'quantile_prediction': False,
            **((enhancements or {}))
        }
        
        # Setup enhancements
        self.loss_fn = self._setup_loss_function()
        self.curriculum = self._setup_curriculum() if self.enhancements['curriculum_learning'] else None
        self.uncertainty_tracker = [] if self.enhancements['uncertainty_quantification'] else None
        
        logger.info(f"Universal trainer initialized with enhancements: {self.enhancements}")
    
    def _setup_device(self):
        """Setup training device"""
        if torch.cuda.is_available() and self.config.get('use_gpu', False):
            device = torch.device(f"cuda:{self.config.get('gpu', 0)}")
        else:
            device = torch.device('cpu')
        return device
    
    def _setup_loss_function(self):
        """Setup enhanced loss function based on configuration"""
        base_loss = self.config.get('loss', 'mse')
        
        # Start with basic loss
        if self.enhancements['adaptive_loss']:
            loss_fn = AdaptiveAutoformerLoss(
                base_loss=base_loss,
                moving_avg=self.config.get('moving_avg', 25)
            )
        else:
            loss_fn = self._get_basic_loss(base_loss)
        
        # Wrap with Bayesian loss if needed
        if self.enhancements['bayesian_training']:
            loss_fn = BayesianLoss(
                base_loss_fn=loss_fn,
                kl_weight=self.config.get('kl_weight', 1e-5),
                uncertainty_weight=self.config.get('uncertainty_weight', 0.1)
            )
        
        return loss_fn
    
    def _setup_curriculum(self):
        """Setup curriculum learning scheduler"""
        return CurriculumLossScheduler(
            start_seq_len=self.config.get('curriculum_start_len', 24),
            target_seq_len=self.config.get('seq_len', 96),
            curriculum_epochs=self.config.get('curriculum_epochs', 50),
            loss_fn=self.loss_fn
        )
    
    def _get_basic_loss(self, loss_name: str):
        """Get basic loss function"""
        loss_map = {
            'mse': nn.MSELoss(),
            'mae': nn.L1Loss(),
            'huber': nn.HuberLoss(),
            'smooth_l1': nn.SmoothL1Loss()
        }
        return loss_map.get(loss_name.lower(), nn.MSELoss())
    
    def train_step(self, batch_data, epoch: int, optimizer: torch.optim.Optimizer):
        """
        Universal training step that works with any model.
        
        Args:
            batch_data: Batch data (format depends on model)
            epoch: Current epoch
            optimizer: Optimizer instance
            
        Returns:
            Dict with loss information
        """
        self.model.train()
        optimizer.zero_grad()
        
        # Apply curriculum learning if enabled
        if self.curriculum:
            batch_data = self._apply_curriculum(batch_data, epoch)
        
        # Forward pass - this is model-specific but standardized interface
        outputs = self._forward_pass(batch_data)
        targets = self._extract_targets(batch_data)
        
        # Compute loss with enhancements
        loss_result = self._compute_enhanced_loss(outputs, targets)
        
        # Backward pass
        loss_result['total_loss'].backward()
        
        # Gradient clipping if enabled
        if self.config.get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['grad_clip']
            )
        
        optimizer.step()
        
        # Track uncertainty if enabled
        if self.enhancements['uncertainty_quantification']:
            self._track_uncertainty(outputs, loss_result)
        
        return loss_result
    
    def _forward_pass(self, batch_data):
        """
        Model-agnostic forward pass.
        Override this method for different model types.
        """
        # Default implementation assumes standard interface
        # Models should implement: model(x, x_mark, dec_inp, y_mark)
        if len(batch_data) == 4:
            x, y, x_mark, y_mark = batch_data
            x, y = x.to(self.device), y.to(self.device)
            x_mark, y_mark = x_mark.to(self.device), y_mark.to(self.device)
            
            # Create decoder input (standard for sequence models)
            pred_len = self.config.get('pred_len', 24)
            label_len = self.config.get('label_len', 48)
            
            dec_inp = torch.zeros_like(y[:, -pred_len:, :]).float()
            dec_inp = torch.cat([y[:, :label_len, :], dec_inp], dim=1).to(self.device)
            
            return self.model(x, x_mark, dec_inp, y_mark)
        else:
            # Simpler interface for some models
            x, y = batch_data[0].to(self.device), batch_data[1].to(self.device)
            return self.model(x)
    
    def _extract_targets(self, batch_data):
        """Extract target values from batch data"""
        if len(batch_data) >= 2:
            y = batch_data[1].to(self.device)
            pred_len = self.config.get('pred_len', 24)
            features = self.config.get('features', 'MS')
            
            # Extract prediction targets
            f_dim = -1 if features == 'MS' else 0
            return y[:, -pred_len:, f_dim:]
        else:
            raise ValueError("Cannot extract targets from batch data")
    
    def _apply_curriculum(self, batch_data, epoch):
        """Apply curriculum learning to batch data"""
        if len(batch_data) == 4:
            x, y, x_mark, y_mark = batch_data
            return self.curriculum.apply_curriculum(x, y, x_mark, y_mark, epoch)
        else:
            # For simpler models, just return as-is
            return batch_data
    
    def _compute_enhanced_loss(self, outputs, targets):
        """Compute loss with all enhancements"""
        if self.enhancements['bayesian_training']:
            # Bayesian loss expects model reference
            return self.loss_fn(self.model, outputs, targets)
        else:
            # Standard loss computation
            if hasattr(self.loss_fn, 'forward'):
                loss = self.loss_fn(outputs, targets)
            else:
                loss = self.loss_fn(outputs, targets)
            
            return {'total_loss': loss}
    
    def _track_uncertainty(self, outputs, loss_result):
        """Track uncertainty metrics if applicable"""
        if isinstance(outputs, dict) and 'uncertainty' in outputs:
            uncertainty = outputs['uncertainty'].detach().cpu().numpy().mean()
            self.uncertainty_tracker.append(uncertainty)


class ModelAdapter:
    """
    Adapter pattern to make any model compatible with the universal trainer.
    """
    
    @staticmethod
    def adapt_timesnet(model):
        """Adapt TimesNet for universal trainer"""
        original_forward = model.forward
        
        def enhanced_forward(x, x_mark, dec_inp, y_mark):
            # TimesNet-specific adaptation
            return original_forward(x, x_mark, dec_inp, y_mark)
        
        model.forward = enhanced_forward
        return model
    
    @staticmethod
    def adapt_informer(model):
        """Adapt Informer for universal trainer"""
        # Similar adaptation for Informer
        return model
    
    @staticmethod
    def adapt_custom_model(model, forward_fn: Callable):
        """Adapt any custom model with provided forward function"""
        model.forward = forward_fn
        return model


def create_universal_trainer(model_name: str, config: Dict[str, Any], 
                           enhancements: Optional[Dict[str, bool]] = None):
    """
    Factory function to create a universal trainer for any model.
    
    Args:
        model_name: Name of the model ('EnhancedAutoformer', 'TimesNet', etc.)
        config: Configuration dictionary
        enhancements: Enhancement flags
        
    Returns:
        Configured UniversalEnhancedTrainer instance
    """
    # Import model dynamically
    if model_name == 'EnhancedAutoformer':
        from models.EnhancedAutoformer import Model as EnhancedAutoformer
        model = EnhancedAutoformer(config)
    elif model_name == 'TimesNet':
        from models.TimesNet import Model as TimesNet
        model = TimesNet(config)
    elif model_name == 'Informer':
        from models.Informer import Model as Informer
        model = Informer(config)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Apply adapter if needed
    if model_name == 'TimesNet':
        model = ModelAdapter.adapt_timesnet(model)
    elif model_name == 'Informer':
        model = ModelAdapter.adapt_informer(model)
    
    return UniversalEnhancedTrainer(model, config, enhancements)


# Example usage:
if __name__ == "__main__":
    # Example: Train TimesNet with all enhancements
    config = {
        'seq_len': 96,
        'pred_len': 24,
        'features': 'MS',
        'curriculum_learning': True,
        'adaptive_loss': True,
        'bayesian_training': False
    }
    
    enhancements = {
        'curriculum_learning': True,
        'adaptive_loss': True,
        'uncertainty_quantification': True
    }
    
    trainer = create_universal_trainer('TimesNet', config, enhancements)
    # Now TimesNet can use curriculum learning, adaptive losses, etc.!
