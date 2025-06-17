"""
Enhanced Loss Functions for Autoformer

This module provides adaptive and component-aware loss functions
that can improve Autoformer training by separately handling trend
and seasonal components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from utils.logger import logger


class AdaptiveAutoformerLoss(nn.Module):
    """
    Adaptive loss function that dynamically weights trend and seasonal components.
    
    Features:
    - Learnable trend/seasonal weights
    - Multiple base loss options
    - Component-wise loss tracking
    - Adaptive weighting based on data characteristics
    """
    
    def __init__(self, base_loss='mse', moving_avg=25, initial_trend_weight=1.0, 
                 initial_seasonal_weight=1.0, adaptive_weights=True):
        super(AdaptiveAutoformerLoss, self).__init__()
        logger.info(f"Initializing AdaptiveAutoformerLoss with base_loss={base_loss}")
        
        self.base_loss = base_loss
        self.adaptive_weights = adaptive_weights
        
        # Decomposition for loss calculation
        self.decomp = series_decomp(kernel_size=moving_avg)
        
        # Learnable weight parameters
        if adaptive_weights:
            self.trend_weight = nn.Parameter(torch.tensor(initial_trend_weight))
            self.seasonal_weight = nn.Parameter(torch.tensor(initial_seasonal_weight))
        else:
            self.register_buffer('trend_weight', torch.tensor(initial_trend_weight))
            self.register_buffer('seasonal_weight', torch.tensor(initial_seasonal_weight))
        
        # Base loss function
        self.loss_fn = self._get_loss_function(base_loss)
        
    def _get_loss_function(self, loss_name):
        """Get the base loss function"""
        if loss_name == 'mse':
            return F.mse_loss
        elif loss_name == 'mae':
            return F.l1_loss
        elif loss_name == 'huber':
            return F.huber_loss
        elif loss_name == 'smooth_l1':
            return F.smooth_l1_loss
        else:
            logger.warning(f"Unknown loss {loss_name}, defaulting to MSE")
            return F.mse_loss
    
    def forward(self, pred, true, return_components=False):
        """
        Compute adaptive loss with trend/seasonal decomposition.
        
        Args:
            pred: [B, L, D] predicted values
            true: [B, L, D] ground truth values
            return_components: whether to return component losses
            
        Returns:
            total_loss: scalar loss value
            components: dict with component losses (if return_components=True)
        """
        # Decompose both predictions and ground truth
        pred_seasonal, pred_trend = self.decomp(pred)
        true_seasonal, true_trend = self.decomp(true)
        
        # Compute component-wise losses
        trend_loss = self.loss_fn(pred_trend, true_trend, reduction='mean')
        seasonal_loss = self.loss_fn(pred_seasonal, true_seasonal, reduction='mean')
        
        # Apply adaptive weighting with softplus for positivity
        if self.adaptive_weights:
            trend_w = F.softplus(self.trend_weight)
            seasonal_w = F.softplus(self.seasonal_weight)
        else:
            trend_w = self.trend_weight
            seasonal_w = self.seasonal_weight
        
        # Total adaptive loss
        total_loss = trend_w * trend_loss + seasonal_w * seasonal_loss
        
        if return_components:
            components = {
                'total_loss': total_loss.item(),
                'trend_loss': trend_loss.item(),
                'seasonal_loss': seasonal_loss.item(),
                'trend_weight': trend_w.item(),
                'seasonal_weight': seasonal_w.item(),
                'trend_contribution': (trend_w * trend_loss).item(),
                'seasonal_contribution': (seasonal_w * seasonal_loss).item()
            }
            return total_loss, components
        else:
            return total_loss


class FrequencyAwareLoss(nn.Module):
    """
    Loss function that considers frequency domain characteristics.
    Useful for time series with strong periodic patterns.
    """
    
    def __init__(self, freq_weight=0.1, base_loss='mse'):
        super(FrequencyAwareLoss, self).__init__()
        logger.info("Initializing FrequencyAwareLoss")
        
        self.freq_weight = freq_weight
        self.base_loss_fn = self._get_loss_function(base_loss)
        
    def _get_loss_function(self, loss_name):
        """Get the base loss function"""
        if loss_name == 'mse':
            return F.mse_loss
        elif loss_name == 'mae':
            return F.l1_loss
        else:
            return F.mse_loss
    
    def forward(self, pred, true):
        """
        Compute frequency-aware loss.
        
        Args:
            pred: [B, L, D] predicted values
            true: [B, L, D] ground truth values
            
        Returns:
            total_loss: scalar loss value
        """
        # Time domain loss
        time_loss = self.base_loss_fn(pred, true, reduction='mean')
        
        # Frequency domain loss
        pred_fft = torch.fft.fft(pred, dim=1)
        true_fft = torch.fft.fft(true, dim=1)
        
        # Focus on magnitude spectrum
        pred_mag = torch.abs(pred_fft)
        true_mag = torch.abs(true_fft)
        
        freq_loss = F.mse_loss(pred_mag, true_mag, reduction='mean')
        
        # Combined loss
        total_loss = time_loss + self.freq_weight * freq_loss
        
        return total_loss


class QuantileLoss(nn.Module):
    """
    Quantile loss for uncertainty quantification in forecasting.
    """
    
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super(QuantileLoss, self).__init__()
        logger.info(f"Initializing QuantileLoss with quantiles={quantiles}")
        self.quantiles = quantiles
        
    def forward(self, pred, true):
        """
        Compute quantile loss.
        
        Args:
            pred: [B, L, D*len(quantiles)] predicted quantiles
            true: [B, L, D] ground truth values
            
        Returns:
            total_loss: scalar loss value
        """
        B, L, D = true.shape
        n_quantiles = len(self.quantiles)
        
        # Reshape predictions to separate quantiles
        pred = pred.view(B, L, D, n_quantiles)
        true = true.unsqueeze(-1).expand(-1, -1, -1, n_quantiles)
        
        losses = []
        for i, q in enumerate(self.quantiles):
            error = true[..., i] - pred[..., i]
            loss = torch.max(q * error, (q - 1) * error)
            losses.append(loss.mean())
        
        return sum(losses) / len(losses)


class CurriculumLossScheduler:
    """
    Curriculum learning scheduler that gradually increases sequence length
    and adjusts loss weighting during training.
    """
    
    def __init__(self, start_seq_len=24, target_seq_len=96, curriculum_epochs=50,
                 loss_fn=None, schedule_type='linear'):
        logger.info(f"Initializing CurriculumLossScheduler: {start_seq_len}→{target_seq_len} over {curriculum_epochs} epochs")
        
        self.start_seq_len = start_seq_len
        self.target_seq_len = target_seq_len
        self.curriculum_epochs = curriculum_epochs
        self.schedule_type = schedule_type
        self.loss_fn = loss_fn or AdaptiveAutoformerLoss()
        
    def get_current_seq_len(self, epoch):
        """Get current sequence length based on curriculum schedule"""
        if epoch >= self.curriculum_epochs:
            return self.target_seq_len
            
        progress = epoch / self.curriculum_epochs
        
        if self.schedule_type == 'linear':
            current_len = self.start_seq_len + progress * (self.target_seq_len - self.start_seq_len)
        elif self.schedule_type == 'exponential':
            # Exponential growth
            ratio = self.target_seq_len / self.start_seq_len
            current_len = self.start_seq_len * (ratio ** progress)
        elif self.schedule_type == 'cosine':
            # Cosine annealing
            current_len = self.start_seq_len + 0.5 * (self.target_seq_len - self.start_seq_len) * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        else:
            current_len = self.target_seq_len
        
        return max(self.start_seq_len, min(int(current_len), self.target_seq_len))
    
    def apply_curriculum(self, batch_x, batch_y, batch_x_mark, batch_y_mark, epoch):
        """
        Apply curriculum learning to batch data.
        
        Args:
            batch_x, batch_y, batch_x_mark, batch_y_mark: input tensors
            epoch: current training epoch
            
        Returns:
            truncated tensors based on curriculum schedule
        """
        current_seq_len = self.get_current_seq_len(epoch)
        
        if current_seq_len < batch_x.size(1):
            # Truncate sequences from the end (keep most recent data)
            batch_x = batch_x[:, -current_seq_len:, :]
            batch_x_mark = batch_x_mark[:, -current_seq_len:, :]
            
            # Adjust decoder inputs accordingly
            # This might need adjustment based on your specific data format
            
        return batch_x, batch_y, batch_x_mark, batch_y_mark
    
    def compute_loss(self, pred, true, epoch, return_components=False):
        """Compute loss with curriculum considerations"""
        # You could adjust loss weighting based on curriculum progress
        progress = min(epoch / self.curriculum_epochs, 1.0)
        
        # Example: gradually increase complexity penalty
        if hasattr(self.loss_fn, 'complexity_weight'):
            self.loss_fn.complexity_weight = 0.1 * progress
            
        return self.loss_fn(pred, true, return_components=return_components)


class ModeAwareLoss(nn.Module):
    """
    Mode-aware loss function that handles M, MS, and S forecasting modes.
    
    - M mode: All features → All features  
    - MS mode: All features → Target features only
    - S mode: Target features → Target features only
    """
    
    def __init__(self, mode='MS', target_features=4, total_features=118, 
                 loss_type='mse', target_weights=None):
        """
        Args:
            mode: Forecasting mode ('M', 'MS', 'S')
            target_features: Number of target features (4 for OHLC)
            total_features: Total number of features (118)
            loss_type: Base loss function ('mse', 'mae', 'huber')
            target_weights: Optional weights for target features
        """
        super().__init__()
        
        self.mode = mode.upper()
        self.target_features = target_features
        self.total_features = total_features
        self.loss_type = loss_type
        
        # Initialize base loss function
        if loss_type == 'mse':
            self.base_loss = nn.MSELoss(reduction='none')
        elif loss_type == 'mae':
            self.base_loss = nn.L1Loss(reduction='none')
        elif loss_type == 'huber':
            self.base_loss = nn.HuberLoss(reduction='none', delta=1.0)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        # Target feature weights (e.g., emphasize Close price)
        if target_weights is not None:
            self.register_buffer('target_weights', torch.tensor(target_weights, dtype=torch.float32))
        else:
            # Default equal weights for all target features
            self.register_buffer('target_weights', torch.ones(target_features, dtype=torch.float32))
        
        logger.info(f"Initialized ModeAwareLoss - Mode: {self.mode}, Loss: {loss_type}")
        
    def forward(self, predictions, targets, return_components=False):
        """
        Compute loss based on forecasting mode
        
        Args:
            predictions: Model predictions [B, L, D]
            targets: Ground truth targets [B, L, D]
            return_components: Whether to return loss components
            
        Returns:
            loss: Computed loss value
            components: Loss components dict (if return_components=True)
        """
        
        if self.mode == 'M':
            # M mode: Use all features
            pred_slice = predictions
            target_slice = targets
            
        elif self.mode == 'MS':
            # MS mode: Use only target features (first target_features columns)
            pred_slice = predictions[:, :, :self.target_features]
            target_slice = targets[:, :, :self.target_features]
            
        elif self.mode == 'S':
            # S mode: All features are targets (input already filtered)
            pred_slice = predictions
            target_slice = targets
            
        else:
            raise ValueError(f"Unknown forecasting mode: {self.mode}")
        
        # Compute base loss
        loss_matrix = self.base_loss(pred_slice, target_slice)  # [B, L, D]
        
        # Apply feature weights if in target mode
        if self.mode in ['MS', 'S'] and len(loss_matrix.shape) == 3:
            # Expand weights to match batch and sequence dimensions
            weights = self.target_weights.view(1, 1, -1).expand_as(loss_matrix)
            loss_matrix = loss_matrix * weights
        
        # Compute mean loss
        total_loss = loss_matrix.mean()
        
        if return_components:
            components = {
                'total_loss': total_loss.item(),
                'mode': self.mode,
                'target_features': self.target_features,
                'prediction_shape': pred_slice.shape,
                'target_shape': target_slice.shape
            }
            return total_loss, components
        
        return total_loss
    
    def get_mode_info(self):
        """Return information about current mode"""
        return {
            'mode': self.mode,
            'target_features': self.target_features,
            'total_features': self.total_features,
            'loss_type': self.loss_type,
            'target_weights': self.target_weights.tolist() if self.target_weights is not None else None
        }


class HierarchicalModeAwareLoss(nn.Module):
    """
    Hierarchical loss for HierarchicalEnhancedAutoformer with mode awareness
    Combines losses from different resolution levels
    """
    
    def __init__(self, mode='MS', target_features=4, 
                 resolution_weights=None, base_loss_type='mse'):
        """
        Args:
            mode: Forecasting mode ('M', 'MS', 'S')
            target_features: Number of target features
            resolution_weights: Weights for different resolution levels
            base_loss_type: Base loss function type
        """
        super().__init__()
        
        self.mode = mode.upper()
        self.target_features = target_features
        self.base_loss_type = base_loss_type
        
        # Initialize base loss for each resolution
        self.mode_aware_loss = ModeAwareLoss(
            mode=mode, 
            target_features=target_features,
            loss_type=base_loss_type
        )
        
        # Resolution weights (if not provided, use decreasing weights)
        if resolution_weights is not None:
            self.register_buffer('resolution_weights', torch.tensor(resolution_weights, dtype=torch.float32))
        else:
            # Default weights: higher weight for original resolution
            self.register_buffer('resolution_weights', torch.tensor([1.0, 0.5, 0.25], dtype=torch.float32))
        
        logger.info(f"Initialized HierarchicalModeAwareLoss - Mode: {self.mode}")
    
    def forward(self, predictions, targets, resolution_outputs=None, return_components=False):
        """
        Compute hierarchical loss with mode awareness
        
        Args:
            predictions: Final model predictions [B, L, D]
            targets: Ground truth targets [B, L, D]
            resolution_outputs: Optional list of outputs from different resolutions
            return_components: Whether to return loss components
            
        Returns:
            total_loss: Weighted combination of losses
            components: Loss components dict (if return_components=True)
        """
        
        # Primary loss on final predictions
        primary_loss = self.mode_aware_loss(predictions, targets)
        total_loss = primary_loss
        
        components = {
            'primary_loss': primary_loss.item(),
            'resolution_losses': [],
            'mode': self.mode
        }
        
        # If no hierarchical outputs provided, return primary loss
        if resolution_outputs is None or len(resolution_outputs) == 0:
            if return_components:
                return total_loss, components
            return total_loss
        
        # Compute losses for each resolution level
        resolution_losses = []
        for i, res_output in enumerate(resolution_outputs):
            if res_output is not None and res_output.shape == predictions.shape:
                res_loss = self.mode_aware_loss(res_output, targets)
                resolution_losses.append(res_loss)
                components['resolution_losses'].append(res_loss.item())
            else:
                # If resolution output doesn't match, skip or use zero loss
                zero_loss = torch.zeros_like(primary_loss)
                resolution_losses.append(zero_loss)
                components['resolution_losses'].append(0.0)
        
        # Weighted combination
        for i, res_loss in enumerate(resolution_losses):
            weight_idx = min(i, len(self.resolution_weights) - 1)
            total_loss += self.resolution_weights[weight_idx] * res_loss
        
        components['total_loss'] = total_loss.item()
        
        if return_components:
            return total_loss, components
        return total_loss


class BayesianModeAwareLoss(nn.Module):
    """
    Bayesian loss for BayesianEnhancedAutoformer with mode awareness
    Combines prediction loss with uncertainty regularization
    """
    
    def __init__(self, mode='MS', target_features=4, 
                 uncertainty_weight=0.1, kl_weight=0.01, base_loss_type='mse'):
        """
        Args:
            mode: Forecasting mode ('M', 'MS', 'S')
            target_features: Number of target features
            uncertainty_weight: Weight for uncertainty loss
            kl_weight: Weight for KL divergence regularization
            base_loss_type: Base loss function type
        """
        super().__init__()
        
        self.mode = mode.upper()
        self.target_features = target_features
        self.uncertainty_weight = uncertainty_weight
        self.kl_weight = kl_weight
        
        # Base prediction loss
        self.mode_aware_loss = ModeAwareLoss(
            mode=mode,
            target_features=target_features,
            loss_type=base_loss_type
        )
        
        logger.info(f"Initialized BayesianModeAwareLoss - Mode: {self.mode}, "
                   f"Uncertainty weight: {uncertainty_weight}, KL weight: {kl_weight}")
    
    def forward(self, predictions, targets, uncertainty=None, kl_divergence=None, return_components=False):
        """
        Compute Bayesian loss with mode awareness
        
        Args:
            predictions: Model predictions [B, L, D]
            targets: Ground truth targets [B, L, D] 
            uncertainty: Predicted uncertainty [B, L, D] (optional)
            kl_divergence: KL divergence from Bayesian layers (optional)
            return_components: Whether to return loss components
            
        Returns:
            total_loss: Combined prediction + uncertainty + KL loss
            components: Loss components dict (if return_components=True)
        """
        
        # Primary prediction loss
        prediction_loss = self.mode_aware_loss(predictions, targets)
        total_loss = prediction_loss
        
        components = {
            'prediction_loss': prediction_loss.item(),
            'uncertainty_loss': 0.0,
            'kl_loss': 0.0,
            'mode': self.mode
        }
        
        # Add uncertainty regularization if provided
        if uncertainty is not None:
            # Uncertainty should be positive and reasonable
            uncertainty_reg = torch.mean(torch.abs(uncertainty)) + \
                             0.1 * torch.mean(torch.square(uncertainty))
            total_loss += self.uncertainty_weight * uncertainty_reg
            components['uncertainty_loss'] = (self.uncertainty_weight * uncertainty_reg).item()
        
        # Add KL divergence regularization if provided
        if kl_divergence is not None:
            total_loss += self.kl_weight * kl_divergence
            components['kl_loss'] = (self.kl_weight * kl_divergence).item()
        
        components['total_loss'] = total_loss.item()
        
        if return_components:
            return total_loss, components
        return total_loss


# Convenience functions for creating mode-aware loss functions

def create_enhanced_loss(model_type='enhanced', mode='MS', target_features=4, **kwargs):
    """
    Create appropriate loss function for enhanced autoformer variants
    
    Args:
        model_type: 'enhanced', 'bayesian', or 'hierarchical'
        mode: Forecasting mode ('M', 'MS', 'S')
        target_features: Number of target features
        **kwargs: Additional arguments for specific loss types
        
    Returns:
        loss_function: Appropriate loss function
    """
    
    if model_type == 'enhanced':
        return ModeAwareLoss(mode=mode, target_features=target_features, **kwargs)
    
    elif model_type == 'bayesian':
        return BayesianModeAwareLoss(mode=mode, target_features=target_features, **kwargs)
    
    elif model_type == 'hierarchical':
        return HierarchicalModeAwareLoss(mode=mode, target_features=target_features, **kwargs)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_mode_aware_criterion(mode='MS', target_features=4, loss_type='mse'):
    """
    Create a simple mode-aware criterion for backward compatibility
    
    Args:
        mode: Forecasting mode ('M', 'MS', 'S')
        target_features: Number of target features
        loss_type: Base loss function type
        
    Returns:
        criterion: Loss function that can be used like nn.MSELoss()
    """
    
    return ModeAwareLoss(mode=mode, target_features=target_features, loss_type=loss_type)


# Example usage and testing
if __name__ == "__main__":
    # Test adaptive loss
    B, L, D = 4, 96, 7
    pred = torch.randn(B, L, D)
    true = torch.randn(B, L, D)
    
    # Test adaptive loss
    adaptive_loss = AdaptiveAutoformerLoss(adaptive_weights=True)
    loss, components = adaptive_loss(pred, true, return_components=True)
    
    print("Adaptive Loss Test:")
    print(f"Total loss: {loss.item():.6f}")
    print(f"Components: {components}")
    
    # Test frequency-aware loss
    freq_loss = FrequencyAwareLoss(freq_weight=0.1)
    freq_loss_val = freq_loss(pred, true)
    
    print(f"\nFrequency-aware loss: {freq_loss_val.item():.6f}")
    
    # Test curriculum scheduler
    curriculum = CurriculumLossScheduler(
        start_seq_len=24,
        target_seq_len=96,
        curriculum_epochs=50
    )
    
    for epoch in [0, 10, 25, 50, 100]:
        seq_len = curriculum.get_current_seq_len(epoch)
        print(f"Epoch {epoch}: sequence length {seq_len}")
    
    # Test mode-aware loss
    mode_loss = ModeAwareLoss(mode='MS', target_features=4, total_features=118, loss_type='mse')
    B, L, D = 2, 10, 118
    predictions = torch.randn(B, L, D)
    targets = torch.randn(B, L, D)
    
    loss_value = mode_loss(predictions, targets, return_components=True)
    print(f"\nMode-Aware Loss (MS mode) Test: {loss_value}")
    
    # Test hierarchical mode-aware loss
    hierarchical_loss = HierarchicalModeAwareLoss(mode='MS', target_features=4)
    resolution_outputs = [predictions, predictions * 0.5, predictions * 0.25]  # Simulated coarser resolutions
    
    total_loss, components = hierarchical_loss(predictions, targets, resolution_outputs=resolution_outputs, return_components=True)
    print(f"\nHierarchical Mode-Aware Loss Test:")
    print(f"Total loss: {total_loss.item():.6f}")
    print(f"Components: {components}")
    
    # Test Bayesian mode-aware loss
    bayesian_loss = BayesianModeAwareLoss(mode='MS', target_features=4)
    uncertainty = torch.abs(torch.randn(B, L, D))  # Simulated uncertainty
    kl_divergence = torch.tensor(0.01)  # Simulated KL divergence
    
    total_loss, components = bayesian_loss(predictions, targets, uncertainty=uncertainty, kl_divergence=kl_divergence, return_components=True)
    print(f"\nBayesian Mode-Aware Loss Test:")
    print(f"Total loss: {total_loss.item():.6f}")
    print(f"Components: {components}")
    
    print("\nEnhanced loss functions test completed!")
