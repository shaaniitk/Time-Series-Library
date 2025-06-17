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
    
    print("\nEnhanced loss functions test completed!")
