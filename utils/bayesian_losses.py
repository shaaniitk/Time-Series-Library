"""
Bayesian Loss Functions for Uncertainty-Aware Training

This module extends the enhanced loss functions to work with Bayesian models,
providing uncertainty-aware loss computation that integrates with any base loss
function including quantile loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple
import math

from utils.enhanced_losses import AdaptiveAutoformerLoss, FrequencyAwareLoss, QuantileLoss
from utils.logger import logger


class BayesianLoss(nn.Module):
    """
    Base class for Bayesian loss functions that incorporate uncertainty.
    
    This class can wrap any standard loss function and add Bayesian regularization
    terms for proper uncertainty quantification training.
    """
    
    def __init__(self, base_loss_fn, kl_weight=1e-5, uncertainty_weight=0.1):
        super(BayesianLoss, self).__init__()
        logger.info("Initializing BayesianLoss wrapper")
        
        self.base_loss_fn = base_loss_fn
        self.kl_weight = kl_weight
        self.uncertainty_weight = uncertainty_weight
        
    def forward(self, model, pred_result, true, **kwargs):
        """
        Compute Bayesian loss including uncertainty regularization.
        
        Args:
            model: Bayesian model (for KL divergence extraction)
            pred_result: Either tensor (standard) or dict (with uncertainty)
            true: Ground truth tensor
            **kwargs: Additional arguments for base loss
            
        Returns:
            Dict with loss components
        """
        # Extract prediction and uncertainty if available
        if isinstance(pred_result, dict):
            pred = pred_result['prediction']
            uncertainty = pred_result.get('uncertainty', None)
        else:
            pred = pred_result
            uncertainty = None
        
        # Compute base loss
        if hasattr(self.base_loss_fn, '__call__'):
            base_loss = self.base_loss_fn(pred, true, **kwargs)
        else:
            base_loss = self.base_loss_fn(pred, true)
        
        # Extract base loss value if it's a tuple/dict
        if isinstance(base_loss, tuple):
            base_loss_value = base_loss[0]
            base_loss_components = base_loss[1] if len(base_loss) > 1 else {}
        elif isinstance(base_loss, dict):
            base_loss_value = base_loss.get('loss', base_loss.get('total_loss', 0))
            base_loss_components = base_loss
        else:
            base_loss_value = base_loss
            base_loss_components = {}
        
        # KL divergence regularization (for Bayesian layers)
        kl_loss = 0.0
        if hasattr(model, 'get_kl_loss'):
            kl_loss = model.get_kl_loss()
        
        # Uncertainty regularization
        uncertainty_loss = 0.0
        if uncertainty is not None:
            # Penalize extreme uncertainties (too high or too low)
            uncertainty_loss = self._compute_uncertainty_regularization(uncertainty, pred, true)
        
        # Total loss
        total_loss = base_loss_value + self.kl_weight * kl_loss + self.uncertainty_weight * uncertainty_loss
        
        # Prepare result
        result = {
            'total_loss': total_loss,
            'base_loss': base_loss_value,
            'kl_loss': kl_loss,
            'uncertainty_loss': uncertainty_loss,
            'kl_weight': self.kl_weight,
            'uncertainty_weight': self.uncertainty_weight
        }
        
        # Add base loss components
        result.update(base_loss_components)
        
        return result
    
    def _compute_uncertainty_regularization(self, uncertainty, pred, true):
        """
        Compute regularization term based on uncertainty.
        
        Encourages the model to be uncertain when predictions are wrong
        and confident when predictions are correct.
        """
        # Prediction error
        pred_error = torch.abs(pred - true)
        
        # Normalize uncertainty and error for fair comparison
        uncertainty_norm = uncertainty / (torch.mean(uncertainty) + 1e-8)
        error_norm = pred_error / (torch.mean(pred_error) + 1e-8)
        
        # Penalize high confidence when error is high, and low confidence when error is low
        # Good: high uncertainty with high error, low uncertainty with low error
        # Bad: low uncertainty with high error, high uncertainty with low error
        mismatch = torch.abs(error_norm - uncertainty_norm)
        
        return torch.mean(mismatch)


class BayesianQuantileLoss(BayesianLoss):
    """
    Bayesian wrapper for quantile loss with uncertainty-aware quantile prediction.
    
    This loss provides certainty measures for specific quantile ranges and
    encourages proper uncertainty calibration for quantile predictions.
    """
    
    def __init__(self, quantiles=[0.1, 0.5, 0.9], kl_weight=1e-5, 
                 uncertainty_weight=0.1, quantile_certainty_weight=0.05):
        base_loss = QuantileLoss(quantiles)
        super().__init__(base_loss, kl_weight, uncertainty_weight)
        
        logger.info(f"Initializing BayesianQuantileLoss with quantiles={quantiles}")
        self.quantiles = quantiles
        self.quantile_certainty_weight = quantile_certainty_weight
        
        # CRITICAL: Bayesian quantile loss requires output dimension = c_out * num_quantiles
        self.output_dim_multiplier = len(quantiles)
    
    def forward(self, model, pred_result, true, **kwargs):
        """Enhanced quantile loss with uncertainty measures"""
        # Get base Bayesian loss
        result = super().forward(model, pred_result, true, **kwargs)
        
        # Add quantile-specific uncertainty analysis
        if isinstance(pred_result, dict) and 'quantile_specific' in pred_result:
            quantile_certainty_loss = self._compute_quantile_certainty_loss(
                pred_result['quantile_specific'], true
            )
            
            result['quantile_certainty_loss'] = quantile_certainty_loss
            result['total_loss'] = result['total_loss'] + self.quantile_certainty_weight * quantile_certainty_loss
            
            # Add detailed quantile metrics
            result.update(self._compute_quantile_metrics(pred_result['quantile_specific'], true))
        
        return result
    
    def _compute_quantile_certainty_loss(self, quantile_results, true):
        """
        Compute loss that encourages proper uncertainty calibration for quantiles.
        
        The model should be:
        - More certain for median predictions (0.5 quantile)
        - Appropriately uncertain for extreme quantiles (0.1, 0.9)
        """
        certainty_losses = []
        
        for q_name, q_data in quantile_results.items():
            q_level = float(q_name.split('_')[1])  # Extract quantile level
            q_pred = q_data['prediction']
            q_certainty = q_data['certainty_score']
            
            # Expected certainty based on quantile level
            # Median (0.5) should have higher certainty than extremes (0.1, 0.9)
            if q_level == 0.5:
                expected_certainty = 0.8  # High certainty for median
            else:
                # Lower certainty for extreme quantiles
                distance_from_median = abs(q_level - 0.5)
                expected_certainty = 0.8 - distance_from_median  # 0.8 -> 0.4 for 0.1/0.9
            
            # Loss penalizes deviation from expected certainty patterns
            certainty_mismatch = torch.abs(q_certainty - expected_certainty)
            certainty_losses.append(torch.mean(certainty_mismatch))
        
        return sum(certainty_losses) / len(certainty_losses) if certainty_losses else 0.0
    
    def _compute_quantile_metrics(self, quantile_results, true):
        """Compute detailed metrics for quantile predictions"""
        metrics = {}
        
        for q_name, q_data in quantile_results.items():
            q_level = float(q_name.split('_')[1])
            q_pred = q_data['prediction']
            q_uncertainty = q_data['uncertainty']
            q_certainty = q_data['certainty_score']
            
            # Coverage probability (what fraction of true values fall below this quantile)
            coverage = torch.mean((true <= q_pred).float())
            coverage_error = torch.abs(coverage - q_level)
            
            # Quantile-specific metrics
            metrics[f'{q_name}_coverage'] = coverage.item()
            metrics[f'{q_name}_coverage_error'] = coverage_error.item()
            metrics[f'{q_name}_mean_uncertainty'] = torch.mean(q_uncertainty).item()
            metrics[f'{q_name}_mean_certainty'] = torch.mean(q_certainty).item()
        
        # Overall quantile performance
        if len(quantile_results) >= 3:
            # Interval coverage for middle quantiles
            q_low = quantile_results.get('quantile_0.1', {}).get('prediction')
            q_high = quantile_results.get('quantile_0.9', {}).get('prediction')
            
            if q_low is not None and q_high is not None:
                interval_coverage = torch.mean(((true >= q_low) & (true <= q_high)).float())
                metrics['interval_80_coverage'] = interval_coverage.item()
                metrics['interval_80_coverage_error'] = abs(interval_coverage.item() - 0.8)
        
        return metrics


class BayesianAdaptiveLoss(BayesianLoss):
    """
    Bayesian wrapper for adaptive autoformer loss with uncertainty-aware
    trend/seasonal component weighting.
    """
    
    def __init__(self, moving_avg=25, kl_weight=1e-5, uncertainty_weight=0.1,
                 component_uncertainty_weight=0.05):
        base_loss = AdaptiveAutoformerLoss(moving_avg=moving_avg)
        super().__init__(base_loss, kl_weight, uncertainty_weight)
        
        logger.info("Initializing BayesianAdaptiveLoss")
        self.component_uncertainty_weight = component_uncertainty_weight
    
    def forward(self, model, pred_result, true, **kwargs):
        """Enhanced adaptive loss with component-wise uncertainty"""
        result = super().forward(model, pred_result, true, **kwargs)
        
        # Add component-wise uncertainty analysis if available
        if isinstance(pred_result, dict) and 'uncertainty' in pred_result:
            component_uncertainty_loss = self._compute_component_uncertainty_loss(
                pred_result, true
            )
            
            result['component_uncertainty_loss'] = component_uncertainty_loss
            result['total_loss'] = result['total_loss'] + self.component_uncertainty_weight * component_uncertainty_loss
        
        return result
    
    def _compute_component_uncertainty_loss(self, pred_result, true):
        """
        Encourage uncertainty to be higher for more volatile components.
        
        Trend components should have lower uncertainty than seasonal components.
        """
        pred = pred_result['prediction']
        uncertainty = pred_result['uncertainty']
        
        # Decompose predictions and true values (simplified)
        # In practice, you'd use the same decomposition as the model
        from layers.Autoformer_EncDec import series_decomp
        
        decomp = series_decomp(kernel_size=25)
        
        pred_seasonal, pred_trend = decomp(pred)
        true_seasonal, true_trend = decomp(true)
        
        # Compute volatility of components
        trend_volatility = torch.std(true_trend, dim=1, keepdim=True)
        seasonal_volatility = torch.std(true_seasonal, dim=1, keepdim=True)
        
        # Normalize volatilities
        total_volatility = trend_volatility + seasonal_volatility + 1e-8
        trend_vol_ratio = trend_volatility / total_volatility
        seasonal_vol_ratio = seasonal_volatility / total_volatility
        
        # Expected uncertainty should correlate with component volatility
        expected_uncertainty = (trend_vol_ratio * 0.3 + seasonal_vol_ratio * 0.7) * torch.mean(uncertainty)
        
        # Loss penalizes mismatch between expected and actual uncertainty patterns
        uncertainty_mismatch = torch.abs(uncertainty - expected_uncertainty)
        
        return torch.mean(uncertainty_mismatch)


class BayesianFrequencyAwareLoss(BayesianLoss):
    """
    Bayesian wrapper for frequency-aware loss with uncertainty in frequency domain.
    """
    
    def __init__(self, freq_weight=0.1, kl_weight=1e-5, uncertainty_weight=0.1):
        base_loss = FrequencyAwareLoss(freq_weight=freq_weight)
        super().__init__(base_loss, kl_weight, uncertainty_weight)
        
        logger.info("Initializing BayesianFrequencyAwareLoss")
    
    def forward(self, model, pred_result, true, **kwargs):
        """Enhanced frequency-aware loss with spectral uncertainty"""
        result = super().forward(model, pred_result, true, **kwargs)
        
        # Add frequency domain uncertainty analysis
        if isinstance(pred_result, dict) and 'uncertainty' in pred_result:
            freq_uncertainty_loss = self._compute_frequency_uncertainty_loss(
                pred_result, true
            )
            
            result['frequency_uncertainty_loss'] = freq_uncertainty_loss
            result['total_loss'] = result['total_loss'] + 0.05 * freq_uncertainty_loss
        
        return result
    
    def _compute_frequency_uncertainty_loss(self, pred_result, true):
        """
        Encourage uncertainty to be calibrated in frequency domain.
        
        High-frequency components should generally have higher uncertainty.
        """
        pred = pred_result['prediction']
        uncertainty = pred_result['uncertainty']
        
        # FFT of predictions and uncertainty
        pred_fft = torch.fft.fft(pred, dim=1)
        uncertainty_fft = torch.fft.fft(uncertainty, dim=1)
        
        # Frequency magnitudes
        pred_freq_mag = torch.abs(pred_fft)
        uncertainty_freq_mag = torch.abs(uncertainty_fft)
        
        # High frequencies should have higher uncertainty
        freq_bins = torch.arange(pred_fft.shape[1], device=pred.device)
        freq_weight = (freq_bins / freq_bins.max()).unsqueeze(0).unsqueeze(-1)
        
        expected_uncertainty_freq = pred_freq_mag * freq_weight
        
        # Normalize
        expected_uncertainty_freq = expected_uncertainty_freq / (torch.mean(expected_uncertainty_freq) + 1e-8)
        uncertainty_freq_norm = uncertainty_freq_mag / (torch.mean(uncertainty_freq_mag) + 1e-8)
        
        # Loss
        freq_uncertainty_mismatch = torch.abs(expected_uncertainty_freq - uncertainty_freq_norm)
        
        return torch.mean(freq_uncertainty_mismatch)


class UncertaintyCalibrationLoss(nn.Module):
    """
    Specialized loss for calibrating uncertainty estimates.
    
    Ensures that uncertainty estimates are well-calibrated:
    - Predicted uncertainty should match actual prediction errors
    - Confidence intervals should have proper coverage
    """
    
    def __init__(self, calibration_weight=0.1):
        super().__init__()
        logger.info("Initializing UncertaintyCalibrationLoss")
        self.calibration_weight = calibration_weight
    
    def forward(self, pred_result, true):
        """
        Compute calibration loss for uncertainty estimates.
        
        Args:
            pred_result: Dict with 'prediction', 'uncertainty', 'confidence_intervals'
            true: Ground truth tensor
            
        Returns:
            Calibration loss value
        """
        if not isinstance(pred_result, dict):
            return 0.0
        
        pred = pred_result.get('prediction')
        uncertainty = pred_result.get('uncertainty')
        confidence_intervals = pred_result.get('confidence_intervals', {})
        
        if pred is None or uncertainty is None:
            return 0.0
        
        calibration_losses = []
        
        # 1. Uncertainty-error correlation
        pred_error = torch.abs(pred - true)
        error_uncertainty_corr = self._compute_correlation_loss(pred_error, uncertainty)
        calibration_losses.append(error_uncertainty_corr)
        
        # 2. Confidence interval coverage
        for conf_name, conf_data in confidence_intervals.items():
            if 'lower' in conf_data and 'upper' in conf_data:
                coverage_loss = self._compute_coverage_loss(
                    true, conf_data['lower'], conf_data['upper'], conf_name
                )
                calibration_losses.append(coverage_loss)
        
        return sum(calibration_losses) / len(calibration_losses) if calibration_losses else 0.0
    
    def _compute_correlation_loss(self, pred_error, uncertainty):
        """Encourage positive correlation between prediction error and uncertainty"""
        # Normalize both to same scale
        error_norm = (pred_error - torch.mean(pred_error)) / (torch.std(pred_error) + 1e-8)
        uncertainty_norm = (uncertainty - torch.mean(uncertainty)) / (torch.std(uncertainty) + 1e-8)
        
        # Correlation should be positive
        correlation = torch.mean(error_norm * uncertainty_norm)
        
        # Loss is higher when correlation is negative or low
        target_correlation = 0.5  # Target positive correlation
        correlation_loss = torch.abs(correlation - target_correlation)
        
        return correlation_loss
    
    def _compute_coverage_loss(self, true, lower_bound, upper_bound, conf_name):
        """Compute loss for confidence interval coverage"""
        # Extract target coverage from name (e.g., "95%" -> 0.95)
        try:
            target_coverage = float(conf_name.replace('%', '')) / 100.0
        except:
            target_coverage = 0.95  # Default
        
        # Actual coverage
        in_interval = ((true >= lower_bound) & (true <= upper_bound)).float()
        actual_coverage = torch.mean(in_interval)
        
        # Coverage should match target
        coverage_error = torch.abs(actual_coverage - target_coverage)
        
        return coverage_error


# Factory function for easy creation
def create_bayesian_loss(loss_type='adaptive', **kwargs):
    """
    Factory function to create Bayesian loss functions.
    
    Args:
        loss_type: Type of base loss ('adaptive', 'quantile', 'frequency', 'mse', 'mae')
        **kwargs: Additional arguments for the specific loss function
        
    Returns:
        Configured Bayesian loss function
    """
    logger.info(f"Creating Bayesian loss of type: {loss_type}")
    
    if loss_type == 'adaptive':
        return BayesianAdaptiveLoss(**kwargs)
    elif loss_type == 'quantile':
        return BayesianQuantileLoss(**kwargs)
    elif loss_type == 'frequency':
        return BayesianFrequencyAwareLoss(**kwargs)
    elif loss_type == 'mse':
        return BayesianLoss(nn.MSELoss(), **kwargs)
    elif loss_type == 'mae':
        return BayesianLoss(nn.L1Loss(), **kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Example usage
if __name__ == "__main__":
    logger.info("Testing Bayesian loss functions...")
    
    # Mock prediction results
    batch_size, seq_len, features = 4, 24, 7
    
    # Standard prediction result
    pred_standard = torch.randn(batch_size, seq_len, features)
    true = torch.randn(batch_size, seq_len, features)
    
    # Bayesian prediction result with uncertainty
    pred_bayesian = {
        'prediction': pred_standard,
        'uncertainty': torch.rand(batch_size, seq_len, features) * 0.5,
        'confidence_intervals': {
            '95%': {
                'lower': pred_standard - 1.96 * 0.3,
                'upper': pred_standard + 1.96 * 0.3
            }
        }
    }
    
    # Mock model with KL loss
    class MockModel:
        def get_kl_loss(self):
            return torch.tensor(0.01)
    
    model = MockModel()
    
    # Test different Bayesian loss functions
    losses_to_test = [
        ('adaptive', BayesianAdaptiveLoss()),
        ('quantile', BayesianQuantileLoss()),
        ('frequency', BayesianFrequencyAwareLoss()),
        ('calibration', UncertaintyCalibrationLoss())
    ]
    
    for name, loss_fn in losses_to_test:
        if name == 'calibration':
            result = loss_fn(pred_bayesian, true)
            print(f"{name} loss: {result.item():.6f}")
        else:
            result = loss_fn(model, pred_bayesian, true)
            print(f"{name} loss components:")
            for k, v in result.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {v.item():.6f}")
                else:
                    print(f"  {k}: {v}")
            print()
    
    logger.info("Bayesian loss functions test completed!")
