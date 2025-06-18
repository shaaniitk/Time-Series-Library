"""
Bayesian Enhanced Autoformer for Uncertainty Quantification

This module implements a Bayesian version of the Enhanced Autoformer that provides
uncertainty estimates along with predictions. It integrates with any loss function
including quantile loss to provide certainty measures for specific quantile ranges.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from models.EnhancedAutoformer import EnhancedAutoformer, LearnableSeriesDecomp
from layers.BayesianLayers import BayesianLinear, BayesianConv1d, convert_to_bayesian, collect_kl_divergence, DropoutSampling
from layers.Embed import DataEmbedding_wo_pos
from utils.logger import logger


class BayesianEnhancedAutoformer(nn.Module):
    """
    Bayesian version of Enhanced Autoformer with uncertainty quantification.
    
    Key Features:
    - Provides prediction uncertainty estimates
    - Works with any loss function (MSE, MAE, Quantile, etc.)
    - Separates aleatoric and epistemic uncertainty
    - Supports multiple uncertainty estimation methods
    """
    
    def __init__(self, configs, uncertainty_method='bayesian', n_samples=50, 
                 bayesian_layers=['projection'], kl_weight=1e-5, 
                 use_quantiles=None, quantile_levels=None):
        super(BayesianEnhancedAutoformer, self).__init__()
        logger.info(f"Initializing BayesianEnhancedAutoformer with method={uncertainty_method}")
        
        self.configs = configs
        self.uncertainty_method = uncertainty_method
        self.n_samples = n_samples
        self.kl_weight = kl_weight
        
        # Quantile configuration - check config first, then parameters
        self.quantile_mode = getattr(configs, 'quantile_mode', False) or use_quantiles or False
        self.quantiles = getattr(configs, 'quantiles', None) or quantile_levels or [0.1, 0.25, 0.5, 0.75, 0.9]
        
        # Store original c_out for reference
        self.original_c_out = configs.c_out
        
        # Don't modify configs.c_out - let base model initialize with original dimensions
        # We'll add quantile expansion later
        
        # Base enhanced autoformer (now with original dimensions)
        self.base_model = EnhancedAutoformer(configs)
        
        # Add quantile expansion layer if needed
        if self.quantile_mode:
            self.quantile_expansion = nn.Linear(self.original_c_out, self.original_c_out * len(self.quantiles))
            logger.info(f"Quantile mode enabled: {len(self.quantiles)} quantiles, added expansion layer {self.original_c_out} -> {self.original_c_out * len(self.quantiles)}")
        
        # Convert specified layers to Bayesian
        if uncertainty_method == 'bayesian':
            self._setup_bayesian_layers(bayesian_layers)
        elif uncertainty_method == 'dropout':
            self._setup_dropout_layers()
        
    def _setup_bayesian_layers(self, bayesian_layers):
        """Convert specified layers to Bayesian versions"""
        logger.info(f"Converting layers to Bayesian: {bayesian_layers}")
        
        if 'projection' in bayesian_layers:
            # Replace final projection layer with Bayesian version
            # Check multiple possible locations for projection layer
            projection_found = False
            
            # Try top-level projection first
            if hasattr(self.base_model, 'projection'):
                original_proj = self.base_model.projection
                self.base_model.projection = BayesianLinear(
                    original_proj.in_features,
                    original_proj.out_features,
                    bias=original_proj.bias is not None
                )
                projection_found = True
                logger.info(f"Converted top-level projection: {original_proj.in_features}->{original_proj.out_features}")
            
            # Try decoder projection (EnhancedAutoformer uses this)
            elif hasattr(self.base_model, 'decoder') and hasattr(self.base_model.decoder, 'projection'):
                original_proj = self.base_model.decoder.projection
                self.base_model.decoder.projection = BayesianLinear(
                    original_proj.in_features,
                    original_proj.out_features,
                    bias=original_proj.bias is not None
                )
                projection_found = True
                logger.info(f"Converted decoder projection: {original_proj.in_features}->{original_proj.out_features}")
            
            if not projection_found:
                logger.warning("No projection layer found to convert to Bayesian!")
        
        if 'encoder' in bayesian_layers:
            # Convert encoder layers
            convert_to_bayesian(self.base_model.enc_embedding, ['Linear'])
            for layer in self.base_model.encoder.attn_layers:
                convert_to_bayesian(layer, ['Linear'])
        
        if 'decoder' in bayesian_layers:
            # Convert decoder layers  
            convert_to_bayesian(self.base_model.dec_embedding, ['Linear'])
            for layer in self.base_model.decoder.layers:
                convert_to_bayesian(layer, ['Linear'])
    
    def _setup_dropout_layers(self):
        """Add Monte Carlo dropout layers for uncertainty estimation"""
        logger.info("Setting up Monte Carlo Dropout layers")
        
        # Add dropout layers at key points
        self.mc_dropout1 = DropoutSampling(p=0.1)
        self.mc_dropout2 = DropoutSampling(p=0.15)
        self.mc_dropout3 = DropoutSampling(p=0.1)
    
    def disable_quantile_mode(self):
        """Disable quantile prediction mode and revert to single predictions."""
        logger.info("Disabling quantile mode")
        self.quantile_mode = False
        self.quantiles = None
        # Note: This doesn't revert layer dimensions - create new model instance if needed
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                return_uncertainty=False, detailed_uncertainty=False):
        """
        Forward pass with optional uncertainty estimation.
        
        Args:
            x_enc, x_mark_enc, x_dec, x_mark_dec: Standard Autoformer inputs
            return_uncertainty: Whether to return uncertainty estimates
            detailed_uncertainty: Whether to return detailed uncertainty breakdown
            
        Returns:
            If return_uncertainty=False: Standard prediction tensor
            If return_uncertainty=True: Dict with predictions and uncertainty measures
        """
        if not return_uncertainty:
            # Standard forward pass
            return self._single_forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Uncertainty estimation
        if self.uncertainty_method == 'bayesian':
            return self._bayesian_forward(x_enc, x_mark_enc, x_dec, x_mark_dec, detailed_uncertainty)
        elif self.uncertainty_method == 'dropout':
            return self._dropout_forward(x_enc, x_mark_enc, x_dec, x_mark_dec, detailed_uncertainty)
        else:
            raise ValueError(f"Unknown uncertainty method: {self.uncertainty_method}")
    
    def _single_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """Single forward pass through the model"""
        if self.uncertainty_method == 'dropout':
            # Apply MC Dropout
            x_enc = self.mc_dropout1(x_enc)
            
        output = self.base_model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        if self.uncertainty_method == 'dropout':
            output = self.mc_dropout3(output)
        
        # Apply quantile expansion if enabled
        if self.quantile_mode:
            output = self.quantile_expansion(output)
            
        return output
    
    def _bayesian_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, detailed=False):
        """Bayesian uncertainty estimation using weight sampling"""
        logger.debug(f"Computing Bayesian uncertainty with {self.n_samples} samples")
        
        predictions = []
        
        # Sample multiple predictions
        for i in range(self.n_samples):
            with torch.no_grad() if i > 0 else torch.enable_grad():  # Save memory for additional samples
                pred = self._single_forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
                predictions.append(pred.detach() if i > 0 else pred)
        
        # Stack predictions: [n_samples, batch, seq_len, features]
        pred_stack = torch.stack(predictions)
        
        return self._compute_uncertainty_statistics(pred_stack, detailed)
    
    def _dropout_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, detailed=False):
        """Monte Carlo Dropout uncertainty estimation"""
        logger.debug(f"Computing MC Dropout uncertainty with {self.n_samples} samples")
        
        predictions = []
        
        # Enable training mode for dropout
        self.train()
        
        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = self._single_forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
                predictions.append(pred)
        
        # Restore original mode
        self.eval()
        
        pred_stack = torch.stack(predictions)
        return self._compute_uncertainty_statistics(pred_stack, detailed)
    
    def _compute_uncertainty_statistics(self, pred_stack, detailed=False):
        """
        Compute uncertainty statistics from prediction samples.
        
        Args:
            pred_stack: [n_samples, batch, seq_len, features] tensor of predictions
            detailed: Whether to compute detailed uncertainty breakdown
            
        Returns:
            Dictionary with uncertainty statistics
        """
        # Basic statistics
        mean_pred = torch.mean(pred_stack, dim=0)
        total_variance = torch.var(pred_stack, dim=0)
        total_std = torch.sqrt(total_variance)
        
        # Confidence intervals
        confidence_intervals = self._compute_confidence_intervals(pred_stack)
        
        result = {
            'prediction': mean_pred,
            'uncertainty': total_std,
            'variance': total_variance,
            'confidence_intervals': confidence_intervals,
            'predictions_samples': pred_stack if detailed else None
        }
        
        # Quantile-specific uncertainty if in quantile mode
        if self.quantile_mode and self.quantiles:
            result.update(self._compute_quantile_uncertainty(pred_stack))
        
        # Detailed uncertainty decomposition
        if detailed:
            result.update(self._compute_detailed_uncertainty(pred_stack))
        
        return result
    
    def _compute_confidence_intervals(self, pred_stack, confidence_levels=[0.68, 0.95, 0.99]):
        """Compute confidence intervals from prediction samples"""
        intervals = {}
        
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_bound = torch.quantile(pred_stack, lower_percentile / 100, dim=0)
            upper_bound = torch.quantile(pred_stack, upper_percentile / 100, dim=0)
            
            intervals[f'{int(conf_level * 100)}%'] = {
                'lower': lower_bound,
                'upper': upper_bound,
                'width': upper_bound - lower_bound
            }
        
        return intervals
    
    def _compute_quantile_uncertainty(self, pred_stack):
        """
        Compute uncertainty specific to quantile predictions.
        
        For quantile predictions, this provides certainty measures for each quantile level.
        """
        if not self.quantiles:
            return {}
        
        n_quantiles = len(self.quantiles)
        batch_size, seq_len, total_features = pred_stack.shape[1], pred_stack.shape[2], pred_stack.shape[3]
        features_per_quantile = total_features // n_quantiles
        
        quantile_results = {}
        
        for i, q_level in enumerate(self.quantiles):
            # Extract predictions for this quantile
            start_idx = i * features_per_quantile
            end_idx = (i + 1) * features_per_quantile
            
            q_predictions = pred_stack[:, :, :, start_idx:end_idx]
            
            # Compute uncertainty for this quantile
            q_mean = torch.mean(q_predictions, dim=0)
            q_std = torch.std(q_predictions, dim=0)
            
            # Quantile-specific confidence intervals
            q_intervals = {}
            for conf in [0.68, 0.95]:
                alpha = 1 - conf
                q_lower = torch.quantile(q_predictions, alpha/2, dim=0)
                q_upper = torch.quantile(q_predictions, 1-alpha/2, dim=0)
                q_intervals[f'{int(conf*100)}%'] = {
                    'lower': q_lower,
                    'upper': q_upper,
                    'width': q_upper - q_lower
                }
            
            quantile_results[f'quantile_{q_level}'] = {
                'prediction': q_mean,
                'uncertainty': q_std,
                'confidence_intervals': q_intervals,
                'certainty_score': self._compute_quantile_certainty(q_predictions, q_level)
            }
        
        return {'quantile_specific': quantile_results}
    
    def _compute_quantile_certainty(self, q_predictions, q_level):
        """
        Compute certainty score for a specific quantile.
        
        Higher scores indicate more confident predictions for this quantile level.
        """
        # Coefficient of variation (lower is more certain)
        mean_pred = torch.mean(q_predictions, dim=0)
        std_pred = torch.std(q_predictions, dim=0)
        
        # Avoid division by zero
        cv = std_pred / (torch.abs(mean_pred) + 1e-8)
        
        # Convert to certainty score (higher is more certain)
        certainty = 1.0 / (1.0 + cv)
        
        return certainty
    
    def _compute_detailed_uncertainty(self, pred_stack):
        """Compute detailed uncertainty decomposition"""
        # This is a simplified version - in practice, you'd need more sophisticated
        # methods to separate aleatoric and epistemic uncertainty
        
        # Total uncertainty
        total_var = torch.var(pred_stack, dim=0)
        
        # Approximate epistemic uncertainty (model uncertainty)
        # Using variance across samples as a proxy
        epistemic_var = total_var * 0.7  # Simplified assumption
        
        # Approximate aleatoric uncertainty (data uncertainty)  
        aleatoric_var = total_var - epistemic_var
        
        return {
            'epistemic_uncertainty': torch.sqrt(epistemic_var),
            'aleatoric_uncertainty': torch.sqrt(aleatoric_var),
            'total_uncertainty_decomposed': torch.sqrt(total_var)
        }
    
    def get_kl_loss(self):
        """Get KL divergence loss for Bayesian regularization"""
        if self.uncertainty_method != 'bayesian':
            return 0.0
        
        return collect_kl_divergence(self) * self.kl_weight
    
    def set_uncertainty_method(self, method, **kwargs):
        """Change uncertainty estimation method"""
        logger.info(f"Switching uncertainty method to: {method}")
        self.uncertainty_method = method
        
        if 'n_samples' in kwargs:
            self.n_samples = kwargs['n_samples']
    
    def get_uncertainty_summary(self, uncertainty_dict):
        """Get a summary of uncertainty measures for logging/monitoring"""
        if 'uncertainty' not in uncertainty_dict:
            return {}
        
        uncertainty = uncertainty_dict['uncertainty']
        
        summary = {
            'mean_uncertainty': torch.mean(uncertainty).item(),
            'max_uncertainty': torch.max(uncertainty).item(),
            'min_uncertainty': torch.min(uncertainty).item(),
            'uncertainty_std': torch.std(uncertainty).item()
        }
        
        # Add quantile-specific summaries if available
        if 'quantile_specific' in uncertainty_dict:
            for q_name, q_data in uncertainty_dict['quantile_specific'].items():
                q_uncertainty = q_data['uncertainty']
                summary[f'{q_name}_mean_uncertainty'] = torch.mean(q_uncertainty).item()
                summary[f'{q_name}_mean_certainty'] = torch.mean(q_data['certainty_score']).item()
        
        return summary
    
    def compute_loss(self, predictions, targets, criterion, return_components=False):
        """
        Compute total Bayesian loss with support for quantile loss and normalized contributions.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets  
            criterion: Loss function (MSE, MAE, Quantile, etc.)
            return_components: Whether to return loss components separately
            
        Returns:
            If return_components=False: total_loss (scalar)
            If return_components=True: dict with loss breakdown
        """
        # Compute primary data loss (quantile, MSE, etc.)
        if hasattr(criterion, 'quantiles') and self.quantile_mode:
            # Quantile loss mode - predictions contain multiple quantiles
            data_loss = self._compute_quantile_loss(predictions, targets, criterion)
        else:
            # Standard loss mode
            data_loss = criterion(predictions, targets)
        
        # Compute KL divergence loss with external weight (not auto-normalized)
        kl_loss_raw = self.get_kl_loss()
        
        # Use the configured KL weight (from KL tuning)
        total_loss = data_loss + self.kl_weight * kl_loss_raw
        
        # Track components for debugging
        data_loss_value = data_loss.item() if hasattr(data_loss, 'item') else float(data_loss)
        kl_loss_value = kl_loss_raw.item() if hasattr(kl_loss_raw, 'item') else float(kl_loss_raw)
        kl_contribution = (self.kl_weight * kl_loss_value)
        
        if return_components:
            return {
                'total_loss': total_loss,
                'data_loss': data_loss,
                'kl_loss': kl_loss_raw,
                'kl_weight_used': self.kl_weight,
                'raw_data_loss': data_loss_value,
                'raw_kl_loss': kl_loss_value,
                'kl_contribution': kl_contribution
            }
        else:
            return total_loss
    
    def _compute_quantile_loss(self, predictions, targets, criterion):
        """
        Compute quantile loss for multi-quantile predictions.
        
        Args:
            predictions: [batch, seq, features * n_quantiles]
            targets: [batch, seq, features]
            criterion: Quantile loss function with .quantiles attribute
            
        Returns:
            Quantile loss tensor
        """
        if not hasattr(criterion, 'quantiles'):
            raise ValueError("Criterion must have 'quantiles' attribute for quantile loss")
        
        quantiles = criterion.quantiles
        n_quantiles = len(quantiles)
        batch_size, seq_len, total_features = predictions.shape
        features_per_quantile = total_features // n_quantiles
        
        # Reshape predictions to [batch, seq, n_quantiles, features]
        pred_reshaped = predictions.view(batch_size, seq_len, n_quantiles, features_per_quantile)
        
        # Expand targets to match quantile structure
        targets_expanded = targets.unsqueeze(2).expand(-1, -1, n_quantiles, -1)
        
        # Compute quantile loss
        total_loss = 0.0
        for i, q in enumerate(quantiles):
            pred_q = pred_reshaped[:, :, i, :]  # [batch, seq, features]
            target_q = targets_expanded[:, :, i, :]  # [batch, seq, features]
            
            # Pinball loss for this quantile
            errors = target_q - pred_q
            loss_q = torch.where(errors >= 0, q * errors, (q - 1) * errors)
            total_loss += loss_q.mean()
        
        return total_loss / n_quantiles
    
    def get_loss_function(self, base_criterion):
        """
        Return a loss function that automatically includes KL divergence.
        
        Args:
            base_criterion: Base loss function (MSE, MAE, etc.)
            
        Returns:
            Wrapped loss function that includes KL divergence
        """
        def bayesian_loss_fn(predictions, targets):
            return self.compute_loss(predictions, targets, base_criterion)
        
        return bayesian_loss_fn
    
    def configure_optimizer_loss(self, base_criterion, verbose=False):
        """
        Configure the model for training with proper Bayesian loss.
        
        Args:
            base_criterion: Base loss function
            verbose: Whether to log loss components during training
            
        Returns:
            Configured loss function
        """
        self._verbose_loss = verbose
        self._loss_history = []
        
        def enhanced_bayesian_loss(predictions, targets):
            loss_components = self.compute_loss(
                predictions, targets, base_criterion, return_components=True
            )
            
            if self._verbose_loss:
                self._loss_history.append({
                    'data_loss': loss_components['data_loss'].item(),
                    'kl_loss': loss_components['kl_contribution'],
                    'total_loss': loss_components['total_loss'].item()
                })
                
                # Log every 10 calls to avoid spam
                if len(self._loss_history) % 10 == 0:
                    recent = self._loss_history[-1]
                    logger.info(f"Bayesian Loss - Data: {recent['data_loss']:.6f}, "
                              f"KL: {recent['kl_loss']:.6f}, "
                              f"Total: {recent['total_loss']:.6f}")
            
            return loss_components['total_loss']
        
        return enhanced_bayesian_loss

# Alias for compatibility with experiment framework
Model = BayesianEnhancedAutoformer

# Example usage and testing
if __name__ == "__main__":
    from argparse import Namespace
    
    # Mock config for testing
    configs = Namespace(
        seq_len=96,
        label_len=48,
        pred_len=24,
        enc_in=7,
        dec_in=7,
        c_out=7,
        d_model=64,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=256,
        factor=1,
        dropout=0.1,
        embed='timeF',
        freq='h',
        activation='gelu'
    )
    
    logger.info("Testing BayesianEnhancedAutoformer...")
    
    # Create model
    model = BayesianEnhancedAutoformer(
        configs, 
        uncertainty_method='bayesian',
        n_samples=10,
        bayesian_layers=['projection']
    )
    
    # Test data
    batch_size = 4
    x_enc = torch.randn(batch_size, configs.seq_len, configs.enc_in)
    x_mark_enc = torch.randn(batch_size, configs.seq_len, 4)
    x_dec = torch.randn(batch_size, configs.label_len + configs.pred_len, configs.dec_in)
    x_mark_dec = torch.randn(batch_size, configs.label_len + configs.pred_len, 4)
    
    # Test standard prediction
    pred = model(x_enc, x_mark_enc, x_dec, x_mark_dec, return_uncertainty=False)
    print(f"Standard prediction shape: {pred.shape}")
    
    # Test uncertainty prediction
    result = model(x_enc, x_mark_enc, x_dec, x_mark_dec, return_uncertainty=True, detailed_uncertainty=True)
    
    print(f"Prediction shape: {result['prediction'].shape}")
    print(f"Uncertainty shape: {result['uncertainty'].shape}")
    print(f"Available confidence intervals: {list(result['confidence_intervals'].keys())}")
    
    # Test quantile mode
    # Create model with quantile mode enabled from start
    model = BayesianEnhancedAutoformer(
        configs, 
        use_quantiles=True, 
        quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9]
    )
    result_quantile = model(x_enc, x_mark_enc, x_dec, x_mark_dec, return_uncertainty=True)
    
    if 'quantile_specific' in result_quantile:
        print(f"Quantile-specific results: {list(result_quantile['quantile_specific'].keys())}")
    
    # Test uncertainty summary
    summary = model.get_uncertainty_summary(result)
    print(f"Uncertainty summary: {summary}")
    
    logger.info("BayesianEnhancedAutoformer test completed!")
