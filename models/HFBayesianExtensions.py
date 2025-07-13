"""
Bayesian Extension Module for HF Models

This module handles Bayesian uncertainty quantification for HF models by:
1. Converting specific HF layers to Bayesian equivalents
2. Implementing Monte Carlo dropout for uncertainty estimation
3. Following existing patterns from BayesianEnhancedAutoformer

Key Technical Solution for Bayesian + HF:
- External layer conversion: Replace deterministic layers with Bayesian versions
- Preserve HF backbone stability while adding uncertainty capabilities
- Use existing BayesianLayers infrastructure
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
from argparse import Namespace

from layers.BayesianLayers import (
    BayesianLinear, 
    convert_to_bayesian, 
    collect_kl_divergence, 
    DropoutSampling
)
from utils.logger import logger


class BayesianExtension(nn.Module):
    """
    Bayesian extension for HF models that handles uncertainty quantification.
    
    This solves the core challenge: HF transformers have deterministic weights,
    but Bayesian methods need weight distributions.
    
    Solution: External layer conversion + sampling-based uncertainty
    """
    
    def __init__(self, configs, hf_backbone):
        super().__init__()
        
        self.uncertainty_method = getattr(configs, 'uncertainty_method', 'dropout')
        self.n_samples = getattr(configs, 'n_samples', 10)
        self.kl_weight = getattr(configs, 'kl_weight', 1e-5)
        
        # Store reference to HF backbone for layer conversion
        self.hf_backbone = hf_backbone
        
        logger.info(f"Initializing BayesianExtension with method: {self.uncertainty_method}")
        
        # Convert layers following existing pattern from BayesianEnhancedAutoformer
        if self.uncertainty_method == 'bayesian':
            self._convert_to_bayesian(
                getattr(configs, 'bayesian_layers', ['output_projection'])
            )
        elif self.uncertainty_method == 'dropout':
            self._setup_mc_dropout()
            
        # Store original training state for MC Dropout
        self._original_training_state = None
            
    def _convert_to_bayesian(self, layer_names):
        """
        Convert HF backbone layers to Bayesian using existing infrastructure.
        
        This is the key solution: Instead of modifying HF internals,
        we replace specific layers with Bayesian equivalents.
        """
        logger.info(f"Converting HF layers to Bayesian: {layer_names}")
        
        # Use existing convert_to_bayesian function from BayesianLayers
        self.bayesian_layers_list = convert_to_bayesian(self.hf_backbone, layer_names)
        
        logger.info(f"Successfully converted {len(self.bayesian_layers_list)} HF layers to Bayesian")
        
        # Log which layers were converted
        for i, layer in enumerate(self.bayesian_layers_list):
            logger.debug(f"  Bayesian layer {i}: {type(layer).__name__}")
        
    def _setup_mc_dropout(self):
        """Setup MC Dropout following existing pattern from BayesianEnhancedAutoformer"""
        logger.info("Setting up Monte Carlo Dropout layers for HF model")
        
        self.mc_dropout1 = DropoutSampling(p=0.1)   # Input dropout
        self.mc_dropout2 = DropoutSampling(p=0.15)  # Intermediate dropout
        self.mc_dropout3 = DropoutSampling(p=0.1)   # Output dropout
        
        logger.debug("MC Dropout layers configured with rates: 0.1, 0.15, 0.1")
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Forward with uncertainty estimation using existing patterns.
        
        Returns either:
        - Standard tensor (for non-uncertainty methods)
        - Dict with uncertainty statistics (for Bayesian/dropout methods)
        """
        
        if self.uncertainty_method == 'bayesian':
            return self._bayesian_forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
        elif self.uncertainty_method == 'dropout':
            return self._dropout_forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
        else:
            # Standard forward through HF backbone
            return self.hf_backbone(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            
    def _bayesian_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Bayesian forward following exact pattern from BayesianEnhancedAutoformer.
        
        Key insight: Each forward pass samples different weights from the
        Bayesian layers, providing natural uncertainty quantification.
        """
        logger.debug(f"Computing Bayesian uncertainty with {self.n_samples} samples")
        
        predictions = []
        
        # Gradient context management (exact pattern from BayesianEnhancedAutoformer)
        grad_context = torch.enable_grad if self.training else torch.no_grad
        
        with grad_context():
            for _ in range(self.n_samples):
                # Each call samples new weights from Bayesian layers
                pred = self.hf_backbone(x_enc, x_mark_enc, x_dec, x_mark_dec)
                predictions.append(pred)
                
        # Stack predictions and detach if evaluating
        pred_stack = torch.stack(predictions)
        
        if not self.training:
            pred_stack = pred_stack.detach()
            
        return self._compute_uncertainty_statistics(pred_stack)
        
    def _dropout_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        MC Dropout forward following existing pattern.
        
        Key insight: Keep HF model unchanged, add uncertainty through
        external dropout sampling.
        """
        logger.debug(f"Computing MC Dropout uncertainty with {self.n_samples} samples")
        
        predictions = []
        self._original_training_state = self.training
        
        # Force training mode for MC Dropout (enables dropout during inference)
        self.train()
        
        with torch.no_grad():
            for _ in range(self.n_samples):
                # Apply dropout to input
                x_enc_dropped = self.mc_dropout1(x_enc)
                
                # Forward through HF backbone (unchanged)
                pred = self.hf_backbone(x_enc_dropped, x_mark_enc, x_dec, x_mark_dec)
                
                # Apply dropout to output
                pred_dropped = self.mc_dropout3(pred)
                predictions.append(pred_dropped)
                
        # Restore original training state
        if self._original_training_state is not None:
            self.train(self._original_training_state)
            
        pred_stack = torch.stack(predictions)
        return self._compute_uncertainty_statistics(pred_stack)
        
    def _compute_uncertainty_statistics(self, pred_stack):
        """
        Compute uncertainty statistics following existing pattern from BayesianEnhancedAutoformer
        
        Returns dict with:
        - prediction: mean prediction
        - uncertainty: standard deviation
        - variance: prediction variance
        - confidence_intervals: 68%, 95%, 99% intervals
        - predictions_samples: all samples (for debugging)
        """
        mean_pred = torch.mean(pred_stack, dim=0)
        total_variance = torch.var(pred_stack, dim=0)
        total_std = torch.sqrt(total_variance)
        
        # Compute confidence intervals following existing pattern
        confidence_intervals = {}
        for conf_level in [0.68, 0.95, 0.99]:
            alpha = 1 - conf_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_bound = torch.quantile(pred_stack, lower_percentile / 100, dim=0)
            upper_bound = torch.quantile(pred_stack, upper_percentile / 100, dim=0)
            
            confidence_intervals[f'{int(conf_level * 100)}%'] = {
                'lower': lower_bound,
                'upper': upper_bound,
                'width': upper_bound - lower_bound
            }
            
        logger.debug(f"Computed uncertainty statistics for {pred_stack.shape[0]} samples")
            
        return {
            'prediction': mean_pred,
            'uncertainty': total_std,
            'variance': total_variance,
            'confidence_intervals': confidence_intervals,
            'predictions_samples': pred_stack
        }
        
    def get_kl_loss(self, max_kl_value=1e6):
        """
        Get KL loss following existing pattern from BayesianEnhancedAutoformer.
        
        This works because we converted specific HF layers to Bayesian versions
        that track KL divergence.
        """
        if self.uncertainty_method != 'bayesian':
            return torch.tensor(0.0, device=next(self.parameters()).device)
            
        # Use existing KL divergence collection
        kl_div = collect_kl_divergence(self.hf_backbone)
        
        # Clip KL divergence to prevent instability (exact pattern)
        kl_div_clipped = torch.clamp(kl_div, max=max_kl_value)
        
        if kl_div > kl_div_clipped:
            logger.warning(f"Clipped KL divergence from {kl_div:.4f} to {kl_div_clipped:.4f}")
            
        return kl_div_clipped * self.kl_weight
        
    def get_uncertainty_info(self):
        """Get information about uncertainty configuration"""
        return {
            'uncertainty_method': self.uncertainty_method,
            'n_samples': self.n_samples,
            'kl_weight': self.kl_weight,
            'bayesian_layers': len(getattr(self, 'bayesian_layers_list', [])),
            'has_mc_dropout': hasattr(self, 'mc_dropout1')
        }


class QuantileExtension(nn.Module):
    """
    Quantile regression extension for HF models.
    
    Follows the quantile expansion pattern from BayesianEnhancedAutoformer.
    """
    
    def __init__(self, configs):
        super().__init__()
        
        self.quantiles = getattr(configs, 'quantile_levels', [0.1, 0.5, 0.9])
        self.num_quantiles = len(self.quantiles)
        self.original_c_out = getattr(configs, 'c_out_evaluation', configs.c_out)
        
        logger.info(f"Initializing QuantileExtension with {self.num_quantiles} quantiles: {self.quantiles}")
        
        # Quantile expansion layer (same pattern as BayesianEnhancedAutoformer)
        self.quantile_expansion = nn.Linear(
            self.original_c_out, 
            self.original_c_out * self.num_quantiles
        )
        
        logger.info(f"Created quantile expansion: {self.original_c_out} -> {self.original_c_out * self.num_quantiles}")
        
    def forward(self, base_output, x_enc=None, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        Expand base output to quantile predictions.
        
        Args:
            base_output: Output from base model or previous processor
            
        Returns:
            Expanded output with quantile predictions
        """
        if isinstance(base_output, dict):
            # Handle uncertainty dict from BayesianExtension
            if 'prediction' in base_output:
                expanded_pred = self.quantile_expansion(base_output['prediction'])
                base_output['prediction'] = expanded_pred
                
                # Update prediction samples if available
                if 'predictions_samples' in base_output and base_output['predictions_samples'] is not None:
                    expanded_samples = self.quantile_expansion(base_output['predictions_samples'])
                    base_output['predictions_samples'] = expanded_samples
                    
            return base_output
        else:
            # Standard tensor input
            return self.quantile_expansion(base_output)
            
    def compute_quantile_uncertainty(self, pred_stack, quantiles):
        """
        Compute quantile-specific uncertainty following existing pattern.
        
        This is used when both Bayesian and quantile extensions are active.
        """
        if not quantiles:
            return {}
            
        n_quantiles = len(quantiles)
        total_features = pred_stack.shape[-1]
        
        if total_features % n_quantiles != 0:
            logger.warning(f"Cannot compute quantile uncertainty: {total_features} not divisible by {n_quantiles}")
            return {}
            
        features_per_quantile = total_features // n_quantiles
        quantile_results = {}
        
        for i, q_level in enumerate(quantiles):
            start_idx = i * features_per_quantile
            end_idx = (i + 1) * features_per_quantile
            q_predictions = pred_stack[:, :, :, start_idx:end_idx]
            
            q_mean = torch.mean(q_predictions, dim=0)
            q_std = torch.std(q_predictions, dim=0)
            
            quantile_results[f'quantile_{q_level}'] = {
                'prediction': q_mean,
                'uncertainty': q_std,
                'certainty_score': self._compute_quantile_certainty(q_predictions, q_level)
            }
            
        return {'quantile_specific': quantile_results}
        
    def _compute_quantile_certainty(self, q_predictions, q_level):
        """Compute quantile certainty following existing pattern"""
        mean_pred = torch.mean(q_predictions, dim=0)
        std_pred = torch.std(q_predictions, dim=0)
        cv = std_pred / (torch.abs(mean_pred) + 1e-8)
        certainty = 1.0 / (1.0 + cv)
        return certainty


# Factory functions for creating extensions
def create_bayesian_extension(configs, hf_backbone):
    """Factory function to create Bayesian extension"""
    return BayesianExtension(configs, hf_backbone)


def create_quantile_extension(configs):
    """Factory function to create quantile extension"""
    return QuantileExtension(configs)
