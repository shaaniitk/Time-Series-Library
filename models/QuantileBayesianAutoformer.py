#!/usr/bin/env python3
"""
Combined Bayesian Autoformer with KL Loss + Quantile Loss
Normalized contributions that sum to 1.0
"""

import torch
import torch.nn as nn
import numpy as np
from models.BayesianEnhancedAutoformer import BayesianEnhancedAutoformer
from utils.losses import get_loss_function
from utils.logger import logger

class QuantileBayesianAutoformer(BayesianEnhancedAutoformer):
    """
    Bayesian Autoformer that outputs quantile predictions with normalized KL + Quantile loss
    """
    
    def __init__(self, configs, quantiles=[0.1, 0.5, 0.9], kl_weight=0.3, 
                 uncertainty_method='bayesian', bayesian_layers=['projection']):
        """
        Args:
            configs: Model configuration
            quantiles: List of quantiles to predict (e.g., [0.1, 0.5, 0.9])
            kl_weight: Weight for KL loss (quantile_weight = 1 - kl_weight)
            uncertainty_method: Type of uncertainty modeling
            bayesian_layers: Which layers to make Bayesian
        """
        
        # Store quantile configuration before parent initialization
        self.quantiles = quantiles
        self.n_quantiles = len(quantiles)
        self.kl_weight = kl_weight
        self.quantile_weight = 1.0 - kl_weight  # Ensures they sum to 1
        
        logger.info(f"Initializing QuantileBayesianAutoformer with {self.n_quantiles} quantiles: {quantiles}")
        logger.info(f"Loss weights: KL={self.kl_weight:.3f}, Quantile={self.quantile_weight:.3f}")
        
        # Modify configs to output quantiles for each target
        original_c_out = configs.c_out
        configs.c_out = original_c_out * self.n_quantiles  # Expand output for quantiles
        
        # Store original target count for later use
        self.original_c_out = original_c_out
        
        # Initialize parent Bayesian model
        super(QuantileBayesianAutoformer, self).__init__(
            configs, 
            uncertainty_method=uncertainty_method,
            bayesian_layers=bayesian_layers,
            kl_weight=1.0  # We'll handle normalization in compute_loss
        )
        
        # Create quantile loss function
        self.quantile_criterion = get_loss_function('pinball', quantiles=self.quantiles)
        
        logger.info(f"Model will output {configs.c_out} values ({original_c_out} targets × {self.n_quantiles} quantiles)")
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        Forward pass returning quantile predictions
        
        Returns:
            torch.Tensor: [batch_size, seq_len, n_targets * n_quantiles]
                         Organized as [target1_q1, target1_q2, target1_q3, target2_q1, ...]
        """
        # Call parent forward - already outputs expanded dimensions
        output = super(QuantileBayesianAutoformer, self).forward(
            x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask, dec_self_mask, dec_enc_mask
        )
        
        return output  # [batch_size, seq_len, original_c_out * n_quantiles]
    
    def compute_loss(self, predictions, targets, fallback_criterion=None, return_components=False):
        """
        Compute normalized combined loss: KL + Quantile
        
        Args:
            predictions: Model outputs [batch_size, seq_len, n_targets * n_quantiles]  
            targets: Ground truth [batch_size, seq_len, n_targets]
            fallback_criterion: Unused (we use quantile loss)
            return_components: Whether to return loss breakdown
            
        Returns:
            torch.Tensor: Combined normalized loss
        """
        
        batch_size, seq_len, _ = predictions.shape
        
        # Reshape predictions to separate quantiles
        # From [batch, seq, n_targets * n_quantiles] to [batch, seq, n_targets, n_quantiles]
        pred_quantiles = predictions.view(batch_size, seq_len, self.original_c_out, self.n_quantiles)
        
        # Expand targets to match quantile structure
        # From [batch, seq, n_targets] to [batch, seq, n_targets, n_quantiles]
        targets_expanded = targets.unsqueeze(-1).expand(-1, -1, -1, self.n_quantiles)
        
        # Compute quantile loss
        quantile_loss = self.quantile_criterion(pred_quantiles, targets_expanded)
        
        # Compute KL divergence from Bayesian layers
        kl_loss = torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype)
        if hasattr(self, 'bayesian_layers_list') and self.bayesian_layers_list:
            for layer in self.bayesian_layers_list:
                if hasattr(layer, 'kl_divergence'):
                    kl_loss += layer.kl_divergence()
        
        # Normalize losses to sum to 1.0
        total_loss = self.quantile_weight * quantile_loss + self.kl_weight * kl_loss
        
        if return_components:
            return {
                'quantile_loss': quantile_loss,
                'kl_loss': kl_loss,
                'quantile_contribution': (self.quantile_weight * quantile_loss).item(),
                'kl_contribution': (self.kl_weight * kl_loss).item(),
                'total_loss': total_loss,
                'quantile_weight': self.quantile_weight,
                'kl_weight': self.kl_weight
            }
        
        return total_loss
    
    def predict_quantiles(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Get quantile predictions in organized format
        
        Returns:
            dict: {
                'quantiles': quantile values,
                'predictions': organized predictions [batch, seq, n_targets, n_quantiles],
                'lower_bound': lower quantile (e.g., 10%),
                'median': median quantile (50%),  
                'upper_bound': upper quantile (e.g., 90%)
            }
        """
        self.eval()
        with torch.no_grad():
            raw_output = self.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            batch_size, seq_len, _ = raw_output.shape
            
            # Reshape to [batch, seq, n_targets, n_quantiles]
            pred_quantiles = raw_output.view(batch_size, seq_len, self.original_c_out, self.n_quantiles)
            
            result = {
                'quantiles': self.quantiles,
                'predictions': pred_quantiles,
            }
            
            # Add convenience accessors for common quantiles
            if 0.1 in self.quantiles:
                result['lower_bound'] = pred_quantiles[:, :, :, self.quantiles.index(0.1)]
            if 0.5 in self.quantiles:
                result['median'] = pred_quantiles[:, :, :, self.quantiles.index(0.5)]
            if 0.9 in self.quantiles:
                result['upper_bound'] = pred_quantiles[:, :, :, self.quantiles.index(0.9)]
                
            return result
    
    def get_uncertainty_metrics(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Extract uncertainty metrics from predictions
        
        Returns:
            dict: Various uncertainty measures
        """
        quantile_results = self.predict_quantiles(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Prediction interval width (measure of uncertainty)
        if 'lower_bound' in quantile_results and 'upper_bound' in quantile_results:
            interval_width = quantile_results['upper_bound'] - quantile_results['lower_bound']
            
            return {
                'prediction_intervals': quantile_results,
                'interval_width': interval_width,
                'mean_interval_width': interval_width.mean().item(),
                'uncertainty_by_target': interval_width.mean(dim=(0,1)).cpu().numpy(),  # [n_targets]
                'uncertainty_by_timestep': interval_width.mean(dim=(0,2)).cpu().numpy()  # [seq_len]
            }
        
        return {'prediction_intervals': quantile_results}

# Factory function for easy creation
def create_quantile_bayesian_autoformer(configs, quantiles=[0.1, 0.5, 0.9], kl_weight=0.3):
    """
    Factory function to create QuantileBayesianAutoformer
    
    Args:
        configs: Model configuration 
        quantiles: Quantile levels to predict
        kl_weight: Weight for KL loss (0.0 to 1.0)
        
    Returns:
        QuantileBayesianAutoformer: Configured model
    """
    return QuantileBayesianAutoformer(
        configs=configs,
        quantiles=quantiles, 
        kl_weight=kl_weight,
        uncertainty_method='bayesian',
        bayesian_layers=['projection']
    )
