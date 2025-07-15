"""
HFBayesianAutoformer: Advanced Bayesian uncertainty quantification using Hugging Face Transformers
Based on HFEnhancedAutoformer with uncertainty estimation and quantile regression.

Original code from: https://github.com/thuml/Time-Series-Library
Enhanced for uncertainty quantification with temporal embedding support.
"""

import torch
import torch.nn as nn
import logging
from .HFEnhancedAutoformer import HFEnhancedAutoformer

logger = logging.getLogger(__name__)

class HFBayesianAutoformer(nn.Module):
    """
    Bayesian Autoformer with uncertainty quantification using Hugging Face Transformers.
    Integrates covariates through temporal embeddings and provides uncertainty estimates.
    """
    
    def __init__(self, configs):
        super(HFBayesianAutoformer, self).__init__()
        self.configs = configs
        
        # Bayesian parameters
        self.mc_samples = getattr(configs, 'mc_samples', 10)
        self.uncertainty_method = getattr(configs, 'uncertainty_method', 'mc_dropout')
        
        # Quantile regression setup
        if hasattr(configs, 'quantile_mode') and configs.quantile_mode:
            self.is_quantile_mode = True
            self.quantiles = getattr(configs, 'quantile_levels', [0.1, 0.25, 0.5, 0.75, 0.9])
        else:
            self.is_quantile_mode = False
            self.quantiles = []
        
        logger.info(f"Quantile mode: {self.is_quantile_mode}")
        if self.is_quantile_mode:
            logger.info(f"Quantiles: {self.quantiles}")
        
        # Base enhanced model (using Step 1 result)
        self.base_model = HFEnhancedAutoformer(configs)
        
        # Add temporal embedding for covariates (following HFEnhancedAutoformer pattern)
        from layers.Embed import TemporalEmbedding, TimeFeatureEmbedding
        
        # Choose embedding type based on config
        if configs.embed == 'timeF':
            self.temporal_embedding = TimeFeatureEmbedding(
                d_model=self.base_model.d_model, 
                embed_type=configs.embed, 
                freq=configs.freq
            )
        else:
            self.temporal_embedding = TemporalEmbedding(d_model=self.base_model.d_model)
        
        # Uncertainty-specific layers
        self.uncertainty_head = nn.Linear(self.base_model.d_model, configs.c_out)
        
        # Quantile heads if needed
        if self.is_quantile_mode and len(self.quantiles) > 0:
            self.quantile_heads = nn.ModuleList([
                nn.Linear(self.base_model.d_model, configs.c_out)
                for _ in self.quantiles
            ])
            logger.info(f"Created {len(self.quantiles)} quantile heads")
        
        # Monte Carlo dropout layers (SAFE - no gradient tracking issues)
        self.mc_dropout1 = nn.Dropout(0.1)
        self.mc_dropout2 = nn.Dropout(0.15)
        self.mc_dropout3 = nn.Dropout(0.1)
        
        logger.info("✅ HFBayesianAutoformer initialized successfully")
        logger.info(f"   Uncertainty method: {self.uncertainty_method}")
        logger.info(f"   Sampling strategy: {self.mc_samples} samples")
        logger.info(f"   Quantile support: {self.is_quantile_mode}")
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """Forward pass with Bayesian uncertainty quantification using covariates"""
        
        # Apply Monte Carlo dropout for uncertainty
        if self.training or self.mc_samples > 1:
            self.mc_dropout1.train()
            self.mc_dropout2.train() 
            self.mc_dropout3.train()
        
        # Encode temporal features from covariates (following HFEnhancedAutoformer pattern)
        if x_mark_enc is not None and x_mark_enc.size(-1) > 0:
            # Get temporal embeddings for encoder covariates
            enc_temporal_embed = self.temporal_embedding(x_mark_enc)
            
            # Integrate temporal embeddings with input (consistent with HFEnhancedAutoformer)
            if enc_temporal_embed.size(1) == x_enc.size(1):
                # If dimensions match, add temporal embeddings
                x_enc_enhanced = x_enc + enc_temporal_embed[:, :, :x_enc.size(-1)]
            else:
                # Otherwise, use original input
                x_enc_enhanced = x_enc
        else:
            x_enc_enhanced = x_enc
        
        # Use enhanced input for base model prediction
        base_output = self.base_model(x_enc_enhanced, x_mark_enc, x_dec, x_mark_dec, mask)
        
        # Apply MC dropout for uncertainty estimation
        hidden_state = self.mc_dropout1(base_output)
        
        # Get prediction and uncertainty
        prediction = self.mc_dropout2(hidden_state)
        uncertainty = torch.abs(self.uncertainty_head(self.mc_dropout3(hidden_state)))
        
        # For quantile mode, generate quantile predictions
        if self.is_quantile_mode and hasattr(self, 'quantile_heads'):
            quantile_outputs = []
            for quantile_head in self.quantile_heads:
                q_output = quantile_head(self.mc_dropout3(hidden_state))
                quantile_outputs.append(q_output)
        
        return prediction
        
    def get_uncertainty_estimates(self, x_enc, x_mark_enc, x_dec, x_mark_dec, n_samples=None):
        """Generate multiple samples for uncertainty estimation"""
        if n_samples is None:
            n_samples = self.mc_samples
            
        predictions = []
        uncertainties = []
        
        # Generate multiple forward passes with dropout
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
                predictions.append(pred)
        
        # Stack predictions and compute statistics
        predictions = torch.stack(predictions, dim=0)  # (n_samples, batch, seq, features)
        
        # Compute uncertainty metrics
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'samples': predictions,
            'confidence_intervals': {
                '68%': {
                    'lower': mean_pred - std_pred,
                    'upper': mean_pred + std_pred
                },
                '95%': {
                    'lower': mean_pred - 2*std_pred,
                    'upper': mean_pred + 2*std_pred
                }
            }
        }
        
    def get_uncertainty_result(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """Get full uncertainty quantification result"""
        with torch.no_grad():
            # Forward pass with covariate support
            prediction = self.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # Get hidden state for uncertainty computation
            if x_mark_enc is not None and x_mark_enc.size(-1) > 0:
                enc_temporal_embed = self.temporal_embedding(x_mark_enc)
                if enc_temporal_embed.size(1) == x_enc.size(1):
                    x_enc_enhanced = x_enc + enc_temporal_embed[:, :, :x_enc.size(-1)]
                else:
                    x_enc_enhanced = x_enc
            else:
                x_enc_enhanced = x_enc
            
            base_output = self.base_model(x_enc_enhanced, x_mark_enc, x_dec, x_mark_dec)
            hidden_state = self.mc_dropout1(base_output)
            
            uncertainty = torch.abs(self.uncertainty_head(self.mc_dropout3(hidden_state)))
            
            # Generate confidence intervals
            confidence_intervals = {
                "68%": {
                    'lower': prediction - uncertainty,
                    'upper': prediction + uncertainty,
                    'width': 2 * uncertainty
                },
                "95%": {
                    'lower': prediction - 2 * uncertainty,
                    'upper': prediction + 2 * uncertainty,
                    'width': 4 * uncertainty
                }
            }
            
            # Generate quantiles if available
            quantiles = {}
            if self.is_quantile_mode and hasattr(self, 'quantile_heads'):
                for i, q in enumerate(self.quantiles):
                    quantiles[f"q{int(q*100)}"] = self.quantile_heads[i](self.mc_dropout3(hidden_state))
                
            return {
                'prediction': prediction,
                'uncertainty': uncertainty,
                'confidence_intervals': confidence_intervals,
                'quantiles': quantiles
            }
