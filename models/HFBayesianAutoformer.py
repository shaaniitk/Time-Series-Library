"""
HFBayesianAutoformer: Advanced Bayesian uncertainty quantification using Hugging Face Transformers
Based on HFEnhancedAutoformer with uncertainty estimation and quantile regression.

Original code from: https://github.com/thuml/Time-Series-Library
Enhanced for uncertainty quantification with temporal embedding support.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, List
from .HFEnhancedAutoformer import HFEnhancedAutoformer

logger = logging.getLogger(__name__)

class HFBayesianAutoformer(nn.Module):
    """
    Bayesian Autoformer with uncertainty quantification using Hugging Face Transformers.
    Integrates covariates through temporal embeddings and provides uncertainty estimates.
    """
    
    def __init__(self, configs):
        super().__init__()
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
        self.base_model = HFEnhancedAutoformer(configs) # HFEnhancedAutoformer should have its own projection layer
        
        # Add temporal embedding for covariates (following HFEnhancedAutoformer pattern)
        from layers.Embed import TemporalEmbedding, TimeFeatureEmbedding
        
        # Choose embedding type based on config
        if configs.embed == 'timeF':
            self.covariate_embedding = TimeFeatureEmbedding(
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
                for _ in self.quantiles # One head per quantile
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
        
    def _get_enhanced_input(self, x_enc: torch.Tensor, x_mark_enc: Optional[torch.Tensor]) -> torch.Tensor:
        """Applies temporal embedding to the input sequence."""
        if x_mark_enc is not None and x_mark_enc.size(-1) > 0:
            # Get temporal embeddings for encoder covariates
            enc_temporal_embed = self.covariate_embedding(x_mark_enc)
            
            # Integrate temporal embeddings with input (consistent with HFEnhancedAutoformer)
            if enc_temporal_embed.size(1) == x_enc.size(1):
                # If dimensions match, add temporal embeddings
                return x_enc + enc_temporal_embed[:, :, :x_enc.size(-1)]
            else:
                # Otherwise, use original input
                logger.warning("Temporal embedding size mismatch. Using original input.")
                return x_enc
        return x_enc
        
    def forward(self, x_enc: torch.Tensor, x_mark_enc: Optional[torch.Tensor], 
                x_dec: torch.Tensor, x_mark_dec: Optional[torch.Tensor], 
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with Bayesian uncertainty quantification using covariates.
        Returns a dictionary of tensors for prediction, uncertainty, and quantiles.
        """
        
        # During inference for uncertainty estimation, dropout should be active.
        is_mc_inference = not self.training and self.mc_samples > 1
        if self.training or is_mc_inference:
            self.mc_dropout1.train()
            self.mc_dropout2.train() 
            self.mc_dropout3.train()
        
        x_enc_enhanced = self._get_enhanced_input(x_enc, x_mark_enc)
        
        # Use enhanced input for base model prediction
        base_output = self.base_model(x_enc_enhanced, x_mark_enc, x_dec, x_mark_dec, mask)
        
        # Apply MC dropout for uncertainty estimation
        hidden_state = self.mc_dropout1(base_output)

        # The main prediction is derived from the base model's output directly
        # or via a deterministic head if one were added. Here we use a dropout path.
        prediction = self.mc_dropout2(hidden_state) 

        # Aleatoric uncertainty prediction (learns to predict noise)
        # The absolute value ensures the uncertainty is non-negative.
        aleatoric_uncertainty = torch.abs(self.uncertainty_head(self.mc_dropout3(hidden_state)))
        
        outputs = {
            'prediction': prediction,
            'aleatoric_uncertainty': aleatoric_uncertainty
        }

        # Generate quantile predictions if in quantile mode
        if self.is_quantile_mode and hasattr(self, 'quantile_heads'):
            quantile_outputs = []
            for quantile_head in self.quantile_heads:
                # Each head gets a slightly different view of the hidden state due to dropout
                q_output = quantile_head(self.mc_dropout3(hidden_state))
                quantile_outputs.append(q_output)
            outputs['quantiles'] = torch.stack(quantile_outputs, dim=-1) # [B, L, C, num_quantiles]

        return outputs
        
    def get_uncertainty_estimates(self, x_enc: torch.Tensor, x_mark_enc: Optional[torch.Tensor], 
                                  x_dec: torch.Tensor, x_mark_dec: Optional[torch.Tensor], 
                                  n_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate multiple samples for uncertainty estimation and compute statistics.
        This is the primary method for obtaining uncertainty results during inference.
        """
        if n_samples is None:
            n_samples = self.mc_samples
            
        with torch.no_grad():
            # Collect samples from multiple stochastic forward passes
            all_outputs: List[Dict[str, torch.Tensor]] = [
                self.forward(x_enc, x_mark_enc, x_dec, x_mark_dec) for _ in range(n_samples)
            ]

        # Stack the collected samples
        # Shape: [n_samples, batch, seq_len, features]
        pred_samples = torch.stack([out['prediction'] for out in all_outputs], dim=0)
        aleatoric_samples = torch.stack([out['aleatoric_uncertainty'] for out in all_outputs], dim=0)

        # --- Compute Statistics ---
        # Mean prediction is the final point forecast
        mean_prediction = pred_samples.mean(dim=0)
        
        # Epistemic uncertainty (model uncertainty) from prediction variance
        epistemic_uncertainty = pred_samples.std(dim=0)
        
        # Aleatoric uncertainty (data uncertainty)
        mean_aleatoric_uncertainty = aleatoric_samples.mean(dim=0)

        # Total uncertainty combines both epistemic and aleatoric
        total_uncertainty = torch.sqrt(epistemic_uncertainty.pow(2) + mean_aleatoric_uncertainty.pow(2))

        results = {
            'prediction': mean_prediction,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': mean_aleatoric_uncertainty,
            'total_uncertainty': total_uncertainty,
            'samples': pred_samples,
            'confidence_intervals': {
                '68%': {'lower': mean_prediction - total_uncertainty, 'upper': mean_prediction + total_uncertainty},
                '95%': {'lower': mean_prediction - 2 * total_uncertainty, 'upper': mean_prediction + 2 * total_uncertainty}
            }
        }

        # Process quantiles if they were generated
        if 'quantiles' in all_outputs[0]:
            # Shape: [n_samples, B, L, C, num_quantiles]
            quantile_samples = torch.stack([out['quantiles'] for out in all_outputs], dim=0)
            # Average the quantile predictions across samples
            mean_quantiles = quantile_samples.mean(dim=0) # Shape: [B, L, C, num_quantiles]
            
            # Create a dictionary mapping quantile level to its tensor
            quantile_dict = {
                f"q{int(q*100)}": mean_quantiles[..., i] for i, q in enumerate(self.quantiles)
            }
            results['quantiles'] = quantile_dict

        return results
