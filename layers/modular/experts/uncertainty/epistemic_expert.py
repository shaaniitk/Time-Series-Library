"""
Epistemic Uncertainty Expert

Specialized expert for modeling epistemic (model) uncertainty in time series.
Handles parameter uncertainty, model structure uncertainty, and knowledge limitations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import math

from ..base_expert import UncertaintyExpert, ExpertOutput


class BayesianLinear(nn.Module):
    """Bayesian linear layer for epistemic uncertainty."""
    
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # Weight parameters (mean and log variance)
        self.weight_mean = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_logvar = nn.Parameter(torch.randn(out_features, in_features) * 0.1 - 2)
        
        # Bias parameters
        self.bias_mean = nn.Parameter(torch.zeros(out_features))
        self.bias_logvar = nn.Parameter(torch.ones(out_features) * 0.1 - 2)
        
    def forward(self, x: torch.Tensor, sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with uncertainty sampling."""
        if sample and self.training:
            # Sample weights and biases
            weight_std = torch.exp(0.5 * self.weight_logvar)
            weight = self.weight_mean + weight_std * torch.randn_like(self.weight_mean)
            
            bias_std = torch.exp(0.5 * self.bias_logvar)
            bias = self.bias_mean + bias_std * torch.randn_like(self.bias_mean)
        else:
            # Use mean parameters
            weight = self.weight_mean
            bias = self.bias_mean
        
        output = F.linear(x, weight, bias)
        
        # Compute epistemic uncertainty (variance of output)
        if sample:
            weight_var = torch.exp(self.weight_logvar)
            bias_var = torch.exp(self.bias_logvar)
            
            # Propagate uncertainty through linear transformation
            output_var = torch.sum((x.unsqueeze(-2) ** 2) * weight_var.unsqueeze(0), dim=-1) + bias_var
            epistemic_uncertainty = torch.sqrt(output_var + 1e-8)
        else:
            epistemic_uncertainty = torch.zeros_like(output)
        
        return output, epistemic_uncertainty
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence for regularization."""
        # KL divergence between posterior and prior
        weight_kl = 0.5 * torch.sum(
            torch.exp(self.weight_logvar) + self.weight_mean ** 2 - self.weight_logvar - 1
        )
        
        bias_kl = 0.5 * torch.sum(
            torch.exp(self.bias_logvar) + self.bias_mean ** 2 - self.bias_logvar - 1
        )
        
        return weight_kl + bias_kl


class EpistemicUncertaintyExpert(UncertaintyExpert):
    """Expert specialized in epistemic uncertainty quantification."""
    
    def __init__(self, config):
        super().__init__(config, 'epistemic_uncertainty_expert')
        
        # Bayesian layers for parameter uncertainty
        self.bayesian_layers = nn.ModuleList([
            BayesianLinear(self.d_model, self.expert_dim),
            BayesianLinear(self.expert_dim, self.expert_dim),
            BayesianLinear(self.expert_dim, self.expert_dim)
        ])
        
        # Model ensemble for structural uncertainty
        self.ensemble_size = getattr(config, 'epistemic_ensemble_size', 5)
        self.model_ensemble = nn.ModuleList()
        for _ in range(self.ensemble_size):
            ensemble_model = nn.Sequential(
                nn.Linear(self.d_model, self.expert_dim),
                nn.ReLU(),
                nn.Linear(self.expert_dim, self.expert_dim // 2),
                nn.ReLU(),
                nn.Linear(self.expert_dim // 2, 1)
            )
            self.model_ensemble.append(ensemble_model)
        
        # Dropout layers for Monte Carlo dropout
        self.mc_dropout_layers = nn.ModuleList([
            nn.Dropout(0.1),
            nn.Dropout(0.2),
            nn.Dropout(0.1)
        ])
        
        # Uncertainty aggregation
        self.uncertainty_aggregator = nn.Sequential(
            nn.Linear(self.expert_dim + self.ensemble_size, self.expert_dim),
            nn.ReLU(),
            nn.Linear(self.expert_dim, 1),
            nn.Softplus()
        )
        
        # Output projection
        self.output_proj = nn.Linear(1, self.expert_dim)
        
    def forward(self, x: torch.Tensor, **kwargs) -> ExpertOutput:
        """Process input through epistemic uncertainty expert."""
        batch_size, seq_len, d_model = x.shape
        
        # 1. Bayesian parameter uncertainty
        bayesian_outputs = []
        bayesian_uncertainties = []
        kl_losses = []
        
        current_input = x
        for layer, dropout in zip(self.bayesian_layers, self.mc_dropout_layers):
            output, uncertainty = layer(current_input, sample=True)
            output = F.relu(dropout(output))
            
            bayesian_outputs.append(output)
            bayesian_uncertainties.append(uncertainty)
            kl_losses.append(layer.kl_divergence())
            
            current_input = output
        
        # Aggregate Bayesian uncertainty
        bayesian_uncertainty = torch.stack(bayesian_uncertainties).mean(dim=0)
        total_kl_loss = sum(kl_losses)
        
        # 2. Model ensemble uncertainty
        ensemble_outputs = []
        for ensemble_model in self.model_ensemble:
            ensemble_out = ensemble_model(x)
            ensemble_outputs.append(ensemble_out)
        
        ensemble_outputs = torch.stack(ensemble_outputs, dim=-1)  # [batch, seq, 1, ensemble_size]
        ensemble_mean = ensemble_outputs.mean(dim=-1, keepdim=True)
        ensemble_variance = ensemble_outputs.var(dim=-1, keepdim=True)
        ensemble_uncertainty = torch.sqrt(ensemble_variance + 1e-8)
        
        # 3. Monte Carlo dropout uncertainty (already applied in Bayesian layers)
        mc_uncertainty = bayesian_uncertainty.mean(dim=-1, keepdim=True)
        
        # 4. Aggregate all epistemic uncertainties
        uncertainty_features = torch.cat([
            current_input,  # Bayesian features
            ensemble_outputs.squeeze(-2)  # Ensemble outputs
        ], dim=-1)
        
        aggregated_uncertainty = self.uncertainty_aggregator(uncertainty_features)
        
        # 5. Apply uncertainty normalization
        normalized_uncertainty = self.uncertainty_norm(aggregated_uncertainty)
        
        # 6. Output projection
        output = self.output_proj(normalized_uncertainty)
        
        # 7. Compute confidence (inverse of uncertainty)
        confidence = 1.0 / (1.0 + normalized_uncertainty)
        
        # 8. Compile metadata
        metadata = {
            'bayesian_uncertainty': bayesian_uncertainty.mean().item(),
            'ensemble_uncertainty': ensemble_uncertainty.mean().item(),
            'mc_dropout_uncertainty': mc_uncertainty.mean().item(),
            'kl_divergence': total_kl_loss.item(),
            'ensemble_disagreement': ensemble_variance.mean().item(),
            'parameter_uncertainty_ratio': (bayesian_uncertainty.mean() / (normalized_uncertainty.mean() + 1e-8)).item()
        }
        
        return ExpertOutput(
            output=output,
            confidence=confidence,
            metadata=metadata,
            attention_weights=None,
            uncertainty=normalized_uncertainty
        )
    
    def get_kl_loss(self) -> torch.Tensor:
        """Get total KL divergence loss for regularization."""
        total_kl = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.bayesian_layers:
            total_kl += layer.kl_divergence()
        return total_kl
    
    def monte_carlo_predict(self, x: torch.Tensor, num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform Monte Carlo prediction for uncertainty estimation."""
        self.train()  # Enable dropout
        
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.forward(x)
                predictions.append(pred.output)
        
        predictions = torch.stack(predictions)  # [num_samples, batch, seq, features]
        
        # Compute mean and uncertainty
        pred_mean = predictions.mean(dim=0)
        pred_uncertainty = predictions.std(dim=0)
        
        return pred_mean, pred_uncertainty