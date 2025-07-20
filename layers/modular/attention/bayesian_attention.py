"""
Bayesian Attention Components for Modular Autoformer

This module implements Bayesian neural network layers for attention mechanisms,
providing uncertainty quantification and robust attention computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base import BaseAttention
from utils.logger import logger


class BayesianLinear(nn.Module):
    """
    Bayesian linear layer with variational inference for uncertainty quantification.
    
    This layer maintains distributions over weights rather than point estimates,
    enabling uncertainty-aware predictions.
    """
    
    def __init__(self, in_features, out_features, prior_std=1.0):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # Mean parameters for weights and biases
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        
        # Log variance parameters (to ensure positivity)
        self.weight_log_var = nn.Parameter(torch.randn(out_features, in_features) * 0.01 - 3)
        self.bias_log_var = nn.Parameter(torch.zeros(out_features) - 3)
        
    def forward(self, x):
        """
        Forward pass with weight sampling from posterior distribution.
        
        Args:
            x: [B, *, in_features] input tensor
            
        Returns:
            output: [B, *, out_features] output tensor
        """
        if self.training:
            # Sample weights from posterior during training
            weight_std = torch.exp(0.5 * self.weight_log_var)
            bias_std = torch.exp(0.5 * self.bias_log_var)
            
            # Reparameterization trick
            weight_eps = torch.randn_like(self.weight_mu)
            bias_eps = torch.randn_like(self.bias_mu)
            
            weight = self.weight_mu + weight_std * weight_eps
            bias = self.bias_mu + bias_std * bias_eps
        else:
            # Use mean parameters during inference
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self):
        """
        Compute KL divergence between posterior and prior distributions.
        
        Returns:
            KL divergence scalar
        """
        # Prior distribution: N(0, prior_std^2)
        # Posterior distribution: N(mu, exp(log_var))
        
        weight_var = torch.exp(self.weight_log_var)
        bias_var = torch.exp(self.bias_log_var)
        
        # KL for weights
        weight_kl = 0.5 * torch.sum(
            (self.weight_mu ** 2 + weight_var) / (self.prior_std ** 2) 
            - 1 - self.weight_log_var + 2 * math.log(self.prior_std)
        )
        
        # KL for biases
        bias_kl = 0.5 * torch.sum(
            (self.bias_mu ** 2 + bias_var) / (self.prior_std ** 2) 
            - 1 - self.bias_log_var + 2 * math.log(self.prior_std)
        )
        
        return weight_kl + bias_kl


class BayesianAttention(BaseAttention):
    """
    Bayesian attention mechanism with uncertainty quantification.
    
    This attention mechanism uses Bayesian linear layers to provide
    uncertainty estimates along with attention weights.
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1, prior_std=1.0, 
                 temperature=1.0, output_attention=False):
        super(BayesianAttention, self).__init__()
        logger.info(f"Initializing BayesianAttention with uncertainty quantification: d_model={d_model}")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.temperature = temperature
        self.output_attention = output_attention
        
        # Bayesian projection layers
        self.w_qs = BayesianLinear(d_model, d_model, prior_std=prior_std)
        self.w_ks = BayesianLinear(d_model, d_model, prior_std=prior_std)
        self.w_vs = BayesianLinear(d_model, d_model, prior_std=prior_std)
        self.w_o = BayesianLinear(d_model, d_model, prior_std=prior_std)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Uncertainty scaling
        self.uncertainty_scale = nn.Parameter(torch.ones(1))
        
        self.output_dim_multiplier = 1
    
    def get_kl_divergence(self):
        """
        Get total KL divergence from all Bayesian layers.
        
        Returns:
            Total KL divergence
        """
        total_kl = (self.w_qs.kl_divergence() + 
                   self.w_ks.kl_divergence() + 
                   self.w_vs.kl_divergence() + 
                   self.w_o.kl_divergence())
        return total_kl
    
    def compute_attention_uncertainty(self, attn_weights):
        """
        Compute uncertainty in attention weights.
        
        Args:
            attn_weights: [B, H, L, L] attention weights
            
        Returns:
            uncertainty: [B, H, L, L] uncertainty estimates
        """
        # Entropy-based uncertainty
        entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=-1, keepdim=True)
        
        # Normalize entropy to [0, 1]
        max_entropy = math.log(attn_weights.size(-1))
        normalized_entropy = entropy / max_entropy
        
        # Scale by learned parameter
        uncertainty = self.uncertainty_scale * normalized_entropy
        
        return uncertainty.expand_as(attn_weights)
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Forward pass with Bayesian attention computation.
        
        Args:
            queries: [B, L, D] query tensor
            keys: [B, L, D] key tensor
            values: [B, L, D] value tensor
            attn_mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights, uncertainty)
        """
        B, L, D = queries.shape
        H = self.n_heads
        
        residual = queries
        
        # Apply Bayesian projections
        q = self.w_qs(queries).view(B, L, H, self.d_k).transpose(1, 2)  # [B, H, L, d_k]
        k = self.w_ks(keys).view(B, L, H, self.d_k).transpose(1, 2)     # [B, H, L, d_k]
        v = self.w_vs(values).view(B, L, H, self.d_k).transpose(1, 2)   # [B, H, L, d_k]
        
        # Scaled dot-product attention with temperature
        scores = torch.matmul(q, k.transpose(-1, -2)) / (math.sqrt(self.d_k) * self.temperature)
        
        # Apply attention mask if provided
        if attn_mask is not None:
            scores.masked_fill_(attn_mask == 0, -1e9)
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Compute uncertainty
        uncertainty = self.compute_attention_uncertainty(attn_weights)
        
        # Apply dropout
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)  # [B, H, L, d_k]
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        
        # Final projection
        output = self.w_o(context)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + residual)
        
        if self.output_attention:
            return output, attn_weights, uncertainty
        else:
            return output, None


class BayesianMultiHeadAttention(BaseAttention):
    """
    Multi-head Bayesian attention with ensemble predictions.
    
    This extends BayesianAttention with multiple sampling for better
    uncertainty estimates through Monte Carlo dropout.
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1, prior_std=1.0, 
                 n_samples=5, output_attention=False):
        super(BayesianMultiHeadAttention, self).__init__()
        logger.info(f"Initializing BayesianMultiHeadAttention with ensemble: samples={n_samples}")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_samples = n_samples
        self.output_attention = output_attention
        
        # Core Bayesian attention
        self.bayesian_attention = BayesianAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            prior_std=prior_std,
            output_attention=True
        )
        
        self.output_dim_multiplier = 1
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Forward pass with Monte Carlo sampling for uncertainty estimation.
        
        Args:
            queries: [B, L, D] query tensor
            keys: [B, L, D] key tensor
            values: [B, L, D] value tensor
            attn_mask: Optional attention mask
            
        Returns:
            Tuple of (mean_output, epistemic_uncertainty, aleatoric_uncertainty)
        """
        if self.training:
            # Single forward pass during training
            output, attn_weights, uncertainty = self.bayesian_attention(queries, keys, values, attn_mask)
            return output, attn_weights
        else:
            # Multiple forward passes during inference for uncertainty estimation
            outputs = []
            attention_weights = []
            uncertainties = []
            
            # Enable training mode temporarily for sampling
            self.bayesian_attention.train()
            
            with torch.no_grad():
                for _ in range(self.n_samples):
                    output, attn_w, uncert = self.bayesian_attention(queries, keys, values, attn_mask)
                    outputs.append(output)
                    if attn_w is not None:
                        attention_weights.append(attn_w)
                    if uncert is not None:
                        uncertainties.append(uncert)
            
            # Restore evaluation mode
            self.bayesian_attention.eval()
            
            # Compute statistics
            outputs_tensor = torch.stack(outputs, dim=0)  # [n_samples, B, L, D]
            mean_output = torch.mean(outputs_tensor, dim=0)
            
            # Epistemic uncertainty (model uncertainty)
            epistemic_uncertainty = torch.var(outputs_tensor, dim=0)
            
            # Aleatoric uncertainty (data uncertainty)
            if uncertainties:
                uncertainties_tensor = torch.stack(uncertainties, dim=0)
                aleatoric_uncertainty = torch.mean(uncertainties_tensor, dim=0)
            else:
                aleatoric_uncertainty = None
            
            # Average attention weights
            if attention_weights:
                mean_attention = torch.mean(torch.stack(attention_weights, dim=0), dim=0)
            else:
                mean_attention = None
            
            if self.output_attention:
                return mean_output, mean_attention, epistemic_uncertainty, aleatoric_uncertainty
            else:
                return mean_output, None


class VariationalAttention(BaseAttention):
    """
    Variational attention mechanism using variational inference.
    
    This implements attention with learnable variance parameters,
    allowing for uncertainty-aware attention computation.
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1, learn_variance=True):
        super(VariationalAttention, self).__init__()
        logger.info(f"Initializing VariationalAttention with learnable variance: d_model={d_model}")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.learn_variance = learn_variance
        
        # Standard projection layers
        self.w_qs = nn.Linear(d_model, d_model)
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Variance projection layers
        if learn_variance:
            self.w_qs_var = nn.Linear(d_model, d_model)
            self.w_ks_var = nn.Linear(d_model, d_model)
            self.w_vs_var = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.output_dim_multiplier = 1
    
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick for variational sampling.
        
        Args:
            mu: mean parameters
            log_var: log variance parameters
            
        Returns:
            sampled values
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Forward pass with variational attention.
        
        Args:
            queries: [B, L, D] query tensor
            keys: [B, L, D] key tensor
            values: [B, L, D] value tensor
            attn_mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        B, L, D = queries.shape
        H = self.n_heads
        
        residual = queries
        
        # Project to get means
        q_mu = self.w_qs(queries).view(B, L, H, self.d_k).transpose(1, 2)
        k_mu = self.w_ks(keys).view(B, L, H, self.d_k).transpose(1, 2)
        v_mu = self.w_vs(values).view(B, L, H, self.d_k).transpose(1, 2)
        
        if self.learn_variance:
            # Project to get variances
            q_log_var = self.w_qs_var(queries).view(B, L, H, self.d_k).transpose(1, 2)
            k_log_var = self.w_ks_var(keys).view(B, L, H, self.d_k).transpose(1, 2)
            v_log_var = self.w_vs_var(values).view(B, L, H, self.d_k).transpose(1, 2)
            
            # Sample using reparameterization trick
            q = self.reparameterize(q_mu, q_log_var)
            k = self.reparameterize(k_mu, k_log_var)
            v = self.reparameterize(v_mu, v_log_var)
        else:
            q, k, v = q_mu, k_mu, v_mu
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)
        
        # Apply attention mask if provided
        if attn_mask is not None:
            scores.masked_fill_(attn_mask == 0, -1e9)
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        
        # Final projection
        output = self.w_o(context)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + residual)
        
        return output, attn_weights


class BayesianCrossAttention(BaseAttention):
    """
    Bayesian cross-attention for encoder-decoder architectures.
    
    This implements cross-attention with Bayesian weights for uncertainty
    quantification in sequence-to-sequence tasks.
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1, prior_std=1.0):
        super(BayesianCrossAttention, self).__init__()
        logger.info(f"Initializing BayesianCrossAttention for encoder-decoder: d_model={d_model}")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Bayesian projections for queries (from decoder)
        self.w_qs = BayesianLinear(d_model, d_model, prior_std=prior_std)
        
        # Standard projections for keys and values (from encoder)
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)
        
        # Bayesian output projection
        self.w_o = BayesianLinear(d_model, d_model, prior_std=prior_std)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.output_dim_multiplier = 1
    
    def get_kl_divergence(self):
        """
        Get total KL divergence from Bayesian layers.
        
        Returns:
            Total KL divergence
        """
        return self.w_qs.kl_divergence() + self.w_o.kl_divergence()
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Forward pass with Bayesian cross-attention.
        
        Args:
            queries: [B, L_dec, D] decoder queries
            keys: [B, L_enc, D] encoder keys
            values: [B, L_enc, D] encoder values
            attn_mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        B, L_dec, D = queries.shape
        L_enc = keys.size(1)
        H = self.n_heads
        
        residual = queries
        
        # Project queries (Bayesian), keys and values (deterministic)
        q = self.w_qs(queries).view(B, L_dec, H, self.d_k).transpose(1, 2)
        k = self.w_ks(keys).view(B, L_enc, H, self.d_k).transpose(1, 2)
        v = self.w_vs(values).view(B, L_enc, H, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)
        
        # Apply attention mask if provided
        if attn_mask is not None:
            scores.masked_fill_(attn_mask == 0, -1e9)
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(B, L_dec, D)
        
        # Bayesian output projection
        output = self.w_o(context)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + residual)
        
        return output, attn_weights
