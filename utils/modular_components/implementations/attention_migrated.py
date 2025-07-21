"""
Migrated Attention Components
Auto-migrated from layers/modular/attention to utils.modular_components.implementations
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
import logging

# Import base interfaces from new framework
from ..base_interfaces import BaseComponent, BaseLoss, BaseAttention
from ..config_schemas import ComponentConfig

# Import AutoCorrelationLayer from layers module
try:
    from .layers import AutoCorrelationLayer
    AUTOCORRELATION_LAYER_AVAILABLE = True
except ImportError:
    # Create a placeholder if not available
    class AutoCorrelationLayer:
        def __init__(self, *args, **kwargs):
            raise ImportError("AutoCorrelationLayer not available - please ensure layers.py is implemented correctly")
    AUTOCORRELATION_LAYER_AVAILABLE = False

logger = logging.getLogger(__name__)

# Essential imports for self-contained implementation
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple

# Migrated Classes
class MetaLearningAdapter(BaseAttention):
    """
    MAML-style Meta-Learning Adapter with proper gradient-based fast adaptation.
    
    Implements Model-Agnostic Meta-Learning (MAML) for rapid adaptation to new
    time series patterns using support sets and gradient-based inner loop optimization.
    """
    
    def __init__(self, d_model, n_heads=None, adaptation_steps=5, meta_lr=0.01, inner_lr=0.1, dropout=0.1):
        super(MetaLearningAdapter, self).__init__()
        logger.info(f"Initializing MAML MetaLearningAdapter: adaptation_steps={adaptation_steps}")
        
        self.adaptation_steps = adaptation_steps
        self.d_model = d_model
        self.inner_lr = inner_lr
        self.output_dim_multiplier = 1
        
        # Base network for meta-learning (theta parameters)
        self.base_network = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Meta-learning rate (learnable)
        self.meta_lr = nn.Parameter(torch.tensor(meta_lr))
        
        # Task context encoder for support set analysis
        self.context_encoder = nn.Sequential(
            nn.Linear(2 * d_model, d_model // 2),  # Accept 2*d_model input (mean + std)
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4)
        )
        
        # Context projection: maps d_model//4 -> d_model for addition with query
        self.context_projector = nn.Linear(d_model // 4, d_model)
        
        # Task-specific adaptation parameters
        self.task_attention = nn.MultiheadAttention(d_model, n_heads or 4, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)

    def extract_task_context(self, support_set):
        """
        Extract task-specific features from support set.
        
        Args:
            support_set: [B, S, D] support examples
            
        Returns:
            context: [B, D//4] task context vector
        """
        # Aggregate support set statistics
        support_mean = torch.mean(support_set, dim=1)  # [B, D]
        support_std = torch.std(support_set, dim=1)    # [B, D]
        
        # Encode task context - concatenate mean and std
        context_features = torch.cat([support_mean, support_std], dim=-1)  # [B, 2*D]
        
        # Project to context space (D//4 dimensions)
        context = self.context_encoder(context_features)  # [B, D//4]
        return context

    def fast_adaptation(self, support_set, query_set, num_steps=None):
        """
        Perform MAML-style fast adaptation using support set.
        
        Args:
            support_set: [B, S, D] support examples for adaptation
            query_set: [B, Q, D] query examples to adapt to
            num_steps: Number of adaptation steps (default: self.adaptation_steps)
            
        Returns:
            adapted_params: Dictionary of adapted parameters
            adaptation_loss: Loss during adaptation process
        """
        if num_steps is None:
            num_steps = self.adaptation_steps
            
        # Extract base parameters
        base_params = {}
        for name, param in self.base_network.named_parameters():
            base_params[name] = param
        
        adapted_params = {name: param.clone() for name, param in base_params.items()}
        total_adaptation_loss = 0.0
        
        # Fast adaptation loop (inner loop of MAML)
        for step in range(num_steps):
            # Forward pass with current adapted parameters
            adapted_output = self._forward_with_params(support_set, adapted_params)
            
            # Compute adaptation loss (task-specific loss on support set)
            adaptation_loss = F.mse_loss(adapted_output, support_set)
            total_adaptation_loss += adaptation_loss.item()
            
            # Compute gradients w.r.t. adapted parameters
            gradients = torch.autograd.grad(
                adaptation_loss, 
                adapted_params.values(), 
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )
            
            # Update adapted parameters using gradient descent
            for (name, param), grad in zip(adapted_params.items(), gradients):
                if grad is not None:
                    adapted_params[name] = param - self.inner_lr * grad
        
        return adapted_params, total_adaptation_loss / num_steps

    def _forward_with_params(self, x, params):
        """
        Forward pass using specified parameters instead of self.parameters()
        
        Args:
            x: [B, L, D] input tensor
            params: Dictionary of parameters to use
            
        Returns:
            output: [B, L, D] network output
        """
        # Manual forward pass through the network layers
        current = x
        
        # Get parameter names in order
        param_names = list(params.keys())
        weight_bias_pairs = []
        
        # Group weights and biases
        for i in range(0, len(param_names), 2):
            if i + 1 < len(param_names):
                weight_name = param_names[i] if 'weight' in param_names[i] else param_names[i+1]
                bias_name = param_names[i+1] if 'bias' in param_names[i+1] else param_names[i]
                weight_bias_pairs.append((weight_name, bias_name))
        
        # Apply linear layers with specified parameters
        for i, (weight_name, bias_name) in enumerate(weight_bias_pairs):
            current = F.linear(current, params[weight_name], params.get(bias_name))
            if i < len(weight_bias_pairs) - 1:  # Apply ReLU except for last layer
                current = F.relu(current)
        
        return current

    def forward(self, query, key, value, attn_mask=None, support_set=None):
        """
        Forward pass with MAML-style fast adaptation.
        
        Args:
            query: [B, L, D] main query tensor
            key: [B, L, D] key tensor (can be used as support_set)
            value: [B, L, D] value tensor  
            attn_mask: Optional attention mask
            support_set: [B, S, D] optional explicit support set
            
        Returns:
            Tuple of (adapted_output, attention_weights)
        """
        # Use key as support set if not provided explicitly
        if support_set is None:
            support_set = key
        
        # Store original query for residual connection
        residual = query
        
        if self.training and support_set is not None:
            # Extract task context from support set
            task_context = self.extract_task_context(support_set)  # [B, D//4]
            
            # Perform fast adaptation using MAML
            adapted_params, adaptation_loss = self.fast_adaptation(support_set, query)
            
            # Apply adapted network to query
            adapted_output = self._forward_with_params(query, adapted_params)
            
            # Task-specific attention using context
            # Project context to match query dimensions for addition
            context_projected = self.context_projector(task_context)  # [B, D]
            context_expanded = context_projected.unsqueeze(1).expand(-1, query.shape[1], -1)  # [B, L, D]
            query_with_context = query + context_expanded
            
            # Apply task attention
            attended_output, attention_weights = self.task_attention(
                query_with_context, adapted_output, adapted_output, attn_mask=attn_mask
            )
            
            # Combine adapted and attended outputs
            final_output = (adapted_output + attended_output) / 2
            
        else:
            # Standard forward pass without adaptation (inference mode)
            base_output = self.base_network(query)
            final_output, attention_weights = self.task_attention(query, base_output, base_output, attn_mask=attn_mask)
        
        # Apply dropout and residual connection
        final_output = self.dropout(final_output)
        final_output = final_output + residual
        
        return final_output, attention_weights if 'attention_weights' in locals() else None

class AdaptiveMixture(BaseAttention):
    """
    Adaptive mixture of experts for different time series patterns.
    Adapted to the BaseAttention interface.
    """
    
    def __init__(self, d_model, n_heads=None, mixture_components=4, gate_hidden_dim=None, dropout=0.1, temperature=1.0):
        super(AdaptiveMixture, self).__init__()
        logger.info(f"Initializing AdaptiveMixture: components={mixture_components}")
        
        self.num_experts = mixture_components
        self.d_model = d_model
        self.output_dim_multiplier = 1
        
        if gate_hidden_dim is None:
            gate_hidden_dim = d_model // 2
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
                nn.Linear(d_model * 2, d_model)
            ) for _ in range(self.num_experts)
        ])
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(d_model, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, self.num_experts),
            nn.Softmax(dim=-1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.temperature = temperature

    def forward(self, query, key, value, attn_mask=None):
        """
        Forward pass for adaptive mixture.
        'query' is the main input tensor.
        """
        x = query
        
        # Compute gating weights with temperature
        gate_weights = self.gate(x) / self.temperature  # [B, L, num_experts]
        
        # Compute expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        
        expert_outputs = torch.stack(expert_outputs, dim=-1)  # [B, L, d_model, num_experts]
        
        # Weighted combination
        output = torch.sum(expert_outputs * gate_weights.unsqueeze(-2), dim=-1)
        
        return self.dropout(output), None
class BaseAttention(nn.Module):
    """
    Base class for all attention components.
    This is not an abstract class, as we are reusing components
    """
    def __init__(self):
        super(BaseAttention, self).__init__()

    def forward(self, 
                query: nn.Module, 
                key: nn.Module, 
                value: nn.Module, 
                attn_mask: Optional[nn.Module] = None) -> Tuple[nn.Module, Optional[nn.Module]]:
        
        raise NotImplementedError("This method should be implemented by subclasses.")

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

class CrossResolutionAttention(BaseAttention):
    """
    Cross-resolution attention using existing MultiWaveletCross components.
    """
    
    def __init__(self, d_model, n_levels, use_multiwavelet=True, n_heads=8):
        super(CrossResolutionAttention, self).__init__()
        logger.info(f"Initializing CrossResolutionAttention for {n_levels} levels")
        
        self.n_levels = n_levels
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = d_model // n_heads
        self.use_multiwavelet = use_multiwavelet
        
        if not self.use_multiwavelet:
            # Use standard multi-head attention
            self.cross_res_attention = nn.ModuleList([
                nn.MultiheadAttention(d_model, n_heads, batch_first=True)
                for _ in range(n_levels - 1)
            ])
        
        # Cross-resolution projection layers
        self.cross_projections = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_levels - 1)
        ])
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Standard attention interface for compatibility.
        
        Args:
            queries: Query tensor
            keys: Key tensor  
            values: Value tensor
            attn_mask: Attention mask (optional, ignored for now)
            tau: Temperature parameter (optional, ignored)
            delta: Delta parameter (optional, ignored)
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # For now, just apply standard attention since this is being called in regular flow
        # TODO: Implement proper cross-resolution logic when multi-resolution data is available
        
        batch_size, seq_len_q, d_model = queries.shape
        _, seq_len_k, _ = keys.shape
        
        # Simple attention mechanism for compatibility
        scale = (d_model // self.n_heads) ** -0.5
        
        # Reshape for multi-head attention
        q = queries.view(batch_size, seq_len_q, self.n_heads, d_model // self.n_heads).transpose(1, 2)
        k = keys.view(batch_size, seq_len_k, self.n_heads, d_model // self.n_heads).transpose(1, 2)
        v = values.view(batch_size, seq_len_k, self.n_heads, d_model // self.n_heads).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, -1e9)
            
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len_q, d_model)
        
        return out, attn_weights
    
    def forward_multi_resolution(self, multi_res_features):
        """
        Apply cross-resolution attention between different scales.
        
        Args:
            multi_res_features: List of tensors at different resolutions
            
        Returns:
            List of cross-attended features
        """
        if len(multi_res_features) < 2:
            return multi_res_features, None
        
        
        attended_features = []
        
        for i in range(len(multi_res_features) - 1):
            coarse_features = multi_res_features[i]     # Lower resolution
            fine_features = multi_res_features[i + 1]   # Higher resolution
            
            # Align feature dimensions
            aligned_fine = self._align_features(fine_features, coarse_features)
            
            if self.use_multiwavelet:
                # Use existing MultiWaveletCross
                try:
                    cross_attended = self._apply_multiwavelet_cross(
                        coarse_features, aligned_fine, i
                    )
                except (AttributeError, RuntimeError) as e:
                    logger.error(f"MultiWavelet cross-attention failed with a critical error: {e}", exc_info=True)
                    cross_attended = self._apply_standard_attention(
                        coarse_features, aligned_fine, i
                    )
            else:
                cross_attended = self._apply_standard_attention(
                    coarse_features, aligned_fine, i
                )
            
            # Apply cross-resolution projection
            cross_attended = self.cross_projections[i](cross_attended)
            attended_features.append(cross_attended)
        
        # Add the finest scale without cross-attention
        attended_features.append(multi_res_features[-1])
        
        return attended_features, None
    
    def _align_features(self, source, target):
        """Align source features to target resolution"""
        if source.size(1) == target.size(1):
            return source
        
        # Resize to match target sequence length
        source_aligned = F.interpolate(
            source.transpose(1, 2),
            size=target.size(1),
            mode='linear',
            align_corners=False
        ).transpose(1, 2)
        
        return source_aligned
    
    def _apply_multiwavelet_cross(self, query_features, key_value_features, layer_idx):
        """Apply MultiWaveletCross attention with proper 4D format."""
        batch_size, seq_len_q, d_model = query_features.shape
        _, seq_len_kv, _ = key_value_features.shape
        
        attention_layer = MultiWaveletCross(
            in_channels=self.head_dim,
            out_channels=self.head_dim,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            modes=min(32, self.head_dim // 2),
            c=64,
            k=8,
            ich=d_model,
            base='legendre',
            activation='tanh'
        ).to(query_features.device)

        # Reshape to 4D format: (B, N, H, E)
        query_4d = query_features.view(batch_size, seq_len_q, self.n_heads, self.head_dim)
        key_4d = key_value_features.view(batch_size, seq_len_kv, self.n_heads, self.head_dim)
        value_4d = key_4d  # Use same as key for cross-attention style
        
        attended, _ = attention_layer(
            query_4d, key_4d, value_4d, seq_len_q, seq_len_kv
        )
        
        # Reshape back to 3D: (B, N, D)
        attended = attended.view(batch_size, seq_len_q, d_model)
        
        return attended

class EnhancedAutoCorrelation(BaseAttention):
    """
    Enhanced AutoCorrelation with adaptive window selection and multi-scale analysis.
    
    Key improvements over standard AutoCorrelation:
    1. Adaptive top-k selection based on correlation energy
    2. Multi-scale correlation analysis
    3. Learnable frequency filtering
    4. Numerical stability enhancements
    """

    def __init__(self, d_model, n_heads, factor=1, scale=None, attention_dropout=0.1, 
                 output_attention=False, adaptive_k=True, multi_scale=True, 
                 scales=[1, 2, 4], eps=1e-8):
        super(EnhancedAutoCorrelation, self).__init__()
        logger.info(f"Initializing EnhancedAutoCorrelation with enhanced features: d_model={d_model}")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
        # Adaptive parameters
        self.adaptive_k = adaptive_k
        self.multi_scale = multi_scale
        self.scales = scales
        self.eps = eps
        
        # Learnable components
        if self.multi_scale:
            self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))
            
        # Frequency filter for noise reduction
        self.frequency_filter = nn.Parameter(torch.ones(1))
        
        # Projection layers
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        
        self.output_dim_multiplier = 1

    def _select_adaptive_k(self, corr_energy, length):
        """
        Intelligently select the number of correlation peaks to use.
        
        Args:
            corr_energy: [B, L] correlation energy per time lag
            length: sequence length
            
        Returns:
            optimal k value
        """
        # Sort correlation energies in descending order
        sorted_energies, _ = torch.sort(corr_energy, dim=-1, descending=True)
        
        # Find the elbow point using second derivative
        if length > 10:  # Only for reasonable sequence lengths
            # Compute first and second derivatives
            first_diff = sorted_energies[:, :-1] - sorted_energies[:, 1:]
            second_diff = first_diff[:, :-1] - first_diff[:, 1:]
            
            # Find elbow as point of maximum curvature
            elbow_candidates = torch.argmax(second_diff, dim=-1) + 2  # +2 due to double differencing
            
            # Ensure reasonable bounds
            min_k = max(2, int(0.1 * math.log(length)))
            max_k = min(int(0.3 * length), int(self.factor * math.log(length) * 2))
            
            if min_k > max_k:
                max_k = min_k
            
            adaptive_k = torch.clamp(elbow_candidates, min_k, max_k)
            
            # Use median across batch for stability
            return int(torch.median(adaptive_k.float()).item())
        else:
            return max(2, int(self.factor * math.log(length)))

    def _multi_scale_correlation(self, queries, keys, length):
        """
        Compute correlations at multiple scales to capture different periodicities.
        
        Args:
            queries, keys: input tensors [B, H, L, E]
            length: sequence length
            
        Returns:
            aggregated multi-scale correlations
        """
        correlations = []
        B, H, L, E = queries.shape
        
        for i, scale in enumerate(self.scales):
            if scale == 1:
                # Original scale - ensure contiguous tensors
                q_input = queries.contiguous()
                k_input = keys.contiguous()
                q_fft = torch.fft.rfft(q_input, dim=-2)  # FFT along sequence dimension
                k_fft = torch.fft.rfft(k_input, dim=-2)
                
            else:
                # Downsample for multi-scale analysis - handle 4D properly
                target_len = max(length // scale, 1)
                q_downsampled = torch.zeros(B, H, target_len, E, device=queries.device)
                k_downsampled = torch.zeros(B, H, target_len, E, device=keys.device)
                
                # Process each head separately for F.interpolate compatibility
                for h in range(H):
                    for e in range(E):
                        q_downsampled[:, h, :, e] = F.interpolate(
                            queries[:, h, :, e].unsqueeze(1), size=target_len, 
                            mode='linear', align_corners=False
                        ).squeeze(1)
                        k_downsampled[:, h, :, e] = F.interpolate(
                            keys[:, h, :, e].unsqueeze(1), size=target_len, 
                            mode='linear', align_corners=False
                        ).squeeze(1)
                
                q_fft = torch.fft.rfft(q_downsampled, dim=-2)
                k_fft = torch.fft.rfft(k_downsampled, dim=-2)
            
            # Compute correlation in frequency domain
            correlation = q_fft * torch.conj(k_fft)
            
            # Apply learnable frequency filtering
            if hasattr(self, 'frequency_filter'):
                correlation = correlation * self.frequency_filter
            
            # Transform back to time domain
            correlation_time = torch.fft.irfft(correlation, n=length if scale == 1 else target_len, dim=-2)
            
            # Upsample back to original length if needed
            if scale != 1:
                correlation_upsampled = torch.zeros(B, H, length, E, device=correlation_time.device)
                for h in range(H):
                    for e in range(E):
                        correlation_upsampled[:, h, :, e] = F.interpolate(
                            correlation_time[:, h, :, e].unsqueeze(1), size=length,
                            mode='linear', align_corners=False
                        ).squeeze(1)
                correlation_time = correlation_upsampled
            
            correlations.append(correlation_time)
        
        # Weighted combination of multi-scale correlations
        if self.multi_scale and len(correlations) > 1:
            weights = F.softmax(self.scale_weights, dim=0)
            correlation_combined = sum(w * corr for w, corr in zip(weights, correlations))
        else:
            correlation_combined = correlations[0]
        
        return correlation_combined

    def _correlation_based_attention(self, queries, keys, values, length):
        """
        Compute attention weights based on autocorrelation.
        
        Args:
            queries: [B, H, L, E] query tensor
            keys: [B, H, L, E] key tensor  
            values: [B, H, L, E] value tensor
            length: sequence length
            
        Returns:
            Tuple of (output, attention_weights)
        """
        B, H, L, E = queries.shape
        
        # Compute multi-scale correlations
        correlation = self._multi_scale_correlation(queries, keys, length)
        
        # Compute correlation energy for adaptive k selection
        correlation_energy = torch.mean(torch.abs(correlation), dim=[1, 3])  # [B, L]
        
        # Select adaptive k
        if self.adaptive_k:
            k = self._select_adaptive_k(correlation_energy, length)
        else:
            k = int(self.factor * math.log(length))
        
        k = max(k, 1)  # Ensure k is at least 1
        
        # Find top-k correlations
        mean_correlation = torch.mean(correlation, dim=1)  # [B, L, E]
        _, top_k_indices = torch.topk(torch.mean(torch.abs(mean_correlation), dim=-1), k, dim=-1)
        
        # Initialize output
        output = torch.zeros_like(values)
        weights = torch.zeros(B, H, L, L, device=queries.device)
        
        # Apply correlation-based attention
        for b in range(B):
            for h in range(H):
                for i, lag in enumerate(top_k_indices[b]):
                    lag = lag.item()
                    
                    # Circular shift for autocorrelation
                    if lag > 0:
                        shifted_values = torch.cat([
                            values[b, h, -lag:, :], 
                            values[b, h, :-lag, :]
                        ], dim=0)
                    else:
                        shifted_values = values[b, h, :, :]
                    
                    # Compute attention weight based on correlation strength
                    corr_strength = torch.abs(mean_correlation[b, lag, :]).mean()
                    weight = F.softmax(torch.tensor([corr_strength]), dim=0)[0]
                    
                    output[b, h, :, :] += weight * shifted_values
                    
                    # Store attention weights for visualization
                    if lag > 0:
                        weights[b, h, :, :] += weight * torch.eye(L, device=queries.device).roll(-lag, dims=0)
                    else:
                        weights[b, h, :, :] += weight * torch.eye(L, device=queries.device)
        
        return output, weights

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Forward pass of enhanced autocorrelation attention.
        
        Args:
            queries: [B, L, D] or [B, H, L, E] query tensor
            keys: [B, L, D] or [B, H, L, E] key tensor
            values: [B, L, D] or [B, H, L, E] value tensor
            attn_mask: Optional attention mask
            tau: Temperature parameter (optional)
            delta: Delta parameter (optional)
            
        Returns:
            Tuple of (output, attention_weights)
        """
        
        # Handle different input shapes
        if queries.dim() == 4:
            # Input is [B, H, L, E] - reshape to [B, L, D]
            B, H, L, E = queries.shape
            D = H * E
            queries = queries.transpose(1, 2).contiguous().view(B, L, D)
            keys = keys.transpose(1, 2).contiguous().view(B, L, D)
            values = values.transpose(1, 2).contiguous().view(B, L, D)
        else:
            # Input is [B, L, D]
            B, L, D = queries.shape
            H = self.n_heads
            E = D // H
        
        # Apply projections
        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)
        
        # Reshape for multi-head processing
        queries = queries.view(B, L, H, E).transpose(1, 2)  # [B, H, L, E]
        keys = keys.view(B, L, H, E).transpose(1, 2)        # [B, H, L, E]
        values = values.view(B, L, H, E).transpose(1, 2)    # [B, H, L, E]
        
        # Apply correlation-based attention
        output, attn_weights = self._correlation_based_attention(queries, keys, values, L)
        
        # Apply dropout
        output = self.dropout(output)
        
        # Reshape back and apply output projection
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.out_projection(output)
        
        if self.output_attention:
            return output, attn_weights
        else:
            return output, None


class AdaptiveAutoCorrelationLayer(BaseAttention):
    """
    Layer wrapper for Enhanced AutoCorrelation with additional processing.
    
    This provides a complete layer interface compatible with transformer architectures.
    """
    
    def __init__(self, d_model, n_heads, factor=1, attention_dropout=0.1, 
                 output_attention=False, adaptive_k=True, multi_scale=True):
        super(AdaptiveAutoCorrelationLayer, self).__init__()
        logger.info(f"Initializing AdaptiveAutoCorrelationLayer: d_model={d_model}")
        
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Core autocorrelation mechanism
        self.autocorrelation = EnhancedAutoCorrelation(
            d_model=d_model,
            n_heads=n_heads,
            factor=factor,
            attention_dropout=attention_dropout,
            output_attention=output_attention,
            adaptive_k=adaptive_k,
            multi_scale=multi_scale
        )
        
        # Additional processing layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(attention_dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(attention_dropout)
        )
        
        self.output_dim_multiplier = 1
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Forward pass with residual connections and normalization.
        
        Args:
            queries: [B, L, D] input tensor
            keys: [B, L, D] key tensor (often same as queries)
            values: [B, L, D] value tensor (often same as queries)
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Self-attention with residual connection
        attn_output, attn_weights = self.autocorrelation(queries, keys, values, attn_mask, tau, delta)
        queries = self.norm1(queries + attn_output)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(queries)
        output = self.norm2(queries + ffn_output)
        
        return output, attn_weights


class HierarchicalAutoCorrelation(BaseAttention):
    """
    Hierarchical autocorrelation for capturing patterns at multiple time horizons.
    
    This component applies autocorrelation at different temporal resolutions
    and combines the results hierarchically.
    """
    
    def __init__(self, d_model, n_heads, hierarchy_levels=[1, 4, 16], factor=1):
        super(HierarchicalAutoCorrelation, self).__init__()
        logger.info(f"Initializing HierarchicalAutoCorrelation: levels={hierarchy_levels}")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.hierarchy_levels = hierarchy_levels
        
        # Autocorrelation for each hierarchy level
        self.level_correlations = nn.ModuleList([
            EnhancedAutoCorrelation(
                d_model=d_model,
                n_heads=n_heads,
                factor=factor,
                scales=[level]
            )
            for level in hierarchy_levels
        ])
        
        # Hierarchical fusion
        self.level_fusion = nn.Linear(d_model * len(hierarchy_levels), d_model)
        self.level_weights = nn.Parameter(torch.ones(len(hierarchy_levels)) / len(hierarchy_levels))
        
        self.output_dim_multiplier = 1
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Forward pass with hierarchical processing.
        
        Args:
            queries: [B, L, D] query tensor
            keys: [B, L, D] key tensor
            values: [B, L, D] value tensor
            
        Returns:
            Tuple of (output, attention_weights)
        """
        B, L, D = queries.shape
        
        level_outputs = []
        level_attentions = []
        
        # Process at each hierarchy level
        for level, correlation_layer in zip(self.hierarchy_levels, self.level_correlations):
            if level == 1:
                # Original resolution
                level_out, level_attn = correlation_layer(queries, keys, values, attn_mask)
            else:
                # Downsample for higher levels
                target_len = max(L // level, 1)
                q_down = F.interpolate(queries.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(1, 2)
                k_down = F.interpolate(keys.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(1, 2)
                v_down = F.interpolate(values.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(1, 2)
                
                level_out, level_attn = correlation_layer(q_down, k_down, v_down)
                
                # Upsample back to original resolution
                level_out = F.interpolate(level_out.transpose(1, 2), size=L, mode='linear', align_corners=False).transpose(1, 2)
            
            level_outputs.append(level_out)
            level_attentions.append(level_attn)
        
        # Hierarchical fusion
        weights = F.softmax(self.level_weights, dim=0)
        
        # Weighted combination
        weighted_outputs = [w * out for w, out in zip(weights, level_outputs)]
        combined_output = sum(weighted_outputs)
        
        # Aggregate attention weights
        if level_attentions[0] is not None:
            avg_attention = torch.stack([attn for attn in level_attentions if attn is not None], dim=0).mean(dim=0)
        else:
            avg_attention = None
        
        return combined_output, avg_attention

class FourierAttention(BaseAttention):
    """
    Fourier-based attention for capturing periodic patterns in time series.
    
    This component uses frequency domain analysis to identify and emphasize
    periodic patterns, making it particularly effective for seasonal forecasting.
    """
    
    def __init__(self, d_model, n_heads, seq_len=96, frequency_selection='adaptive', 
                 dropout=0.1, temperature=1.0, learnable_filter=True):
        super(FourierAttention, self).__init__()
        logger.info(f"Initializing FourierAttention: d_model={d_model}, n_heads={n_heads}, seq_len={seq_len}")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.head_dim = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        self.temperature = temperature
        self.learnable_filter = learnable_filter
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # Learnable frequency components - fix initialization
        max_freq_dim = max(seq_len // 2 + 1, 64)  # Ensure minimum size
        self.freq_weights = nn.Parameter(torch.randn(max_freq_dim))
        self.phase_weights = nn.Parameter(torch.zeros(max_freq_dim))
        
        # Standard attention components
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Frequency selection strategy
        self.frequency_selection = frequency_selection
        if frequency_selection == 'adaptive':
            self.freq_selector = nn.Linear(d_model, max_freq_dim)
        
        self.output_dim_multiplier = 1
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Forward pass using Fourier-domain attention with complex filtering.
        
        Args:
            queries: [B, L, D] query tensor
            keys: [B, L, D] key tensor (same as queries for self-attention)
            values: [B, L, D] value tensor
            attn_mask: Optional attention mask
            tau: Temperature parameter (optional)
            delta: Delta parameter (optional)
            
        Returns:
            Tuple of (output, attention_weights)
        """
        B, L, D = queries.shape
        
        # Ensure input is contiguous for FFT operations
        queries = queries.contiguous()
        
        try:
            # Use alternative FFT approach to avoid Intel MKL issues
            queries = queries.contiguous()
            
            # Check if sequence length is compatible with FFT
            if L < 2:
                raise RuntimeError("Sequence length too short for FFT")
            
            # Try multiple FFT approaches
            queries_freq = None
            approaches = [
                lambda x: torch.fft.rfft(x, dim=1, norm='ortho'),  # Orthogonal normalization
                lambda x: torch.fft.fft(x, dim=1)[:, :L//2+1],    # Manual truncation
                lambda x: self._manual_dft(x)                      # Manual DFT implementation
            ]
            
            for i, approach in enumerate(approaches):
                try:
                    queries_freq = approach(queries)
                    freq_dim = queries_freq.shape[1]
                    logger.debug(f"FFT approach {i+1} succeeded with freq_dim={freq_dim}")
                    break
                except Exception as e:
                    logger.debug(f"FFT approach {i+1} failed: {e}")
                    continue
            
            if queries_freq is None:
                raise RuntimeError("All FFT approaches failed")
                
            # Validate frequency dimensions
            if freq_dim <= 0:
                raise RuntimeError("Invalid frequency dimension")
                
        except (RuntimeError, Exception) as e:
            logger.warning(f"All FFT operations failed: {e}, using fallback frequency simulation")
            # Fallback: simulate frequency filtering using conv1d
            queries_filtered = self._frequency_fallback(queries)
            return self._standard_attention(queries_filtered, attn_mask)
        
        try:
            
            # Adaptive frequency selection if enabled
            if self.frequency_selection == 'adaptive':
                freq_weights_input = queries.mean(dim=1)  # [B, D]
                
                # Ensure freq_selector output matches frequency dimensions
                if hasattr(self, 'freq_selector'):
                    freq_logits = self.freq_selector(freq_weights_input)  # [B, max_freq_dim]
                    # Trim to actual frequency dimension
                    freq_logits = freq_logits[:, :freq_dim]
                    freq_selection_weights = F.softmax(freq_logits, dim=-1)  # [B, freq_dim]
                else:
                    # Fallback: uniform weights
                    freq_selection_weights = torch.ones(B, freq_dim, device=queries.device) / freq_dim
                
                # Create frequency filter with proper dimensions
                phase_weights = self.phase_weights[:freq_dim].unsqueeze(0)  # [1, freq_dim]
                freq_weights = self.freq_weights[:freq_dim].unsqueeze(0)    # [1, freq_dim]
                
                # Complex frequency filter with adaptive selection
                freq_filter = torch.complex(
                    torch.cos(phase_weights) * freq_weights * freq_selection_weights,
                    torch.sin(phase_weights) * freq_weights * freq_selection_weights
                )  # [B, freq_dim]
            else:
                # Create frequency filter with proper dimensions
                phase_weights = self.phase_weights[:freq_dim]  # [freq_dim]
                freq_weights = self.freq_weights[:freq_dim]    # [freq_dim]
                
                freq_filter = torch.complex(
                    torch.cos(phase_weights) * freq_weights,
                    torch.sin(phase_weights) * freq_weights
                )  # [freq_dim]
                freq_filter = freq_filter.unsqueeze(0)  # [1, freq_dim]
            
            # Apply complex frequency filtering with correct broadcasting
            freq_filter = freq_filter.unsqueeze(-1)  # [B, freq_dim, 1] or [1, freq_dim, 1]
            queries_freq_filtered = queries_freq * freq_filter  # [B, freq_dim, D]
            
            # Transform back to time domain with error handling and multiple approaches
            try:
                inverse_approaches = [
                    lambda x: torch.fft.irfft(x, n=L, dim=1, norm='ortho'),
                    lambda x: torch.real(torch.fft.ifft(F.pad(x, (0, 0, 0, L-x.shape[1])), dim=1)),
                    lambda x: self._manual_idft(x, L)
                ]
                
                for i, inv_approach in enumerate(inverse_approaches):
                    try:
                        queries_filtered = inv_approach(queries_freq_filtered)
                        logger.debug(f"Inverse FFT approach {i+1} succeeded")
                        break
                    except Exception as e:
                        logger.debug(f"Inverse FFT approach {i+1} failed: {e}")
                        continue
                        
                if 'queries_filtered' not in locals():
                    raise RuntimeError("All inverse FFT approaches failed")
                    
            except Exception as e:
                logger.warning(f"Inverse FFT failed: {e}, using frequency fallback")
                queries_filtered = self._frequency_fallback(queries)
                return self._standard_attention(queries_filtered, attn_mask)
            
            # Ensure correct output length
            if queries_filtered.shape[1] != L:
                queries_filtered = F.interpolate(
                    queries_filtered.transpose(1, 2), 
                    size=L, 
                    mode='linear', 
                    align_corners=False
                ).transpose(1, 2)
                
        except Exception as e:
            logger.warning(f"FFT operation failed: {e}, falling back to standard attention")
            # Fallback to standard processing without FFT
            queries_filtered = queries
        
        # Standard multi-head attention on filtered signal
        qkv = self.qkv(queries_filtered).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, D/H]
        
        # Compute attention scores
        scale = math.sqrt(self.head_dim)
        attn_scores = (q @ k.transpose(-2, -1)) / scale
        
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -float('inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = (attn_weights @ v).transpose(1, 2).reshape(B, L, D)
        
        return self.out_proj(out), attn_weights

    def _manual_dft(self, x):
        """
        Manual DFT implementation as fallback for problematic FFT configurations.
        
        Args:
            x: [B, L, D] input tensor
            
        Returns:
            x_freq: [B, L//2+1, D] frequency domain representation
        """
        B, L, D = x.shape
        
        # Create frequency indices
        freqs = torch.arange(0, L//2 + 1, device=x.device, dtype=torch.float32)
        times = torch.arange(0, L, device=x.device, dtype=torch.float32)
        
        # Compute DFT manually (simplified version)
        x_freq = torch.zeros(B, L//2 + 1, D, dtype=torch.complex64, device=x.device)
        
        for k in range(L//2 + 1):
            # e^(-2πi * k * n / L)
            phase = -2 * math.pi * k * times / L
            kernel_real = torch.cos(phase)
            kernel_imag = torch.sin(phase)
            
            # Apply DFT kernel
            real_part = torch.sum(x * kernel_real.view(1, -1, 1), dim=1)
            imag_part = torch.sum(x * kernel_imag.view(1, -1, 1), dim=1)
            
            x_freq[:, k, :] = torch.complex(real_part, imag_part)
        
        return x_freq

    def _manual_idft(self, x_freq, target_length):
        """
        Manual inverse DFT implementation.
        
        Args:
            x_freq: [B, F, D] frequency domain tensor
            target_length: Target time series length
            
        Returns:
            x_time: [B, target_length, D] time domain representation
        """
        B, F, D = x_freq.shape
        L = target_length
        
        # Create time and frequency indices
        times = torch.arange(0, L, device=x_freq.device, dtype=torch.float32)
        freqs = torch.arange(0, F, device=x_freq.device, dtype=torch.float32)
        
        # Initialize output
        x_time = torch.zeros(B, L, D, device=x_freq.device)
        
        for n in range(L):
            # e^(2πi * k * n / L)
            phase = 2 * math.pi * freqs * n / L
            kernel_real = torch.cos(phase)
            kernel_imag = torch.sin(phase)
            
            # Apply inverse DFT
            real_contrib = torch.sum(
                torch.real(x_freq) * kernel_real.view(1, -1, 1) - 
                torch.imag(x_freq) * kernel_imag.view(1, -1, 1), 
                dim=1
            )
            
            x_time[:, n, :] = real_contrib / L
        
        return x_time

    def _frequency_fallback(self, x):
        """
        Fallback frequency filtering using 1D convolution when FFT fails.
        
        Args:
            x: [B, L, D] input tensor
            
        Returns:
            filtered: [B, L, D] frequency-filtered output
        """
        B, L, D = x.shape
        
        # Transpose for conv1d: [B, D, L]
        x_conv = x.transpose(1, 2)
        
        # Apply learnable frequency filters (simulate band-pass filtering)
        freq_kernels = self.freq_weights[:min(3, len(self.freq_weights))].view(1, 1, -1)
        
        # Pad for convolution
        padding = freq_kernels.shape[-1] // 2
        x_padded = F.pad(x_conv, (padding, padding), mode='reflect')
        
        # Apply convolution as frequency filter
        filtered = F.conv1d(x_padded, freq_kernels.expand(D, 1, -1), groups=D)
        
        # Ensure correct output length and transpose back
        if filtered.shape[-1] != L:
            filtered = F.interpolate(filtered, size=L, mode='linear', align_corners=False)
        
        return filtered.transpose(1, 2)  # [B, L, D]

    def _standard_attention(self, x, attn_mask=None):
        """
        Standard multi-head attention fallback.
        
        Args:
            x: [B, L, D] input tensor
            attn_mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        B, L, D = x.shape
        
        # Standard multi-head attention
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, D/H]
        
        # Compute attention scores
        scale = math.sqrt(self.head_dim)
        attn_scores = (q @ k.transpose(-2, -1)) / scale
        
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -float('inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = (attn_weights @ v).transpose(1, 2).reshape(B, L, D)
        
        return self.out_proj(out), attn_weights


class FourierBlock(BaseAttention):
    """
    1D Fourier block for frequency domain representation learning.
    
    Performs FFT, learnable linear transformation in frequency domain,
    and inverse FFT to return to time domain.
    """
    
    def __init__(self, in_channels, out_channels, n_heads, seq_len, modes=64, mode_select_method='random'):
        super(FourierBlock, self).__init__()
        logger.info(f"Initializing FourierBlock: {in_channels} -> {out_channels}, modes={modes}")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.seq_len = seq_len
        
        # Get frequency modes
        self.index = self._get_frequency_modes(seq_len, modes, mode_select_method)
        logger.info(f"Selected frequency modes: {self.index}")
        
        self.scale = (1 / (in_channels * out_channels))
        
        # Learnable frequency domain weights (complex)
        self.weights_real = nn.Parameter(
            self.scale * torch.rand(n_heads, in_channels // n_heads, out_channels // n_heads, len(self.index))
        )
        self.weights_imag = nn.Parameter(
            self.scale * torch.rand(n_heads, in_channels // n_heads, out_channels // n_heads, len(self.index))
        )
        
        self.output_dim_multiplier = out_channels / in_channels
    
    def _get_frequency_modes(self, seq_len, modes=64, mode_select_method='random'):
        """Select frequency modes for processing."""
        modes = min(modes, seq_len // 2)
        if mode_select_method == 'random':
            index = list(range(0, seq_len // 2))
            np.random.shuffle(index)
            index = index[:modes]
        else:
            index = list(range(0, modes))
        index.sort()
        return index
    
    def _complex_mul1d(self, x, weights_real, weights_imag):
        """Complex multiplication in frequency domain."""
        # Convert to complex if needed
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))
        
        weights = torch.complex(weights_real, weights_imag)
        
        # Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        return torch.einsum("bhio,bhio->bho", x, weights)
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Forward pass through Fourier block.
        
        Args:
            queries: [B, L, H, E] input tensor (reshaped from [B, L, D])
            keys: Not used in this implementation
            values: Not used in this implementation
            
        Returns:
            Tuple of (output, None)
        """
        # Assume queries is [B, L, D] and reshape to [B, H, E, L]
        B, L, D = queries.shape
        H = self.n_heads
        E = D // H
        
        x = queries.view(B, L, H, E).permute(0, 2, 3, 1)  # [B, H, E, L]
        
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x, dim=-1)
        
        # Perform Fourier neural operations
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        
        for wi, i in enumerate(self.index):
            if i >= x_ft.shape[3] or wi >= len(self.index):
                continue
            out_ft[:, :, :, wi] = self._complex_mul1d(
                x_ft[:, :, :, i:i+1], 
                self.weights_real[:, :, :, wi:wi+1],
                self.weights_imag[:, :, :, wi:wi+1]
            ).squeeze(-1)
        
        # Return to time domain
        x_out = torch.fft.irfft(out_ft, n=L, dim=-1)
        
        # Reshape back to [B, L, D]
        output = x_out.permute(0, 3, 1, 2).reshape(B, L, -1)
        
        return output, None


class FourierCrossAttention(BaseAttention):
    """
    Fourier-enhanced cross attention for encoder-decoder architectures.
    
    Performs cross-attention in frequency domain with different modes
    for queries and key-value pairs.
    """
    
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=64, 
                 mode_select_method='random', activation='tanh', num_heads=8):
        super(FourierCrossAttention, self).__init__()
        logger.info(f"Initializing FourierCrossAttention: {in_channels} -> {out_channels}")
        
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        
        # Get frequency modes for queries and keys/values
        self.index_q = self._get_frequency_modes(seq_len_q, modes, mode_select_method)
        self.index_kv = self._get_frequency_modes(seq_len_kv, modes, mode_select_method)
        
        logger.info(f"Query modes: {len(self.index_q)}, KV modes: {len(self.index_kv)}")
        
        self.scale = (1 / (in_channels * out_channels))
        
        # Learnable weights for cross-attention in frequency domain
        self.weights_q = nn.Parameter(self.scale * torch.rand(num_heads, in_channels, len(self.index_q), dtype=torch.cfloat))
        self.weights_kv = nn.Parameter(self.scale * torch.rand(num_heads, in_channels, len(self.index_kv), dtype=torch.cfloat))
        
        self.output_dim_multiplier = out_channels / in_channels
    
    def _get_frequency_modes(self, seq_len, modes=64, mode_select_method='random'):
        """Select frequency modes for processing."""
        modes = min(modes, seq_len // 2)
        if mode_select_method == 'random':
            index = list(range(0, seq_len // 2))
            np.random.shuffle(index)
            index = index[:modes]
        else:
            index = list(range(0, modes))
        index.sort()
        return index
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Forward pass for cross-attention in frequency domain.
        
        Args:
            queries: [B, L_q, D] query tensor
            keys: [B, L_kv, D] key tensor
            values: [B, L_kv, D] value tensor
            
        Returns:
            Tuple of (output, attention_weights)
        """
        B, L_q, D = queries.shape
        _, L_kv, _ = keys.shape
        
        # Transform to frequency domain
        q_ft = torch.fft.rfft(queries, dim=1)  # [B, L_q//2+1, D]
        k_ft = torch.fft.rfft(keys, dim=1)     # [B, L_kv//2+1, D]
        v_ft = torch.fft.rfft(values, dim=1)   # [B, L_kv//2+1, D]
        
        # Apply frequency domain transformations
        q_filtered = torch.zeros_like(q_ft)
        kv_filtered = torch.zeros_like(k_ft)
        
        # Process selected frequency modes
        for i, freq_idx in enumerate(self.index_q):
            if freq_idx < q_ft.shape[1]:
                q_filtered[:, i, :] = q_ft[:, freq_idx, :] * self.weights_q[0, :, i]
        
        for i, freq_idx in enumerate(self.index_kv):
            if freq_idx < k_ft.shape[1]:
                kv_filtered[:, i, :] = k_ft[:, freq_idx, :] * self.weights_kv[0, :, i]
        
        # Convert back to time domain for attention computation
        q_processed = torch.fft.irfft(q_filtered, n=L_q, dim=1)
        k_processed = torch.fft.irfft(kv_filtered, n=L_kv, dim=1)
        v_processed = torch.fft.irfft(kv_filtered, n=L_kv, dim=1)
        
        # Standard cross-attention
        scale = math.sqrt(D)
        attn_scores = torch.bmm(q_processed, k_processed.transpose(1, 2)) / scale
        
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -float('inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.bmm(attn_weights, v_processed)
        
        return output, attn_weights

class AttentionRegistry:
    """
    A registry for all available attention components.
    """
    _registry = {
        # Legacy components
        "autocorrelation_layer": AutoCorrelationLayer,
        "adaptive_autocorrelation_layer": AdaptiveAutoCorrelationLayer,
        "cross_resolution_attention": CrossResolutionAttention,
        
        # Phase 2: Fourier Attention Components
        "fourier_attention": FourierAttention,
        "fourier_block": FourierBlock,
        "fourier_cross_attention": FourierCrossAttention,
        
        # Phase 2: Wavelet Attention Components
        "wavelet_attention": WaveletAttention,
        "wavelet_decomposition": WaveletDecomposition,
        "adaptive_wavelet_attention": AdaptiveWaveletAttention,
        "multi_scale_wavelet_attention": MultiScaleWaveletAttention,

        # Modular Advanced Attention Components
        "two_stage_attention": TwoStageAttention,
        "exponential_smoothing_attention": ExponentialSmoothingAttention,
        "multi_wavelet_cross_attention": MultiWaveletCrossAttention,
        
        # Phase 2: Enhanced AutoCorrelation Components
        "enhanced_autocorrelation": EnhancedAutoCorrelation,
        "new_adaptive_autocorrelation_layer": NewAdaptiveAutoCorrelationLayer,
        "hierarchical_autocorrelation": HierarchicalAutoCorrelation,
        
        # Phase 2: Bayesian Attention Components
        "bayesian_attention": BayesianAttention,
        "bayesian_multi_head_attention": BayesianMultiHeadAttention,
        "variational_attention": VariationalAttention,
        "bayesian_cross_attention": BayesianCrossAttention,
        
        # Phase 2: Adaptive Components
        "meta_learning_adapter": MetaLearningAdapter,
        "adaptive_mixture": AdaptiveMixture,
        
        # Phase 2: Temporal Convolution Attention Components
        "causal_convolution": CausalConvolution,
        "temporal_conv_net": TemporalConvNet,
        "convolutional_attention": ConvolutionalAttention,
    }

    @classmethod
    def register(cls, name, component_class):
        if name in cls._registry:
            logger.warning(f"Component '{name}' is already registered and will be overwritten.")
        cls._registry[name] = component_class
        logger.info(f"Registered attention component: {name}")

    @classmethod
    def get(cls, name):
        component = cls._registry.get(name)
        if component is None:
            logger.error(f"Attention component '{name}' not found.")
            raise ValueError(f"Attention component '{name}' not found.")
        return component

    @classmethod
    def list_components(cls):
        return list(cls._registry.keys())

def get_attention_component(name, **kwargs):
    component_class = AttentionRegistry.get(name)
    
    # Legacy components
    if name == "autocorrelation_layer":
        autocorrelation = AutoCorrelation(
            mask_flag=kwargs.get('mask_flag', True),
            factor=kwargs.get('factor', 1),
            attention_dropout=kwargs.get('dropout', 0.1),
            output_attention=kwargs.get('output_attention', False)
        )
        return component_class(
            autocorrelation,
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads')
        )
    elif name == "adaptive_autocorrelation_layer":
        autocorrelation = AdaptiveAutoCorrelation(
            mask_flag=kwargs.get('mask_flag', True),
            factor=kwargs.get('factor', 1),
            attention_dropout=kwargs.get('dropout', 0.1),
            output_attention=kwargs.get('output_attention', False)
        )
        return component_class(
            autocorrelation,
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads')
        )
    elif name == "cross_resolution_attention":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_levels=kwargs.get('n_levels'),
            n_heads=kwargs.get('n_heads')
        )
    
    # Phase 2: Fourier Attention Components
    elif name == "fourier_attention":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            seq_len=kwargs.get('seq_len', 96),
            frequency_selection=kwargs.get('frequency_selection', 'adaptive'),
            dropout=kwargs.get('dropout', 0.1),
            temperature=kwargs.get('temperature', 1.0),
            learnable_filter=kwargs.get('learnable_filter', True)
        )
    elif name == "fourier_block":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            seq_len_q=kwargs.get('seq_len_q', 96),
            dropout=kwargs.get('dropout', 0.1),
            activation=kwargs.get('activation', 'gelu')
        )
    elif name == "fourier_cross_attention":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            dropout=kwargs.get('dropout', 0.1),
            temperature=kwargs.get('temperature', 1.0)
        )
    
    # Phase 2: Wavelet Attention Components
    elif name == "wavelet_attention":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            levels=kwargs.get('levels', 3),
            n_levels=kwargs.get('n_levels'),  # Pass as alias
            wavelet_type=kwargs.get('wavelet_type', 'learnable'),
            dropout=kwargs.get('dropout', 0.1)
        )
    elif name == "wavelet_decomposition":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            n_levels=kwargs.get('n_levels', 4),
            wavelet_type=kwargs.get('wavelet_type', 'learnable'),
            dropout=kwargs.get('dropout', 0.1)
        )
    elif name == "adaptive_wavelet_attention":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            max_levels=kwargs.get('max_levels', 5),
            dropout=kwargs.get('dropout', 0.1)
        )
    elif name == "multi_scale_wavelet_attention":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            scales=kwargs.get('scales', [1, 2, 4, 8]),
            dropout=kwargs.get('dropout', 0.1)
        )
    
    # Phase 2: Enhanced AutoCorrelation Components
    elif name == "enhanced_autocorrelation":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            factor=kwargs.get('factor', 1),
            attention_dropout=kwargs.get('dropout', 0.1),
            output_attention=kwargs.get('output_attention', False),
            adaptive_k=kwargs.get('adaptive_k', True),
            multi_scale=kwargs.get('multi_scale', True)
        )
    elif name == "new_adaptive_autocorrelation_layer":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            factor=kwargs.get('factor', 1),
            attention_dropout=kwargs.get('dropout', 0.1),
            output_attention=kwargs.get('output_attention', False),
            adaptive_k=kwargs.get('adaptive_k', True),
            multi_scale=kwargs.get('multi_scale', True)
        )
    elif name == "hierarchical_autocorrelation":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            hierarchy_levels=kwargs.get('hierarchy_levels', [1, 4, 16]),
            factor=kwargs.get('factor', 1)
        )
    
    # Phase 2: Bayesian Attention Components
    elif name == "bayesian_attention":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            dropout=kwargs.get('dropout', 0.1),
            prior_std=kwargs.get('prior_std', 1.0),
            temperature=kwargs.get('temperature', 1.0),
            output_attention=kwargs.get('output_attention', False)
        )
    elif name == "bayesian_multi_head_attention":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            dropout=kwargs.get('dropout', 0.1),
            prior_std=kwargs.get('prior_std', 1.0),
            n_samples=kwargs.get('n_samples', 5),
            output_attention=kwargs.get('output_attention', False)
        )
    elif name == "variational_attention":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            dropout=kwargs.get('dropout', 0.1),
            learn_variance=kwargs.get('learn_variance', True)
        )
    elif name == "bayesian_cross_attention":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            dropout=kwargs.get('dropout', 0.1),
            prior_std=kwargs.get('prior_std', 1.0)
        )
    
    # Phase 2: Adaptive Components
    elif name == "meta_learning_adapter":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            adaptation_steps=kwargs.get('adaptation_steps', 3),
            meta_lr=kwargs.get('meta_lr', 0.01),
            inner_lr=kwargs.get('inner_lr', 0.1),
            dropout=kwargs.get('dropout', 0.1)
        )
    elif name == "adaptive_mixture":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            mixture_components=kwargs.get('mixture_components', 3),
            gate_hidden_dim=kwargs.get('gate_hidden_dim', None),
            dropout=kwargs.get('dropout', 0.1),
            temperature=kwargs.get('temperature', 1.0)
        )
    
    # Phase 2: Temporal Convolution Attention Components
    elif name == "causal_convolution":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            kernel_sizes=kwargs.get('kernel_sizes', [3, 5, 7]),
            dilation_rates=kwargs.get('dilation_rates', [1, 2, 4]),
            dropout=kwargs.get('dropout', 0.1),
            activation=kwargs.get('activation', 'gelu')
        )
    elif name == "temporal_conv_net":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            num_levels=kwargs.get('num_levels', 4),
            kernel_size=kwargs.get('kernel_size', 3),
            dropout=kwargs.get('dropout', 0.1)
        )
    elif name == "convolutional_attention":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads'),
            conv_kernel_size=kwargs.get('conv_kernel_size', 3),
            pool_size=kwargs.get('pool_size', 2),
            dropout=kwargs.get('dropout', 0.1)
        )
    
    raise ValueError(f"Factory not implemented for attention type: {name}")

class CausalConvolution(BaseAttention):
    """
    Causal convolution attention mechanism for temporal sequence modeling.
    
    This component uses dilated causal convolutions to capture temporal
    dependencies while maintaining causality constraints.
    """
    
    def __init__(self, d_model, n_heads, kernel_sizes=[3, 5, 7], dilation_rates=[1, 2, 4],
                 dropout=0.1, activation='gelu'):
        super(CausalConvolution, self).__init__()
        logger.info(f"Initializing CausalConvolution: d_model={d_model}, kernels={kernel_sizes}")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.kernel_sizes = kernel_sizes
        self.dilation_rates = dilation_rates
        
        # Multi-scale causal convolutions
        self.causal_convs = nn.ModuleList()
        for kernel_size in kernel_sizes:
            conv_layers = nn.ModuleList()
            for dilation in dilation_rates:
                # Causal padding calculation
                padding = (kernel_size - 1) * dilation
                conv = nn.Conv1d(
                    d_model, d_model, 
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding
                )
                conv_layers.append(conv)
            self.causal_convs.append(conv_layers)
        
        # Attention projection layers
        self.query_conv = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.key_conv = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.value_conv = nn.Conv1d(d_model, d_model, kernel_size=1)
        
        # Output projection
        self.output_projection = nn.Linear(d_model * len(kernel_sizes), d_model)
        
        # Normalization and activation
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()
        
        # Positional encoding for temporal awareness
        self.positional_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        
        self.output_dim_multiplier = 1
    
    def apply_causal_mask(self, x, kernel_size, dilation):
        """
        Apply causal masking by removing future information.
        
        Args:
            x: [B, D, L] convolution output
            kernel_size: Convolution kernel size
            dilation: Dilation rate
            
        Returns:
            masked output with causality preserved
        """
        # Calculate how much to trim from the end
        trim_amount = (kernel_size - 1) * dilation
        if trim_amount > 0:
            x = x[:, :, :-trim_amount]
        return x
    
    def multi_scale_causal_conv(self, x):
        """
        Apply multi-scale causal convolutions.
        
        Args:
            x: [B, L, D] input tensor
            
        Returns:
            List of convolution outputs at different scales
        """
        B, L, D = x.shape
        x_conv = x.transpose(1, 2)  # [B, D, L] for convolution
        
        scale_outputs = []
        
        for kernel_idx, conv_layers in enumerate(self.causal_convs):
            kernel_size = self.kernel_sizes[kernel_idx]
            
            # Apply dilated convolutions at current kernel size
            dilated_outputs = []
            for dilation_idx, conv_layer in enumerate(conv_layers):
                dilation = self.dilation_rates[dilation_idx]
                
                # Apply convolution
                conv_out = conv_layer(x_conv)
                
                # Apply causal masking
                conv_out = self.apply_causal_mask(conv_out, kernel_size, dilation)
                
                # Pad to original length if needed
                if conv_out.size(-1) < L:
                    padding_needed = L - conv_out.size(-1)
                    conv_out = F.pad(conv_out, (0, padding_needed))
                
                # Apply activation
                conv_out = self.activation(conv_out)
                dilated_outputs.append(conv_out)
            
            # Combine dilated outputs (average)
            combined_output = torch.stack(dilated_outputs, dim=0).mean(dim=0)
            scale_outputs.append(combined_output)
        
        return scale_outputs
    
    def temporal_attention(self, queries, keys, values):
        """
        Compute temporal attention with convolution features.
        
        Args:
            queries: [B, D, L] query features
            keys: [B, D, L] key features  
            values: [B, D, L] value features
            
        Returns:
            attention output and weights
        """
        B, D, L = queries.shape
        H = self.n_heads
        d_k = D // H
        
        # Reshape for multi-head processing
        q = queries.view(B, H, d_k, L).transpose(2, 3)  # [B, H, L, d_k]
        k = keys.view(B, H, d_k, L).transpose(2, 3)     # [B, H, L, d_k]
        v = values.view(B, H, d_k, L).transpose(2, 3)   # [B, H, L, d_k]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
        
        # Apply causal mask (future positions set to -inf)
        causal_mask = torch.triu(torch.ones(L, L, device=queries.device), diagonal=1).bool()
        scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)  # [B, H, L, d_k]
        
        # Reshape back
        context = context.transpose(2, 3).contiguous().view(B, D, L)
        
        return context, attn_weights
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Forward pass with causal convolution attention.
        
        Args:
            queries: [B, L, D] query tensor
            keys: [B, L, D] key tensor
            values: [B, L, D] value tensor
            attn_mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        B, L, D = queries.shape
        
        residual = queries
        
        # Apply positional encoding
        pos_encoded = queries + self.positional_conv(queries.transpose(1, 2)).transpose(1, 2)
        
        # Multi-scale causal convolutions
        conv_outputs = self.multi_scale_causal_conv(pos_encoded)
        
        # Convert to attention format
        q_conv = self.query_conv(queries.transpose(1, 2))    # [B, D, L]
        k_conv = self.key_conv(keys.transpose(1, 2))         # [B, D, L]
        v_conv = self.value_conv(values.transpose(1, 2))     # [B, D, L]
        
        # Add convolution features to keys and values
        enhanced_keys = k_conv
        enhanced_values = v_conv
        
        for conv_out in conv_outputs:
            enhanced_keys = enhanced_keys + conv_out
            enhanced_values = enhanced_values + conv_out
        
        # Apply temporal attention
        attn_output, attn_weights = self.temporal_attention(q_conv, enhanced_keys, enhanced_values)
        
        # Combine multi-scale outputs
        combined_conv = torch.cat(conv_outputs, dim=1)  # [B, D*len(kernels), L]
        combined_conv = combined_conv.transpose(1, 2)   # [B, L, D*len(kernels)]
        
        # Project to original dimension
        conv_features = self.output_projection(combined_conv)
        
        # Combine attention and convolution features
        final_output = attn_output.transpose(1, 2) + conv_features
        
        # Residual connection and normalization
        output = self.layer_norm(final_output + residual)
        
        return output, attn_weights


class TemporalConvNet(BaseAttention):
    """
    Temporal Convolution Network (TCN) based attention mechanism.
    
    This implements a full TCN architecture for sequence modeling
    with attention-like interfaces.
    """
    
    def __init__(self, d_model, n_heads, num_levels=4, kernel_size=3, dropout=0.1):
        super(TemporalConvNet, self).__init__()
        logger.info(f"Initializing TemporalConvNet: levels={num_levels}, kernel={kernel_size}")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_levels = num_levels
        self.kernel_size = kernel_size
        
        # Build TCN layers with exponentially increasing dilation
        self.tcn_layers = nn.ModuleList()
        dilation_size = 1
        
        for i in range(num_levels):
            # Residual block with dilated convolution
            layer = TemporalBlock(
                d_model, d_model, kernel_size, 
                stride=1, dilation=dilation_size, 
                padding=(kernel_size-1) * dilation_size, 
                dropout=dropout
            )
            self.tcn_layers.append(layer)
            dilation_size *= 2  # Exponential dilation
        
        # Attention mechanism for combining temporal features
        self.temporal_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # Output processing
        self.output_norm = nn.LayerNorm(d_model)
        self.output_dropout = nn.Dropout(dropout)
        
        self.output_dim_multiplier = 1
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Forward pass through TCN with attention combination.
        
        Args:
            queries: [B, L, D] query tensor
            keys: [B, L, D] key tensor (used as input)
            values: [B, L, D] value tensor (used as input)
            attn_mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        B, L, D = queries.shape
        
        # Use keys as primary input for TCN processing
        x = keys
        residual = x
        
        # Apply TCN layers sequentially
        temporal_features = []
        current_input = x.transpose(1, 2)  # [B, D, L] for conv
        
        for tcn_layer in self.tcn_layers:
            current_input = tcn_layer(current_input)
            # Store features from each level
            temporal_features.append(current_input.transpose(1, 2))  # Back to [B, L, D]
        
        # Final TCN output
        tcn_output = current_input.transpose(1, 2)  # [B, L, D]
        
        # Apply attention to combine with queries
        attn_output, attn_weights = self.temporal_attention(
            queries, tcn_output, values, attn_mask=attn_mask
        )
        
        # Apply output processing
        output = self.output_norm(attn_output + residual)
        output = self.output_dropout(output)
        
        return output, attn_weights


class TemporalBlock(nn.Module):
    """
    Temporal block for TCN with residual connections and causal convolutions.
    """
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # First convolution layer
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        
        # Apply causal masking by chopping off the future
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second convolution layer
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Combine layers
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # Downsample for residual connection if needed
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize convolution weights."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        """Forward pass through temporal block."""
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """
    Remove rightmost elements to ensure causality in convolutions.
    """
    
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        """Remove future information from convolution output."""
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size].contiguous()
        return x


class ConvolutionalAttention(BaseAttention):
    """
    Convolutional attention mechanism combining spatial and temporal convolutions.
    
    This component uses 2D convolutions to capture both spatial (feature)
    and temporal relationships in attention computation.
    """
    
    def __init__(self, d_model, n_heads, conv_kernel_size=3, pool_size=2, dropout=0.1):
        super(ConvolutionalAttention, self).__init__()
        logger.info(f"Initializing ConvolutionalAttention: d_model={d_model}")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.conv_kernel_size = conv_kernel_size
        
        # 2D convolutions for attention feature extraction
        self.spatial_conv = nn.Conv2d(
            1, n_heads, 
            kernel_size=(conv_kernel_size, conv_kernel_size),
            padding=(conv_kernel_size//2, conv_kernel_size//2)
        )
        
        # Temporal convolutions
        self.temporal_conv_q = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.temporal_conv_k = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.temporal_conv_v = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        
        # Standard attention projections
        self.w_qs = nn.Linear(d_model, d_model)
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Pooling for dimension reduction
        self.adaptive_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.output_dim_multiplier = 1
    
    def compute_conv_attention_weights(self, queries, keys):
        """
        Compute attention weights using 2D convolutions.
        
        Args:
            queries: [B, L, D] query tensor
            keys: [B, L, D] key tensor
            
        Returns:
            conv_attention_weights: [B, H, L, L] attention weights
        """
        B, L, D = queries.shape
        H = self.n_heads
        
        # Compute pairwise interactions
        q_expanded = queries.unsqueeze(2).expand(B, L, L, D)  # [B, L, L, D]
        k_expanded = keys.unsqueeze(1).expand(B, L, L, D)     # [B, L, L, D]
        
        # Element-wise product for interaction
        interactions = q_expanded * k_expanded  # [B, L, L, D]
        
        # Average across feature dimension and add channel dimension
        interaction_map = interactions.mean(dim=-1, keepdim=True)  # [B, L, L, 1]
        interaction_map = interaction_map.permute(0, 3, 1, 2)     # [B, 1, L, L]
        
        # Apply 2D convolution to capture spatial patterns
        conv_weights = self.spatial_conv(interaction_map)  # [B, H, L, L]
        
        # Apply softmax to get attention weights
        conv_attention_weights = F.softmax(conv_weights, dim=-1)
        
        return conv_attention_weights
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Forward pass with convolutional attention.
        
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
        
        # Apply temporal convolutions
        q_temp = self.temporal_conv_q(queries.transpose(1, 2)).transpose(1, 2)
        k_temp = self.temporal_conv_k(keys.transpose(1, 2)).transpose(1, 2)
        v_temp = self.temporal_conv_v(values.transpose(1, 2)).transpose(1, 2)
        
        # Apply standard projections
        q = self.w_qs(q_temp).view(B, L, H, self.d_k).transpose(1, 2)
        k = self.w_ks(k_temp).view(B, L, H, self.d_k).transpose(1, 2)
        v = self.w_vs(v_temp).view(B, L, H, self.d_k).transpose(1, 2)
        
        # Compute standard attention scores
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)
        
        # Compute convolutional attention weights
        conv_attn_weights = self.compute_conv_attention_weights(q_temp, k_temp)
        
        # Combine standard and convolutional attention
        combined_scores = scores + conv_attn_weights
        
        # Apply attention mask if provided
        if attn_mask is not None:
            combined_scores.masked_fill_(attn_mask == 0, -1e9)
        
        # Compute final attention weights
        attn_weights = F.softmax(combined_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        
        # Apply output projection
        output = self.w_o(context)
        
        # Residual connection and normalization
        output = self.layer_norm(output + residual)
        
        return output, attn_weights

class WaveletAttention(BaseAttention):
    """
    Wavelet-based attention using learnable wavelet decomposition.
    
    This component performs multi-resolution analysis by decomposing
    time series into different frequency bands using wavelet transforms.
    """
    
    def __init__(self, d_model, n_heads, levels=3, n_levels=None, wavelet_type='learnable', 
                 dropout=0.1):
        super().__init__()
        
        # Handle both levels and n_levels parameters for compatibility
        if n_levels is not None:
            levels = n_levels
        
        logger.info(f"Initializing WaveletAttention: d_model={d_model}, levels={levels}")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.levels = levels
        self.head_dim = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # Wavelet decomposition
        self.wavelet_decomp = WaveletDecomposition(d_model, levels)
        
        # Multi-head attention for each decomposition level
        self.level_attentions = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            for _ in range(levels + 1)  # +1 for final low-frequency component
        ])
        
        # Level fusion
        self.level_weights = nn.Parameter(torch.ones(levels + 1) / (levels + 1))
        self.fusion_proj = nn.Linear(d_model * (levels + 1), d_model)
        
        self.output_dim_multiplier = 1

class TwoStageAttention(BaseAttention):
    """
    Modular implementation of Two-Stage Attention (TSA).
    This combines cross-time and cross-dimension attention for segment merging and multi-variate time series.
    Algorithm adapted from TwoStageAttentionLayer in SelfAttention_Family.py.
    """
    def __init__(self, d_model: int, n_heads: int, seg_num: int, factor: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.dim_sender = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.dim_receiver = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x: torch.Tensor, attn_mask=None, tau=None, delta=None):
        # x: [batch, ts_d, seg_num, d_model]
        batch = x.shape[0]
        ts_d = x.shape[1]
        seg_num = x.shape[2]
        d_model = x.shape[3]
        # Cross Time Stage
        time_in = x.reshape(-1, seg_num, d_model)
        time_enc, _ = self.time_attention(time_in, time_in, time_in, attn_mask=attn_mask)
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)
        # Cross Dimension Stage
        dim_send = dim_in.reshape(batch * seg_num, ts_d, d_model)
        batch_router = self.router.repeat(batch, 1, 1)
        dim_buffer, _ = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=attn_mask)
        dim_receive, _ = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=attn_mask)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)
        final_out = dim_enc.reshape(batch, ts_d, seg_num, d_model)
        return final_out, None


class ExponentialSmoothingAttention(BaseAttention):
    """
    Modular implementation of Exponential Smoothing Attention.
    Implements ETS-style attention using learnable smoothing weights.
    Algorithm adapted from ExponentialSmoothing in ETSformer_EncDec.py.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self._smoothing_weight = nn.Parameter(torch.randn(n_heads, 1))
        self.v0 = nn.Parameter(torch.randn(1, 1, n_heads, d_model // n_heads))
        self.dropout = nn.Dropout(dropout)

    @property
    def weight(self):
        return torch.sigmoid(self._smoothing_weight)

    def get_exponential_weight(self, T):
        powers = torch.arange(T, dtype=torch.float, device=self.weight.device)
        weight = (1 - self.weight) * (self.weight ** torch.flip(powers, dims=(0,)))
        init_weight = self.weight ** (powers + 1)
        return init_weight.unsqueeze(0).unsqueeze(-1), weight.unsqueeze(0).unsqueeze(-1)

    def forward(self, values: torch.Tensor, attn_mask=None, tau=None, delta=None):
        # values: [B, L, D]
        B, L, D = values.shape
        H = self.n_heads
        d_head = D // H
        values = values.view(B, L, H, d_head)
        init_weight, weight = self.get_exponential_weight(L)
        output = F.conv1d(self.dropout(values.permute(0, 2, 3, 1)), weight.squeeze(-1), groups=H)
        output = init_weight * self.v0 + output
        output = output.permute(0, 3, 1, 2).reshape(B, L, D)
        return output, None


class MultiWaveletCrossAttention(BaseAttention):
    """
    Modular implementation of Multi-Wavelet Cross Attention.
    Applies cross-attention using multi-resolution wavelet transforms.
    Algorithm adapted from MultiWaveletCorrelation and wavelet attention patterns.
    """
    def __init__(self, d_model: int, n_heads: int, levels: int = 3, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.levels = levels
        self.dropout = nn.Dropout(dropout)
        self.wavelet_decomp = WaveletDecomposition(d_model, levels)
        self.cross_attentions = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            for _ in range(levels + 1)
        ])
        self.level_weights = nn.Parameter(torch.ones(levels + 1) / (levels + 1))
        self.fusion_proj = nn.Linear(d_model * (levels + 1), d_model)

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, D = queries.shape
        _, q_components = self.wavelet_decomp(queries)
        _, k_components = self.wavelet_decomp(keys)
        _, v_components = self.wavelet_decomp(values)
        level_outputs = []
        level_attentions = []
        for i, (q_comp, k_comp, v_comp, attention_layer) in enumerate(
            zip(q_components, k_components, v_components, self.cross_attentions)
        ):
            if q_comp.size(1) != L:
                q_comp = F.interpolate(q_comp.transpose(1, 2), size=L, mode='linear', align_corners=False).transpose(1, 2)
                k_comp = F.interpolate(k_comp.transpose(1, 2), size=L, mode='linear', align_corners=False).transpose(1, 2)
                v_comp = F.interpolate(v_comp.transpose(1, 2), size=L, mode='linear', align_corners=False).transpose(1, 2)
            level_out, level_attn = attention_layer(q_comp, k_comp, v_comp, attn_mask=attn_mask)
            level_outputs.append(level_out)
            level_attentions.append(level_attn)
        weights = F.softmax(self.level_weights, dim=0)
        concatenated = torch.cat(level_outputs, dim=-1)
        fused_output = self.fusion_proj(concatenated)
        avg_attention = torch.stack(level_attentions, dim=0).mean(dim=0)
        return fused_output, avg_attention
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Forward pass using wavelet-based multi-resolution attention.
        
        Args:
            queries: [B, L, D] query tensor
            keys: [B, L, D] key tensor  
            values: [B, L, D] value tensor
            attn_mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        B, L, D = queries.shape
        
        # Wavelet decomposition of queries, keys, values
        _, q_components = self.wavelet_decomp(queries)
        _, k_components = self.wavelet_decomp(keys)
        _, v_components = self.wavelet_decomp(values)
        
        # Apply attention at each decomposition level
        level_outputs = []
        level_attentions = []
        
        for i, (q_comp, k_comp, v_comp, attention_layer) in enumerate(
            zip(q_components, k_components, v_components, self.level_attentions)
        ):
            # Ensure components have the right shape
            if q_comp.size(1) != L:
                q_comp = F.interpolate(q_comp.transpose(1, 2), size=L, mode='linear', align_corners=False).transpose(1, 2)
                k_comp = F.interpolate(k_comp.transpose(1, 2), size=L, mode='linear', align_corners=False).transpose(1, 2)
                v_comp = F.interpolate(v_comp.transpose(1, 2), size=L, mode='linear', align_corners=False).transpose(1, 2)
            
            # Apply attention
            level_out, level_attn = attention_layer(q_comp, k_comp, v_comp, attn_mask=attn_mask)
            level_outputs.append(level_out)
            level_attentions.append(level_attn)
        
        # Weighted fusion of multi-resolution outputs
        weights = F.softmax(self.level_weights, dim=0)
        
        # Concatenate and fuse
        concatenated = torch.cat(level_outputs, dim=-1)  # [B, L, D*(levels+1)]
        fused_output = self.fusion_proj(concatenated)     # [B, L, D]
        
        # Aggregate attention weights (simple average)
        avg_attention = torch.stack(level_attentions, dim=0).mean(dim=0)
        
        return fused_output, avg_attention


class WaveletDecomposition(nn.Module):
    """
    Learnable wavelet decomposition for multi-resolution analysis.
    
    This module learns wavelet-like filters to decompose time series
    into multiple frequency bands for hierarchical processing.
    """
    
    def __init__(self, input_dim, levels=3, kernel_size=4):
        super(WaveletDecomposition, self).__init__()
        logger.info(f"Initializing WaveletDecomposition: input_dim={input_dim}, levels={levels}")
        
        self.levels = levels
        self.kernel_size = kernel_size
        
        # Learnable wavelet filters (low-pass and high-pass)
        self.low_pass = nn.ModuleList([
            nn.Conv1d(input_dim, input_dim, kernel_size, stride=2, padding=kernel_size//2, groups=input_dim)
            for _ in range(levels)
        ])
        
        self.high_pass = nn.ModuleList([
            nn.Conv1d(input_dim, input_dim, kernel_size, stride=2, padding=kernel_size//2, groups=input_dim)
            for _ in range(levels)
        ])
        
        # Reconstruction weights for combining components
        self.recon_weights = nn.Parameter(torch.ones(levels + 1) / (levels + 1))
        
        self.output_dim_multiplier = 1
    
    def forward(self, x):
        """
        Perform wavelet decomposition.
        
        Args:
            x: [B, L, D] input tensor
            
        Returns:
            Tuple of (reconstructed, components_list)
        """
        B, L, D = x.shape
        x_conv = x.transpose(1, 2)  # [B, D, L] for conv1d
        
        components = []
        current = x_conv
        
        # Multi-level decomposition
        for level in range(self.levels):
            # Apply low-pass and high-pass filters
            low = self.low_pass[level](current)
            high = self.high_pass[level](current)
            
            # Store high-frequency component
            components.append(high.transpose(1, 2))  # Convert back to [B, L', D]
            
            # Continue with low-frequency component
            current = low
            
        # Store final low-frequency component
        components.append(current.transpose(1, 2))
        
        # Weighted reconstruction
        weights = F.softmax(self.recon_weights, dim=0)
        
        # Upsample and combine all components
        reconstructed = torch.zeros_like(x)
        for i, (comp, weight) in enumerate(zip(components, weights)):
            if comp.size(1) < L:
                # Upsample to original length
                comp_upsampled = F.interpolate(
                    comp.transpose(1, 2), size=L, mode='linear', align_corners=False
                ).transpose(1, 2)
            else:
                comp_upsampled = comp
                
            reconstructed += comp_upsampled * weight
            
        return reconstructed, components


class AdaptiveWaveletAttention(BaseAttention):
    """
    Adaptive wavelet attention with learnable decomposition levels.
    
    This component adaptively selects the optimal number of decomposition
    levels based on the input characteristics.
    """
    
    def __init__(self, d_model, n_heads, max_levels=5):
        super(AdaptiveWaveletAttention, self).__init__()
        logger.info(f"Initializing AdaptiveWaveletAttention: d_model={d_model}, max_levels={max_levels}")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_levels = max_levels
        
        # Level selection network
        self.level_selector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, max_levels),
            nn.Softmax(dim=-1)
        )
        
        # Multiple wavelet attention modules for different levels
        self.wavelet_attentions = nn.ModuleList([
            WaveletAttention(d_model, n_heads, levels=i+1)
            for i in range(max_levels)
        ])
        
        self.output_dim_multiplier = 1
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Forward pass with adaptive level selection.
        
        Args:
            queries: [B, L, D] query tensor
            keys: [B, L, D] key tensor
            values: [B, L, D] value tensor
            attn_mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        B, L, D = queries.shape
        
        # Select optimal decomposition levels based on input characteristics
        input_summary = queries.mean(dim=1)  # [B, D]
        level_weights = self.level_selector(input_summary)  # [B, max_levels]
        
        # Compute outputs for all levels
        level_outputs = []
        level_attentions = []
        
        for i, wavelet_attn in enumerate(self.wavelet_attentions):
            output, attention = wavelet_attn(queries, keys, values, attn_mask)
            level_outputs.append(output)
            level_attentions.append(attention)
        
        # Weighted combination based on adaptive selection
        level_outputs = torch.stack(level_outputs, dim=-1)  # [B, L, D, max_levels]
        level_attentions = torch.stack(level_attentions, dim=-1)  # [B, H, L, L, max_levels]
        
        # Apply level weights
        final_output = torch.sum(level_outputs * level_weights.view(B, 1, 1, -1), dim=-1)
        final_attention = torch.sum(level_attentions * level_weights.view(B, 1, 1, 1, -1), dim=-1)
        
        return final_output, final_attention


class MultiScaleWaveletAttention(BaseAttention):
    """
    Multi-scale wavelet attention for capturing patterns at different time scales.
    
    This component applies wavelet attention at multiple predefined scales
    and combines the results for comprehensive temporal modeling.
    """
    
    def __init__(self, d_model, n_heads, scales=[1, 2, 4, 8]):
        super(MultiScaleWaveletAttention, self).__init__()
        logger.info(f"Initializing MultiScaleWaveletAttention: scales={scales}")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.scales = scales
        
        # Wavelet attention for each scale
        self.scale_attentions = nn.ModuleList([
            WaveletAttention(d_model, n_heads, levels=3)
            for _ in scales
        ])
        
        # Scale fusion
        self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))
        self.scale_proj = nn.Linear(d_model * len(scales), d_model)
        
        self.output_dim_multiplier = 1
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Forward pass with multi-scale wavelet processing.
        
        Args:
            queries: [B, L, D] query tensor
            keys: [B, L, D] key tensor
            values: [B, L, D] value tensor
            attn_mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        B, L, D = queries.shape
        
        scale_outputs = []
        scale_attentions = []
        
        for scale, scale_attn in zip(self.scales, self.scale_attentions):
            if scale == 1:
                # Original scale
                q_scaled, k_scaled, v_scaled = queries, keys, values
            else:
                # Downsample for larger scales
                target_len = max(L // scale, 1)
                q_scaled = F.interpolate(queries.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(1, 2)
                k_scaled = F.interpolate(keys.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(1, 2)
                v_scaled = F.interpolate(values.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(1, 2)
            
            # Apply wavelet attention at this scale
            output, attention = scale_attn(q_scaled, k_scaled, v_scaled, attn_mask)
            
            # Upsample back to original length if needed
            if output.size(1) != L:
                output = F.interpolate(output.transpose(1, 2), size=L, mode='linear', align_corners=False).transpose(1, 2)
            
            scale_outputs.append(output)
            scale_attentions.append(attention)
        
        # Weighted fusion of multi-scale outputs
        weights = F.softmax(self.scale_weights, dim=0)
        
        # Concatenate and project
        concatenated = torch.cat(scale_outputs, dim=-1)  # [B, L, D*len(scales)]
        fused_output = self.scale_proj(concatenated)      # [B, L, D]
        
        # Average attention weights
        avg_attention = torch.stack(scale_attentions, dim=0).mean(dim=0)
        
        return fused_output, avg_attention


# Migrated Functions  


# Registry function for attention components
def get_attention_component(name: str, config: ComponentConfig = None, **kwargs):
    """Get attention component by name"""
    # This will be implemented based on the migrated components
    pass

def register_attention_components(registry):
    """Register all attention components with the registry"""
    # This will be implemented to register all migrated components
    pass
