"""
Adaptive Attention Components for Modular Autoformer

This module implements adaptive mechanisms like meta-learning and
mixture-of-experts, adapted for the modular attention framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseAttention
from utils.logger import logger

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
        
        # Core MAML parameters - these will be adapted
        self.fast_weights = nn.ParameterDict({
            'layer1_weight': nn.Parameter(torch.randn(d_model, d_model) * 0.01),
            'layer1_bias': nn.Parameter(torch.zeros(d_model)),
            'layer2_weight': nn.Parameter(torch.randn(d_model, d_model) * 0.01),
            'layer2_bias': nn.Parameter(torch.zeros(d_model)),
            'output_weight': nn.Parameter(torch.randn(d_model, d_model) * 0.01),
            'output_bias': nn.Parameter(torch.zeros(d_model))
        })
        
        # Meta-learning rate (learnable)
        self.meta_lr_param = nn.Parameter(torch.tensor(meta_lr))
        
        # Task context encoder for support set analysis
        self.support_encoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model)
        )
        
        # Task-specific adaptation network
        self.task_adapter = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, attn_mask=None, support_set=None, support_labels=None):
        """
        Forward pass with optional MAML-style fast adaptation.
        
        Args:
            query: [B, L, D] query tensor
            key: Optional key tensor (not used in this implementation, for compatibility)
            value: Optional value tensor (not used in this implementation, for compatibility) 
            attn_mask: Optional attention mask
            support_set: [B, S, D] support examples for adaptation
            support_labels: [B, S, D] support labels for supervised adaptation
            
        Returns:
            Tuple of (adapted_output, None)
        """
        B, L, D = query.shape
        
        if support_set is not None and self.training:
            # Perform MAML-style fast adaptation
            adapted_output = self._maml_forward(query, support_set, support_labels)
        else:
            # Standard forward pass without adaptation
            adapted_output = self._standard_forward(query)
        
        return adapted_output, None
    
    def _maml_forward(self, query, support_set, support_labels=None):
        """
        MAML-style forward pass with gradient-based adaptation.
        """
        B, L, D = query.shape
        B_s, S, D_s = support_set.shape
        
        # Ensure support set has same feature dimension
        assert D == D_s, f"Query dim {D} != Support dim {D_s}"
        
        # Extract task context from support set
        support_context = self.support_encoder(support_set.mean(dim=1))  # [B, D]
        
        # Initialize adapted parameters as copies of fast_weights
        adapted_params = {}
        for name, param in self.fast_weights.items():
            adapted_params[name] = param.clone()
        
        # MAML inner loop: adapt parameters using support set
        for step in range(self.adaptation_steps):
            # Forward pass through support set with current adapted parameters
            support_pred = self._forward_with_params(support_set, adapted_params)
            
            # Compute adaptation loss
            if support_labels is not None:
                # Supervised adaptation
                adapt_loss = F.mse_loss(support_pred, support_labels)
            else:
                # Self-supervised adaptation (reconstruction)
                adapt_loss = F.mse_loss(support_pred, support_set)
            
            # Compute gradients and update adapted parameters
            gradients = torch.autograd.grad(
                adapt_loss, 
                adapted_params.values(), 
                create_graph=True, 
                retain_graph=True,
                allow_unused=True
            )
            
            # Update adapted parameters using computed gradients
            for (name, param), grad in zip(adapted_params.items(), gradients):
                if grad is not None:
                    adapted_params[name] = param - self.inner_lr * grad
        
        # Apply adapted parameters to query
        adapted_query = self._forward_with_params(query, adapted_params)
        
        # Apply task-specific adaptation based on support context
        task_adaptation = self.task_adapter(support_context).unsqueeze(1)  # [B, 1, D]
        adapted_output = adapted_query + task_adaptation
        
        return self.dropout(adapted_output)
    
    def _forward_with_params(self, x, params):
        """
        Forward pass using specific parameter set.
        """
        # Layer 1
        h1 = F.linear(x, params['layer1_weight'], params['layer1_bias'])
        h1 = F.relu(h1)
        
        # Layer 2  
        h2 = F.linear(h1, params['layer2_weight'], params['layer2_bias'])
        h2 = F.relu(h2)
        
        # Output layer
        output = F.linear(h2, params['output_weight'], params['output_bias'])
        
        return output
    
    def _standard_forward(self, query):
        """
        Standard forward pass without adaptation.
        """
        # Use base fast_weights for standard forward pass
        return self._forward_with_params(query, self.fast_weights)
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
        
        # Encode task context - just use support mean  
        context = self.support_encoder(support_mean)  # [B, D] -> [B, D]
        return context


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