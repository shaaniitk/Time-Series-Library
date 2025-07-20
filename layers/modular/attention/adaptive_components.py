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