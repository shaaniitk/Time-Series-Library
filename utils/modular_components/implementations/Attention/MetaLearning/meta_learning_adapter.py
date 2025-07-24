"""
MetaLearningAdapter: MAML-style Meta-Learning Adapter with gradient-based fast adaptation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseMetaLearningAdapter
import logging

logger = logging.getLogger(__name__)

class MetaLearningAdapter(BaseMetaLearningAdapter):
    """MAML-style Meta-Learning Adapter with proper gradient-based fast adaptation."""
    def __init__(self, d_model, n_heads=None, adaptation_steps=5, meta_lr=0.01, inner_lr=0.1, dropout=0.1):
        super().__init__()
        logger.info(f"Initializing MAML MetaLearningAdapter: adaptation_steps={adaptation_steps}")
        self.adaptation_steps = adaptation_steps
        self.d_model = d_model
        self.inner_lr = inner_lr
        self.output_dim_multiplier = 1
        self.base_network = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.meta_lr = nn.Parameter(torch.tensor(meta_lr))
        self.context_encoder = nn.Sequential(
            nn.Linear(2 * d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4)
        )
        self.context_projector = nn.Linear(d_model // 4, d_model)
        self.task_attention = nn.MultiheadAttention(d_model, n_heads or 4, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def extract_task_context(self, support_set):
        support_mean = torch.mean(support_set, dim=1)
        support_std = torch.std(support_set, dim=1)
        context_features = torch.cat([support_mean, support_std], dim=-1)
        context = self.context_encoder(context_features)
        return context

    def fast_adaptation(self, support_set, query_set, num_steps=None):
        if num_steps is None:
            num_steps = self.adaptation_steps
        base_params = {name: param for name, param in self.base_network.named_parameters()}
        adapted_params = {name: param.clone() for name, param in base_params.items()}
        total_adaptation_loss = 0.0
        for step in range(num_steps):
            adapted_output = self._forward_with_params(support_set, adapted_params)
            adaptation_loss = F.mse_loss(adapted_output, support_set)
            total_adaptation_loss += adaptation_loss.item()
            gradients = torch.autograd.grad(
                adaptation_loss,
                adapted_params.values(),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )
            for (name, param), grad in zip(adapted_params.items(), gradients):
                if grad is not None:
                    adapted_params[name] = param - self.inner_lr * grad
        return adapted_params, total_adaptation_loss / num_steps

    def _forward_with_params(self, x, params):
        current = x
        param_names = list(params.keys())
        weight_bias_pairs = []
        for i in range(0, len(param_names), 2):
            if i + 1 < len(param_names):
                weight_name = param_names[i] if 'weight' in param_names[i] else param_names[i+1]
                bias_name = param_names[i+1] if 'bias' in param_names[i+1] else param_names[i]
                weight_bias_pairs.append((weight_name, bias_name))
        for i, (weight_name, bias_name) in enumerate(weight_bias_pairs):
            current = F.linear(current, params[weight_name], params.get(bias_name))
            if i < len(weight_bias_pairs) - 1:
                current = F.relu(current)
        return current

    def forward(self, query, key, value, attn_mask=None, support_set=None):
        if support_set is None:
            support_set = key
        residual = query
        if self.training and support_set is not None:
            task_context = self.extract_task_context(support_set)
            adapted_params, adaptation_loss = self.fast_adaptation(support_set, query)
            adapted_output = self._forward_with_params(query, adapted_params)
            context_projected = self.context_projector(task_context)
            context_expanded = context_projected.unsqueeze(1).expand(-1, query.shape[1], -1)
            query_with_context = query + context_expanded
            attended_output, attention_weights = self.task_attention(
                query_with_context, adapted_output, adapted_output, attn_mask=attn_mask
            )
            final_output = (adapted_output + attended_output) / 2
        else:
            base_output = self.base_network(query)
            final_output, attention_weights = self.task_attention(query, base_output, base_output, attn_mask=attn_mask)
        final_output = self.dropout(final_output)
        final_output = final_output + residual
        return final_output, attention_weights if 'attention_weights' in locals() else None
