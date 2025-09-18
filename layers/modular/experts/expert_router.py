"""
Expert Routing Mechanisms for Mixture of Experts

Provides various routing strategies to select and weight experts
based on input characteristics and learned patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from abc import ABC, abstractmethod
import numpy as np


class BaseRouter(nn.Module, ABC):
    """Abstract base class for expert routers."""
    
    def __init__(self, d_model: int, num_experts: int, routing_strategy: str):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.routing_strategy = routing_strategy
        
        # Load balancing parameters
        self.load_balancing = True
        self.load_balancing_weight = 0.01
        
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict]:
        """
        Route input to experts.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            routing_weights: Expert weights [batch_size, num_experts]
            routing_info: Additional routing information
        """
        pass
    
    def compute_load_balancing_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """Compute load balancing loss to encourage uniform expert usage."""
        if not self.load_balancing:
            return torch.tensor(0.0, device=routing_weights.device)
        
        # Compute expert usage frequency
        expert_usage = routing_weights.mean(dim=0)  # [num_experts]
        
        # Encourage uniform distribution
        uniform_target = torch.ones_like(expert_usage) / self.num_experts
        load_loss = F.mse_loss(expert_usage, uniform_target)
        
        return self.load_balancing_weight * load_loss


class ExpertRouter(BaseRouter):
    """Simple learned routing with MLP."""
    
    def __init__(self, d_model: int, num_experts: int, hidden_dim: Optional[int] = None):
        super().__init__(d_model, num_experts, 'learned')
        
        hidden_dim = hidden_dim or d_model // 2
        
        self.router = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        
        # Temperature for softmax
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict]:
        # Aggregate sequence dimension
        x_agg = x.mean(dim=1)  # [batch_size, d_model]
        
        # Compute routing logits
        routing_logits = self.router(x_agg)  # [batch_size, num_experts]
        
        # Apply temperature scaling
        routing_logits = routing_logits / torch.clamp(self.temperature, min=0.1)
        
        # Softmax to get weights
        routing_weights = F.softmax(routing_logits, dim=-1)
        
        # Compute load balancing loss
        load_loss = self.compute_load_balancing_loss(routing_weights)
        
        routing_info = {
            'routing_logits': routing_logits,
            'temperature': self.temperature.item(),
            'load_balancing_loss': load_loss,
            'expert_entropy': -torch.sum(routing_weights * torch.log(routing_weights + 1e-8), dim=-1).mean()
        }
        
        return routing_weights, routing_info


class AdaptiveExpertRouter(BaseRouter):
    """Adaptive routing that considers input characteristics."""
    
    def __init__(self, d_model: int, num_experts: int, adaptation_features: int = 8):
        super().__init__(d_model, num_experts, 'adaptive')
        
        self.adaptation_features = adaptation_features
        
        # Input characteristic analyzer
        self.characteristic_analyzer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, adaptation_features),
            nn.Tanh()
        )
        
        # Expert affinity matrix
        self.expert_affinity = nn.Parameter(
            torch.randn(adaptation_features, num_experts) * 0.1
        )
        
        # Context-aware routing
        self.context_router = nn.Sequential(
            nn.Linear(d_model + adaptation_features, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_experts)
        )
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict]:
        batch_size, seq_len, d_model = x.shape
        
        # Analyze input characteristics
        x_agg = x.mean(dim=1)  # [batch_size, d_model]
        characteristics = self.characteristic_analyzer(x_agg)  # [batch_size, adaptation_features]
        
        # Compute affinity-based routing
        affinity_scores = torch.mm(characteristics, self.expert_affinity)  # [batch_size, num_experts]
        
        # Context-aware routing
        context_input = torch.cat([x_agg, characteristics], dim=-1)
        context_scores = self.context_router(context_input)  # [batch_size, num_experts]
        
        # Gating between affinity and context
        gate_weight = self.gate(x_agg)  # [batch_size, 1]
        
        # Combine routing scores
        routing_logits = gate_weight * affinity_scores + (1 - gate_weight) * context_scores
        routing_weights = F.softmax(routing_logits, dim=-1)
        
        # Load balancing
        load_loss = self.compute_load_balancing_loss(routing_weights)
        
        routing_info = {
            'characteristics': characteristics,
            'affinity_scores': affinity_scores,
            'context_scores': context_scores,
            'gate_weights': gate_weight,
            'load_balancing_loss': load_loss,
            'routing_diversity': torch.std(routing_weights, dim=-1).mean()
        }
        
        return routing_weights, routing_info


class AttentionBasedRouter(BaseRouter):
    """Attention-based routing using expert embeddings."""
    
    def __init__(self, d_model: int, num_experts: int, num_heads: int = 4):
        super().__init__(d_model, num_experts, 'attention')
        
        self.num_heads = num_heads
        
        # Expert embeddings
        self.expert_embeddings = nn.Parameter(
            torch.randn(num_experts, d_model) * 0.1
        )
        
        # Multi-head attention for routing
        self.routing_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=0.1, batch_first=True
        )
        
        # Query projection
        self.query_projection = nn.Linear(d_model, d_model)
        
        # Final routing layer
        self.routing_head = nn.Linear(d_model, 1)
        
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict]:
        batch_size, seq_len, d_model = x.shape
        
        # Project input to query
        query = self.query_projection(x.mean(dim=1, keepdim=True))  # [batch_size, 1, d_model]
        
        # Expert embeddings as keys and values
        expert_emb = self.expert_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_experts, d_model]
        
        # Attention-based routing
        attended_experts, attention_weights = self.routing_attention(
            query, expert_emb, expert_emb
        )  # attended_experts: [batch_size, 1, d_model], attention_weights: [batch_size, 1, num_experts]
        
        # Extract routing weights
        routing_weights = attention_weights.squeeze(1)  # [batch_size, num_experts]
        
        # Additional routing refinement
        routing_scores = self.routing_head(attended_experts).squeeze(-1)  # [batch_size, 1]
        routing_weights = routing_weights * torch.sigmoid(routing_scores)
        
        # Renormalize
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        # Load balancing
        load_loss = self.compute_load_balancing_loss(routing_weights)
        
        routing_info = {
            'attention_weights': attention_weights,
            'attended_experts': attended_experts,
            'routing_scores': routing_scores,
            'load_balancing_loss': load_loss,
            'expert_embeddings': self.expert_embeddings
        }
        
        return routing_weights, routing_info


class HierarchicalRouter(BaseRouter):
    """Hierarchical routing with coarse-to-fine expert selection."""
    
    def __init__(self, d_model: int, num_experts: int, hierarchy_levels: int = 2):
        super().__init__(d_model, num_experts, 'hierarchical')
        
        self.hierarchy_levels = hierarchy_levels
        
        # Create hierarchical structure
        experts_per_level = [num_experts // (2 ** i) for i in range(hierarchy_levels)]
        experts_per_level[-1] = max(experts_per_level[-1], 1)  # Ensure at least 1 expert at top
        
        self.level_routers = nn.ModuleList()
        for level, num_level_experts in enumerate(experts_per_level):
            if level == 0:
                input_dim = d_model
            else:
                input_dim = d_model + experts_per_level[level - 1]
            
            router = nn.Sequential(
                nn.Linear(input_dim, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, num_level_experts)
            )
            self.level_routers.append(router)
        
        # Expert hierarchy mapping
        self.expert_hierarchy = self._build_expert_hierarchy(num_experts, hierarchy_levels)
        
    def _build_expert_hierarchy(self, num_experts: int, levels: int) -> Dict:
        """Build hierarchical mapping of experts."""
        hierarchy = {}
        experts_per_level = [num_experts // (2 ** i) for i in range(levels)]
        experts_per_level[-1] = max(experts_per_level[-1], 1)
        
        for level in range(levels):
            hierarchy[level] = list(range(experts_per_level[level]))
        
        return hierarchy
    
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict]:
        batch_size = x.size(0)
        x_agg = x.mean(dim=1)  # [batch_size, d_model]
        
        level_weights = []
        routing_input = x_agg
        
        # Forward through hierarchy levels
        for level, router in enumerate(self.level_routers):
            level_logits = router(routing_input)
            level_weight = F.softmax(level_logits, dim=-1)
            level_weights.append(level_weight)
            
            # Prepare input for next level
            if level < len(self.level_routers) - 1:
                routing_input = torch.cat([x_agg, level_weight], dim=-1)
        
        # Combine hierarchical weights to get final expert weights
        final_weights = self._combine_hierarchical_weights(level_weights)
        
        # Load balancing across all levels
        total_load_loss = sum(
            self.compute_load_balancing_loss(weights) for weights in level_weights
        )
        
        routing_info = {
            'level_weights': level_weights,
            'hierarchy_levels': self.hierarchy_levels,
            'load_balancing_loss': total_load_loss,
            'expert_hierarchy': self.expert_hierarchy
        }
        
        return final_weights, routing_info
    
    def _combine_hierarchical_weights(self, level_weights: List[torch.Tensor]) -> torch.Tensor:
        """Combine weights from different hierarchy levels."""
        # Simple combination: multiply weights down the hierarchy
        final_weights = level_weights[0]
        
        for level in range(1, len(level_weights)):
            # Expand lower level weights to match expert count
            expanded_weights = level_weights[level].repeat_interleave(
                final_weights.size(-1) // level_weights[level].size(-1), dim=-1
            )
            final_weights = final_weights * expanded_weights
        
        # Renormalize
        final_weights = F.softmax(final_weights, dim=-1)
        return final_weights


class DynamicRouter(BaseRouter):
    """Dynamic routing that adapts based on temporal context."""
    
    def __init__(self, d_model: int, num_experts: int, memory_size: int = 100):
        super().__init__(d_model, num_experts, 'dynamic')
        
        self.memory_size = memory_size
        
        # Routing memory
        self.register_buffer('routing_memory', torch.zeros(memory_size, d_model))
        self.register_buffer('memory_pointer', torch.zeros(1, dtype=torch.long))
        
        # Context encoder
        self.context_encoder = nn.LSTM(d_model, d_model // 2, batch_first=True)
        
        # Dynamic routing network
        self.dynamic_router = nn.Sequential(
            nn.Linear(d_model + d_model // 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_experts)
        )
        
        # Memory update gate
        self.memory_gate = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict]:
        batch_size, seq_len, d_model = x.shape
        x_agg = x.mean(dim=1)  # [batch_size, d_model]
        
        # Encode temporal context
        context_output, _ = self.context_encoder(x)
        context_agg = context_output.mean(dim=1)  # [batch_size, d_model//2]
        
        # Combine with memory context
        memory_context = self.routing_memory.mean(dim=0).unsqueeze(0).expand(batch_size, -1)
        
        # Dynamic routing
        routing_input = torch.cat([x_agg, context_agg], dim=-1)
        routing_logits = self.dynamic_router(routing_input)
        routing_weights = F.softmax(routing_logits, dim=-1)
        
        # Update memory
        if self.training:
            self._update_memory(x_agg)
        
        # Load balancing
        load_loss = self.compute_load_balancing_loss(routing_weights)
        
        routing_info = {
            'context_features': context_agg,
            'memory_context': memory_context,
            'load_balancing_loss': load_loss,
            'memory_utilization': (self.memory_pointer.item() / self.memory_size)
        }
        
        return routing_weights, routing_info
    
    def _update_memory(self, x_agg: torch.Tensor):
        """Update routing memory with current context."""
        batch_size = x_agg.size(0)
        
        for i in range(batch_size):
            # Decide whether to update memory
            gate_score = self.memory_gate(x_agg[i:i+1])
            
            if gate_score.item() > 0.5:  # Threshold for memory update
                # Update memory at current pointer
                ptr = self.memory_pointer.item()
                self.routing_memory[ptr] = x_agg[i].detach()
                
                # Update pointer
                self.memory_pointer[0] = (ptr + 1) % self.memory_size