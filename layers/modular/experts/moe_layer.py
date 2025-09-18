"""
Mixture of Experts Layer Implementation

Provides complete MoE layers that combine experts with routing mechanisms
for efficient and specialized processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple, Union
import math

from .base_expert import BaseExpert, ExpertOutput
from .expert_router import BaseRouter


class MoELayer(nn.Module):
    """
    Standard Mixture of Experts Layer.
    
    Combines multiple experts with a routing mechanism to process inputs
    based on learned or adaptive routing strategies.
    """
    
    def __init__(self, 
                 experts: List[BaseExpert], 
                 router: BaseRouter,
                 top_k: int = 2,
                 expert_capacity_factor: float = 1.25,
                 use_residual: bool = True):
        """
        Initialize MoE Layer.
        
        Args:
            experts: List of expert modules
            router: Routing mechanism
            top_k: Number of top experts to use per input
            expert_capacity_factor: Capacity factor for load balancing
            use_residual: Whether to use residual connections
        """
        super().__init__()
        
        self.experts = nn.ModuleList(experts)
        self.router = router
        self.top_k = min(top_k, len(experts))
        self.expert_capacity_factor = expert_capacity_factor
        self.use_residual = use_residual
        
        # Validation
        assert len(experts) > 0, "At least one expert required"
        assert self.top_k > 0, "top_k must be positive"
        
        # Expert capacity calculation
        self.expert_capacity = None  # Will be calculated dynamically
        
        # Output projection (if experts have different output dims)
        expert_dims = [expert.expert_dim for expert in experts]
        if len(set(expert_dims)) > 1:  # Different dimensions
            self.output_projection = nn.ModuleList([
                nn.Linear(dim, experts[0].d_model) for dim in expert_dims
            ])
        else:
            self.output_projection = None
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(experts[0].d_model)
        
        # Metrics tracking
        self.register_buffer('expert_usage_count', torch.zeros(len(experts)))
        self.register_buffer('total_tokens_processed', torch.zeros(1))
        
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through MoE layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            **kwargs: Additional arguments for experts
            
        Returns:
            output: Combined expert outputs
            moe_info: MoE routing and processing information
        """
        batch_size, seq_len, d_model = x.shape
        original_shape = x.shape
        
        # Flatten for expert processing
        x_flat = x.view(-1, d_model)  # [batch_size * seq_len, d_model]
        num_tokens = x_flat.size(0)
        
        # Get routing weights
        routing_weights, routing_info = self.router(x, **kwargs)  # [batch_size, num_experts]
        
        # Expand routing weights for all tokens
        routing_weights_expanded = routing_weights.unsqueeze(1).expand(-1, seq_len, -1)
        routing_weights_flat = routing_weights_expanded.contiguous().view(num_tokens, -1)
        
        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(
            routing_weights_flat, self.top_k, dim=-1
        )  # [num_tokens, top_k]
        
        # Renormalize top-k weights
        top_k_weights = F.softmax(top_k_weights, dim=-1)
        
        # Process through selected experts
        expert_outputs = []
        expert_confidences = []
        expert_metadata = []
        
        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]  # [num_tokens]
            expert_weights = top_k_weights[:, k]  # [num_tokens]
            
            # Group tokens by expert
            unique_experts = torch.unique(expert_indices)
            k_outputs = torch.zeros_like(x_flat)
            k_confidences = torch.zeros(num_tokens, 1, device=x.device)
            
            for expert_idx in unique_experts:
                expert_idx = expert_idx.item()
                
                # Find tokens for this expert
                token_mask = (expert_indices == expert_idx)
                if not token_mask.any():
                    continue
                
                # Get tokens for this expert
                expert_tokens = x_flat[token_mask]  # [num_expert_tokens, d_model]
                
                # Process through expert
                expert_output = self.experts[expert_idx](
                    expert_tokens.view(-1, 1, d_model), **kwargs
                )
                
                # Handle expert output
                if isinstance(expert_output, ExpertOutput):
                    output_tensor = expert_output.output.view(-1, d_model)
                    confidence_tensor = expert_output.confidence.view(-1, 1)
                else:
                    output_tensor = expert_output.view(-1, d_model)
                    confidence_tensor = torch.ones(output_tensor.size(0), 1, device=x.device)
                
                # Apply output projection if needed
                if self.output_projection is not None:
                    output_tensor = self.output_projection[expert_idx](output_tensor)
                
                # Store outputs
                k_outputs[token_mask] = output_tensor
                k_confidences[token_mask] = confidence_tensor
                
                # Update usage statistics
                if self.training:
                    self.expert_usage_count[expert_idx] += token_mask.sum().float()
            
            # Weight by routing weights
            weighted_output = k_outputs * expert_weights.unsqueeze(-1)
            weighted_confidence = k_confidences * expert_weights.unsqueeze(-1)
            
            expert_outputs.append(weighted_output)
            expert_confidences.append(weighted_confidence)
        
        # Combine expert outputs
        combined_output = sum(expert_outputs)  # [num_tokens, d_model]
        combined_confidence = sum(expert_confidences)  # [num_tokens, 1]
        
        # Reshape back to original shape
        output = combined_output.view(original_shape)
        confidence = combined_confidence.view(batch_size, seq_len, 1)
        
        # Apply residual connection and layer norm
        if self.use_residual:
            output = self.layer_norm(output + x)
        else:
            output = self.layer_norm(output)
        
        # Update total tokens processed
        if self.training:
            self.total_tokens_processed += num_tokens
        
        # Compile MoE information
        moe_info = {
            'routing_info': routing_info,
            'top_k_weights': top_k_weights,
            'top_k_indices': top_k_indices,
            'expert_usage': self.expert_usage_count.clone(),
            'total_tokens': self.total_tokens_processed.item(),
            'average_confidence': combined_confidence.mean().item(),
            'load_balancing_loss': routing_info.get('load_balancing_loss', 0.0)
        }
        
        return output, moe_info
    
    def get_expert_utilization(self) -> Dict[str, float]:
        """Get expert utilization statistics."""
        if self.total_tokens_processed.item() == 0:
            return {f'expert_{i}': 0.0 for i in range(len(self.experts))}
        
        utilization = self.expert_usage_count / self.total_tokens_processed
        return {f'expert_{i}': util.item() for i, util in enumerate(utilization)}
    
    def reset_statistics(self):
        """Reset usage statistics."""
        self.expert_usage_count.zero_()
        self.total_tokens_processed.zero_()


class SparseMoELayer(MoELayer):
    """
    Sparse Mixture of Experts Layer with capacity constraints.
    
    Implements capacity-based routing to ensure balanced expert usage
    and prevent expert overloading.
    """
    
    def __init__(self, 
                 experts: List[BaseExpert], 
                 router: BaseRouter,
                 top_k: int = 2,
                 expert_capacity_factor: float = 1.25,
                 use_residual: bool = True,
                 capacity_balancing: bool = True):
        """
        Initialize Sparse MoE Layer.
        
        Args:
            experts: List of expert modules
            router: Routing mechanism
            top_k: Number of top experts to use per input
            expert_capacity_factor: Capacity factor for load balancing
            use_residual: Whether to use residual connections
            capacity_balancing: Whether to enforce capacity constraints
        """
        super().__init__(experts, router, top_k, expert_capacity_factor, use_residual)
        
        self.capacity_balancing = capacity_balancing
        
        # Capacity tracking
        self.register_buffer('expert_capacities', torch.zeros(len(experts)))
        self.register_buffer('expert_loads', torch.zeros(len(experts)))
        
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass with capacity constraints."""
        batch_size, seq_len, d_model = x.shape
        num_tokens = batch_size * seq_len
        
        # Calculate expert capacities
        if self.capacity_balancing:
            capacity_per_expert = int(
                math.ceil(num_tokens * self.expert_capacity_factor / len(self.experts))
            )
            self.expert_capacities.fill_(capacity_per_expert)
            self.expert_loads.zero_()
        
        # Get routing weights
        routing_weights, routing_info = self.router(x, **kwargs)
        
        # Apply capacity constraints if enabled
        if self.capacity_balancing:
            routing_weights = self._apply_capacity_constraints(routing_weights, num_tokens)
        
        # Continue with standard MoE processing
        return self._process_with_constraints(x, routing_weights, routing_info, **kwargs)
    
    def _apply_capacity_constraints(self, routing_weights: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """Apply capacity constraints to routing weights."""
        batch_size, num_experts = routing_weights.shape
        
        # Calculate expected load per expert
        expected_loads = routing_weights.sum(dim=0) * (num_tokens / batch_size)
        
        # Apply capacity constraints
        constrained_weights = routing_weights.clone()
        
        for expert_idx in range(num_experts):
            if expected_loads[expert_idx] > self.expert_capacities[expert_idx]:
                # Scale down weights for overloaded experts
                scale_factor = self.expert_capacities[expert_idx] / expected_loads[expert_idx]
                constrained_weights[:, expert_idx] *= scale_factor
        
        # Renormalize
        constrained_weights = F.softmax(constrained_weights, dim=-1)
        
        return constrained_weights
    
    def _process_with_constraints(self, x: torch.Tensor, routing_weights: torch.Tensor, 
                                routing_info: Dict, **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Process inputs with capacity constraints."""
        batch_size, seq_len, d_model = x.shape
        original_shape = x.shape
        
        # Flatten for processing
        x_flat = x.view(-1, d_model)
        num_tokens = x_flat.size(0)
        
        # Expand routing weights
        routing_weights_expanded = routing_weights.unsqueeze(1).expand(-1, seq_len, -1)
        routing_weights_flat = routing_weights_expanded.contiguous().view(num_tokens, -1)
        
        # Select top-k with capacity awareness
        top_k_weights, top_k_indices = self._capacity_aware_top_k(routing_weights_flat)
        
        # Process through experts (similar to base class but with capacity tracking)
        output, expert_info = self._process_through_experts(
            x_flat, top_k_weights, top_k_indices, **kwargs
        )
        
        # Reshape and apply normalization
        output = output.view(original_shape)
        if self.use_residual:
            output = self.layer_norm(output + x)
        else:
            output = self.layer_norm(output)
        
        # Compile information
        moe_info = {
            'routing_info': routing_info,
            'top_k_weights': top_k_weights,
            'top_k_indices': top_k_indices,
            'expert_capacities': self.expert_capacities.clone(),
            'expert_loads': self.expert_loads.clone(),
            'capacity_utilization': (self.expert_loads / self.expert_capacities).mean().item(),
            **expert_info
        }
        
        return output, moe_info
    
    def _capacity_aware_top_k(self, routing_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select top-k experts with capacity awareness."""
        num_tokens, num_experts = routing_weights.shape
        
        # Standard top-k selection
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        if not self.capacity_balancing:
            return F.softmax(top_k_weights, dim=-1), top_k_indices
        
        # Apply capacity constraints
        constrained_weights = top_k_weights.clone()
        constrained_indices = top_k_indices.clone()
        
        for token_idx in range(num_tokens):
            for k in range(self.top_k):
                expert_idx = top_k_indices[token_idx, k].item()
                
                # Check if expert has capacity
                if self.expert_loads[expert_idx] >= self.expert_capacities[expert_idx]:
                    # Find alternative expert with capacity
                    alternative_found = False
                    for alt_k in range(self.top_k, num_experts):
                        if alt_k < routing_weights.size(-1):
                            alt_expert_idx = torch.topk(routing_weights[token_idx], alt_k + 1)[1][alt_k].item()
                            if self.expert_loads[alt_expert_idx] < self.expert_capacities[alt_expert_idx]:
                                constrained_indices[token_idx, k] = alt_expert_idx
                                constrained_weights[token_idx, k] = routing_weights[token_idx, alt_expert_idx]
                                alternative_found = True
                                break
                    
                    if not alternative_found:
                        # Zero out this expert if no alternative
                        constrained_weights[token_idx, k] = 0.0
                
                # Update load tracking
                if constrained_weights[token_idx, k] > 0:
                    expert_idx = constrained_indices[token_idx, k].item()
                    self.expert_loads[expert_idx] += 1
        
        # Renormalize weights
        constrained_weights = F.softmax(constrained_weights, dim=-1)
        
        return constrained_weights, constrained_indices
    
    def _process_through_experts(self, x_flat: torch.Tensor, top_k_weights: torch.Tensor, 
                               top_k_indices: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict]:
        """Process tokens through selected experts."""
        num_tokens, d_model = x_flat.shape
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        total_confidence = torch.zeros(num_tokens, 1, device=x_flat.device)
        
        # Process each k
        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]
            expert_weights = top_k_weights[:, k]
            
            # Group by expert
            unique_experts = torch.unique(expert_indices)
            
            for expert_idx in unique_experts:
                expert_idx = expert_idx.item()
                
                # Get tokens for this expert
                token_mask = (expert_indices == expert_idx)
                if not token_mask.any():
                    continue
                
                expert_tokens = x_flat[token_mask]
                expert_token_weights = expert_weights[token_mask]
                
                # Process through expert
                expert_output = self.experts[expert_idx](
                    expert_tokens.view(-1, 1, d_model), **kwargs
                )
                
                # Handle output
                if isinstance(expert_output, ExpertOutput):
                    output_tensor = expert_output.output.view(-1, d_model)
                    confidence_tensor = expert_output.confidence.view(-1, 1)
                else:
                    output_tensor = expert_output.view(-1, d_model)
                    confidence_tensor = torch.ones(output_tensor.size(0), 1, device=x_flat.device)
                
                # Apply projection if needed
                if self.output_projection is not None:
                    output_tensor = self.output_projection[expert_idx](output_tensor)
                
                # Weight and accumulate
                weighted_output = output_tensor * expert_token_weights.unsqueeze(-1)
                weighted_confidence = confidence_tensor * expert_token_weights.unsqueeze(-1)
                
                output[token_mask] += weighted_output
                total_confidence[token_mask] += weighted_confidence
        
        expert_info = {
            'average_confidence': total_confidence.mean().item(),
            'expert_usage': self.expert_usage_count.clone()
        }
        
        return output, expert_info