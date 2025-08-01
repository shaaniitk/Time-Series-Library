"""
Gated Mixture of Experts Feed-Forward Network

This module implements a Gated Feed-Forward Network (FFN) using a
Mixture of Experts (MoE) layer with a top-1 gating mechanism. This allows
the model to learn specialized expert sub-networks and dynamically route
tokens to the most appropriate one.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedExpert(nn.Module):
    """
    A single gated feed-forward expert network.
    Implements a Gated Linear Unit (GLU) for richer representations.
    """
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_model, d_ff)
        self.w_3 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        """Forward pass for the gated expert."""
        x_1 = self.w_1(x)
        x_2 = self.w_2(x)
        hidden = self.activation(x_1) * x_2  # Gated Linear Unit
        return self.w_3(self.dropout(hidden))

class TopKGating(nn.Module):
    """
    Top-k Gating mechanism (router).
    
    Selects the top-k experts for each token and computes their weights.
    Includes optional noise for improved load balancing during training.
    """
    def __init__(self, d_model, num_experts, top_k=2, use_aux_loss=True, noise_epsilon=1e-2):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts)
        self.use_aux_loss = use_aux_loss
        self.top_k = top_k
        self.noise_epsilon = noise_epsilon
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Determines which expert to route each token to.
        
        Returns:
            A tuple of (combined_weights, top_indices, aux_loss).
        """
        logits = self.gate(x)
        
        # Add noise for load balancing during training
        if self.training and self.noise_epsilon > 0:
            noise = torch.randn_like(logits) * self.noise_epsilon
            logits += noise
        
        # Get the top-k experts and their logits
        top_k_logits, top_indices = torch.topk(logits, self.top_k, dim=-1)
        
        # The weights are the softmax of the top-k logits
        top_k_weights = self.softmax(top_k_logits)
        
        # Create a sparse tensor for the combined weights
        # This will have shape [num_tokens, num_experts]
        combined_weights = torch.zeros_like(logits)
        combined_weights.scatter_(1, top_indices, top_k_weights)
        
        # Optional: Auxiliary load balancing loss
        aux_loss = 0.0
        if self.use_aux_loss and self.training:
            # Use the full softmax over all experts for the loss calculation
            full_softmax_probs = self.softmax(logits)
            
            # Calculate the fraction of tokens routed to each expert
            # We count how many times each expert was in the top-k
            top_k_one_hot = F.one_hot(top_indices, num_classes=self.gate.out_features).float()
            tokens_per_expert = top_k_one_hot.sum(dim=(0, 1))
            fraction_per_expert = tokens_per_expert / x.size(0) # Changed from x.size(0) to top_k_one_hot.size(0)
            
            # Calculate the router's dispatch probability for each expert
            dispatch_prob_per_expert = full_softmax_probs.mean(dim=0)
            
            # Load balancing loss from the Switch Transformer paper
            # The loss is the dot product of the fraction of tokens and the dispatch probability
            # We multiply by num_experts to keep the loss scale consistent with the original paper
            aux_loss = self.gate.out_features * torch.sum(fraction_per_expert * dispatch_prob_per_expert)

        return combined_weights.unsqueeze(-1), top_indices, aux_loss

class GatedMoEFFN(nn.Module):
    """
    A Gated Feed-Forward Network using a Mixture of Experts.
    
    Args:
        d_model (int): Dimension of the model.
        d_ff (int): Dimension of the feed-forward layer in each expert.
        num_experts (int): The number of expert networks to use.
        dropout (float): Dropout probability.
        top_k (int): The number of experts to route each token to.
    """
    def __init__(self, d_model, d_ff, num_experts=4, top_k=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        # Create the expert networks
        self.experts = nn.ModuleList([
            GatedExpert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])
        
        # Create the gating network
        self.gating_network = TopKGating(d_model, num_experts, top_k)

    def forward(self, x):
        """
        Forward pass for the MoE layer.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            The output of the MoE layer and the auxiliary loss.
        """
        batch_size, seq_len, d_model = x.shape
        x = x.reshape(-1, d_model) # Flatten to [batch*seq_len, d_model] using reshape instead of view for non-contiguous tensors
        
        # Get routing decisions: gating_weights are [B*L, num_experts, 1], top_indices are [B*L, top_k]
        gating_weights, top_indices, aux_loss = self.gating_network(x)
        
        # Initialize final output tensor
        final_output = torch.zeros_like(x)
        
        # Iterate over experts to process tokens in batches. This is a clear and correct approach.
        for i in range(self.num_experts):
            # Find all tokens that should be routed to this expert (i.e., it is in their top_k)
            token_indices_for_expert = (top_indices == i).any(dim=1)

            if token_indices_for_expert.any():
                # Get the tokens for this expert
                tokens_for_expert = x[token_indices_for_expert]

                # Process them through the expert
                expert_output = self.experts[i](tokens_for_expert)

                # Get the gating weights for these specific tokens and this expert
                expert_weights = gating_weights[token_indices_for_expert, i]

                # Weight the expert output and add it to the final output (scatter-like operation)
                final_output[token_indices_for_expert] += expert_output * expert_weights
        
        # Reshape back to the original sequence format using reshape instead of view for non-contiguous tensors
        final_output = final_output.reshape(batch_size, seq_len, d_model)
        
        # Debug logging for auxiliary loss
        if self.training and aux_loss > 0:
            from utils.logger import logger
            if logger.isEnabledFor(10):  # DEBUG level
                logger.debug(f"GatedMoEFFN aux_loss: {aux_loss.item():.6f}")
        
        return final_output, aux_loss
