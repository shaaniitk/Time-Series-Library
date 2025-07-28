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

class Top1Gating(nn.Module):
    """
    Top-1 Gating mechanism (router).
    
    Selects the single best expert for each token in the sequence.
    """
    def __init__(self, d_model, num_experts, use_aux_loss=True):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts)
        self.use_aux_loss = use_aux_loss
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Determines which expert to route each token to.
        
        Returns:
            A tuple of (indices, combined_weights, separate_weights, aux_loss).
        """
        logits = self.gate(x)
        
        # Get the index of the top expert
        top_expert_indices = logits.argmax(dim=-1)
        
        # Create a one-hot mask for the chosen experts
        one_hot_mask = F.one_hot(top_expert_indices, num_classes=self.gate.out_features).float()
        
        # Calculate soft weights for combining expert outputs
        soft_weights = self.softmax(logits)
        
        # The final weight for combining outputs is the product of the hard mask and soft weights
        combined_weights = (one_hot_mask * soft_weights).unsqueeze(-1)
        
        # Optional: Auxiliary load balancing loss
        # This encourages the router to use all experts roughly equally.
        aux_loss = 0.0
        if self.use_aux_loss and self.training:
            # Calculate the fraction of tokens routed to each expert
            tokens_per_expert = one_hot_mask.sum(dim=0)
            fraction_per_expert = tokens_per_expert / x.size(0)
            
            # Calculate the router's dispatch probability for each expert
            dispatch_prob_per_expert = soft_weights.mean(dim=0)
            
            # Load balancing loss from the Switch Transformer paper
            aux_loss = self.gate.out_features * torch.sum(fraction_per_expert * dispatch_prob_per_expert)

        return top_expert_indices, combined_weights, soft_weights, aux_loss

class GatedMoEFFN(nn.Module):
    """
    A Gated Feed-Forward Network using a Mixture of Experts.
    
    Args:
        d_model (int): Dimension of the model.
        d_ff (int): Dimension of the feed-forward layer in each expert.
        num_experts (int): The number of expert networks to use.
        dropout (float): Dropout probability.
    """
    def __init__(self, d_model, d_ff, num_experts=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        
        # Create the expert networks
        self.experts = nn.ModuleList([
            GatedExpert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])
        
        # Create the gating network
        self.gating_network = Top1Gating(d_model, num_experts)

    def forward(self, x):
        """
        Forward pass for the MoE layer.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            The output of the MoE layer and the auxiliary loss.
        """
        batch_size, seq_len, d_model = x.shape
        x = x.view(-1, d_model) # Flatten to [batch*seq_len, d_model]
        
        # Get routing decisions from the gating network
        top_indices, combined_weights, _, aux_loss = self.gating_network(x)
        
        # Initialize a tensor to store the outputs of all experts
        expert_outputs = torch.zeros(x.shape[0], self.num_experts, d_model, device=x.device)
        
        # Pass tokens through their assigned experts
        for i, expert in enumerate(self.experts):
            # Find which tokens are assigned to this expert
            expert_mask = (top_indices == i)
            if expert_mask.any():
                # Process only the relevant tokens
                expert_outputs[expert_mask, i] = expert(x[expert_mask])
        
        # Combine the expert outputs using the weights from the gating network
        # (B*L, num_experts, d_model) * (B*L, num_experts, 1) -> (B*L, num_experts, d_model)
        weighted_outputs = expert_outputs * combined_weights
        
        # Sum the weighted outputs to get the final result
        final_output = weighted_outputs.sum(dim=1)
        
        # Reshape back to the original sequence format
        final_output = final_output.view(batch_size, seq_len, d_model)
        
        return final_output, aux_loss
