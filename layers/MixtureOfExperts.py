
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Type

class MixtureOfExperts(nn.Module):
    """
    A generalized Mixture of Experts (MoE) component.

    This component can wrap any set of 'expert' modules and uses a gating
    network to dynamically select and combine their outputs.
    """

    def __init__(self, input_dim: int, num_experts: int, expert_class: Type[nn.Module], expert_configs: List[dict]):
        """
        Args:
            input_dim (int): The input dimension for the gating network.
            num_experts (int): The number of expert modules.
            expert_class (Type[nn.Module]): The class of the expert modules to be instantiated.
            expert_configs (List[dict]): A list of configuration dictionaries, one for each expert.
        """
        super().__init__()
        assert num_experts == len(expert_configs), "Number of experts must match the number of expert configurations."

        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1)
        )

        self.experts = nn.ModuleList([expert_class(**config) for config in expert_configs])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Mixture of Experts.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The combined output from the experts.
        """
        # The gating network determines the weight for each expert based on the input.
        # For a sequence, we might want to use the mean of the sequence to decide on the expert weights.
        if x.dim() > 2:
            gating_input = x.mean(dim=1)
        else:
            gating_input = x
            
        expert_weights = self.gating_network(gating_input)

        # Get the output from each expert
        expert_outputs = [expert(x) for expert in self.experts]
        
        # Stack the expert outputs
        # expert_outputs is a list of tensors of shape [batch_size, seq_len, output_dim]
        # We stack them along a new dimension to get [batch_size, seq_len, output_dim, num_experts]
        stacked_expert_outputs = torch.stack(expert_outputs, dim=-1)

        # Weight the expert outputs by the gating network's decisions.
        # expert_weights shape: [batch_size, num_experts]
        # We need to reshape it to be broadcastable with the stacked_expert_outputs.
        # Reshape weights to [batch_size, 1, 1, num_experts] to align with stacked_expert_outputs
        reshaped_weights = expert_weights.unsqueeze(1).unsqueeze(1)

        # The weighted sum of the expert outputs.
        # The sum is over the last dimension (the experts dimension).
        weighted_output = torch.sum(stacked_expert_outputs * reshaped_weights, dim=-1)

        return weighted_output
