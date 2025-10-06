import torch
import torch.nn as nn
from layers.modular.graph.registry import GraphComponentRegistry

@GraphComponentRegistry.register("gated_graph_combiner")
class GatedGraphCombiner(nn.Module):
    """
    Combines two graph structures (adjacency matrices and optional edge weights)
    using a learnable, dynamic gating mechanism.
    """
    def __init__(self, num_nodes, d_model):
        super().__init__()
        # A simple approach is a single parameter to balance the two graphs globally.
        self.gate_parameter = nn.Parameter(torch.zeros(1))

    def forward(self, base_adj, adaptive_adj, base_weights=None, adaptive_weights=None):
        """
        Forward pass to combine the graph structures.

        Args:
            base_adj (torch.Tensor): The base adjacency matrix.
            adaptive_adj (torch.Tensor): The adaptively learned adjacency matrix.
            base_weights (torch.Tensor, optional): Edge weights for the base graph.
            adaptive_weights (torch.Tensor, optional): Edge weights for the adaptive graph.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The combined adjacency matrix and edge weights.
        """
        gate = torch.sigmoid(self.gate_parameter)

        # Combine adjacency matrices
        if base_adj is not None and adaptive_adj is not None:
            if isinstance(base_adj, torch.Tensor) and isinstance(adaptive_adj, torch.Tensor):
                combined_adj = gate * base_adj + (1.0 - gate) * adaptive_adj
            else:
                # If not a tensor (e.g. HeteroData), prefer the adaptive one
                combined_adj = adaptive_adj
        else:
            combined_adj = adaptive_adj if adaptive_adj is not None else base_adj

        # Combine edge weights if they exist
        if base_weights is not None and adaptive_weights is not None:
            combined_weights = gate * base_weights + (1.0 - gate) * adaptive_weights
        else:
            combined_weights = adaptive_weights if adaptive_weights is not None else base_weights
            
        return combined_adj, combined_weights