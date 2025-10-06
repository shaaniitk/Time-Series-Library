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
            base_adj (torch.Tensor or dict): The base adjacency matrix.
            adaptive_adj (torch.Tensor or dict): The adaptively learned adjacency matrix.
            base_weights (torch.Tensor, dict, or None): Edge weights for the base graph.
            adaptive_weights (torch.Tensor, dict, or None): Edge weights for the adaptive graph.

        Returns:
            Tuple[torch.Tensor or dict, torch.Tensor or dict or None]: The combined adjacency matrix and edge weights.
        """
        gate = torch.sigmoid(self.gate_parameter)

        # Combine adjacency matrices
        if base_adj is not None and adaptive_adj is not None:
            if isinstance(base_adj, torch.Tensor) and isinstance(adaptive_adj, torch.Tensor):
                combined_adj = gate * base_adj + (1.0 - gate) * adaptive_adj
            else:
                # If not tensors (e.g. HeteroData dict), prefer the adaptive one
                combined_adj = adaptive_adj
        else:
            combined_adj = adaptive_adj if adaptive_adj is not None else base_adj

        # Combine edge weights if they exist - handle both tensor and dict types
        combined_weights = None
        if base_weights is not None and adaptive_weights is not None:
            if isinstance(base_weights, torch.Tensor) and isinstance(adaptive_weights, torch.Tensor):
                # Both are tensors - can combine directly
                combined_weights = gate * base_weights + (1.0 - gate) * adaptive_weights
            elif isinstance(base_weights, dict) and isinstance(adaptive_weights, dict):
                # Both are dicts - combine matching keys
                combined_weights = {}
                all_keys = set(base_weights.keys()) | set(adaptive_weights.keys())
                for key in all_keys:
                    base_val = base_weights.get(key)
                    adaptive_val = adaptive_weights.get(key)
                    if base_val is not None and adaptive_val is not None:
                        if isinstance(base_val, torch.Tensor) and isinstance(adaptive_val, torch.Tensor):
                            combined_weights[key] = gate * base_val + (1.0 - gate) * adaptive_val
                        else:
                            # Prefer adaptive if types don't match
                            combined_weights[key] = adaptive_val
                    else:
                        # Use whichever is available
                        combined_weights[key] = adaptive_val if adaptive_val is not None else base_val
            else:
                # Mixed types - prefer adaptive weights
                combined_weights = adaptive_weights
        else:
            # Use whichever is available
            combined_weights = adaptive_weights if adaptive_weights is not None else base_weights
            
        return combined_adj, combined_weights