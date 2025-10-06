
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
from layers.modular.graph.registry import GraphComponentRegistry

@GraphComponentRegistry.register("gated_graph_combiner")
class GatedGraphCombiner(nn.Module):
    """
    Combines multiple graph structures (adjacency matrices and optional edge weights)
    using a learnable, context-aware attention mechanism.
    """
    def __init__(self, num_nodes: int, d_model: int, num_graphs: int = 2, max_graphs: int = 5):
        """
        Args:
            num_nodes (int): The number of nodes in the graph.
            d_model (int): The dimensionality of the model (for context).
            num_graphs (int): The initial number of input graphs to combine.
            max_graphs (int): Maximum number of graphs that can be combined (for dynamic sizing).
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.num_graphs = num_graphs
        self.max_graphs = max_graphs

        # Flexible attention mechanism that can handle variable number of graphs
        self.context_encoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU()
        )
        
        # Dynamic attention head that can output variable number of gates
        self.attention_head = nn.Linear(d_model // 4, max_graphs)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, graph_proposals: List[Tuple[torch.Tensor, Optional[torch.Tensor]]], context: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass to combine the graph structures.

        Args:
            graph_proposals (List[Tuple[torch.Tensor, Optional[torch.Tensor]]]): A list of tuples,
                where each tuple contains an adjacency matrix and an optional edge weight tensor.
            context (torch.Tensor): A context tensor (e.g., mean of node features) to inform the gating.
                                    Shape: [batch_size, d_model]

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: The combined adjacency matrix and edge weights.
        """
        num_proposals = len(graph_proposals)
        if num_proposals > self.max_graphs:
            raise ValueError(f"Too many graph proposals ({num_proposals}), maximum supported is {self.max_graphs}")
        
        if num_proposals == 0:
            raise ValueError("At least one graph proposal is required")

        # Compute attention-based gates from the context
        context_encoded = self.context_encoder(context)  # [batch_size, d_model//4]
        all_gates = self.attention_head(context_encoded)  # [batch_size, max_graphs]
        
        # Only use the gates for the actual number of proposals
        gates = all_gates[:, :num_proposals]  # [batch_size, num_proposals]
        gates = self.softmax(gates)  # Normalize only over actual proposals

        # Separate adjacency matrices and weights
        adj_matrices = [g[0] for g in graph_proposals]
        edge_weights = [g[1] for g in graph_proposals]

        # Combine adjacency matrices using a weighted sum
        # Reshape gates for broadcasting: [B, num_graphs] -> [B, num_graphs, 1, 1]
        reshaped_gates = gates.unsqueeze(-1).unsqueeze(-1)
        stacked_adj = torch.stack(adj_matrices, dim=1) # [B, num_graphs, N, N]
        combined_adj = torch.sum(stacked_adj * reshaped_gates, dim=1) # [B, N, N]

        # Combine edge weights if they exist
        combined_weights = None
        if all(w is not None for w in edge_weights):
            # Assuming all weights are tensors of the same shape
            # This part is tricky if edge weights have different shapes (e.g., for different edge types)
            # For now, we assume they are dense [B, N, N] tensors like the adjacency matrix.
            # A more robust implementation would handle dicts of weights.
            try:
                # Ensure all weight tensors are broadcastable
                max_dims = max(w.ndim for w in edge_weights)
                aligned_weights = []
                for w in edge_weights:
                    while w.ndim < max_dims:
                        w = w.unsqueeze(0)
                    aligned_weights.append(w)
                stacked_weights = torch.stack(aligned_weights, dim=1) # [B, num_graphs, ...]
                combined_weights = torch.sum(stacked_weights * reshaped_gates, dim=1)
            except RuntimeError as e:
                print(f"Warning: Could not combine edge weights due to shape mismatch: {e}. Using the first one.")
                combined_weights = edge_weights[0]
        else:
            # If some weights are None, we can't easily combine. Fallback to the first available.
            for w in edge_weights:
                if w is not None:
                    combined_weights = w
                    break
            
        return combined_adj, combined_weights