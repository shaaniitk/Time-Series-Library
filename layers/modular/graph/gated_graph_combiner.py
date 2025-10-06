import math

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Union
from layers.modular.graph.registry import GraphComponentRegistry

@GraphComponentRegistry.register("gated_graph_combiner")
class GatedGraphCombiner(nn.Module):
    """Combine multiple graph proposals with attention-style gating."""

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
        self.proposal_summary_dim = 8
        self.attention_hidden_dim = max(32, d_model // 2)
        self.edge_activation_threshold = 1e-3

        context_input_dim = d_model + self.proposal_summary_dim
        self.context_encoder = nn.Sequential(
            nn.Linear(context_input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.proposal_encoder = nn.Sequential(
            nn.Linear(self.proposal_summary_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )

        self.query_projection = nn.Linear(d_model, self.attention_hidden_dim)
        self.key_projection = nn.Linear(d_model, self.attention_hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.attention_scale = 1.0 / math.sqrt(float(self.attention_hidden_dim))

    def forward(
        self,
        graph_proposals: List[Tuple[torch.Tensor, Optional[Union[torch.Tensor, Dict]]]],
        context: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[Union[torch.Tensor, Dict[Tuple[str, str, str], torch.Tensor]]]]:
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

        summaries: List[torch.Tensor] = []
        encoded_proposals: List[torch.Tensor] = []
        adj_matrices: List[torch.Tensor] = []
        edge_weight_collection: List[Optional[Union[torch.Tensor, Dict]]] = []

        for adjacency, weights in graph_proposals:
            if adjacency.dim() == 2:
                adjacency = adjacency.unsqueeze(0)
            adjacency = adjacency.float()
            summary = self._summarize_proposal(adjacency, weights)
            summaries.append(summary)
            encoded_proposals.append(self.proposal_encoder(summary))
            adj_matrices.append(adjacency)
            edge_weight_collection.append(weights)

        proposal_summary_stack = torch.stack(summaries, dim=1)
        global_stats = proposal_summary_stack.mean(dim=1)
        context_features = torch.cat([context, global_stats], dim=-1)
        context_encoded = self.context_encoder(context_features)
        query = self.query_projection(context_encoded)  # [batch, attn_dim]
        keys = torch.stack([self.key_projection(encoded) for encoded in encoded_proposals], dim=1)
        attention_logits = (keys * query.unsqueeze(1)).sum(dim=-1) * self.attention_scale
        gates = self.softmax(attention_logits)

        reshaped_gates = gates.unsqueeze(-1).unsqueeze(-1)
        stacked_adj = torch.stack(adj_matrices, dim=1)
        combined_adj = torch.sum(stacked_adj * reshaped_gates, dim=1)

        combined_weights = self._combine_edge_weights(edge_weight_collection, gates)

        return combined_adj, combined_weights

    def _summarize_proposal(
        self,
        adjacency: torch.Tensor,
        weights: Optional[Union[torch.Tensor, Dict]]
    ) -> torch.Tensor:
        """Create a compact summary vector for a graph proposal."""

        if adjacency.dim() != 3:
            raise ValueError("adjacency must have shape [batch, N, N]")

        adj_mean = adjacency.mean(dim=(1, 2))
        adj_std = adjacency.std(dim=(1, 2), unbiased=False)
        adj_max = adjacency.abs().amax(dim=(1, 2))
        adj_min = adjacency.amin(dim=(1, 2))
        adj_density = (adjacency.abs() > self.edge_activation_threshold).float().mean(dim=(1, 2))
        adj_energy = adjacency.pow(2).mean(dim=(1, 2))

        if isinstance(weights, torch.Tensor):
            if weights.dim() == 2:
                weights = weights.unsqueeze(0)
            weight_mean = weights.mean(dim=(1, 2))
            weight_std = weights.std(dim=(1, 2), unbiased=False)
        else:
            device = adjacency.device
            weight_mean = torch.zeros(adjacency.size(0), device=device, dtype=adjacency.dtype)
            weight_std = torch.zeros_like(weight_mean)

        stats = torch.stack(
            [adj_mean, adj_std, adj_max, adj_min, adj_density, adj_energy, weight_mean, weight_std],
            dim=-1
        )
        return stats

    def _combine_edge_weights(
        self,
        edge_weights: List[Optional[Union[torch.Tensor, Dict]]],
        gates: torch.Tensor
    ) -> Optional[Union[torch.Tensor, Dict[Tuple[str, str, str], torch.Tensor]]]:
        """Blend edge weights using the learned attention gates."""

        if all(weight is None for weight in edge_weights):
            return None

        if all(isinstance(weight, torch.Tensor) or weight is None for weight in edge_weights):
            template = next((weight for weight in edge_weights if isinstance(weight, torch.Tensor)), None)
            if template is None:
                return None
            template = template if template.dim() == 3 else template.unsqueeze(0)

            tensors: List[torch.Tensor] = []
            for weight in edge_weights:
                if isinstance(weight, torch.Tensor):
                    tensor = weight if weight.dim() == 3 else weight.unsqueeze(0)
                else:
                    tensor = torch.zeros_like(template)
                tensors.append(tensor)

            stacked_weights = torch.stack(tensors, dim=1)
            reshaped_gates = gates.unsqueeze(-1).unsqueeze(-1)
            return torch.sum(stacked_weights * reshaped_gates, dim=1)

        # Mixed formats are not supported; fallback to first available representation.
        for weight in edge_weights:
            if weight is not None:
                return weight
        return None