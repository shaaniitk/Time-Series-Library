
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class StochasticGraphLearner(nn.Module):
    """
    Learns a stochastic graph structure from node features.
    It predicts a probability distribution over edges and samples from it.
    """
    def __init__(self, d_model: int, num_nodes: int):
        """
        Args:
            d_model (int): The dimensionality of the node features.
            num_nodes (int): The total number of nodes in the graph.
        """
        super().__init__()
        self.d_model = d_model
        self.num_nodes = num_nodes

        # A simple feed-forward network to predict edge logits
        # It takes concatenated features of two nodes as input
        self.edge_predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

        # Temperature for Gumbel-Softmax sampling
        self.temperature = 1.0

    def forward(self, node_features: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features (torch.Tensor): A tensor of node features of shape [batch_size, num_nodes, d_model].
            training (bool): Flag for training or inference mode.

        Returns:
            torch.Tensor: The sampled adjacency matrix of shape [batch_size, num_nodes, num_nodes].
            torch.Tensor: The edge probabilities (logits) of shape [batch_size, num_nodes, num_nodes].
        """
        batch_size, _, _ = node_features.shape

        # Expand node features to create all possible pairs for edge prediction
        # [B, N, D] -> [B, N, 1, D] -> [B, N, N, D]
        f1 = node_features.unsqueeze(2).expand(-1, -1, self.num_nodes, -1)
        # [B, N, D] -> [B, 1, N, D] -> [B, N, N, D]
        f2 = node_features.unsqueeze(1).expand(-1, self.num_nodes, -1, -1)

        # Concatenate pairs of node features
        edge_features = torch.cat([f1, f2], dim=-1) # [B, N, N, 2*D]

        # Predict edge logits
        # The predictor expects a 2D tensor, so we reshape
        edge_logits = self.edge_predictor(edge_features.view(-1, self.d_model * 2))
        edge_logits = edge_logits.view(batch_size, self.num_nodes, self.num_nodes) # [B, N, N]

        # Symmetrize the logits
        edge_logits = (edge_logits + edge_logits.transpose(1, 2)) / 2

        # Sample adjacency matrix
        if training:
            # Use Gumbel-Softmax for differentiable sampling of edges
            # We treat the binary decision (edge or no edge) as a 2-class classification
            # We need logits for both classes (edge, no_edge). Let's assume no_edge logit is 0.
            adj_logits_paired = torch.stack([edge_logits, torch.zeros_like(edge_logits)], dim=-1)
            adj_matrix_sampled = F.gumbel_softmax(adj_logits_paired, tau=self.temperature, hard=True)[..., 0]
        else:
            # During inference, use the most probable edge
            adj_matrix_sampled = (torch.sigmoid(edge_logits) > 0.5).float()

        # Zero out diagonal
        adj_matrix_sampled = adj_matrix_sampled * (1 - torch.eye(self.num_nodes, device=node_features.device))

        return adj_matrix_sampled, edge_logits

    def regularization_loss(self, edge_logits: torch.Tensor) -> torch.Tensor:
        """
        Computes a regularization loss to encourage sparsity and connectivity.
        """
        # L1 sparsity on probabilities
        probs = torch.sigmoid(edge_logits)
        sparsity_loss = torch.mean(torch.abs(probs))

        # Connectivity loss (encourage some edges to exist)
        connectivity_loss = torch.mean(torch.relu(0.05 - probs))

        return 0.01 * sparsity_loss + 0.1 * connectivity_loss
