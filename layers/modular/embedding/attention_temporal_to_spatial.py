
import torch
import torch.nn as nn

class AttentionTemporalToSpatial(nn.Module):
    """
    Converts a temporal sequence representation [Batch, SeqLen, Dim] into a 
    spatial (node-based) representation [Batch, NumNodes, Dim] using an 
    attention mechanism.

    Each target node representation is formed by attending to the entire input time series.
    """
    def __init__(self, d_model, num_nodes, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.n_heads = n_heads

        # Learnable queries, one for each target node
        self.queries = nn.Parameter(torch.randn(1, self.num_nodes, self.d_model))

        # Standard multi-head attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=self.d_model, 
            num_heads=self.n_heads, 
            batch_first=True
        )
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input temporal sequence of shape [Batch, SeqLen, Dim].

        Returns:
            torch.Tensor: The output spatial representation of shape [Batch, NumNodes, Dim].
        """
        batch_size = x.shape[0]
        
        # Expand queries to match the batch size
        queries = self.queries.expand(batch_size, -1, -1)

        # The input sequence `x` serves as both keys and values
        # `queries` ask for the summary of the sequence to form the node representation
        attended_output, _ = self.attention(query=queries, key=x, value=x)
        
        # Add & Norm
        return self.norm(attended_output)
