import torch
import torch.nn as nn

class TemporalAttention(nn.Module):
    """A standard Multi-Head Attention block to find relevant past events for the current target state."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, current_target_state, historical_event_messages):
        context, attn_weights = self.attention(query=current_target_state, key=historical_event_messages, value=historical_event_messages)
        return self.norm(current_target_state + context), attn_weights