import torch
import torch.nn as nn
from typing import Optional, Tuple, Union


class TemporalAttention(nn.Module):
    """Multi-head temporal attention with optional attention weight inspection."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, return_attention: bool = True) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"TemporalAttention requires d_model divisible by n_heads; received d_model={d_model}, n_heads={n_heads}."
            )

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self._return_attention = return_attention
        self._embed_dim = d_model
        self._num_heads = n_heads

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        *,
        return_attention: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Apply temporal attention and optional residual normalization.

        Args:
            query: Tensor shaped `[batch, sequence, d_model]` representing the target trajectory.
            key: Optional tensor for keys; defaults to ``query`` when omitted.
            value: Optional tensor for values; defaults to ``key`` when omitted.
            return_attention: Override flag to emit attention weights alongside the output.

        Returns:
            Either the normalized attention output or a tuple of ``(output, attn_weights)`` when
            attention inspection is requested.
        """
        attn_return_flag = self._return_attention if return_attention is None else return_attention
        key = query if key is None else key
        value = key if value is None else value
        self._validate_inputs(query=query, key=key, value=value)
        attn_output, attn_weights = self.attention(query=query, key=key, value=value)
        output = self.norm(query + attn_output)
        if attn_return_flag:
            return output, attn_weights
        return output

    def _validate_inputs(self, *, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
        """Validate dimensional compatibility of attention inputs.

        Raises:
            ValueError: If tensors do not conform to the expected `[batch, sequence, d_model]` shape
                or incompatible embeddings are supplied.
        """

        if query.ndim != 3:
            raise ValueError(
                "TemporalAttention expects `query` shaped as [batch, sequence, d_model]."
            )
        if key.ndim != 3:
            raise ValueError(
                "TemporalAttention expects `key` shaped as [batch, sequence, d_model]."
            )
        if value.ndim != 3:
            raise ValueError(
                "TemporalAttention expects `value` shaped as [batch, sequence, d_model]."
            )

        batch_size = query.size(0)
        if key.size(0) != batch_size or value.size(0) != batch_size:
            raise ValueError("TemporalAttention requires identical batch dimension for query, key, and value tensors.")

        embed_dim = query.size(-1)
        if embed_dim != self._embed_dim:
            raise ValueError(
                f"TemporalAttention configured with d_model={self._embed_dim} but received query embedding {embed_dim}."
            )

        for tensor_name, tensor in {"key": key, "value": value}.items():
            tensor_embed_dim = tensor.size(-1)
            if tensor_embed_dim != self._embed_dim:
                raise ValueError(
                    f"TemporalAttention expects {tensor_name} embedding dimension {self._embed_dim}; received {tensor_embed_dim}."
                )

        if key.dtype != query.dtype or value.dtype != query.dtype:
            raise ValueError("TemporalAttention requires query, key, and value tensors to share the same dtype.")

        if key.device != query.device or value.device != query.device:
            raise ValueError("TemporalAttention requires all tensors to reside on the same device.")