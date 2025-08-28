import torch
import torch.nn as nn
from typing import Optional, Tuple
from abc import ABC, abstractmethod

class BaseAttention(nn.Module, ABC):
    """
    Abstract Base Class for all attention components.

    This class enforces a common interface for all attention mechanisms,
    ensuring they can be used interchangeably in a modular framework.
    """
    def __init__(self, d_model: int, n_heads: int, **kwargs):
        """
        Initializes the base attention component.

        Args:
            d_model (int): The dimensionality of the model.
            n_heads (int): The number of attention heads.
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # This multiplier can be overridden by subclasses if their output
        # dimension differs from d_model.
        self.output_dim_multiplier = 1.0

    @abstractmethod
    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Defines the forward pass of the attention mechanism.

        Args:
            queries (torch.Tensor): Query tensor. Shape: [B, L, D]
            keys (torch.Tensor): Key tensor. Shape: [B, S, D]
            values (torch.Tensor): Value tensor. Shape: [B, S, D]
            attn_mask (Optional[torch.Tensor]): An optional mask to prevent
                                                attention to certain positions.
            **kwargs: For additional, implementation-specific arguments.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - The output tensor of the attention mechanism.
                - An optional tensor containing the attention weights.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def returnOutputDimension(self) -> int:
        """
        Returns the feature dimension of the output tensor.

        This is typically d_model, but can be different for some components
        (e.g., those that concatenate features).
        
        Returns:
            int: The output dimension.
        """
        return int(self.d_model * self.output_dim_multiplier)