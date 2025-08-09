import torch
import torch.nn as nn
from utils.logger import logger
from typing import Union


class Normalize(nn.Module):
    """
    Base normalization class implementing RevIN (Reversible Instance Normalization).
    
    Features:
    - Reversible normalization/denormalization
    - Optional affine transformation
    - Optional last-value subtraction mode
    - Optional bypass mode
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False, 
                 subtract_last: bool = False, non_norm: bool = False) -> None:
        """
        Initialize normalization layer.
        
        Args:
            num_features: Number of features or channels
            eps: Value added for numerical stability
            affine: If True, has learnable affine parameters
            subtract_last: If True, subtract last timestep instead of mean
            non_norm: If True, bypass normalization
        """
        super(Normalize, self).__init__()
        logger.info(f"Initializing {self.__class__.__name__} with num_features={num_features}, "
                    f"eps={eps}, affine={affine}, subtract_last={subtract_last}, non_norm={non_norm}")
        
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        
        # Statistics storage
        self.mean: Union[torch.Tensor, None] = None
        self.stdev: Union[torch.Tensor, None] = None
        self.last: Union[torch.Tensor, None] = None
        
        if self.affine:
            self._init_params()

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Forward pass with mode selection.
        
        Args:
            x: Input tensor
            mode: 'norm' for normalization, 'denorm' for denormalization
            
        Returns:
            Normalized or denormalized tensor
        """
        logger.debug(f"{self.__class__.__name__} forward, x shape: {x.shape}, mode: {mode}")
        
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError(f"Mode '{mode}' not supported. Use 'norm' or 'denorm'.")
        return x

    def _init_params(self) -> None:
        """Initialize learnable affine parameters."""
        logger.info("Initializing affine parameters")
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x: torch.Tensor) -> None:
        """Compute and store normalization statistics."""
        logger.debug(f"Getting statistics, input shape: {x.shape}")
        dim2reduce = tuple(range(1, x.ndim - 1))
        
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
            logger.debug("Subtracting last timestep for normalization")
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            logger.debug(f"Computed mean: {self.mean.shape if self.mean is not None else None}")
            
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
        logger.debug(f"Computed std deviation: {self.stdev.shape}")

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization transformation."""
        logger.debug("Normalizing the input")
        
        if self.non_norm:
            return x
            
        if self.subtract_last:
            if self.last is None:
                raise RuntimeError("Statistics not computed. Call _get_statistics first.")
            x = x - self.last
        else:
            if self.mean is None:
                raise RuntimeError("Statistics not computed. Call _get_statistics first.")
            x = x - self.mean
            
        if self.stdev is None:
            raise RuntimeError("Statistics not computed. Call _get_statistics first.")
        x = x / self.stdev
        
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
            
        logger.debug("Normalization complete")
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply denormalization transformation."""
        logger.debug("Denormalizing the input")
        
        if self.non_norm:
            return x
            
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
            
        if self.stdev is None:
            raise RuntimeError("Statistics not available. Normalize first before denormalizing.")
        x = x * self.stdev
        
        if self.subtract_last:
            if self.last is None:
                raise RuntimeError("Statistics not available. Normalize first before denormalizing.")
            x = x + self.last
        else:
            if self.mean is None:
                raise RuntimeError("Statistics not available. Normalize first before denormalizing.")
            x = x + self.mean
            
        logger.debug("Denormalization complete")
        return x


class StandardNorm(Normalize):
    """
    Standard normalization implementation - inherits from Normalize.
    
    This class is kept for backward compatibility and specific use cases.
    It inherits all functionality from the base Normalize class.
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False, 
                 subtract_last: bool = False, non_norm: bool = False) -> None:
        """
        Initialize StandardNorm (identical to Normalize).
        
        Args:
            num_features: Number of features or channels
            eps: Value added for numerical stability  
            affine: If True, has learnable affine parameters
            subtract_last: If True, subtract last timestep instead of mean
            non_norm: If True, bypass normalization
        """
        super().__init__(num_features, eps, affine, subtract_last, non_norm)
        logger.info(f"StandardNorm initialized as alias to Normalize")


# Backward compatibility aliases
ReversibleInstanceNorm = Normalize
RevIN = Normalize
