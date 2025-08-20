"""
Standardized output format for all decoder implementations.
"""
import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class DecoderOutput:
    """
    Standardized output format for decoder forward passes.
    
    Attributes:
        seasonal: The seasonal component output tensor
        trend: The trend component output tensor  
        aux_loss: Auxiliary loss (e.g., from MoE routing), defaults to 0.0
    """
    seasonal: torch.Tensor
    trend: torch.Tensor
    aux_loss: float = 0.0
    
    def __iter__(self):
        """Allow tuple unpacking for backward compatibility."""
        yield self.seasonal
        yield self.trend
        if self.aux_loss != 0.0:
            yield self.aux_loss
    
    def __len__(self):
        """Return length for backward compatibility checks."""
        return 3 if self.aux_loss != 0.0 else 2
    
    def to_tuple(self, include_aux_loss: bool = True):
        """Convert to tuple format for backward compatibility."""
        if include_aux_loss:
            return (self.seasonal, self.trend, self.aux_loss)
        return (self.seasonal, self.trend)


def standardize_decoder_output(output) -> DecoderOutput:
    """
    Convert various decoder output formats to standardized DecoderOutput.
    
    Args:
        output: Can be tuple of (seasonal, trend) or (seasonal, trend, aux_loss)
                or already a DecoderOutput instance
    
    Returns:
        DecoderOutput instance
    """
    if isinstance(output, DecoderOutput):
        return output
    
    if isinstance(output, (tuple, list)):
        if len(output) == 2:
            return DecoderOutput(seasonal=output[0], trend=output[1])
        elif len(output) == 3:
            return DecoderOutput(seasonal=output[0], trend=output[1], aux_loss=output[2])
        else:
            raise ValueError(f"Invalid decoder output format: expected 2 or 3 elements, got {len(output)}")
    
    raise TypeError(f"Invalid decoder output type: {type(output)}")