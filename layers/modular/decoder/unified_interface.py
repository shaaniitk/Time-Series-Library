"""
Unified decoder interface providing consistent behavior across all decoder types.
"""
import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
from .base import BaseDecoder
from .decoder_output import DecoderOutput, standardize_decoder_output


class UnifiedDecoderInterface(BaseDecoder):
    """
    Unified interface that wraps any decoder implementation to provide consistent behavior.
    
    This class ensures:
    - Consistent return signatures (always returns DecoderOutput)
    - Input validation
    - Error handling
    - Performance monitoring hooks
    """
    
    def __init__(self, decoder_impl: nn.Module, validate_inputs: bool = True):
        super().__init__()
        self.decoder_impl = decoder_impl
        self.validate_inputs = validate_inputs
        self._forward_count = 0
    
    def _validate_decoder_inputs(self, x: torch.Tensor, cross: torch.Tensor, 
                                trend: Optional[torch.Tensor] = None):
        """Validate input tensors for common issues."""
        if not isinstance(x, torch.Tensor) or not isinstance(cross, torch.Tensor):
            raise TypeError("x and cross must be torch.Tensor")
        
        if x.dim() != 3 or cross.dim() != 3:
            raise ValueError(f"Expected 3D tensors, got x: {x.dim()}D, cross: {cross.dim()}D")
        
        if x.size(-1) != cross.size(-1):
            raise ValueError(f"Feature dimension mismatch: x={x.size(-1)}, cross={cross.size(-1)}")
        
        if trend is not None:
            if not isinstance(trend, torch.Tensor):
                raise TypeError("trend must be torch.Tensor or None")
            if trend.size(0) != x.size(0):
                raise ValueError(f"Batch size mismatch: x={x.size(0)}, trend={trend.size(0)}")
    
    def forward(self, x: torch.Tensor, cross: torch.Tensor, 
                x_mask: Optional[torch.Tensor] = None, 
                cross_mask: Optional[torch.Tensor] = None, 
                trend: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unified forward pass with consistent interface.
        
        Returns:
            Tuple of (seasonal, trend) for backward compatibility
        """
        if self.validate_inputs:
            self._validate_decoder_inputs(x, cross, trend)
        
        self._forward_count += 1
        
        try:
            # Call the underlying decoder implementation
            raw_output = self.decoder_impl(x, cross, x_mask, cross_mask, trend)
            
            # Standardize the output format
            standardized_output = standardize_decoder_output(raw_output)
            
            # Return tuple for backward compatibility
            return standardized_output.seasonal, standardized_output.trend
            
        except Exception as e:
            raise RuntimeError(f"Decoder forward pass failed at step {self._forward_count}: {str(e)}") from e
    
    def get_full_output(self, x: torch.Tensor, cross: torch.Tensor, 
                       x_mask: Optional[torch.Tensor] = None, 
                       cross_mask: Optional[torch.Tensor] = None, 
                       trend: Optional[torch.Tensor] = None) -> DecoderOutput:
        """
        Get full DecoderOutput including auxiliary loss.
        """
        if self.validate_inputs:
            self._validate_decoder_inputs(x, cross, trend)
        
        raw_output = self.decoder_impl(x, cross, x_mask, cross_mask, trend)
        return standardize_decoder_output(raw_output)
    
    def reset_stats(self):
        """Reset internal statistics."""
        self._forward_count = 0
    
    def get_stats(self) -> dict:
        """Get decoder usage statistics."""
        return {
            'forward_count': self._forward_count,
            'decoder_type': type(self.decoder_impl).__name__
        }


class DecoderFactory:
    """
    Factory for creating unified decoder interfaces.
    """
    
    @staticmethod
    def create_unified_decoder(decoder_impl: nn.Module, 
                             validate_inputs: bool = True) -> UnifiedDecoderInterface:
        """
        Create a unified decoder interface wrapping the given implementation.
        
        Args:
            decoder_impl: The underlying decoder implementation
            validate_inputs: Whether to validate inputs on each forward pass
            
        Returns:
            UnifiedDecoderInterface instance
        """
        return UnifiedDecoderInterface(decoder_impl, validate_inputs)
    
    @staticmethod
    def wrap_existing_decoder(decoder: nn.Module) -> UnifiedDecoderInterface:
        """
        Wrap an existing decoder with the unified interface.
        
        This is useful for gradually migrating existing code.
        """
        return UnifiedDecoderInterface(decoder, validate_inputs=True)