"""
Core decoder implementations moved from models/ to avoid circular imports.
"""
import torch
import torch.nn as nn
from typing import List, Optional
from .decoder_output import DecoderOutput


class CoreAutoformerDecoder(nn.Module):
    """
    Core Autoformer decoder implementation without model-specific dependencies.
    """
    def __init__(self, layers: List[nn.Module], norm_layer: Optional[nn.Module] = None, 
                 projection: Optional[nn.Module] = None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x: torch.Tensor, cross: torch.Tensor, 
                x_mask: Optional[torch.Tensor] = None, 
                cross_mask: Optional[torch.Tensor] = None, 
                trend: Optional[torch.Tensor] = None) -> DecoderOutput:
        
        if trend is None:
            trend = torch.zeros_like(x)
        
        total_aux_loss = 0.0
        
        for layer in self.layers:
            layer_output = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            
            if len(layer_output) == 3:
                x, residual_trend, aux_loss = layer_output
                if isinstance(aux_loss, torch.Tensor):
                    total_aux_loss += aux_loss.item()
            else:
                x, residual_trend = layer_output
            
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            if isinstance(self.projection, nn.Conv1d) or (
                isinstance(self.projection, nn.Sequential) and 
                any(isinstance(m, nn.Conv1d) for m in self.projection.modules())
            ):
                x = self.projection(x.permute(0, 2, 1)).transpose(1, 2)
            else:
                x = self.projection(x)
        
        return DecoderOutput(seasonal=x, trend=trend, aux_loss=total_aux_loss)


class CoreEnhancedDecoder(nn.Module):
    """
    Core enhanced decoder with advanced features.
    """
    def __init__(self, layers: List[nn.Module], d_model: int, 
                 norm_layer: Optional[nn.Module] = None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.d_model = d_model

    def forward(self, x: torch.Tensor, cross: torch.Tensor, 
                x_mask: Optional[torch.Tensor] = None, 
                cross_mask: Optional[torch.Tensor] = None, 
                trend: Optional[torch.Tensor] = None) -> DecoderOutput:
        
        if trend is None:
            trend = torch.zeros(x.size(0), x.size(1), self.d_model, 
                              device=x.device, dtype=x.dtype)
        
        total_aux_loss = 0.0
        
        for layer in self.layers:
            layer_output = layer(x, cross, x_mask, cross_mask)
            
            if len(layer_output) == 3:
                x, residual_trend, aux_loss = layer_output
                if isinstance(aux_loss, torch.Tensor):
                    total_aux_loss += aux_loss.item()
            else:
                x, residual_trend = layer_output
            
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)
        
        return DecoderOutput(seasonal=x, trend=trend, aux_loss=total_aux_loss)