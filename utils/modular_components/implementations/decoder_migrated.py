"""
Migrated Decoder Components
Auto-migrated from layers/modular/decoder to utils.modular_components.implementations
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
import logging

# Import base interfaces from new framework
from ..base_interfaces import BaseComponent, BaseLoss, BaseAttention
from ..config_schemas import ComponentConfig

logger = logging.getLogger(__name__)

# Migrated imports
from utils.logger import logger
from models.EnhancedAutoformer import EnhancedDecoder as EnhancedAutoformerDecoder
from layers.Autoformer_EncDec import Decoder as AutoformerDecoder
import torch.nn as nn
from typing import Tuple, Optional
from models.EnhancedAutoformer_Fixed import EnhancedDecoder as StableAutoformerDecoder
from abc import ABC, abstractmethod

# Migrated Classes
class BaseDecoder(nn.Module, ABC):
    """
    Abstract base class for all decoder components.
    """
    def __init__(self):
        super(BaseDecoder, self).__init__()

    @abstractmethod
    def forward(self, 
                x: nn.Module, 
                cross: nn.Module, 
                x_mask: Optional[nn.Module] = None, 
                cross_mask: Optional[nn.Module] = None, 
                trend: Optional[nn.Module] = None) -> Tuple[nn.Module, nn.Module]:
        """
        The forward pass for the decoder.

        Args:
            x (torch.Tensor): The input tensor for the decoder.
            cross (torch.Tensor): The output from the encoder.
            x_mask (torch.Tensor, optional): The self-attention mask for the decoder. Defaults to None.
            cross_mask (torch.Tensor, optional): The cross-attention mask. Defaults to None.
            trend (torch.Tensor, optional): The trend component. Defaults to None.

        Returns:
            tuple: A tuple containing:
                   - torch.Tensor: The seasonal component output.
                   - torch.Tensor: The trend component output.
        """
        pass

class EnhancedDecoder(BaseDecoder):
    """
    The enhanced Autoformer decoder, now built with modular layers.
    """
    def __init__(self, d_layers, d_model, c_out, n_heads, d_ff, dropout, activation,
                 self_attention_comp, cross_attention_comp, decomp_comp, 
                 norm_layer=None, projection=None):
        super(EnhancedDecoder, self).__init__()
        
        self.decoder = EnhancedAutoformerDecoder(
            [
                EnhancedDecoderLayer(
                    self_attention_comp,
                    cross_attention_comp,
                    decomp_comp,
                    d_model,
                    c_out,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            c_out=c_out,
            norm_layer=norm_layer,
            projection=projection
        )

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        return self.decoder(x, cross, x_mask, cross_mask, trend)

class DecoderRegistry:
    """
    A registry for all available decoder components.
    """
    _registry = {
        "standard": StandardDecoder,
        "enhanced": EnhancedDecoder,
        "stable": StableDecoder,
    }

    @classmethod
    def register(cls, name, component_class):
        if name in cls._registry:
            logger.warning(f"Component '{name}' is already registered and will be overwritten.")
        cls._registry[name] = component_class
        logger.info(f"Registered decoder component: {name}")

    @classmethod
    def get(cls, name):
        component = cls._registry.get(name)
        if component is None:
            logger.error(f"Decoder component '{name}' not found.")
            raise ValueError(f"Decoder component '{name}' not found.")
        return component

    @classmethod
    def list_components(cls):
        return list(cls._registry.keys())

def get_decoder_component(name, **kwargs):
    component_class = DecoderRegistry.get(name)
    return component_class(**kwargs)

class StableDecoder(BaseDecoder):
    """
    The stable Autoformer decoder, built with modular enhanced layers.
    """
    def __init__(self, d_layers, d_model, c_out, n_heads, d_ff, dropout, activation,
                 self_attention_comp, cross_attention_comp, decomp_comp, 
                 norm_layer=None, projection=None):
        super(StableDecoder, self).__init__()
        
        self.decoder = StableAutoformerDecoder(
            [
                EnhancedDecoderLayer(
                    self_attention_comp,
                    cross_attention_comp,
                    decomp_comp,
                    d_model,
                    c_out,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=norm_layer,
            projection=projection
        )

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        return self.decoder(x, cross, x_mask, cross_mask, trend)

class StandardDecoder(BaseDecoder):
    """
    The standard Autoformer decoder, now built with modular layers.
    """
    def __init__(self, d_layers, d_model, c_out, n_heads, d_ff, dropout, activation,
                 self_attention_comp, cross_attention_comp, decomp_comp, 
                 norm_layer=None, projection=None):
        super(StandardDecoder, self).__init__()
        
        self.decoder = AutoformerDecoder(
            [
                StandardDecoderLayer(
                    self_attention_comp,
                    cross_attention_comp,
                    decomp_comp,
                    d_model,
                    c_out,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=norm_layer,
            projection=projection
        )

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        return self.decoder(x, cross, x_mask, cross_mask, trend)


# Migrated Functions  


# Registry function for decoder components
def get_decoder_component(name: str, config: ComponentConfig = None, **kwargs):
    """Get decoder component by name"""
    # This will be implemented based on the migrated components
    pass

def register_decoder_components(registry):
    """Register all decoder components with the registry"""
    # This will be implemented to register all migrated components
    pass
