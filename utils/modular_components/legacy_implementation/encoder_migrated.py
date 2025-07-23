"""
Migrated Encoder Components
Auto-migrated from layers/modular/encoder to utils.modular_components.implementations
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
from layers.EnhancedAutoCorrelation import AdaptiveAutoCorrelationLayer, AdaptiveAutoCorrelation
from typing import Tuple, Optional
from layers.MultiWaveletCorrelation import MultiWaveletTransform
        from argparse import Namespace
from abc import ABC, abstractmethod
import torch.nn as nn
from utils.logger import logger
import torch
from models.EnhancedAutoformer import EnhancedEncoder as EnhancedAutoformerEncoder
from layers.Autoformer_EncDec import Encoder as AutoformerEncoder
from models.HierarchicalEnhancedAutoformer import HierarchicalEncoder as HierarchicalAutoformerEncoder
from models.EnhancedAutoformer import EnhancedEncoder, EnhancedEncoderLayer
from models.EnhancedAutoformer_Fixed import EnhancedEncoder as StableAutoformerEncoder

# Migrated Classes
class BaseEncoder(nn.Module, ABC):
    """
    Abstract base class for all encoder components.
    """
    def __init__(self):
        super(BaseEncoder, self).__init__()

    @abstractmethod
    def forward(self, x: nn.Module, attn_mask: Optional[nn.Module] = None) -> Tuple[nn.Module, Optional[nn.Module]]:
        """
        The forward pass for the encoder.

        Args:
            x (torch.Tensor): The input tensor.
            attn_mask (torch.Tensor, optional): The attention mask. Defaults to None.

        Returns:
            tuple: A tuple containing:
                   - torch.Tensor: The encoder output.
                   - torch.Tensor or None: The attention weights, if available.
        """
        pass

class EnhancedEncoder(BaseEncoder):
    """
    The enhanced Autoformer encoder, now built with modular layers.
    """
    def __init__(self, e_layers, d_model, n_heads, d_ff, dropout, activation, 
                 attention_comp, decomp_comp, conv_layers=None, norm_layer=None):
        super(EnhancedEncoder, self).__init__()
        
        self.encoder = EnhancedAutoformerEncoder(
            [
                EnhancedEncoderLayer(
                    attention_comp,
                    decomp_comp,
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            conv_layers=conv_layers,
            norm_layer=norm_layer
        )

    def forward(self, x, attn_mask=None):
        return self.encoder(x, attn_mask)

class HierarchicalEncoder(BaseEncoder):
    """
    The hierarchical Autoformer encoder, now properly modularized.
    """
    def __init__(self, e_layers, d_model, n_heads, d_ff, dropout, activation, 
                 attention_type, decomp_type, decomp_params, n_levels=3, share_weights=False, **kwargs):
        super(HierarchicalEncoder, self).__init__()
        
        from ..attention import get_attention_component

        mock_configs = Namespace(
            e_layers=e_layers, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            dropout=dropout, activation=activation, factor=1
        )

        def attention_factory():
            return get_attention_component(attention_type, d_model=d_model, n_heads=n_heads, dropout=dropout)

        self.encoder = HierarchicalAutoformerEncoder(
            configs=mock_configs, 
            n_levels=n_levels, 
            share_weights=share_weights,
            _attention_factory=attention_factory,
            decomp_params=decomp_params
        )

    def forward(self, x, attn_mask=None):
        # The original hierarchical encoder expects a list of tensors
        if not isinstance(x, list):
            x = [x]
        multi_res_output = self.encoder(x, attn_mask)
        
        # For compatibility with standard decoder interface, 
        # we need to return a single tensor for cross-attention
        # We'll concatenate or take the finest resolution
        if isinstance(multi_res_output, list) and len(multi_res_output) > 0:
            # Use the finest resolution (last one) as the main output
            output = multi_res_output[-1]
        else:
            output = multi_res_output
            
        # Return output and None for attention weights to match standard encoder interface
        return output, None
class EncoderRegistry:
    """
    A registry for all available encoder components.
    """
    _registry = {
        "standard": StandardEncoder,
        "enhanced": EnhancedEncoder,
        "stable": StableEncoder,
        "hierarchical": HierarchicalEncoder,
    }

    @classmethod
    def register(cls, name, component_class):
        if name in cls._registry:
            logger.warning(f"Component '{name}' is already registered and will be overwritten.")
        cls._registry[name] = component_class
        logger.info(f"Registered encoder component: {name}")

    @classmethod
    def get(cls, name):
        component = cls._registry.get(name)
        if component is None:
            logger.error(f"Encoder component '{name}' not found.")
            raise ValueError(f"Encoder component '{name}' not found.")
        return component

    @classmethod
    def list_components(cls):
        return list(cls._registry.keys())

def get_encoder_component(name, **kwargs):
    component_class = EncoderRegistry.get(name)
    if name == 'hierarchical':
        return component_class(**kwargs)
    return component_class(**kwargs)

class StableEncoder(BaseEncoder):
    """
    The stable Autoformer encoder, built with modular enhanced layers.
    """
    def __init__(self, e_layers, d_model, n_heads, d_ff, dropout, activation, 
                 attention_comp, decomp_comp, conv_layers=None, norm_layer=None):
        super(StableEncoder, self).__init__()
        
        self.encoder = StableAutoformerEncoder(
            [
                EnhancedEncoderLayer(
                    attention_comp,
                    decomp_comp,
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=norm_layer
        )

    def forward(self, x, attn_mask=None):
        return self.encoder(x, attn_mask)

class StandardEncoder(BaseEncoder):
    """
    The standard Autoformer encoder, now built with modular layers.
    """
    def __init__(self, e_layers, d_model, n_heads, d_ff, dropout, activation, 
                 attention_comp, decomp_comp, conv_layers=None, norm_layer=None):
        super(StandardEncoder, self).__init__()
        
        self.encoder = AutoformerEncoder(
            [
                StandardEncoderLayer(
                    attention_comp,
                    decomp_comp,
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            conv_layers=conv_layers,
            norm_layer=norm_layer
        )

    def forward(self, x, attn_mask=None):
        return self.encoder(x, attn_mask)


# Migrated Functions  


# Registry function for encoder components
def get_encoder_component(name: str, config: ComponentConfig = None, **kwargs):
    """Get encoder component by name"""
    # This will be implemented based on the migrated components
    pass

def register_encoder_components(registry):
    """Register all encoder components with the registry"""
    # This will be implemented to register all migrated components
    pass
