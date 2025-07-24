import torch
from ..base_interfaces import BaseComponent
from ..config_schemas import ComponentConfig

class BaseDecomposition(BaseComponent):
    """Base class for all decomposition components"""
    def __init__(self, config: ComponentConfig):
        super().__init__()
        self.config = config
    def forward(self, x: torch.Tensor):
        raise NotImplementedError
