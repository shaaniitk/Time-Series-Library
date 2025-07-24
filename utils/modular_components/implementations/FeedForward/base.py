import torch
import torch.nn as nn
from ..base_interfaces import BaseComponent
from ..config_schemas import ComponentConfig

class BaseFeedForward(BaseComponent):
    """Base class for all feedforward/backbone components"""
    def __init__(self, config: ComponentConfig):
        super().__init__()
        self.config = config
    def forward(self, x: torch.Tensor):
        raise NotImplementedError
