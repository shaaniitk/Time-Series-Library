"""Configuration schemas for modular components."""
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ComponentConfig:
    """Base configuration class for modular components."""
    name: str
    type: str
    params: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class BackboneConfig(ComponentConfig):
    """Configuration for backbone components."""
    d_model: int = 512
    dropout: float = 0.1
    pretrained: bool = True
    model_name: str = "default"


@dataclass
class LossConfig(ComponentConfig):
    """Base configuration for loss components."""
    reduction: str = "mean"
    weight: float = 1.0


__all__ = [
    "ComponentConfig",
    "BackboneConfig", 
    "LossConfig"
]