"""
Dynamic registry for all modular loss functions.
"""
import importlib
import os
from typing import Dict, Any, Type
from ..base_interfaces import BaseLoss

LOSS_REGISTRY: Dict[str, Type[BaseLoss]] = {}

# Dynamically import all loss classes in this directory
loss_dir = os.path.dirname(__file__)
for fname in os.listdir(loss_dir):
    if fname.endswith('_loss.py'):
        module_name = fname[:-3]
        module = importlib.import_module(f".{{module_name}}", package=__package__)
        for attr in dir(module):
            obj = getattr(module, attr)
            if isinstance(obj, type) and issubclass(obj, BaseLoss) and obj is not BaseLoss:
                LOSS_REGISTRY[obj.__name__.lower().replace('loss', '')] = obj


def get_loss_function(loss_type: str, config: Any) -> BaseLoss:
    cls = LOSS_REGISTRY.get(loss_type)
    if cls is None:
        raise ValueError(f"Loss type '{loss_type}' not found in registry.")
    return cls(config)


def list_available_losses() -> Dict[str, str]:
    return {k: v.__name__ for k, v in LOSS_REGISTRY.items()}


def get_loss_capabilities(loss_type: str) -> Dict[str, Any]:
    cls = LOSS_REGISTRY.get(loss_type)
    if cls is None:
        return {}
    return getattr(cls, 'get_capabilities', lambda self: {})(cls)
