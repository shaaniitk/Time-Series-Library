"""
Registry for encoder components.
"""

from typing import Type, Dict, Any
import torch.nn as nn

class EncoderRegistry:
    """A registry for encoder components."""

    _registry: Dict[str, Type[nn.Module]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a new encoder component.

        Args:
            name (str): The name of the encoder component.
        """
        def wrapper(wrapped_class: Type[nn.Module]):
            if name in cls._registry:
                raise ValueError(f"Encoder component '{name}' is already registered.")
            cls._registry[name] = wrapped_class
            return wrapped_class
        return wrapper

    @classmethod
    def get(cls, name: str) -> Type[nn.Module]:
        """
        Get an encoder component from the registry.

        Args:
            name (str): The name of the encoder component.

        Returns:
            Type[nn.Module]: The encoder component class.
        """
        if name not in cls._registry:
            raise ValueError(f"Encoder component '{name}' not found in registry. Available: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def list_available(cls) -> list[str]:
        """List all available encoder components."""
        return list(cls._registry.keys())

def get_encoder_component(name: str, **kwargs: Any) -> nn.Module:
    """
    Instantiate an encoder component from the registry.

    Args:
        name (str): The name of the encoder component.
        **kwargs: The arguments to pass to the component's constructor.

    Returns:
        nn.Module: An instance of the encoder component.
    """
    component_class = EncoderRegistry.get(name)
    return component_class(**kwargs)