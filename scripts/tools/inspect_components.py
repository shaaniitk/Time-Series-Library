#!/usr/bin/env python3
"""
Quick component registry inspection to see what's available
"""

import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from layers.modular.core import unified_registry, ComponentFamily
import layers.modular.core.register_components  # noqa: F401
from layers.modular.sampling.registry import SamplingRegistry
from layers.modular.encoder.registry import EncoderRegistry
from layers.modular.decoder.registry import DecoderRegistry
from layers.modular.decomposition.registry import DecompositionRegistry
from layers.modular.output_heads.registry import OutputHeadRegistry
from layers.modular.losses.registry import LossRegistry
from utils.modular_components.registry import create_global_registry
from utils.modular_components.example_components import register_example_components

print("🔍 Available Modular Components")
print("=" * 50)

# Traditional components
print("\n📋 Traditional Components:")
print(f"Attention: {sorted(unified_registry.list(ComponentFamily.ATTENTION)[ComponentFamily.ATTENTION.value])}")
print(f"Sampling: {SamplingRegistry.list_components()}")
print(f"Encoder: {EncoderRegistry.list_components()}")
print(f"Decoder: {DecoderRegistry.list_components()}")
print(f"Decomposition: {DecompositionRegistry.list_components()}")
print(f"Output Head: {OutputHeadRegistry.list_components()}")
print(f"Loss: {LossRegistry.list_components()}")

# Modular components
print("\n🔧 Modular Components (Global Registry):")
registry = create_global_registry()
register_example_components(registry)

for component_type in ['backbone', 'processor', 'attention', 'loss', 'embedding', 'output']:
    components = registry.list_components(component_type)
    if components:
        print(f"{component_type.title()}: {components}")

print("\n✅ Component inspection complete!")
