"""Normalization components module.

This module provides normalization components for the modular architecture,
including layer normalization, data normalization, and tensor processing.
"""

from .registry import (
    normalization_registry,
    LayerNormWrapper,
    RMSNormWrapper,
    TSNormalizerWrapper,
    NormalizationProcessorWrapper,
    create_layer_norm,
    create_data_normalizer,
    create_tensor_normalizer,
    get_normalization_component,
    list_normalization_components,
    register_custom_normalization
)

__all__ = [
    'normalization_registry',
    'LayerNormWrapper',
    'RMSNormWrapper',
    'TSNormalizerWrapper', 
    'NormalizationProcessorWrapper',
    'create_layer_norm',
    'create_data_normalizer',
    'create_tensor_normalizer',
    'get_normalization_component',
    'list_normalization_components',
    'register_custom_normalization'
]