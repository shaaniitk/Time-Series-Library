"""
Celestial Enhanced PGAT - Modular Components

This package contains the modular components for the Celestial Enhanced PGAT model,
refactored for improved readability, maintainability, and extensibility.
"""

from .config import CelestialPGATConfig
from .embedding import EmbeddingModule
from .graph import GraphModule
from .encoder import EncoderModule
from .postprocessing import PostProcessingModule
from .decoder import DecoderModule

__all__ = [
    'CelestialPGATConfig',
    'EmbeddingModule',
    'GraphModule', 
    'EncoderModule',
    'PostProcessingModule',
    'DecoderModule'
]