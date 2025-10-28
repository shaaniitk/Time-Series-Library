"""
Celestial Enhanced PGAT - Modular Components

This package contains the modular components for the Celestial Enhanced PGAT model,
refactored for improved readability, maintainability, and extensibility.

Enhanced with full feature parity to the original implementation including:
- Comprehensive diagnostics and logging
- Advanced fusion mechanisms
- Stochastic graph learning
- Sequential mixture decoders
- Efficient covariate interaction
- Future celestial processing
"""

from .config import CelestialPGATConfig
from .embedding import EmbeddingModule
from .graph import GraphModule
from .encoder import EncoderModule
from .postprocessing import PostProcessingModule
from .decoder import DecoderModule
from .utils import ModelUtils
from .diagnostics import ModelDiagnostics
from .context_fusion import MultiScaleContextFusion, ContextFusionFactory

__all__ = [
    'CelestialPGATConfig',
    'EmbeddingModule',
    'GraphModule', 
    'EncoderModule',
    'PostProcessingModule',
    'DecoderModule',
    'ModelUtils',
    'ModelDiagnostics',
    'MultiScaleContextFusion',
    'ContextFusionFactory'
]