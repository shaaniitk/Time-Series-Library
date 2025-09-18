"""
Spatial Relationship Experts

Specialized experts for different spatial patterns in time series:
- Local spatial dependencies (short-range interactions)
- Global spatial relationships (long-range interactions)  
- Hierarchical spatial patterns (multi-scale spatial structures)
"""

from .local_expert import LocalSpatialExpert
from .global_expert import GlobalSpatialExpert
from .hierarchical_expert import HierarchicalSpatialExpert

__all__ = [
    'LocalSpatialExpert',
    'GlobalSpatialExpert', 
    'HierarchicalSpatialExpert'
]