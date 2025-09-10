#!/usr/bin/env python3
"""
Fusion components for the modular framework.

This module provides fusion components that can be used to combine
multiple input streams or features in time series models.
"""

from .hierarchical_fusion import HierarchicalFusion

__all__ = [
    'HierarchicalFusion'
]