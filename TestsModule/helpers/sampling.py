"""Sampling / parametrization strategies (placeholder)."""
from __future__ import annotations
from typing import List, Dict, Any


def backbone_param_grid() -> List[Dict[str, Any]]:
    return [
        {"d_model": 16, "hidden_factor": 2},
        {"d_model": 32, "hidden_factor": 4},
    ]
