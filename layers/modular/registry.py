"""Thin facade exposing registry helpers for tests.

Provides create_component and list_components wrappers over the unified registry.
"""
from __future__ import annotations
from typing import Any, Dict, List

from .core.registry import unified_registry, ComponentFamily


_STR_TO_FAMILY = {f.value: f for f in ComponentFamily}


def create_component(kind: str, name: str, params: Dict[str, Any] | None = None):
    fam = _STR_TO_FAMILY.get(kind)
    if fam is None:
        raise ValueError(f"Unknown component family: {kind}")
    return unified_registry.create(fam, name, **(params or {}))


def list_components(kind: str | None = None) -> Dict[str, List[str]]:
    if kind is None:
        return {k: v for k, v in unified_registry.list().items()}
    fam = _STR_TO_FAMILY.get(kind)
    if fam is None:
        raise ValueError(f"Unknown component family: {kind}")
    return unified_registry.list(fam)
