"""Registry/component construction helpers (placeholder)."""
from __future__ import annotations
from typing import Any, Dict

try:
    from layers.modular.core.registry import unified_registry, ComponentFamily
except Exception:  # pragma: no cover
    unified_registry = None  # type: ignore


def build_backbone(config: Dict[str, Any] | None = None) -> Any:  # type: ignore[override]
    """Return a backbone component instance via registry (placeholder)."""
    if unified_registry is None:  # pragma: no cover
        raise RuntimeError("Registry unavailable")
    cfg = config or {"type": "feedforward", "name": "standard_ffn", "params": {"d_model": 16, "hidden_factor": 2, "dropout": 0.0}}
    fam = ComponentFamily.FEEDFORWARD
    return unified_registry.create(fam, cfg["name"], **cfg.get("params", {}))
