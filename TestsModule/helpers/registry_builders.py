"""Registry/component construction helpers (placeholder)."""
from __future__ import annotations
from typing import Any, Dict

try:
    from utils.modular_components.registry import get_global_registry
except Exception:  # pragma: no cover
    get_global_registry = None  # type: ignore


def build_backbone(config: Dict[str, Any] | None = None) -> Any:  # type: ignore[override]
    """Return a backbone component instance via registry (placeholder)."""
    if get_global_registry is None:  # pragma: no cover
        raise RuntimeError("Registry unavailable")
    reg = get_global_registry()
    cfg = config or {"type": "feedforward", "name": "standard_ffn", "params": {"d_model": 16, "hidden_factor": 2, "dropout": 0.0}}
    return reg.create_component(cfg["type"], cfg["name"], cfg.get("params", {}))
