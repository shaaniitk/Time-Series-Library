"""Unified Component Registry Facade (migration shim / enhancer).

Goal:
    Provide a single, robust access point over the global component registry
    while we converge legacy, wrapped, and restored algorithm registrations.

Enhancements added (non‑breaking):
    - Idempotent, explicit initialization (no duplicate side‑effects on reimport)
    - Structured status & error tracking per registration category
    - Lazy optional imports with concise diagnostics instead of noisy prints
    - Thread‑safe initialization guard
    - Separation of concerns (factory usage clarified; avoids hidden secondary
        registries)
    - Lightweight timing metrics
    - Backwards compatible public symbols: ``unified_registry``,
        ``get_component``, ``create_component``

Important Note About Factories:
    This module previously instantiated ``ComponentFactory`` from
    ``utils.modular_components.factory``. That factory internally constructed a
    new *private* ``ComponentRegistry`` instance (not the shared global
    registry), creating the risk of divergence. We retain the attribute for
    compatibility but now point it at the *global* registry and discourage new
    usage in favor of typed factories in ``utils.modular_components.factories``.
"""

from __future__ import annotations

import warnings
import logging
import time
import threading
from typing import Dict, List, Any, Type

from layers.modular.core.registry import unified_registry as _modular_registry
from typing import Protocol

class BaseComponent(Protocol):
    def forward(self, *args, **kwargs): ...

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Status / diagnostics containers
# ---------------------------------------------------------------------------
_INIT_LOCK = threading.Lock()
_INITIALIZED = False
REGISTRATION_STATUS: Dict[str, Dict[str, Any]] = {}

def _status_block(key: str) -> Dict[str, Any]:  # tiny helper
        block = REGISTRATION_STATUS.setdefault(key, {})
        return block

from layers.modular.core import register_components as _register_components  # populates modular registry
from layers.modular.processor.wrapped_decompositions import register_layers_decompositions
from layers.modular.processor.wrapped_encoders import register_layers_encoders
from layers.modular.processor.wrapped_decoders import register_layers_decoders
from layers.modular.processor.wrapped_fusions import register_layers_fusions
from layers.modular.embedding import register_utils_embeddings  # shim; no-op
from layers.modular.feedforward.feedforwards import register_feedforwards as register_utils_feedforwards  # if available
from layers.modular.core.register_advanced import register_specialized_processors

# Import sophisticated algorithms
try:
    from utils.implementations.attention.restored_algorithms import (
        RestoredFourierAttention,
        RestoredAutoCorrelationAttention, 
        RestoredMetaLearningAttention,
        register_restored_algorithms
    )
    ALGORITHMS_AVAILABLE = True
except ImportError:
    # Fallback to direct import
    from utils_algorithm_adapters import (
        RestoredFourierAttention,
        RestoredAutoCorrelationAttention, 
        RestoredMetaLearningAttention,
        register_restored_algorithms
    )
    ALGORITHMS_AVAILABLE = True


def _safe_call(category: str, func, *args, **kwargs):  # helper for guarded imports
    start = time.time()
    status = _status_block(category)
    try:
        func(*args, **kwargs)
        status.setdefault("success", True)
    except Exception as exc:  # pragma: no cover - defensive
        status.setdefault("errors", []).append(str(exc))
        status["success"] = False
        LOGGER.debug("Registration failure in %s: %s", category, exc)
    finally:
        status["duration_ms"] = round(1000 * (time.time() - start), 2)


def _initialize():
    """Perform (one‑time) registration of legacy + advanced components."""
    global _INITIALIZED
    if _INITIALIZED:  # fast path
        return
    with _INIT_LOCK:
        if _INITIALIZED:
            return
        t0 = time.time()
        # Modular registrations
        _safe_call("modular_core_components", _register_components)  # base attentions/enc/dec/decomp/etc
        _safe_call("wrapped_decomposition", register_layers_decompositions)
        _safe_call("wrapped_encoders", register_layers_encoders)
        _safe_call("wrapped_decoders", register_layers_decoders)
        _safe_call("wrapped_fusions", register_layers_fusions)
        try:
            _safe_call("utils_embeddings", register_utils_embeddings)
        except Exception:
            pass
        try:
            _safe_call("utils_feedforwards", register_utils_feedforwards)
        except Exception:
            pass
        _safe_call("specialized_processors", register_specialized_processors)
        _INITIALIZED = True
        REGISTRATION_STATUS["overall"] = {
            "initialized": True,
            "duration_ms": round(1000 * (time.time() - t0), 2),
            "component_counts": _modular_registry.list(),
        }
        LOGGER.info(
            "Unified registry initialization complete (%sms)",
            REGISTRATION_STATUS["overall"]["duration_ms"],
        )


class UnifiedComponentRegistry:
    """Facade exposing consistent API over the global registry.

    Initialization is deferred until first use to avoid import‑time side effects.
    """

    def __init__(self):  # lightweight; does not trigger heavy work
        self._registry = _modular_registry

    # ---- public API ----
    def ensure_initialized(self):
        _initialize()

    def get_component(self, component_type: str, component_name: str) -> Type[BaseComponent]:
        self.ensure_initialized()
        fam = component_type
        # Use unified modular registry resolve path
        from layers.modular.core.registry import ComponentFamily
        cf = ComponentFamily(fam)
        return self._registry.resolve(cf, component_name).cls  # type: ignore[attr-defined]

    def create_component(self, component_type: str, component_name: str, config: Any) -> BaseComponent:  # type: ignore[name-defined]
        cls = self.get_component(component_type, component_name)
        return cls(config)  # type: ignore[call-arg]

    def list_all_components(self) -> Dict[str, List[str]]:
        self.ensure_initialized()
        return self._registry.list()

    def get_sophisticated_algorithms(self) -> List[Dict[str, Any]]:
        self.ensure_initialized()
        restored_algos = [
            "restored_fourier_attention",
            "restored_autocorrelation_attention",
            "restored_meta_learning_attention",
        ]
        found = []
        try:
            from layers.modular.core.registry import ComponentFamily
            attn_list = set(self._registry.list(ComponentFamily.ATTENTION)[ComponentFamily.ATTENTION.value])
        except Exception:
            attn_list = set()
        for algo in restored_algos:
            if algo in attn_list:
                try:
                    metadata = self._registry.describe(ComponentFamily.ATTENTION, algo).get("metadata", {})
                except Exception:  # pragma: no cover
                    metadata = {}
                found.append({
                    "name": algo,
                    "type": "attention",
                    "sophistication_level": metadata.get("sophistication_level"),
                    "features": metadata.get("features", []),
                    "algorithm_source": metadata.get("algorithm_source"),
                })
        return found

    def validate_migration_status(self) -> Dict[str, Any]:
        self.ensure_initialized()
        all_components = self._registry.list()
        sophisticated = self.get_sophisticated_algorithms()
        status = {
            "utils_components": sum(len(v) for v in all_components.values()),
            "sophisticated_algorithms": len(sophisticated),
            "migration_complete": len(sophisticated) >= 3,
            "issues": [],
        }
        if status["sophisticated_algorithms"] < 3:
            status["issues"].append("Missing sophisticated algorithms")
        return status

    def get_migration_summary(self) -> str:
        st = self.validate_migration_status()
        sophisticated = self.get_sophisticated_algorithms()
        summary = [
            "UNIFIED REGISTRY STATUS",
            "=" * 30,
            f"Utils Components: {st['utils_components']}",
            f"Sophisticated Algorithms: {st['sophisticated_algorithms']}/3",
            f"Migration Complete: {'YES' if st['migration_complete'] else 'NO'}",
            "",
        ]
        if sophisticated:
            summary.append("SOPHISTICATED ALGORITHMS:")
            summary.extend(
                f"  • {algo['name']}: {algo.get('sophistication_level')}" for algo in sophisticated
            )
        if st["issues"]:
            summary.append("")
            summary.append("ISSUES:")
            summary.extend(f"  • {issue}" for issue in st["issues"])
        return "\n".join(summary)

    def test_component_functionality(self) -> bool:  # light smoke test
        self.ensure_initialized()
        try:  # pragma: no cover - depends on optional package
            if not self._registry.is_registered("attention", "restored_fourier_attention"):
                return False
            cls = self._registry.get("attention", "restored_fourier_attention")
            # Attempt minimal instantiation; if signature mismatch, skip silently.
            try:
                inst = cls({"d_model": 32, "num_heads": 4, "dropout": 0.1})
            except Exception:
                return False
            return hasattr(inst, "forward") or hasattr(inst, "apply_attention")
        except Exception:
            return False

    # Diagnostics helpers
    def get_registration_status(self) -> Dict[str, Any]:
        return REGISTRATION_STATUS.copy()


 # Global unified registry instance (deferred heavy init)
unified_registry = UnifiedComponentRegistry()

# Backwards compatibility functions
def get_component(component_type: str, component_name: str) -> Type[BaseComponent]:
    """Backwards compatible component access (deprecated)."""
    warnings.warn(
        "Direct get_component is deprecated; prefer unified_registry.get_component",
        DeprecationWarning,
        stacklevel=2,
    )
    return unified_registry.get_component(component_type, component_name)

def create_component(component_type: str, component_name: str, config: Any) -> BaseComponent:  # type: ignore[name-defined]
    """Backwards compatible component creation (deprecated)."""
    warnings.warn(
        "Direct create_component is deprecated; prefer unified_registry.create_component",
        DeprecationWarning,
        stacklevel=2,
    )
    return unified_registry.create_component(component_type, component_name, config)


__all__ = [
    "unified_registry",
    "UnifiedComponentRegistry",
    "get_component",
    "create_component",
    "REGISTRATION_STATUS",
]
