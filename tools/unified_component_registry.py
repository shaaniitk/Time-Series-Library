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

from utils.modular_components.registry import _global_registry
from utils.modular_components.base_interfaces import BaseComponent

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

# Register wrapped legacy attentions from layers folder
try:
    from utils.implementations.attention.layers_wrapped_attentions import register_layers_attentions
    _HAS_LAYER_ATTENTION_WRAPPERS = True
except Exception:
    _HAS_LAYER_ATTENTION_WRAPPERS = False

# Register wrapped legacy decomposition processors
try:
    from utils.implementations.decomposition.wrapped_decompositions import register_layers_decompositions
    _HAS_LAYER_DECOMP_WRAPPERS = True
except Exception:
    _HAS_LAYER_DECOMP_WRAPPERS = False

# Register wrapped legacy encoder processors
try:
    from utils.implementations.encoder.wrapped_encoders import register_layers_encoders
    _HAS_LAYER_ENCODER_WRAPPERS = True
except Exception:
    _HAS_LAYER_ENCODER_WRAPPERS = False

# Register wrapped legacy decoder processors
try:
    from utils.implementations.decoder.wrapped_decoders import register_layers_decoders
    _HAS_LAYER_DECODER_WRAPPERS = True
except Exception:
    _HAS_LAYER_DECODER_WRAPPERS = False

# Register wrapped legacy fusion processors
try:
    from utils.implementations.fusion.wrapped_fusions import register_layers_fusions
    _HAS_LAYER_FUSION_WRAPPERS = True
except Exception:
    _HAS_LAYER_FUSION_WRAPPERS = False

# Register utils loss implementations
try:
    from utils.implementations.loss.wrapped_losses import register_utils_losses
    _HAS_UTILS_LOSSES = True
except Exception:
    _HAS_UTILS_LOSSES = False

# Register utils output implementations
try:
    from utils.implementations.output.wrapped_outputs import register_utils_outputs
    _HAS_UTILS_OUTPUTS = True
except Exception:
    _HAS_UTILS_OUTPUTS = False

# Register legacy layers output head wrappers
try:
    from utils.implementations.output.layers_wrapped_outputs import register_layers_output_heads
    _HAS_LAYER_OUTPUT_WRAPPERS = True
except Exception:
    _HAS_LAYER_OUTPUT_WRAPPERS = False

# Register utils embedding implementations
try:
    from utils.implementations.embedding.wrapped_embeddings import register_utils_embeddings
    _HAS_UTILS_EMBEDDINGS = True
except Exception:
    _HAS_UTILS_EMBEDDINGS = False

# Register utils feedforward implementations
try:
    from utils.implementations.feedforward.wrapped_feedforward import register_utils_feedforwards
    _HAS_UTILS_FFN = True
except Exception:
    _HAS_UTILS_FFN = False

# Register utils adapter implementations
try:
    from utils.implementations.adapter.wrapped_adapters import register_utils_adapters
    _HAS_UTILS_ADAPTERS = True
except Exception:
    _HAS_UTILS_ADAPTERS = False

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
        # Sophisticated / restored algorithms
        if ALGORITHMS_AVAILABLE:
            _safe_call("restored_algorithms", register_restored_algorithms)
        # Legacy / wrapped groups (each individually guarded)
        if _HAS_LAYER_ATTENTION_WRAPPERS:
            _safe_call("wrapped_attentions", register_layers_attentions)
        if _HAS_LAYER_DECOMP_WRAPPERS:
            _safe_call("wrapped_decomposition", register_layers_decompositions)
        if _HAS_LAYER_ENCODER_WRAPPERS:
            _safe_call("wrapped_encoders", register_layers_encoders)
        if _HAS_LAYER_DECODER_WRAPPERS:
            _safe_call("wrapped_decoders", register_layers_decoders)
        if _HAS_LAYER_FUSION_WRAPPERS:
            _safe_call("wrapped_fusions", register_layers_fusions)
        if _HAS_UTILS_LOSSES:
            _safe_call("utils_losses", register_utils_losses)
        if _HAS_UTILS_OUTPUTS:
            _safe_call("utils_outputs", register_utils_outputs)
        if _HAS_LAYER_OUTPUT_WRAPPERS:
            _safe_call("legacy_output_heads", register_layers_output_heads)
        if _HAS_UTILS_EMBEDDINGS:
            _safe_call("utils_embeddings", register_utils_embeddings)
        if _HAS_UTILS_FFN:
            _safe_call("utils_feedforwards", register_utils_feedforwards)
        if _HAS_UTILS_ADAPTERS:
            _safe_call("utils_adapters", register_utils_adapters)
        _INITIALIZED = True
        REGISTRATION_STATUS["overall"] = {
            "initialized": True,
            "duration_ms": round(1000 * (time.time() - t0), 2),
            "component_counts": {
                k: len(v) for k, v in _global_registry.list_components().items()
            },
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
        self._registry = _global_registry

    # ---- public API ----
    def ensure_initialized(self):
        _initialize()

    def get_component(self, component_type: str, component_name: str) -> Type[BaseComponent]:
        self.ensure_initialized()
        return self._registry.get(component_type, component_name)

    def create_component(self, component_type: str, component_name: str, config: Any) -> BaseComponent:  # type: ignore[name-defined]
        cls = self.get_component(component_type, component_name)
        return cls(config)

    def list_all_components(self) -> Dict[str, List[str]]:
        self.ensure_initialized()
        return self._registry.list_components()

    def get_sophisticated_algorithms(self) -> List[Dict[str, Any]]:
        self.ensure_initialized()
        restored_algos = [
            "restored_fourier_attention",
            "restored_autocorrelation_attention",
            "restored_meta_learning_attention",
        ]
        found = []
        for algo in restored_algos:
            if self._registry.is_registered("attention", algo):
                try:
                    metadata = self._registry.get_metadata("attention", algo)
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
        all_components = self._registry.list_components()
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
