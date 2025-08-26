"""Core unified modular infrastructure package.

Usage pattern:

		from layers.modular.core import unified_registry, ComponentFamily
		import layers.modular.core.register_components  # populate registry
		attn = unified_registry.create(ComponentFamily.ATTENTION, 'fourier_attention', d_model=512, n_heads=8, seq_len=96)

Design notes:
* Registration is opt‑in (no heavy side effects on import of this package).
* All component families share a single `unified_registry` keyed by
	`ComponentFamily` + name for easier global introspection and tooling.
* Legacy per‑family registries forward with DeprecationWarning until removed.
"""
from .registry import unified_registry, ComponentFamily, get_attention_component  # re-export for convenience

# Also re-export the advanced registration hook for specialized processors expected in tests
try:
	from .register_advanced import register_specialized_processors  # type: ignore
except Exception:  # pragma: no cover
	register_specialized_processors = None  # type: ignore

__all__ = ["unified_registry", "ComponentFamily", "get_attention_component", "register_specialized_processors"]
