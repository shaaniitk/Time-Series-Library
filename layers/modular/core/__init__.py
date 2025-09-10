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
from .registry import ComponentFamily, component_registry as unified_registry  # re-export for convenience

__all__ = ["ComponentFamily", "unified_registry"]
