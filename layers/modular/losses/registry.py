"""Compatibility shim for plural losses import path used in tests.

Provides LossRegistry and a stable get_loss_component wrapper that enforces
expected quantile alias semantics regardless of import order.
"""
from __future__ import annotations

from typing import Any, Tuple

from layers.modular.loss.registry import LossRegistry as _LossRegistry  # re-exported
from layers.modular.loss.registry import get_loss_component as _get_loss_component


def get_loss_component(name: str, **kwargs: Any) -> Tuple[Any, int]:
	"""Wrapper around singular-path get_loss_component with stricter semantics.

	For 'quantile'/'pinball', always return multiplier == len(quantiles)
	when available, falling back to instance attribute if needed.
	"""
	loss, mult = _get_loss_component(name, **kwargs)
	if name in {"quantile", "pinball"}:
		q = kwargs.get("quantiles", getattr(loss, "quantiles", None))
		if q is not None:
			try:
				mult = len(list(q))
			except Exception:
				pass
	return loss, mult


# Public re-exports
LossRegistry = _LossRegistry
__all__ = ["LossRegistry", "get_loss_component"]
