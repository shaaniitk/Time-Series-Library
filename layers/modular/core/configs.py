"""Compatibility shim for configuration schemas.

This module preserves legacy import paths used by tests and examples:

    from layers.modular.core.configs import ...

It re-exports all schema definitions from the single source-of-truth
`configs.schemas` to avoid duplication.
"""

# Re-export everything from the central schemas module
from configs.schemas import *  # noqa: F401,F403

# Optional: define __all__ if present in configs.schemas, otherwise allow wildcard
try:  # pragma: no cover - best-effort compatibility
    from configs.schemas import __all__  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover
    pass
