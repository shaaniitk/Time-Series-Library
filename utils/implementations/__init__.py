"""Implementation shims for registering components into unified registry.

This package exposes helper functions used by tests to register utils components.
"""

try:  # re-export for convenience
    from .embedding.wrapped_embeddings import register_utils_embeddings  # noqa: F401
except Exception:  # pragma: no cover
    pass
