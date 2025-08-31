from .base import BaseAttention

# Keep the package init lightweight and future-proof.
# Avoid importing concrete implementations here. Each module registers itself
# with the unified component registry on import; tests explicitly import
# submodules to trigger registration.

__all__ = ["BaseAttention"]
