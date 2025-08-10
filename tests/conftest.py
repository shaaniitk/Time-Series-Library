"""Central pytest configuration: seeds, lightweight fixtures, and shared utilities.

Part of progressive test consolidation.
"""
from __future__ import annotations

import os
import random
from typing import Iterator

import numpy as np
import pytest

GLOBAL_SEED = int(os.environ.get("TSL_GLOBAL_TEST_SEED", "1337"))


def _set_seeds(seed: int) -> None:
    """Set seeds for deterministic test behavior (torch optional)."""
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch optional
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover
        pass


_set_seeds(GLOBAL_SEED)


def pytest_configure(config: pytest.Config) -> None:  # noqa: D401
    """Register custom markers (avoid PytestUnknownMark warnings)."""
    config.addinivalue_line("markers", "smoke: Fast minimal surface tests")
    config.addinivalue_line("markers", "perf: Performance / benchmark tests")


@pytest.fixture(scope="session")
def synthetic_series() -> np.ndarray:
    """Return a small deterministic synthetic time series for quick tests."""
    x = np.linspace(0, 4 * np.pi, 128)
    return np.sin(x) + 0.1 * np.cos(3 * x)


@pytest.fixture(scope="function")
def seed_context() -> Iterator[int]:
    """Ensure each test function runs with a fresh deterministic seed.

    Yields the seed used so tests can reference it if needed.
    """
    _set_seeds(GLOBAL_SEED)
    yield GLOBAL_SEED
    _set_seeds(GLOBAL_SEED)
