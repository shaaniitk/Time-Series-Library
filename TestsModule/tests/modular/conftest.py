"""Shared test fixtures for modular test suite."""
from __future__ import annotations

import pytest

from configs.modular_components import register_all_components, component_registry


@pytest.fixture(scope="session", autouse=True)
def register_components() -> None:
    """Auto-register all concrete components once per test session."""
    if not component_registry._components:
        register_all_components()
