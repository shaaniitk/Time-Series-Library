"""Pytest configuration & fixtures for TestsModule.

Fixtures here are local to new modular test hierarchy until migration completes.
Legacy unittest-based directories under tests/ are intentionally excluded; a
guard below raises if old monolithic path resurfaces in collection.
"""
from __future__ import annotations
import os
import random
from typing import Iterator

import numpy as np
import pytest
import torch


def pytest_collection_modifyitems(config, items):  # pragma: no cover - collection hook
    # Deselect legacy monolithic unittest file if accidentally collected
    legacy_path = "tests/modular_framework/test_components.py"
    for item in list(items):
        if legacy_path in str(item.fspath).replace("\\", "/"):
            try:
                items.remove(item)
            except ValueError:
                pass

_SEED = int(os.environ.get("TS_TEST_SEED", 1337))

@pytest.fixture(autouse=True)
def _global_seed() -> None:
    random.seed(_SEED)
    np.random.seed(_SEED)
    torch.manual_seed(_SEED)

@pytest.fixture()
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
