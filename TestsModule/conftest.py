"""Pytest configuration & fixtures for TestsModule.

Fixtures here are local to new modular test hierarchy until migration completes.
"""
from __future__ import annotations
import os
import random
from typing import Iterator

import numpy as np
import pytest
import torch

_SEED = int(os.environ.get("TS_TEST_SEED", 1337))

@pytest.fixture(autouse=True)
def _global_seed() -> None:
    random.seed(_SEED)
    np.random.seed(_SEED)
    torch.manual_seed(_SEED)

@pytest.fixture()
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
