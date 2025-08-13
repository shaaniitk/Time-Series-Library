#!/usr/bin/env python3
"""Deprecated monolithic loss functionality tests.

Replaced by split TestsModule integration tests:
    - TestsModule/integration/test_loss_functionality_loss.py
    - TestsModule/integration/test_loss_functionality_quantile.py
    - TestsModule/integration/test_loss_functionality_output.py
    - TestsModule/integration/test_loss_functionality_registry.py

Kept as a tiny shim to avoid re-collection & preserve git history. Remove
after migration sign-off.
"""
import pytest

pytest.skip("Deprecated loss monolith replaced by split tests", allow_module_level=True)
