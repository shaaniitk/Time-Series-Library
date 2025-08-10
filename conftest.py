"""Global pytest configuration utilities.

Purpose: Gate collection of legacy test tree under ``tests/`` until migration completes.
Only collect legacy tests when either environment variable ``INCLUDE_LEGACY_TESTS=1`` is set
or the CLI flag ``--include-legacy`` is provided. New modular split tests live under
``TestsModule`` and are always collected.
"""
from __future__ import annotations

import os
from pathlib import Path
import pytest


def pytest_addoption(parser: pytest.Parser) -> None:  # type: ignore[name-defined]
    """Register custom CLI flag to include legacy tests explicitly."""
    parser.addoption(
        "--include-legacy",
        action="store_true",
        default=False,
        help="Include legacy tests under tests/ (disabled by default while migration in progress)",
    )


def _is_legacy_path(p: Path) -> bool:
    """Return True if path belongs to legacy test tree.

    Accepts either Path-like or py.path.local provided by pytest.
    """
    try:
        path = Path(str(p))
    except Exception:  # pragma: no cover
        return False
    parts = {seg.lower() for seg in path.parts}
    return ("tests" in parts) and ("testsmodule" not in parts)


def pytest_ignore_collect(path, config: pytest.Config) -> bool:  # type: ignore[name-defined]
    """Dynamically ignore legacy test files unless explicitly enabled.

    This prevents import-time errors (missing optional deps, stale monoliths) from
    breaking CI focused on migrated modular suite.
    """
    try:
        include_legacy = config.getoption("include_legacy")  # CLI flag
    except Exception:  # pragma: no cover - defensive
        include_legacy = False

    # Don't block root discovery object '.'
    try:
        path_str = str(path)
    except Exception:  # pragma: no cover
        path_str = ""
    if path_str in {".", str(Path.cwd())}:
        return False
    if _is_legacy_path(path) and not (include_legacy or os.environ.get("INCLUDE_LEGACY_TESTS") == "1"):
        return True
    return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:  # type: ignore[name-defined]
    """Auto-mark legacy tests when they are explicitly included for visibility."""
    include_legacy = config.getoption("include_legacy") or os.environ.get("INCLUDE_LEGACY_TESTS") == "1"
    if not include_legacy:
        return
    for item in items:
        path = Path(str(item.fspath))
        if _is_legacy_path(path):
            item.add_marker(pytest.mark.legacy)  # type: ignore[attr-defined]
