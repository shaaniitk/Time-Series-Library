"""Global pytest configuration utilities.

Purpose: Gate collection of legacy test tree under ``tests/`` until migration completes.
Only collect legacy tests when either environment variable ``INCLUDE_LEGACY_TESTS=1`` is set
or the CLI flag ``--include-legacy`` is provided. New modular split tests live under
``TestsModule`` and are always collected.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import random
import pytest


def pytest_addoption(parser: pytest.Parser) -> None:  # type: ignore[name-defined]
    """Register custom CLI flag to include legacy tests explicitly."""
    parser.addoption(
        "--include-legacy",
        action="store_true",
        default=False,
        help="Include legacy tests under tests/ (disabled by default while migration in progress)",
    )
    parser.addoption(
        "--enforce-venv",
        action="store_true",
        default=True,
        help="Fail fast if not running inside expected project virtual environment (tsl-env)",
    )
    parser.addoption(
        "--smoke-sample",
        action="store",
        type=int,
        default=0,
        help="Randomly mark N extended tests as smoke if they lack an explicit marker (for progressive curation)",
    )
    parser.addoption(
        "--enforce-classification",
        action="store_true",
        default=True,
        help="Ensure every test has at least one classification marker (smoke/extended/perf/quarantine/legacy)",
    )
    parser.addoption(
        "--auto-classify-missing",
        action="store_true",
        default=True,
        help="Automatically assign a marker (heuristic) to unmarked tests instead of failing",
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
    """Auto-mark legacy tests when included; optionally sample smoke tests; enforce venv."""
    # 1. Enforce virtual environment guard
    if config.getoption("enforce_venv"):
        expected_name = "tsl-env"
        venv_prefix = sys.prefix.lower()
        if expected_name not in venv_prefix:
            pytest.exit(
                f"Test run aborted: expected to run inside virtual environment containing '{expected_name}' in path, got: {sys.prefix}\n"
                "Activate with: .\\tsl-env\\Scripts\\Activate.ps1", returncode=2
            )

    # 2. Legacy marking
    include_legacy = config.getoption("include_legacy") or os.environ.get("INCLUDE_LEGACY_TESTS") == "1"
    if not include_legacy:
        # proceed to potential smoke sampling
        pass
    else:
        for item in items:
            path = Path(str(item.fspath))
            if _is_legacy_path(path):
                item.add_marker(pytest.mark.legacy)  # type: ignore[attr-defined]

    # 3. Opportunistic smoke sampling (only for extended tests without explicit smoke marker)
    sample_n = config.getoption("smoke_sample")
    if sample_n and sample_n > 0:
        extended_items = [it for it in items if any(m.name == "extended" for m in it.own_markers) and not any(m.name == "smoke" for m in it.own_markers)]
        if sample_n < len(extended_items):
            random.seed(42)
            sampled = random.sample(extended_items, sample_n)
        else:
            sampled = extended_items
        for s in sampled:
            s.add_marker(pytest.mark.smoke)  # type: ignore[attr-defined]

    # 4. Marker enforcement & auto-classification
    enforce = config.getoption("enforce_classification") and os.environ.get("DISABLE_MARKER_ENFORCEMENT") != "1"
    auto_classify = config.getoption("auto_classify_missing") or os.environ.get("AUTO_CLASSIFY_MISSING") == "1"
    classification_set = {"smoke", "extended", "perf", "quarantine", "legacy"}
    unmarked: list[pytest.Item] = []
    auto_assigned: list[tuple[str, str]] = []  # (nodeid, marker)
    if enforce:
        for it in items:
            if any(m.name in classification_set for m in it.own_markers):
                continue
            # Ignore internal pytest items (doctest, etc.)
            nodeid = it.nodeid.lower()
            if nodeid.startswith("::"):
                continue
            if auto_classify:
                # Heuristic classification
                if "smoke" in nodeid or "/smoke/" in nodeid:
                    mark_name = "smoke"
                elif "perf" in nodeid or "/perf/" in nodeid:
                    mark_name = "perf"
                elif "quarantine" in nodeid or "/quarantine/" in nodeid:
                    mark_name = "quarantine"
                elif "legacy" in nodeid or "/tests/" in nodeid:
                    mark_name = "legacy"
                else:
                    mark_name = "extended"
                it.add_marker(getattr(pytest.mark, mark_name))  # type: ignore[attr-defined]
                auto_assigned.append((it.nodeid, mark_name))
            else:
                unmarked.append(it)
    # Stash stats on config for terminal summary
    config._marker_enforcement_stats = {  # type: ignore[attr-defined]
        "auto_assigned": auto_assigned,
        "unmarked": [u.nodeid for u in unmarked],
    }
    if enforce and not auto_classify and unmarked:
        missing_list = "\n".join(u.nodeid for u in unmarked)
        pytest.exit(
            f"Marker enforcement failed: {len(unmarked)} test(s) lack a classification marker (smoke/extended/perf/quarantine/legacy).\n"\
            f"List (first 20):\n{os.linesep.join(list(missing_list.splitlines())[:20])}\n"\
            "Add appropriate @pytest.mark.<marker> or disable enforcement with --enforce-classification=0.",
            returncode=3,
        )


def pytest_terminal_summary(config: pytest.Config, terminalreporter):  # type: ignore[no-untyped-def]
    stats = getattr(config, "_marker_enforcement_stats", None)
    if not stats:
        return
    auto_assigned = stats.get("auto_assigned", [])
    unmarked = stats.get("unmarked", [])
    if auto_assigned:
        terminalreporter.write_line(
            f"[marker-enforcement] Auto-assigned markers to {len(auto_assigned)} test(s): "
            + ", ".join(f"{m}:{n}" for m, n in auto_assigned[:5])
            + (" ..." if len(auto_assigned) > 5 else "")
        )
    if unmarked:
        terminalreporter.write_line(f"[marker-enforcement] {len(unmarked)} unmarked test(s) (enforcement disabled or auto-classify off)")
