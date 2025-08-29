"""Pytest plugin: enforce maximum allowed skipped tests.

Environment variable EXPECTED_MAX_SKIPS defines ceiling (default 3).
Fails session if actual skip count exceeds this.
"""
from __future__ import annotations

import os
import pytest

_skip_reports = []


def pytest_runtest_logreport(report: pytest.TestReport):  # type: ignore
    if report.when == "call" and report.skipped:
        _skip_reports.append(report)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int):  # type: ignore
    max_skips = int(os.getenv("EXPECTED_MAX_SKIPS", "3"))
    skip_count = len(_skip_reports)
    if skip_count > max_skips:
        session.exitstatus = 1
        print(f"[SKIP_GUARD] Fail: {skip_count} skips > allowed {max_skips}")
    else:
        print(f"[SKIP_GUARD] OK: {skip_count} skips (limit {max_skips})")
