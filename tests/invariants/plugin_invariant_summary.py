"""Pytest plugin to emit invariant summary JSON at session end (Phase 4 - Step 4)."""
from __future__ import annotations

import json
import time
import pathlib
from typing import List, Dict, Any

import pytest

_records: List[Dict[str, Any]] = []
_start_time = time.perf_counter()


def pytest_runtest_logreport(report: pytest.TestReport):  # type: ignore
    if report.when != "call":
        return
    nodeid = report.nodeid
    status = "passed" if report.passed else "failed" if report.failed else "skipped"
    _records.append({
        "nodeid": nodeid,
        "outcome": status,
        "duration": getattr(report, 'duration', None),
        "keywords": list(report.keywords),
    })


def pytest_sessionfinish(session: pytest.Session, exitstatus: int):  # type: ignore
    duration = time.perf_counter() - _start_time
    summary = {
        "total": len(_records),
        "passed": sum(r['outcome'] == 'passed' for r in _records),
        "failed": sum(r['outcome'] == 'failed' for r in _records),
        "skipped": sum(r['outcome'] == 'skipped' for r in _records),
        "exitstatus": exitstatus,
        "duration_sec": duration,
        "tests": _records,
    }
    out_dir = pathlib.Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    with (out_dir / "invariant_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[INVARIANT_SUMMARY] written to {out_dir / 'invariant_summary.json'}")

__all__ = []
