"""Lightweight test runtime logging utilities.

Writes per-test rows to a CSV for performance/regression tracking.

CSV path: `.test_artifacts/metrics_runtime.csv`
Columns:
  timestamp_iso, commit, test_nodeid, duration_sec, status, markers, extra_metrics_json

Set environment variable `METRICS_RUNTIME_DISABLE=1` to disable logging.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List
import json
import os
import datetime as _dt
import subprocess

ARTIFACT_ROOT = Path('.test_artifacts')
CSV_PATH = ARTIFACT_ROOT / 'metrics_runtime.csv'
HEADER = ['timestamp_iso','commit','test_nodeid','duration_sec','status','markers','extra_metrics_json']

def _now_iso() -> str:
    return _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc).isoformat()

def _git_commit() -> str:
    try:
        return subprocess.check_output(['git','rev-parse','--short','HEAD'], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'

def ensure_header() -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        CSV_PATH.write_text(','.join(HEADER) + '\n', encoding='utf-8')

def append_rows(rows: Iterable[List[str]]) -> None:
    if os.getenv('METRICS_RUNTIME_DISABLE') == '1':
        return
    ensure_header()
    with CSV_PATH.open('a', encoding='utf-8') as f:
        for r in rows:
            f.write(','.join(r) + '\n')

def build_row(nodeid: str, duration: float, status: str, markers: Iterable[str], extra: dict | None = None) -> List[str]:
    extra_json = json.dumps(extra or {})
    return [
        _now_iso(),
        _git_commit(),
        nodeid,
        f"{duration:.6f}",
        status,
        ';'.join(sorted(set(markers))),
        extra_json.replace(',', ';'),  # keep CSV simple; semi-colon inside JSON
    ]

__all__ = ['build_row','append_rows','CSV_PATH']
