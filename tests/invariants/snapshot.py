"""Metric snapshot management for invariant tests.

Snapshots live under .test_artifacts/component_metrics/<component>/<name>.json
Environment variable UPDATE_METRIC_SNAPSHOTS=1 forces regeneration.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any


ARTIFACT_ROOT = Path('.test_artifacts') / 'component_metrics'


@dataclass
class SnapshotResult:
    created: bool
    updated: bool
    path: Path
    metrics: Dict[str, float]
    diffs: Dict[str, float]


def _load_existing(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    try:
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        return {k: float(v) for k, v in data.items()}
    except Exception:
        return {}


def save_or_compare(component: str, name: str, metrics: Dict[str, float], rel_tol: float = 0.2) -> SnapshotResult:
    """Save metrics snapshot or compare with existing.

    Args:
        component: component type (e.g. 'decomposition')
        name: specific implementation label
        metrics: mapping metric->value
        rel_tol: allowable relative drift before flagging (test should assert on diffs)
    """
    update = os.getenv('UPDATE_METRIC_SNAPSHOTS') == '1'
    comp_dir = ARTIFACT_ROOT / component
    comp_dir.mkdir(parents=True, exist_ok=True)
    path = comp_dir / f'{name}.json'
    existing = _load_existing(path)
    if not existing or update:
        with path.open('w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        return SnapshotResult(created=not bool(existing), updated=update and bool(existing), path=path, metrics=metrics, diffs={})
    diffs: Dict[str, float] = {}
    for k, v in metrics.items():
        if k in existing:
            base = existing[k] if existing[k] != 0 else 1e-12
            diffs[k] = abs(v - existing[k]) / abs(base)
    exceed = {k: d for k, d in diffs.items() if d > rel_tol}
    return SnapshotResult(created=False, updated=False, path=path, metrics=metrics, diffs=exceed)


__all__ = ["save_or_compare", "SnapshotResult"]
