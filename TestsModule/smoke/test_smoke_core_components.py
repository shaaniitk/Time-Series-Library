"""Minimal smoke tests for core modular components.

Fast (<1s) existence + instantiation + forward shape checks.
Skips gracefully if registry or specific components unavailable.
Maintains a simple inventory snapshot to detect component add/remove impact.
"""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, List

import pytest

pytestmark = [pytest.mark.smoke]


def _import_registry():  # type: ignore[return-type]
    """Return (create_component, list_components) from whichever registry is present.

    Tries new path first (layers.modular.registry) then legacy path.
    Returns (None, None) if neither present so tests can skip.
    """
    try:  # New style
        from layers.modular.registry import create_component, list_components  # type: ignore
        return create_component, list_components
    except Exception:  # pragma: no cover
        pass
    try:  # Legacy style
        from utils.modular_components.registry import create_component, get_global_registry  # type: ignore

        def list_components(kind: str):  # adapter for legacy
            reg = get_global_registry()
            return reg.list_components(kind) if hasattr(reg, "list_components") else []  # type: ignore

        return create_component, list_components
    except Exception:  # pragma: no cover
        return None, None


@pytest.fixture(scope="session")
def registry_api():  # type: ignore[return-type]
    api = _import_registry()
    if api == (None, None):
        pytest.skip("No component registry available in this build")
    return api


def test_loss_instantiation_and_value(registry_api):  # type: ignore
    create_component, list_components = registry_api
    # Pick two deterministic looking losses if present; fall back to any.
    try:
        losses_raw = list_components("loss")  # type: ignore[arg-type]
        if isinstance(losses_raw, dict):  # registry style {name: class}
            losses = [n for n in losses_raw.keys() if n != "loss"]
        else:
            losses = [n for n in losses_raw if n != "loss"]
    except Exception:  # pragma: no cover
        losses = []
    if not losses:
        pytest.skip("No loss components registered")
    preferred = [n for n in losses if "mse" in n or "mae" in n or "frequency" in n]
    sample = (preferred or losses)[:2]
    import torch
    x = torch.randn(2, 4, 3)
    y = torch.randn(2, 4, 3)
    for name in sample:
        loss = create_component("loss", name, {"reduction": "mean"})  # type: ignore[arg-type]
        val = loss(x, y)
        assert val.dim() == 0 and torch.isfinite(val)


def test_attention_forward_minimal(registry_api):  # type: ignore
    create_component, list_components = registry_api
    names = []
    try:
        res = list_components("attention")  # type: ignore[arg-type]
        if isinstance(res, dict):
            names = list(res.keys())
        else:
            names = list(res)
    except Exception:  # pragma: no cover
        pass
    target_name = None
    # Prefer basic or optimized autocorrelation style for speed
    if names:
        for pref in ["optimized_autocorrelation", "autocorrelation", "adaptive_autocorrelation", "enhanced_autocorrelation", "multi_head"]:
            if pref in names:
                target_name = pref
                break
        if target_name is None:
            target_name = names[0]
    if not target_name:
        pytest.skip("No attention components registered")
    # Minimal config guesses; ignore unexpected param failures gracefully
    base_cfg = {"d_model": 16, "n_heads": 2, "dropout": 0.0}
    try:
        attn = create_component("attention", target_name, base_cfg)  # type: ignore[arg-type]
    except Exception:
        # Try empty config; if still failing skip (smoke shouldn't hard fail on cfg drift)
        try:
            attn = create_component("attention", target_name, {})  # type: ignore[arg-type]
        except Exception:
            pytest.skip(f"Could not instantiate attention component '{target_name}'")
    import torch

    x = torch.randn(1, 12, 16)
    out = attn(x, x, x)
    if isinstance(out, tuple):  # (output, weights)
        out = out[0]
    assert out.shape == x.shape


def test_component_inventory_change_detection(registry_api):  # type: ignore
    """Detect addition/removal of registered components vs baseline snapshot.

    Baseline stored at .test_baseline/component_inventory.json. If absent we create
    it and xfail once (bootstrap). Removals fail. Additions only fail when
    STRICT_COMPONENT_DIFF=1; optionally auto-update when ALLOW_COMPONENT_SNAPSHOT_UPDATE=1.
    """
    _, list_components = registry_api
    kinds = ["backbone", "attention", "processor", "loss", "output"]
    inventory: Dict[str, List[str]] = {}
    for k in kinds:
        try:
            res = list_components(k)  # type: ignore[arg-type]
            if isinstance(res, dict):
                inventory[k] = sorted(res.keys())
            else:
                inventory[k] = sorted(list(res))
        except Exception:  # pragma: no cover
            inventory[k] = []
    base_dir = Path(".test_baseline")
    base_dir.mkdir(exist_ok=True)
    snap = base_dir / "component_inventory.json"
    if not snap.exists():
        snap.write_text(json.dumps(inventory, indent=2))
        pytest.xfail("Baseline created; re-run for change detection")
    old = json.loads(snap.read_text())
    removed = {k: sorted(set(old.get(k, [])) - set(inventory.get(k, []))) for k in kinds}
    added = {k: sorted(set(inventory.get(k, [])) - set(old.get(k, []))) for k in kinds}
    removed_flat = [r for rlist in removed.values() for r in rlist]
    added_flat = [a for alist in added.values() for a in alist]
    if removed_flat:
        pytest.fail(f"Removed components detected: {removed}")
    if added_flat and os.environ.get("STRICT_COMPONENT_DIFF") == "1":
        pytest.fail(f"New components added (STRICT mode): {added}")
    if added_flat and os.environ.get("ALLOW_COMPONENT_SNAPSHOT_UPDATE") == "1":
        snap.write_text(json.dumps(inventory, indent=2))
