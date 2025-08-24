"""Threshold auto-tuning utility (Phase 4 - Step 2).

Run selected metric sampling functions multiple times to propose tightened
threshold values. It DOES NOT overwrite `thresholds.py`; instead it prints a
JSON object with suggested new thresholds based on robust statistics.

Method:
  For each metric function we collect N samples (default 15). For metrics
  expected to be small (error-like), we propose: new_threshold = median + k*IQR
  (k=1.5). For metrics with lower-bound expectations (effect sizes), we output
  observed minimum effect (median - k*IQR) so caller can ensure threshold stays
  BELOW that lower whisker (or choose a fraction of median).

Usage (example):
  pwsh -c "python -m tests.invariants.auto_threshold_tuner --samples 20 --json"

Extend METRICS dict with additional lambdas returning floats to include them.
"""
from __future__ import annotations

import argparse
import json
import statistics
import re
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch

from . import thresholds as TH
from . import metrics as M
from tests.helpers import time_series_generators as gen  # type: ignore


def _safe_import(path: str, attr: str):  # pragma: no cover - utility
    try:
        mod = __import__(path, fromlist=[attr])
        return getattr(mod, attr)
    except Exception:
        return None


def metric_encoder_determinism() -> float:
    fn = _safe_import("tests.invariants.test_encoder_invariants", "_get_structured_encoder")
    if fn is None:
        raise RuntimeError("encoder helper unavailable")
    enc = fn()
    enc.eval()
    x = torch.randn(2, 64, enc.get_d_model())
    with torch.no_grad():
        a = enc(x)
        b = enc(x)
    return (torch.linalg.norm(a - b) / (torch.linalg.norm(a) + 1e-12)).item()


def metric_adapter_pass_through() -> float:
    fn = _safe_import("tests.invariants.test_adapter_invariants", "_get_covariate_adapter")
    if fn is None:
        raise RuntimeError("adapter helper unavailable")
    ad = fn()
    ad.eval()
    seq = gen.seasonal_with_trend(batch=1, length=64, dim=1)
    out1 = ad._prepare_backbone_input(seq)
    out2 = ad._fuse_covariates(seq, None)
    return M.l2_relative_error(out1, out2)


def metric_multiscale_reconstruction_err() -> float:
    fn = _safe_import("tests.invariants.test_multiscale_energy_invariants", "_build_adapter")
    if fn is None:
        raise RuntimeError("multiscale adapter helper unavailable")
    ad = fn()
    ad.eval()
    seq = gen.seasonal_with_trend(batch=1, length=96, dim=1)
    with torch.no_grad():
        final = ad(seq)
    if not hasattr(ad, "_last_scale_outputs"):
        raise RuntimeError("scale outputs not captured")
    scales = getattr(ad, "_last_scale_outputs")
    concat = torch.cat(scales, dim=-1)
    B, L, _ = concat.shape
    X = concat.view(B * L, -1)
    Y = final.view(B * L, -1)
    XtX = X.T @ X + 1e-6 * torch.eye(X.size(1))
    w = torch.linalg.solve(XtX, X.T @ Y)
    recon = (X @ w).view_as(final)
    return (torch.linalg.norm(recon - final) / (torch.linalg.norm(final) + 1e-12)).item()


def metric_covariate_effect() -> float:
    fn = _safe_import("tests.invariants.test_adapter_invariants", "_get_covariate_adapter")
    if fn is None:
        raise RuntimeError("adapter helper unavailable")
    base = fn()
    from layers.modular.backbone.adapters import CovariateAdapter  # type: ignore
    cfg = {"covariate_dim": 3, "fusion_method": "project", "embedding_dim": base.backbone.get_d_model()}
    adapter = CovariateAdapter(base.backbone, cfg)
    seq = gen.seasonal_with_trend(batch=1, length=64, dim=1)
    cov = torch.randn(seq.size(0), seq.size(1), cfg["covariate_dim"])
    adapter.eval()
    with torch.no_grad():
        fused = adapter(seq, covariates=cov) if 'covariates' in adapter.forward.__code__.co_varnames else adapter._fuse_covariates(seq, cov)
        no_cov = adapter(seq, covariates=torch.zeros_like(cov)) if 'covariates' in adapter.forward.__code__.co_varnames else adapter._fuse_covariates(seq, torch.zeros_like(cov))
    delta = torch.linalg.norm(fused - no_cov) / (torch.linalg.norm(no_cov) + 1e-12)
    return float(delta)


METRICS: Dict[str, Tuple[Callable[[], float], str]] = {
    # key -> (callable, kind) where kind in {"upper", "lower"}
    "encoder_determinism_tol": (metric_encoder_determinism, "upper"),
    "adapter_pass_through_rel_err": (metric_adapter_pass_through, "upper"),
    "multiscale_energy_reconstruction_rel_err": (metric_multiscale_reconstruction_err, "upper"),
    "adapter_covariate_effect_min_delta": (metric_covariate_effect, "lower"),
}


def robust_suggest(values: List[float], kind: str, k: float) -> float:
    med = statistics.median(values)
    q1 = statistics.quantiles(values, n=4)[0]
    q3 = statistics.quantiles(values, n=4)[2]
    iqr = q3 - q1
    if kind == "upper":
        return med + k * iqr
    else:  # lower bound metric (effect size) -> propose whisker floor
        return max(0.0, med - k * iqr)


def main() -> None:  # pragma: no cover - CLI utility
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=15, help="Samples per metric")
    parser.add_argument("--k", type=float, default=1.5, help="IQR multiplier")
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    parser.add_argument("--apply", action="store_true", help="Write suggested thresholds into thresholds.py (in-place)")
    args = parser.parse_args()

    torch.manual_seed(0)

    proposals = {}
    for key, (fn, kind) in METRICS.items():
        current = TH.get_threshold(key)
        vals: List[float] = []
        for _ in range(args.samples):
            try:
                vals.append(float(fn()))
            except Exception as e:  # skip metric if missing dependency
                vals.append(float("nan"))
                print(f"[WARN] metric {key} failed: {e}")
                break
        clean = [v for v in vals if v == v]
        if not clean:
            continue
        suggestion = robust_suggest(clean, kind, args.k)
        proposals[key] = {
            "current": current,
            "median": statistics.median(clean),
            "iqr_multiplier": args.k,
            "suggested": suggestion,
            "kind": kind,
            "samples": len(clean),
            "values": clean,
        }

    if args.apply:
        # Load thresholds.py text and perform pattern substitution for keys we have suggestions for.
        th_path = Path(__file__).parent / "thresholds.py"
        text = th_path.read_text(encoding="utf-8")
        updated = text
        for k, info in proposals.items():
            # Only adjust if suggestion tighter (for upper) or higher (for lower effect size)
            kind = info["kind"]
            current = info["current"]
            suggested = info["suggested"]
            if kind == "upper" and suggested < current:
                pattern = rf"(\"{re.escape(k)}\"\s*:\s*)([0-9eE\.\-]+)"
                updated = re.sub(pattern, lambda m: f"{m.group(1)}{suggested:.6g}", updated, count=1)
            elif kind == "lower" and suggested > current:
                pattern = rf"(\"{re.escape(k)}\"\s*:\s*)([0-9eE\.\-]+)"
                updated = re.sub(pattern, lambda m: f"{m.group(1)}{suggested:.6g}", updated, count=1)
        if updated != text:
            th_path.write_text(updated + ("\n" if not updated.endswith("\n") else ""), encoding="utf-8")
            print("[APPLY] thresholds.py updated with tighter bounds")
        else:
            print("[APPLY] no changes applied (no tighter suggestions)")

    if args.json:
        print(json.dumps(proposals, indent=2))
    else:
        print("Suggested threshold updates:")
        for k, info in proposals.items():
            print(f"  {k}: current={info['current']:.3g} -> suggested={info['suggested']:.3g} (median={info['median']:.3g})")


if __name__ == "__main__":  # pragma: no cover
    main()

__all__ = ["METRICS", "robust_suggest"]
