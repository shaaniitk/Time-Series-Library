"""Unified time series normalization utilities.

Provides a small, typed wrapper (TSNormalizer) that offers a consistent
fit/transform/inverse API across several normalization modes used in the
codebase (StandardScaler style, global min-max, and the dataframe-based
Normalizers already present in data_provider.uea).

Modes:
    standard        - Global mean/std per feature across all rows.
    minmax          - Global min/max per feature across all rows.
    normalizer_std  - Uses uea.Normalizer('standardization').
    normalizer_minmax - Uses uea.Normalizer('minmax').

The *normalizer_* variants retain the existing behaviour of Normalizer but
wrap it so consumers can use a uniform interface. Inverse transforms for the
Normalizer modes are supported because the Normalizer stores mean/std or
min/max after first call.

Note: Per-sample modes from Normalizer (per_sample_std, per_sample_minmax)
are not exposed here because FinancialDataManager handles aggregated single
continuous frames; introducing grouping semantics would require additional
context.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd

try:  # Re-use existing implementation if available
    from data_provider.uea import Normalizer as _UEANormalizer  # type: ignore
except Exception:  # pragma: no cover - defensive
    _UEANormalizer = None  # type: ignore


@dataclass
class NormalizationStats:
    """Container for normalization statistics."""
    mean: Optional[pd.Series] = None
    std: Optional[pd.Series] = None
    min_val: Optional[pd.Series] = None
    max_val: Optional[pd.Series] = None

    def as_dict(self) -> Dict[str, Dict[str, float]]:
        def _to(d: Optional[pd.Series]) -> Dict[str, float]:
            return {} if d is None else {k: float(v) for k, v in d.items()}
        return {
            "mean": _to(self.mean),
            "std": _to(self.std),
            "min": _to(self.min_val),
            "max": _to(self.max_val),
        }


class TSNormalizer:
    """Unified time series normalizer.

    Parameters
    ----------
    mode: str
        One of {'standard','minmax','normalizer_std','normalizer_minmax'}.
    eps: float
        Numerical stability offset.
    """
    SUPPORTED = {"standard", "minmax", "normalizer_std", "normalizer_minmax"}

    def __init__(self, mode: str = "standard", eps: float = 1e-8):
        if mode not in self.SUPPORTED:
            raise ValueError(f"Unsupported normalization mode: {mode}")
        self.mode = mode
        self.eps = eps
        self.stats = NormalizationStats()
        self._fitted = False
        self._uea_normalizer = None
        if mode.startswith("normalizer_") and _UEANormalizer is None:
            raise ImportError("UEA Normalizer not available; can't use mode='" + mode + "'")

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "TSNormalizer":
        """Fit statistics on provided DataFrame (numeric columns only)."""
        numeric = df.select_dtypes(include=["number"]).copy()
        if numeric.empty:
            raise ValueError("No numeric columns to normalize")

        if self.mode == "standard":
            self.stats.mean = numeric.mean()
            self.stats.std = numeric.std().replace(0, 1.0)
        elif self.mode == "minmax":
            self.stats.min_val = numeric.min()
            self.stats.max_val = numeric.max()
        elif self.mode == "normalizer_std":
            self._uea_normalizer = _UEANormalizer("standardization")
            # Invocation sets mean/std internally on first call
            _ = self._uea_normalizer.normalize(numeric)
            self.stats.mean = self._uea_normalizer.mean
            self.stats.std = self._uea_normalizer.std.replace(0, 1.0)
        elif self.mode == "normalizer_minmax":
            self._uea_normalizer = _UEANormalizer("minmax")
            _ = self._uea_normalizer.normalize(numeric)
            self.stats.min_val = self._uea_normalizer.min_val
            self.stats.max_val = self._uea_normalizer.max_val
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply normalization (numeric columns only); returns new DataFrame."""
        if not self._fitted:
            raise RuntimeError("TSNormalizer must be fit before transform().")
        out = df.copy()
        numeric_cols = out.select_dtypes(include=["number"]).columns
        if self.mode == "standard" or self.mode == "normalizer_std":
            out[numeric_cols] = (out[numeric_cols] - self.stats.mean) / (self.stats.std + self.eps)
        elif self.mode == "minmax" or self.mode == "normalizer_minmax":
            rng = (self.stats.max_val - self.stats.min_val + self.eps)
            out[numeric_cols] = (out[numeric_cols] - self.stats.min_val) / rng
        return out

    # ------------------------------------------------------------------
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse normalization if stats permit; returns new DataFrame."""
        if not self._fitted:
            raise RuntimeError("TSNormalizer must be fit before inverse_transform().")
        out = df.copy()
        numeric_cols = out.select_dtypes(include=["number"]).columns
        if self.mode == "standard" or self.mode == "normalizer_std":
            out[numeric_cols] = out[numeric_cols] * (self.stats.std + self.eps) + self.stats.mean
        elif self.mode == "minmax" or self.mode == "normalizer_minmax":
            rng = (self.stats.max_val - self.stats.min_val + self.eps)
            out[numeric_cols] = out[numeric_cols] * rng + self.stats.min_val
        return out

    # ------------------------------------------------------------------
    def fitted(self) -> bool:
        return self._fitted

    def info(self) -> Dict[str, Dict[str, float]]:
        return self.stats.as_dict()
