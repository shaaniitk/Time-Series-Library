"""Deprecated monolithic Bayesian attention module.

All classes were split into dedicated modules under
`layers/modular/attention/bayesian/`.

Temporary re-export is kept for backward compatibility and will be removed
once downstream references are migrated.
"""
from __future__ import annotations
from .bayesian.bayesian_linear import BayesianLinear  # type: ignore F401
from .bayesian.bayesian_attention import BayesianAttention  # type: ignore F401
from .bayesian.bayesian_multi_head_attention import BayesianMultiHeadAttention  # type: ignore F401
from .bayesian.variational_attention import VariationalAttention  # type: ignore F401
from .bayesian.bayesian_cross_attention import BayesianCrossAttention  # type: ignore F401

__all__ = [
    "BayesianLinear",
    "BayesianAttention",
    "BayesianMultiHeadAttention",
    "VariationalAttention",
    "BayesianCrossAttention",
]

