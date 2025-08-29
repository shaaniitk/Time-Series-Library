"""Lightweight invariant rule harness."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any, List


@dataclass
class RuleResult:
    name: str
    metric: float
    passed: bool
    details: Dict[str, Any]


InvariantRule = Callable[[], RuleResult]


def run_rules(rules: List[InvariantRule]) -> List[RuleResult]:
    return [r() for r in rules]


__all__ = ["RuleResult", "run_rules"]
