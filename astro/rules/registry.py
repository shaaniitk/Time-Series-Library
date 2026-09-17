"""Closed operator vocabulary for the rule DSL.

Rules are data.  They name an operator from this registry; they never carry
code.  Adding a new operator is a deliberate code change with tests, which is
the only way new arithmetic enters the feature pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, NamedTuple

import numpy as np


class RuleContractError(ValueError):
    """Raised when a rule violates the operator contract."""


class EmitDef(NamedTuple):
    """One output channel of an operator instance."""

    key: str
    low: float
    high: float


# ayanamsha_rotation semantics:
#   "invariant" - adding a constant to every longitude leaves the output unchanged
#                 (relative geometry: separations, aspects, speeds)
#   "covariant" - the output depends on absolute zodiacal position and must change
#                 (rashi, nakshatra)
VALID_INVARIANCE = ("invariant", "covariant")


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    version: int
    fn: Callable
    inputs: tuple[str, ...]
    optional_inputs: tuple[str, ...]
    params: tuple[str, ...]
    optional_params: tuple[str, ...]
    invariance: str
    emits_resolver: Callable[[Mapping], tuple[EmitDef, ...]]
    doc: str = ""

    def resolve_emits(self, params: Mapping) -> tuple[EmitDef, ...]:
        emits = self.emits_resolver(params)
        keys = [emit.key for emit in emits]
        if len(set(keys)) != len(keys):
            raise RuleContractError(
                f"Operator {self.name!r} produced duplicate emit keys: {keys}."
            )
        return emits

    def validate_call(self, inputs: Mapping, params: Mapping) -> None:
        missing_inputs = [key for key in self.inputs if key not in inputs]
        if missing_inputs:
            raise RuleContractError(
                f"Operator {self.name!r} requires inputs {missing_inputs} which are absent."
            )
        allowed_inputs = set(self.inputs) | set(self.optional_inputs)
        unknown_inputs = sorted(set(inputs) - allowed_inputs)
        if unknown_inputs:
            raise RuleContractError(
                f"Operator {self.name!r} received unknown inputs {unknown_inputs}; "
                f"allowed keys are {sorted(allowed_inputs)}."
            )
        missing_params = [key for key in self.params if key not in params]
        if missing_params:
            raise RuleContractError(
                f"Operator {self.name!r} requires params {missing_params} which are absent."
            )
        allowed_params = set(self.params) | set(self.optional_params)
        unknown_params = sorted(set(params) - allowed_params)
        if unknown_params:
            raise RuleContractError(
                f"Operator {self.name!r} received unknown params {unknown_params}; "
                f"allowed keys are {sorted(allowed_params)}."
            )


OPERATOR_REGISTRY: dict[str, OperatorSpec] = {}


def _static_resolver(
    emits: Iterable[str], ranges: Mapping[str, tuple[float, float]]
) -> Callable[[Mapping], tuple[EmitDef, ...]]:
    resolved = tuple(
        EmitDef(key, float(ranges[key][0]), float(ranges[key][1])) for key in emits
    )

    def resolver(params: Mapping) -> tuple[EmitDef, ...]:
        return resolved

    return resolver


def operator(
    *,
    name: str,
    version: int,
    inputs: Iterable[str] = (),
    optional_inputs: Iterable[str] = (),
    params: Iterable[str] = (),
    optional_params: Iterable[str] = (),
    emits: Iterable[str] = (),
    ranges: Mapping[str, tuple[float, float]] | None = None,
    emits_resolver: Callable[[Mapping], tuple[EmitDef, ...]] | None = None,
    invariance: str,
):
    """Register a pure ephemeris transform under the closed vocabulary.

    The decorated function must have signature
    ``(provider, rows, inputs, params) -> dict[str, np.ndarray]`` and must be
    deterministic: identical arguments must yield bit-identical output.
    """

    if invariance not in VALID_INVARIANCE:
        raise ValueError(f"invariance must be one of {VALID_INVARIANCE}.")
    if emits_resolver is None:
        missing = [key for key in emits if key not in (ranges or {})]
        if missing:
            raise ValueError(f"Operator {name!r} is missing declared ranges for {missing}.")
        emits_resolver = _static_resolver(emits, ranges or {})
    elif emits:
        raise ValueError(
            f"Operator {name!r} declares both emits and emits_resolver; choose one."
        )

    def decorator(fn: Callable) -> Callable:
        if name in OPERATOR_REGISTRY:
            raise ValueError(f"Operator {name!r} is already registered.")
        OPERATOR_REGISTRY[name] = OperatorSpec(
            name=name,
            version=version,
            fn=fn,
            inputs=tuple(inputs),
            optional_inputs=tuple(optional_inputs),
            params=tuple(params),
            optional_params=tuple(optional_params),
            invariance=invariance,
            emits_resolver=emits_resolver,
            doc=(fn.__doc__ or "").strip(),
        )
        return fn

    return decorator


def get_operator(name: str) -> OperatorSpec:
    if name not in OPERATOR_REGISTRY:
        raise RuleContractError(
            f"Unknown operator {name!r}. Registered operators are "
            f"{sorted(OPERATOR_REGISTRY)}."
        )
    return OPERATOR_REGISTRY[name]


def operator_versions() -> dict[str, int]:
    """Operator name to version, folded into the feature manifest."""

    return {name: spec.version for name, spec in sorted(OPERATOR_REGISTRY.items())}
