"""Compile a declarative ruleset into a named known-future feature matrix.

Compilation runs on the full daily ephemeris grid so event and response
channels see real calendar time; the loader subsequently samples market
sessions out of the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from astro.compile.response_bank import (
    elapsed_days_from_index,
    ewma_response,
    is_uniform_daily,
)
from astro.ephemeris.provider import EphemerisProvider
from astro.rules.naming import channel_name, channel_name_hash
from astro.rules.registry import RuleContractError, get_operator, operator_versions
from astro.rules.schema import RuleSet, RuleSpec

_RANGE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ChannelMeta:
    """Provenance for one compiled channel. The index-to-rule source of truth."""

    name: str
    rule_id: str
    family: str
    school: str
    operator: str
    emit_key: str
    half_life_days: int | None
    low: float
    high: float
    prior_importance: float
    l2_anchor: float
    l1_weight: float
    arm: str

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "rule_id": self.rule_id,
            "family": self.family,
            "school": self.school,
            "operator": self.operator,
            "emit_key": self.emit_key,
            "half_life_days": self.half_life_days,
            "range": [self.low, self.high],
            "prior_importance": self.prior_importance,
            "arm": self.arm,
        }


@dataclass(frozen=True)
class CompiledFeatures:
    names: tuple[str, ...]
    matrix: np.ndarray
    channel_meta: tuple[ChannelMeta, ...]
    index: pd.DatetimeIndex
    manifest: dict

    def __post_init__(self) -> None:
        if self.matrix.shape != (len(self.index), len(self.names)):
            raise RuleContractError(
                f"Compiled matrix shape {self.matrix.shape} does not match "
                f"({len(self.index)}, {len(self.names)})."
            )

    @property
    def channel_count(self) -> int:
        return len(self.names)

    def select_rows(self, rows: np.ndarray) -> np.ndarray:
        return self.matrix[rows, :]

    def families(self) -> tuple[str, ...]:
        seen: list[str] = []
        for meta in self.channel_meta:
            if meta.family not in seen:
                seen.append(meta.family)
        return tuple(seen)

    def family_channel_indices(self) -> dict[str, tuple[int, ...]]:
        grouped: dict[str, list[int]] = {}
        for position, meta in enumerate(self.channel_meta):
            grouped.setdefault(meta.family, []).append(position)
        return {family: tuple(items) for family, items in grouped.items()}

    def rule_channel_indices(self) -> dict[str, tuple[int, ...]]:
        grouped: dict[str, list[int]] = {}
        for position, meta in enumerate(self.channel_meta):
            grouped.setdefault(meta.rule_id, []).append(position)
        return {rule_id: tuple(items) for rule_id, items in grouped.items()}


def _check_ephemeris_requirements(ruleset: RuleSet, provider: EphemerisProvider) -> None:
    requirements = dict(ruleset.requires_ephemeris)
    manifest = provider.manifest
    for key, expected in requirements.items():
        if key == "frame":
            wanted = expected if isinstance(expected, list) else [expected]
            missing = [frame for frame in wanted if frame not in manifest.available_frames]
            if missing:
                raise RuleContractError(
                    f"Ruleset {ruleset.ruleset_id!r} requires frames {wanted} but the "
                    f"ephemeris provides {list(manifest.available_frames)}."
                )
            continue
        actual = getattr(manifest, key, None)
        if actual is None:
            raise RuleContractError(
                f"Ruleset requires ephemeris property {key!r}, which the manifest does not declare."
            )
        if actual != expected:
            raise RuleContractError(
                f"Ruleset {ruleset.ruleset_id!r} requires {key}={expected!r} but the "
                f"ephemeris declares {actual!r}."
            )


def _enforce_range(values: np.ndarray, low: float, high: float, name: str) -> np.ndarray:
    if not np.isfinite(values).all():
        raise RuleContractError(f"Channel {name!r} produced a non-finite value.")
    minimum = float(values.min())
    maximum = float(values.max())
    if minimum < low - _RANGE_TOLERANCE or maximum > high + _RANGE_TOLERANCE:
        raise RuleContractError(
            f"Channel {name!r} left its declared range [{low}, {high}]: observed "
            f"[{minimum}, {maximum}]. Known-future channels bypass the scaler, so "
            "operators must emit bounded values."
        )
    return values


def _compile_rule(
    rule: RuleSpec,
    provider: EphemerisProvider,
    elapsed_days: np.ndarray | None,
    arm: str,
) -> tuple[list[str], list[np.ndarray], list[ChannelMeta]]:
    spec = get_operator(rule.operator)
    params = rule.param_dict
    emitted = spec.fn(provider, rule.input_dict, params)
    emit_ranges = {emit.key: emit for emit in spec.resolve_emits(params)}

    names: list[str] = []
    columns: list[np.ndarray] = []
    metas: list[ChannelMeta] = []

    for emit_key in rule.emits:
        if emit_key not in emitted:
            raise RuleContractError(
                f"Operator {spec.name!r} declared emit {emit_key!r} but did not return it."
            )
        definition = emit_ranges[emit_key]
        base = np.asarray(emitted[emit_key], dtype=np.float64)

        variants: list[tuple[int | None, np.ndarray]] = [(None, base)]
        for half_life in rule.half_lives:
            variants.append(
                (half_life, ewma_response(base, half_life, elapsed_days))
            )

        for half_life, values in variants:
            name = channel_name(
                rule.school, rule.family, rule.rule_id, emit_key, half_life
            )
            _enforce_range(values, definition.low, definition.high, name)
            names.append(name)
            columns.append(values)
            metas.append(
                ChannelMeta(
                    name=name,
                    rule_id=rule.rule_id,
                    family=rule.family,
                    school=rule.school,
                    operator=rule.operator,
                    emit_key=emit_key,
                    half_life_days=half_life,
                    low=definition.low,
                    high=definition.high,
                    prior_importance=rule.prior.importance,
                    l2_anchor=rule.prior.l2_anchor,
                    l1_weight=rule.prior.l1_weight,
                    arm=arm,
                )
            )

    return names, columns, metas


def compile_ruleset(
    ruleset: RuleSet,
    provider: EphemerisProvider,
    arm: str = "real",
) -> CompiledFeatures:
    """Compile every enabled rule against ``provider``'s full ephemeris grid."""

    _check_ephemeris_requirements(ruleset, provider)

    index = provider.index
    elapsed = elapsed_days_from_index(index)
    # A uniform daily grid takes the vectorized IIR path; anything else falls
    # back to the explicit variable-gap recursion.
    elapsed_argument = None if is_uniform_daily(elapsed) else elapsed

    names: list[str] = []
    columns: list[np.ndarray] = []
    metas: list[ChannelMeta] = []

    for rule in ruleset.enabled_rules:
        rule_names, rule_columns, rule_metas = _compile_rule(
            rule, provider, elapsed_argument, arm
        )
        names.extend(rule_names)
        columns.extend(rule_columns)
        metas.extend(rule_metas)

    if not names:
        raise RuleContractError(
            f"Ruleset {ruleset.ruleset_id!r} compiled to zero channels; every rule is disabled."
        )
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise RuleContractError(f"Compiled duplicate channel names: {duplicates}.")

    matrix = np.stack(columns, axis=1) if columns else np.empty((len(index), 0))

    manifest = {
        "schema_version": 1,
        "arm": arm,
        "ruleset_id": ruleset.ruleset_id,
        "ruleset_hash": ruleset.ruleset_hash,
        "convention_hash": provider.manifest.convention_hash,
        "channel_name_hash": channel_name_hash(names),
        "channel_count": len(names),
        "families": sorted({meta.family for meta in metas}),
        "operator_versions": operator_versions(),
        "coverage_start": index[0].isoformat(),
        "coverage_end": index[-1].isoformat(),
    }

    return CompiledFeatures(
        names=tuple(names),
        matrix=matrix,
        channel_meta=tuple(metas),
        index=index,
        manifest=manifest,
    )
