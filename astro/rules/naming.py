"""Deterministic channel naming.

Confirmatory runs reject anonymous ``f0/f1`` features, so every compiled channel
carries a name that identifies the school, family, rule and emit that produced
it.  Names are stable: adding a rule appends channels without renaming existing
ones.
"""

from __future__ import annotations

import re
from typing import Iterable

from utils.reproducibility import stable_json_hash

CHANNEL_PREFIX = "astro"
SLUG_PATTERN = re.compile(r"^[a-z0-9_]+$")
EMIT_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


def validate_slug(value: str, label: str) -> str:
    if not isinstance(value, str) or not SLUG_PATTERN.match(value):
        raise ValueError(
            f"{label} must match {SLUG_PATTERN.pattern} (lowercase, digits, underscore); "
            f"got {value!r}."
        )
    return value


def validate_emit_key(value: str) -> str:
    if not isinstance(value, str) or not EMIT_PATTERN.match(value):
        raise ValueError(
            f"emit key must match {EMIT_PATTERN.pattern}; got {value!r}."
        )
    return value


def channel_name(
    school: str,
    family: str,
    rule_id: str,
    emit_key: str,
    half_life_days: int | None = None,
) -> str:
    """Build the canonical channel name for one compiled feature."""

    parts = [CHANNEL_PREFIX, school, family, rule_id, emit_key]
    if half_life_days is not None:
        parts.append(f"hl{int(half_life_days)}")
    return ".".join(parts)


def channel_name_hash(names: Iterable[str]) -> str:
    """Hash the ordered channel-name list.

    Folded into the run digest so a checkpoint cannot silently be reused with a
    different feature layout.
    """

    return stable_json_hash(list(names))
