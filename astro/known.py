"""Bridge between compiled rules and the TFT known-future input path.

The native model is constructed from ``args`` before any dataset is loaded, so
the known-channel layout must be resolved first.  :func:`prepare_astro_known`
compiles the requested arm once, caches it, and writes the layout onto
``args``; the dataset then samples the cached block by market date.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from astro.compile.compiler import CompiledFeatures, compile_ruleset
from astro.compile.nulls import compile_null_arm
from astro.ephemeris.contract import EphemerisManifest
from astro.ephemeris.provider import EphemerisProvider
from astro.rules.registry import RuleContractError
from astro.rules.schema import RuleSet, load_ruleset
from utils.reproducibility import stable_json_hash
from utils.tft_schema import resolve_known_feature_names

REAL_ARM = "real"
ZERO_ARM = "zero"

_COMPILED_CACHE: dict[tuple, tuple[RuleSet, CompiledFeatures]] = {}


def load_ephemeris(data_path: str, manifest_path: str) -> EphemerisProvider:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = EphemerisManifest.from_dict(json.load(handle))
    extension = os.path.splitext(data_path)[1].lower()
    if extension == ".parquet":
        frame = pd.read_parquet(data_path)
    elif extension == ".csv":
        frame = pd.read_csv(data_path)
    else:
        raise RuleContractError(
            f"Ephemeris file must be .csv or .parquet, got {data_path!r}."
        )
    return EphemerisProvider(frame, manifest)


def _cache_key(args) -> tuple:
    return (
        os.path.abspath(str(args.astro_ruleset)),
        os.path.abspath(str(args.astro_ephemeris_path)),
        os.path.abspath(str(args.astro_ephemeris_manifest)),
        str(getattr(args, "astro_arm", REAL_ARM)),
    )


def compile_arm(args) -> tuple[RuleSet, CompiledFeatures]:
    """Compile (or fetch from cache) the arm requested by ``args.astro_arm``."""

    key = _cache_key(args)
    if key in _COMPILED_CACHE:
        return _COMPILED_CACHE[key]

    ruleset = load_ruleset(args.astro_ruleset)
    provider = load_ephemeris(args.astro_ephemeris_path, args.astro_ephemeris_manifest)
    real = compile_ruleset(ruleset, provider)
    arm = key[-1]

    if arm == REAL_ARM:
        features = real
    elif arm == ZERO_ARM or arm.startswith(ZERO_ARM + ":"):
        # Capacity-matched knockout: identical layout, so initialization and data
        # order stay paired with the real arm. "zero" removes every family;
        # "zero:<family>" removes one for leave-one-family-out importance.
        matrix = real.matrix.copy()
        if arm == ZERO_ARM:
            matrix[:] = 0.0
        else:
            family = arm.split(":", 1)[1]
            columns = real.family_channel_indices().get(family)
            if columns is None:
                raise RuleContractError(
                    f"astro_arm {arm!r} names unknown family {family!r}; families are "
                    f"{list(real.families())}."
                )
            matrix[:, list(columns)] = 0.0
        features = CompiledFeatures(
            names=real.names,
            matrix=matrix,
            channel_meta=real.channel_meta,
            index=real.index,
            manifest={**real.manifest, "arm": arm},
        )
    else:
        matches = [null for null in ruleset.null_arms if null.arm_name == arm]
        if not matches:
            available = [REAL_ARM, ZERO_ARM] + [n.arm_name for n in ruleset.null_arms]
            raise RuleContractError(f"Unknown astro_arm {arm!r}; available arms: {available}.")
        features = compile_null_arm(ruleset, provider, matches[0], real=real)

    _COMPILED_CACHE[key] = (ruleset, features)
    return ruleset, features


def calendar_feature_names(args) -> tuple[str, ...]:
    if getattr(args, "embed", None) != "timeF":
        raise RuleContractError(
            "planetary_market requires --embed timeF so calendar mark names are well defined."
        )
    return resolve_known_feature_names("timeF", str(args.freq), allow_custom_known=False)


def prepare_astro_known(args) -> dict:
    """Resolve the known-channel layout and write it onto ``args``.

    Idempotent: repeated calls with the same configuration are no-ops, and a
    conflicting user-supplied known layout is rejected rather than overwritten.
    """

    ruleset, features = compile_arm(args)
    names = list(calendar_feature_names(args)) + list(features.names)

    max_channels = int(getattr(args, "tft_known_max_channels", 512))
    if len(names) > max_channels:
        raise RuleContractError(
            f"Ruleset compiles to {len(names)} known channels, above "
            f"tft_known_max_channels={max_channels}. Disable rules or raise the cap."
        )

    existing = getattr(args, "tft_known_feature_names", None)
    if existing and list(existing) != names:
        raise RuleContractError(
            "tft_known_feature_names was set explicitly and disagrees with the compiled "
            "astro layout; let planetary_market derive it."
        )

    args.tft_allow_custom_known = True
    args.tft_known_len = len(names)
    args.tft_known_feature_names = names

    manifest = {
        **features.manifest,
        "calendar_channels": list(calendar_feature_names(args)),
        "fold": str(getattr(args, "astro_fold", "")),
    }
    args.astro_manifest_hash = stable_json_hash(manifest)[:12]
    return {
        "manifest": manifest,
        "channel_meta": features.channel_meta,
        "calendar_channel_count": len(names) - features.channel_count,
    }


def build_known_block(args, market_dates: pd.Series) -> np.ndarray:
    """Rule channels for each market date, sampled from the daily ephemeris grid.

    Each session maps to the single ephemeris row on the same UTC calendar day.
    """

    _, features = compile_arm(args)
    session_days = pd.DatetimeIndex(pd.to_datetime(market_dates, errors="raise")).normalize()
    if session_days.tz is None:
        session_days = session_days.tz_localize("UTC")
    ephemeris_days = features.index.normalize()
    if ephemeris_days.has_duplicates:
        raise RuleContractError(
            "Ephemeris has more than one row per UTC day; session lookup is ambiguous."
        )
    rows = ephemeris_days.get_indexer(session_days)
    if np.any(rows < 0):
        first_missing = session_days[int(np.flatnonzero(rows < 0)[0])]
        raise RuleContractError(
            f"Ephemeris does not cover market session {first_missing.date()}; coverage is "
            f"{features.index[0].date()}..{features.index[-1].date()}."
        )
    return features.select_rows(rows)


def clear_cache() -> None:
    _COMPILED_CACHE.clear()
