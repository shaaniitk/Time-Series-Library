"""Synthetic market + ephemeris fixtures with a planted astrological effect.

Used for the known-answer test: if the pipeline cannot recover an effect that
was deliberately planted, a null result on real data would be uninformative.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from astro.ephemeris.provider import EphemerisProvider, build_synthetic_ephemeris
from astro.rules.registry import get_operator
from astro.rules import operators as _operators  # noqa: F401

EPHEMERIS_START = "1995-01-01"
MARKET_START = "1996-11-05"
MARKET_END = "2020-12-31"


def write_synthetic_study(
    directory: str,
    effect_size: float = 0.8,
    noise_scale: float = 1.0,
    seed: int = 0,
    market_end: str = MARKET_END,
) -> dict:
    """Write ephemeris, manifest and market CSVs; return their paths.

    The planted effect: next-session close return shifts by ``effect_size``
    standard deviations while Mercury is retrograde.  ``effect_size=0`` gives a
    pure-noise market for null checks.
    """

    os.makedirs(directory, exist_ok=True)
    periods = (pd.Timestamp(market_end) - pd.Timestamp(EPHEMERIS_START)).days + 400
    frame, manifest = build_synthetic_ephemeris(
        start=EPHEMERIS_START, periods=periods, seed=seed
    )
    provider = EphemerisProvider(frame, manifest)

    ephemeris_path = os.path.join(directory, "ephemeris.csv")
    manifest_path = os.path.join(directory, "ephemeris_manifest.json")
    frame.to_csv(ephemeris_path, index=False)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2)

    sessions = pd.bdate_range(MARKET_START, market_end)
    retro = get_operator("retrograde_state").fn(provider, {"body": "Mercury"}, {})[
        "is_retrograde"
    ]
    day_index = provider.index.normalize().tz_localize(None)
    rows = day_index.get_indexer(sessions)
    signal = retro[rows] - retro[rows].mean()

    rng = np.random.default_rng(seed + 1)
    close = effect_size * signal + noise_scale * rng.standard_normal(len(sessions))
    market = pd.DataFrame(
        {
            "date": sessions.strftime("%Y-%m-%d"),
            "log_Open": 0.5 * close + 0.5 * rng.standard_normal(len(sessions)),
            "log_High": rng.standard_normal(len(sessions)),
            "log_Low": rng.standard_normal(len(sessions)),
            "log_Close": close,
        }
    )
    market_path = os.path.join(directory, "market.csv")
    market.to_csv(market_path, index=False)

    return {
        "root_path": directory,
        "data_path": "market.csv",
        "ephemeris_path": ephemeris_path,
        "manifest_path": manifest_path,
        "session_count": len(sessions),
    }
