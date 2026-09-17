import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from astro.ephemeris.contract import EphemerisContractError, EphemerisManifest
from astro.ephemeris.provider import (
    EphemerisProvider,
    build_synthetic_ephemeris,
    synthetic_provider,
)
from astro.ephemeris.validator import AYANAMSHA_COLUMN


@pytest.fixture(scope="module")
def synthetic():
    return build_synthetic_ephemeris(periods=800)


def _provider(frame, manifest):
    return EphemerisProvider(frame, manifest)


def test_synthetic_ephemeris_passes_validation(synthetic):
    frame, manifest = synthetic
    provider = _provider(frame.copy(), manifest)
    assert len(provider.index) == 800
    assert set(provider.bodies) >= {"Sun", "Moon", "Rahu", "Ketu"}


def test_synthetic_retrograde_is_realistic():
    provider = synthetic_provider(periods=4000)
    assert 0.15 < float((provider.speed("Mercury") < 0).mean()) < 0.23
    assert float((provider.speed("Sun") < 0).mean()) == 0.0
    assert bool((provider.speed("Rahu") < 0).all())


def test_missing_column_rejected(synthetic):
    frame, manifest = synthetic
    with pytest.raises(EphemerisContractError, match="missing columns"):
        _provider(frame.drop(columns=["Mars_decl"]), manifest)


def test_non_finite_value_rejected(synthetic):
    frame, manifest = synthetic
    broken = frame.copy()
    broken.loc[10, "Venus_speed"] = np.nan
    with pytest.raises(EphemerisContractError, match="non-finite"):
        _provider(broken, manifest)


def test_non_monotonic_timestamps_rejected(synthetic):
    frame, manifest = synthetic
    broken = frame.copy()
    broken.loc[5, "timestamp"] = broken.loc[4, "timestamp"]
    with pytest.raises(EphemerisContractError, match="strictly increasing"):
        _provider(broken, manifest)


def test_longitude_out_of_range_rejected(synthetic):
    frame, manifest = synthetic
    broken = frame.copy()
    broken.loc[3, "Sun_lon"] = 360.0
    with pytest.raises(EphemerisContractError, match=r"\[0, 360\)"):
        _provider(broken, manifest)


def test_ketu_must_oppose_rahu(synthetic):
    frame, manifest = synthetic
    broken = frame.copy()
    broken["Ketu_lon"] = np.mod(broken["Ketu_lon"] + 0.01, 360.0)
    with pytest.raises(EphemerisContractError, match="opposite Rahu"):
        _provider(broken, manifest)


def test_mean_node_must_be_retrograde(synthetic):
    frame, manifest = synthetic
    broken = frame.copy()
    broken.loc[7, "Rahu_speed"] = 0.01
    broken.loc[7, "Ketu_speed"] = 0.01
    with pytest.raises(EphemerisContractError, match="retrograde"):
        _provider(broken, manifest)


def test_discontinuous_longitude_rejected(synthetic):
    frame, manifest = synthetic
    broken = frame.copy()
    broken.loc[100:, "Saturn_lon"] = np.mod(broken.loc[100:, "Saturn_lon"] + 90.0, 360.0)
    with pytest.raises(EphemerisContractError, match="budget"):
        _provider(broken, manifest)


def test_ayanamsha_column_must_match_manifest(synthetic):
    frame, manifest = synthetic
    with pytest.raises(EphemerisContractError, match="has_ayanamsha_column"):
        _provider(frame.drop(columns=[AYANAMSHA_COLUMN]), manifest)


def test_row_count_and_coverage_must_match(synthetic):
    frame, manifest = synthetic
    with pytest.raises(EphemerisContractError, match="row_count"):
        _provider(frame.iloc[:-1], manifest)


def test_unavailable_frame_rejected(synthetic):
    frame, manifest = synthetic
    payload = manifest.to_dict()
    payload["has_ayanamsha_column"] = False
    tropical_only = EphemerisManifest.from_dict(payload)
    provider = _provider(frame.drop(columns=[AYANAMSHA_COLUMN]), tropical_only)
    provider.longitude("Sun", "tropical")
    with pytest.raises(EphemerisContractError, match="unavailable"):
        provider.longitude("Sun", "sidereal")


def test_sidereal_is_tropical_minus_ayanamsha(synthetic):
    frame, manifest = synthetic
    provider = _provider(frame.copy(), manifest)
    expected = np.mod(provider.longitude("Moon", "tropical") - provider.ayanamsha(), 360.0)
    np.testing.assert_allclose(provider.longitude("Moon", "sidereal"), expected)


def test_manifest_rejects_unknown_keys_and_missing_commit(synthetic):
    _, manifest = synthetic
    payload = manifest.to_dict()
    with pytest.raises(EphemerisContractError, match="Unknown ephemeris manifest keys"):
        EphemerisManifest.from_dict({**payload, "extra": 1})
    with pytest.raises(EphemerisContractError, match="source_commit"):
        EphemerisManifest.from_dict({**payload, "source_commit": ""})


def test_convention_hash_tracks_convention(synthetic):
    _, manifest = synthetic
    payload = manifest.to_dict()
    changed = EphemerisManifest.from_dict({**payload, "ayanamsha": "raman"})
    assert changed.convention_hash != manifest.convention_hash
    assert EphemerisManifest.from_dict(payload).convention_hash == manifest.convention_hash


def test_provider_exposes_no_market_surface(synthetic):
    frame, manifest = synthetic
    provider = _provider(frame.copy(), manifest)
    public = {name for name in dir(provider) if not name.startswith("_")}
    assert not any("market" in name or "price" in name or "return" in name for name in public)
