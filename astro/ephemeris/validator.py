"""Hard validation of a supplied ephemeris table against its manifest.

Every check here defends a downstream invariant.  Failures name the offending
column and row so a producer can fix the generator rather than guess.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from astro.ephemeris.contract import (
    EphemerisContractError,
    EphemerisManifest,
    NODE_BODIES,
    column_name,
    required_columns,
    required_fields,
)

TIMESTAMP_COLUMN = "timestamp"
AYANAMSHA_COLUMN = "ayanamsha"

# Maximum apparent geocentric speed, degrees/day, with generous headroom.
# The Moon is the fastest body at roughly 15 deg/day.
_MAX_SPEED_DEG_PER_DAY = 20.0

_NODE_ANTIPODE_TOLERANCE_DEG = 1e-6
_UNIT_CIRCLE_TOLERANCE = 1e-9


def _first_bad_row(mask: np.ndarray) -> int:
    return int(np.flatnonzero(mask)[0])


def _check_timestamps(frame: pd.DataFrame) -> pd.DatetimeIndex:
    if TIMESTAMP_COLUMN not in frame.columns:
        raise EphemerisContractError(
            f"Ephemeris table must contain a {TIMESTAMP_COLUMN!r} column."
        )
    timestamps = pd.to_datetime(frame[TIMESTAMP_COLUMN], errors="coerce", utc=True)
    if timestamps.isna().any():
        raise EphemerisContractError(
            f"{TIMESTAMP_COLUMN} contains unparseable values at row "
            f"{_first_bad_row(timestamps.isna().to_numpy())}."
        )
    index = pd.DatetimeIndex(timestamps)
    deltas = np.diff(index.asi8)
    if deltas.size and np.any(deltas <= 0):
        raise EphemerisContractError(
            f"{TIMESTAMP_COLUMN} must be strictly increasing; violation at row "
            f"{_first_bad_row(deltas <= 0) + 1}."
        )
    return index


def _check_finite(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        values = frame[column].to_numpy(dtype=np.float64)
        bad = ~np.isfinite(values)
        if bad.any():
            raise EphemerisContractError(
                f"Column {column!r} has a non-finite value at row {_first_bad_row(bad)}."
            )


def _check_ranges(frame: pd.DataFrame, manifest: EphemerisManifest) -> None:
    for body in manifest.bodies:
        fields = required_fields(body)

        longitude = frame[column_name(body, "lon")].to_numpy(dtype=np.float64)
        out_of_range = (longitude < 0.0) | (longitude >= 360.0)
        if out_of_range.any():
            row = _first_bad_row(out_of_range)
            raise EphemerisContractError(
                f"{column_name(body, 'lon')} must lie in [0, 360); got "
                f"{longitude[row]} at row {row}."
            )

        speed = frame[column_name(body, "speed")].to_numpy(dtype=np.float64)
        too_fast = np.abs(speed) > _MAX_SPEED_DEG_PER_DAY
        if too_fast.any():
            row = _first_bad_row(too_fast)
            raise EphemerisContractError(
                f"{column_name(body, 'speed')} exceeds {_MAX_SPEED_DEG_PER_DAY} deg/day; "
                f"got {speed[row]} at row {row}. Check the declared speed_unit."
            )

        if "lat" in fields:
            latitude = frame[column_name(body, "lat")].to_numpy(dtype=np.float64)
            bad = np.abs(latitude) > 90.0
            if bad.any():
                row = _first_bad_row(bad)
                raise EphemerisContractError(
                    f"{column_name(body, 'lat')} must lie in [-90, 90]; got "
                    f"{latitude[row]} at row {row}."
                )

        if "decl" in fields:
            declination = frame[column_name(body, "decl")].to_numpy(dtype=np.float64)
            bad = np.abs(declination) > 90.0
            if bad.any():
                row = _first_bad_row(bad)
                raise EphemerisContractError(
                    f"{column_name(body, 'decl')} must lie in [-90, 90]; got "
                    f"{declination[row]} at row {row}."
                )

        if "dist" in fields:
            distance = frame[column_name(body, "dist")].to_numpy(dtype=np.float64)
            bad = distance <= 0.0
            if bad.any():
                row = _first_bad_row(bad)
                raise EphemerisContractError(
                    f"{column_name(body, 'dist')} must be positive; got "
                    f"{distance[row]} at row {row}."
                )


def _check_node_geometry(frame: pd.DataFrame, manifest: EphemerisManifest) -> None:
    if not {"Rahu", "Ketu"}.issubset(set(manifest.bodies)):
        return

    rahu = frame[column_name("Rahu", "lon")].to_numpy(dtype=np.float64)
    ketu = frame[column_name("Ketu", "lon")].to_numpy(dtype=np.float64)
    separation = np.abs(((ketu - rahu - 180.0 + 180.0) % 360.0) - 180.0)
    bad = separation > _NODE_ANTIPODE_TOLERANCE_DEG
    if bad.any():
        row = _first_bad_row(bad)
        raise EphemerisContractError(
            f"Ketu must be exactly opposite Rahu; separation error {separation[row]} deg "
            f"at row {row}."
        )

    rahu_speed = frame[column_name("Rahu", "speed")].to_numpy(dtype=np.float64)
    if manifest.node_policy == "mean":
        # The mean node regresses uniformly; any prograde sample means the
        # producer mislabelled a true node as mean.
        bad_speed = rahu_speed >= 0.0
        if bad_speed.any():
            row = _first_bad_row(bad_speed)
            raise EphemerisContractError(
                f"node_policy='mean' requires a strictly retrograde Rahu speed; got "
                f"{rahu_speed[row]} at row {row}."
            )


def _check_longitude_continuity(
    frame: pd.DataFrame, manifest: EphemerisManifest, index: pd.DatetimeIndex
) -> None:
    """Reject jumps larger than the body could physically travel.

    Guards against row shuffling, duplicated blocks, and silent wrap errors.
    """

    elapsed_days = np.diff(index.asi8) / (24.0 * 60.0 * 60.0 * 1e9)
    if elapsed_days.size == 0:
        return

    for body in manifest.bodies:
        longitude = frame[column_name(body, "lon")].to_numpy(dtype=np.float64)
        stepped = np.abs(((np.diff(longitude) + 180.0) % 360.0) - 180.0)
        budget = _MAX_SPEED_DEG_PER_DAY * elapsed_days
        bad = stepped > budget
        if bad.any():
            row = _first_bad_row(bad)
            raise EphemerisContractError(
                f"{column_name(body, 'lon')} moves {stepped[row]:.4f} deg across "
                f"{elapsed_days[row]:.4f} days between rows {row} and {row + 1}, which "
                f"exceeds the {_MAX_SPEED_DEG_PER_DAY} deg/day budget. The table may be "
                "unsorted or contain duplicated blocks."
            )


def _check_ayanamsha(frame: pd.DataFrame, manifest: EphemerisManifest) -> None:
    present = AYANAMSHA_COLUMN in frame.columns
    if present != manifest.has_ayanamsha_column:
        raise EphemerisContractError(
            f"Manifest declares has_ayanamsha_column={manifest.has_ayanamsha_column} but "
            f"the table {'has' if present else 'does not have'} an {AYANAMSHA_COLUMN!r} column."
        )
    if not present:
        return
    values = frame[AYANAMSHA_COLUMN].to_numpy(dtype=np.float64)
    bad = ~np.isfinite(values) | (values < 0.0) | (values >= 360.0)
    if bad.any():
        row = _first_bad_row(bad)
        raise EphemerisContractError(
            f"{AYANAMSHA_COLUMN} must lie in [0, 360); got {values[row]} at row {row}."
        )


def validate_ephemeris(
    frame: pd.DataFrame, manifest: EphemerisManifest
) -> pd.DatetimeIndex:
    """Validate ``frame`` against ``manifest`` and return its UTC index.

    Raises :class:`EphemerisContractError` on the first violation found.
    """

    expected = required_columns(manifest.bodies)
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise EphemerisContractError(f"Ephemeris table is missing columns: {missing}.")

    if len(frame) != manifest.row_count:
        raise EphemerisContractError(
            f"Manifest declares row_count={manifest.row_count} but the table has {len(frame)} rows."
        )

    index = _check_timestamps(frame)
    _check_finite(frame, expected)
    _check_ranges(frame, manifest)
    _check_node_geometry(frame, manifest)
    _check_longitude_continuity(frame, manifest, index)
    _check_ayanamsha(frame, manifest)

    declared_start = pd.Timestamp(manifest.coverage_start, tz="UTC")
    declared_end = pd.Timestamp(manifest.coverage_end, tz="UTC")
    if index[0] != declared_start or index[-1] != declared_end:
        raise EphemerisContractError(
            f"Manifest coverage {manifest.coverage_start}..{manifest.coverage_end} does not "
            f"match table range {index[0].isoformat()}..{index[-1].isoformat()}."
        )

    return index


def assert_unit_circle(sin_values: np.ndarray, cos_values: np.ndarray, label: str) -> None:
    """Assert a sin/cos pair lies on the unit circle.

    Used by the compiler as a tripwire: marks are never scaled, so any drift
    here means a transform corrupted the circular encoding.
    """

    residual = np.abs(sin_values**2 + cos_values**2 - 1.0)
    worst = float(residual.max()) if residual.size else 0.0
    if worst > _UNIT_CIRCLE_TOLERANCE:
        raise EphemerisContractError(
            f"{label} violates sin^2+cos^2=1 by {worst:.3e}."
        )
