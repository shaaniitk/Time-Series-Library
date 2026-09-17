"""Read access to a validated ephemeris.

This is the only ephemeris surface the rule compiler sees.  It deliberately
exposes no market accessor, which makes market leakage into a "known future"
channel structurally impossible rather than merely discouraged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from astro.ephemeris.contract import (
    CANONICAL_BODIES,
    EphemerisContractError,
    EphemerisManifest,
    NODE_BODIES,
    column_name,
    required_fields,
)
from astro.ephemeris.validator import AYANAMSHA_COLUMN, validate_ephemeris

OBLIQUITY_DEG = 23.4392911


class EphemerisProvider:
    """A validated ephemeris table with frame-aware accessors."""

    def __init__(self, frame: pd.DataFrame, manifest: EphemerisManifest):
        self._index = validate_ephemeris(frame, manifest)
        self._frame = frame.reset_index(drop=True)
        self._manifest = manifest
        self._row_of_timestamp = pd.Series(
            np.arange(len(self._index)), index=self._index
        )

    @property
    def manifest(self) -> EphemerisManifest:
        return self._manifest

    @property
    def index(self) -> pd.DatetimeIndex:
        return self._index

    @property
    def bodies(self) -> tuple[str, ...]:
        return self._manifest.bodies

    def has_body(self, body: str) -> bool:
        return body in self._manifest.bodies

    def _require_body(self, body: str) -> None:
        if body not in CANONICAL_BODIES:
            raise EphemerisContractError(
                f"Unknown body {body!r}; canonical names are {list(CANONICAL_BODIES)}."
            )
        if body not in self._manifest.bodies:
            raise EphemerisContractError(
                f"Body {body!r} is not present in this ephemeris. Available: "
                f"{list(self._manifest.bodies)}."
            )

    def _require_field(self, body: str, field: str) -> None:
        if field not in required_fields(body):
            raise EphemerisContractError(
                f"Field {field!r} is not defined for {body!r}; available fields are "
                f"{list(required_fields(body))}."
            )

    def _column(self, body: str, field: str) -> np.ndarray:
        self._require_body(body)
        self._require_field(body, field)
        return self._frame[column_name(body, field)].to_numpy(dtype=np.float64)

    def ayanamsha(self) -> np.ndarray:
        if not self._manifest.has_ayanamsha_column:
            raise EphemerisContractError(
                "This ephemeris carries no ayanamsha column, so frame conversion is "
                f"unavailable. Only {self._manifest.stored_frame!r} longitudes can be served."
            )
        return self._frame[AYANAMSHA_COLUMN].to_numpy(dtype=np.float64)

    def longitude(self, body: str, frame: str = "sidereal") -> np.ndarray:
        """Return ecliptic longitude in degrees, in the requested frame."""

        if frame not in self._manifest.available_frames:
            raise EphemerisContractError(
                f"Frame {frame!r} is unavailable. This ephemeris stores "
                f"{self._manifest.stored_frame!r} longitudes and "
                f"{'has' if self._manifest.has_ayanamsha_column else 'has no'} ayanamsha "
                f"column, so available frames are {list(self._manifest.available_frames)}."
            )
        stored = self._column(body, "lon")
        if frame == self._manifest.stored_frame:
            return stored
        ayanamsha = self.ayanamsha()
        if frame == "sidereal":
            return np.mod(stored - ayanamsha, 360.0)
        return np.mod(stored + ayanamsha, 360.0)

    def speed(self, body: str) -> np.ndarray:
        """Signed longitudinal speed in degrees/day. Negative means retrograde."""

        return self._column(body, "speed")

    def latitude(self, body: str) -> np.ndarray:
        if body in NODE_BODIES:
            return np.zeros(len(self._index), dtype=np.float64)
        return self._column(body, "lat")

    def distance(self, body: str) -> np.ndarray:
        return self._column(body, "dist")

    def declination(self, body: str) -> np.ndarray:
        return self._column(body, "decl")

    def rows_for(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """Map a timestamp index onto row positions, requiring exact coverage."""

        missing = timestamps.difference(self._index)
        if len(missing) > 0:
            raise EphemerisContractError(
                f"Ephemeris does not cover {len(missing)} requested timestamps; first "
                f"missing is {missing[0].isoformat()}. Coverage is "
                f"{self._index[0].isoformat()}..{self._index[-1].isoformat()}."
            )
        return self._row_of_timestamp.loc[timestamps].to_numpy(dtype=np.int64)


def ecliptic_to_declination(
    longitude_deg: np.ndarray, latitude_deg: np.ndarray
) -> np.ndarray:
    """Exact ecliptic-to-equatorial declination conversion, in degrees."""

    obliquity = np.radians(OBLIQUITY_DEG)
    longitude = np.radians(longitude_deg)
    latitude = np.radians(latitude_deg)
    sin_declination = np.sin(latitude) * np.cos(obliquity) + np.cos(
        latitude
    ) * np.sin(obliquity) * np.sin(longitude)
    return np.degrees(np.arcsin(np.clip(sin_declination, -1.0, 1.0)))


# (mean motion deg/day, synodic period days, retrograde fraction, max latitude deg).
# The epicycle amplitude is derived from the retrograde fraction below, so the
# generated motion reproduces realistic retrograde episodes.
_SYNTHETIC_BODIES = {
    "Sun": (0.985647, 365.25, 0.0, 0.0),
    "Moon": (13.176358, 27.55, 0.0, 5.14),
    "Mercury": (4.092339, 115.88, 0.19, 3.38),
    "Venus": (1.602136, 583.92, 0.073, 3.39),
    "Mars": (0.524039, 779.94, 0.093, 1.85),
    "Jupiter": (0.083091, 398.88, 0.30, 1.30),
    "Saturn": (0.033460, 378.09, 0.36, 2.49),
    "Uranus": (0.011725, 369.66, 0.41, 0.77),
    "Neptune": (0.005981, 367.49, 0.43, 1.77),
    "Pluto": (0.003964, 366.73, 0.44, 17.16),
}


def _epicycle_amplitude(
    mean_motion: float, synodic_days: float, retrograde_fraction: float
) -> float:
    """Amplitude giving a target retrograde fraction for lon = M*t + A*sin(w*t).

    Speed is ``M + A*w*cos(w*t)``, which is negative on the fraction of the
    cycle where ``cos(w*t) < -M/(A*w)``.  Inverting that relation gives
    ``A = M / (w * cos(pi * f))``.
    """

    if retrograde_fraction <= 0.0:
        return 0.0
    if not 0.0 < retrograde_fraction < 0.5:
        raise ValueError("retrograde_fraction must lie in (0, 0.5).")
    omega = 2.0 * np.pi / synodic_days
    return mean_motion / (omega * np.cos(np.pi * retrograde_fraction))

_SYNTHETIC_DISTANCE_AU = {
    "Sun": 1.0,
    "Moon": 0.00257,
    "Mercury": 0.72,
    "Venus": 0.95,
    "Mars": 1.20,
    "Jupiter": 5.20,
    "Saturn": 9.58,
    "Uranus": 19.2,
    "Neptune": 30.1,
    "Pluto": 39.5,
}

_MEAN_NODE_SPEED_DEG_PER_DAY = -0.052992


def build_synthetic_ephemeris(
    start: str = "1990-01-01",
    periods: int = 4000,
    bodies: tuple[str, ...] = CANONICAL_BODIES,
    seed: int = 0,
) -> tuple[pd.DataFrame, EphemerisManifest]:
    """Generate a physically plausible ephemeris for tests and development.

    Longitudes follow mean motion plus a synodic epicycle, and speeds are the
    exact analytic derivative, so retrograde episodes are real and internally
    consistent.  This unblocks the whole pipeline while the authoritative
    ephemeris is produced elsewhere.
    """

    index = pd.date_range(start=start, periods=periods, freq="D", tz="UTC")
    days = (index - index[0]).days.to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)

    columns: dict[str, np.ndarray] = {"timestamp": index}

    for body in bodies:
        if body in NODE_BODIES:
            continue
        mean_motion, synodic_days, retrograde_fraction, max_latitude = _SYNTHETIC_BODIES[
            body
        ]
        amplitude = _epicycle_amplitude(mean_motion, synodic_days, retrograde_fraction)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        omega = 2.0 * np.pi / synodic_days
        longitude = mean_motion * days + amplitude * np.sin(omega * days + phase)
        speed = mean_motion + amplitude * omega * np.cos(omega * days + phase)
        latitude = max_latitude * np.sin(
            2.0 * np.pi * days / (synodic_days * 1.13) + phase
        )
        wrapped = np.mod(longitude, 360.0)
        columns[column_name(body, "lon")] = wrapped
        columns[column_name(body, "speed")] = speed
        columns[column_name(body, "lat")] = latitude
        columns[column_name(body, "dist")] = np.full(
            periods, _SYNTHETIC_DISTANCE_AU[body], dtype=np.float64
        )
        columns[column_name(body, "decl")] = ecliptic_to_declination(wrapped, latitude)

    if "Rahu" in bodies:
        rahu = np.mod(120.0 + _MEAN_NODE_SPEED_DEG_PER_DAY * days, 360.0)
        columns[column_name("Rahu", "lon")] = rahu
        columns[column_name("Rahu", "speed")] = np.full(
            periods, _MEAN_NODE_SPEED_DEG_PER_DAY, dtype=np.float64
        )
    if "Ketu" in bodies:
        if "Rahu" not in bodies:
            raise ValueError("Ketu requires Rahu in the synthetic body list.")
        columns[column_name("Ketu", "lon")] = np.mod(rahu + 180.0, 360.0)
        columns[column_name("Ketu", "speed")] = np.full(
            periods, _MEAN_NODE_SPEED_DEG_PER_DAY, dtype=np.float64
        )

    # Lahiri precesses at roughly 50.29 arcsec/year from a J2000 value near 23.85.
    columns[AYANAMSHA_COLUMN] = 23.85 + (50.29 / 3600.0) * (days / 365.25)

    frame = pd.DataFrame(columns)
    manifest = EphemerisManifest(
        schema_version=1,
        source_repo="synthetic",
        source_commit=f"synthetic-seed-{seed}",
        stored_frame="tropical",
        ayanamsha="lahiri",
        has_ayanamsha_column=True,
        node_policy="mean",
        center="geocentric",
        bodies=tuple(bodies),
        coverage_start=index[0].isoformat(),
        coverage_end=index[-1].isoformat(),
        row_count=periods,
    )
    return frame, manifest


def synthetic_provider(**kwargs) -> EphemerisProvider:
    frame, manifest = build_synthetic_ephemeris(**kwargs)
    return EphemerisProvider(frame, manifest)
