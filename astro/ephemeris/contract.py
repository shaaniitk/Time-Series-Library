"""Ingest contract for externally supplied ephemeris tables.

The ephemeris is produced by a separate repository.  This module defines the
manifest that producer must ship alongside the data, so that every downstream
feature can be traced to a fully specified astronomical convention.
"""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass
from typing import Literal

from utils.reproducibility import stable_json_hash

CONTRACT_SCHEMA_VERSION = 1

PLANET_BODIES = (
    "Sun",
    "Moon",
    "Mercury",
    "Venus",
    "Mars",
    "Jupiter",
    "Saturn",
    "Uranus",
    "Neptune",
    "Pluto",
)

NODE_BODIES = ("Rahu", "Ketu")

CANONICAL_BODIES = PLANET_BODIES + NODE_BODIES

# Nodes are defined points on the ecliptic: latitude is identically zero and
# distance is undefined, so they carry a reduced field set.
PLANET_FIELDS = ("lon", "lat", "speed", "dist", "decl")
NODE_FIELDS = ("lon", "speed")

VALID_FRAMES = ("tropical", "sidereal")
VALID_NODE_POLICIES = ("mean", "true")
VALID_CENTERS = ("geocentric", "topocentric")


class EphemerisContractError(ValueError):
    """Raised when a supplied ephemeris violates the declared contract."""


def required_fields(body: str) -> tuple[str, ...]:
    if body in NODE_BODIES:
        return NODE_FIELDS
    return PLANET_FIELDS


def column_name(body: str, field: str) -> str:
    return f"{body}_{field}"


def required_columns(bodies: tuple[str, ...]) -> tuple[str, ...]:
    columns: list[str] = []
    for body in bodies:
        for field in required_fields(body):
            columns.append(column_name(body, field))
    return tuple(columns)


@dataclass(frozen=True)
class EphemerisManifest:
    """Fully specifies the astronomical convention of an ephemeris table.

    ``stored_frame`` is the frame the ``*_lon`` columns are expressed in.  When
    ``has_ayanamsha_column`` is true the table carries a per-row ``ayanamsha``
    column and both frames are derivable; otherwise only ``stored_frame`` is
    available and requesting the other frame is an error.
    """

    schema_version: int
    source_repo: str
    source_commit: str
    stored_frame: Literal["tropical", "sidereal"]
    ayanamsha: str
    has_ayanamsha_column: bool
    node_policy: Literal["mean", "true"]
    center: Literal["geocentric", "topocentric"]
    bodies: tuple[str, ...]
    coverage_start: str
    coverage_end: str
    row_count: int
    topocentric_lat_deg: float | None = None
    topocentric_lon_deg: float | None = None
    topocentric_alt_m: float | None = None
    time_scale: str = "UTC"
    angle_unit: str = "degrees"
    speed_unit: str = "degrees_per_day"
    distance_unit: str = "au"

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_SCHEMA_VERSION:
            raise EphemerisContractError(
                f"Unsupported ephemeris schema_version {self.schema_version}; "
                f"expected {CONTRACT_SCHEMA_VERSION}."
            )
        if self.stored_frame not in VALID_FRAMES:
            raise EphemerisContractError(
                f"stored_frame must be one of {VALID_FRAMES}, got {self.stored_frame!r}."
            )
        if self.node_policy not in VALID_NODE_POLICIES:
            raise EphemerisContractError(
                f"node_policy must be one of {VALID_NODE_POLICIES}, got {self.node_policy!r}."
            )
        if self.center not in VALID_CENTERS:
            raise EphemerisContractError(
                f"center must be one of {VALID_CENTERS}, got {self.center!r}."
            )
        if self.time_scale != "UTC":
            raise EphemerisContractError(
                f"time_scale must be UTC, got {self.time_scale!r}."
            )
        if self.angle_unit != "degrees":
            raise EphemerisContractError(
                f"angle_unit must be degrees, got {self.angle_unit!r}."
            )
        if self.speed_unit != "degrees_per_day":
            raise EphemerisContractError(
                f"speed_unit must be degrees_per_day, got {self.speed_unit!r}."
            )
        if not self.bodies:
            raise EphemerisContractError("bodies must not be empty.")
        unknown = [body for body in self.bodies if body not in CANONICAL_BODIES]
        if unknown:
            raise EphemerisContractError(
                f"Unknown bodies {unknown}; canonical names are {list(CANONICAL_BODIES)}."
            )
        if len(set(self.bodies)) != len(self.bodies):
            raise EphemerisContractError("bodies must be unique.")
        if self.center == "topocentric" and (
            self.topocentric_lat_deg is None or self.topocentric_lon_deg is None
        ):
            raise EphemerisContractError(
                "topocentric center requires topocentric_lat_deg and topocentric_lon_deg."
            )
        if self.row_count <= 0:
            raise EphemerisContractError("row_count must be positive.")
        if not self.source_commit:
            raise EphemerisContractError(
                "source_commit is required so features are traceable to a generator revision."
            )

    @property
    def available_frames(self) -> tuple[str, ...]:
        if self.has_ayanamsha_column:
            return VALID_FRAMES
        return (self.stored_frame,)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["bodies"] = list(self.bodies)
        return payload

    @property
    def convention_hash(self) -> str:
        return stable_json_hash(self.to_dict())

    @classmethod
    def from_dict(cls, payload: dict) -> "EphemerisManifest":
        known = {field for field in cls.__dataclass_fields__}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise EphemerisContractError(
                f"Unknown ephemeris manifest keys: {unknown}."
            )
        missing = sorted(
            field
            for field, spec in cls.__dataclass_fields__.items()
            if field not in payload and spec.default is MISSING
        )
        if missing:
            raise EphemerisContractError(f"Missing ephemeris manifest keys: {missing}.")
        data = dict(payload)
        if "bodies" in data:
            data["bodies"] = tuple(data["bodies"])
        return cls(**data)
