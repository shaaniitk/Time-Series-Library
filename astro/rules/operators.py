"""The v1 operator vocabulary: Vedic and Western transforms of ephemeris.

Every operator is a pure function of the ephemeris alone.  Operators receive the
full daily ephemeris grid (not the market grid) so that event solving and
response kernels see real calendar time, including weekends.  The compiler
subsets to market sessions afterwards.

All outputs are bounded to their declared range because known-future channels
bypass the dataset scaler entirely.
"""

from __future__ import annotations

import numpy as np

from astro.ephemeris.provider import EphemerisProvider
from astro.rules.registry import EmitDef, RuleContractError, operator

RASHI_COUNT = 12
RASHI_WIDTH_DEG = 360.0 / RASHI_COUNT
NAKSHATRA_COUNT = 27
NAKSHATRA_WIDTH_DEG = 360.0 / NAKSHATRA_COUNT
PADA_PER_NAKSHATRA = 4

# Bounds time-to-event channels; beyond this the signal is not meaningful.
_EVENT_HORIZON_DAYS = 90.0
# Relative speeds below this are treated as stationary, where linear
# time-to-exactness estimates are invalid.
_STATIONARY_SPEED_DEG_PER_DAY = 1e-4


def wrap_signed(degrees: np.ndarray) -> np.ndarray:
    """Wrap degrees into (-180, 180]."""

    return ((np.asarray(degrees, dtype=np.float64) + 180.0) % 360.0) - 180.0


def directed_separation(
    longitude_from: np.ndarray, longitude_to: np.ndarray
) -> np.ndarray:
    """Signed separation from one body to another, in (-180, 180]."""

    return wrap_signed(longitude_to - longitude_from)


def _frame_of(inputs: dict, default: str) -> str:
    frame = inputs.get("frame", default)
    if frame not in ("sidereal", "tropical"):
        raise RuleContractError(
            f"frame must be 'sidereal' or 'tropical', got {frame!r}."
        )
    return frame


def _gaussian_activation(orb_deg: np.ndarray, sigma_deg: float) -> np.ndarray:
    if sigma_deg <= 0.0:
        raise RuleContractError("sigma_deg must be positive.")
    return np.exp(-0.5 * (orb_deg / sigma_deg) ** 2)


def _angle_tag(angle_deg: float) -> str:
    """Stable channel-name fragment for an aspect angle."""

    if float(angle_deg).is_integer():
        return f"a{int(angle_deg)}"
    return f"a{angle_deg}".replace(".", "p").replace("-", "m")


def _require_angle_list(params: dict, key: str = "aspect_deg") -> tuple[float, ...]:
    raw = params[key]
    if not isinstance(raw, (list, tuple)) or not raw:
        raise RuleContractError(f"{key} must be a non-empty list of angles in degrees.")
    angles = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuleContractError(f"{key} entries must be numbers, got {value!r}.")
        angle = float(value) % 360.0
        angles.append(angle)
    if len(set(angles)) != len(angles):
        raise RuleContractError(f"{key} contains duplicate angles after wrapping.")
    return tuple(angles)


def _bounded_time_to_exact(
    signed_error_deg: np.ndarray, relative_speed: np.ndarray
) -> np.ndarray:
    """Signed days to exactness, squashed to (-1, 1).

    Near a station the linear estimate diverges, so the stationary case is
    reported as zero rather than as a spuriously large lead time.
    """

    stationary = np.abs(relative_speed) < _STATIONARY_SPEED_DEG_PER_DAY
    safe_speed = np.where(stationary, 1.0, relative_speed)
    days = -signed_error_deg / safe_speed
    days = np.where(stationary, 0.0, days)
    return np.tanh(days / _EVENT_HORIZON_DAYS)


# --------------------------------------------------------------------------
# Relative geometry (school-neutral)
# --------------------------------------------------------------------------


@operator(
    name="wrapped_separation",
    version=1,
    inputs=("from_body", "to_body"),
    optional_inputs=("frame",),
    emits=("sep_sin", "sep_cos", "relative_speed"),
    ranges={"sep_sin": (-1.0, 1.0), "sep_cos": (-1.0, 1.0), "relative_speed": (-1.0, 1.0)},
    invariance="invariant",
)
def wrapped_separation(provider: EphemerisProvider, inputs: dict, params: dict) -> dict:
    """Circular encoding of the directed separation between two bodies."""

    frame = _frame_of(inputs, "sidereal")
    lon_from = provider.longitude(inputs["from_body"], frame)
    lon_to = provider.longitude(inputs["to_body"], frame)
    separation = np.radians(directed_separation(lon_from, lon_to))
    relative_speed = provider.speed(inputs["to_body"]) - provider.speed(
        inputs["from_body"]
    )
    return {
        "sep_sin": np.sin(separation),
        "sep_cos": np.cos(separation),
        "relative_speed": np.tanh(relative_speed),
    }


def _harmonic_emits(params: dict) -> tuple[EmitDef, ...]:
    harmonics = params["harmonics"]
    if not isinstance(harmonics, (list, tuple)) or not harmonics:
        raise RuleContractError("harmonics must be a non-empty list of integers.")
    emits: list[EmitDef] = []
    for order in harmonics:
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            raise RuleContractError(f"harmonics entries must be positive integers, got {order!r}.")
        emits.append(EmitDef(f"h{order}_sin", -1.0, 1.0))
        emits.append(EmitDef(f"h{order}_cos", -1.0, 1.0))
    return tuple(emits)


@operator(
    name="harmonic_phase",
    version=1,
    inputs=("from_body", "to_body"),
    optional_inputs=("frame",),
    params=("harmonics",),
    emits_resolver=_harmonic_emits,
    invariance="invariant",
)
def harmonic_phase(provider: EphemerisProvider, inputs: dict, params: dict) -> dict:
    """Harmonic decomposition of a pair separation, sin/cos of k*delta."""

    frame = _frame_of(inputs, "sidereal")
    lon_from = provider.longitude(inputs["from_body"], frame)
    lon_to = provider.longitude(inputs["to_body"], frame)
    separation = np.radians(directed_separation(lon_from, lon_to))
    output = {}
    for order in params["harmonics"]:
        output[f"h{order}_sin"] = np.sin(order * separation)
        output[f"h{order}_cos"] = np.cos(order * separation)
    return output


def _aspect_emits(params: dict) -> tuple[EmitDef, ...]:
    angles = _require_angle_list(params)
    emits: list[EmitDef] = []
    for angle in angles:
        tag = _angle_tag(angle)
        emits.append(EmitDef(f"act_{tag}", 0.0, 1.0))
        emits.append(EmitDef(f"applying_{tag}", -1.0, 1.0))
        emits.append(EmitDef(f"tte_{tag}", -1.0, 1.0))
    return tuple(emits)


def _aspect_channels(
    provider: EphemerisProvider, inputs: dict, params: dict, frame_default: str
) -> dict:
    frame = _frame_of(inputs, frame_default)
    body_a = inputs["from_body"]
    body_b = inputs["to_body"]
    lon_a = provider.longitude(body_a, frame)
    lon_b = provider.longitude(body_b, frame)
    separation = directed_separation(lon_a, lon_b)
    relative_speed = provider.speed(body_b) - provider.speed(body_a)
    sigma = float(params["sigma_deg"])

    output = {}
    for angle in _require_angle_list(params):
        tag = _angle_tag(angle)
        signed_error = wrap_signed(separation - angle)
        activation = _gaussian_activation(np.abs(signed_error), sigma)
        # Applying when the orb is shrinking; weighted by activation so distant
        # configurations do not emit a full-strength direction signal.
        approaching = -np.sign(signed_error * relative_speed)
        output[f"act_{tag}"] = activation
        output[f"applying_{tag}"] = approaching * activation
        output[f"tte_{tag}"] = _bounded_time_to_exact(signed_error, relative_speed)
    return output


@operator(
    name="aspect_activation",
    version=1,
    inputs=("from_body", "to_body"),
    optional_inputs=("frame",),
    params=("aspect_deg", "sigma_deg"),
    emits_resolver=_aspect_emits,
    invariance="invariant",
)
def aspect_activation(provider: EphemerisProvider, inputs: dict, params: dict) -> dict:
    """Western aspect activation with a Gaussian orb, defaulting to tropical."""

    return _aspect_channels(provider, inputs, params, "tropical")


@operator(
    name="degree_orb_drishti",
    version=1,
    inputs=("from_body", "to_body"),
    optional_inputs=("frame",),
    params=("aspect_deg", "sigma_deg"),
    emits_resolver=_aspect_emits,
    invariance="invariant",
)
def degree_orb_drishti(provider: EphemerisProvider, inputs: dict, params: dict) -> dict:
    """Vedic graha drishti on a degree-orb basis, defaulting to sidereal.

    Kept distinct from :func:`whole_sign_drishti` so the degree-orb and
    whole-sign schools are never silently averaged together.
    """

    return _aspect_channels(provider, inputs, params, "sidereal")


def _whole_sign_emits(params: dict) -> tuple[EmitDef, ...]:
    houses = params["house_offsets"]
    if not isinstance(houses, (list, tuple)) or not houses:
        raise RuleContractError("house_offsets must be a non-empty list of integers 1..12.")
    emits: list[EmitDef] = []
    for offset in houses:
        if isinstance(offset, bool) or not isinstance(offset, int) or not 1 <= offset <= 12:
            raise RuleContractError(
                f"house_offsets entries must be integers in 1..12, got {offset!r}."
            )
        emits.append(EmitDef(f"drishti_h{offset}", 0.0, 1.0))
    return tuple(emits)


@operator(
    name="whole_sign_drishti",
    version=1,
    inputs=("from_body", "to_body"),
    params=("house_offsets",),
    emits_resolver=_whole_sign_emits,
    invariance="covariant",
)
def whole_sign_drishti(provider: EphemerisProvider, inputs: dict, params: dict) -> dict:
    """Parashari whole-sign drishti: a discrete sign-count relationship.

    This is covariant under an ayanamsha shift because sign boundaries are
    absolute positions, unlike the degree-orb form.
    """

    lon_from = provider.longitude(inputs["from_body"], "sidereal")
    lon_to = provider.longitude(inputs["to_body"], "sidereal")
    sign_from = np.floor(lon_from / RASHI_WIDTH_DEG).astype(np.int64)
    sign_to = np.floor(lon_to / RASHI_WIDTH_DEG).astype(np.int64)
    house_distance = np.mod(sign_to - sign_from, RASHI_COUNT) + 1
    return {
        f"drishti_h{offset}": (house_distance == offset).astype(np.float64)
        for offset in params["house_offsets"]
    }


# --------------------------------------------------------------------------
# Absolute zodiacal position (Vedic)
# --------------------------------------------------------------------------


@operator(
    name="rashi_cyclic",
    version=1,
    inputs=("body",),
    emits=("rashi_sin", "rashi_cos", "degree_in_sign", "to_next_sign"),
    ranges={
        "rashi_sin": (-1.0, 1.0),
        "rashi_cos": (-1.0, 1.0),
        "degree_in_sign": (0.0, 1.0),
        "to_next_sign": (0.0, 1.0),
    },
    invariance="covariant",
)
def rashi_cyclic(provider: EphemerisProvider, inputs: dict, params: dict) -> dict:
    """Sidereal rashi as a cyclic embedding plus position within the sign.

    Regenerated from longitude here rather than read from any supplied
    ``*_sign_sin/cos`` column, which the source audit rejected.
    """

    longitude = provider.longitude(inputs["body"], "sidereal")
    rashi_id = np.floor(longitude / RASHI_WIDTH_DEG)
    degree_in_sign = longitude - rashi_id * RASHI_WIDTH_DEG
    angle = 2.0 * np.pi * rashi_id / RASHI_COUNT
    return {
        "rashi_sin": np.sin(angle),
        "rashi_cos": np.cos(angle),
        "degree_in_sign": degree_in_sign / RASHI_WIDTH_DEG,
        "to_next_sign": (RASHI_WIDTH_DEG - degree_in_sign) / RASHI_WIDTH_DEG,
    }


@operator(
    name="nakshatra_pada",
    version=1,
    inputs=("body",),
    emits=("nak_sin", "nak_cos", "pada_sin", "pada_cos", "to_next_nakshatra"),
    ranges={
        "nak_sin": (-1.0, 1.0),
        "nak_cos": (-1.0, 1.0),
        "pada_sin": (-1.0, 1.0),
        "pada_cos": (-1.0, 1.0),
        "to_next_nakshatra": (0.0, 1.0),
    },
    invariance="covariant",
)
def nakshatra_pada(provider: EphemerisProvider, inputs: dict, params: dict) -> dict:
    """27-mansion nakshatra and pada as cyclic embeddings."""

    longitude = provider.longitude(inputs["body"], "sidereal")
    nakshatra_id = np.floor(longitude / NAKSHATRA_WIDTH_DEG)
    within = longitude - nakshatra_id * NAKSHATRA_WIDTH_DEG
    pada_id = np.floor(within / (NAKSHATRA_WIDTH_DEG / PADA_PER_NAKSHATRA))
    nak_angle = 2.0 * np.pi * nakshatra_id / NAKSHATRA_COUNT
    pada_angle = 2.0 * np.pi * pada_id / PADA_PER_NAKSHATRA
    return {
        "nak_sin": np.sin(nak_angle),
        "nak_cos": np.cos(nak_angle),
        "pada_sin": np.sin(pada_angle),
        "pada_cos": np.cos(pada_angle),
        "to_next_nakshatra": (NAKSHATRA_WIDTH_DEG - within) / NAKSHATRA_WIDTH_DEG,
    }


@operator(
    name="ingress_pulse",
    version=1,
    inputs=("body",),
    params=("width_days",),
    emits=("pre_ingress", "post_ingress"),
    ranges={"pre_ingress": (0.0, 1.0), "post_ingress": (0.0, 1.0)},
    invariance="covariant",
)
def ingress_pulse(provider: EphemerisProvider, inputs: dict, params: dict) -> dict:
    """Proximity to a sign ingress, with anticipation and aftermath separated.

    Pre and post channels stay separate so an anticipatory hypothesis is not
    forced to be symmetric with an aftermath hypothesis.
    """

    width = float(params["width_days"])
    if width <= 0.0:
        raise RuleContractError("width_days must be positive.")

    longitude = provider.longitude(inputs["body"], "sidereal")
    rashi_id = np.floor(longitude / RASHI_WIDTH_DEG).astype(np.int64)
    ingress = np.flatnonzero(np.diff(rashi_id) != 0) + 1

    count = longitude.shape[0]
    positions = np.arange(count, dtype=np.float64)
    if ingress.size == 0:
        zeros = np.zeros(count, dtype=np.float64)
        return {"pre_ingress": zeros, "post_ingress": zeros.copy()}

    ingress_positions = positions[ingress]
    # Nearest ingress on each side; the ephemeris grid is daily, so row
    # distance is calendar-day distance.
    next_slot = np.searchsorted(ingress_positions, positions, side="left")
    previous_slot = next_slot - 1

    days_to_next = np.where(
        next_slot < ingress_positions.size,
        ingress_positions[np.clip(next_slot, 0, ingress_positions.size - 1)] - positions,
        np.inf,
    )
    days_since_previous = np.where(
        previous_slot >= 0,
        positions - ingress_positions[np.clip(previous_slot, 0, ingress_positions.size - 1)],
        np.inf,
    )

    return {
        "pre_ingress": np.exp(-0.5 * (days_to_next / width) ** 2),
        "post_ingress": np.exp(-0.5 * (days_since_previous / width) ** 2),
    }


# --------------------------------------------------------------------------
# Motion state
# --------------------------------------------------------------------------


@operator(
    name="retrograde_state",
    version=1,
    inputs=("body",),
    emits=("is_retrograde", "speed_norm"),
    ranges={"is_retrograde": (0.0, 1.0), "speed_norm": (-1.0, 1.0)},
    invariance="invariant",
)
def retrograde_state(provider: EphemerisProvider, inputs: dict, params: dict) -> dict:
    """Retrograde indicator and a bounded signed speed."""

    speed = provider.speed(inputs["body"])
    return {
        "is_retrograde": (speed < 0.0).astype(np.float64),
        "speed_norm": np.tanh(speed),
    }


@operator(
    name="station_proximity",
    version=1,
    inputs=("body",),
    params=("speed_scale",),
    emits=("station_score",),
    ranges={"station_score": (0.0, 1.0)},
    invariance="invariant",
)
def station_proximity(provider: EphemerisProvider, inputs: dict, params: dict) -> dict:
    """Closeness to a station, where longitudinal speed passes through zero."""

    scale = float(params["speed_scale"])
    if scale <= 0.0:
        raise RuleContractError("speed_scale must be positive.")
    speed = provider.speed(inputs["body"])
    return {"station_score": np.exp(-((np.abs(speed) / scale) ** 2))}


# --------------------------------------------------------------------------
# Sun-relative and nodal
# --------------------------------------------------------------------------


@operator(
    name="combustion",
    version=1,
    inputs=("body",),
    optional_inputs=("frame",),
    params=("width_deg",),
    emits=("sun_proximity", "sun_sep_sin", "sun_sep_cos"),
    ranges={
        "sun_proximity": (0.0, 1.0),
        "sun_sep_sin": (-1.0, 1.0),
        "sun_sep_cos": (-1.0, 1.0),
    },
    invariance="invariant",
)
def combustion(provider: EphemerisProvider, inputs: dict, params: dict) -> dict:
    """Smooth proximity to the Sun.

    A continuous basis is used rather than a binary flag because published
    combustion thresholds differ by planet and retrograde state.
    """

    frame = _frame_of(inputs, "sidereal")
    width = float(params["width_deg"])
    if width <= 0.0:
        raise RuleContractError("width_deg must be positive.")
    separation = directed_separation(
        provider.longitude("Sun", frame), provider.longitude(inputs["body"], frame)
    )
    radians = np.radians(separation)
    return {
        "sun_proximity": _gaussian_activation(np.abs(separation), width),
        "sun_sep_sin": np.sin(radians),
        "sun_sep_cos": np.cos(radians),
    }


@operator(
    name="node_axis",
    version=1,
    inputs=("body",),
    optional_inputs=("frame",),
    params=("width_deg",),
    emits=("axis_proximity", "axis_sep_sin", "axis_sep_cos"),
    ranges={
        "axis_proximity": (0.0, 1.0),
        "axis_sep_sin": (-1.0, 1.0),
        "axis_sep_cos": (-1.0, 1.0),
    },
    invariance="invariant",
)
def node_axis(provider: EphemerisProvider, inputs: dict, params: dict) -> dict:
    """Proximity to the Rahu/Ketu axis, the eclipse-relevant geometry.

    The axis is undirected, so proximity folds Rahu and Ketu together while the
    sin/cos pair retains the directed position along it.
    """

    frame = _frame_of(inputs, "sidereal")
    width = float(params["width_deg"])
    if width <= 0.0:
        raise RuleContractError("width_deg must be positive.")
    separation = directed_separation(
        provider.longitude("Rahu", frame), provider.longitude(inputs["body"], frame)
    )
    # Distance to the nearest end of the axis: fold onto [-90, 90].
    axis_distance = np.abs(wrap_signed(2.0 * separation)) / 2.0
    radians = np.radians(separation)
    return {
        "axis_proximity": _gaussian_activation(axis_distance, width),
        "axis_sep_sin": np.sin(radians),
        "axis_sep_cos": np.cos(radians),
    }


# --------------------------------------------------------------------------
# Western declination and midpoints
# --------------------------------------------------------------------------


@operator(
    name="declination_parallel",
    version=1,
    inputs=("body_a", "body_b"),
    params=("orb_deg",),
    emits=("parallel", "contraparallel", "declination_gap"),
    ranges={
        "parallel": (0.0, 1.0),
        "contraparallel": (0.0, 1.0),
        "declination_gap": (-1.0, 1.0),
    },
    invariance="invariant",
)
def declination_parallel(provider: EphemerisProvider, inputs: dict, params: dict) -> dict:
    """Western declination parallels and contraparallels.

    A parallel holds when two bodies share a declination; a contraparallel when
    their declinations are equal and opposite.
    """

    orb = float(params["orb_deg"])
    if orb <= 0.0:
        raise RuleContractError("orb_deg must be positive.")
    declination_a = provider.declination(inputs["body_a"])
    declination_b = provider.declination(inputs["body_b"])
    return {
        "parallel": _gaussian_activation(np.abs(declination_a - declination_b), orb),
        "contraparallel": _gaussian_activation(
            np.abs(declination_a + declination_b), orb
        ),
        "declination_gap": np.tanh((declination_a - declination_b) / 45.0),
    }


@operator(
    name="midpoint_activation",
    version=1,
    inputs=("body_a", "body_b", "target_body"),
    optional_inputs=("frame",),
    params=("orb_deg",),
    emits=("direct_midpoint", "indirect_midpoint"),
    ranges={"direct_midpoint": (0.0, 1.0), "indirect_midpoint": (0.0, 1.0)},
    invariance="invariant",
)
def midpoint_activation(provider: EphemerisProvider, inputs: dict, params: dict) -> dict:
    """Contact of a third body with the midpoint of a pair.

    The near midpoint and its opposite are both reported, since the direct and
    indirect midpoint traditions treat them differently.
    """

    frame = _frame_of(inputs, "tropical")
    orb = float(params["orb_deg"])
    if orb <= 0.0:
        raise RuleContractError("orb_deg must be positive.")
    lon_a = provider.longitude(inputs["body_a"], frame)
    lon_b = provider.longitude(inputs["body_b"], frame)
    # Circular midpoint: advance from a by half the wrapped separation.
    midpoint = np.mod(lon_a + directed_separation(lon_a, lon_b) / 2.0, 360.0)
    target = provider.longitude(inputs["target_body"], frame)
    offset = np.abs(wrap_signed(target - midpoint))
    return {
        "direct_midpoint": _gaussian_activation(offset, orb),
        "indirect_midpoint": _gaussian_activation(np.abs(180.0 - offset), orb),
    }
