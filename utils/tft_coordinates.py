"""Explicit temporal-coordinate contracts for the native TFT.

The native TFT must not infer elapsed time from tensor position when samples can
be irregularly spaced.  :class:`TemporalCoordinateContract` carries numeric
coordinates and their validity mask together with the unit and provenance that
give those numbers meaning.

Mask policy
-----------
``True`` always means a usable coordinate.  Holes are allowed: monotonicity is
checked between successive *valid* tokens, even when invalid tokens occur
between them.  Every batch row must contain at least one valid token.  This
last rule deliberately prevents all-masked attention rows, which otherwise tend
to produce undefined normalisation or NaNs downstream.  Non-finite values are
permitted only at invalid positions and are canonicalised to zero.

There is no implicit ``arange`` fallback.  Row-index coordinates can only be
created through :meth:`TemporalCoordinateContract.from_regular_row_index`, and
that factory requires the caller to explicitly declare regular sampling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Literal, Sequence

import torch


TemporalUnit = Literal["steps", "trading_sessions", "calendar_days"]
TemporalSource = Literal[
    "row_index", "named_known_feature", "explicit_argument"
]

SUPPORTED_TEMPORAL_UNITS: tuple[TemporalUnit, ...] = (
    "steps",
    "trading_sessions",
    "calendar_days",
)
SUPPORTED_TEMPORAL_SOURCES: tuple[TemporalSource, ...] = (
    "row_index",
    "named_known_feature",
    "explicit_argument",
)
TEMPORAL_COORDINATE_SCHEMA_VERSION = 1

_ROW_INDEX_FACTORY_TOKEN = object()


def _as_numeric_positions(value: Any) -> torch.Tensor:
    try:
        tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError("positions must contain numeric values.") from exc

    if tensor.dtype == torch.bool or tensor.is_complex() or not (
        tensor.is_floating_point()
        or tensor.dtype
        in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
    ):
        raise TypeError(
            "positions must use a real numeric dtype (integer or floating point)."
        )
    if tensor.ndim not in (1, 2):
        raise ValueError(
            "positions must have shape [T] or [B,T], "
            f"got {tuple(tensor.shape)}."
        )
    if tensor.shape[-1] == 0:
        raise ValueError("positions must contain at least one time token.")
    if tensor.ndim == 2 and tensor.shape[0] == 0:
        raise ValueError("positions must contain at least one batch row.")

    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.detach().to(dtype=torch.float64).clone()


def _as_valid_mask(
    value: Any | None,
    *,
    batch_size: int,
    time_length: int,
    device: torch.device,
) -> torch.Tensor:
    if value is None:
        return torch.ones(
            (batch_size, time_length), dtype=torch.bool, device=device
        )
    try:
        mask = value if torch.is_tensor(value) else torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError("valid_mask must contain boolean values.") from exc
    if mask.dtype != torch.bool:
        raise TypeError("valid_mask must have boolean dtype (True means valid).")
    if mask.ndim == 1:
        if mask.shape[0] != time_length:
            raise ValueError(
                "A rank-1 valid_mask must have shape [T] matching positions; "
                f"got {tuple(mask.shape)} for T={time_length}."
            )
        mask = mask.unsqueeze(0).expand(batch_size, -1)
    elif mask.ndim == 2:
        if tuple(mask.shape) != (batch_size, time_length):
            raise ValueError(
                "A rank-2 valid_mask must have shape [B,T] matching positions; "
                f"got {tuple(mask.shape)} versus {(batch_size, time_length)}."
            )
    else:
        raise ValueError(
            "valid_mask must have shape [T] or [B,T], "
            f"got {tuple(mask.shape)}."
        )
    return mask.detach().to(device=device).clone()


def _validate_unit(unit: str) -> TemporalUnit:
    if not isinstance(unit, str) or unit not in SUPPORTED_TEMPORAL_UNITS:
        raise ValueError(
            f"unit must be one of {SUPPORTED_TEMPORAL_UNITS}, got {unit!r}."
        )
    return unit  # type: ignore[return-value]


def _validate_source(source: str) -> TemporalSource:
    if not isinstance(source, str) or source not in SUPPORTED_TEMPORAL_SOURCES:
        raise ValueError(
            f"source must be one of {SUPPORTED_TEMPORAL_SOURCES}, got {source!r}."
        )
    return source  # type: ignore[return-value]


def _validate_index_tensor(
    indices: Any,
    *,
    batch_size: int,
    time_length: int,
    device: torch.device,
) -> torch.Tensor:
    try:
        index = indices if torch.is_tensor(indices) else torch.as_tensor(indices)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError("time indices must contain integers.") from exc
    if index.dtype == torch.bool or index.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError("time indices must have an integer dtype.")
    if index.ndim not in (1, 2):
        raise ValueError("time indices must have shape [K] or [B,K].")
    if index.shape[-1] == 0:
        raise ValueError("time indices must select at least one token.")
    if index.ndim == 2 and index.shape[0] != batch_size:
        raise ValueError(
            "Per-batch time indices must have shape [B,K]; "
            f"got batch {index.shape[0]} versus {batch_size}."
        )
    index = index.detach().to(device=device, dtype=torch.long).clone()
    if torch.any(index < 0) or torch.any(index >= time_length):
        raise IndexError(f"time indices must be inside [0, {time_length - 1}].")
    return index


@dataclass(frozen=True, eq=False, init=False)
class TemporalCoordinateContract:
    """Immutable, canonical ``[B,T]`` temporal coordinates and validity.

    Inputs may be ``[T]`` or ``[B,T]``.  The canonical coordinate dtype is
    ``torch.float64`` so calendar/session offsets and fractional coordinates
    share one predictable representation.  Tensor inputs are defensively
    copied and public tensor properties also return copies.

    ``source_name`` is optional provenance detail (for example the column name
    when ``source='named_known_feature'``); when present it must be non-empty.
    """

    _positions: torch.Tensor = field(repr=False)
    _valid_mask: torch.Tensor = field(repr=False)
    unit: TemporalUnit
    source: TemporalSource
    source_name: str | None
    _declared_regular_sampling: bool = field(repr=False)

    def __init__(
        self,
        positions: Any,
        *,
        unit: TemporalUnit,
        valid_mask: Any | None = None,
        source: TemporalSource = "explicit_argument",
        source_name: str | None = None,
        _factory_token: object | None = None,
    ) -> None:
        resolved_unit = _validate_unit(unit)
        resolved_source = _validate_source(source)
        if resolved_source == "row_index" and _factory_token is not _ROW_INDEX_FACTORY_TOKEN:
            raise ValueError(
                "row_index coordinates may only be created by "
                "TemporalCoordinateContract.from_regular_row_index(..., "
                "declared_regular_sampling=True)."
            )
        if source_name is not None and (
            not isinstance(source_name, str) or not source_name.strip()
        ):
            raise ValueError("source_name must be None or a non-empty string.")

        canonical_positions = _as_numeric_positions(positions)
        batch_size, time_length = canonical_positions.shape
        canonical_mask = _as_valid_mask(
            valid_mask,
            batch_size=batch_size,
            time_length=time_length,
            device=canonical_positions.device,
        )

        valid_counts = canonical_mask.sum(dim=1)
        empty_rows = torch.nonzero(valid_counts == 0, as_tuple=False).flatten()
        if empty_rows.numel():
            rows = [int(value) for value in empty_rows.cpu().tolist()]
            raise ValueError(
                "Every batch row must contain at least one valid coordinate; "
                f"all-invalid rows: {rows}."
            )

        finite = torch.isfinite(canonical_positions)
        invalid_finite_rows = torch.nonzero(
            canonical_mask & ~finite, as_tuple=False
        )
        if invalid_finite_rows.numel():
            first = tuple(int(value) for value in invalid_finite_rows[0].tolist())
            raise ValueError(
                "Every valid coordinate must be finite; first non-finite "
                f"coordinate at batch/time index {first}."
            )
        # Masked padding may carry NaN/Inf from an upstream collation step.  It
        # has no coordinate meaning, so normalise it to a deterministic value.
        canonical_positions = torch.where(
            finite, canonical_positions, torch.zeros_like(canonical_positions)
        )

        for batch_index in range(batch_size):
            values = canonical_positions[batch_index][canonical_mask[batch_index]]
            if values.numel() > 1 and torch.any(values[1:] <= values[:-1]):
                raise ValueError(
                    "Valid coordinates must be strictly increasing over "
                    "successive valid tokens; violation in batch row "
                    f"{batch_index}."
                )

        object.__setattr__(self, "_positions", canonical_positions.contiguous())
        object.__setattr__(self, "_valid_mask", canonical_mask.contiguous())
        object.__setattr__(self, "unit", resolved_unit)
        object.__setattr__(self, "source", resolved_source)
        object.__setattr__(
            self, "source_name", source_name.strip() if source_name is not None else None
        )
        object.__setattr__(
            self,
            "_declared_regular_sampling",
            resolved_source == "row_index",
        )

    @classmethod
    def from_regular_row_index(
        cls,
        length: int,
        *,
        batch_size: int = 1,
        start: Real = 0,
        step: Real = 1,
        unit: TemporalUnit = "steps",
        valid_mask: Any | None = None,
        declared_regular_sampling: bool,
        device: torch.device | str | None = None,
        source_name: str | None = None,
    ) -> "TemporalCoordinateContract":
        """Create row-index coordinates after an explicit regularity claim.

        The mandatory ``declared_regular_sampling`` keyword must be exactly
        ``True``.  This makes row positions an intentional modelling choice,
        never an automatic substitute for unavailable timestamps.
        """

        if declared_regular_sampling is not True:
            raise ValueError(
                "Row-index coordinates require declared_regular_sampling=True."
            )
        if isinstance(length, bool) or not isinstance(length, Integral) or length <= 0:
            raise ValueError("length must be a positive integer.")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, Integral)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer.")
        if isinstance(start, bool) or not isinstance(start, Real) or not math.isfinite(float(start)):
            raise ValueError("start must be a finite real number.")
        if isinstance(step, bool) or not isinstance(step, Real) or not math.isfinite(float(step)) or step <= 0:
            raise ValueError("step must be a finite positive real number.")

        row = torch.arange(int(length), dtype=torch.float64, device=device)
        row = float(start) + row * float(step)
        positions = row.unsqueeze(0).expand(int(batch_size), -1).clone()
        return cls(
            positions,
            unit=unit,
            valid_mask=valid_mask,
            source="row_index",
            source_name=source_name,
            _factory_token=_ROW_INDEX_FACTORY_TOKEN,
        )

    @property
    def positions(self) -> torch.Tensor:
        """A defensive copy of canonical coordinates with shape ``[B,T]``."""

        return self._positions.clone()

    @property
    def coordinates(self) -> torch.Tensor:
        """Alias for :attr:`positions`."""

        return self.positions

    @property
    def valid_mask(self) -> torch.Tensor:
        """A defensive copy of the canonical mask; ``True`` means valid."""

        return self._valid_mask.clone()

    @property
    def batch_size(self) -> int:
        return int(self._positions.shape[0])

    @property
    def time_length(self) -> int:
        return int(self._positions.shape[1])

    @property
    def declared_regular_sampling(self) -> bool:
        return self._declared_regular_sampling

    @property
    def device(self) -> torch.device:
        return self._positions.device

    def _new(
        self,
        positions: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> "TemporalCoordinateContract":
        return TemporalCoordinateContract(
            positions,
            unit=self.unit,
            valid_mask=valid_mask,
            source=self.source,
            source_name=self.source_name,
            _factory_token=(
                _ROW_INDEX_FACTORY_TOKEN if self.source == "row_index" else None
            ),
        )

    def broadcast_to(self, batch_size: int) -> "TemporalCoordinateContract":
        """Broadcast a single-row contract to ``batch_size`` rows."""

        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, Integral)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer.")
        batch_size = int(batch_size)
        if batch_size == self.batch_size:
            return self
        if self.batch_size != 1:
            raise ValueError(
                "Only a single-row coordinate contract can be broadcast; "
                f"current batch size is {self.batch_size}."
            )
        return self._new(
            self._positions.expand(batch_size, -1).clone(),
            self._valid_mask.expand(batch_size, -1).clone(),
        )

    def slice_time(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> "TemporalCoordinateContract":
        """Apply a forward Python slice along the time axis."""

        resolved_step = 1 if step is None else step
        if (
            isinstance(resolved_step, bool)
            or not isinstance(resolved_step, Integral)
            or resolved_step <= 0
        ):
            raise ValueError("slice step must be a positive integer.")
        time_slice = slice(start, stop, int(resolved_step))
        positions = self._positions[:, time_slice]
        if positions.shape[1] == 0:
            raise ValueError("The requested time slice is empty.")
        return self._new(positions, self._valid_mask[:, time_slice])

    def gather_time(self, indices: Any) -> "TemporalCoordinateContract":
        """Gather shared ``[K]`` or per-batch ``[B,K]`` source positions.

        Output order must still be strictly chronological over valid tokens; a
        duplicated or reordered gather is rejected by contract validation.
        """

        index = _validate_index_tensor(
            indices,
            batch_size=self.batch_size,
            time_length=self.time_length,
            device=self.device,
        )
        if index.ndim == 1:
            positions = self._positions.index_select(1, index)
            mask = self._valid_mask.index_select(1, index)
        else:
            positions = torch.gather(self._positions, 1, index)
            mask = torch.gather(self._valid_mask, 1, index)
        return self._new(positions, mask)

    def index_select_time(self, indices: Any) -> "TemporalCoordinateContract":
        """Alias for :meth:`gather_time`."""

        return self.gather_time(indices)

    def shifted_source(self, offset: int) -> "TemporalCoordinateContract":
        """Map output token ``t`` to source token ``t + offset``.

        Tokens whose shifted source lies outside the sequence are invalid.  A
        shift that leaves an entire batch row invalid is rejected by the normal
        all-invalid policy.
        """

        if isinstance(offset, bool) or not isinstance(offset, Integral):
            raise TypeError("offset must be an integer.")
        offset = int(offset)
        if offset == 0:
            return self
        positions = torch.zeros_like(self._positions)
        mask = torch.zeros_like(self._valid_mask)
        if offset > 0 and offset < self.time_length:
            positions[:, :-offset] = self._positions[:, offset:]
            mask[:, :-offset] = self._valid_mask[:, offset:]
        elif offset < 0 and -offset < self.time_length:
            amount = -offset
            positions[:, amount:] = self._positions[:, :-amount]
            mask[:, amount:] = self._valid_mask[:, :-amount]
        return self._new(positions, mask)

    def concat(self, *others: "TemporalCoordinateContract") -> "TemporalCoordinateContract":
        """Concatenate compatible contracts along time."""

        contracts: Sequence[TemporalCoordinateContract] = (self, *others)
        for index, contract in enumerate(contracts[1:], start=1):
            if not isinstance(contract, TemporalCoordinateContract):
                raise TypeError(
                    f"concat operand {index} must be a TemporalCoordinateContract."
                )
            if contract.batch_size != self.batch_size:
                raise ValueError("All concatenated contracts must share batch size.")
            if contract.device != self.device:
                raise ValueError("All concatenated contracts must share device.")
            if (
                contract.unit != self.unit
                or contract.source != self.source
                or contract.source_name != self.source_name
            ):
                raise ValueError(
                    "All concatenated contracts must share unit and source provenance."
                )
        return self._new(
            torch.cat([contract._positions for contract in contracts], dim=1),
            torch.cat([contract._valid_mask for contract in contracts], dim=1),
        )

    @classmethod
    def concatenate(
        cls, contracts: Sequence["TemporalCoordinateContract"]
    ) -> "TemporalCoordinateContract":
        """Sequence-oriented equivalent of :meth:`concat`."""

        if not isinstance(contracts, Sequence) or not contracts:
            raise ValueError("contracts must be a non-empty sequence.")
        first, *rest = contracts
        if not isinstance(first, TemporalCoordinateContract):
            raise TypeError("Every item must be a TemporalCoordinateContract.")
        return first.concat(*rest)

    def compose_valid_mask(self, *masks: Any) -> "TemporalCoordinateContract":
        """Intersect one or more masks with existing validity.

        Validity composition is intentionally AND-only: an external mask may
        remove usable coordinates but cannot resurrect a coordinate whose
        source marked it invalid.
        """

        if not masks:
            raise ValueError("At least one mask is required for composition.")
        composed = self._valid_mask.clone()
        for candidate in masks:
            composed &= _as_valid_mask(
                candidate,
                batch_size=self.batch_size,
                time_length=self.time_length,
                device=self.device,
            )
        return self._new(self._positions, composed)

    def to_metadata(self) -> dict[str, Any]:
        """Return compact, JSON-safe provenance and validity metadata."""

        counts = [int(value) for value in self._valid_mask.sum(dim=1).cpu().tolist()]
        first_valid: list[float] = []
        last_valid: list[float] = []
        has_internal_holes: list[bool] = []
        for positions, mask in zip(self._positions, self._valid_mask):
            valid_indices = torch.nonzero(mask, as_tuple=False).flatten()
            first = int(valid_indices[0])
            last = int(valid_indices[-1])
            values = positions[mask]
            first_valid.append(float(values[0].item()))
            last_valid.append(float(values[-1].item()))
            has_internal_holes.append(bool(torch.any(~mask[first : last + 1]).item()))

        return {
            "schema_version": TEMPORAL_COORDINATE_SCHEMA_VERSION,
            "unit": self.unit,
            "source": self.source,
            "source_name": self.source_name,
            "shape": [self.batch_size, self.time_length],
            "dtype": "float64",
            "device": str(self.device),
            "true_means_valid": True,
            "holes_allowed": True,
            "all_invalid_rows_allowed": False,
            "declared_regular_sampling": self.declared_regular_sampling,
            "valid_count_by_batch": counts,
            "first_valid_coordinate_by_batch": first_valid,
            "last_valid_coordinate_by_batch": last_valid,
            "has_internal_holes_by_batch": has_internal_holes,
        }

