from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest
import torch

from utils.tft_coordinates import TemporalCoordinateContract


def test_rank_one_positions_are_canonicalized_to_float64_bt():
    source = torch.tensor([0, 1, 3], dtype=torch.int32)
    contract = TemporalCoordinateContract(
        source,
        unit="calendar_days",
        source="explicit_argument",
    )

    assert contract.positions.shape == (1, 3)
    assert contract.positions.dtype == torch.float64
    assert torch.equal(contract.positions, torch.tensor([[0.0, 1.0, 3.0]]))
    assert torch.equal(contract.valid_mask, torch.ones(1, 3, dtype=torch.bool))


def test_rank_two_positions_and_shared_rank_one_mask_are_canonicalized():
    contract = TemporalCoordinateContract(
        [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
        unit="trading_sessions",
        valid_mask=[True, False, True],
        source="named_known_feature",
        source_name="session_number",
    )

    assert contract.positions.shape == (2, 3)
    assert torch.equal(
        contract.valid_mask,
        torch.tensor([[True, False, True], [True, False, True]]),
    )
    assert contract.source_name == "session_number"


@pytest.mark.parametrize("unit", ["hours", "", None])
def test_invalid_unit_is_rejected(unit):
    with pytest.raises(ValueError, match="unit must be one of"):
        TemporalCoordinateContract([0, 1], unit=unit)


@pytest.mark.parametrize("source", ["timestamp", "", None])
def test_invalid_source_is_rejected(source):
    with pytest.raises(ValueError, match="source must be one of"):
        TemporalCoordinateContract([0, 1], unit="steps", source=source)


@pytest.mark.parametrize(
    "positions",
    [
        torch.tensor(1.0),
        torch.ones(1, 2, 3),
        torch.empty(0),
        torch.empty(0, 3),
    ],
)
def test_invalid_position_shape_is_rejected(positions):
    with pytest.raises(ValueError, match="positions must|at least one"):
        TemporalCoordinateContract(positions, unit="steps")


@pytest.mark.parametrize(
    "positions",
    [
        torch.tensor([True, False]),
        torch.tensor([1 + 0j, 2 + 0j]),
        ["one", "two"],
    ],
)
def test_invalid_position_dtype_is_rejected(positions):
    with pytest.raises(TypeError, match="numeric dtype|numeric values"):
        TemporalCoordinateContract(positions, unit="steps")


def test_invalid_mask_dtype_and_shape_are_rejected():
    with pytest.raises(TypeError, match="boolean dtype"):
        TemporalCoordinateContract(
            [0, 1, 2], unit="steps", valid_mask=[1, 1, 0]
        )
    with pytest.raises(ValueError, match="matching positions"):
        TemporalCoordinateContract(
            [[0, 1, 2], [0, 1, 2]],
            unit="steps",
            valid_mask=torch.ones(1, 3, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match=r"shape \[T\] or \[B,T\]"):
        TemporalCoordinateContract(
            [0, 1, 2],
            unit="steps",
            valid_mask=torch.ones(1, 1, 3, dtype=torch.bool),
        )


def test_nonfinite_valid_coordinate_is_rejected_but_masked_padding_is_canonicalized():
    with pytest.raises(ValueError, match="valid coordinate must be finite"):
        TemporalCoordinateContract(
            [0.0, float("nan"), 2.0], unit="steps"
        )

    contract = TemporalCoordinateContract(
        [0.0, float("nan"), 2.0],
        unit="steps",
        valid_mask=[True, False, True],
    )
    assert contract.positions[0, 1].item() == 0.0
    assert torch.isfinite(contract.positions).all()


@pytest.mark.parametrize(
    "positions,mask",
    [
        ([0.0, 1.0, 1.0], None),
        ([0.0, 2.0, 1.0], None),
        ([3.0, 100.0, 2.0], [True, False, True]),
    ],
)
def test_duplicate_or_decreasing_successive_valid_coordinates_are_rejected(
    positions, mask
):
    with pytest.raises(ValueError, match="strictly increasing"):
        TemporalCoordinateContract(
            positions, unit="calendar_days", valid_mask=mask
        )


def test_mask_holes_are_allowed_and_reported():
    contract = TemporalCoordinateContract(
        [0.0, 100.0, 4.0, 5.0],
        unit="calendar_days",
        valid_mask=[True, False, True, True],
    )

    metadata = contract.to_metadata()
    assert metadata["holes_allowed"] is True
    assert metadata["has_internal_holes_by_batch"] == [True]
    assert metadata["valid_count_by_batch"] == [3]


def test_every_batch_row_must_have_a_valid_coordinate():
    with pytest.raises(ValueError, match=r"all-invalid rows: \[1\]"):
        TemporalCoordinateContract(
            [[0.0, 1.0], [0.0, 1.0]],
            unit="steps",
            valid_mask=[[True, False], [False, False]],
        )


def test_friday_to_monday_calendar_and_trading_coordinates_are_distinct():
    # Friday -> Monday spans three calendar days but only one next trading
    # session.  Both contracts have two tensor tokens; their units/deltas retain
    # the materially different elapsed-time meanings.
    calendar = TemporalCoordinateContract(
        [0.0, 3.0], unit="calendar_days", source="explicit_argument"
    )
    sessions = TemporalCoordinateContract(
        [0.0, 1.0], unit="trading_sessions", source="explicit_argument"
    )

    assert (calendar.positions[:, 1] - calendar.positions[:, 0]).item() == 3.0
    assert (sessions.positions[:, 1] - sessions.positions[:, 0]).item() == 1.0
    assert calendar.unit != sessions.unit


def test_row_index_requires_explicit_regular_sampling_factory():
    with pytest.raises(ValueError, match="only be created"):
        TemporalCoordinateContract(
            [0, 1, 2], unit="steps", source="row_index"
        )
    with pytest.raises(ValueError, match="declared_regular_sampling=True"):
        TemporalCoordinateContract.from_regular_row_index(
            3, declared_regular_sampling=False
        )

    contract = TemporalCoordinateContract.from_regular_row_index(
        4,
        batch_size=2,
        start=10,
        step=2,
        unit="steps",
        declared_regular_sampling=True,
    )
    assert contract.source == "row_index"
    assert contract.declared_regular_sampling is True
    assert torch.equal(
        contract.positions,
        torch.tensor([[10.0, 12.0, 14.0, 16.0]]).expand(2, -1),
    )


def test_slice_concat_and_shared_gather_preserve_coordinate_semantics():
    full = TemporalCoordinateContract(
        [0.0, 1.0, 3.0, 4.0, 7.0],
        unit="calendar_days",
        source="explicit_argument",
        source_name="days_since_epoch",
    )

    left = full.slice_time(0, 2)
    right = full.slice_time(2, None)
    rebuilt = left.concat(right)
    gathered = full.gather_time([0, 2, 4])

    assert torch.equal(rebuilt.positions, full.positions)
    assert torch.equal(gathered.positions, torch.tensor([[0.0, 3.0, 7.0]]))
    assert gathered.unit == "calendar_days"
    assert gathered.source_name == "days_since_epoch"


def test_per_batch_gather_and_index_select_alias():
    contract = TemporalCoordinateContract(
        [[0.0, 1.0, 3.0], [10.0, 12.0, 20.0]],
        unit="calendar_days",
    )
    gathered = contract.gather_time([[0, 2], [1, 2]])
    selected = contract.index_select_time([0, 2])

    assert torch.equal(
        gathered.positions, torch.tensor([[0.0, 3.0], [12.0, 20.0]])
    )
    assert torch.equal(
        selected.positions, torch.tensor([[0.0, 3.0], [10.0, 20.0]])
    )


@pytest.mark.parametrize("indices", [[1, 0], [0, 0]])
def test_gather_rejects_nonchronological_or_duplicate_indices(indices):
    contract = TemporalCoordinateContract([0.0, 1.0, 2.0], unit="steps")
    with pytest.raises(ValueError, match="strictly increasing"):
        contract.gather_time(indices)


def test_shifted_source_maps_t_to_t_plus_offset_and_masks_boundaries():
    contract = TemporalCoordinateContract(
        [10.0, 11.0, 13.0, 14.0],
        unit="calendar_days",
        valid_mask=[True, False, True, True],
    )

    forward = contract.shifted_source(1)
    backward = contract.shifted_source(-1)

    assert torch.equal(forward.positions, torch.tensor([[11.0, 13.0, 14.0, 0.0]]))
    assert torch.equal(
        forward.valid_mask, torch.tensor([[False, True, True, False]])
    )
    assert torch.equal(backward.positions, torch.tensor([[0.0, 10.0, 11.0, 13.0]]))
    assert torch.equal(
        backward.valid_mask, torch.tensor([[False, True, False, True]])
    )
    with pytest.raises(ValueError, match="all-invalid"):
        contract.shifted_source(contract.time_length)


def test_broadcast_and_mask_composition_are_explicit_and_and_only():
    contract = TemporalCoordinateContract(
        [0.0, 1.0, 2.0, 3.0],
        unit="steps",
        valid_mask=[True, True, False, True],
    ).broadcast_to(2)
    composed = contract.compose_valid_mask(
        [True, False, True, True],
        [[True, True, True, False], [True, True, True, True]],
    )

    assert torch.equal(
        composed.valid_mask,
        torch.tensor([[True, False, False, False], [True, False, False, True]]),
    )
    with pytest.raises(ValueError, match="all-invalid"):
        contract.compose_valid_mask(torch.zeros(2, 4, dtype=torch.bool))


def test_concat_rejects_incompatible_provenance_and_boundary_order():
    left = TemporalCoordinateContract([0.0, 1.0], unit="steps")
    wrong_unit = TemporalCoordinateContract([2.0, 3.0], unit="calendar_days")
    wrong_order = TemporalCoordinateContract([0.5, 2.0], unit="steps")

    with pytest.raises(ValueError, match="unit and source provenance"):
        left.concat(wrong_unit)
    with pytest.raises(ValueError, match="strictly increasing"):
        left.concat(wrong_order)


def test_metadata_is_json_safe_and_contract_is_defensively_immutable():
    source = torch.tensor([0.0, 1.0, 2.0])
    source_mask = torch.tensor([True, False, True])
    contract = TemporalCoordinateContract(
        source,
        unit="steps",
        valid_mask=source_mask,
        source="named_known_feature",
        source_name="step_id",
    )

    source[0] = 99.0
    source_mask[0] = False
    exposed_positions = contract.positions
    exposed_mask = contract.valid_mask
    exposed_positions[0, 0] = 88.0
    exposed_mask[0, 0] = False

    assert contract.positions[0, 0].item() == 0.0
    assert contract.valid_mask[0, 0].item() is True
    with pytest.raises(FrozenInstanceError):
        contract.unit = "calendar_days"

    encoded = json.dumps(contract.to_metadata(), sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["shape"] == [1, 3]
    assert decoded["source"] == "named_known_feature"
    assert decoded["all_invalid_rows_allowed"] is False
