from __future__ import annotations

import pytest
import torch

from layers.TemporalFusion_layers import (
    InterpretableCrossAttention,
    MultiScaleLagAttention,
    PositionalMultiHeadAttention,
    apply_rotary_embedding,
    build_alibi_bias,
)


def _assert_exact(left: torch.Tensor, right: torch.Tensor) -> None:
    torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)


def test_rope_accepts_shared_and_batched_coordinates():
    torch.manual_seed(1)
    query = torch.randn(2, 2, 4, 6)
    key = torch.randn(2, 2, 3, 6)
    query_positions = torch.tensor(
        [[0.0, 1.0, 4.0, 5.0], [0.0, 2.0, 3.0, 9.0]]
    )
    key_positions = torch.tensor([[0.0, 3.0, 8.0], [1.0, 2.0, 7.0]])

    batch_query, batch_key = apply_rotary_embedding(
        query,
        key,
        query_positions=query_positions,
        key_positions=key_positions,
    )

    assert batch_query.shape == query.shape
    assert batch_key.shape == key.shape
    for batch_index in range(2):
        row_query, row_key = apply_rotary_embedding(
            query[batch_index : batch_index + 1],
            key[batch_index : batch_index + 1],
            query_positions=query_positions[batch_index],
            key_positions=key_positions[batch_index],
        )
        _assert_exact(batch_query[batch_index], row_query[0])
        _assert_exact(batch_key[batch_index], row_key[0])


def test_rope_omitted_and_shared_arange_paths_are_exactly_compatible():
    torch.manual_seed(2)
    query = torch.randn(2, 3, 5, 4)
    key = torch.randn(2, 3, 5, 4)

    omitted = apply_rotary_embedding(query, key)
    explicit = apply_rotary_embedding(
        query,
        key,
        query_positions=torch.arange(5),
        key_positions=torch.arange(5),
    )

    _assert_exact(omitted[0], explicit[0])
    _assert_exact(omitted[1], explicit[1])


@pytest.mark.parametrize(
    "positions,error",
    [
        (torch.ones(2, 2, 2), ValueError),
        (torch.ones(3), ValueError),
        (torch.tensor([True, False, True, True]), TypeError),
        (torch.tensor([0.0, 1.0, float('nan'), 3.0]), ValueError),
    ],
)
def test_rope_rejects_invalid_coordinate_contracts(positions, error):
    query = torch.randn(2, 1, 4, 4)
    key = torch.randn(2, 1, 4, 4)
    with pytest.raises(error):
        apply_rotary_embedding(query, key, query_positions=positions)


def test_alibi_accepts_per_sample_coordinates_and_matches_rowwise_calls():
    query_positions = torch.tensor(
        [[0.0, 1.0, 4.0], [0.0, 3.0, 10.0]], dtype=torch.float64
    )
    key_positions = torch.tensor(
        [[0.0, 2.0, 5.0, 9.0], [1.0, 2.0, 8.0, 13.0]], dtype=torch.float64
    )
    bias = build_alibi_bias(
        3,
        3,
        4,
        torch.device("cpu"),
        torch.float32,
        query_positions=query_positions,
        key_positions=key_positions,
    )

    assert bias.shape == (2, 3, 3, 4)
    for batch_index in range(2):
        row = build_alibi_bias(
            3,
            3,
            4,
            torch.device("cpu"),
            torch.float32,
            query_positions=query_positions[batch_index],
            key_positions=key_positions[batch_index],
        )
        _assert_exact(bias[batch_index], row[0])


def test_alibi_shared_and_repeated_batch_coordinates_have_same_values():
    positions = torch.tensor([0.0, 1.0, 4.0, 5.0])
    shared = build_alibi_bias(
        2, 4, 4, torch.device("cpu"), torch.float32,
        query_positions=positions, key_positions=positions,
    )
    batched = build_alibi_bias(
        2, 4, 4, torch.device("cpu"), torch.float32,
        query_positions=positions.expand(3, -1),
        key_positions=positions.expand(3, -1),
    )

    assert shared.shape == (1, 2, 4, 4)
    assert batched.shape == (3, 2, 4, 4)
    _assert_exact(batched, shared.expand(3, -1, -1, -1))


@pytest.mark.parametrize("position_bias_type", ["none", "rope", "alibi"])
def test_positional_attention_masks_invalid_keys_and_queries(position_bias_type):
    torch.manual_seed(3)
    attention = PositionalMultiHeadAttention(
        d_model=8,
        n_heads=2,
        dropout=0.0,
        position_bias_type=position_bias_type,
        attention_backend="exact",
    ).eval()
    query = torch.randn(2, 4, 8)
    key = torch.randn(2, 5, 8)
    value = torch.randn(2, 5, 8)
    query_positions = torch.tensor(
        [[0.0, 1.0, 4.0, 8.0], [0.0, 2.0, 6.0, 7.0]]
    )
    key_positions = torch.tensor(
        [[0.0, 1.0, 3.0, 5.0, 9.0], [0.0, 2.0, 4.0, 8.0, 12.0]]
    )
    query_valid = torch.tensor(
        [[True, False, True, True], [False, True, True, False]]
    )
    key_valid = torch.tensor(
        [[True, False, True, True, False], [False, True, True, False, True]]
    )

    output, weights = attention(
        query,
        key,
        value,
        return_attention=True,
        query_positions=query_positions,
        key_positions=key_positions,
        query_valid_mask=query_valid,
        key_valid_mask=key_valid,
    )

    assert torch.isfinite(output).all()
    assert torch.isfinite(weights).all()
    assert torch.count_nonzero(output.masked_select(~query_valid[:, :, None])) == 0
    assert torch.count_nonzero(
        weights.masked_select(~query_valid[:, None, :, None])
    ) == 0
    assert torch.count_nonzero(
        weights.masked_select(~key_valid[:, None, None, :])
    ) == 0


def test_positional_attention_invalid_key_values_cannot_affect_valid_outputs():
    torch.manual_seed(4)
    attention = PositionalMultiHeadAttention(
        8, 2, dropout=0.0, position_bias_type="alibi"
    ).eval()
    x = torch.randn(2, 5, 8)
    corrupted = x.clone()
    key_valid = torch.tensor(
        [[True, False, True, True, True], [True, True, True, False, True]]
    )
    corrupted[~key_valid] = 1e6
    positions = torch.tensor(
        [[0.0, 1.0, 4.0, 6.0, 7.0], [0.0, 2.0, 3.0, 8.0, 11.0]]
    )

    clean = attention(
        x,
        x,
        x,
        query_positions=positions,
        key_positions=positions,
        key_valid_mask=key_valid,
    )
    changed = attention(
        x,
        corrupted,
        corrupted,
        query_positions=positions,
        key_positions=positions,
        key_valid_mask=key_valid,
    )

    torch.testing.assert_close(clean, changed)


@pytest.mark.parametrize("backend", ["exact", "sdpa"])
def test_positional_attention_all_invalid_keys_and_queries_are_nan_safe(backend):
    torch.manual_seed(5)
    attention = PositionalMultiHeadAttention(
        8, 2, dropout=0.0, attention_backend=backend
    ).eval()
    query = torch.randn(2, 3, 8)
    context = torch.randn(2, 4, 8)
    query_valid = torch.tensor([[True, False, True], [False, False, False]])
    key_valid = torch.tensor(
        [[False, False, False, False], [True, True, True, True]]
    )

    output = attention(
        query,
        context,
        context,
        query_valid_mask=query_valid,
        key_valid_mask=key_valid,
    )

    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[0]) == 0
    assert torch.count_nonzero(output[1]) == 0


def test_positional_attention_omitted_and_all_valid_masks_are_exactly_equal():
    torch.manual_seed(6)
    attention = PositionalMultiHeadAttention(
        8, 2, dropout=0.0, position_bias_type="rope"
    ).eval()
    x = torch.randn(2, 5, 8)
    positions = torch.arange(5, dtype=torch.float32)

    omitted, omitted_weights = attention(
        x, x, x, return_attention=True, query_positions=positions,
        key_positions=positions,
    )
    explicit, explicit_weights = attention(
        x, x, x, return_attention=True, query_positions=positions,
        key_positions=positions,
        query_valid_mask=torch.ones(5, dtype=torch.bool),
        key_valid_mask=torch.ones(2, 5, dtype=torch.bool),
    )

    _assert_exact(omitted, explicit)
    _assert_exact(omitted_weights, explicit_weights)


def test_true_validity_composes_with_legacy_true_padding_mask():
    torch.manual_seed(7)
    attention = PositionalMultiHeadAttention(8, 2, dropout=0.0).eval()
    x = torch.randn(1, 4, 8)
    _, weights = attention(
        x,
        x,
        x,
        return_attention=True,
        key_padding_mask=torch.tensor([[False, True, False, False]]),
        key_valid_mask=torch.tensor([[True, True, False, True]]),
    )

    assert torch.count_nonzero(weights[..., 1]) == 0
    assert torch.count_nonzero(weights[..., 2]) == 0
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones_like(weights[..., 0]))


@pytest.mark.parametrize("position_bias_type", ["none", "rope", "alibi"])
def test_interpretable_cross_attention_honours_batched_coordinates_and_validity(
    position_bias_type,
):
    torch.manual_seed(8)
    attention = InterpretableCrossAttention(
        8, 2, dropout=0.0, position_bias_type=position_bias_type
    ).eval()
    query = torch.randn(2, 3, 8)
    context = torch.randn(2, 5, 8)
    query_valid = torch.tensor([[True, False, True], [False, True, True]])
    key_valid = torch.tensor(
        [[True, False, True, True, False], [False, True, True, False, True]]
    )
    query_positions = torch.tensor([[0.0, 1.0, 4.0], [0.0, 3.0, 7.0]])
    key_positions = torch.tensor(
        [[0.0, 1.0, 2.0, 5.0, 9.0], [0.0, 2.0, 6.0, 8.0, 12.0]]
    )

    output, weights = attention(
        query,
        context,
        return_attention=True,
        query_positions=query_positions,
        key_positions=key_positions,
        query_valid_mask=query_valid,
        key_valid_mask=key_valid,
    )

    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output.masked_select(~query_valid[:, :, None])) == 0
    assert torch.count_nonzero(
        weights.masked_select(~query_valid[:, None, :, None])
    ) == 0
    assert torch.count_nonzero(
        weights.masked_select(~key_valid[:, None, None, :])
    ) == 0


def test_interpretable_cross_attention_all_valid_masks_preserve_exact_output():
    torch.manual_seed(9)
    attention = InterpretableCrossAttention(
        8, 2, dropout=0.0, position_bias_type="alibi"
    ).eval()
    query = torch.randn(2, 3, 8)
    context = torch.randn(2, 4, 8)

    omitted = attention(query, context)
    explicit = attention(
        query,
        context,
        query_valid_mask=torch.ones(3, dtype=torch.bool),
        key_valid_mask=torch.ones(4, dtype=torch.bool),
    )
    _assert_exact(omitted, explicit)


def test_multiscale_lag_attention_shifts_irregular_coordinates_and_validity():
    torch.manual_seed(10)
    lag_attention = MultiScaleLagAttention(
        8,
        2,
        lag_scales=[1, 2],
        dropout=0.0,
        position_bias_type="alibi",
    ).eval()
    x = torch.randn(2, 6, 8)
    positions = torch.tensor(
        [[0.0, 1.0, 4.0, 5.0, 9.0, 10.0], [0.0, 3.0, 4.0, 8.0, 12.0, 13.0]]
    )
    valid = torch.tensor(
        [[True, False, True, True, False, True], [False, True, True, False, True, True]]
    )

    output, payload = lag_attention(
        x,
        return_attention=True,
        positions=positions,
        valid_mask=valid,
    )

    assert payload["lag_key_positions"].shape == (2, 2, 6)
    assert payload["lag_key_valid_masks"].shape == (2, 2, 6)
    assert torch.equal(payload["lag_query_valid_mask"], valid)
    assert torch.count_nonzero(output.masked_select(~valid[:, :, None])) == 0
    for branch_index, lag in enumerate((1, 2)):
        expected_key_valid = torch.zeros_like(valid)
        expected_key_valid[:, lag:] = valid[:, :-lag]
        assert torch.equal(
            payload["lag_key_valid_masks"][branch_index], expected_key_valid
        )
        _assert_exact(
            payload["lag_key_positions"][branch_index, :, lag:],
            positions[:, :-lag],
        )
        weights = payload["lag_attention"][..., branch_index]
        assert torch.count_nonzero(
            weights.masked_select(~valid[:, None, :, None])
        ) == 0
        assert torch.count_nonzero(
            weights.masked_select(~expected_key_valid[:, None, None, :])
        ) == 0


def test_lag_position_arithmetic_is_versioned_between_v1_replay_and_v2_physical_sources():
    torch.manual_seed(101)
    legacy = MultiScaleLagAttention(
        8,
        2,
        lag_scales=[2],
        dropout=0.0,
        position_bias_type="alibi",
        extension_semantics_version=1,
    ).eval()
    current = MultiScaleLagAttention(
        8,
        2,
        lag_scales=[2],
        dropout=0.0,
        position_bias_type="alibi",
        extension_semantics_version=2,
    ).eval()
    current.load_state_dict(legacy.state_dict(), strict=True)
    x = torch.randn(2, 6, 8)
    positions = torch.tensor(
        [[0.0, 1.0, 4.0, 8.0, 9.0, 15.0], [2.0, 5.0, 6.0, 12.0, 20.0, 21.0]]
    )

    with torch.no_grad():
        _, legacy_payload = legacy(x, return_attention=True, positions=positions)
        _, current_payload = current(x, return_attention=True, positions=positions)

    # Frozen v1 subtracted the integer lag from every coordinate.  V2 maps
    # every usable shifted key back to the actual irregular source token.
    _assert_exact(legacy_payload["lag_key_positions"][0], positions - 2.0)
    _assert_exact(
        current_payload["lag_key_positions"][0, :, 2:], positions[:, :-2]
    )
    assert not torch.equal(
        legacy_payload["lag_key_positions"],
        current_payload["lag_key_positions"],
    )


def test_multiscale_lag_attention_omitted_shared_and_all_valid_paths_match():
    torch.manual_seed(11)
    lag_attention = MultiScaleLagAttention(
        8, 2, lag_scales=[1, 2], dropout=0.0, position_bias_type="alibi"
    ).eval()
    x = torch.randn(2, 6, 8)
    positions = torch.arange(6, dtype=torch.float32)

    omitted = lag_attention(x)
    shared = lag_attention(x, positions=positions)
    explicit_valid = lag_attention(
        x, positions=positions, valid_mask=torch.ones(6, dtype=torch.bool)
    )

    _assert_exact(omitted, shared)
    _assert_exact(shared, explicit_valid)


@pytest.mark.parametrize(
    "mask,error",
    [
        (torch.ones(4), TypeError),
        (torch.ones(3, dtype=torch.bool), ValueError),
        (torch.ones(3, 4, dtype=torch.bool), ValueError),
    ],
)
def test_attention_rejects_invalid_true_valid_masks(mask, error):
    attention = PositionalMultiHeadAttention(8, 2, dropout=0.0)
    x = torch.randn(2, 4, 8)
    with pytest.raises(error):
        attention(x, x, x, query_valid_mask=mask)
