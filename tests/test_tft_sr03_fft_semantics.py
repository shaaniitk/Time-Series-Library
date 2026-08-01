"""Semantic-v2 contracts for the native TFT spectral extension.

These tests deliberately use the public names introduced by TFT-SR03.  The
old ``low``, ``top_amplitude``, and ``learned`` operator remains available only
inside semantics-v1 replay; semantics v2 describes hard selection and soft
all-bin filtering with different, truthful names.
"""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
import torch

from layers.TemporalFusion_layers import SpectralBranch
from models.TemporalFusionTransformer import Model as NativeTFT
from run import build_parser
from utils.tft_config import (
    TFT_EXTENSION_MIGRATION_CAPABILITIES,
    apply_tft_profile,
    pending_v2_artifact_extensions,
    validate_tft_v2_artifact_readiness,
)


CANONICAL_MODES = ("low_k", "top_amplitude_k", "learned_filter")


def _raw_config(**overrides):
    values = {
        "model": "TemporalFusionTransformer",
        "task_name": "long_term_forecast",
        "model_id": "tft-sr03-fft-contract",
        "data": "ETTh1",
        "features": "MS",
        "seq_len": 8,
        "label_len": 4,
        "pred_len": 3,
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 1,
        "d_model": 8,
        "n_heads": 2,
        "e_layers": 1,
        "d_layers": 1,
        "d_ff": 2048,
        "dropout": 0.0,
        "embed": "timeF",
        "freq": "h",
        "tft_profile": "extended_safe",
        "tft_extension_semantics_version": 2,
        "tft_temporal_backbone": "lstm",
        "tft_temporal_backbone_layers": 1,
        "tft_target_pos": [6],
        "tft_full_attention": False,
        "tft_attention_dropout": 0.0,
        "tft_use_fft_branch": True,
        "tft_fft_modes": 3,
        "tft_fft_mode_select": "low_k",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _resolved_config(**overrides):
    return apply_tft_profile(_raw_config(**overrides))


def _inputs():
    generator = torch.Generator().manual_seed(17011993)
    return (
        torch.randn(2, 8, 7, generator=generator),
        torch.randn(2, 8, 4, generator=generator),
        torch.randn(2, 7, 1, generator=generator),
        torch.randn(2, 7, 4, generator=generator),
    )


def _copy_shared_reference_state(reference, variant):
    reference_state = reference.state_dict()
    variant_state = variant.state_dict()
    missing_or_changed = {
        name
        for name, tensor in reference_state.items()
        if name not in variant_state
        or variant_state[name].shape != tensor.shape
        or variant_state[name].dtype != tensor.dtype
    }
    assert missing_or_changed == set()
    with torch.no_grad():
        for name, tensor in reference_state.items():
            variant_state[name].copy_(tensor)
    variant.load_state_dict(variant_state, strict=True)


def _assert_nested_exact(left, right):
    assert type(left) is type(right)
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            _assert_nested_exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_exact(left_item, right_item)
    else:
        assert left == right


def _multi_sine(frequencies, *, length=32):
    """Build ``[B,T,D]`` fixtures from one or more integer bins per channel."""

    time = torch.arange(length, dtype=torch.float32)
    samples = []
    for sample_frequencies in frequencies:
        channels = []
        for channel_frequencies in sample_frequencies:
            signal = sum(
                (1.0 / (index + 1))
                * torch.sin(2.0 * torch.pi * frequency * time / length)
                for index, frequency in enumerate(channel_frequencies)
            )
            channels.append(signal)
        samples.append(torch.stack(channels, dim=-1))
    return torch.stack(samples, dim=0)


@pytest.mark.parametrize("mode", CANONICAL_MODES)
def test_v2_accepts_only_truthfully_named_canonical_fft_operations(mode):
    args = _resolved_config(tft_fft_mode_select=mode)
    model = NativeTFT(args)

    assert args.tft_fft_mode_select == mode
    assert model.temporal_fusion_decoder.layers[0].fft_branch.mode_select == mode


def test_v2_learned_alias_warns_and_has_the_canonical_digest():
    canonical = _resolved_config(tft_fft_mode_select="learned_filter")
    with pytest.warns(DeprecationWarning, match="learned_filter"):
        alias = _resolved_config(tft_fft_mode_select="learned")

    assert alias.tft_fft_mode_select == "learned_filter"
    assert alias.tft_config_digest == canonical.tft_config_digest


def test_cli_exposes_canonical_modes_and_truthful_fft_modes_help():
    parser = build_parser()
    help_text = parser.format_help()
    required = [
        "--task_name", "long_term_forecast",
        "--is_training", "0",
        "--model_id", "sr03-cli",
        "--model", "TemporalFusionTransformer",
        "--data", "ETTh1",
    ]

    for mode in CANONICAL_MODES:
        parsed = parser.parse_args(required + ["--tft_fft_mode_select", mode])
        assert parsed.tft_fft_mode_select == mode
    assert "retained bins" in help_text
    assert "hard-selection" in help_text
    assert "spectral control points" in help_text
    assert "learned_filter" in help_text


def test_semantics_v1_preserves_legacy_mode_and_concat_scope():
    # This is an affirmative replay test, not merely a v2 rejection test.  The
    # legacy graph keeps its old token, one FFT over history+future, and its old
    # diagnostic names so frozen checkpoints remain interpretable as v1.
    model = NativeTFT(
        _raw_config(
            tft_extension_semantics_version=1,
            tft_fft_mode_select="learned",
        )
    ).eval()
    branch = model.temporal_fusion_decoder.layers[0].fft_branch
    observed_lengths = []
    original_forward = branch.forward

    def recording_forward(value, *args, **kwargs):
        observed_lengths.append(value.shape[1])
        return original_forward(value, *args, **kwargs)

    branch.forward = recording_forward
    with torch.no_grad():
        payload = model(*_inputs(), return_interpretation=True)

    assert model.configs.tft_fft_mode_select == "learned"
    assert branch.mode_select == "learned"
    assert observed_lengths == [model.seq_len + model.pred_len]
    assert "fft_gate_mean" in payload
    assert "fft_learned_mask_mean" in payload
    assert "fft_learned_mask_std" in payload
    assert "fft_learned_mask_peak_bin_mean" in payload
    assert "fft_diagnostics" not in payload


def test_low_k_has_exactly_k_active_bins_when_k_is_available():
    branch = SpectralBranch(
        d_model=3, modes=4, mode_select="low_k", dropout=0.0
    )
    values = torch.randn(2, 32, 3)
    spectrum = torch.fft.rfft(values.permute(0, 2, 1), dim=-1)

    selected = branch._select_modes(spectrum, spectrum.shape[-1])

    assert selected.shape == (4,)
    assert torch.equal(selected.cpu(), torch.tensor([0, 1, 2, 3]))


def test_top_amplitude_k_recovers_each_single_sine_bin_exactly():
    expected = torch.tensor([[3, 11], [7, 13]])
    signal = _multi_sine(
        [
            [[3], [11]],
            [[7], [13]],
        ],
        length=32,
    )
    branch = SpectralBranch(
        d_model=2, modes=1, mode_select="top_amplitude_k", dropout=0.0
    )
    spectrum = torch.fft.rfft(signal.permute(0, 2, 1), dim=-1)

    selected = branch._select_modes(spectrum, spectrum.shape[-1])

    assert selected.shape == (2, 2, 1)
    assert torch.equal(selected.squeeze(-1).cpu(), expected)


def test_top_amplitude_k_selects_exactly_k_unique_bins_per_sample_channel():
    signal = _multi_sine(
        [
            [[2, 9], [4, 12]],
            [[3, 10], [6, 14]],
        ],
        length=32,
    )
    branch = SpectralBranch(
        d_model=2, modes=2, mode_select="top_amplitude_k", dropout=0.0
    )
    spectrum = torch.fft.rfft(signal.permute(0, 2, 1), dim=-1)

    selected = branch._select_modes(spectrum, spectrum.shape[-1])

    assert selected.shape == (2, 2, 2)
    assert torch.all(selected.sort(dim=-1).values.diff(dim=-1) > 0)
    expected_sets = (
        ({2, 9}, {4, 12}),
        ({3, 10}, {6, 14}),
    )
    for batch_index in range(2):
        for channel_index in range(2):
            assert set(selected[batch_index, channel_index].tolist()) == (
                expected_sets[batch_index][channel_index]
            )


def test_hard_selection_truthfully_reports_requested_available_and_clamped_k():
    # A request cannot activate more than rFFT makes available.  Clamping is
    # permitted only when it is explicit in diagnostics rather than silently
    # claiming that the requested number of bins was retained.
    branch = SpectralBranch(
        d_model=2, modes=20, mode_select="low_k", dropout=0.0
    ).eval()
    with torch.no_grad():
        _, diagnostics = branch(
            torch.randn(1, 16, 2), return_diagnostics=True, scope="history"
        )

    assert diagnostics["requested_bin_count"] == 20
    assert diagnostics["available_bin_count"] == 9
    assert diagnostics["active_bin_count"] == 9
    assert diagnostics["selection_was_clamped"] is True
    assert diagnostics["selected_bins"].numel() == 9


def test_dc_and_nyquist_are_eligible_and_reported_for_even_runtime_length():
    length = 32
    time = torch.arange(length, dtype=torch.float32)
    # DC and the alternating-sign Nyquist component are the two nonzero bins.
    values = (2.0 + 3.0 * torch.pow(-1.0, time)).view(1, length, 1)
    branch = SpectralBranch(
        d_model=1, modes=2, mode_select="top_amplitude_k", dropout=0.0
    ).eval()
    spectrum = torch.fft.rfft(values.permute(0, 2, 1), dim=-1)

    selected = branch._select_modes(spectrum, spectrum.shape[-1])
    with torch.no_grad():
        _, diagnostics = branch(
            values, return_diagnostics=True, scope="known_future"
        )

    assert set(selected[0, 0].tolist()) == {0, length // 2}
    assert set(diagnostics["selected_bins"][0, 0].tolist()) == {
        0,
        length // 2,
    }
    assert diagnostics["dc_bin"] == 0
    assert diagnostics["nyquist_bin"] == length // 2
    assert diagnostics["dc_policy"] == (
        "eligible_self_conjugate_real_response"
    )
    assert diagnostics["nyquist_policy"] == (
        "eligible_self_conjugate_real_response"
    )
    assert diagnostics["imaginary_endpoint_policy"] == (
        "irfft_discards_imaginary_dc_and_nyquist"
    )


def test_odd_runtime_reports_that_no_nyquist_bin_exists():
    branch = SpectralBranch(
        d_model=2, modes=3, mode_select="low_k", dropout=0.0
    ).eval()

    with torch.no_grad():
        _, diagnostics = branch(
            torch.randn(1, 31, 2),
            return_diagnostics=True,
            scope="history",
        )

    assert diagnostics["available_bin_count"] == 16
    assert diagnostics["dc_bin"] == 0
    assert diagnostics["nyquist_bin"] is None
    assert diagnostics["nyquist_policy"] == (
        "absent_for_odd_sequence_length"
    )


def test_learned_filter_reports_all_bin_filtering_and_physical_grid_truthfully():
    branch = SpectralBranch(
        d_model=3, modes=5, mode_select="learned_filter", dropout=0.0
    ).eval()
    with torch.no_grad():
        _, diagnostics = branch(
            torch.randn(2, 32, 3), return_diagnostics=True, scope="history"
        )

    assert diagnostics["mode"] == "learned_filter"
    assert diagnostics["operation"] == "soft_all_bin_filter"
    assert diagnostics["scope"] == "history"
    assert diagnostics["available_bin_count"] == 17
    assert diagnostics["active_bin_count"] == 17
    assert diagnostics["selected_bins"] is None
    assert diagnostics["peak_bins"].shape == (3,)
    assert diagnostics["mask_entropy_per_channel"].shape == (3,)
    assert diagnostics["effective_active_bin_count_per_channel"].shape == (3,)
    assert diagnostics["filter_norm_per_channel"].shape == (3,)
    assert torch.isfinite(diagnostics["mask_entropy_per_channel"]).all()
    assert torch.isfinite(diagnostics["filter_norm_per_channel"]).all()

    normalized = diagnostics["normalized_frequency_grid"]
    periods = diagnostics["equivalent_token_period_grid"]
    torch.testing.assert_close(
        normalized,
        torch.arange(17, dtype=normalized.dtype, device=normalized.device) / 32.0,
    )
    assert torch.isinf(periods[0])
    torch.testing.assert_close(periods[1:], 1.0 / normalized[1:])
    assert diagnostics["period_unit"] == "tokens"
    assert diagnostics["physical_period_claim"] is False


def test_learned_filter_all_parameters_receive_finite_nonzero_gradients():
    torch.manual_seed(7)
    branch = SpectralBranch(
        d_model=3, modes=5, mode_select="learned_filter", dropout=0.0
    )
    values = torch.randn(2, 24, 3, requires_grad=True)
    weights = torch.randn(2, 24, 3)

    output = branch(values)
    (output * weights).sum().backward()

    assert values.grad is not None
    assert torch.isfinite(values.grad).all()
    for name, parameter in branch.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad).item() > 0, name


def test_learned_filter_runtime_interpolation_and_strict_reload_are_stable():
    torch.manual_seed(8)
    branch = SpectralBranch(
        d_model=2, modes=4, mode_select="learned_filter", dropout=0.0
    ).eval()
    short = torch.randn(2, 16, 2)
    long = torch.randn(2, 30, 2)

    with torch.no_grad():
        short_output, short_diagnostics = branch(
            short, return_diagnostics=True, scope="history"
        )
        long_output, long_diagnostics = branch(
            long, return_diagnostics=True, scope="history"
        )

    buffer = io.BytesIO()
    torch.save(branch.state_dict(), buffer)
    buffer.seek(0)
    state = torch.load(buffer, map_location="cpu", weights_only=True)
    restored = SpectralBranch(
        d_model=2, modes=4, mode_select="learned_filter", dropout=0.0
    ).eval()
    incompatibilities = restored.load_state_dict(state, strict=True)
    with torch.no_grad():
        restored_short, restored_short_diagnostics = restored(
            short, return_diagnostics=True, scope="history"
        )
        restored_long, restored_long_diagnostics = restored(
            long, return_diagnostics=True, scope="history"
        )

    assert incompatibilities.missing_keys == []
    assert incompatibilities.unexpected_keys == []
    assert short_diagnostics["available_bin_count"] == 9
    assert long_diagnostics["available_bin_count"] == 16
    _assert_nested_exact(restored_short_diagnostics, short_diagnostics)
    _assert_nested_exact(restored_long_diagnostics, long_diagnostics)
    assert torch.equal(restored_short, short_output)
    assert torch.equal(restored_long, long_output)


def test_v2_model_applies_fft_to_history_and_known_future_separately():
    torch.manual_seed(9)
    model = NativeTFT(
        _raw_config(tft_fft_mode_select="low_k")
    ).eval()
    branch = model.temporal_fusion_decoder.layers[0].fft_branch
    calls = []
    original_forward = branch.forward

    def recording_forward(value, *args, **kwargs):
        calls.append((value.shape[1], value.detach().clone(), kwargs.get("scope")))
        return original_forward(value, *args, **kwargs)

    branch.forward = recording_forward
    first_inputs = _inputs()
    changed_inputs = tuple(value.clone() for value in first_inputs)
    changed_inputs[3][:, -model.pred_len :, :] += 100.0

    with torch.no_grad():
        payload = model(*first_inputs, return_interpretation=True)
        first_calls = calls[:]
        calls.clear()
        model(*changed_inputs)
        changed_calls = calls[:]

    assert [item[0] for item in first_calls] == [model.seq_len, model.pred_len]
    assert [item[2] for item in first_calls] == ["history", "known_future"]
    assert [item[0] for item in changed_calls] == [model.seq_len, model.pred_len]
    assert torch.equal(first_calls[0][1], changed_calls[0][1])
    assert not torch.equal(first_calls[1][1], changed_calls[1][1])

    diagnostics = payload["fft_diagnostics"]
    assert diagnostics["scope"] == "separate_history_future"
    assert diagnostics["mode"] == "low_k"
    assert diagnostics["history"]["sequence_length"] == model.seq_len
    assert diagnostics["known_future"]["sequence_length"] == model.pred_len
    assert diagnostics["residual_contribution_rms"] == 0.0
    assert "temporal_path_weight_mean" in diagnostics
    if "fft_gate_mean" in payload:
        assert payload["fft_gate_mean"] == pytest.approx(
            diagnostics["temporal_path_weight_mean"]
        )


def test_canonical_fft_mode_retains_sr02_exact_neutral_model_parity():
    torch.manual_seed(10)
    reference = NativeTFT(
        _raw_config(tft_use_fft_branch=False, tft_fft_mode_select="low_k")
    ).eval()
    torch.manual_seed(11)
    variant = NativeTFT(
        _raw_config(tft_fft_mode_select="low_k")
    ).eval()
    _copy_shared_reference_state(reference, variant)

    with torch.no_grad():
        reference_output = reference(*_inputs())
        variant_output = variant(*_inputs())

    assert torch.equal(variant_output, reference_output)


def test_sr03_releases_only_the_repaired_fft_operator_for_v2_artifacts():
    args = _resolved_config(tft_fft_mode_select="learned_filter")

    assert (
        TFT_EXTENSION_MIGRATION_CAPABILITIES["fft_branch"]["v2_artifact_status"]
        == "released"
    )
    assert pending_v2_artifact_extensions(args) == []
    assert validate_tft_v2_artifact_readiness(args) is args
