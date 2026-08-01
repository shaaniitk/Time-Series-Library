"""Model-level contracts for TFT-SR02 neutral extension integration.

These tests intentionally compare complete native TFT models, rather than only
the residual adapter primitive.  A semantics-v2 extension in ``neutral`` mode
must be a bitwise no-op after the common state is paired with an all-off
reference model.  This includes the loss, input gradients, and every shared
parameter gradient when the optional branch itself is stochastic.
"""

from __future__ import annotations

import hashlib
import io
from types import SimpleNamespace

import pytest
import torch

from layers.TemporalFusion_layers import (
    ExtensionResidualAdapter,
    temporarily_zero_extensions,
)
from models.TemporalFusionTransformer import Model


EXTENSION_CASES = (
    (
        "fft_branch",
        {"tft_use_fft_branch": True, "tft_fft_modes": 4},
    ),
    (
        "explicit_cross_attention",
        {"tft_use_explicit_cross_attention": True},
    ),
    (
        "lag_attention",
        {"tft_use_lag_attention": True, "tft_lag_scales": [1, 2]},
    ),
    (
        "higher_order_interaction",
        {"tft_use_higher_order": True},
    ),
    (
        "graph_cross_mixing",
        {"tft_cross_variable_mixing": True},
    ),
    (
        "covariate_reattention",
        {"tft_covariate_reattention": True},
    ),
    (
        "regime_moe",
        {
            "tft_use_regime_moe": True,
            "tft_num_moe_experts": 2,
            "tft_num_regimes": 2,
            "tft_moe_top_k": 1,
        },
    ),
    (
        "dual_attention_fusion",
        {"tft_dual_attention_fusion": True},
    ),
    (
        "vsn_residual_bypass",
        {"tft_vsn_residual_bypass": True},
    ),
    (
        "temporal_compression",
        {
            "tft_use_temporal_compression": True,
            "tft_tc_stride": 2,
            # Force the compact fixture through the active compression path.
            "tft_tc_threshold": 1,
        },
    ),
)

ALL_EXTENSION_OVERRIDES = {
    key: value
    for _, overrides in EXTENSION_CASES
    for key, value in overrides.items()
}


def _config(**overrides):
    values = {
        "model": "TemporalFusionTransformer",
        "task_name": "long_term_forecast",
        "model_id": "tft-sr02-model-contract",
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
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _inputs(*, requires_grad=False):
    generator = torch.Generator().manual_seed(31051995)
    x_enc = torch.randn(2, 8, 7, generator=generator)
    x_mark_enc = torch.randn(2, 8, 4, generator=generator)
    x_dec = torch.randn(2, 7, 1, generator=generator)
    x_mark_dec = torch.randn(2, 7, 4, generator=generator)
    if requires_grad:
        x_enc.requires_grad_(True)
    return x_enc, x_mark_enc, x_dec, x_mark_dec


def _paired_models(
    variant_overrides,
    *,
    dropout=0.0,
    common_overrides=None,
):
    common_overrides = {} if common_overrides is None else dict(common_overrides)
    torch.manual_seed(1701)
    reference = Model(
        _config(dropout=dropout, **common_overrides)
    ).train(dropout > 0.0)
    torch.manual_seed(8675309)
    variant = Model(
        _config(dropout=dropout, **common_overrides, **variant_overrides)
    ).train(dropout > 0.0)
    _copy_reference_state(reference, variant)
    return reference, variant


def _copy_reference_state(reference, variant):
    """Copy every reference tensor into its name/shape-compatible peer.

    The subset assertion is important: exact neutral comparisons are invalid
    if enabling an extension silently removes or renames part of the base path.
    """

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


def _adapters(model):
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, ExtensionResidualAdapter)
    }


def _assert_optional_grad_equal(reference_grad, variant_grad):
    if reference_grad is None or variant_grad is None:
        assert reference_grad is None and variant_grad is None
    else:
        assert torch.equal(reference_grad, variant_grad)


@pytest.mark.parametrize(
    ("extension_name", "variant_overrides"),
    EXTENSION_CASES,
    ids=[case[0] for case in EXTENSION_CASES],
)
def test_each_neutral_extension_is_exactly_the_all_off_model(
    extension_name,
    variant_overrides,
):
    reference, variant = _paired_models(variant_overrides)
    reference.eval()
    variant.eval()

    assert reference.configs.tft_resolved_extension_modes[extension_name] == "off"
    assert variant.configs.tft_resolved_extension_modes[extension_name] == "neutral"
    assert _adapters(reference) == {}
    matching_adapters = [
        adapter
        for adapter in _adapters(variant).values()
        if adapter.extension_name == extension_name
    ]
    assert matching_adapters
    assert all(adapter.is_effectively_zero() for adapter in matching_adapters)

    inputs = _inputs()
    with torch.no_grad():
        reference_output = reference(*inputs)
        variant_output = variant(*inputs)

    assert torch.equal(variant_output, reference_output)


def test_stochastic_neutral_branches_preserve_loss_input_and_shared_gradients():
    # Both branches contain dropout-capable paths.  Their lazy neutral adapter
    # calls must restore RNG before the next shared operation is evaluated.
    reference, variant = _paired_models(
        {
            "tft_use_fft_branch": True,
            "tft_fft_modes": 4,
            "tft_use_explicit_cross_attention": True,
        },
        dropout=0.25,
        common_overrides={"tft_attention_dropout": 0.2},
    )
    reference_inputs = _inputs(requires_grad=True)
    variant_inputs = tuple(
        tensor.detach().clone().requires_grad_(index == 0)
        for index, tensor in enumerate(reference_inputs)
    )
    target_generator = torch.Generator().manual_seed(271828)
    target = torch.randn(2, 11, 1, generator=target_generator)

    torch.manual_seed(424242)
    reference_output = reference(*reference_inputs)
    reference_loss = (reference_output - target).square().mean()
    reference_loss.backward()

    torch.manual_seed(424242)
    variant_output = variant(*variant_inputs)
    variant_loss = (variant_output - target).square().mean()
    variant_loss.backward()

    assert torch.equal(variant_output, reference_output)
    assert torch.equal(variant_loss, reference_loss)
    assert torch.equal(variant_inputs[0].grad, reference_inputs[0].grad)

    variant_parameters = dict(variant.named_parameters())
    for name, reference_parameter in reference.named_parameters():
        assert name in variant_parameters
        _assert_optional_grad_equal(
            reference_parameter.grad,
            variant_parameters[name].grad,
        )


def test_neutral_strength_learns_before_fft_branch_parameters():
    torch.manual_seed(314159)
    model = Model(
        _config(tft_use_fft_branch=True, tft_fft_modes=4)
    ).train()
    adapter = model.temporal_fusion_decoder.layers[0].fft_residual_adapter
    branch_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if ".fft_branch." in name or ".fft_fusion_gate." in name
    ]
    assert branch_parameters

    inputs = _inputs(requires_grad=True)
    target_generator = torch.Generator().manual_seed(161803)
    target = torch.randn(2, 11, 1, generator=target_generator)
    first_loss = (model(*inputs) - target).square().mean()
    first_loss.backward()

    assert adapter.residual_strength.item() == 0.0
    assert adapter.residual_strength.grad is not None
    assert torch.isfinite(adapter.residual_strength.grad)
    assert adapter.residual_strength.grad.item() != 0.0
    assert all(parameter.grad is not None for parameter in branch_parameters)
    assert all(torch.count_nonzero(parameter.grad).item() == 0 for parameter in branch_parameters)

    strength_optimizer = torch.optim.SGD([adapter.residual_strength], lr=0.1)
    strength_optimizer.step()
    assert adapter.residual_strength.item() != 0.0

    model.zero_grad(set_to_none=True)
    inputs[0].grad = None
    second_loss = (model(*inputs) - target).square().mean()
    second_loss.backward()

    assert any(
        parameter.grad is not None
        and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in branch_parameters
    )


def test_model_interpretation_reports_exact_neutral_residual_diagnostics():
    torch.manual_seed(112358)
    model = Model(
        _config(tft_use_fft_branch=True, tft_fft_modes=4)
    ).eval()

    with torch.no_grad():
        payload = model(*_inputs(), return_interpretation=True)

    diagnostics = payload["extension_residuals"]["fft_branch"]
    assert set(diagnostics) == {
        "raw_residual_strength",
        "effective_residual_strength",
        "residual_strength",
        "base_rms",
        "delta_rms",
        "combined_minus_base_rms",
    }
    assert torch.equal(
        diagnostics["raw_residual_strength"],
        torch.zeros_like(diagnostics["raw_residual_strength"]),
    )
    assert torch.equal(
        diagnostics["effective_residual_strength"],
        torch.zeros_like(diagnostics["effective_residual_strength"]),
    )
    assert diagnostics["base_rms"].item() > 0.0
    assert diagnostics["delta_rms"].item() > 0.0
    assert diagnostics["combined_minus_base_rms"].item() == 0.0

    assert payload["temporal_coordinate_metadata"]["unit"] == "steps"
    assert payload["temporal_coordinate_metadata"]["source"] == "row_index"
    assert payload["temporal_positions"].shape == (2, 11)
    assert payload["temporal_valid_mask"].all()


def test_model_knockout_restores_the_exact_all_off_counterfactual():
    reference, variant = _paired_models(
        {"tft_use_fft_branch": True, "tft_fft_modes": 4}
    )
    reference.eval()
    variant.eval()
    adapter = variant.temporal_fusion_decoder.layers[0].fft_residual_adapter
    with torch.no_grad():
        adapter.residual_strength.fill_(0.35)
    state_before = {
        name: tensor.detach().clone()
        for name, tensor in variant.state_dict().items()
    }
    inputs = _inputs()

    with torch.no_grad():
        reference_output = reference(*inputs)
        active_output = variant(*inputs)
        with temporarily_zero_extensions(variant, "fft_branch"):
            knockout_output = variant(*inputs)
        restored_output = variant(*inputs)

    assert not torch.equal(active_output, reference_output)
    assert torch.equal(knockout_output, reference_output)
    assert torch.equal(restored_output, active_output)
    assert adapter.residual_strength.item() == pytest.approx(0.35)
    for name, tensor in variant.state_dict().items():
        assert torch.equal(tensor, state_before[name])


@pytest.mark.parametrize(
    ("extension_name", "overrides"),
    [
        ("graph_cross_mixing", {"tft_cross_variable_mixing": True}),
        ("vsn_residual_bypass", {"tft_vsn_residual_bypass": True}),
    ],
)
def test_model_semantic_knockout_selects_every_vsn_adapter_for_one_extension(
    extension_name,
    overrides,
):
    model = Model(_config(**overrides)).eval()
    matching = [
        adapter
        for adapter in _adapters(model).values()
        if adapter.extension_name == extension_name
    ]
    # History and future VSNs are separate modules implementing one semantic
    # extension. A named counterfactual must disable the whole extension.
    assert len(matching) >= 2
    with torch.no_grad():
        for adapter in matching:
            adapter.residual_strength.fill_(0.25)

    with temporarily_zero_extensions(model, extension_name):
        assert all(adapter.is_temporarily_zeroed for adapter in matching)
        assert all(adapter.is_effectively_zero() for adapter in matching)

    assert all(not adapter.is_temporarily_zeroed for adapter in matching)
    assert all(adapter.residual_strength.item() == pytest.approx(0.25) for adapter in matching)


def test_dual_attention_strength_one_retains_base_and_adds_supplementary_path():
    torch.manual_seed(9091)
    model = Model(_config(tft_dual_attention_fusion=True)).eval()
    layer = model.temporal_fusion_decoder.layers[0]
    with torch.no_grad():
        layer.dual_attention_residual_adapter.residual_strength.fill_(1.0)

    captured = {}

    def adapter_pre_hook(_module, args):
        captured["base"] = args[0].detach().clone()

    def supplementary_hook(_module, _args, output):
        captured["supplementary"] = output.detach().clone()

    def adapter_hook(_module, _args, output):
        captured["combined"] = output[0].detach().clone()

    handles = (
        layer.dual_attention_residual_adapter.register_forward_pre_hook(
            adapter_pre_hook
        ),
        layer.dual_attention_module.register_forward_hook(supplementary_hook),
        layer.dual_attention_residual_adapter.register_forward_hook(adapter_hook),
    )
    try:
        with torch.no_grad():
            model(*_inputs())
    finally:
        for handle in handles:
            handle.remove()

    assert torch.equal(
        captured["combined"],
        captured["base"] + captured["supplementary"],
    )


def test_regime_moe_strength_one_retains_base_ff_and_adds_expert_residual():
    torch.manual_seed(9092)
    model = Model(
        _config(
            tft_use_regime_moe=True,
            tft_num_moe_experts=2,
            tft_num_regimes=2,
            tft_moe_top_k=1,
        )
    ).eval()
    layer = model.temporal_fusion_decoder.layers[0]
    with torch.no_grad():
        layer.regime_moe_residual_adapter.residual_strength.fill_(1.0)

    captured = {}

    def adapter_pre_hook(_module, args):
        captured["base_ff"] = args[0].detach().clone()

    def moe_pre_hook(_module, args):
        captured["moe_input"] = args[0].detach().clone()

    def moe_hook(_module, _args, output):
        captured["moe_output"] = output[0].detach().clone()

    def adapter_hook(_module, _args, output):
        captured["combined"] = output[0].detach().clone()

    handles = (
        layer.regime_moe_residual_adapter.register_forward_pre_hook(
            adapter_pre_hook
        ),
        layer.regime_moe.register_forward_pre_hook(moe_pre_hook),
        layer.regime_moe.register_forward_hook(moe_hook),
        layer.regime_moe_residual_adapter.register_forward_hook(adapter_hook),
    )
    try:
        with torch.no_grad():
            model(*_inputs())
    finally:
        for handle in handles:
            handle.remove()

    expected = captured["base_ff"] + (
        captured["moe_output"] - captured["moe_input"]
    )
    assert torch.equal(captured["combined"], expected)


def test_all_extension_adapter_state_has_a_strict_exact_round_trip():
    torch.manual_seed(12345)
    model = Model(_config(**ALL_EXTENSION_OVERRIDES)).eval()
    adapters = _adapters(model)
    assert len(adapters) == 12
    with torch.no_grad():
        for index, adapter in enumerate(adapters.values(), start=1):
            adapter.residual_strength.fill_(index / 100.0)

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    serialized_state = torch.load(buffer, map_location="cpu", weights_only=True)

    torch.manual_seed(54321)
    restored = Model(_config(**ALL_EXTENSION_OVERRIDES)).eval()
    incompatibilities = restored.load_state_dict(serialized_state, strict=True)
    assert incompatibilities.missing_keys == []
    assert incompatibilities.unexpected_keys == []
    assert set(restored.state_dict()) == set(model.state_dict())
    for name, tensor in model.state_dict().items():
        assert torch.equal(restored.state_dict()[name], tensor)

    inputs = _inputs()
    with torch.no_grad():
        assert torch.equal(restored(*inputs), model(*inputs))


def test_semantics_v1_keeps_the_frozen_adapter_free_state_topology():
    torch.manual_seed(1701)
    legacy = Model(
        _config(tft_extension_semantics_version=1)
    )
    assert _adapters(legacy) == {}
    assert all("residual_adapter" not in name for name in legacy.state_dict())

    topology_material = "\n".join(
        f"{name}:{tuple(tensor.shape)}:{tensor.dtype}"
        for name, tensor in legacy.state_dict().items()
    )
    topology_sha256 = hashlib.sha256(topology_material.encode("utf-8")).hexdigest()

    # A compact guard for the immutable semantics-v1 graph used by the legacy
    # replay matrix.  Values are intentionally excluded; names, shapes and
    # dtypes are the checkpoint compatibility surface protected here.
    assert topology_sha256 == (
        "6eb7cbcfd4b04c1d2932344f8ecf5feefe319c92da0f826a4411fbefd42283a8"
    )
    assert len(legacy.state_dict()) == 280
    assert sum(parameter.numel() for parameter in legacy.parameters()) == 14222
