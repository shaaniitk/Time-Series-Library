from __future__ import annotations

from types import SimpleNamespace

import torch

from layers.TemporalFusion_layers import (
    HigherOrderInteractionBlock,
    NamedCovariateInteractionEncoder,
)
from models.TemporalFusionTransformer import Model as NativeTFT
from utils.tft_config import (
    TFT_EXTENSION_MIGRATION_CAPABILITIES,
    apply_tft_profile,
    pending_v2_artifact_extensions,
    validate_tft_v2_artifact_readiness,
)


def _raw_config(**overrides):
    values = {
        "model": "TemporalFusionTransformer",
        "task_name": "long_term_forecast",
        "model_id": "tft-sr06-interaction-contract",
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
        "tft_attention_dropout": 0.0,
        "tft_use_higher_order": True,
        "tft_vsn_per_feature_gating": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _resolved_config(**overrides):
    return apply_tft_profile(_raw_config(**overrides))


def _inputs():
    generator = torch.Generator().manual_seed(8072000)
    return (
        torch.randn(2, 8, 7, generator=generator),
        torch.randn(2, 8, 4, generator=generator),
        torch.randn(2, 7, 1, generator=generator),
        torch.randn(2, 7, 4, generator=generator),
    )


def test_latent_polynomial_payload_reports_projected_residual_and_strength():
    block = HigherOrderInteractionBlock(
        d_model=6,
        interaction_order=3,
        interaction_rank=4,
        dropout=0.0,
    ).eval()
    x = torch.randn(2, 5, 6)

    with torch.no_grad():
        out, payload = block(x, return_payload=True)

    residual = payload["latent_polynomial_residual"]
    expected = block.layer_norm(x + residual)
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(payload["interaction_contribution"], residual)
    assert torch.isfinite(payload["latent_polynomial_strength"])


def test_v2_model_surfaces_latent_polynomial_semantics_and_diagnostics():
    model = NativeTFT(_raw_config()).eval()

    with torch.no_grad():
        payload = model(*_inputs(), return_interpretation=True)

    expected_time = _raw_config().seq_len + _raw_config().pred_len
    assert payload["latent_polynomial_residual"].shape == (2, expected_time, _raw_config().d_model)
    assert torch.isfinite(payload["latent_polynomial_strength"])
    assert "latent_polynomial_block" in payload["extension_residuals"]


def test_per_feature_vsn_gating_runs_forward_backward_with_live_gradients():
    model = NativeTFT(_raw_config(tft_vsn_per_feature_gating=True)).train()
    batch = _inputs()

    output = model(*batch)
    loss = output.square().mean()
    loss.backward()

    feature_gate_params = {
        name: param
        for name, param in model.named_parameters()
        if "feature_gate_grn" in name and param.requires_grad
    }
    assert feature_gate_params
    active_grads = {
        name: param.grad
        for name, param in feature_gate_params.items()
        if param.grad is not None
    }
    assert active_grads
    assert all(torch.isfinite(grad).all() for grad in active_grads.values())


def test_named_covariate_encoder_is_declared_pair_only_and_name_order_stable():
    encoder = NamedCovariateInteractionEncoder(
        d_model=4,
        rank=2,
        declared_pairs=[("moon", "saturn")],
    ).eval()
    x = torch.randn(2, 6, 3, 4)
    names = ["moon", "saturn", "mercury"]

    with torch.no_grad():
        direct, direct_meta = encoder(x, names)
        permuted, permuted_meta = encoder(
            x[:, :, [2, 0, 1], :],
            ["mercury", "moon", "saturn"],
        )

    assert direct_meta["active_pairs"] == (("moon", "saturn"),)
    assert direct_meta["pair_indices"] == ((0, 1),)
    assert permuted_meta["active_pairs"] == (("moon", "saturn"),)
    assert permuted_meta["pair_indices"] == ((1, 2),)
    torch.testing.assert_close(direct, permuted)


def test_named_covariate_encoder_returns_empty_channels_without_declared_pairs():
    encoder = NamedCovariateInteractionEncoder(
        d_model=4,
        rank=2,
        declared_pairs=[("moon", "saturn")],
    ).eval()

    with torch.no_grad():
        channels, metadata = encoder(torch.randn(1, 3, 2, 4), ["venus", "mars"])

    assert channels.shape == (1, 3, 0, 4)
    assert metadata["active_pairs"] == tuple()


def test_sr06_releases_higher_order_and_per_feature_vsn_v2_artifacts():
    args = _resolved_config(
        tft_use_higher_order=True,
        tft_vsn_per_feature_gating=True,
    )

    for name in ("higher_order_interaction", "per_feature_vsn"):
        capability = TFT_EXTENSION_MIGRATION_CAPABILITIES[name]
        assert capability["repair_task"] == "TFT-SR06"
        assert capability["v2_artifact_status"] == "released"

    assert pending_v2_artifact_extensions(args) == []
    assert validate_tft_v2_artifact_readiness(args) is args
