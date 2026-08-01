from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from layers.AdvancedDynamicGraph import AdvancedDynamicGraphLearner
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
        "model_id": "tft-sr08-graph-contract",
        "data": "ETTh1",
        "features": "MS",
        "seq_len": 12,
        "label_len": 6,
        "pred_len": 4,
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
        "tft_cross_variable_mixing": True,
        "tft_graph_type": "sparse",
        "tft_graph_top_k": 2,
        "tft_graph_head_mode": "single",
        "tft_graph_self_edge_policy": "required",
        "tft_graph_residual_strength_init": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _resolved_config(**overrides):
    return apply_tft_profile(_raw_config(**overrides))


def _inputs():
    generator = torch.Generator().manual_seed(9082000)
    cfg = _raw_config()
    return (
        torch.randn(2, cfg.seq_len, 7, generator=generator),
        torch.randn(2, cfg.seq_len, 4, generator=generator),
        torch.randn(2, cfg.label_len + cfg.pred_len, 1, generator=generator),
        torch.randn(2, cfg.label_len + cfg.pred_len, 4, generator=generator),
    )


def test_single_head_mode_reports_one_adjacency_head_only():
    model = NativeTFT(_raw_config()).eval()

    with torch.no_grad():
        payload = model(*_inputs(), return_interpretation=True)

    history_graph = payload["history_graph_attention"]
    history_meta = payload["history_graph_metadata"]
    assert history_graph.shape[-3] == 1
    assert history_meta["num_reported_heads"] == 1
    assert history_meta["head_mode"] == "single"


def test_true_multihead_mode_learns_distinct_adjacency_tensors():
    torch.manual_seed(13)
    model = AdvancedDynamicGraphLearner(
        d_model=8,
        n_heads=3,
        top_k=2,
        head_mode="true_multihead",
        self_edge_policy="required",
        residual_strength_init=0.0,
        num_layers=1,
    ).eval()
    x = torch.randn(2, 4, 6, 8)

    with torch.no_grad():
        _, adj = model(x, return_attention=True)

    assert adj.shape[2] == 3
    assert not torch.equal(adj[:, :, 0], adj[:, :, 1])


def test_self_edge_policy_and_topk_nonself_budget_hold_exactly():
    model = AdvancedDynamicGraphLearner(
        d_model=8,
        n_heads=2,
        top_k=2,
        head_mode="single",
        self_edge_policy="required",
        residual_strength_init=0.0,
        num_layers=1,
    ).eval()
    x = torch.randn(1, 3, 6, 8)

    with torch.no_grad():
        _, adj = model(x, return_attention=True)

    support = adj[0, 0, 0] > 1e-7
    per_row_nonzero = support.sum(dim=-1)
    assert torch.all(per_row_nonzero == 3)
    assert torch.all(torch.diagonal(support, dim1=-2, dim2=-1))

    excluded = AdvancedDynamicGraphLearner(
        d_model=8,
        n_heads=2,
        top_k=2,
        head_mode="single",
        self_edge_policy="excluded",
        residual_strength_init=0.0,
        num_layers=1,
    ).eval()
    with torch.no_grad():
        _, excluded_adj = excluded(x, return_attention=True)
    excluded_diag = torch.diagonal(excluded_adj[0, 0, 0], dim1=-2, dim2=-1)
    assert torch.allclose(excluded_diag, torch.zeros_like(excluded_diag))


def test_invalid_required_topk_rejects_more_than_available_nonself_neighbors():
    model = AdvancedDynamicGraphLearner(
        d_model=8,
        n_heads=2,
        top_k=6,
        head_mode="single",
        self_edge_policy="required",
        residual_strength_init=0.0,
        num_layers=1,
    ).eval()
    x = torch.randn(1, 2, 6, 8)

    with pytest.raises(ValueError, match="exceeds available candidates"):
        model(x)


def test_model_payload_exports_graph_metadata_diagnostics():
    model = NativeTFT(
        _raw_config(
            tft_graph_head_mode="single",
            tft_graph_self_edge_policy="required",
            tft_graph_history_top_k=2,
            tft_graph_future_top_k=1,
        )
    ).eval()

    with torch.no_grad():
        payload = model(*_inputs(), return_interpretation=True)

    history_meta = payload["history_graph_metadata"]
    future_meta = payload["future_graph_metadata"]
    for metadata in (history_meta, future_meta):
        assert metadata["self_edge_policy"] == "required"
        assert torch.isfinite(metadata["adjacency_entropy"])
        assert torch.isfinite(metadata["selected_support_frequency"])
        assert torch.isfinite(metadata["support_turnover"])
        assert torch.isfinite(metadata["self_edge_mass"])


def test_sr08_releases_graph_cross_mixing_v2_artifacts():
    args = _resolved_config(tft_cross_variable_mixing=True)

    graph = TFT_EXTENSION_MIGRATION_CAPABILITIES["graph_cross_mixing"]
    assert graph["repair_task"] == "TFT-SR08"
    assert graph["v2_artifact_status"] == "released"

    assert pending_v2_artifact_extensions(args) == []
    assert validate_tft_v2_artifact_readiness(args) is args
