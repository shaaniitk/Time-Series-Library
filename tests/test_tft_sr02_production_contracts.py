"""Production-boundary coverage for the ten TFT-SR02 extension adapters."""

from __future__ import annotations

import copy
import hashlib

import pytest
import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from layers.TemporalFusion_layers import ExtensionResidualAdapter
from models.TemporalFusionTransformer import Model as NativeTFT
from run import build_parser, build_setting, normalize_args, resolve_run_args
from utils.reproducibility import isolated_rng, state_dict_sha256
from utils.tft_checkpoint import load_tft_checkpoint
from utils.tft_config import (
    apply_tft_profile,
    read_tft_semantics_metadata,
    write_tft_semantics_metadata,
)


def _cli_args(*extra):
    argv = (
        "--task_name long_term_forecast --is_training 1 "
        "--model_id sr02-production-contract --model TemporalFusionTransformer "
        "--data ETTh1 --features MS --seq_len 24 --label_len 12 --pred_len 4 "
        "--enc_in 7 --dec_in 7 --c_out 1 --tft_target_pos 6 "
        "--d_model 8 --n_heads 2 --e_layers 1 --d_layers 1 "
        "--tft_profile extended_safe --tft_temporal_backbone lstm "
        "--dropout 0.0 --tft_attention_dropout 0.0 "
        "--train_epochs 1 --batch_size 2 --num_workers 0 --no_use_gpu"
    ).split()
    return normalize_args(build_parser().parse_args(argv + list(extra)))


# Counts and namespaces are intentionally independent of the production
# allowlist.  This catches both unapproved topology drift and an accidentally
# over-broad production regular expression.
PAIR_CASES = (
    pytest.param(
        "fft_branch",
        "tft_use_fft_branch",
        ("--tft_use_fft_branch", "--tft_fft_modes", "4"),
        9,
        (
            "temporal_fusion_decoder.layers.0.fft_branch.",
            "temporal_fusion_decoder.layers.0.fft_fusion_gate.",
            "temporal_fusion_decoder.layers.0.fft_residual_adapter.",
        ),
        id="fft",
    ),
    pytest.param(
        "explicit_cross_attention",
        "tft_use_explicit_cross_attention",
        ("--tft_use_explicit_cross_attention",),
        11,
        (
            "temporal_fusion_decoder.layers.0.cross_attention.",
            "temporal_fusion_decoder.layers.0.gate_after_cross_attention.",
            "temporal_fusion_decoder.layers.0.cross_attention_residual_adapter.",
        ),
        id="explicit-cross-attention",
    ),
    pytest.param(
        "lag_attention",
        "tft_use_lag_attention",
        ("--tft_use_lag_attention", "--tft_lag_scales", "1,2"),
        12,
        (
            "temporal_fusion_decoder.layers.0.lag_attention_module.",
            "temporal_fusion_decoder.layers.0.lag_residual_adapter.",
        ),
        id="lag-attention",
    ),
    pytest.param(
        "higher_order_interaction",
        "tft_use_higher_order",
        ("--tft_use_higher_order",),
        13,
        (
            "temporal_fusion_decoder.layers.0.higher_order_block.",
            "temporal_fusion_decoder.layers.0.higher_order_residual_adapter.",
        ),
        id="higher-order",
    ),
    pytest.param(
        "temporal_compression",
        "tft_use_temporal_compression",
        ("--tft_use_temporal_compression", "--tft_tc_threshold", "1"),
        11,
        (
            "temporal_fusion_decoder.layers.0.temporal_compression.",
            "temporal_fusion_decoder.layers.0.compression_residual_adapter.",
        ),
        id="temporal-compression",
    ),
    pytest.param(
        "graph_cross_mixing",
        "tft_cross_variable_mixing",
        ("--tft_cross_variable_mixing",),
        22,
        (
            "history_vsn.cross_mixing.",
            "history_vsn.graph_residual_adapter.",
            "future_vsn.cross_mixing.",
            "future_vsn.graph_residual_adapter.",
        ),
        id="graph-cross-mixing",
    ),
    pytest.param(
        "covariate_reattention",
        "tft_covariate_reattention",
        ("--tft_covariate_reattention",),
        11,
        (
            "temporal_fusion_decoder.layers.0.covariate_cross_attention.",
            "temporal_fusion_decoder.layers.0.gate_after_reattention.",
            "temporal_fusion_decoder.layers.0.covariate_reattention_residual_adapter.",
        ),
        id="covariate-reattention",
    ),
    pytest.param(
        "regime_moe",
        "tft_use_regime_moe",
        (
            "--tft_use_regime_moe",
            "--tft_num_moe_experts",
            "2",
            "--tft_num_regimes",
            "2",
            "--tft_moe_top_k",
            "1",
        ),
        15,
        (
            "temporal_fusion_decoder.layers.0.regime_moe.",
            "temporal_fusion_decoder.layers.0.regime_moe_residual_adapter.",
        ),
        id="regime-moe",
    ),
    pytest.param(
        "dual_attention_fusion",
        "tft_dual_attention_fusion",
        ("--tft_dual_attention_fusion",),
        5,
        (
            "temporal_fusion_decoder.layers.0.dual_attention_module.",
            "temporal_fusion_decoder.layers.0.dual_attention_residual_adapter.",
        ),
        id="dual-attention",
    ),
    pytest.param(
        "vsn_residual_bypass",
        "tft_vsn_residual_bypass",
        ("--tft_vsn_residual_bypass",),
        6,
        (
            "history_vsn.residual_projection.",
            "history_vsn.vsn_bypass_residual_adapter.",
            "future_vsn.residual_projection.",
            "future_vsn.vsn_bypass_residual_adapter.",
        ),
        id="vsn-residual-bypass",
    ),
)


def _independent_reference(run_args, disable_flag, model_init_seed):
    reference_args = copy.deepcopy(run_args)
    setattr(reference_args, disable_flag, False)
    reference_args.tft_paired_initialization = False
    if hasattr(reference_args, "_tft_resolved_schema"):
        delattr(reference_args, "_tft_resolved_schema")
    reference_args = apply_tft_profile(reference_args)
    with isolated_rng(model_init_seed):
        return NativeTFT(reference_args).float()


@pytest.mark.parametrize(
    (
        "extension_name",
        "disable_flag",
        "extension_cli",
        "variant_only_count",
        "variant_namespaces",
    ),
    PAIR_CASES,
)
def test_real_exp_pairing_copies_all_shared_state_and_only_adds_declared_namespaces(
    extension_name,
    disable_flag,
    extension_cli,
    variant_only_count,
    variant_namespaces,
):
    args = _cli_args(
        *extension_cli,
        "--tft_paired_initialization",
        "--tft_paired_reference_disable",
        disable_flag,
    )
    run_args, seed_bundle = resolve_run_args(args, 0)
    experiment = Exp_Long_Term_Forecast(run_args)
    report = run_args._paired_initialization_report
    variant_state = experiment.model.state_dict()
    reference_state = _independent_reference(
        run_args,
        disable_flag,
        seed_bundle.model_init_seed,
    ).state_dict()

    assert run_args.tft_resolved_extension_modes[extension_name] == "neutral"
    assert report["binding_verified"] is True
    assert report["reference_only_names"] == []
    assert report["copied_names"] == sorted(reference_state)
    assert report["copied_tensor_count"] == len(reference_state)
    assert report["reference_state_sha256"] == state_dict_sha256(reference_state)
    assert report["shared_state_sha256"] == report["shared_state_hash"]

    variant_only = sorted(set(variant_state) - set(reference_state))
    assert variant_only == report["variant_only_names"]
    assert len(variant_only) == variant_only_count
    assert all(
        any(name.startswith(namespace) for namespace in variant_namespaces)
        for name in variant_only
    )
    assert all(
        any(name.startswith(namespace) for name in variant_only)
        for namespace in variant_namespaces
    )
    for name, reference_tensor in reference_state.items():
        assert torch.equal(variant_state[name], reference_tensor), name


RELEASED_CASES = (
    pytest.param(
        "regime_moe",
        "tft_use_regime_moe",
        (
            "--tft_use_regime_moe",
            "--tft_num_moe_experts",
            "2",
            "--tft_num_regimes",
            "2",
            "--tft_moe_top_k",
            "1",
        ),
        id="regime-moe",
    ),
    pytest.param(
        "dual_attention_fusion",
        "tft_dual_attention_fusion",
        ("--tft_dual_attention_fusion",),
        id="dual-attention",
    ),
    pytest.param(
        "vsn_residual_bypass",
        "tft_vsn_residual_bypass",
        ("--tft_vsn_residual_bypass",),
        id="vsn-residual-bypass",
    ),
)


def _released_run_args(extension_cli, disable_flag):
    args = _cli_args(
        *extension_cli,
        "--tft_paired_initialization",
        "--tft_paired_reference_disable",
        disable_flag,
    )
    return resolve_run_args(args, 0)[0]


def _learn_residual_strengths(model, extension_name):
    adapters = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, ExtensionResidualAdapter)
        and module.extension_name == extension_name
    ]
    assert adapters
    assert all(adapter.is_effectively_zero() for _, adapter in adapters)

    generator = torch.Generator().manual_seed(31051995)
    inputs = (
        torch.randn(2, 24, 7, generator=generator),
        torch.randn(2, 24, 4, generator=generator),
        torch.randn(2, 16, 1, generator=generator),
        torch.randn(2, 16, 4, generator=generator),
    )
    model.train()
    output = model(*inputs)
    target = torch.randn(output.shape, generator=generator)
    strengths = [adapter.residual_strength for _, adapter in adapters]
    optimizer = torch.optim.SGD(strengths, lr=0.1)
    optimizer.zero_grad(set_to_none=True)
    (output - target).square().mean().backward()
    assert all(
        strength.grad is not None
        and torch.isfinite(strength.grad).all()
        and torch.count_nonzero(strength.grad).item() > 0
        for strength in strengths
    )
    optimizer.step()

    learned = {
        f"{name}.residual_strength": adapter.residual_strength.detach().clone()
        for name, adapter in adapters
    }
    assert all(torch.count_nonzero(value).item() > 0 for value in learned.values())
    return learned


@pytest.mark.parametrize(
    ("extension_name", "disable_flag", "extension_cli"),
    RELEASED_CASES,
)
def test_released_v2_extension_checkpoint_roundtrip_is_metadata_and_hash_bound(
    tmp_path,
    extension_name,
    disable_flag,
    extension_cli,
):
    source_args = _released_run_args(extension_cli, disable_flag)
    source = Exp_Long_Term_Forecast(source_args)
    learned_strengths = _learn_residual_strengths(source.model, extension_name)
    source_state = {
        name: tensor.detach().clone()
        for name, tensor in source.model.state_dict().items()
    }

    setting = build_setting(source.args, 0)
    artifact_dir = tmp_path / extension_name
    artifact_dir.mkdir()
    checkpoint = artifact_dir / "checkpoint.pth"
    torch.save(source_state, checkpoint)
    metadata_path = write_tft_semantics_metadata(
        artifact_dir,
        source.args,
        artifact_kind="checkpoint",
        setting=setting,
        checkpoint_path=checkpoint,
    )
    metadata = read_tft_semantics_metadata(checkpoint)

    assert metadata_path.is_file()
    assert metadata["extension_semantics_version"] == 2
    assert metadata["config_digest"] == source.args.tft_config_digest
    assert metadata["active_extensions"][extension_name] is True
    assert (
        metadata["extension_integration"]["resolved_modes"][extension_name]
        == "neutral"
    )
    assert metadata["checkpoint"] == {
        "filename": "checkpoint.pth",
        "bytes": checkpoint.stat().st_size,
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    checkpoint_extension_state = metadata["checkpoint_extension_state"]
    assert checkpoint_extension_state["mode_semantics"] == "initialization_policy"
    assert checkpoint_extension_state["nonzero_scalar_count"] >= len(
        learned_strengths
    )
    assert set(checkpoint_extension_state["parameter_values"]) == set(
        learned_strengths
    )
    for name, expected in learned_strengths.items():
        assert checkpoint_extension_state["parameter_values"][name] == pytest.approx(
            expected.detach().cpu().reshape(-1).tolist()
        )

    target_args = _released_run_args(extension_cli, disable_flag)
    target = Exp_Long_Term_Forecast(target_args)
    policy = load_tft_checkpoint(
        target.model,
        checkpoint,
        target.args,
        map_location="cpu",
        external_load=True,
        expected_setting=setting,
    )
    assert policy["load_mode"] == "current"
    assert policy["metadata_present"] is True
    for name, expected in source_state.items():
        assert torch.equal(target.model.state_dict()[name], expected), name
    for name, expected in learned_strengths.items():
        assert torch.count_nonzero(expected).item() > 0
        assert torch.equal(target.model.state_dict()[name], expected), name

    # Prove that the successful load above depended on the sidecar's exact
    # checkpoint record, not merely on compatible semantic configuration.
    tampered_state = dict(source_state)
    strength_name = next(iter(learned_strengths))
    tampered_state[strength_name] = tampered_state[strength_name] + 1.0
    torch.save(tampered_state, checkpoint)
    with pytest.raises(RuntimeError, match="hash/size"):
        load_tft_checkpoint(
            target.model,
            checkpoint,
            target.args,
            map_location="cpu",
            external_load=True,
            expected_setting=setting,
        )
