import json
import pickle
import random
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.TemporalFusionTransformer import Model as NativeTFTModel
from utils.reproducibility import (
    DeterministicEpochSampler,
    SeedBundle,
    atomic_write_json,
    build_paired_models,
    derive_seed,
    isolated_rng,
    make_torch_generator,
    resolve_seed_bundle,
    seed_worker,
    set_experiment_seed,
    set_global_seed,
    stable_json_hash,
    state_dict_sha256,
)


def _args(**overrides):
    values = {
        "seed": 2021,
        "model_init_seed": None,
        "extension_init_seed": None,
        "data_order_seed": None,
        "worker_seed": None,
        "training_seed": None,
        "deterministic_mode": "warn",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _small_native_tft_args():
    """Small real native-TFT contract; no mock modules or patched initializers."""

    return SimpleNamespace(
        task_name="long_term_forecast",
        model="TemporalFusionTransformer",
        data="ETTh1",
        features="M",
        seq_len=12,
        label_len=6,
        pred_len=3,
        enc_in=7,
        dec_in=7,
        c_out=7,
        d_model=8,
        d_ff=2048,
        n_heads=2,
        e_layers=1,
        dropout=0.1,
        embed="timeF",
        freq="h",
        tft_extension_semantics_version=2,
        tft_use_revin=False,
        tft_revin_affine=False,
        tft_use_quantile_head=False,
        tft_output_mode="point",
        tft_temporal_backbone="lstm",
    )


class _ReferenceModel(nn.Module):
    def __init__(self, width=3):
        super().__init__()
        self.shared = nn.Linear(2, width)
        self.register_buffer("shared_counter", torch.tensor(7, dtype=torch.int64))


class _VariantModel(nn.Module):
    def __init__(self, width=3):
        super().__init__()
        self.shared = nn.Linear(2, width)
        self.register_buffer("shared_counter", torch.tensor(7, dtype=torch.int64))
        self.extension = nn.Linear(width, 1)


def _numpy_states_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_sha256_seed_stream_is_versioned_stable_and_independent():
    expected = derive_seed(2021, "model_init", iteration=0, version=1)
    assert expected == derive_seed(2021, "model_init", iteration=0, version=1)
    assert 0 <= expected < 2**63 - 1
    assert expected != derive_seed(2021, "extension_init", iteration=0)
    assert expected != derive_seed(2021, "model_init", iteration=1)
    assert expected != derive_seed(2022, "model_init", iteration=0)

    with pytest.raises(ValueError, match="Unsupported seed derivation version"):
        derive_seed(2021, "model_init", version=2)
    with pytest.raises(ValueError, match="non-empty"):
        derive_seed(2021, "")
    with pytest.raises(ValueError, match="non-negative"):
        derive_seed(-1, "model_init")


def test_seed_bundle_resolves_all_streams_and_is_dataclass_serializable():
    bundle = resolve_seed_bundle(_args(), iteration=2)
    assert isinstance(bundle, SeedBundle)
    assert bundle.base_seed == 2021
    assert bundle.iteration == 2
    values = {
        bundle.experiment_seed,
        bundle.model_init_seed,
        bundle.extension_init_seed,
        bundle.data_order_seed,
        bundle.worker_seed,
        bundle.training_seed,
    }
    assert len(values) == 6
    assert json.loads(json.dumps(asdict(bundle))) == asdict(bundle)

    repeated = resolve_seed_bundle(_args(), iteration=2)
    next_iteration = resolve_seed_bundle(_args(), iteration=3)
    changed_base = resolve_seed_bundle(_args(seed=2022), iteration=2)
    assert repeated == bundle
    assert next_iteration != bundle
    assert changed_base != bundle


def test_explicit_stream_roots_still_receive_iteration_schedule():
    args = _args(
        model_init_seed=11,
        extension_init_seed=22,
        data_order_seed=33,
        worker_seed=44,
        training_seed=55,
    )
    first = resolve_seed_bundle(args, iteration=0)
    second = resolve_seed_bundle(args, iteration=1)
    assert first.model_init_seed == derive_seed(11, "model_init", 0)
    assert first.extension_init_seed == derive_seed(22, "extension_init", 0)
    assert first.data_order_seed == derive_seed(33, "data_order", 0)
    assert first.worker_seed == derive_seed(44, "worker", 0)
    assert first.training_seed == derive_seed(55, "training", 0)
    assert asdict(first) != asdict(second)


def _native_tft_initialization_and_epoch_zero_order(base_seed):
    bundle = resolve_seed_bundle(_args(seed=base_seed), iteration=0)
    set_global_seed(bundle.model_init_seed, deterministic_mode="off")
    model_hash = state_dict_sha256(NativeTFTModel(_small_native_tft_args()).state_dict())
    sampler = DeterministicEpochSampler(
        list(range(41)),
        seed=bundle.data_order_seed,
    )
    return bundle, model_hash, tuple(sampler)


def test_base_seed_controls_real_native_tft_initialization_and_sampler_order():
    first_bundle, first_model_hash, first_order = (
        _native_tft_initialization_and_epoch_zero_order(2021)
    )
    changed_bundle, changed_model_hash, changed_order = (
        _native_tft_initialization_and_epoch_zero_order(2022)
    )
    replay_bundle, replay_model_hash, replay_order = (
        _native_tft_initialization_and_epoch_zero_order(2021)
    )

    assert replay_bundle == first_bundle
    assert replay_model_hash == first_model_hash
    assert replay_order == first_order

    assert changed_bundle.model_init_seed != first_bundle.model_init_seed
    assert changed_bundle.data_order_seed != first_bundle.data_order_seed
    assert changed_model_hash != first_model_hash
    assert changed_order != first_order
    assert sorted(first_order) == sorted(changed_order) == list(range(41))


def test_set_global_seed_replays_all_cpu_rngs_and_honors_modes():
    algorithms_before = torch.are_deterministic_algorithms_enabled()
    warn_before = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic_before = torch.backends.cudnn.deterministic
    cudnn_benchmark_before = torch.backends.cudnn.benchmark
    try:
        set_global_seed(8123, deterministic_mode="warn")
        first = (random.random(), np.random.random(), torch.rand(3))
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()

        set_global_seed(8123, deterministic_mode="warn")
        second = (random.random(), np.random.random(), torch.rand(3))
        assert first[0] == second[0]
        assert first[1] == second[1]
        assert torch.equal(first[2], second[2])

        set_global_seed(8123, deterministic_mode="strict")
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()

        set_global_seed(8123, deterministic_mode="off")
        assert not torch.are_deterministic_algorithms_enabled()
        with pytest.raises(ValueError, match="deterministic_mode"):
            set_global_seed(1, deterministic_mode="sometimes")
    finally:
        torch.use_deterministic_algorithms(
            algorithms_before, warn_only=warn_before if algorithms_before else False
        )
        torch.backends.cudnn.deterministic = cudnn_deterministic_before
        torch.backends.cudnn.benchmark = cudnn_benchmark_before


def test_set_experiment_seed_can_select_model_then_training_stream():
    args = _args(deterministic_mode="warn")
    bundle = resolve_seed_bundle(args, iteration=4)
    returned = set_experiment_seed(args, iteration=4, selected_seed=bundle.model_init_seed)
    first_model_draw = torch.rand(2)
    assert returned == bundle
    set_global_seed(bundle.model_init_seed, deterministic_mode="warn")
    assert torch.equal(first_model_draw, torch.rand(2))

    set_experiment_seed(bundle, selected_seed=bundle.training_seed)
    first_training_draw = torch.rand(2)
    set_global_seed(bundle.training_seed, deterministic_mode="warn")
    assert torch.equal(first_training_draw, torch.rand(2))


@pytest.mark.parametrize(
    ("mode", "enabled", "warn_only"),
    (("off", False, False), ("warn", True, True), ("strict", True, False)),
)
def test_set_experiment_seed_bundle_honors_explicit_backend_mode(
    mode, enabled, warn_only
):
    algorithms_before = torch.are_deterministic_algorithms_enabled()
    warning_before = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic_before = torch.backends.cudnn.deterministic
    cudnn_benchmark_before = torch.backends.cudnn.benchmark
    try:
        bundle = resolve_seed_bundle(_args(), iteration=0)
        returned = set_experiment_seed(
            bundle,
            selected_seed=bundle.model_init_seed,
            deterministic_mode=mode,
        )
        assert returned == bundle
        assert torch.are_deterministic_algorithms_enabled() is enabled
        assert torch.is_deterministic_algorithms_warn_only_enabled() is warn_only
    finally:
        torch.use_deterministic_algorithms(
            algorithms_before,
            warn_only=warning_before if algorithms_before else False,
        )
        torch.backends.cudnn.deterministic = cudnn_deterministic_before
        torch.backends.cudnn.benchmark = cudnn_benchmark_before


def test_isolated_rng_replays_inner_draws_and_restores_outer_states():
    random.seed(51)
    np.random.seed(52)
    torch.manual_seed(53)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    with isolated_rng(991):
        inner_first = (random.random(), np.random.random(), torch.rand(4))
    assert random.getstate() == python_before
    assert _numpy_states_equal(np.random.get_state(), numpy_before)
    assert torch.equal(torch.get_rng_state(), torch_before)

    with isolated_rng(991):
        inner_second = (random.random(), np.random.random(), torch.rand(4))
    assert inner_first[0] == inner_second[0]
    assert inner_first[1] == inner_second[1]
    assert torch.equal(inner_first[2], inner_second[2])


def test_isolated_rng_restores_states_when_body_raises():
    random.seed(71)
    np.random.seed(72)
    torch.manual_seed(73)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="sentinel"):
        with isolated_rng(88):
            random.random()
            np.random.random()
            torch.rand(1)
            raise RuntimeError("sentinel")
    assert random.getstate() == python_before
    assert _numpy_states_equal(np.random.get_state(), numpy_before)
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_stable_json_hash_is_canonical_and_supports_seed_bundle():
    left = {"beta": [2, 3], "alpha": {"x", "y"}}
    right = {"alpha": {"y", "x"}, "beta": (2, 3)}
    assert stable_json_hash(left) == stable_json_hash(right)
    assert stable_json_hash(left) != stable_json_hash({"beta": [3, 2], "alpha": {"x", "y"}})
    assert len(stable_json_hash(resolve_seed_bundle(_args()))) == 64
    with pytest.raises(TypeError, match="string keys"):
        stable_json_hash({1: "not canonical"})
    with pytest.raises(ValueError):
        stable_json_hash({"bad": float("nan")})


def test_state_dict_hash_includes_name_dtype_shape_and_exact_bytes():
    raw = torch.arange(4, dtype=torch.float32)
    baseline = state_dict_sha256({"weight": raw})
    assert baseline == state_dict_sha256({"weight": raw.clone()})
    assert baseline != state_dict_sha256({"bias": raw.clone()})
    assert baseline != state_dict_sha256({"weight": raw.reshape(2, 2)})
    assert baseline != state_dict_sha256({"weight": raw.to(torch.float64)})
    changed = raw.clone()
    changed[-1] += 1
    assert baseline != state_dict_sha256({"weight": changed})
    assert len(baseline) == 64

    with pytest.raises(TypeError, match="must be a tensor"):
        state_dict_sha256({"weight": [1, 2, 3]})


def test_seed_worker_is_top_level_picklable_and_replays_python_numpy():
    assert pickle.loads(pickle.dumps(seed_worker)) is seed_worker
    torch.manual_seed(123456)
    seed_worker(0)
    first = (random.random(), np.random.random())
    torch.manual_seed(123456)
    seed_worker(999)
    second = (random.random(), np.random.random())
    assert first == second


def test_epoch_sampler_is_global_rng_independent_and_epoch_addressable():
    dataset = list(range(20))
    sampler_a = DeterministicEpochSampler(dataset, seed=456)
    sampler_b = DeterministicEpochSampler(dataset, seed=456)
    epoch_zero = list(sampler_a)
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    assert list(sampler_b) == epoch_zero
    assert sorted(epoch_zero) == list(range(20))

    sampler_a.set_epoch(1)
    sampler_b.set_epoch(1)
    assert list(sampler_a) == list(sampler_b)
    assert list(sampler_a) != epoch_zero

    ordered = DeterministicEpochSampler(dataset, seed=1, shuffle=False)
    ordered.set_epoch(100)
    assert list(ordered) == list(range(20))


def test_sampler_order_matches_with_zero_and_multiple_loader_workers():
    dataset = TensorDataset(torch.arange(24))

    def read_order(worker_count):
        loader = DataLoader(
            dataset,
            batch_size=5,
            sampler=DeterministicEpochSampler(dataset, seed=987),
            num_workers=worker_count,
            worker_init_fn=seed_worker,
            generator=make_torch_generator(654),
        )
        return torch.cat([batch[0] for batch in loader]).tolist()

    assert read_order(0) == read_order(2)


def test_atomic_json_writer_replaces_manifest_without_partial_temp(tmp_path):
    path = tmp_path / "nested" / "manifest.json"
    bundle = resolve_seed_bundle(_args(), iteration=2)
    assert atomic_write_json(path, {"bundle": bundle, "value": 1}) == path
    assert json.loads(path.read_text()) == {"bundle": asdict(bundle), "value": 1}
    atomic_write_json(path, {"replacement": True})
    assert json.loads(path.read_text()) == {"replacement": True}
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


def test_build_paired_models_copies_all_shared_state_and_reports_hashes():
    random.seed(301)
    np.random.seed(302)
    torch.manual_seed(303)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    reference, variant, report = build_paired_models(
        _ReferenceModel,
        _VariantModel,
        reference_seed=1001,
        variant_seed=2002,
        allowed_variant_only_prefixes=("extension.",),
    )
    assert random.getstate() == python_before
    assert _numpy_states_equal(np.random.get_state(), numpy_before)
    assert torch.equal(torch.get_rng_state(), torch_before)

    reference_state = reference.state_dict()
    variant_state = variant.state_dict()
    assert report["copied_names"] == [
        "shared.bias",
        "shared.weight",
        "shared_counter",
    ]
    assert report["reference_only_names"] == []
    assert report["variant_only_names"] == ["extension.bias", "extension.weight"]
    assert report["copied_tensor_count"] == 3
    assert len(report["shared_state_sha256"]) == 64
    assert report["shared_state_hash"] == report["shared_state_sha256"]
    assert report["reference_state_sha256"] == state_dict_sha256(reference_state)
    assert report["variant_state_sha256"] == state_dict_sha256(variant_state)
    for name in report["copied_names"]:
        assert torch.equal(reference_state[name], variant_state[name])


def test_build_paired_models_rejects_unexpected_unmatched_names():
    with pytest.raises(ValueError, match="unexpected unmatched names"):
        build_paired_models(
            _ReferenceModel,
            _VariantModel,
            reference_seed=1,
            variant_seed=2,
        )


def test_build_paired_models_accepts_explicit_regex_allowlist():
    _, _, report = build_paired_models(
        _ReferenceModel,
        _VariantModel,
        reference_seed=1,
        variant_seed=2,
        allowed_variant_only_regexes=(r"^extension\.(weight|bias)$",),
    )
    assert report["variant_only_names"] == ["extension.bias", "extension.weight"]


def test_build_paired_models_rejects_shared_name_shape_mismatch_even_if_allowed():
    with pytest.raises(ValueError, match="shape mismatch"):
        build_paired_models(
            lambda: _ReferenceModel(width=3),
            lambda: _VariantModel(width=4),
            reference_seed=1,
            variant_seed=2,
            allowed_variant_only_prefixes=("",),
        )
