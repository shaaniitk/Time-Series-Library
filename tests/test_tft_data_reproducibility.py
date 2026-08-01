import hashlib
import os
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from data_provider.data_factory import (
    data_provider,
)
from data_provider.data_loader import (
    Dataset_Custom,
    Dataset_ETT_hour,
    Dataset_ETT_minute,
)
from utils.reproducibility import (
    DeterministicEpochSampler,
    resolve_seed_bundle,
    stable_json_hash,
)


def _write_series(path, rows=160, freq="h"):
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2000-01-01", periods=rows, freq=freq),
            "feature": np.arange(rows, dtype=np.float64),
            "OT": np.arange(rows, dtype=np.float64) * 0.5,
        }
    )
    frame.to_csv(path, index=False)
    return frame


def _base_args(tmp_path, *, num_workers=0, semantics=2, **overrides):
    values = {
        "model": "TemporalFusionTransformer",
        "tft_extension_semantics_version": semantics,
        "task_name": "long_term_forecast",
        "data": "custom",
        "root_path": str(tmp_path),
        "data_path": "series.csv",
        "features": "M",
        "target": "OT",
        "embed": "timeF",
        "freq": "h",
        "seq_len": 8,
        "label_len": 4,
        "pred_len": 3,
        "seasonal_patterns": "Monthly",
        "batch_size": 9,
        "num_workers": num_workers,
        "augmentation_ratio": 0,
        "seed": 41,
        "model_init_seed": 101,
        "extension_init_seed": 102,
        "data_order_seed": 103,
        "worker_seed": 104,
        "run_index": 0,
    }
    values.update(overrides)
    args = SimpleNamespace(**values)
    if semantics == 2 and args.data_order_seed is not None and args.worker_seed is not None:
        bundle = resolve_seed_bundle(args, iteration=args.run_index)
        args._seed_bundle = bundle
        args.data_order_seed = bundle.data_order_seed
        args.worker_seed = bundle.worker_seed
        args.training_seed = bundle.training_seed
    return args


def _ordered_first_values(loader):
    values = []
    for batch_x, _, _, _ in loader:
        values.extend(batch_x[:, 0, 0].tolist())
    return values


def _augmentation_args(args, extra_tag=""):
    for name in (
        "jitter",
        "scaling",
        "rotation",
        "permutation",
        "randompermutation",
        "magwarp",
        "timewarp",
        "windowslice",
        "windowwarp",
        "spawner",
        "dtwwarp",
        "shapedtwwarp",
        "wdba",
        "discdtw",
        "discsdtw",
    ):
        setattr(args, name, name == "jitter")
    args.extra_tag = extra_tag
    args.augmentation_ratio = 1
    return args


def test_v2_train_order_ignores_global_model_rng_and_is_epoch_addressable(tmp_path):
    _write_series(tmp_path / "series.csv")
    args = _base_args(tmp_path)

    _, loader_a = data_provider(args, "train")
    order_a0 = list(iter(loader_a.sampler))

    # Simulate an optional model branch consuming every process-global stream
    # before the otherwise identical loader is built.
    random.random()
    np.random.random(37)
    torch.rand(211)
    _, loader_b = data_provider(args, "train")
    order_b0 = list(iter(loader_b.sampler))

    assert isinstance(loader_a.sampler, DeterministicEpochSampler)
    assert loader_a.sampler.shuffle is True
    assert order_a0 == order_b0
    assert loader_a.reproducibility_metadata == loader_b.reproducibility_metadata

    loader_a.sampler.set_epoch(1)
    loader_b.sampler.set_epoch(1)
    order_a1 = list(iter(loader_a.sampler))
    assert order_a1 == list(iter(loader_b.sampler))
    assert order_a1 != order_a0


def test_v2_worker_count_does_not_change_sample_order(tmp_path):
    _write_series(tmp_path / "series.csv")
    args_zero = _base_args(tmp_path, num_workers=0)
    args_two = _base_args(tmp_path, num_workers=2)

    _, loader_zero = data_provider(args_zero, "train")
    _, loader_two = data_provider(args_two, "train")

    assert list(iter(loader_zero.sampler)) == list(iter(loader_two.sampler))
    assert _ordered_first_values(loader_zero) == _ordered_first_values(loader_two)
    assert (
        loader_zero.reproducibility_metadata["epoch0_order_hash"]
        == loader_two.reproducibility_metadata["epoch0_order_hash"]
    )
    assert (
        loader_zero.reproducibility_metadata["first_batch_index_hash"]
        == loader_two.reproducibility_metadata["first_batch_index_hash"]
    )


def test_v2_shuffles_train_only_and_attaches_complete_metadata(tmp_path):
    _write_series(tmp_path / "series.csv")
    args = _base_args(tmp_path)

    train_set, train_loader = data_provider(args, "train")
    val_set, val_loader = data_provider(args, "val")
    test_set, test_loader = data_provider(args, "test")

    assert train_loader.sampler.shuffle is True
    assert val_loader.sampler.shuffle is False
    assert test_loader.sampler.shuffle is False
    assert list(iter(val_loader.sampler)) == list(range(len(val_set)))
    assert list(iter(test_loader.sampler)) == list(range(len(test_set)))

    metadata = train_loader.reproducibility_metadata
    assert set(
        (
            "split",
            "sampler_policy",
            "fold_manifest",
            "fold_manifest_hash",
            "epoch0_order_hash",
            "first_batch_index_hash",
            "data_order_seed",
            "worker_seed",
        )
    ).issubset(metadata)
    assert metadata["split"] == "train"
    assert metadata["sampler_policy"] == "deterministic_epoch_shuffle"
    assert metadata["fold_manifest"] == train_set.fold_manifest
    assert metadata["fold_manifest_hash"] == train_set.fold_manifest_hash
    assert val_loader.reproducibility_metadata["sampler_policy"] == "ordered"
    assert test_loader.reproducibility_metadata["sampler_policy"] == "ordered"


def test_semantics_v1_and_non_tft_keep_legacy_shuffle_contract(tmp_path):
    _write_series(tmp_path / "series.csv")
    legacy = _base_args(tmp_path, semantics=1)

    _, legacy_val = data_provider(legacy, "val")
    _, legacy_test = data_provider(legacy, "test")
    assert isinstance(legacy_val.sampler, RandomSampler)
    assert isinstance(legacy_test.sampler, SequentialSampler)
    assert legacy_val.reproducibility_metadata["epoch0_order_hash"] is None
    assert legacy_val.reproducibility_metadata["data_order_seed"] is None

    non_tft = _base_args(tmp_path, model="DLinear")
    _, non_tft_val = data_provider(non_tft, "val")
    assert isinstance(non_tft_val.sampler, RandomSampler)


def test_direct_v2_caller_auto_resolves_missing_stream_roots(tmp_path):
    _write_series(tmp_path / "series.csv")
    args = _base_args(tmp_path)
    del args._seed_bundle
    args.data_order_seed = None
    args.worker_seed = None

    dataset, loader = data_provider(args, "val")

    assert isinstance(loader.sampler, DeterministicEpochSampler)
    assert loader.sampler.shuffle is False
    assert list(loader.sampler) == list(range(len(dataset)))
    assert loader.reproducibility_metadata["data_order_seed"] is not None
    assert loader.reproducibility_metadata["worker_seed"] is not None
    assert loader.reproducibility_metadata["epoch0_order_hash"] is not None


def test_v2_dataset_construction_does_not_leak_augmentation_rng(tmp_path):
    _write_series(tmp_path / "series.csv")
    args = _augmentation_args(_base_args(tmp_path))

    random.seed(700)
    np.random.seed(701)
    torch.manual_seed(702)
    expected = (random.random(), np.random.random(), torch.rand(1).item())

    random.seed(700)
    np.random.seed(701)
    torch.manual_seed(702)
    dataset, loader = data_provider(args, "train")
    actual = (random.random(), np.random.random(), torch.rand(1).item())

    assert actual == expected
    augmentation = dataset.fold_manifest["augmentation"]
    assert augmentation["applied"] is True
    assert augmentation["config"]["augmentation_ratio"] == 1
    assert augmentation["config"]["flags"]["jitter"] is True
    assert augmentation["effective_seed"] == args._seed_bundle.training_seed
    assert augmentation["effective_numpy_seed"] == (
        args._seed_bundle.training_seed % (1 << 32)
    )
    assert augmentation["effective_seed"] != args.seed
    assert augmentation["seed_source"] == "training_seed"
    assert augmentation["tag"] == "_jitter"
    assert len(augmentation["transformed_data_content_hash"]) == 64
    assert dataset.fold_manifest["positional_sample_id_range"]["prefix"].startswith(
        "transformed:"
    )
    assert loader.reproducibility_metadata["epoch0_order_hash"] is not None
    assert loader.reproducibility_metadata["first_batch_index_hash"] is not None


def test_seeded_augmentation_manifest_repeats_and_changes_with_training_seed(tmp_path):
    _write_series(tmp_path / "series.csv")
    first_args = _augmentation_args(_base_args(tmp_path, seed=41), "audit")
    repeat_args = _augmentation_args(_base_args(tmp_path, seed=41), "audit")
    changed_args = _augmentation_args(_base_args(tmp_path, seed=42), "audit")

    first, first_loader = data_provider(first_args, "train")
    repeat, repeat_loader = data_provider(repeat_args, "train")
    changed, changed_loader = data_provider(changed_args, "train")

    first_aug = first.fold_manifest["augmentation"]
    repeat_aug = repeat.fold_manifest["augmentation"]
    changed_aug = changed.fold_manifest["augmentation"]
    assert first_aug == repeat_aug
    assert first_aug["tag"] == "_jitter_audit"
    assert first.fold_manifest_hash == repeat.fold_manifest_hash
    assert first_loader.reproducibility_metadata == repeat_loader.reproducibility_metadata
    assert first_aug["effective_seed"] != changed_aug["effective_seed"]
    assert (
        first_aug["transformed_data_content_hash"]
        != changed_aug["transformed_data_content_hash"]
    )
    assert first.fold_manifest_hash != changed.fold_manifest_hash
    for loader in (first_loader, repeat_loader, changed_loader):
        assert loader.reproducibility_metadata["epoch0_order_hash"] is not None
        assert loader.reproducibility_metadata["first_batch_index_hash"] is not None


def test_unaugmented_fold_identity_does_not_claim_or_depend_on_unused_seed(tmp_path):
    _write_series(tmp_path / "series.csv")
    first, _ = data_provider(_base_args(tmp_path, seed=41), "train")
    changed_seed, _ = data_provider(_base_args(tmp_path, seed=42), "train")

    first_aug = first.fold_manifest["augmentation"]
    changed_aug = changed_seed.fold_manifest["augmentation"]
    for augmentation in (first_aug, changed_aug):
        assert augmentation["applied"] is False
        assert augmentation["effective_seed"] is None
        assert augmentation["effective_numpy_seed"] is None
        assert augmentation["seed_source"] is None
    assert first_aug == changed_aug
    assert first.fold_manifest_hash == changed_seed.fold_manifest_hash


def test_local_source_sha_rehashes_same_size_file_with_restored_mtime(tmp_path):
    path = tmp_path / "series.csv"
    _write_series(path)
    args = _base_args(tmp_path)
    first, _ = data_provider(args, "val")
    first_sha = first.fold_manifest["source_identity"]["sha256"]
    original_stat = path.stat()

    original = path.read_bytes()
    marker = b",0.5\n"
    replacement = b",0.6\n"
    assert marker in original and len(marker) == len(replacement)
    path.write_bytes(original.replace(marker, replacement, 1))
    os.utime(
        path,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    assert path.stat().st_size == original_stat.st_size
    assert path.stat().st_mtime_ns == original_stat.st_mtime_ns

    second, _ = data_provider(args, "val")
    second_sha = second.fold_manifest["source_identity"]["sha256"]
    assert second_sha == hashlib.sha256(path.read_bytes()).hexdigest()
    assert second_sha != first_sha
    assert second.fold_manifest_hash != first.fold_manifest_hash


def _direct_dataset_args():
    return SimpleNamespace(augmentation_ratio=0)


def test_equal_length_ett_folds_have_distinct_truthful_hashes(tmp_path):
    frame = _write_series(tmp_path / "ETTh1.csv", rows=12 * 30 * 24 + 8 * 30 * 24)
    kwargs = {
        "args": _direct_dataset_args(),
        "root_path": str(tmp_path),
        "size": [8, 4, 3],
        "features": "M",
        "data_path": "ETTh1.csv",
        "target": "OT",
        "timeenc": 1,
        "freq": "h",
    }
    val = Dataset_ETT_hour(flag="val", **kwargs)
    test = Dataset_ETT_hour(flag="test", **kwargs)

    assert len(val) == len(test)
    assert val.fold_manifest_hash != test.fold_manifest_hash
    assert val.fold_manifest_hash == stable_json_hash(val.fold_manifest)
    assert test.fold_manifest_hash == stable_json_hash(test.fold_manifest)
    assert val.fold_manifest["split"] == "val"
    assert test.fold_manifest["split"] == "test"
    assert val.fold_manifest["first_forecast_timestamp"] == pd.Timestamp(
        frame.date.iloc[12 * 30 * 24]
    ).isoformat()
    assert test.fold_manifest["last_forecast_timestamp"] == pd.Timestamp(
        frame.date.iloc[-1]
    ).isoformat()
    assert (
        val.fold_manifest["positional_sample_id_range"]["end_inclusive"]
        == val.fold_manifest["positional_sample_id_range"]["start"] + len(val) - 1
    )


@pytest.mark.parametrize(
    ("dataset_class", "filename", "rows", "freq", "loader_freq"),
    (
        (Dataset_ETT_minute, "ETTm1.csv", 12 * 30 * 24 * 4, "15min", "t"),
        (Dataset_Custom, "custom.csv", 160, "h", "h"),
    ),
)
def test_forecast_dataset_classes_expose_content_addressed_fold_manifests(
    tmp_path, dataset_class, filename, rows, freq, loader_freq
):
    path = tmp_path / filename
    _write_series(path, rows=rows, freq=freq)
    dataset = dataset_class(
        args=_direct_dataset_args(),
        root_path=str(tmp_path),
        flag="train",
        size=[8, 4, 3],
        features="M",
        data_path=filename,
        target="OT",
        timeenc=1,
        freq=loader_freq,
    )

    manifest = dataset.fold_manifest
    assert dataset.fold_manifest_hash == stable_json_hash(manifest)
    assert manifest["source_identity"]["sha256"] == hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    assert manifest["features"]["columns"] == ["feature", "OT"]
    assert manifest["lengths"]["sample_count"] == len(dataset)
    assert manifest["first_forecast_timestamp"] is not None
    assert manifest["last_forecast_timestamp"] is not None
    assert len(dataset[0]) == 4
