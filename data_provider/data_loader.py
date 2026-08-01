import os
import hashlib
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
from utils.augmentation import augment, run_augmentation_single
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from utils.reproducibility import stable_json_hash

HUGGINGFACE_REPO = "thuml/Time-Series-Library"

_AUGMENTATION_FLAG_NAMES = (
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
)


def _local_source_identity(path):
    """Hash current bytes every time; file stat metadata is never authoritative."""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    size_bytes = os.path.getsize(path)
    return {
        "kind": "local_file",
        "path": path,
        "size_bytes": int(size_bytes),
        "sha256": digest.hexdigest(),
    }


def _resolve_local_source_identity(path):
    resolved = os.path.realpath(path)
    return _local_source_identity(resolved)


def _resolve_huggingface_source_identity(dataset, config_name, split_name):
    split = dataset[split_name]
    identity = {
        "kind": "huggingface_dataset",
        "repository": HUGGINGFACE_REPO,
        "config": str(config_name),
        "split": str(split_name),
        "row_count": int(len(split)),
    }
    fingerprint = getattr(split, "_fingerprint", None)
    if fingerprint is not None:
        identity["fingerprint"] = str(fingerprint)
    return identity


def _iso_timestamp(value):
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _array_content_sha256(value):
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256(b"time-series-library:ndarray-content:v1\n")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\n")
    digest.update(",".join(str(int(size)) for size in array.shape).encode("ascii"))
    digest.update(b"\n")
    digest.update(array.view(np.uint8).reshape(-1).tobytes(order="C"))
    return digest.hexdigest()


def _run_forecast_augmentation(dataset):
    """Run the unchanged legacy transform while retaining exact provenance."""
    augmentation_tag = str(getattr(dataset.args, "extra_tag", "") or "")
    if dataset.set_type == 0 and int(getattr(dataset.args, "augmentation_ratio", 0)) > 0:
        if getattr(dataset.args, "tft_augmentation_seed_source", None) == "training_seed":
            # `isolated_rng(training_seed)` already seeded NumPy using its
            # portable uint32 projection. Calling the legacy wrapper here would
            # overwrite that state with the original CLI seed (and cannot
            # accept Torch's full 63-bit derived seed range).
            x_input = dataset.data_x[np.newaxis, :]
            transformed = x_input
            for _ in range(int(dataset.args.augmentation_ratio)):
                transformed, augmentation_tag = augment(
                    x_input, dataset.data_y, dataset.args
                )
            if getattr(dataset.args, "extra_tag", ""):
                augmentation_tag += "_" + str(dataset.args.extra_tag)
            dataset.data_x = transformed.squeeze(0)
        else:
            dataset.data_x, dataset.data_y, augmentation_tag = run_augmentation_single(
                dataset.data_x, dataset.data_y, dataset.args
            )
    dataset.augmentation_tag = str(augmentation_tag)


def _augmentation_manifest(dataset):
    ratio = int(getattr(dataset.args, "augmentation_ratio", 0))
    applied = int(dataset.set_type) == 0 and ratio > 0
    config = {
        "augmentation_ratio": ratio,
        "extra_tag": str(getattr(dataset.args, "extra_tag", "") or ""),
        "flags": {
            name: bool(getattr(dataset.args, name, False))
            for name in _AUGMENTATION_FLAG_NAMES
        },
    }
    effective_seed = getattr(dataset.args, "seed", None) if applied else None
    if effective_seed is not None:
        effective_seed = int(effective_seed)
    seed_source = (
        str(getattr(dataset.args, "tft_augmentation_seed_source", "args.seed"))
        if applied
        else None
    )
    content_hashes = {
        "data_x_sha256": _array_content_sha256(dataset.data_x),
        "data_y_sha256": _array_content_sha256(dataset.data_y),
    }
    content_hashes["combined_sha256"] = stable_json_hash(content_hashes)
    return {
        "applied": applied,
        "config": config,
        "config_hash": stable_json_hash(config),
        "effective_seed": effective_seed,
        "effective_numpy_seed": (
            effective_seed % (1 << 32) if effective_seed is not None else None
        ),
        "seed_source": seed_source,
        "tag": str(getattr(dataset, "augmentation_tag", "")),
        "transformed_data_content_hash": content_hashes["combined_sha256"],
        "content_hashes": content_hashes,
    }


def _attach_forecast_fold_manifest(dataset, df_raw, source_identity, border1, border2):
    """Attach a truthful, JSON-safe split manifest without changing samples.

    ``border1`` includes the look-back overlap for validation and test folds.
    The separately recorded forecast interval excludes that overlap.
    """
    source_rows = int(len(df_raw))
    split_rows = int(len(dataset.data_x))
    sample_count = max(0, split_rows - dataset.seq_len - dataset.pred_len + 1)
    augmentation = _augmentation_manifest(dataset)

    effective_start = min(max(int(border1), 0), source_rows)
    effective_end = min(effective_start + split_rows, source_rows)
    first_forecast_pos = effective_start + int(dataset.seq_len) if sample_count else None
    last_forecast_pos = effective_end - 1 if sample_count else None

    dates = df_raw["date"] if "date" in df_raw.columns else None
    first_forecast_timestamp = None
    last_forecast_timestamp = None
    if dates is not None and first_forecast_pos is not None:
        if 0 <= first_forecast_pos < source_rows:
            first_forecast_timestamp = _iso_timestamp(dates.iloc[first_forecast_pos])
        if 0 <= last_forecast_pos < source_rows:
            last_forecast_timestamp = _iso_timestamp(dates.iloc[last_forecast_pos])

    positional_ids_supported = sample_count > 0
    sample_id_start = effective_start if positional_ids_supported else None
    sample_id_end = (
        effective_start + sample_count - 1 if positional_ids_supported else None
    )
    feature_names = [str(name) for name in getattr(dataset, "feature_names", ())]
    split_name = {0: "train", 1: "val", 2: "test"}[int(dataset.set_type)]
    sample_id_prefix = None
    if augmentation["applied"]:
        sample_id_prefix = (
            "transformed:"
            + str(augmentation["transformed_data_content_hash"])
        )

    manifest = {
        "schema_version": 1,
        "dataset_class": type(dataset).__name__,
        "split": split_name,
        "source_identity": dict(source_identity),
        "split_boundaries": {
            "declared_start_pos": int(border1),
            "declared_end_exclusive_pos": int(border2),
            "effective_start_pos": effective_start,
            "effective_end_exclusive_pos": effective_end,
            "forecast_start_pos": first_forecast_pos,
            "forecast_end_exclusive_pos": effective_end if sample_count else None,
        },
        "lengths": {
            "source_rows": source_rows,
            "split_rows_including_history_overlap": split_rows,
            "sample_count": sample_count,
            "seq_len": int(dataset.seq_len),
            "label_len": int(dataset.label_len),
            "pred_len": int(dataset.pred_len),
        },
        "features": {
            "mode": str(dataset.features),
            "target": str(dataset.target),
            "columns": feature_names,
        },
        "augmentation": augmentation,
        "first_forecast_timestamp": first_forecast_timestamp,
        "last_forecast_timestamp": last_forecast_timestamp,
        "positional_sample_id_range": {
            "prefix": sample_id_prefix,
            "start": sample_id_start,
            "end_inclusive": sample_id_end,
        },
    }
    coordinate_manifest = getattr(dataset, "temporal_coordinate_manifest", None)
    if coordinate_manifest is not None:
        manifest["temporal_coordinates"] = dict(coordinate_manifest)
    dataset.fold_manifest = manifest
    dataset.fold_manifest_hash = stable_json_hash(manifest)
    dataset.positional_sample_id_start = sample_id_start if sample_id_start is not None else 0
    dataset.positional_sample_id_end = sample_id_end
    dataset.positional_sample_ids_supported = positional_ids_supported
    dataset.positional_sample_id_prefix = sample_id_prefix


def _configure_tft_temporal_coordinates(dataset, df_raw, border1, border2):
    """Attach physical coordinates for native semantics-v2 explicit batches.

    The stored array is aligned with ``data_x`` rather than the full source.
    Calendar-day coordinates retain weekend/holiday gaps; trading-session and
    step coordinates intentionally count source rows.
    """

    args = dataset.args
    if (
        getattr(args, "model", None) != "TemporalFusionTransformer"
        or int(getattr(args, "tft_extension_semantics_version", 1)) != 2
        or str(getattr(args, "tft_position_source", "row_index"))
        != "explicit_argument"
    ):
        return
    if "date" not in df_raw.columns:
        raise ValueError(
            "Native TFT explicit temporal coordinates require a date column."
        )

    timestamps = pd.DatetimeIndex(
        pd.to_datetime(df_raw["date"], errors="raise", utc=True)
    )
    if timestamps.hasnans:
        raise ValueError("TFT temporal-coordinate dates must not contain NaT.")
    timestamp_ns = timestamps.asi8
    if len(timestamp_ns) > 1 and np.any(np.diff(timestamp_ns) <= 0):
        raise ValueError(
            "TFT temporal-coordinate dates must be strictly increasing."
        )

    unit = str(getattr(args, "tft_position_unit", "steps"))
    if unit == "calendar_days":
        full_positions = (
            timestamp_ns - timestamp_ns[0]
        ).astype(np.float64) / (24.0 * 60.0 * 60.0 * 1e9)
    elif unit in {"steps", "trading_sessions"}:
        full_positions = np.arange(len(df_raw), dtype=np.float64)
    else:
        raise ValueError(
            "tft_position_unit must be steps, trading_sessions, or calendar_days."
        )

    split_positions = np.asarray(
        full_positions[int(border1):int(border2)], dtype=np.float64
    )
    if split_positions.shape[0] != len(dataset.data_x):
        raise RuntimeError(
            "Temporal coordinates no longer align with the transformed forecast split."
        )
    dataset.data_temporal_positions = split_positions
    dataset.data_temporal_valid_mask = np.ones(
        split_positions.shape, dtype=np.bool_
    )
    dataset.temporal_coordinate_manifest = {
        "schema_version": 1,
        "source": "explicit_argument",
        "unit": unit,
        "valid_mask_semantics": "true_is_valid",
        "origin_timestamp": timestamps[0].isoformat(),
        "split_position_start": (
            float(split_positions[0]) if split_positions.size else None
        ),
        "split_position_end": (
            float(split_positions[-1]) if split_positions.size else None
        ),
    }


def _with_tft_temporal_coordinates(dataset, s_begin, s_end, sample):
    positions = getattr(dataset, "data_temporal_positions", None)
    if positions is None:
        return sample
    forecast_end = s_end + dataset.pred_len
    temporal_positions = positions[s_begin:forecast_end]
    temporal_valid_mask = dataset.data_temporal_valid_mask[
        s_begin:forecast_end
    ]
    expected = dataset.seq_len + dataset.pred_len
    if temporal_positions.shape[0] != expected:
        raise RuntimeError(
            "TFT sample temporal-coordinate length does not match seq_len+pred_len."
        )
    return (*sample, temporal_positions, temporal_valid_mask)

class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        local_fp = os.path.join(self.root_path, self.data_path)
        cfg_name = os.path.splitext(os.path.basename(self.data_path))[0]

        if os.path.exists(local_fp):
            df_raw = pd.read_csv(local_fp)
            source_identity = _resolve_local_source_identity(local_fp)
        else:
            ds = load_dataset(HUGGINGFACE_REPO, name=cfg_name)
            df_raw = ds["train"].to_pandas()
            source_identity = _resolve_huggingface_source_identity(ds, cfg_name, "train")
            
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        self.feature_names = tuple(df_data.columns.tolist())

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        _run_forecast_augmentation(self)

        self.data_stamp = data_stamp
        _configure_tft_temporal_coordinates(
            self, df_raw, border1=border1, border2=border2
        )
        _attach_forecast_fold_manifest(
            self, df_raw, source_identity, border1=border1, border2=border2
        )

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return _with_tft_temporal_coordinates(
            self,
            s_begin,
            s_end,
            (seq_x, seq_y, seq_x_mark, seq_y_mark),
        )

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        
        local_fp = os.path.join(self.root_path, self.data_path)
        cfg_name = os.path.splitext(os.path.basename(self.data_path))[0]

        if os.path.exists(local_fp):
            df_raw = pd.read_csv(local_fp)
            source_identity = _resolve_local_source_identity(local_fp)
        else:
            ds = load_dataset(HUGGINGFACE_REPO, name=cfg_name)
            df_raw = ds["train"].to_pandas()
            source_identity = _resolve_huggingface_source_identity(ds, cfg_name, "train")

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        self.feature_names = tuple(df_data.columns.tolist())

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        _run_forecast_augmentation(self)

        self.data_stamp = data_stamp
        _configure_tft_temporal_coordinates(
            self, df_raw, border1=border1, border2=border2
        )
        _attach_forecast_fold_manifest(
            self, df_raw, source_identity, border1=border1, border2=border2
        )

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return _with_tft_temporal_coordinates(
            self,
            s_begin,
            s_end,
            (seq_x, seq_y, seq_x_mark, seq_y_mark),
        )

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        local_fp = os.path.join(self.root_path, self.data_path)
        cfg_name = os.path.splitext(os.path.basename(self.data_path))[0]

        if os.path.exists(local_fp):
            df_raw = pd.read_csv(local_fp)
            source_identity = _resolve_local_source_identity(local_fp)
        else:
            ds = load_dataset(HUGGINGFACE_REPO, name=cfg_name)
            split_name = "train" if "train" in ds else list(ds.keys())[0]
            df_raw = ds[split_name].to_pandas()
            source_identity = _resolve_huggingface_source_identity(ds, cfg_name, split_name)

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        self.feature_names = tuple(df_data.columns.tolist())

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        _run_forecast_augmentation(self)

        self.data_stamp = data_stamp
        _configure_tft_temporal_coordinates(
            self, df_raw, border1=border1, border2=border2
        )
        _attach_forecast_fold_manifest(
            self, df_raw, source_identity, border1=border1, border2=border2
        )

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return _with_tft_temporal_coordinates(
            self,
            s_begin,
            s_end,
            (seq_x, seq_y, seq_x_mark, seq_y_mark),
        )

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, args, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           max(0, cut_point - self.label_len):min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class PSMSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        train_path = os.path.join(root_path, "train.csv")
        test_path = os.path.join(root_path, "test.csv")
        label_path = os.path.join(root_path, "test_label.csv")

        if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_df      = pd.read_csv(train_path)
            test_df       = pd.read_csv(test_path)
            test_label_df = pd.read_csv(label_path)
        else:
            ds_data  = load_dataset(HUGGINGFACE_REPO, name="PSM-data")
            ds_label = load_dataset(HUGGINGFACE_REPO, name="PSM-label")
            train_df      = ds_data["train"].to_pandas()
            test_df       = ds_data["test"].to_pandas()
            test_label_df = ds_label[next(iter(ds_label))].to_pandas()

        data = train_df.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        
        test_data = test_df.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = test_label_df.values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        
        train_path = os.path.join(root_path, "MSL_train.npy")
        test_path  = os.path.join(root_path, "MSL_test.npy")
        label_path = os.path.join(root_path, "MSL_test_label.npy")

        if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_data = np.load(train_path)
            test_data  = np.load(test_path)
            test_label = np.load(label_path)
        else:
            train_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="MSL/MSL_train.npy",repo_type="dataset")
            test_path  = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="MSL/MSL_test.npy",repo_type="dataset")
            label_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="MSL/MSL_test_label.npy",repo_type="dataset")

            train_data  = np.load(train_path)
            test_data   = np.load(test_path)
            test_label  = np.load(label_path)

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data  = self.scaler.transform(test_data)

        self.train = train_data
        self.test  = test_data
        self.test_labels = test_label

        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8):]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        
        train_path = os.path.join(root_path, "SMAP_train.npy")
        test_path  = os.path.join(root_path, "SMAP_test.npy")
        label_path = os.path.join(root_path, "SMAP_test_label.npy")

        if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_data = np.load(train_path)
            test_data  = np.load(test_path)
            test_label = np.load(label_path)
        else:
            train_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMAP/SMAP_train.npy",repo_type="dataset")
            test_path  = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMAP/SMAP_test.npy",repo_type="dataset")
            label_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMAP/SMAP_test_label.npy",repo_type="dataset")

            train_data  = np.load(train_path)
            test_data   = np.load(test_path)
            test_label = np.load(label_path)

        # 标准化
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data  = self.scaler.transform(test_data)

        self.train = train_data
        self.test  = test_data
        self.test_labels = test_label

        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8):]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        
        train_path = os.path.join(root_path, "SMD_train.npy")
        test_path  = os.path.join(root_path, "SMD_test.npy")
        label_path = os.path.join(root_path, "SMD_test_label.npy")

        if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_data = np.load(train_path)
            test_data  = np.load(test_path)
            test_label = np.load(label_path)
        else:
            train_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMD/SMD_train.npy",repo_type="dataset")
            test_path  = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMD/SMD_test.npy",repo_type="dataset")
            label_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMD/SMD_test_label.npy",repo_type="dataset")

            train_data  = np.load(train_path)
            test_data   = np.load(test_path)
            test_label = np.load(label_path)
            
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = test_label
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train2_path = os.path.join(root_path, "swat_train2.csv")
        test_path   = os.path.join(root_path, "swat2.csv")
        if all(os.path.exists(p) for p in [train2_path, test_path]):
            train_data = pd.read_csv(train2_path)
            test_data   = pd.read_csv(test_path)
        else:
            ds = load_dataset(HUGGINGFACE_REPO, name="SWaT")
            train_data = ds["train"].to_pandas()
            test_data  = ds["test"].to_pandas()
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def _resolve_ts_path(self, root_path, dataset_name, flag):
        split = "TRAIN" if "train" in str(flag).lower() else "TEST"
        fname = f"{dataset_name}_{split}.ts"
        local = os.path.join(root_path, fname)
        if os.path.exists(local):
            return local
        return hf_hub_download(HUGGINGFACE_REPO, filename=f"{dataset_name}/{fname}", repo_type="dataset")

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from ts files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .ts files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        dataset_name = self.args.model_id
        ts_path = self._resolve_ts_path(root_path, dataset_name, flag or "train")

        all_df, labels_df = self.load_single(ts_path)
        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                             replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values
        labels = self.labels_df.loc[self.all_IDs[ind]].values
        if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
            num_samples = len(self.all_IDs)
            num_columns = self.feature_df.shape[1]
            seq_len = int(self.feature_df.shape[0] / num_samples)
            batch_x = batch_x.reshape((1, seq_len, num_columns))
            batch_x, labels, augmentation_tags = run_augmentation_single(batch_x, labels, self.args)

            batch_x = batch_x.reshape((1 * seq_len, num_columns))

        return self.instance_norm(torch.from_numpy(batch_x)), \
               torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)
