from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import torch
from pandas.tseries.frequencies import to_offset

from utils.timefeatures import time_features_from_frequency_str


_DATASET_DEFAULT_OBSERVED = {
    "ETTh1": tuple(range(7)),
    "ETTh2": tuple(range(7)),
    "ETTm1": tuple(range(7)),
    "ETTm2": tuple(range(7)),
}


@dataclass(frozen=True)
class ResolvedTFTSchema:
    feature_names: tuple[str, ...]
    observed_positions: tuple[int, ...]
    static_positions: tuple[int, ...]
    target_positions: tuple[int, ...]
    known_feature_names: tuple[str, ...]
    features_mode: Literal["M", "MS", "S"]
    enc_in: int
    c_out: int


def is_tft_model(args) -> bool:
    return getattr(args, "model", None) == "TemporalFusionTransformer"


def _validate_index_list(name: str, values, upper_bound: int) -> tuple[int, ...]:
    if values is None:
        return tuple()
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple of integers.")
    resolved = tuple(int(v) for v in values)
    for value in resolved:
        if value < 0 or value >= upper_bound:
            raise ValueError(f"{name} value {value} is outside [0, {upper_bound - 1}].")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{name} contains duplicated indices.")
    return resolved


def resolve_known_feature_names(
    embed_type: str,
    freq: str,
    *,
    allow_custom_known: bool = False,
    known_len: int | None = None,
    custom_names: Iterable[str] | None = None,
) -> tuple[str, ...]:
    if allow_custom_known:
        if known_len is None or int(known_len) <= 0:
            raise ValueError("tft_known_len must be a positive integer when tft_allow_custom_known=True.")
        if custom_names is None:
            raise ValueError("tft_known_feature_names is required when tft_allow_custom_known=True.")
        names = tuple(str(name) for name in custom_names)
        if len(names) != int(known_len):
            raise ValueError(
                f"tft_known_feature_names length ({len(names)}) must equal tft_known_len ({int(known_len)})."
            )
        if len(set(names)) != len(names):
            raise ValueError("tft_known_feature_names must be unique.")
        return names

    if embed_type == "timeF":
        return tuple(feature.__class__.__name__ for feature in time_features_from_frequency_str(freq))

    offset = to_offset(freq)
    if offset.name in {"min", "T", "s", "S"}:
        return ("month", "day", "weekday", "hour", "minute")
    return ("month", "day", "weekday", "hour")


def resolve_tft_schema(args, feature_names: Iterable[str] | None = None) -> ResolvedTFTSchema:
    enc_in = int(getattr(args, "enc_in"))
    c_out = int(getattr(args, "c_out"))
    features_mode = str(getattr(args, "features", "M"))
    if features_mode not in {"M", "MS", "S"}:
        raise ValueError(f"Unsupported TFT features mode: {features_mode!r}.")

    if feature_names is None:
        feature_names = getattr(args, "tft_feature_names", None)
    if feature_names is None:
        feature_names = tuple(f"f{i}" for i in range(enc_in))
    else:
        feature_names = tuple(str(name) for name in feature_names)
    if len(feature_names) != enc_in:
        raise ValueError(f"feature_names length ({len(feature_names)}) must equal enc_in ({enc_in}).")

    explicit_observed = getattr(args, "tft_observed_pos", None)
    explicit_static = getattr(args, "tft_static_pos", None)
    explicit_target = getattr(args, "tft_target_pos", None)

    if explicit_observed is not None:
        observed_positions = _validate_index_list("tft_observed_pos", explicit_observed, enc_in)
    elif features_mode == "S" and enc_in == 1:
        observed_positions = (0,)
    else:
        observed_positions = _DATASET_DEFAULT_OBSERVED.get(getattr(args, "data", ""), tuple())
        observed_positions = _validate_index_list("tft_observed_pos", observed_positions, enc_in)
        if not observed_positions:
            raise KeyError(
                f"Dataset '{getattr(args, 'data', None)}' is not registered in datatype_dict. "
                "You must provide tft_observed_pos explicitly."
            )

    static_positions = _validate_index_list("tft_static_pos", explicit_static or tuple(), enc_in)
    if set(static_positions) & set(observed_positions):
        overlap = sorted(set(static_positions) & set(observed_positions))
        raise ValueError(f"tft_static_pos and tft_observed_pos overlap at indices {overlap}.")

    target_positions = resolve_target_positions(args)
    if any(target not in observed_positions for target in target_positions):
        missing = [target for target in target_positions if target not in observed_positions]
        raise ValueError(f"Every tft_target_pos must also be historically observed; missing {missing}.")

    known_names = resolve_known_feature_names(
        getattr(args, "embed", "timeF"),
        getattr(args, "freq", "h"),
        allow_custom_known=getattr(args, "tft_allow_custom_known", False),
        known_len=getattr(args, "tft_known_len", None),
        custom_names=getattr(args, "tft_known_feature_names", None),
    )

    return ResolvedTFTSchema(
        feature_names=feature_names,
        observed_positions=observed_positions,
        static_positions=static_positions,
        target_positions=target_positions,
        known_feature_names=known_names,
        features_mode=features_mode,
        enc_in=enc_in,
        c_out=c_out,
    )


def resolve_target_positions(args) -> tuple[int, ...]:
    enc_in = int(getattr(args, "enc_in"))
    c_out = int(getattr(args, "c_out"))
    target_pos = getattr(args, "tft_target_pos", None)

    if target_pos is None:
        if c_out == enc_in:
            return tuple(range(c_out))
        if c_out == 1 and enc_in > 1:
            return (enc_in - 1,)
        raise ValueError(
            "tft_target_pos is required when c_out != enc_in for TemporalFusionTransformer."
        )

    if not isinstance(target_pos, (list, tuple)):
        raise TypeError("tft_target_pos must be a list or tuple of integers.")
    if len(target_pos) != c_out:
        raise ValueError(f"tft_target_pos must have length c_out={c_out}.")

    resolved = tuple(int(pos) for pos in target_pos)
    for pos in resolved:
        if pos < 0 or pos >= enc_in:
            raise ValueError(f"tft_target_pos value {pos} is outside [0, {enc_in - 1}].")
    if len(set(resolved)) != len(resolved):
        raise ValueError("tft_target_pos contains duplicated indices.")
    return resolved


def _target_index_tensor(
    target_positions: Iterable[int],
    device: torch.device | None = None,
) -> torch.Tensor:
    return torch.as_tensor(tuple(target_positions), dtype=torch.long, device=device)


def select_tft_truth(
    batch_y: torch.Tensor,
    pred_len: int,
    target_positions: Iterable[int],
) -> torch.Tensor:
    if batch_y.ndim != 3:
        raise ValueError(f"batch_y must be rank-3 [B,T,C], got shape {tuple(batch_y.shape)}.")
    truth = batch_y[:, -pred_len:, :]
    index = _target_index_tensor(target_positions, device=truth.device)
    selected = truth.index_select(-1, index)
    if selected.shape[-1] != index.numel():
        raise AssertionError("Selected TFT truth width does not match resolved target positions.")
    return selected


def inverse_transform_selected(
    values,
    scaler,
    target_positions: Iterable[int],
):
    target_positions = tuple(target_positions)
    target_array = np.asarray(target_positions, dtype=np.int64)

    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        mean = np.asarray(scaler.mean_, dtype=np.float64)[target_array]
        scale = np.asarray(scaler.scale_, dtype=np.float64)[target_array]
    elif hasattr(scaler, "mean") and hasattr(scaler, "std"):
        mean = np.asarray(scaler.mean, dtype=np.float64)[target_array]
        scale = np.asarray(scaler.std, dtype=np.float64)[target_array]
    else:
        raise TypeError("Scaler must expose mean_/scale_ or mean/std for TFT inverse transforms.")

    if torch.is_tensor(values):
        mean_tensor = torch.as_tensor(mean, dtype=values.dtype, device=values.device)
        scale_tensor = torch.as_tensor(scale, dtype=values.dtype, device=values.device)
        return values * scale_tensor + mean_tensor

    values_array = np.asarray(values)
    return values_array * scale + mean
