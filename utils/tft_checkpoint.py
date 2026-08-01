from pathlib import Path
import re

import torch

from utils.tft_config import validate_tft_checkpoint_compatibility


def _align_data_parallel_prefix(state_dict, model):
    """Normalize a uniform ``module.`` wrapper without weakening strict load.

    ``nn.DataParallel`` changes only state-dict names, not model semantics.  We
    permit that one reversible topology wrapper when—and only when—the
    normalized key set exactly matches the receiving model.  Mixed prefixes,
    collisions, and every other mismatch still fail through strict loading.
    """

    source_keys = tuple(state_dict)
    target_keys = tuple(model.state_dict())
    if set(source_keys) == set(target_keys) or not source_keys:
        return state_dict

    source_is_wrapped = all(key.startswith("module.") for key in source_keys)
    target_is_wrapped = bool(target_keys) and all(
        key.startswith("module.") for key in target_keys
    )

    if source_is_wrapped and not target_is_wrapped:
        normalized = {key[len("module."):]: value for key, value in state_dict.items()}
    elif target_is_wrapped and not source_is_wrapped:
        normalized = {f"module.{key}": value for key, value in state_dict.items()}
    else:
        return state_dict

    if len(normalized) != len(state_dict) or set(normalized) != set(target_keys):
        return state_dict
    return normalized


def legacy_setting_id(setting):
    """Return the pre-SR00 setting ID for an explicit semantics-v1 ID."""

    return re.sub(r"_tsv1_(?=td[0-9a-f]+$)", "_", str(setting), count=1)


def resolve_tft_checkpoint_path(checkpoints_root, setting, args):
    """Resolve a current setting, or a guarded historical no-``_tsv`` path."""

    root = Path(checkpoints_root)
    current = root / str(setting) / "checkpoint.pth"
    if current.is_file():
        return current

    semantics_version = int(getattr(args, "tft_extension_semantics_version", 2))
    allow_legacy = bool(
        getattr(args, "tft_allow_legacy_extension_checkpoint", False)
    )
    if semantics_version == 1 and allow_legacy:
        historical_setting = legacy_setting_id(setting)
        historical = root / historical_setting / "checkpoint.pth"
        if historical.is_file():
            return historical
    return current


def load_tft_checkpoint(
    model,
    checkpoint_path,
    args,
    *,
    map_location=None,
    external_load=True,
    strict=True,
    expected_setting=None,
):
    """Load a native-TFT artifact only after semantic-version validation.

    Low-level in-memory ``load_state_dict`` remains unrestricted for paired
    initialization and unit tests. This helper protects filesystem artifact
    boundaries, where a missing sidecar is deliberately classified as v1.
    """

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"TFT checkpoint does not exist: {checkpoint_path}")
    policy = validate_tft_checkpoint_compatibility(
        args,
        checkpoint_path,
        external_load=external_load,
        expected_setting=expected_setting,
    )
    state_dict = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=True,
    )
    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Expected a bare state_dict mapping in {checkpoint_path}, "
            f"got {type(state_dict).__name__}."
        )
    state_dict = _align_data_parallel_prefix(state_dict, model)
    model.load_state_dict(state_dict, strict=strict)
    return policy
