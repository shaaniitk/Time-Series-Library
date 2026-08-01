"""Reproducibility primitives for native TFT experiments.

The helpers in this module make randomness explicit and auditable.  Hashes of
tensor state are intended to prove exact identity within an experiment and its
recorded software/hardware environment; they do not promise cross-hardware
bitwise identity.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import struct
import tempfile
from argparse import Namespace
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Mapping, Pattern, Sequence, TypeVar

import numpy as np
import torch
from torch import nn
from torch.utils.data import Sampler


SEED_DERIVATION_VERSION = 1
SUPPORTED_SEED_DERIVATION_VERSIONS = (SEED_DERIVATION_VERSION,)
_MAX_SEED = (1 << 63) - 1
_NUMPY_SEED_MODULUS = 1 << 32
_DETERMINISTIC_MODES = ("off", "warn", "strict")
_STABLE_JSON_HASH_VERSION = 1
_STATE_DICT_HASH_VERSION = 1


@dataclass(frozen=True)
class SeedBundle:
    """Resolved, JSON-serializable seed schedule for one experiment iteration."""

    derivation_version: int
    base_seed: int
    iteration: int
    experiment_seed: int
    model_init_seed: int
    extension_init_seed: int
    data_order_seed: int
    worker_seed: int
    training_seed: int


def _validated_seed(seed: Any, *, field: str = "seed") -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"{field} must be an integer, got {type(seed).__name__}.")
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"{field} must be non-negative, got {seed}.")
    return seed


def derive_seed(
    base_seed: int,
    stream_name: str,
    iteration: int = 0,
    version: int = SEED_DERIVATION_VERSION,
) -> int:
    """Derive a stable, independent integer seed using a versioned SHA-256 stream.

    The output is constrained to Torch's portable non-negative 63-bit range.
    Changing the stream name, iteration, base seed, or derivation version changes
    the output without consuming any process-global random state.
    """

    base_seed = _validated_seed(base_seed, field="base_seed")
    iteration = _validated_seed(iteration, field="iteration")
    version = _validated_seed(version, field="version")
    if version not in SUPPORTED_SEED_DERIVATION_VERSIONS:
        raise ValueError(
            f"Unsupported seed derivation version {version}; supported versions: "
            f"{SUPPORTED_SEED_DERIVATION_VERSIONS}."
        )
    if not isinstance(stream_name, str) or not stream_name.strip():
        raise ValueError("stream_name must be a non-empty string.")

    payload = (
        f"time-series-library:tft-seed-stream:v{version}\n"
        f"base_seed={base_seed}\n"
        f"iteration={iteration}\n"
        f"stream={stream_name}\n"
    ).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % _MAX_SEED


def _optional_seed_root(args: Any, name: str, default: int) -> int:
    value = getattr(args, name, None)
    return default if value is None else _validated_seed(value, field=name)


def resolve_seed_bundle(args: Any, iteration: int = 0) -> SeedBundle:
    """Resolve all independent seed streams for an experiment iteration.

    A stream-specific CLI value is a root seed, not a final seed: the iteration
    is still mixed into it.  Consequently ``itr > 1`` has an explicit schedule
    even when the user pins individual stream roots.
    """

    iteration = _validated_seed(iteration, field="iteration")
    base_seed = _validated_seed(getattr(args, "seed", 2021), field="seed")
    version = _validated_seed(
        getattr(args, "seed_derivation_version", SEED_DERIVATION_VERSION),
        field="seed_derivation_version",
    )
    if version not in SUPPORTED_SEED_DERIVATION_VERSIONS:
        raise ValueError(
            f"Unsupported seed derivation version {version}; supported versions: "
            f"{SUPPORTED_SEED_DERIVATION_VERSIONS}."
        )

    roots = {
        "model_init": _optional_seed_root(args, "model_init_seed", base_seed),
        "extension_init": _optional_seed_root(
            args, "extension_init_seed", base_seed
        ),
        "data_order": _optional_seed_root(args, "data_order_seed", base_seed),
        "worker": _optional_seed_root(args, "worker_seed", base_seed),
        "training": _optional_seed_root(args, "training_seed", base_seed),
    }
    return SeedBundle(
        derivation_version=version,
        base_seed=base_seed,
        iteration=iteration,
        experiment_seed=derive_seed(base_seed, "experiment", iteration, version),
        model_init_seed=derive_seed(
            roots["model_init"], "model_init", iteration, version
        ),
        extension_init_seed=derive_seed(
            roots["extension_init"], "extension_init", iteration, version
        ),
        data_order_seed=derive_seed(
            roots["data_order"], "data_order", iteration, version
        ),
        worker_seed=derive_seed(roots["worker"], "worker", iteration, version),
        training_seed=derive_seed(
            roots["training"], "training", iteration, version
        ),
    )


def _normalize_deterministic_mode(mode: str) -> str:
    if not isinstance(mode, str):
        raise TypeError(
            f"deterministic_mode must be one of {_DETERMINISTIC_MODES}, "
            f"got {type(mode).__name__}."
        )
    mode = mode.strip().lower()
    if mode not in _DETERMINISTIC_MODES:
        raise ValueError(
            f"deterministic_mode must be one of {_DETERMINISTIC_MODES}, got {mode!r}."
        )
    return mode


def _seed_accelerators(seed: int) -> None:
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and hasattr(xpu, "is_available") and xpu.is_available():
        xpu.manual_seed_all(seed)
    mps = getattr(torch, "mps", None)
    if mps is not None and hasattr(mps, "manual_seed"):
        backend = getattr(torch.backends, "mps", None)
        if backend is not None and backend.is_available():
            mps.manual_seed(seed)


def set_global_seed(seed: int, deterministic_mode: str = "warn") -> int:
    """Seed Python, NumPy, Torch CPU/accelerators and set backend policy.

    ``warn`` requests deterministic algorithms but turns unsupported-operation
    failures into warnings.  ``strict`` makes those failures errors.  ``off``
    disables Torch's deterministic-algorithm guard.  The function cannot alter
    Python's hash randomization for the already-running interpreter; setting
    ``PYTHONHASHSEED`` only makes the value explicit for child processes.
    """

    seed = _validated_seed(seed)
    mode = _normalize_deterministic_mode(deterministic_mode)
    random.seed(seed)
    np.random.seed(seed % _NUMPY_SEED_MODULUS)
    torch.manual_seed(seed)
    _seed_accelerators(seed)
    os.environ["PYTHONHASHSEED"] = str(seed % _NUMPY_SEED_MODULUS)

    cudnn = getattr(torch.backends, "cudnn", None)
    if mode == "off":
        torch.use_deterministic_algorithms(False)
        if cudnn is not None:
            cudnn.deterministic = False
    else:
        # Required by deterministic CUDA matrix multiplication on supported
        # CUDA/ROCm builds.  setdefault respects an explicit operator choice.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=mode == "warn")
        if cudnn is not None:
            cudnn.deterministic = True
            cudnn.benchmark = False
    return seed


def set_experiment_seed(
    args_or_bundle: Any,
    iteration: int = 0,
    selected_seed: int | None = None,
    deterministic_mode: str | None = None,
) -> SeedBundle:
    """Resolve an experiment seed bundle and apply one selected global seed.

    By default the bundle's ``experiment_seed`` is selected. Callers may pass
    ``model_init_seed`` before construction and ``training_seed`` afterward to
    isolate model creation from training-time randomness. When a resolved
    ``SeedBundle`` is supplied, pass ``deterministic_mode`` explicitly whenever
    the experiment policy is not the conservative ``warn`` default.
    """

    if isinstance(args_or_bundle, SeedBundle):
        bundle = args_or_bundle
        mode = "warn" if deterministic_mode is None else deterministic_mode
    else:
        bundle = resolve_seed_bundle(args_or_bundle, iteration=iteration)
        mode = (
            getattr(args_or_bundle, "deterministic_mode", "warn")
            if deterministic_mode is None
            else deterministic_mode
        )
    selected = bundle.experiment_seed if selected_seed is None else selected_seed
    set_global_seed(selected, deterministic_mode=mode)
    return bundle


def _capture_accelerator_rng_states() -> list[tuple[str, Any, Callable[[Any], None]]]:
    states: list[tuple[str, Any, Callable[[Any], None]]] = []
    if torch.cuda.is_available():
        states.append(("cuda", torch.cuda.get_rng_state_all(), torch.cuda.set_rng_state_all))

    xpu = getattr(torch, "xpu", None)
    if (
        xpu is not None
        and hasattr(xpu, "is_available")
        and xpu.is_available()
        and hasattr(xpu, "get_rng_state_all")
        and hasattr(xpu, "set_rng_state_all")
    ):
        states.append(("xpu", xpu.get_rng_state_all(), xpu.set_rng_state_all))

    mps = getattr(torch, "mps", None)
    mps_backend = getattr(torch.backends, "mps", None)
    if (
        mps is not None
        and mps_backend is not None
        and mps_backend.is_available()
        and hasattr(mps, "get_rng_state")
        and hasattr(mps, "set_rng_state")
    ):
        states.append(("mps", mps.get_rng_state(), mps.set_rng_state))
    return states


@contextmanager
def isolated_rng(seed: int) -> Iterator[None]:
    """Temporarily seed all RNGs and restore their exact entry states on exit."""

    seed = _validated_seed(seed)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    accelerator_states = _capture_accelerator_rng_states()
    try:
        random.seed(seed)
        np.random.seed(seed % _NUMPY_SEED_MODULUS)
        torch.manual_seed(seed)
        _seed_accelerators(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        for _, state, restore in accelerator_states:
            restore(state)


def _json_ready(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, (Namespace, SimpleNamespace)):
        return _json_ready(vars(value))
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    "Stable JSON mappings require string keys; "
                    f"got {type(key).__name__}."
                )
            result[key] = _json_ready(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_json_ready(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ),
        )
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _json_ready(value.value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def stable_json_hash(value: Any) -> str:
    """Return a domain-separated SHA-256 hash of canonical JSON content."""

    digest = hashlib.sha256(
        f"time-series-library:stable-json:v{_STABLE_JSON_HASH_VERSION}\n".encode(
            "ascii"
        )
    )
    digest.update(_canonical_json_bytes(value))
    return digest.hexdigest()


def _update_length_prefixed(digest: Any, value: bytes) -> None:
    digest.update(struct.pack(">Q", len(value)))
    digest.update(value)


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    tensor = tensor.detach().cpu().contiguous()
    if tensor.layout != torch.strided:
        raise TypeError(
            f"state_dict_sha256 supports dense strided tensors, got {tensor.layout}."
        )
    if tensor.numel() == 0:
        return b""
    # ``view(dtype)`` rejects zero-dimensional tensors when element sizes
    # differ, so flatten before reinterpreting the storage as bytes.
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash state names, dtypes, shapes, and exact local tensor bytes."""

    if not isinstance(state_dict, Mapping):
        raise TypeError("state_dict must be a mapping of names to tensors.")
    digest = hashlib.sha256(
        f"time-series-library:state-dict:v{_STATE_DICT_HASH_VERSION}\n".encode(
            "ascii"
        )
    )
    for name in sorted(state_dict):
        if not isinstance(name, str):
            raise TypeError("state_dict keys must be strings.")
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"state_dict entry {name!r} must be a tensor, "
                f"got {type(tensor).__name__}."
            )
        _update_length_prefixed(digest, name.encode("utf-8"))
        _update_length_prefixed(digest, str(tensor.dtype).encode("ascii"))
        _update_length_prefixed(
            digest,
            json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"),
        )
        _update_length_prefixed(digest, _tensor_bytes(tensor))
    return digest.hexdigest()


def seed_worker(worker_id: int) -> None:
    """Top-level, spawn-picklable DataLoader worker initializer."""

    del worker_id  # Torch already mixes the worker ID into ``initial_seed``.
    worker_seed = torch.initial_seed() % _NUMPY_SEED_MODULUS
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_torch_generator(seed: int) -> torch.Generator:
    """Create a CPU generator without consuming global Torch RNG state."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(_validated_seed(seed))
    return generator


class DeterministicEpochSampler(Sampler[int]):
    """Epoch-addressable sampler whose order is independent of global RNG state."""

    def __init__(self, data_source: Sequence[Any], seed: int, shuffle: bool = True):
        self.data_source = data_source
        self.seed = _validated_seed(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        size = len(self.data_source)
        if not self.shuffle:
            return iter(range(size))
        epoch_seed = derive_seed(self.seed, "epoch_sampler", self.epoch)
        order = torch.randperm(size, generator=make_torch_generator(epoch_seed))
        return iter(order.tolist())

    def __len__(self) -> int:
        return len(self.data_source)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = _validated_seed(epoch, field="epoch")


def atomic_write_json(path: str | os.PathLike[str], payload: Any) -> Path:
    """Atomically replace a UTF-8 JSON manifest and fsync its file contents."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        _json_ready(payload),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def _compile_patterns(patterns: Sequence[str | Pattern[str]]) -> tuple[Pattern[str], ...]:
    return tuple(re.compile(pattern) if isinstance(pattern, str) else pattern for pattern in patterns)


def _is_allowed_name(
    name: str,
    prefixes: Sequence[str],
    patterns: Sequence[Pattern[str]],
) -> bool:
    return any(name.startswith(prefix) for prefix in prefixes) or any(
        pattern.search(name) is not None for pattern in patterns
    )


_ModuleT = TypeVar("_ModuleT", bound=nn.Module)


def build_paired_models(
    reference_factory: Callable[[], _ModuleT],
    variant_factory: Callable[[], _ModuleT],
    *,
    reference_seed: int,
    variant_seed: int,
    allowed_reference_only_prefixes: Sequence[str] = (),
    allowed_variant_only_prefixes: Sequence[str] = (),
    allowed_reference_only_regexes: Sequence[str | Pattern[str]] = (),
    allowed_variant_only_regexes: Sequence[str | Pattern[str]] = (),
) -> tuple[_ModuleT, _ModuleT, dict[str, Any]]:
    """Construct a strict paired model comparison and copy all shared state.

    Unmatched names are legal only when an explicit side-specific prefix or
    regular-expression allowlist accepts them.  A common name with different
    shape or dtype is always an error because it is not the same shared tensor.
    """

    reference_seed = _validated_seed(reference_seed, field="reference_seed")
    variant_seed = _validated_seed(variant_seed, field="variant_seed")
    with isolated_rng(reference_seed):
        reference = reference_factory()
    with isolated_rng(variant_seed):
        variant = variant_factory()
    if not isinstance(reference, nn.Module) or not isinstance(variant, nn.Module):
        raise TypeError("Paired model factories must return torch.nn.Module instances.")

    reference_state = reference.state_dict()
    variant_state = variant.state_dict()
    reference_names = set(reference_state)
    variant_names = set(variant_state)
    shared_names = sorted(reference_names & variant_names)
    reference_only = sorted(reference_names - variant_names)
    variant_only = sorted(variant_names - reference_names)

    shape_mismatches = [
        name
        for name in shared_names
        if tuple(reference_state[name].shape) != tuple(variant_state[name].shape)
    ]
    dtype_mismatches = [
        name
        for name in shared_names
        if reference_state[name].dtype != variant_state[name].dtype
    ]
    if shape_mismatches or dtype_mismatches:
        details = []
        if shape_mismatches:
            details.append(f"shape mismatch: {shape_mismatches}")
        if dtype_mismatches:
            details.append(f"dtype mismatch: {dtype_mismatches}")
        raise ValueError("Paired shared-state contract failed; " + "; ".join(details))

    reference_patterns = _compile_patterns(allowed_reference_only_regexes)
    variant_patterns = _compile_patterns(allowed_variant_only_regexes)
    unexpected_reference = [
        name
        for name in reference_only
        if not _is_allowed_name(
            name, allowed_reference_only_prefixes, reference_patterns
        )
    ]
    unexpected_variant = [
        name
        for name in variant_only
        if not _is_allowed_name(name, allowed_variant_only_prefixes, variant_patterns)
    ]
    if unexpected_reference or unexpected_variant:
        raise ValueError(
            "Paired state contains unexpected unmatched names; "
            f"reference_only={unexpected_reference}, variant_only={unexpected_variant}. "
            "Declare explicit side-specific prefix/regex allowlists for optional state."
        )

    with torch.no_grad():
        for name in shared_names:
            variant_state[name].copy_(reference_state[name])

    variant_after = variant.state_dict()
    unequal = [
        name
        for name in shared_names
        if not torch.equal(reference_state[name], variant_after[name])
    ]
    if unequal:
        raise RuntimeError(
            f"Paired shared-state copy was not bitwise exact for: {unequal}."
        )

    reference_shared = {name: reference_state[name] for name in shared_names}
    variant_shared = {name: variant_after[name] for name in shared_names}
    reference_shared_hash = state_dict_sha256(reference_shared)
    variant_shared_hash = state_dict_sha256(variant_shared)
    if reference_shared_hash != variant_shared_hash:
        raise RuntimeError("Paired shared-state hashes differ after exact tensor checks.")

    report = {
        "schema_version": 1,
        "reference_seed": reference_seed,
        "variant_seed": variant_seed,
        "copied_tensor_count": len(shared_names),
        "copied_names": shared_names,
        "reference_only_names": reference_only,
        "variant_only_names": variant_only,
        # The generic key is convenient for experiment manifests; the explicit
        # SHA-256 key makes the algorithm unambiguous for standalone consumers.
        "shared_state_hash": reference_shared_hash,
        "shared_state_sha256": reference_shared_hash,
        "reference_state_sha256": state_dict_sha256(reference_state),
        "variant_state_sha256": state_dict_sha256(variant_after),
    }
    return reference, variant, report


__all__ = [
    "SEED_DERIVATION_VERSION",
    "SUPPORTED_SEED_DERIVATION_VERSIONS",
    "SeedBundle",
    "derive_seed",
    "resolve_seed_bundle",
    "set_global_seed",
    "set_experiment_seed",
    "isolated_rng",
    "stable_json_hash",
    "state_dict_sha256",
    "seed_worker",
    "make_torch_generator",
    "DeterministicEpochSampler",
    "atomic_write_json",
    "build_paired_models",
]
