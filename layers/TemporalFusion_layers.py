import math
import random
from contextlib import contextmanager
from typing import Callable, Iterable, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


@contextmanager
def _preserve_extension_rng(device: torch.device):
    """Restore process RNG streams after evaluating a neutral extension.

    ``torch.random.fork_rng`` always snapshots the CPU generator and can also
    snapshot the accelerator generator used by the branch.  Python and NumPy
    are included because an extension callback is ordinary user code and is
    not limited to Torch operators.
    """

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    device_type = device.type
    accelerator_devices = []
    fork_device_type = 'cuda'
    if device_type in {'cuda', 'xpu'}:
        fork_device_type = device_type
        device_module = getattr(torch, device_type)
        device_index = device.index
        if device_index is None:
            device_index = device_module.current_device()
        accelerator_devices = [device_index]

    # MPS does not expose the indexed-generator interface consumed by
    # torch.random.fork_rng, so preserve its single generator explicitly.
    mps_state = None
    if device_type == 'mps' and hasattr(torch, 'mps') and hasattr(torch.mps, 'get_rng_state'):
        mps_state = torch.mps.get_rng_state()

    try:
        with torch.random.fork_rng(
            devices=accelerator_devices,
            enabled=True,
            device_type=fork_device_type,
        ):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


class ExtensionResidualAdapter(nn.Module):
    """Combine an existing path with an optional extension as a residual.

    The adapter deliberately performs no normalization or activation after the
    combination.  In ``neutral`` mode its learnable strength starts at exactly
    zero, so the initial value and the shared-path gradient are exactly those
    of ``base_output``.  The extension is still evaluated, allowing the
    strength to learn on the first optimization step; its own parameters begin
    receiving gradients after the strength moves away from zero.

    ``delta_output`` may be a tensor or a zero-argument callable.  Prefer the
    callable form for stochastic branches.  While the effective strength is
    exactly zero, the callback runs with process RNG state restored afterward,
    preventing extension dropout/noise from perturbing downstream shared RNG.

    Args:
        channels: Last-dimension size required by ``strength_type='channel'``.
        strength_type: A single shared ``'scalar'`` strength or one ``'channel'``
            strength per last-dimension channel.
        mode: Initialization policy: ``'neutral'`` (0), ``'small_residual'``
            (``small_residual_strength``), or ``'legacy'``
            (``legacy_strength``).  The ``off`` mode is represented by not
            constructing an adapter.
        small_residual_strength: Explicit exploratory nonzero initialization.
        legacy_strength: Compatibility initialization for the old full branch.
        extension_name: Optional stable selector for
            :func:`temporarily_zero_extensions`.
    """

    VALID_STRENGTH_TYPES = frozenset({'scalar', 'channel'})
    VALID_MODES = frozenset({'neutral', 'small_residual', 'legacy'})

    def __init__(
        self,
        channels: Optional[int] = None,
        *,
        strength_type: str = 'scalar',
        mode: str = 'neutral',
        small_residual_strength: float = 1e-3,
        legacy_strength: float = 1.0,
        extension_name: Optional[str] = None,
    ):
        super().__init__()
        if strength_type not in self.VALID_STRENGTH_TYPES:
            raise ValueError(
                "strength_type must be one of: "
                f"{', '.join(sorted(self.VALID_STRENGTH_TYPES))}."
            )
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"mode must be one of: {', '.join(sorted(self.VALID_MODES))}."
            )
        if strength_type == 'channel':
            if not isinstance(channels, int) or isinstance(channels, bool) or channels <= 0:
                raise ValueError(
                    "channels must be a positive integer when strength_type='channel'."
                )
            parameter_shape = (channels,)
        else:
            if channels is not None and (
                not isinstance(channels, int) or isinstance(channels, bool) or channels <= 0
            ):
                raise ValueError("channels must be a positive integer or None.")
            parameter_shape = ()
        if extension_name is not None and (
            not isinstance(extension_name, str) or not extension_name.strip()
        ):
            raise ValueError("extension_name must be a non-empty string or None.")

        if mode == 'neutral':
            initial_strength = 0.0
        elif mode == 'small_residual':
            initial_strength = float(small_residual_strength)
            if not math.isfinite(initial_strength) or initial_strength == 0.0:
                raise ValueError(
                    "small_residual_strength must be finite and nonzero in "
                    "small_residual mode."
                )
        else:
            initial_strength = float(legacy_strength)
            if not math.isfinite(initial_strength):
                raise ValueError("legacy_strength must be finite in legacy mode.")

        self.channels = channels
        self.strength_type = strength_type
        self.mode = mode
        self.extension_name = extension_name.strip() if extension_name is not None else None
        self.residual_strength = nn.Parameter(
            torch.full(parameter_shape, initial_strength, dtype=torch.float32)
        )

        # Plain Python attributes are intentional: neither a temporary
        # counterfactual override nor cached diagnostics belong in state_dict.
        self._temporary_zero_override = False
        self._last_diagnostics = None

    @property
    def strength(self):
        """Alias for the learnable ``residual_strength`` parameter."""

        return self.residual_strength

    @property
    def is_temporarily_zeroed(self) -> bool:
        return bool(self._temporary_zero_override)

    def _effective_strength(self, base_output: torch.Tensor) -> torch.Tensor:
        if self.residual_strength.device != base_output.device:
            raise ValueError(
                "Residual strength and base_output must be on the same device; "
                f"got {self.residual_strength.device} and {base_output.device}. "
                "Move the adapter with model.to(device)."
            )
        strength = self.residual_strength
        if self._temporary_zero_override:
            # Keep a graph edge with an exactly-zero derivative so a context
            # override cannot accidentally mutate or replace the parameter.
            strength = strength * 0.0
        strength = strength.to(dtype=base_output.dtype)
        if self.strength_type == 'channel':
            strength = strength.view(*([1] * (base_output.ndim - 1)), self.channels)
        return strength

    def is_effectively_zero(self) -> bool:
        """Return whether the current (possibly overridden) strength is zero."""

        if self._temporary_zero_override:
            return True
        return bool(torch.count_nonzero(self.residual_strength.detach()).item() == 0)

    def _validate_outputs(
        self,
        base_output: torch.Tensor,
        delta_output: torch.Tensor,
    ) -> None:
        if not isinstance(base_output, torch.Tensor):
            raise TypeError("base_output must be a torch.Tensor.")
        if not isinstance(delta_output, torch.Tensor):
            raise TypeError("delta_output (or its callable result) must be a torch.Tensor.")
        if base_output.layout != torch.strided or delta_output.layout != torch.strided:
            raise ValueError("base_output and delta_output must use strided tensor layout.")
        if base_output.ndim == 0:
            raise ValueError("base_output and delta_output must have at least one dimension.")
        if base_output.numel() == 0:
            raise ValueError("base_output and delta_output must be non-empty.")
        if tuple(base_output.shape) != tuple(delta_output.shape):
            raise ValueError(
                "base_output and delta_output must have identical shapes; "
                f"got {tuple(base_output.shape)} and {tuple(delta_output.shape)}."
            )
        if base_output.device != delta_output.device:
            raise ValueError(
                "base_output and delta_output must be on the same device; "
                f"got {base_output.device} and {delta_output.device}."
            )
        if base_output.dtype != delta_output.dtype:
            raise ValueError(
                "base_output and delta_output must have the same dtype; "
                f"got {base_output.dtype} and {delta_output.dtype}."
            )
        if not base_output.is_floating_point():
            raise ValueError("base_output and delta_output must use a floating-point dtype.")
        if self.strength_type == 'channel' and base_output.shape[-1] != self.channels:
            raise ValueError(
                "Channel-wise residual strength must match the output's last dimension; "
                f"configured {self.channels}, received {base_output.shape[-1]}."
            )

    @staticmethod
    def _rms(value: torch.Tensor) -> torch.Tensor:
        # Float32 accumulation keeps half/bfloat16 diagnostics stable without
        # modifying the combination dtype.
        diagnostic_value = value.detach()
        if diagnostic_value.dtype in {torch.float16, torch.bfloat16}:
            diagnostic_value = diagnostic_value.float()
        return torch.sqrt(torch.mean(torch.square(diagnostic_value)))

    def _record_diagnostics(
        self,
        base_output: torch.Tensor,
        delta_output: torch.Tensor,
        combined: torch.Tensor,
        effective_strength: torch.Tensor,
    ) -> None:
        raw_strength = self.residual_strength.detach().reshape(-1).clone()
        effective_strength = effective_strength.detach().reshape(-1).clone()
        self._last_diagnostics = {
            'raw_residual_strength': raw_strength,
            'effective_residual_strength': effective_strength,
            # Compatibility/convenience alias: the residual used for this
            # forward, including a temporary counterfactual override.
            'residual_strength': effective_strength.clone(),
            'base_rms': self._rms(base_output).clone(),
            'delta_rms': self._rms(delta_output).clone(),
            'combined_minus_base_rms': self._rms(combined - base_output).clone(),
        }

    def diagnostics(self):
        """Return detached copies of diagnostics from the latest forward pass."""

        if self._last_diagnostics is None:
            return None
        return {key: value.clone() for key, value in self._last_diagnostics.items()}

    def forward(
        self,
        base_output: torch.Tensor,
        delta_output: torch.Tensor | Callable[[], torch.Tensor],
        *,
        return_diagnostics: bool = False,
    ):
        if not isinstance(base_output, torch.Tensor):
            raise TypeError("base_output must be a torch.Tensor.")

        exact_zero = self.is_effectively_zero()
        if callable(delta_output):
            if exact_zero:
                with _preserve_extension_rng(base_output.device):
                    resolved_delta = delta_output()
            else:
                resolved_delta = delta_output()
        else:
            resolved_delta = delta_output

        self._validate_outputs(base_output, resolved_delta)
        effective_strength = self._effective_strength(base_output)
        delta_is_finite = bool(torch.isfinite(resolved_delta.detach()).all().item())
        if not delta_is_finite:
            if not exact_zero:
                raise ValueError(
                    "A non-finite extension delta cannot be combined with a "
                    "nonzero residual strength."
                )
            # IEEE arithmetic makes ``0 * NaN`` a NaN.  A neutral or
            # counterfactually zeroed extension must instead fail closed to
            # its exact base path.  Quarantining the complete broken delta
            # also gives the strength a zero gradient, so an invalid branch
            # cannot activate itself on the next optimizer step.  The normal
            # finite-delta path below remains differentiable with respect to
            # strength at exact zero.
            delta_for_combination = torch.zeros_like(resolved_delta)
        else:
            delta_for_combination = resolved_delta
        combined = base_output + effective_strength * delta_for_combination
        self._record_diagnostics(
            base_output,
            resolved_delta,
            combined,
            effective_strength,
        )
        if return_diagnostics:
            return combined, self.diagnostics()
        return combined

    def combine_lazy(
        self,
        base_output: torch.Tensor,
        delta_fn: Callable[[], torch.Tensor],
        *,
        return_diagnostics: bool = False,
    ):
        """Named alias for the RNG-neutral callable form of :meth:`forward`."""

        if not callable(delta_fn):
            raise TypeError("delta_fn must be callable.")
        return self.forward(
            base_output,
            delta_fn,
            return_diagnostics=return_diagnostics,
        )


@contextmanager
def temporarily_zero_extensions(model: nn.Module, names: Optional[Iterable[str] | str] = None):
    """Temporarily force selected residual adapters to exact zero.

    ``names`` may contain qualified names from ``model.named_modules()`` or
    semantic ``extension_name`` values declared by adapters.  A qualified name
    selects exactly one module; a semantic name selects every matching module
    (for example the history and future VSN adapters belonging to one graph
    extension).  ``None`` selects every adapter.  The override is not a
    parameter or buffer, so checkpoints written inside the context retain the
    trained strengths.  Previous overrides are restored in ``finally``, making
    nested and exceptional use safe.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module.")
    adapters = [
        (qualified_name, module)
        for qualified_name, module in model.named_modules()
        if isinstance(module, ExtensionResidualAdapter)
    ]

    if names is None:
        selected = adapters
    else:
        requested = [names] if isinstance(names, str) else list(names)
        if any(not isinstance(name, str) or not name for name in requested):
            raise ValueError("names must contain non-empty strings only.")
        selected = []
        missing = []
        for requested_name in requested:
            # Qualified module names are authoritative and intentionally
            # select one adapter.  Otherwise a semantic extension selector
            # applies to every adapter that implements that extension.
            matches = [item for item in adapters if item[0] == requested_name]
            if not matches:
                matches = [
                    item for item in adapters
                    if item[1].extension_name == requested_name
                ]
            if not matches:
                missing.append(requested_name)
                continue
            for match in matches:
                if all(match[1] is not module for _, module in selected):
                    selected.append(match)
        if missing:
            available = sorted(
                name or '<root>' for name, _ in adapters
            )
            raise ValueError(
                f"Unknown extension adapter(s): {missing}. Available qualified names: {available}."
            )

    previous = [(module, module._temporary_zero_override) for _, module in selected]
    try:
        for module, _ in previous:
            module._temporary_zero_override = True
        yield model
    finally:
        for module, prior_override in reversed(previous):
            module._temporary_zero_override = prior_override


def _debug_check_finite(enabled: bool, tensor: torch.Tensor, message: str):
    if enabled and not torch.isfinite(tensor).all():
        raise ValueError(message)


def build_causal_mask(seq_len: int, device, dtype):
    return torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype), 1)


def _resolve_positions(
    length: int,
    positions,
    device,
    *,
    batch_size: Optional[int] = None,
    name: str = 'positions',
):
    """Resolve shared ``[T]`` or sample-specific ``[B,T]`` coordinates.

    Keeping omitted/shared coordinates rank-1 preserves the historical RoPE and
    ALiBi arithmetic.  A rank-2 input is never silently collapsed: its batch
    dimension must match the associated query/key tensor.
    """

    if positions is None:
        return torch.arange(length, device=device, dtype=torch.float32)
    try:
        resolved = positions if torch.is_tensor(positions) else torch.as_tensor(positions)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(f"{name} must contain real numeric coordinates.") from exc
    if resolved.dtype == torch.bool or resolved.is_complex() or not (
        resolved.is_floating_point()
        or resolved.dtype in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
    ):
        raise TypeError(f"{name} must contain real numeric coordinates.")
    if resolved.ndim == 1:
        if resolved.shape[0] != length:
            raise ValueError(
                f"{name} must have time length {length}, got shape {tuple(resolved.shape)}."
            )
    elif resolved.ndim == 2:
        if resolved.shape[1] != length:
            raise ValueError(
                f"{name} must have time length {length}, got shape {tuple(resolved.shape)}."
            )
        if batch_size is not None and resolved.shape[0] != batch_size:
            raise ValueError(
                f"{name} batch size must be {batch_size}, got shape {tuple(resolved.shape)}."
            )
    else:
        raise ValueError(
            f"{name} must have shape [T] or [B,T], got {tuple(resolved.shape)}."
        )
    resolved = resolved.detach().to(device=device, dtype=torch.float32)
    if not torch.isfinite(resolved).all():
        raise ValueError(f"{name} must contain finite coordinates.")
    return resolved


def _resolve_true_valid_mask(mask, batch_size: int, length: int, device, *, name: str):
    """Canonicalise a True-valid mask to ``[B,T]`` without polarity flips."""

    if mask is None:
        return None
    try:
        resolved = mask if torch.is_tensor(mask) else torch.as_tensor(mask)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(f"{name} must contain boolean values (True means valid).") from exc
    if resolved.dtype != torch.bool:
        raise TypeError(f"{name} must have boolean dtype (True means valid).")
    if resolved.ndim == 1:
        if resolved.shape[0] != length:
            raise ValueError(
                f"{name} must have time length {length}, got shape {tuple(resolved.shape)}."
            )
        resolved = resolved.unsqueeze(0).expand(batch_size, -1)
    elif resolved.ndim == 2:
        if tuple(resolved.shape) != (batch_size, length):
            raise ValueError(
                f"{name} must have shape [B,T]={(batch_size, length)}, "
                f"got {tuple(resolved.shape)}."
            )
    else:
        raise ValueError(
            f"{name} must have shape [T] or [B,T], got {tuple(resolved.shape)}."
        )
    return resolved.detach().to(device=device)


def _zero_invalid_queries(value: torch.Tensor, query_valid_mask):
    if query_valid_mask is None or bool(query_valid_mask.all().item()):
        return value
    if value.ndim == 4:
        return value.masked_fill(~query_valid_mask[:, None, :, None], 0.0)
    if value.ndim == 3:
        return value.masked_fill(~query_valid_mask[:, :, None], 0.0)
    raise ValueError(
        "Query-masked attention values must be rank-3 outputs or rank-4 weights."
    )


def _shift_source_positions(positions: torch.Tensor, lag: int) -> torch.Tensor:
    """Map destination token ``t`` to its physical source token ``t-lag``.

    Prefix coordinates are invalid in lag attention.  They retain the legacy
    ``position-lag`` extrapolation for stable diagnostics, while every usable
    coordinate is gathered from its actual (possibly irregular) source token.
    """

    shifted = positions - float(lag)
    shifted[..., lag:] = positions[..., :-lag]
    return shifted


def _rotate_half(x):
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1)
    return rotated.flatten(start_dim=-2)


def apply_rotary_embedding(query, key, query_positions=None, key_positions=None, base=10000.0):
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("RoPE expects query/key shaped [B,H,T,D].")
    if query.shape[0] != key.shape[0]:
        raise ValueError("RoPE query/key batch sizes must match.")
    if query.shape[-1] % 2 != 0:
        raise ValueError("RoPE requires an even head dimension.")
    if key.shape[-1] != query.shape[-1]:
        raise ValueError("RoPE query/key head dimensions must match.")

    device = query.device
    query_positions = _resolve_positions(
        query.shape[-2],
        query_positions,
        device,
        batch_size=query.shape[0],
        name='query_positions',
    )
    key_positions = _resolve_positions(
        key.shape[-2],
        key_positions,
        device,
        batch_size=key.shape[0],
        name='key_positions',
    )
    half_dim = query.shape[-1] // 2
    index = torch.arange(half_dim, device=device, dtype=torch.float32)
    inv_freq = base ** (-index / max(1, half_dim))

    def _apply(x, positions):
        if positions.ndim == 1:
            angles = torch.einsum('t,d->td', positions, inv_freq)
            broadcast_dims = (0, 0)
        else:
            angles = torch.einsum('bt,d->btd', positions, inv_freq)
            broadcast_dims = (1,)
        sin = torch.repeat_interleave(torch.sin(angles), 2, dim=-1).to(dtype=x.dtype)
        cos = torch.repeat_interleave(torch.cos(angles), 2, dim=-1).to(dtype=x.dtype)
        for dim in broadcast_dims:
            sin = sin.unsqueeze(dim)
            cos = cos.unsqueeze(dim)
        return x * cos + _rotate_half(x) * sin

    return _apply(query, query_positions), _apply(key, key_positions)


def _get_alibi_slopes(num_heads: int, device, dtype):
    def _power_of_two_slopes(power_of_two: int):
        start = 2 ** (-(2 ** -(math.log2(power_of_two) - 3)))
        ratio = start
        return [start * (ratio ** idx) for idx in range(power_of_two)]

    if num_heads <= 0:
        raise ValueError("num_heads must be positive for ALiBi.")
    if math.log2(num_heads).is_integer():
        slopes = _power_of_two_slopes(num_heads)
    else:
        closest_power_of_two = 2 ** math.floor(math.log2(num_heads))
        slopes = _power_of_two_slopes(closest_power_of_two)
        extra = _power_of_two_slopes(2 * closest_power_of_two)
        slopes.extend(extra[0::2][: num_heads - closest_power_of_two])
    return torch.tensor(slopes, device=device, dtype=dtype)


def build_alibi_bias(num_heads, query_len, key_len, device, dtype, query_positions=None, key_positions=None, scale=1.0):
    query_positions = _resolve_positions(
        query_len, query_positions, device, name='query_positions'
    )
    key_positions = _resolve_positions(
        key_len, key_positions, device, name='key_positions'
    )
    query_positions = (
        query_positions.unsqueeze(0)
        if query_positions.ndim == 1
        else query_positions
    )
    key_positions = (
        key_positions.unsqueeze(0)
        if key_positions.ndim == 1
        else key_positions
    )
    coordinate_batch = max(query_positions.shape[0], key_positions.shape[0])
    if query_positions.shape[0] not in (1, coordinate_batch):
        raise ValueError("query_positions and key_positions batch sizes are incompatible.")
    if key_positions.shape[0] not in (1, coordinate_batch):
        raise ValueError("query_positions and key_positions batch sizes are incompatible.")
    query_positions = query_positions.expand(coordinate_batch, -1)
    key_positions = key_positions.expand(coordinate_batch, -1)
    relative_distance = (
        query_positions.unsqueeze(-1) - key_positions.unsqueeze(-2)
    ).abs()
    slopes = _get_alibi_slopes(num_heads, device, dtype)
    return -(
        scale
        * slopes.view(1, num_heads, 1, 1)
        * relative_distance.unsqueeze(1).to(dtype=dtype)
    )


class PositionalMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, position_bias_type='none', rope_base=10000.0, alibi_scale=1.0, attention_backend='exact', debug_checks: bool = False, attn_dropout: float = 0.0):
        super(PositionalMultiHeadAttention, self).__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads for PositionalMultiHeadAttention.")
        if position_bias_type not in {'none', 'rope', 'alibi'}:
            raise ValueError("position_bias_type must be one of: none, rope, alibi.")
        if attention_backend not in {'exact', 'sdpa'}:
            raise ValueError("attention_backend must be one of: exact, sdpa.")

        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.position_bias_type = position_bias_type
        self.rope_base = float(rope_base)
        self.alibi_scale = float(alibi_scale)
        self.attention_backend = attention_backend
        self.debug_checks = debug_checks
        self.attn_dropout = float(attn_dropout)
        self.last_attention_backend = None
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.out_projection = nn.Linear(d_model, d_model, bias=False)
        self.out_dropout = nn.Dropout(dropout)
        self.scale = self.d_head ** -0.5

    def _build_attention_bias(
        self,
        query,
        query_len,
        key_len,
        attn_mask=None,
        query_positions=None,
        key_positions=None,
        key_padding_mask=None,
        key_valid_mask=None,
    ):
        attention_bias = None
        if self.position_bias_type == 'alibi':
            attention_bias = build_alibi_bias(
                self.n_heads,
                query_len,
                key_len,
                query.device,
                query.dtype,
                query_positions=query_positions,
                key_positions=key_positions,
                scale=self.alibi_scale,
            )
            if attention_bias.shape[0] not in (1, query.shape[0]):
                raise ValueError(
                    "Batched ALiBi coordinates must match the attention batch size."
                )
        if attn_mask is not None:
            if attn_mask.ndim != 2:
                raise ValueError(f"attn_mask must be rank-2 [T,S], got shape {tuple(attn_mask.shape)}.")
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            attention_bias = attn_mask if attention_bias is None else attention_bias + attn_mask
        if key_padding_mask is not None:
            if key_padding_mask.ndim != 2 or key_padding_mask.shape != (query.shape[0], key_len):
                raise ValueError(
                    f"key_padding_mask must be rank-2 [B,S] matching batch/key length, got shape {tuple(key_padding_mask.shape)}."
                )
            if key_padding_mask.dtype != torch.bool:
                raise TypeError("key_padding_mask must have boolean dtype (True means padding).")
        invalid_keys = None
        if key_padding_mask is not None:
            invalid_keys = key_padding_mask.to(device=query.device)
        if key_valid_mask is not None and not bool(key_valid_mask.all().item()):
            invalid_from_validity = ~key_valid_mask
            invalid_keys = (
                invalid_from_validity
                if invalid_keys is None
                else invalid_keys | invalid_from_validity
            )
        if invalid_keys is not None and bool(invalid_keys.any().item()):
            padding_bias = torch.zeros(
                query.shape[0], 1, 1, key_len, device=query.device, dtype=query.dtype
            ).masked_fill(invalid_keys.unsqueeze(1).unsqueeze(1), float('-inf'))
            attention_bias = padding_bias if attention_bias is None else attention_bias + padding_bias
        return attention_bias

    def forward(
        self,
        query,
        key,
        value,
        return_attention: bool = False,
        attn_mask=None,
        query_positions=None,
        key_positions=None,
        key_padding_mask=None,
        query_valid_mask=None,
        key_valid_mask=None,
    ):
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError(
                f"PositionalMultiHeadAttention expects rank-3 tensors, got {tuple(query.shape)}, {tuple(key.shape)}, {tuple(value.shape)}."
            )
        batch_size, query_len, _ = query.shape
        key_len = key.shape[1]
        if key.shape[0] != batch_size or value.shape[0] != batch_size:
            raise ValueError("Attention query/key/value batch sizes must match.")
        if value.shape[1] != key_len:
            raise ValueError("Attention key/value time lengths must match.")
        if query_positions is not None:
            query_positions = _resolve_positions(
                query_len,
                query_positions,
                query.device,
                batch_size=batch_size,
                name='query_positions',
            )
        if key_positions is not None:
            key_positions = _resolve_positions(
                key_len,
                key_positions,
                query.device,
                batch_size=batch_size,
                name='key_positions',
            )
        resolved_query_valid = _resolve_true_valid_mask(
            query_valid_mask,
            batch_size,
            query_len,
            query.device,
            name='query_valid_mask',
        )
        resolved_key_valid = _resolve_true_valid_mask(
            key_valid_mask,
            batch_size,
            key_len,
            query.device,
            name='key_valid_mask',
        )
        q = self.q_linear(query).view(batch_size, query_len, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        k = self.k_linear(key).view(batch_size, key_len, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = self.v_linear(value).view(batch_size, key_len, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        if self.position_bias_type == 'rope':
            q, k = apply_rotary_embedding(q, k, query_positions=query_positions, key_positions=key_positions, base=self.rope_base)

        _debug_check_finite(self.debug_checks, q, "Attention projections contain NaN/Inf values.")
        _debug_check_finite(self.debug_checks, k, "Attention projections contain NaN/Inf values.")
        _debug_check_finite(self.debug_checks, v, "Attention projections contain NaN/Inf values.")

        attention_bias = self._build_attention_bias(
            query,
            query_len,
            key_len,
            attn_mask=attn_mask,
            query_positions=query_positions,
            key_positions=key_positions,
            key_padding_mask=key_padding_mask,
            key_valid_mask=resolved_key_valid,
        )

        if self.attention_backend == 'sdpa' and not return_attention:
            attention_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_bias,
                dropout_p=(self.attn_dropout if self.training else 0.0),
                is_causal=False,
            )
            self.last_attention_backend = 'sdpa'
            attention_out = attention_out.permute(0, 2, 1, 3).contiguous().view(batch_size, query_len, self.n_heads * self.d_head)
            out = self.out_projection(attention_out)
            out = self.out_dropout(out)
            out = _zero_invalid_queries(out, resolved_query_valid)
            return out

        attention_score = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        _debug_check_finite(self.debug_checks, attention_score, "Attention scores contain NaN/Inf values before biasing.")
        if attention_bias is not None:
            attention_score = attention_score + attention_bias.to(dtype=attention_score.dtype)
        if self.debug_checks and torch.isnan(attention_score).any():
            raise ValueError("Attention scores contain NaN values after biasing.")
        clamp_limit = 1e4
        finite_mask = torch.isfinite(attention_score)
        if finite_mask.any() and (attention_score.masked_select(finite_mask).abs() > clamp_limit).any():
            clamped = attention_score.clamp(min=-clamp_limit, max=clamp_limit)
            attention_score = torch.where(finite_mask, clamped, attention_score)
        fully_masked = ~torch.isfinite(attention_score).any(dim=-1, keepdim=True)
        safe_scores = torch.where(fully_masked, torch.zeros_like(attention_score), attention_score)
        attention_prob = F.softmax(safe_scores, dim=-1)
        attention_prob = torch.where(fully_masked, torch.zeros_like(attention_prob), attention_prob)
        attention_prob = _zero_invalid_queries(attention_prob, resolved_query_valid)
        _debug_check_finite(self.debug_checks, attention_prob, "Attention probabilities contain NaN/Inf values.")
        attention_prob_used = F.dropout(attention_prob, p=self.attn_dropout, training=self.training)

        self.last_attention_backend = 'exact'
        attention_out = torch.matmul(attention_prob_used, v)
        attention_out = attention_out.permute(0, 2, 1, 3).contiguous().view(batch_size, query_len, self.n_heads * self.d_head)
        out = self.out_projection(attention_out)
        out = self.out_dropout(out)
        out = _zero_invalid_queries(out, resolved_query_valid)
        if return_attention:
            return out, attention_prob
        return out


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True):
        super(CausalConv1d, self).__init__()
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive for CausalConv1d.")
        if dilation <= 0:
            raise ValueError("dilation must be positive for CausalConv1d.")
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f"CausalConv1d expects rank-3 [B,C,T] input, got shape {tuple(x.shape)}.")
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class GatedDilatedTemporalBlock(nn.Module):
    def __init__(self, d_model, hidden_size=None, kernel_size=3, dilation=1, dropout=0.0, debug_checks: bool = False):
        super(GatedDilatedTemporalBlock, self).__init__()
        hidden_size = d_model if not hidden_size or hidden_size <= 0 else hidden_size
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive for GatedDilatedTemporalBlock.")
        self.filter_conv = CausalConv1d(d_model, hidden_size, kernel_size=kernel_size, dilation=dilation)
        self.gate_conv = CausalConv1d(d_model, hidden_size, kernel_size=kernel_size, dilation=dilation)
        self.out_projection = nn.Conv1d(hidden_size, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.debug_checks = debug_checks

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f"GatedDilatedTemporalBlock expects rank-3 [B,T,D] input, got shape {tuple(x.shape)}.")
        _debug_check_finite(self.debug_checks, x, "Temporal block input contains NaN/Inf values.")
        x_conv = x.transpose(1, 2)
        filtered = torch.tanh(self.filter_conv(x_conv))
        gated = torch.sigmoid(self.gate_conv(x_conv))
        mixed = self.out_projection(self.dropout(filtered * gated)).transpose(1, 2)
        out = self.layer_norm(x + mixed)
        _debug_check_finite(self.debug_checks, out, "Temporal block output contains NaN/Inf values.")
        return out


class GatedDilatedTemporalBackbone(nn.Module):
    def __init__(self, d_model, num_layers=3, kernel_size=3, hidden_size=None, dropout=0.0, debug_checks: bool = False):
        super(GatedDilatedTemporalBackbone, self).__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive for GatedDilatedTemporalBackbone.")
        self.blocks = nn.ModuleList([
            GatedDilatedTemporalBlock(
                d_model,
                hidden_size=hidden_size,
                kernel_size=kernel_size,
                dilation=2 ** idx,
                dropout=dropout,
                debug_checks=debug_checks,
            )
            for idx in range(num_layers)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class HybridTemporalBackbone(nn.Module):
    def __init__(self, d_model, num_layers=3, kernel_size=3, hidden_size=None, dropout=0.0, debug_checks: bool = False):
        super(HybridTemporalBackbone, self).__init__()
        self.tcn_backbone = GatedDilatedTemporalBackbone(
            d_model,
            num_layers=num_layers,
            kernel_size=kernel_size,
            hidden_size=hidden_size,
            dropout=dropout,
            debug_checks=debug_checks,
        )
        self.recurrent_backbone = nn.LSTM(d_model, d_model, batch_first=True)
        self.fusion_gate = nn.Linear(d_model * 2, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, state=None):
        if x.ndim != 3:
            raise ValueError(f"HybridTemporalBackbone expects rank-3 [B,T,D] input, got shape {tuple(x.shape)}.")
        tcn_features = self.tcn_backbone(x)
        recurrent_features, next_state = self.recurrent_backbone(tcn_features, state)
        gate = torch.sigmoid(self.fusion_gate(torch.cat([tcn_features, recurrent_features], dim=-1)))
        fused = gate * recurrent_features + (1.0 - gate) * tcn_features
        return self.layer_norm(fused), next_state


class SpectralBranch(nn.Module):
    """Parallel frequency-domain processing branch for TFT.

    Semantics-v2 uses three truthful public operations:

    ``low_k``
        Hard-retain the first ``k`` rFFT bins.
    ``top_amplitude_k``
        Hard-retain the strongest ``k`` bins independently per sample/channel.
    ``learned_filter``
        Interpolate ``modes`` learned spectral control points across every
        runtime rFFT bin and apply a soft all-bin filter.

    The legacy names ``low``, ``top_amplitude``, and ``learned`` deliberately
    keep their historical state topology and arithmetic for semantics-v1
    replay.  Config-level semantic-version resolution decides which family is
    legal; this component accepts both so old checkpoints remain loadable.

    DC is eligible in every operation.  The Nyquist bin is eligible when the
    runtime length is even.  Because both are self-conjugate, ``irfft`` uses
    only their real response; diagnostics state that policy explicitly.
    """

    LEGACY_MODES = frozenset({'low', 'top_amplitude', 'learned'})
    CANONICAL_MODES = frozenset(
        {'low_k', 'top_amplitude_k', 'learned_filter'}
    )
    LEARNED_MODES = frozenset({'learned', 'learned_filter'})
    LOW_MODES = frozenset({'low', 'low_k'})
    TOP_AMPLITUDE_MODES = frozenset(
        {'top_amplitude', 'top_amplitude_k'}
    )
    VALID_SCOPES = frozenset(
        {'unspecified', 'history', 'known_future', 'combined'}
    )

    def __init__(self, d_model: int, modes: int = 32, mode_select: str = 'low', dropout: float = 0.0):
        super(SpectralBranch, self).__init__()
        if modes < 1:
            raise ValueError("tft_fft_modes must be >= 1.")
        valid_modes = self.LEGACY_MODES | self.CANONICAL_MODES
        if mode_select not in valid_modes:
            raise ValueError(
                "tft_fft_mode_select must be one of: "
                f"{', '.join(sorted(valid_modes))}."
            )
        self.d_model = d_model
        self.modes = modes
        self.mode_select = mode_select
        # Learnable complex weights: [d_model, modes] real + imaginary
        self.weight_real = nn.Parameter(torch.empty(d_model, modes))
        self.weight_imag = nn.Parameter(torch.empty(d_model, modes))
        nn.init.xavier_uniform_(self.weight_real)
        nn.init.xavier_uniform_(self.weight_imag)
        # Learned soft mask over all frequency bins.  The canonical
        # ``learned_filter`` name intentionally registers exactly the same
        # tensors as legacy ``learned`` so topology-compatible state can be
        # inspected or migrated explicitly rather than silently rewritten.
        if mode_select in self.LEARNED_MODES:
            self.freq_mask_logits = nn.Parameter(torch.zeros(1, d_model, modes))
        self.out_projection = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _interpolate_learned_spectral_params(self, n_freqs: int):
        if self.mode_select not in self.LEARNED_MODES:
            raise RuntimeError(
                "_interpolate_learned_spectral_params is only valid for "
                "learned/learned_filter modes."
            )
        if n_freqs < 1:
            raise ValueError("n_freqs must be positive.")
        if self.freq_mask_logits.shape[-1] == n_freqs:
            mask_logits = self.freq_mask_logits
            weight_real = self.weight_real.unsqueeze(0)
            weight_imag = self.weight_imag.unsqueeze(0)
        else:
            mask_logits = F.interpolate(
                self.freq_mask_logits,
                size=n_freqs,
                mode='linear',
                align_corners=False,
            )
            weight_real = F.interpolate(
                self.weight_real.unsqueeze(0),
                size=n_freqs,
                mode='linear',
                align_corners=False,
            )
            weight_imag = F.interpolate(
                self.weight_imag.unsqueeze(0),
                size=n_freqs,
                mode='linear',
                align_corners=False,
            )
        return mask_logits, weight_real.squeeze(0), weight_imag.squeeze(0)

    def summarize_learned_mask(self, n_freqs: int):
        if self.mode_select not in self.LEARNED_MODES:
            return None
        mask_logits, _, _ = self._interpolate_learned_spectral_params(n_freqs)
        soft_mask = torch.sigmoid(mask_logits)
        mean = float(soft_mask.mean().item())
        std = float(soft_mask.std(unbiased=False).item())
        peak_bins = soft_mask.argmax(dim=-1).to(dtype=torch.float32)
        peak_bin_mean = float(peak_bins.mean().item())
        return {
            'fft_learned_mask_mean': mean,
            'fft_learned_mask_std': std,
            'fft_learned_mask_peak_bin_mean': peak_bin_mean,
        }

    def _select_modes(self, x_ft, n_freqs: int):
        """Return indices of frequency modes to process (hard selection), or None for learned soft mask."""
        if self.mode_select in self.LEARNED_MODES:
            return None  # soft mask applied in forward instead of hard selection
        k = min(self.modes, n_freqs)
        if self.mode_select in self.LOW_MODES:
            return torch.arange(k, device=x_ft.device)
        if self.mode_select in self.TOP_AMPLITUDE_MODES:
            # top_amplitude: pick modes with highest energy per sample/channel.
            # NOTE: topk selection is non-differentiable — the model learns what to
            # do with selected modes but cannot learn *which* modes to select.
            amplitudes = x_ft.abs()  # [B, D, n_freqs]
            _, indices = torch.topk(amplitudes, k, dim=-1)
            return indices
        raise RuntimeError(f"Unhandled FFT mode {self.mode_select!r}.")

    @staticmethod
    def _frequency_grid(n_fft: int, n_freqs: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = torch.arange(
            n_freqs, device=device, dtype=torch.float32
        ) / float(n_fft)
        periods = torch.empty_like(normalized)
        periods[0] = float('inf')
        if n_freqs > 1:
            periods[1:] = normalized[1:].reciprocal()
        return normalized, periods

    def _build_diagnostics(
        self,
        *,
        x_ft: torch.Tensor,
        n_fft: int,
        mode_indices,
        learned_mask,
        learned_weight_real,
        learned_weight_imag,
        scope: str,
    ) -> dict:
        n_freqs = int(x_ft.shape[-1])
        effective_k = min(self.modes, n_freqs)
        normalized, periods = self._frequency_grid(
            n_fft, n_freqs, x_ft.device
        )
        has_nyquist = n_fft % 2 == 0
        diagnostics = {
            'mode': self.mode_select,
            'scope': scope,
            'sequence_length': int(n_fft),
            'requested_bin_count': (
                None if self.mode_select in self.LEARNED_MODES else int(self.modes)
            ),
            'spectral_control_point_count': (
                int(self.modes) if self.mode_select in self.LEARNED_MODES else None
            ),
            'available_bin_count': n_freqs,
            'selection_was_clamped': (
                False
                if self.mode_select in self.LEARNED_MODES
                else bool(self.modes > n_freqs)
            ),
            'normalized_frequency_grid': normalized.detach().clone(),
            'equivalent_token_period_grid': periods.detach().clone(),
            'period_unit': 'tokens',
            'physical_period_claim': False,
            'dc_bin': 0,
            'dc_policy': 'eligible_self_conjugate_real_response',
            'nyquist_bin': n_freqs - 1 if has_nyquist else None,
            'nyquist_policy': (
                'eligible_self_conjugate_real_response'
                if has_nyquist else 'absent_for_odd_sequence_length'
            ),
            'imaginary_endpoint_policy': (
                'irfft_discards_imaginary_dc_and_nyquist'
            ),
        }

        if self.mode_select in self.LEARNED_MODES:
            mask = learned_mask.detach().squeeze(0)
            weight = torch.complex(
                learned_weight_real.detach(), learned_weight_imag.detach()
            )
            response = weight * mask
            epsilon = torch.finfo(mask.dtype).eps
            probabilities = mask.clamp(min=epsilon, max=1.0 - epsilon)
            entropy = -(
                probabilities * probabilities.log()
                + (1.0 - probabilities) * (1.0 - probabilities).log()
            ).mean(dim=-1)
            effective_count = mask.sum(dim=-1).square() / (
                mask.square().sum(dim=-1) + epsilon
            )
            diagnostics.update({
                'operation': 'soft_all_bin_filter',
                'active_bin_count': n_freqs,
                'selected_bins': None,
                'peak_bins': mask.argmax(dim=-1).detach().clone(),
                'mask_entropy_per_channel': entropy.detach().clone(),
                'effective_active_bin_count_per_channel': effective_count.detach().clone(),
                'filter_norm_per_channel': response.abs().square().sum(dim=-1).sqrt().detach().clone(),
            })
            return diagnostics

        selected = mode_indices.detach().clone()
        hard_weight = torch.complex(
            self.weight_real[:, :effective_k].detach(),
            self.weight_imag[:, :effective_k].detach(),
        )
        if selected.ndim == 1:
            selected_amplitude = x_ft.detach().abs().index_select(-1, selected)
            peak_bins = selected[
                selected_amplitude.argmax(dim=-1)
            ]
            operation = 'hard_low_frequency_selection'
        else:
            # ``torch.topk`` is sorted by descending amplitude, so the first
            # selected bin is the actual peak for each sample/channel.
            peak_bins = selected[..., 0]
            operation = 'hard_sample_channel_amplitude_selection'
        diagnostics.update({
            'operation': operation,
            'active_bin_count': effective_k,
            'selected_bins': selected,
            'peak_bins': peak_bins.detach().clone(),
            'mask_entropy_per_channel': None,
            'effective_active_bin_count_per_channel': None,
            'filter_norm_per_channel': hard_weight.abs().square().sum(dim=-1).sqrt().detach().clone(),
        })
        return diagnostics

    def forward(self, x, return_diagnostics: bool = False, scope: str = 'unspecified'):
        """x: [B, L, d_model] -> [B, L, d_model]"""
        if x.ndim != 3:
            raise ValueError(
                f"SpectralBranch expects rank-3 [B,T,D] input, got {tuple(x.shape)}."
            )
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"SpectralBranch expected d_model={self.d_model}, got {x.shape[-1]}."
            )
        if x.shape[1] < 1:
            raise ValueError("SpectralBranch requires at least one time token.")
        if scope not in self.VALID_SCOPES:
            raise ValueError(
                f"scope must be one of {sorted(self.VALID_SCOPES)}, got {scope!r}."
            )
        B, L, D = x.shape
        # Permute to [B, D, L] for FFT along temporal axis
        x_perm = x.permute(0, 2, 1)  # [B, D, L]
        x_ft = torch.fft.rfft(x_perm, dim=-1)  # [B, D, n_freqs] complex
        n_freqs = x_ft.shape[-1]

        mode_indices = self._select_modes(x_ft, n_freqs)
        learned_mask = None
        learned_weight_real = None
        learned_weight_imag = None

        if mode_indices is None:
            mask_logits, weight_real, weight_imag = self._interpolate_learned_spectral_params(n_freqs)
            soft_mask = torch.sigmoid(mask_logits)
            learned_mask = soft_mask
            learned_weight_real = weight_real
            learned_weight_imag = weight_imag
            w_complex = torch.complex(weight_real, weight_imag).unsqueeze(0)
            out_ft = x_ft * w_complex * soft_mask
            x_reconstructed = torch.fft.irfft(out_ft, n=L)
        else:
            if mode_indices.ndim == 1:
                k = mode_indices.shape[0]
                selected = x_ft[:, :, mode_indices]
                w_real = self.weight_real[:, :k]
                w_imag = self.weight_imag[:, :k]
                w_complex = torch.complex(w_real, w_imag)
                transformed = selected * w_complex.unsqueeze(0)
                out_ft = torch.zeros_like(x_ft)
                out_ft[:, :, mode_indices] = transformed
            else:
                k = mode_indices.shape[-1]
                gather_index = mode_indices
                selected = torch.gather(x_ft, dim=-1, index=gather_index)
                w_real = self.weight_real[:, :k].unsqueeze(0)
                w_imag = self.weight_imag[:, :k].unsqueeze(0)
                w_complex = torch.complex(w_real, w_imag)
                transformed = selected * w_complex
                out_ft = torch.zeros_like(x_ft)
                out_ft.scatter_(dim=-1, index=gather_index, src=transformed)

            # Inverse FFT back to time domain: [B, D, L]
            x_reconstructed = torch.fft.irfft(out_ft, n=L)  # [B, D, L]
        x_reconstructed = x_reconstructed.permute(0, 2, 1)  # [B, L, D]

        output = self.layer_norm(x + self.dropout(self.out_projection(x_reconstructed)))
        if not return_diagnostics:
            return output
        diagnostics = self._build_diagnostics(
            x_ft=x_ft,
            n_fft=L,
            mode_indices=mode_indices,
            learned_mask=learned_mask,
            learned_weight_real=learned_weight_real,
            learned_weight_imag=learned_weight_imag,
            scope=scope,
        )
        return output, diagnostics


class TemporalCompression(nn.Module):
    """Learned strided temporal reduction for sequence compression.

    Compresses a sequence from length T to T // stride using a depthwise-separable
    strided convolution, and decompresses back via transposed convolution.
    Acts as a no-op when the input length is at or below *threshold*.

    Designed to reduce the O(T^2) cost of downstream attention branches by
    compressing the history portion of the sequence before attention.
    """

    def __init__(self, d_model: int, stride: int = 2, threshold: int = 256,
                 kernel_size: int | None = None, dropout: float = 0.0):
        super().__init__()
        if stride < 1:
            raise ValueError("stride must be >= 1.")
        self.d_model = d_model
        self.stride = stride
        self.threshold = threshold
        # Default kernel = 2 * stride (covers one full stride window on each side)
        self.kernel_size = kernel_size if kernel_size is not None else 2 * stride
        padding = (self.kernel_size - 1) // 2

        # Compress: depthwise-separable strided conv  (groups=d_model → depthwise)
        self.compress_dw = nn.Conv1d(
            d_model, d_model, kernel_size=self.kernel_size, stride=stride,
            padding=padding, groups=d_model, bias=False,
        )
        self.compress_pw = nn.Conv1d(d_model, d_model, kernel_size=1, bias=True)
        self.compress_norm = nn.LayerNorm(d_model)
        self.compress_act = nn.GELU()
        self.compress_drop = nn.Dropout(dropout)

        # Decompress: transposed conv (mirrors compress)
        self.decompress = nn.ConvTranspose1d(
            d_model, d_model, kernel_size=self.kernel_size, stride=stride,
            padding=padding, groups=d_model, bias=False,
        )
        self.decompress_pw = nn.Conv1d(d_model, d_model, kernel_size=1, bias=True)
        self.decompress_norm = nn.LayerNorm(d_model)
        self.decompress_drop = nn.Dropout(dropout)

    def should_compress(self, seq_len: int) -> bool:
        """Return True when compression is beneficial (long sequences)."""
        return self.stride > 1 and seq_len > self.threshold

    def compress(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Compress [B, T, D] -> [B, T', D] where T' ≈ T // stride.

        Returns (compressed, original_length) so decompress can restore size.
        """
        original_len = x.shape[1]
        # Conv1d expects [B, D, T]
        h = x.permute(0, 2, 1)
        h = self.compress_pw(self.compress_dw(h))  # [B, D, T']
        h = h.permute(0, 2, 1)  # [B, T', D]
        h = self.compress_drop(self.compress_act(self.compress_norm(h)))
        return h, original_len

    def decompress_to(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """Decompress [B, T', D] -> [B, target_len, D]."""
        h = x.permute(0, 2, 1)  # [B, D, T']
        h = self.decompress_pw(self.decompress(h))  # [B, D, ~T]
        h = h.permute(0, 2, 1)  # [B, ~T, D]
        # Transposed conv output may differ from target_len by ±1; trim or pad
        curr_len = h.shape[1]
        if curr_len > target_len:
            h = h[:, :target_len, :]
        elif curr_len < target_len:
            h = F.pad(h, (0, 0, 0, target_len - curr_len))
        h = self.decompress_drop(self.decompress_norm(h))
        return h


class MultiScaleLagAttention(nn.Module):
    def __init__(self, d_model, n_heads, lag_scales, dropout=0.0, position_bias_type='none', rope_base=10000.0, alibi_scale=1.0, attention_backend='exact', max_seq_len=None, debug_checks: bool = False, attn_dropout: float = 0.0, extension_semantics_version: int = 2):
        super(MultiScaleLagAttention, self).__init__()
        if not isinstance(lag_scales, (list, tuple)) or len(lag_scales) == 0:
            raise ValueError("tft_lag_scales must be a non-empty list/tuple of positive integers.")
        normalized_lags = []
        for lag in lag_scales:
            if not isinstance(lag, int) or lag <= 0:
                raise ValueError("tft_lag_scales must contain positive integers only.")
            normalized_lags.append(lag)
        if len(set(normalized_lags)) != len(normalized_lags):
            raise ValueError("tft_lag_scales contains duplicated lag values.")
        if extension_semantics_version not in (1, 2):
            raise ValueError("extension_semantics_version must be 1 or 2.")

        self.lag_scales = tuple(normalized_lags)
        self.extension_semantics_version = int(extension_semantics_version)
        self.attention_layers = nn.ModuleList([
            PositionalMultiHeadAttention(
                d_model,
                n_heads,
                dropout=dropout,
                position_bias_type=position_bias_type,
                rope_base=rope_base,
                alibi_scale=alibi_scale,
                attention_backend=attention_backend,
                debug_checks=debug_checks,
                attn_dropout=attn_dropout,
            )
            for _ in self.lag_scales
        ])
        self.scale_logits = nn.Parameter(torch.zeros(len(self.lag_scales)))
        self.out_projection = nn.Linear(d_model, d_model)
        self.out_dropout = nn.Dropout(dropout)
        if max_seq_len is not None:
            self.register_buffer('_causal_mask_buf', build_causal_mask(max_seq_len, torch.device('cpu'), torch.float32), persistent=False)
        else:
            self._causal_mask_buf = None
        self.debug_checks = debug_checks

    def _shift_sequence(self, x, lag: int):
        if lag <= 0 or lag >= x.shape[1]:
            raise ValueError(f"lag={lag} must be in [1, sequence_length-1] for active sequence length {x.shape[1]}.")
        # F.pad avoids allocating a full zero tensor; single fused op
        return F.pad(x[:, :-lag, :], (0, 0, lag, 0))

    def forward(
        self,
        x,
        return_attention: bool = False,
        positions=None,
        valid_mask=None,
    ):
        if x.ndim != 3:
            raise ValueError(f"MultiScaleLagAttention expects rank-3 [B,T,D] input, got shape {tuple(x.shape)}.")
        _debug_check_finite(self.debug_checks, x, "Lag attention input contains NaN/Inf values.")

        seq_len = x.shape[1]
        query_positions = _resolve_positions(
            seq_len,
            positions,
            x.device,
            batch_size=x.shape[0],
            name='positions',
        )
        resolved_valid = _resolve_true_valid_mask(
            valid_mask,
            x.shape[0],
            seq_len,
            x.device,
            name='valid_mask',
        )
        # Supplying an explicit all-valid mask is numerically identical to
        # omitting it, including the historical attention/dropout path.
        active_valid = (
            None
            if resolved_valid is None or bool(resolved_valid.all().item())
            else resolved_valid
        )
        if self._causal_mask_buf is not None and seq_len <= self._causal_mask_buf.shape[0]:
            attn_mask = self._causal_mask_buf[:seq_len, :seq_len].to(x.dtype)
        else:
            attn_mask = build_causal_mask(seq_len, x.device, x.dtype)
        branch_outputs = []
        branch_weights = []
        branch_key_positions = []
        branch_key_valid_masks = []
        for lag, attention_layer in zip(self.lag_scales, self.attention_layers):
            shifted = self._shift_sequence(x, lag)
            key_padding_mask = torch.zeros(x.shape[0], seq_len, dtype=torch.bool, device=x.device)
            key_padding_mask[:, :lag] = True
            if self.extension_semantics_version == 1:
                # Immutable legacy-v1 arithmetic treated a lag as a scalar
                # coordinate offset, including after lossy/compressed index
                # selection.  Semantics v2 below gathers each usable key's
                # actual source coordinate, which is required for irregular
                # physical time but is intentionally not backported to replay.
                key_positions = query_positions - float(lag)
            else:
                key_positions = _shift_source_positions(query_positions, lag)
            shifted_valid = torch.zeros(
                x.shape[0], seq_len, dtype=torch.bool, device=x.device
            )
            if active_valid is None:
                shifted_valid[:, lag:] = True
                key_valid_for_attention = None
            else:
                shifted_valid[:, lag:] = active_valid[:, :-lag]
                key_valid_for_attention = shifted_valid
            if return_attention:
                attn_out, attn_prob = attention_layer(
                    x,
                    shifted,
                    shifted,
                    return_attention=True,
                    attn_mask=attn_mask,
                    query_positions=query_positions,
                    key_positions=key_positions,
                    key_padding_mask=key_padding_mask,
                    query_valid_mask=active_valid,
                    key_valid_mask=key_valid_for_attention,
                )
            else:
                attn_out = attention_layer(
                    x,
                    shifted,
                    shifted,
                    attn_mask=attn_mask,
                    query_positions=query_positions,
                    key_positions=key_positions,
                    key_padding_mask=key_padding_mask,
                    query_valid_mask=active_valid,
                    key_valid_mask=key_valid_for_attention,
                )
                attn_prob = None
            _debug_check_finite(self.debug_checks, attn_out, "Lag attention output contains NaN/Inf values.")
            branch_outputs.append(attn_out)
            if return_attention:
                if attn_prob is None:
                    raise ValueError("Lag attention weights are missing.")
                _debug_check_finite(self.debug_checks, attn_prob, "Lag attention weights contain NaN/Inf values.")
                branch_weights.append(attn_prob)
                branch_key_positions.append(key_positions.detach())
                branch_key_valid_masks.append(shifted_valid.detach())

        scale_weights = torch.softmax(self.scale_logits, dim=0)
        fused = sum(weight * branch for weight, branch in zip(scale_weights, branch_outputs))
        fused = self.out_projection(self.out_dropout(fused))
        fused = _zero_invalid_queries(fused, active_valid)

        if return_attention:
            return fused, {
                'lag_attention': torch.stack(branch_weights, dim=-1),
                'lag_scale_weights': scale_weights.detach(),
                'lag_scales': self.lag_scales,
                'lag_attention_mode': 'shifted_history_attention',
                'lag_query_positions': query_positions.detach(),
                'lag_key_positions': torch.stack(branch_key_positions, dim=0),
                'lag_query_valid_mask': (
                    torch.ones(
                        x.shape[0], seq_len, dtype=torch.bool, device=x.device
                    )
                    if resolved_valid is None
                    else resolved_valid.detach()
                ),
                'lag_key_valid_masks': torch.stack(branch_key_valid_masks, dim=0),
            }
        return fused


class InterpretableCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, position_bias_type='none', rope_base=10000.0, alibi_scale=1.0, debug_checks: bool = False, attn_dropout: float = 0.0):
        super(InterpretableCrossAttention, self).__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads for InterpretableCrossAttention.")
        if position_bias_type not in {'none', 'rope', 'alibi'}:
            raise ValueError("position_bias_type must be one of: none, rope, alibi.")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.position_bias_type = position_bias_type
        self.rope_base = float(rope_base)
        self.alibi_scale = float(alibi_scale)
        self.debug_checks = debug_checks
        self.attn_dropout = float(attn_dropout)
        self.q_linear = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.k_linear = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.v_linear = nn.Linear(d_model, self.d_head, bias=False)
        self.out_projection = nn.Linear(self.d_head, d_model, bias=False)
        self.out_dropout = nn.Dropout(dropout)
        self.scale = self.d_head ** -0.5

    def forward(
        self,
        query,
        context,
        return_attention: bool = False,
        query_positions=None,
        key_positions=None,
        query_valid_mask=None,
        key_valid_mask=None,
    ):
        if query.ndim != 3 or context.ndim != 3:
            raise ValueError(
                f"InterpretableCrossAttention expects rank-3 query/context, got {tuple(query.shape)} and {tuple(context.shape)}."
            )
        if query.shape[0] != context.shape[0]:
            raise ValueError("Cross-attention query/context batch sizes must match.")
        batch_size = query.shape[0]
        query_len = query.shape[1]
        key_len = context.shape[1]
        if query_positions is not None:
            query_positions = _resolve_positions(
                query_len,
                query_positions,
                query.device,
                batch_size=batch_size,
                name='query_positions',
            )
        if key_positions is not None:
            key_positions = _resolve_positions(
                key_len,
                key_positions,
                query.device,
                batch_size=batch_size,
                name='key_positions',
            )
        resolved_query_valid = _resolve_true_valid_mask(
            query_valid_mask,
            batch_size,
            query_len,
            query.device,
            name='query_valid_mask',
        )
        resolved_key_valid = _resolve_true_valid_mask(
            key_valid_mask,
            batch_size,
            key_len,
            query.device,
            name='key_valid_mask',
        )
        q = self.q_linear(query).view(query.shape[0], query.shape[1], self.n_heads, self.d_head).permute(0, 2, 1, 3)
        k = self.k_linear(context).view(context.shape[0], context.shape[1], self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = self.v_linear(context)

        if self.position_bias_type == 'rope':
            q, k = apply_rotary_embedding(q, k, query_positions=query_positions, key_positions=key_positions, base=self.rope_base)

        attention_score = torch.matmul(q, k.transpose(-2, -1))
        attention_score = attention_score * self.scale
        if self.position_bias_type == 'alibi':
            alibi_bias = build_alibi_bias(
                self.n_heads,
                query.shape[1],
                context.shape[1],
                query.device,
                attention_score.dtype,
                query_positions=query_positions,
                key_positions=key_positions,
                scale=self.alibi_scale,
            )
            if alibi_bias.shape[0] not in (1, batch_size):
                raise ValueError(
                    "Batched ALiBi coordinates must match the attention batch size."
                )
            attention_score = attention_score + alibi_bias
        _debug_check_finite(self.debug_checks, attention_score, "Cross-attention scores contain NaN/Inf values.")
        if resolved_key_valid is not None and not bool(resolved_key_valid.all().item()):
            attention_score = attention_score.masked_fill(
                ~resolved_key_valid[:, None, None, :], float('-inf')
            )
            fully_masked = ~torch.isfinite(attention_score).any(dim=-1, keepdim=True)
            safe_scores = torch.where(
                fully_masked, torch.zeros_like(attention_score), attention_score
            )
            attention_prob = torch.softmax(safe_scores, dim=-1)
            attention_prob = torch.where(
                fully_masked, torch.zeros_like(attention_prob), attention_prob
            )
        else:
            # Preserve the historical all-valid arithmetic exactly.
            attention_prob = torch.softmax(attention_score, dim=-1)
        attention_prob = _zero_invalid_queries(attention_prob, resolved_query_valid)
        _debug_check_finite(self.debug_checks, attention_prob, "Cross-attention probabilities contain NaN/Inf values.")
        attention_prob_used = F.dropout(attention_prob, p=self.attn_dropout, training=self.training)

        attention_out = torch.matmul(attention_prob_used, v.unsqueeze(1))
        attention_out = attention_out.mean(dim=1)
        out = self.out_projection(attention_out)
        out = self.out_dropout(out)
        out = _zero_invalid_queries(out, resolved_query_valid)
        if return_attention:
            return out, attention_prob
        return out


class HigherOrderInteractionBlock(nn.Module):
    def __init__(self, d_model, interaction_order=2, interaction_rank=None, dropout=0.0, debug_checks: bool = False):
        super(HigherOrderInteractionBlock, self).__init__()
        if interaction_order not in (2, 3):
            raise ValueError("tft_interaction_order must be either 2 or 3.")
        if interaction_rank is None:
            interaction_rank = max(4, d_model // 4)
        if not isinstance(interaction_rank, int) or interaction_rank <= 0:
            raise ValueError("tft_interaction_rank must be a positive integer.")

        self.interaction_order = interaction_order
        self.num_terms = self.interaction_order - 1
        self.interaction_rank = interaction_rank
        self.left_projection = nn.Linear(d_model, interaction_rank)
        self.right_projection = nn.Linear(d_model, interaction_rank)
        self.pair_projection = nn.Linear(interaction_rank, d_model)
        if self.interaction_order == 3:
            self.third_projection = nn.Linear(d_model, interaction_rank)
            self.triple_projection = nn.Linear(interaction_rank, d_model)
        else:
            self.third_projection = None
            self.triple_projection = None
        self.gate_projection = nn.Linear(d_model, self.num_terms)
        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.debug_checks = debug_checks

    def forward(self, x, return_payload: bool = False):
        if x.ndim != 3:
            raise ValueError(f"HigherOrderInteractionBlock expects rank-3 [B,T,D] input, got shape {tuple(x.shape)}.")
        _debug_check_finite(self.debug_checks, x, "Higher-order interaction input contains NaN/Inf values.")

        left = self.left_projection(x)
        right = self.right_projection(x)
        pair_term = self.pair_projection((left * right) / (self.interaction_rank ** 0.5))
        interaction_terms = [pair_term]

        if self.interaction_order == 3:
            third = self.third_projection(x)
            triple_term = self.triple_projection((left * right * third) / self.interaction_rank)
            interaction_terms.append(triple_term)

        gates = torch.sigmoid(self.gate_projection(x))
        interaction_stack = torch.stack(interaction_terms, dim=-2)
        contribution = torch.sum(gates.unsqueeze(-1) * interaction_stack, dim=-2)
        out = self.layer_norm(x + self.out_projection(self.dropout(contribution)))
        _debug_check_finite(self.debug_checks, out, "Higher-order interaction output contains NaN/Inf values.")

        if return_payload:
            return out, {
                'interaction_contribution': contribution.detach(),
                'interaction_gates': gates.detach(),
            }
        return out


class RegimeAwareSparseMoE(nn.Module):
    def __init__(
        self,
        d_model,
        num_experts=4,
        top_k=2,
        num_regimes=4,
        hidden_size=None,
        dropout=0.0,
        noise_epsilon=1e-2,
    ):
        super(RegimeAwareSparseMoE, self).__init__()
        if num_experts <= 0:
            raise ValueError("tft_num_moe_experts must be positive.")
        if num_regimes <= 0:
            raise ValueError("tft_num_regimes must be positive.")
        if top_k <= 0 or top_k > num_experts:
            raise ValueError("tft_moe_top_k must be in [1, tft_num_moe_experts].")
        hidden_size = d_model if not hidden_size or hidden_size <= 0 else hidden_size

        self.num_experts = num_experts
        self.top_k = top_k
        self.num_regimes = num_regimes
        self.noise_epsilon = noise_epsilon
        self.capacity_factor = 1.25  # default; can be overridden via config
        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()
        self.regime_detector = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_regimes),
        )
        self.context_to_regime = nn.Linear(d_model, num_regimes, bias=False)
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.noise = nn.Linear(d_model, num_experts, bias=False)
        self.regime_expert_bias = nn.Parameter(torch.zeros(num_regimes, num_experts))
        # Fused expert parameters for batched computation (no sequential loop)
        self.expert_w1 = nn.Parameter(torch.empty(num_experts, d_model, hidden_size))
        self.expert_b1 = nn.Parameter(torch.zeros(num_experts, hidden_size))
        self.expert_w2 = nn.Parameter(torch.empty(num_experts, hidden_size, d_model))
        self.expert_b2 = nn.Parameter(torch.zeros(num_experts, d_model))
        for i in range(num_experts):
            nn.init.xavier_uniform_(self.expert_w1.data[i])
            nn.init.xavier_uniform_(self.expert_w2.data[i])
        self.expert_activation = nn.GELU()
        self.expert_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.debug_checks = False

    def cv_squared(self, x):
        eps = 1e-10
        if x.numel() <= 1:
            return x.new_tensor(0.0)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _compute_sparse_routing(self, x, regime_probs):
        if regime_probs.ndim != 3:
            raise ValueError(f"Expected timestep regime probabilities [B,T,R], got shape {tuple(regime_probs.shape)}.")
        logits = self.gate(x)
        if self.training:
            raw_noise = self.noise(x)
            noise_std = self.softplus(raw_noise) + self.noise_epsilon
            logits = logits + torch.randn_like(logits) * noise_std

        logits = logits + torch.einsum('btr,re->bte', regime_probs, self.regime_expert_bias)
        dense_probs = self.softmax(logits)
        top_values, top_indices = torch.topk(dense_probs, self.top_k, dim=-1)
        sparse_probs = torch.zeros_like(dense_probs)
        sparse_probs.scatter_(-1, top_indices, top_values)

        # Expert capacity constraint: cap tokens per expert to prevent monopolization
        if self.training and hasattr(self, 'capacity_factor'):
            B_dim, T_dim = sparse_probs.shape[0], sparse_probs.shape[1]
            token_count = B_dim * T_dim
            if token_count > 0:
                capacity = max(1, math.ceil(self.capacity_factor * token_count * self.top_k / self.num_experts))
                flat_probs = sparse_probs.reshape(-1, self.num_experts).contiguous()  # [B*T, E]
                keep_k = min(capacity, flat_probs.shape[0])
                expert_scores = flat_probs.transpose(0, 1)  # [E, B*T]
                top_scores, top_token_idx = torch.topk(expert_scores, keep_k, dim=-1)
                keep_mask = torch.zeros_like(expert_scores, dtype=torch.bool)
                keep_mask.scatter_(1, top_token_idx, top_scores > 0)
                pruned = torch.where(keep_mask.transpose(0, 1), flat_probs, torch.zeros_like(flat_probs))
                zero_token_mask = pruned.sum(dim=-1, keepdim=True) <= 0
                if zero_token_mask.any():
                    dense_flat = dense_probs.reshape(-1, self.num_experts)
                    fallback_idx = dense_flat.argmax(dim=-1, keepdim=True)
                    fallback_weight = dense_flat.gather(-1, fallback_idx)
                    fallback = torch.zeros_like(pruned).scatter_(-1, fallback_idx, fallback_weight)
                    pruned = torch.where(zero_token_mask, fallback, pruned)
                sparse_probs = pruned.reshape(B_dim, T_dim, self.num_experts)

        sparse_probs = sparse_probs / sparse_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        importance = sparse_probs.sum(dim=(0, 1))
        load = (sparse_probs > 0).to(sparse_probs.dtype).sum(dim=(0, 1))
        aux_loss = self.cv_squared(importance) + self.cv_squared(load)
        return sparse_probs, aux_loss, importance, load

    def forward(self, x, context: Optional[torch.Tensor] = None, return_payload: bool = False):
        if x.ndim != 3:
            raise ValueError(f"RegimeAwareSparseMoE expects rank-3 [B,T,D] input, got shape {tuple(x.shape)}.")
        _debug_check_finite(self.debug_checks, x, "MoE input contains NaN/Inf values.")

        regime_logits = self.regime_detector(x)
        if context is not None:
            if context.ndim == 2:
                regime_logits = regime_logits + self.context_to_regime(context).unsqueeze(1)
            elif context.ndim == 3:
                regime_logits = regime_logits + self.context_to_regime(context)
            else:
                raise ValueError(f"MoE context must be rank-2 [B,D] or rank-3 [B,T,D], got shape {tuple(context.shape)}.")
        regime_probs = self.softmax(regime_logits)
        pooled_regime_probs = regime_probs.mean(dim=1)
        routing, aux_loss, importance_sum, load_sum = self._compute_sparse_routing(x, regime_probs)
        token_count = routing.new_tensor([routing.shape[0] * routing.shape[1]])
        # Batched expert evaluation: all experts in parallel via fused parameters
        # x: [B,T,D], expert_w1: [E,D,H] -> hidden: [B,T,E,H]
        hidden = torch.einsum('btd,edh->bteh', x, self.expert_w1) + self.expert_b1
        hidden = self.expert_activation(hidden)
        hidden = self.expert_dropout(hidden)
        # hidden: [B,T,E,H], expert_w2: [E,H,D] -> expert_outputs: [B,T,E,D]
        expert_outputs = torch.einsum('bteh,ehd->bted', hidden, self.expert_w2) + self.expert_b2
        mixed = torch.sum(routing.unsqueeze(-1) * expert_outputs, dim=-2)
        out = self.layer_norm(x + self.out_dropout(mixed))
        _debug_check_finite(self.debug_checks, out, "MoE output contains NaN/Inf values.")

        if return_payload:
            return out, aux_loss, {
                'expert_routing': routing.detach(),
                'regime_probabilities': regime_probs.detach(),
                'regime_probabilities_pooled': pooled_regime_probs.detach(),
                'moe_routing_mode': 'dense_compute_topk_mixing',
                'expert_importance_sum': importance_sum.unsqueeze(0),
                'expert_load_sum': load_sum.unsqueeze(0),
                'expert_token_count': token_count,
            }
        return out, aux_loss
