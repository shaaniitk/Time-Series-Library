"""Matched placebo arms for a compiled ruleset.

Every null arm recompiles or transforms the *same* ruleset, so its channel
names, count, dtype and declared ranges are identical to the real arm by
construction.  Only the alignment between astronomy and calendar is broken.

- ``date_shift``: rolls the whole compiled block coherently in time, keeping
  every cross-channel relationship but misaligning it with market dates.
- ``phase_randomize``: a multivariate phase surrogate (one shared random phase
  per frequency) that preserves each channel's power spectrum and the
  cross-spectrum, then rank-remaps each channel onto its original values so the
  marginal distribution and declared range are preserved exactly.
- ``body_permute``: reassigns planetary tracks to the wrong body names before
  compilation, keeping realistic orbital motion but destroying body identity.
"""

from __future__ import annotations

import numpy as np

from astro.compile.compiler import ChannelMeta, CompiledFeatures, compile_ruleset
from astro.ephemeris.contract import PLANET_BODIES, column_name, required_fields
from astro.ephemeris.provider import EphemerisProvider
from astro.rules.naming import channel_name_hash
from astro.rules.registry import RuleContractError
from astro.rules.schema import NullSpec, RuleSet

_MIN_SHIFT_DAYS = 30


def _relabel(features: CompiledFeatures, matrix: np.ndarray, arm: str) -> CompiledFeatures:
    metas = tuple(
        ChannelMeta(**{**meta.__dict__, "arm": arm}) for meta in features.channel_meta
    )
    manifest = {**features.manifest, "arm": arm}
    manifest["channel_name_hash"] = channel_name_hash(features.names)
    return CompiledFeatures(
        names=features.names,
        matrix=matrix,
        channel_meta=metas,
        index=features.index,
        manifest=manifest,
    )


def _date_shift(features: CompiledFeatures, params: dict, arm: str) -> CompiledFeatures:
    days = params.get("days")
    if isinstance(days, bool) or not isinstance(days, int) or days < _MIN_SHIFT_DAYS:
        raise RuleContractError(
            f"date_shift requires integer days >= {_MIN_SHIFT_DAYS}, got {days!r}."
        )
    if days >= features.matrix.shape[0]:
        raise RuleContractError(
            f"date_shift of {days} days exceeds the {features.matrix.shape[0]}-day ephemeris."
        )
    return _relabel(features, np.roll(features.matrix, days, axis=0), arm)


def _phase_randomize(features: CompiledFeatures, params: dict, arm: str) -> CompiledFeatures:
    seed = params.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise RuleContractError(f"phase_randomize requires an integer seed, got {seed!r}.")

    matrix = features.matrix
    length = matrix.shape[0]
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(matrix, axis=0)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=spectrum.shape[0])
    # Keep the mean (and the real-valued Nyquist bin) untouched.
    phases[0] = 0.0
    if length % 2 == 0:
        phases[-1] = 0.0
    surrogate = np.fft.irfft(spectrum * np.exp(1j * phases)[:, None], n=length, axis=0)

    remapped = np.empty_like(matrix)
    for column in range(matrix.shape[1]):
        ranks = np.argsort(np.argsort(surrogate[:, column], kind="stable"), kind="stable")
        remapped[:, column] = np.sort(matrix[:, column])[ranks]
    return _relabel(features, remapped, arm)


def _permuted_provider(provider: EphemerisProvider, seed: int) -> EphemerisProvider:
    bodies = [body for body in PLANET_BODIES if body in provider.bodies]
    if len(bodies) < 2:
        raise RuleContractError("body_permute needs at least two planetary bodies.")
    rng = np.random.default_rng(seed)
    # Require a derangement so no body keeps its own track.
    while True:
        order = rng.permutation(len(bodies))
        if not np.any(order == np.arange(len(bodies))):
            break

    source = provider._frame
    permuted = source.copy()
    for target, origin in zip(bodies, (bodies[i] for i in order)):
        for field in required_fields(target):
            permuted[column_name(target, field)] = source[column_name(origin, field)].to_numpy()
    return EphemerisProvider(permuted, provider.manifest)


def compile_null_arm(
    ruleset: RuleSet,
    provider: EphemerisProvider,
    null: NullSpec,
    real: CompiledFeatures | None = None,
) -> CompiledFeatures:
    """Build one placebo arm with the same layout as the real arm."""

    arm = null.arm_name
    params = null.param_dict

    if null.type == "body_permute":
        seed = params.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise RuleContractError(f"body_permute requires an integer seed, got {seed!r}.")
        features = compile_ruleset(ruleset, _permuted_provider(provider, seed), arm=arm)
    else:
        base = real if real is not None else compile_ruleset(ruleset, provider)
        if null.type == "date_shift":
            features = _date_shift(base, params, arm)
        else:
            features = _phase_randomize(base, params, arm)

    if real is not None:
        assert_arms_matched(real, features)
    return features


def assert_arms_matched(real: CompiledFeatures, null: CompiledFeatures) -> None:
    """Fail unless a null arm is capacity-matched and date-aligned with the real arm."""

    if real.names != null.names:
        raise RuleContractError("Null arm channel names differ from the real arm.")
    if real.matrix.shape != null.matrix.shape or real.matrix.dtype != null.matrix.dtype:
        raise RuleContractError(
            f"Null arm shape/dtype {null.matrix.shape}/{null.matrix.dtype} differs from "
            f"real arm {real.matrix.shape}/{real.matrix.dtype}."
        )
    if not real.index.equals(null.index):
        raise RuleContractError("Null arm dates differ from the real arm.")
    for meta in null.channel_meta:
        column = null.matrix[:, null.names.index(meta.name)]
        if column.min() < meta.low - 1e-9 or column.max() > meta.high + 1e-9:
            raise RuleContractError(f"Null arm channel {meta.name!r} left its declared range.")
