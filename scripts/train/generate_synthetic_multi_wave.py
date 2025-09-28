"""Utility for generating a synthetic multi-wave / multi-target dataset.

The generator creates sinusoidal waveforms whose amplitudes oscillate between
0.9 and 1.0 while their phase velocities drift within [-1.0, 1.0]. Targets are
constructed as noisy linear combinations of the wave signals with additional
non-linear perturbations, yielding a rich training corpus for PGAT experiments.

Example
-------
python scripts/train/generate_synthetic_multi_wave.py \
    --output data/synthetic_multi_wave.csv \
    --rows 2048 --waves 7 --targets 3 --seed 1337
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def _smooth_random_walk(
    base: np.ndarray,
    steps: int,
    amplitude: float,
    low: float,
    high: float,
) -> np.ndarray:
    """Return smoothly varying sequences centred at ``base``.

    Parameters
    ----------
    base:
        Initial mean value per series, shape ``(series,)``.
    steps:
        Number of timesteps to generate.
    amplitude:
        Magnitude of the random walk increments per step.
    low, high:
        Hard clipping bounds applied elementwise to keep the process stable.
    """

    noise = amplitude * np.random.uniform(-1.0, 1.0, size=(base.size, steps))
    walk = np.cumsum(noise, axis=1)
    series = base[:, None] + walk
    return np.clip(series, low, high)


def _build_wave_signals(
    timesteps: np.ndarray,
    waves: int,
    *,
    amplitude_bounds: Tuple[float, float] = (0.9, 1.0),
    phase_velocity_bounds: Tuple[float, float] = (-1.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct synthetic wave inputs with drifting amplitude and phase velocity."""

    base_amplitude = np.random.uniform(amplitude_bounds[0], amplitude_bounds[1], size=waves)
    base_phase_velocity = np.random.uniform(phase_velocity_bounds[0], phase_velocity_bounds[1], size=waves)
    base_frequency = np.random.uniform(0.005, 0.02, size=waves)
    phase_offset = np.random.uniform(0.0, 2.0 * math.pi, size=waves)

    amplitudes = _smooth_random_walk(
        base_amplitude,
        timesteps.size,
        amplitude=0.01,
        low=amplitude_bounds[0],
        high=amplitude_bounds[1],
    )
    phase_velocity = _smooth_random_walk(
        base_phase_velocity,
        timesteps.size,
        amplitude=0.05,
        low=phase_velocity_bounds[0],
        high=phase_velocity_bounds[1],
    )

    wave_values = []
    for idx in range(waves):
        phase = 2.0 * math.pi * base_frequency[idx] * timesteps
        phase += phase_offset[idx]
        phase += phase_velocity[idx]
        signal = amplitudes[idx] * np.sin(phase)
        noise = 0.03 * np.random.normal(size=timesteps.size)
        wave_values.append(signal + noise)

    wave_matrix = np.stack(wave_values, axis=1)
    return wave_matrix, amplitudes, phase_velocity, base_frequency


def _build_targets(waves: np.ndarray, targets: int) -> np.ndarray:
    """Generate target series from the synthetic wave signals."""

    num_samples, num_waves = waves.shape
    mixing = np.random.uniform(-0.6, 0.6, size=(num_waves, targets))
    base_targets = waves @ mixing

    # Inject non-linear harmonics to avoid trivial linear dependence
    harmonics = []
    for tgt in range(targets):
        mod_wave = np.sin(0.5 * base_targets[:, tgt]) + 0.2 * np.cos(0.25 * base_targets[:, tgt])
        harmonics.append(mod_wave)
    harmonic_matrix = np.stack(harmonics, axis=1)

    seasonal = 0.1 * np.sin(np.linspace(0, 6 * math.pi, num_samples))[:, None]
    noise = 0.02 * np.random.normal(size=(num_samples, targets))
    return base_targets + harmonic_matrix + seasonal + noise


def generate_dataset(args: argparse.Namespace) -> pd.DataFrame:
    """Construct the synthetic dataset respecting amplitude/phase constraints."""

    timesteps = np.arange(args.rows, dtype=float)
    wave_matrix, amplitudes, phase_velocity, frequencies = _build_wave_signals(timesteps, args.waves)
    target_matrix = _build_targets(wave_matrix, args.targets)

    covariate_1 = np.gradient(wave_matrix.mean(axis=1))
    covariate_2 = np.var(wave_matrix[:, : max(1, args.waves // 2)], axis=1)

    index = pd.date_range(start=args.start, periods=args.rows, freq=args.freq)

    data: Dict[str, Any] = {
        "date": index,
    }
    for idx in range(args.waves):
        data[f"wave_{idx}"] = wave_matrix[:, idx]
    for idx in range(args.targets):
        data[f"target_{idx}"] = target_matrix[:, idx]

    data["covariate_gradient"] = covariate_1
    data["covariate_dispersion"] = covariate_2

    df = pd.DataFrame(data)

    metadata = {
        "wave_amplitude_mean": amplitudes.mean(axis=1),
        "wave_amplitude_std": amplitudes.std(axis=1),
        "wave_phase_velocity_mean": phase_velocity.mean(axis=1),
        "wave_frequency": frequencies,
    }

    meta_frame = pd.DataFrame(metadata)
    meta_path = Path(args.output).with_suffix(".meta.csv")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_frame.to_csv(meta_path, index=False)

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_multi_wave.csv"))
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--waves", type=int, default=7)
    parser.add_argument("--targets", type=int, default=3)
    parser.add_argument("--freq", type=str, default="H", help="Pandas offset alias (e.g., H, 15T, D).")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start timestamp for generated data.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    df = generate_dataset(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Synthetic dataset written to {args.output}")
    print(f"Metadata (amplitude/phase velocity summary) -> {args.output.with_suffix('.meta.csv')}")
    print(
        f"Column summary: {len([c for c in df.columns if c.startswith('wave_')])} waves, "
        f"{len([c for c in df.columns if c.startswith('target_')])} targets, "
        f"{df.shape[0]} rows"
    )


if __name__ == "__main__":
    main()
