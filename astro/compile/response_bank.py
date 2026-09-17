"""Calendar-time exponential response kernels.

Orbital phase says where a body is in its cycle; a response kernel says how long
an alleged effect persists.  Because the kernel runs on the daily ephemeris grid
rather than the trading grid, state evolves across weekends and holidays with
real elapsed time.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter


def ewma_response(
    intensity: np.ndarray, half_life_days: int, elapsed_days: np.ndarray | None = None
) -> np.ndarray:
    """Apply ``h(t) = a^dt*h(t-1) + (1 - a^dt)*z(t)`` with ``a = 2^(-1/H)``.

    The state is initialized at ``z(0)`` rather than zero, so a long half-life
    does not encode an arbitrary cold-start value as if it were a real slow
    planetary state.

    Being a convex combination of inputs, the output stays within the input's
    declared range.
    """

    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive.")
    values = np.asarray(intensity, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"ewma_response expects a 1-D series, got shape {values.shape}.")
    if values.size == 0:
        return values.copy()

    decay = 2.0 ** (-1.0 / float(half_life_days))

    if elapsed_days is None:
        # Uniform daily grid: a first-order IIR filter, seeded at the steady
        # state so the response starts warm.
        filtered = lfilter([1.0 - decay], [1.0, -decay], values - values[0])
        return filtered + values[0]

    gaps = np.asarray(elapsed_days, dtype=np.float64)
    if gaps.shape != values.shape:
        raise ValueError("elapsed_days must have the same shape as intensity.")
    output = np.empty_like(values)
    output[0] = values[0]
    for position in range(1, values.size):
        step_decay = decay ** gaps[position]
        output[position] = step_decay * output[position - 1] + (1.0 - step_decay) * values[position]
    return output


def elapsed_days_from_index(index) -> np.ndarray:
    """Calendar days elapsed since the previous row, first entry zero."""

    nanoseconds = np.asarray(index.asi8, dtype=np.float64)
    gaps = np.diff(nanoseconds) / (24.0 * 60.0 * 60.0 * 1e9)
    return np.concatenate([[0.0], gaps])


def is_uniform_daily(elapsed_days: np.ndarray) -> bool:
    if elapsed_days.size <= 1:
        return True
    return bool(np.allclose(elapsed_days[1:], 1.0, atol=1e-9))
