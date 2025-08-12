"""
Simple ChronosX Real Data Demo - Working Example

Run with --smoke for a fast offline demo (no downloads). This script avoids
Unicode issues on Windows consoles by using ASCII fallbacks when stdout is not UTF-8.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def _icon(utf8_symbol: str, fallback: str) -> str:
    """Return a UTF-8 symbol or a safe ASCII fallback based on stdout encoding."""
    enc = (sys.stdout.encoding or "").upper()
    return utf8_symbol if "UTF-8" in enc else fallback


def _safe_load_series(csv_path: str) -> np.ndarray:
    """Load a numeric series from CSV or generate a fallback synthetic series.

    Parameters
    ----------
    csv_path: str
        Path to a CSV file that contains at least one numeric column.

    Returns
    -------
    np.ndarray
        1D array of values representing a time series.
    """
    try:
        df = pd.read_csv(csv_path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found")
        return df[numeric_cols[0]].values
    except Exception:
        # Fallback synthetic series
        rng = np.random.default_rng(42)
        t = np.arange(400)
        trend = 0.02 * t
        seasonal = 0.5 * np.sin(2 * np.pi * t / 24)
        noise = 0.1 * rng.normal(size=len(t))
        return (100 + trend + seasonal + noise).astype(np.float32)


def simple_chronos_demo(smoke: bool = False) -> Dict[str, Any]:
    """Run a simple ChronosX demo with optional smoke (offline) mode.

    Parameters
    ----------
    smoke: bool
        When True, avoid external downloads and simulate a forecast quickly.

    Returns
    -------
    Dict[str, Any]
        Dictionary with forecast arrays, timings, and success flag.
    """
    rocket = _icon("🚀", "[GO]")
    stats_icon = _icon("📊", "Data")
    setup_icon = _icon("🔮", "Setup")
    fast_icon = _icon("⚡", "[FAST]")
    ok_icon = _icon("✅", "OK")
    predict_icon = _icon("🎯", "Predict")
    plot_icon = _icon("📈", "Plot")
    done_icon = _icon("🎉", "Done")

    print(f"{rocket} ChronosX Real Data Forecasting Demo")
    print("=" * 45)

    # Load real/synthetic data
    print(f"{stats_icon} Loading time series data...")
    data = _safe_load_series("synthetic_timeseries.csv")
    print(f"   Loaded {len(data)} data points")
    print(f"   Range: [{data.min():.3f}, {data.max():.3f}]")

    # Forecasting parameters
    context_length = 96
    forecast_length = 24

    print(f"\n{setup_icon} Forecasting Setup:")
    print(f"   Context length: {context_length} points")
    print(f"   Forecast length: {forecast_length} points")

    load_time = 0.0
    forecast_time = 0.0

    if smoke or os.environ.get("DEMO_SMOKE") == "1":
        print(f"\n{fast_icon} Smoke mode enabled: simulating forecast...")
        # Create deterministic pseudo-forecast from context (e.g., last value + noise)
        context = torch.tensor(data[-context_length:], dtype=torch.float32).unsqueeze(0)
        start_time = time.time()
        rng = np.random.default_rng(0)
        last = context.squeeze().numpy()[-1]
        mean_forecast = last + np.cumsum(0.01 * rng.standard_normal(forecast_length))
        std_forecast = np.full_like(mean_forecast, 0.05, dtype=np.float64)
        # Create 10 sample trajectories around the mean for UQ display
        samples = np.stack(
            [mean_forecast + 0.05 * rng.standard_normal(forecast_length) for _ in range(10)],
            axis=0,
        )
        forecast_time = time.time() - start_time
        forecast_np = samples.astype(np.float32)
    else:
        # Load ChronosX (may download weights)
        print(f"\n{fast_icon} Loading ChronosX tiny model...")
        start_time = time.time()
        from chronos import ChronosPipeline  # noqa: WPS433 - local import to avoid dependency in smoke

        pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-tiny",
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        load_time = time.time() - start_time
        print(f"   {ok_icon} Model loaded in {load_time:.2f} seconds")

        # Prepare context data
        context = torch.tensor(data[-context_length:], dtype=torch.float32).unsqueeze(0)
        print(f"   Context shape: {context.shape}")

        # Generate forecast
        print(f"\n{predict_icon} Generating forecast...")
        start_time = time.time()
        forecast = pipeline.predict(
            context=context,
            prediction_length=forecast_length,
            num_samples=10,
        )
        forecast_time = time.time() - start_time
        print(f"   {ok_icon} Forecast generated in {forecast_time:.2f} seconds")

        forecast_np = forecast.cpu().numpy()
        mean_forecast = np.mean(forecast_np, axis=0).flatten()
        std_forecast = np.std(forecast_np, axis=0).flatten()

    # Visualization
    print(f"\n{plot_icon} Creating visualization...")

    plt.figure(figsize=(14, 8))

    # Historical data
    hist_x = np.arange(len(data))
    plt.plot(
        hist_x[-200:],
        data[-200:],
        label="Historical Data",
        color="blue",
        alpha=0.7,
        linewidth=1,
    )

    # Context
    context_x = hist_x[-context_length:]
    plt.plot(
        context_x,
        data[-context_length:],
        label="Context (Input)",
        color="green",
        linewidth=2,
    )

    # Forecast
    forecast_x = np.arange(len(data), len(data) + forecast_length)
    plt.plot(
        forecast_x,
        mean_forecast,
        label=("ChronosX Forecast" if not smoke else "Simulated Forecast"),
        color="red",
        linewidth=2,
        marker="o",
        markersize=4,
    )

    # Uncertainty bands
    plt.fill_between(
        forecast_x,
        mean_forecast - std_forecast,
        mean_forecast + std_forecast,
        alpha=0.3,
        color="red",
        label="±1σ Uncertainty",
    )
    plt.fill_between(
        forecast_x,
        mean_forecast - 2 * std_forecast,
        mean_forecast + 2 * std_forecast,
        alpha=0.2,
        color="red",
        label="±2σ Uncertainty",
    )

    # Formatting
    plt.title(
        "ChronosX Real Time Series Forecasting with Uncertainty",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Performance annotation
    plt.figtext(
        0.02,
        0.02,
        f"Performance: Load {load_time:.1f}s | Forecast {forecast_time:.1f}s | "
        f"Avg Uncertainty ±{np.mean(std_forecast):.3f}",
        fontsize=10,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig("chronos_x_simple_demo.png", dpi=300, bbox_inches="tight")
    print(f"   {stats_icon} Saved plot: chronos_x_simple_demo.png")

    # Summary
    print(f"\n{done_icon} Demo Complete! Results Summary:")
    print("   🔸 Mode: Smoke" if smoke else "   🔸 Mode: Full ChronosX")
    print("   🔸 Forecast Points:", forecast_length)

    return {
        "mean_forecast": np.asarray(mean_forecast),
        "std_forecast": np.asarray(std_forecast),
        "context": data[-context_length:],
        "load_time": load_time,
        "forecast_time": forecast_time,
        "success": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChronosX simple demo")
    parser.add_argument("--smoke", action="store_true", help="Run in fast offline mode")
    args = parser.parse_args()

    _ = simple_chronos_demo(smoke=args.smoke)
