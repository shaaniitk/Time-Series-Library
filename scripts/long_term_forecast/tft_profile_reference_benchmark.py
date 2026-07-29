import argparse
import io
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.TemporalFusionTransformer import Model
from utils.metrics import metric, quantile_metric
from utils.tft_interpretation import summarize_tft_interpretation
from utils.tft_config import apply_tft_profile
from utils.tft_synthetic import make_multiscale_tft_tensors


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark canonical vs extended_safe TFT profiles.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=str, default="results/tft_profile_reference_benchmark")
    parser.add_argument("--seeds", type=str, default="7,13")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=48)
    parser.add_argument("--val-samples", type=int, default=16)
    parser.add_argument("--test-samples", type=int, default=16)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def resolve_device(choice):
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_base_config():
    return SimpleNamespace(
        task_name="long_term_forecast",
        is_training=1,
        model_id="tft_profile_ref",
        model="TemporalFusionTransformer",
        data="synthetic_tft_profile_ref",
        root_path="",
        data_path="",
        features="M",
        target="OT",
        freq="h",
        checkpoints="./checkpoints/",
        seq_len=24,
        label_len=12,
        pred_len=4,
        seasonal_patterns="Monthly",
        inverse=False,
        mask_rate=0.25,
        anomaly_ratio=0.25,
        expand=2,
        d_conv=4,
        top_k=5,
        num_kernels=6,
        enc_in=6,
        dec_in=2,
        c_out=2,
        d_model=32,
        n_heads=4,
        e_layers=2,
        d_layers=1,
        d_ff=128,
        moving_avg=25,
        factor=1,
        distil=True,
        dropout=0.1,
        embed="timeF",
        activation="gelu",
        channel_independence=1,
        decomp_method="moving_avg",
        use_norm=1,
        down_sampling_layers=0,
        down_sampling_window=1,
        down_sampling_method=None,
        seg_len=24,
        num_workers=0,
        itr=1,
        train_epochs=4,
        batch_size=8,
        patience=2,
        learning_rate=3e-3,
        des="benchmark",
        loss="MSE",
        lradj="type1",
        use_amp=False,
        use_gpu=False,
        gpu=0,
        gpu_type="cuda",
        use_multi_gpu=False,
        devices="0",
        p_hidden_dims=[8, 8],
        p_hidden_layers=2,
        use_dtw=False,
        augmentation_ratio=0,
        seed=7,
        jitter=False,
        scaling=False,
        permutation=False,
        randompermutation=False,
        magwarp=False,
        timewarp=False,
        windowslice=False,
        windowwarp=False,
        rotation=False,
        spawner=False,
        dtwwarp=False,
        shapedtwwarp=False,
        wdba=False,
        discdtw=False,
        discsdtw=False,
        extra_tag="",
        patch_len=16,
        node_dim=10,
        gcn_depth=2,
        gcn_dropout=0.3,
        propalpha=0.3,
        conv_channel=32,
        skip_channel=32,
        individual=False,
        tft_profile="extended_safe",
        tft_observed_pos=list(range(6)),
        tft_static_pos=[],
        tft_target_pos=[0, 1],
        tft_allow_custom_known=True,
        tft_known_len=6,
        tft_known_max_channels=16,
        tft_known_feature_names=[f"known_{i}" for i in range(6)],
        tft_output_quantiles=[0.1, 0.5, 0.9],
    )


def make_dataset(cfg, n_samples, device):
    tensors = make_multiscale_tft_tensors(
        seq_len=cfg.seq_len,
        label_len=cfg.label_len,
        pred_len=cfg.pred_len,
        enc_in=cfg.enc_in,
        c_out=cfg.c_out,
        known_len=cfg.tft_known_len,
        n_samples=n_samples,
        noise_std=0.01,
        device=device,
    )
    x_dec_full = torch.zeros(n_samples, cfg.label_len + cfg.pred_len, cfg.c_out, device=device)
    x_dec_full[:, -cfg.pred_len:, :] = tensors["y_future"]
    dataset = TensorDataset(
        tensors["x_enc"],
        x_dec_full,
        tensors["x_mark_enc"],
        tensors["x_mark_dec"],
        tensors["y_future"],
    )
    return dataset


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def checkpoint_size_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(buf.getvalue()) / (1024 ** 2)


def measure_latency_and_memory(model, batch, device, steps=5):
    model.eval()
    x_enc, x_dec, x_mark_enc, x_mark_dec, _ = batch
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    with torch.inference_mode():
        for _ in range(2):
            _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(steps):
            _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
    peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else None
    return 1000.0 * elapsed / max(steps, 1), peak_memory_mb


def train_and_evaluate_profile(cfg, profile, seed, loaders, device):
    set_seed(seed)
    run_cfg = SimpleNamespace(**vars(cfg))
    run_cfg.tft_profile = profile
    run_cfg = apply_tft_profile(run_cfg)
    model = Model(run_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=run_cfg.learning_rate)
    criterion = torch.nn.MSELoss()

    train_loader, val_loader, test_loader = loaders
    for _ in range(run_cfg.train_epochs):
        model.train()
        for x_enc, x_dec, x_mark_enc, x_mark_dec, y_future in train_loader:
            optimizer.zero_grad()
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec, return_auxiliary=True)
            pred = output.point_forecast
            loss = criterion(pred, y_future)
            if output.quantile_forecast is not None:
                loss = loss + 0.1 * (output.quantile_forecast[:, :, 1, :] - y_future).abs().mean()
            loss.backward()
            optimizer.step()

    model.eval()
    preds = []
    truths = []
    quantiles = []
    interpretation_summaries = []
    with torch.inference_mode():
        for batch_idx, (x_enc, x_dec, x_mark_enc, x_mark_dec, y_future) in enumerate(test_loader):
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec, return_auxiliary=True)
            preds.append(output.point_forecast.detach().cpu())
            truths.append(y_future.detach().cpu())
            if output.quantile_forecast is not None:
                quantiles.append(output.quantile_forecast.detach().cpu())
            if batch_idx == 0:
                payload = model(x_enc[:1], x_mark_enc[:1], x_dec[:1], x_mark_dec[:1], return_interpretation=True)
                interpretation_summaries.append(summarize_tft_interpretation(payload, top_k=3))

    pred = torch.cat(preds, dim=0).numpy()
    true = torch.cat(truths, dim=0).numpy()
    mse, mae, rmse, mape, mspe = metric(pred, true)

    quantile_summary = None
    if quantiles:
        quantile_pred = torch.cat(quantiles, dim=0).numpy()
        quantile_summary = quantile_metric(quantile_pred, true, run_cfg.tft_output_quantiles)

    latency_ms, peak_memory_mb = measure_latency_and_memory(model, next(iter(test_loader)), device)
    interpretation_signature = json.dumps(interpretation_summaries[0]["top_history_vsn_entries"], sort_keys=True) if interpretation_summaries else ""

    return {
        "profile": profile,
        "seed": seed,
        "mse": float(mse),
        "mae": float(mae),
        "pinball": None if quantile_summary is None else float(quantile_summary["pinball"]),
        "coverage": None if quantile_summary is None else float(quantile_summary["coverage"]),
        "interval_width": None if quantile_summary is None else float(quantile_summary["interval_width"]),
        "latency_ms": float(latency_ms),
        "peak_memory_mb": None if peak_memory_mb is None else float(peak_memory_mb),
        "parameter_count": count_params(model),
        "checkpoint_size_mb": checkpoint_size_mb(model),
        "interpretation_signature": interpretation_signature,
        "tft_profile": run_cfg.tft_profile,
        "tft_config_digest": run_cfg.tft_config_digest,
    }


def summarize_runs(rows):
    by_profile = {}
    for row in rows:
        by_profile.setdefault(row["profile"], []).append(row)

    summary = []
    for profile, items in by_profile.items():
        sigs = [item["interpretation_signature"] for item in items]
        stability = sum(sig == sigs[0] for sig in sigs) / max(len(sigs), 1)
        summary.append({
            "profile": profile,
            "seeds": [item["seed"] for item in items],
            "mean_mse": float(np.mean([item["mse"] for item in items])),
            "mean_mae": float(np.mean([item["mae"] for item in items])),
            "mean_pinball": None if all(item["pinball"] is None for item in items) else float(np.mean([item["pinball"] for item in items if item["pinball"] is not None])),
            "mean_coverage": None if all(item["coverage"] is None for item in items) else float(np.mean([item["coverage"] for item in items if item["coverage"] is not None])),
            "mean_interval_width": None if all(item["interval_width"] is None for item in items) else float(np.mean([item["interval_width"] for item in items if item["interval_width"] is not None])),
            "mean_latency_ms": float(np.mean([item["latency_ms"] for item in items])),
            "peak_memory_mb": items[0]["peak_memory_mb"],
            "parameter_count": items[0]["parameter_count"],
            "checkpoint_size_mb": items[0]["checkpoint_size_mb"],
            "interpretation_stability": float(stability),
            "tft_config_digest": items[0]["tft_config_digest"],
        })
    summary.sort(key=lambda item: item["profile"])
    return summary


def write_markdown(summary, output_path):
    lines = [
        "# TFT Canonical Reference Benchmark",
        "",
        "| Profile | Mean MSE | Mean MAE | Mean Pinball | Mean Coverage | Mean Interval Width | Mean Latency ms | Peak Memory MB | Params | Checkpoint MB | Interpretation Stability | Digest |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in summary:
        def fmt(value):
            if value is None:
                return "n/a"
            if isinstance(value, float):
                return f"{value:.4f}"
            return str(value)
        lines.append(
            f"| {item['profile']} | {fmt(item['mean_mse'])} | {fmt(item['mean_mae'])} | {fmt(item['mean_pinball'])} | "
            f"{fmt(item['mean_coverage'])} | {fmt(item['mean_interval_width'])} | {fmt(item['mean_latency_ms'])} | "
            f"{fmt(item['peak_memory_mb'])} | {item['parameter_count']} | {fmt(item['checkpoint_size_mb'])} | "
            f"{fmt(item['interpretation_stability'])} | {item['tft_config_digest']} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    device = resolve_device(args.device)
    cfg = build_base_config()
    seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]
    if args.quick:
        args.epochs = min(args.epochs, 2)
        args.batch_size = min(args.batch_size, 4)
        args.train_samples = min(args.train_samples, 16)
        args.val_samples = min(args.val_samples, 8)
        args.test_samples = min(args.test_samples, 8)
        seeds = seeds[:2]
        cfg.d_model = 24
        cfg.n_heads = 4
        cfg.e_layers = 1
        cfg.dropout = 0.0
    cfg.train_epochs = args.epochs
    cfg.batch_size = args.batch_size

    set_seed(seeds[0] if seeds else 7)
    train_ds = make_dataset(cfg, args.train_samples, device)
    val_ds = make_dataset(cfg, args.val_samples, device)
    test_ds = make_dataset(cfg, args.test_samples, device)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    loaders = (train_loader, val_loader, test_loader)

    rows = []
    for profile in ("canonical", "extended_safe"):
        for seed in seeds:
            rows.append(train_and_evaluate_profile(cfg, profile, seed, loaders, device))

    summary = summarize_runs(rows)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "tft_profile_reference_summary.json"
    md_path = output_dir / "tft_profile_reference_summary.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, md_path)

    print("Benchmark: TFT canonical vs extended_safe reference")
    print(f"Device: {device} | seeds={seeds} | epochs={cfg.train_epochs} | batch_size={cfg.batch_size}")
    for item in summary:
        print(
            f"{item['profile']}: mse={item['mean_mse']:.4f}, mae={item['mean_mae']:.4f}, "
            f"pinball={item['mean_pinball'] if item['mean_pinball'] is not None else 'n/a'}, "
            f"latency_ms={item['mean_latency_ms']:.3f}, params={item['parameter_count']}, "
            f"stability={item['interpretation_stability']:.3f}"
        )
    print(f"Saved JSON: {json_path}")
    print(f"Saved Markdown: {md_path}")


if __name__ == "__main__":
    main()
