import argparse
import json
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.TemporalFusionTransformer import Model
from scripts.long_term_forecast.tft_ablation_full_attention_vs_vsn_bypass import build_base_config, resolve_device, set_seed
from utils.tft_synthetic import make_multiscale_tft_tensors


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark TFT exact vs SDPA attention backend.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def _make_inputs(cfg, batch_size, device):
    tensors = make_multiscale_tft_tensors(
        seq_len=cfg.seq_len,
        label_len=cfg.label_len,
        pred_len=cfg.pred_len,
        enc_in=cfg.enc_in,
        c_out=cfg.c_out,
        known_len=cfg.tft_known_len,
        n_samples=batch_size,
        noise_std=0.0,
        device=device,
    )
    return tensors["x_enc"], tensors["x_mark_enc"], tensors["x_dec"], tensors["x_mark_dec"]


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_backend(cfg, backend, x_enc, x_mark_enc, x_dec, x_mark_dec, warmup, steps, device):
    run_cfg = argparse.Namespace(**vars(cfg))
    run_cfg.tft_attention_backend = backend
    model = Model(run_cfg).to(device)
    model.eval()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        _sync(device)
        start = time.perf_counter()
        for _ in range(steps):
            _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        _sync(device)
        elapsed = time.perf_counter() - start

        payload = model(x_enc[:1], x_mark_enc[:1], x_dec[:1], x_mark_dec[:1], return_interpretation=True)

    result = {
        "backend": backend,
        "avg_step_ms": 1000.0 * elapsed / max(steps, 1),
        "attention_backend_config": payload.get("attention_backend_config"),
        "attention_backend_used": payload.get("attention_backend_used"),
        "cross_attention_backend_used": payload.get("cross_attention_backend_used"),
    }
    if device.type == "cuda":
        result["peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        result["peak_memory_mb"] = None
    return result


def main():
    args = parse_args()
    device = resolve_device(args.device)
    cfg = build_base_config()
    cfg.tft_full_attention = True
    cfg.tft_dual_attention_fusion = True
    cfg.tft_use_explicit_cross_attention = True
    cfg.tft_cross_attention_type = "full"
    cfg.tft_attention_position_bias = "rope"
    cfg.tft_use_revin = True
    cfg.tft_temporal_backbone = "hybrid_tcn_lstm"
    cfg.tft_temporal_backbone_layers = 2
    cfg.tft_temporal_kernel_size = 3
    cfg.tft_temporal_hidden_size = cfg.d_model

    warmup = args.warmup
    steps = args.steps
    batch_size = args.batch_size
    if args.quick:
        cfg.d_model = 32
        cfg.n_heads = 4
        warmup = min(warmup, 1)
        steps = min(steps, 3)
        batch_size = min(batch_size, 2)

    set_seed(42)
    x_enc, x_mark_enc, x_dec, x_mark_dec = _make_inputs(cfg, batch_size, device)

    results = [
        benchmark_backend(cfg, backend, x_enc, x_mark_enc, x_dec, x_mark_dec, warmup, steps, device)
        for backend in ("exact", "sdpa")
    ]
    results = sorted(results, key=lambda item: item["avg_step_ms"])

    print("Benchmark: TFT exact vs SDPA attention backend")
    print(f"Device: {device} | warmup={warmup} | steps={steps} | batch_size={batch_size}")
    print("-" * 100)
    print(f"{'backend':<10}{'avg_step_ms':<16}{'peak_memory_mb':<18}{'interp_backend':<18}{'cross_interp_backend'}")
    for item in results:
        peak_memory = "n/a" if item["peak_memory_mb"] is None else f"{item['peak_memory_mb']:.2f}"
        print(
            f"{item['backend']:<10}{item['avg_step_ms']:<16.3f}{peak_memory:<18}{str(item['attention_backend_used']):<18}{item['cross_attention_backend_used']}"
        )
    print("Best backend by avg_step_ms:", results[0]["backend"])
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()