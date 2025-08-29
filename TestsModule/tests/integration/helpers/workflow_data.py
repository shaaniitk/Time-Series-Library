"""Shared utilities for workflow integration tests.

Provides synthetic data generation consistent across task variants.
"""
from __future__ import annotations

from types import SimpleNamespace
import torch
import numpy as np
from typing import Tuple

__all__ = ["make_base_config", "generate_synthetic_data"]

def make_base_config() -> SimpleNamespace:
    """Return a base configuration namespace for Autoformer-style models."""
    cfg = SimpleNamespace()
    cfg.task_name = "long_term_forecast"
    cfg.seq_len = 96
    cfg.label_len = 48
    cfg.pred_len = 24
    cfg.enc_in = 7
    cfg.dec_in = 7
    cfg.c_out = 7
    cfg.d_model = 64
    cfg.n_heads = 8
    cfg.e_layers = 2
    cfg.d_layers = 1
    cfg.d_ff = 256
    cfg.moving_avg = 25
    cfg.factor = 1
    cfg.dropout = 0.1
    cfg.activation = "gelu"
    cfg.embed = "timeF"
    cfg.freq = "h"
    cfg.norm_type = "LayerNorm"
    return cfg

def generate_synthetic_data(cfg: SimpleNamespace, batch_size: int = 8):
    """Generate synthetic encoder/decoder series + marks and target.

    Mirrors logic from original monolith; trimmed batch size for speed.
    """
    seq_len = cfg.seq_len
    pred_len = cfg.pred_len
    label_len = cfg.label_len
    features = cfg.enc_in

    t_enc = torch.arange(0, seq_len).float() / seq_len
    t_dec = torch.arange(seq_len - label_len, seq_len + pred_len).float() / seq_len

    def build_stack(t):
        cols = []
        for i in range(features):
            trend = 0.1 * i * t
            seasonal = 0.5 * torch.sin(2 * np.pi * (i + 1) * t)
            noise = torch.randn_like(t) * 0.1
            cols.append(trend + seasonal + noise)
        return torch.stack(cols, dim=1).unsqueeze(0).repeat(batch_size, 1, 1).transpose(1, 2)

    x_enc = build_stack(t_enc)
    x_dec = build_stack(t_dec)
    target = x_dec[:, -pred_len:, :]

    x_mark_enc = torch.randn(batch_size, seq_len, 4)
    x_mark_dec = torch.randn(batch_size, label_len + pred_len, 4)
    return x_enc, x_mark_enc, x_dec, x_mark_dec, target
