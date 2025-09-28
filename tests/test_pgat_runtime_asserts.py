from typing import Tuple

import pytest
import torch

from models.SOTA_Temporal_PGAT import SOTA_Temporal_PGAT


class DummyDims:
    def __init__(self, d_model):
        self.d_model = d_model


class MinimalConfig:
    def __init__(self, d_model=8, n_heads=2, dropout=0.0, seq_len=2, pred_len=1, enc_in=7):
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.seq_len = seq_len
        self.pred_len = pred_len
        # keep forward lightweight
        self.use_dynamic_edge_weights = False
        self.use_autocorr_attention = False
        self.enable_dynamic_graph = False
        self.enable_graph_positional_encoding = False
        self.use_mixture_density = False
        # misc defaults referenced in model
        self.features = 'M'
        self.enc_in = enc_in
        self.mdn_components = 1


def minimal_model(d_model=8):
    # Build a minimally-initialized model with a lightweight config.
    cfg = MinimalConfig(d_model=d_model, n_heads=2, dropout=0.0, seq_len=2, pred_len=1)
    return SOTA_Temporal_PGAT(cfg)


def minimal_windows(batch=1, seq_len=2, pred_len=1, d_model=8):
    torch.manual_seed(0)
    wave_window = torch.randn(batch, seq_len, d_model)
    target_window = torch.randn(batch, pred_len, d_model)
    return wave_window, target_window


def fallback_windows(batch: int = 2, seq_len: int = 2, pred_len: int = 1, feature_dim: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic windows whose feature dim differs from d_model to trigger fallback."""
    torch.manual_seed(1)
    wave_window = torch.randn(batch, seq_len, feature_dim)
    target_window = torch.randn(batch, pred_len, feature_dim)
    return wave_window, target_window


def make_enhanced_dicts(d_model=8):
    torch.manual_seed(0)
    enhanced_x_dict = {
        'wave': torch.randn(3, d_model),
        'transition': torch.randn(2, d_model),
        'target': torch.randn(4, d_model),
    }
    e1 = torch.tensor([[0, 1, 2], [0, 1, 1]], dtype=torch.long)
    e2 = torch.tensor([[0, 1, 1], [0, 2, 3]], dtype=torch.long)
    enhanced_edge_index_dict = {
        ('wave', 'interacts_with', 'transition'): e1,
        ('transition', 'influences', 'target'): e2,
    }
    return enhanced_x_dict, enhanced_edge_index_dict


# Unskipped: now uses a minimal fixture and forces a validation error via bad edge indices
def test_pgat_validations_trigger_on_bad_inputs():
    model = minimal_model(d_model=8)
    # Monkeypatch _create_edge_index to produce invalid (negative) indices to trigger validation
    def bad_edge_index(num_source, num_target, device):
        return torch.tensor([[0, -1], [0, 0]], dtype=torch.long, device=device)
    model._create_edge_index = bad_edge_index  # type: ignore[attr-defined]

    wave_window, target_window = minimal_windows(d_model=8)

    with pytest.raises(ValueError, match=r"contains negative indices"):
        model.forward(wave_window, target_window, graph=None)


def test_pgat_fallback_embedding_path() -> None:
    """Ensure PGAT uses tensor-compatible fallback embedding when registry entry is absent."""
    cfg = MinimalConfig(d_model=8, n_heads=2, dropout=0.0, seq_len=2, pred_len=1, enc_in=3)
    model = SOTA_Temporal_PGAT(cfg)

    assert model._embedding_source in {'linear_fallback', 'lazy_linear_fallback'}

    wave_window, target_window = fallback_windows(
        batch=2,
        seq_len=cfg.seq_len,
        pred_len=cfg.pred_len,
        feature_dim=cfg.enc_in,
    )
    output = model(wave_window, target_window, graph=None)
    output_tensor = output[0] if isinstance(output, tuple) else output

    assert output_tensor.shape[0] == 2
    assert output_tensor.shape[1] == cfg.pred_len
    assert torch.isfinite(output_tensor).all()


def test_pgat_rejects_mismatched_batch_sizes() -> None:
    """Forward should raise when wave and target windows disagree on batch size."""
    model = minimal_model(d_model=8)
    wave_window, target_window = minimal_windows(batch=2, d_model=8)
    with pytest.raises(ValueError, match="Batch size mismatch"):
        model.forward(wave_window, target_window[:1], graph=None)


def test_pgat_rejects_dtype_mismatch() -> None:
    """Forward should enforce consistent dtypes across wave and target inputs."""
    model = minimal_model(d_model=8)
    wave_window, target_window = minimal_windows(d_model=8)
    with pytest.raises(ValueError, match="same dtype"):
        model.forward(wave_window, target_window.to(torch.float64), graph=None)


def test_pgat_config_requires_divisible_heads() -> None:
    """Initialization should fail when d_model is not divisible by n_heads."""
    cfg = MinimalConfig(d_model=10, n_heads=3, dropout=0.0, seq_len=2, pred_len=1)
    with pytest.raises(ValueError, match="divisible"):
        SOTA_Temporal_PGAT(cfg)
