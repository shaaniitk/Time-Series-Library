import pytest
import torch

from models.SOTA_Temporal_PGAT import SOTA_Temporal_PGAT


class DummyDims:
    def __init__(self, d_model):
        self.d_model = d_model


class MinimalConfig:
    def __init__(self, d_model=8, n_heads=2, dropout=0.0, seq_len=2, pred_len=1):
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
        self.enc_in = 7
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