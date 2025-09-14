import torch
import pytest
from layers.modular.attention.temporal_attention import TemporalAttention

@pytest.fixture
def temporal_attention_model():
    return TemporalAttention(d_model=64, n_heads=4)

def test_temporal_attention_forward(temporal_attention_model):
    current_target_state = torch.randn(32, 1, 64)
    historical_event_messages = torch.randn(32, 10, 64)
    output, attn_weights = temporal_attention_model(current_target_state, historical_event_messages)
    assert output.shape == (32, 1, 64)
    assert attn_weights.shape == (32, 1, 10)