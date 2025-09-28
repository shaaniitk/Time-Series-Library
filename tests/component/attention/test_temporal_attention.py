import torch
import pytest
from layers.modular.attention.temporal_attention import TemporalAttention


@pytest.fixture
def temporal_attention_model() -> TemporalAttention:
    """Create a temporal attention module mirroring default configuration."""
    return TemporalAttention(d_model=64, n_heads=4)


def test_temporal_attention_forward(temporal_attention_model: TemporalAttention) -> None:
    """Temporal attention should return normalized context and averaged weights."""
    current_target_state = torch.randn(32, 1, 64)
    historical_event_messages = torch.randn(32, 10, 64)
    output, attn_weights = temporal_attention_model(current_target_state, historical_event_messages)
    assert output.shape == (32, 1, 64)
    assert attn_weights.shape == (32, 1, 10)


def test_temporal_attention_can_skip_weights(temporal_attention_model: TemporalAttention) -> None:
    """Temporal attention should optionally omit attention weights when disabled."""
    current_target_state = torch.randn(6, 1, 64)
    historical_event_messages = torch.randn(6, 5, 64)
    output = temporal_attention_model(
        current_target_state,
        historical_event_messages,
        return_attention=False,
    )
    assert isinstance(output, torch.Tensor)
    assert output.shape == (6, 1, 64)


def test_temporal_attention_rejects_mismatched_embedding(temporal_attention_model: TemporalAttention) -> None:
    """Temporal attention should raise when embeddings do not align with configuration."""
    current_target_state = torch.randn(8, 1, 32)
    with pytest.raises(ValueError, match="configured with d_model=64"):
        temporal_attention_model(current_target_state)


def test_temporal_attention_rejects_batch_mismatch(temporal_attention_model: TemporalAttention) -> None:
    """Temporal attention should detect mismatched batch dimensions across tensors."""
    current_target_state = torch.randn(4, 1, 64)
    historical_event_messages = torch.randn(2, 10, 64)
    with pytest.raises(ValueError, match="identical batch dimension"):
        temporal_attention_model(current_target_state, historical_event_messages)
