import torch
import pytest
from layers.modular.decoder.custom_decoders import StandardDecoder, ProbabilisticDecoder

@pytest.fixture
def standard_decoder_model():
    return StandardDecoder(d_model=64)

@pytest.fixture
def probabilistic_decoder_model():
    return ProbabilisticDecoder(d_model=64)

def test_standard_decoder_forward(standard_decoder_model):
    input_tensor = torch.randn(32, 10, 64)
    output = standard_decoder_model(input_tensor)
    assert output.shape == (32, 10)

def test_probabilistic_decoder_forward(probabilistic_decoder_model):
    input_tensor = torch.randn(32, 10, 64)
    mean, std_dev = probabilistic_decoder_model(input_tensor)
    assert mean.shape == (32, 10)
    assert std_dev.shape == (32, 10)
    assert torch.all(std_dev > 0)