"""
Model Sanity Test Suite
End-to-end sanity tests for all Autoformer models.
"""
import pytest
import torch
from models import Autoformer, EnhancedAutoformer, BayesianEnhancedAutoformer  # Add more as needed

def test_autoformer_sanity():
    batch_size, seq_len, d_model = 2, 96, 8
    x = torch.randn(batch_size, seq_len, d_model)
    model = Autoformer(d_model=d_model)
    out = model.forward(x)
    assert out is not None
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

def test_enhanced_autoformer_sanity():
    batch_size, seq_len, d_model = 2, 96, 8
    x = torch.randn(batch_size, seq_len, d_model)
    model = EnhancedAutoformer(d_model=d_model)
    out = model.forward(x)
    assert out is not None
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

def test_bayesian_autoformer_sanity():
    batch_size, seq_len, d_model = 2, 96, 8
    x = torch.randn(batch_size, seq_len, d_model)
    model = BayesianEnhancedAutoformer(d_model=d_model)
    out = model.forward(x)
    assert out is not None
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
