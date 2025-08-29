import pytest
import torch
from types import SimpleNamespace
from models.refactored.RefactoredAutoformer import RefactoredAutoformer
from models.refactored.RefactoredBayesianEnhancedAutoformer import RefactoredBayesianEnhancedAutoformer
from models.refactored.RefactoredHierarchicalEnhancedAutoformer import RefactoredHierarchicalEnhancedAutoformer
from models.refactored.RefactoredHybridAutoformer import RefactoredHybridAutoformer

@pytest.fixture
def sample_config():
    return SimpleNamespace(
        enc_in=7, dec_in=7, c_out=7, seq_len=96, label_len=48, pred_len=96,
        d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048,
        dropout=0.05, embed='timeF', freq='h', task_name='long_term_forecast'
    )

def test_refactored_autoformer(sample_config):
    model = RefactoredAutoformer(sample_config)
    x_enc = torch.randn(2, 96, 7)
    x_mark_enc = torch.randn(2, 96, 4)
    x_dec = torch.randn(2, 144, 7)
    x_mark_dec = torch.randn(2, 144, 4)
    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    assert output.shape == (2, 96, 7)

def test_refactored_bayesian(sample_config):
    sample_config.quantiles = [0.1, 0.5, 0.9]
    model = RefactoredBayesianEnhancedAutoformer(sample_config)
    x_enc = torch.randn(2, 96, 7)
    x_mark_enc = torch.randn(2, 96, 4)
    x_dec = torch.randn(2, 144, 7)
    x_mark_dec = torch.randn(2, 144, 4)
    output, _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    assert output.shape == (2, 96, 7 * 3)

def test_refactored_hierarchical(sample_config):
    sample_config.n_levels = 3
    model = RefactoredHierarchicalEnhancedAutoformer(sample_config)
    x_enc = torch.randn(2, 96, 7)
    x_mark_enc = torch.randn(2, 96, 4)
    x_dec = torch.randn(2, 144, 7)
    x_mark_dec = torch.randn(2, 144, 4)
    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    assert output.shape == (2, 96, 7)

def test_refactored_hybrid(sample_config):
    model = RefactoredHybridAutoformer(sample_config)
    x_enc = torch.randn(2, 96, 7)
    x_mark_enc = torch.randn(2, 96, 4)
    x_dec = torch.randn(2, 144, 7)
    x_mark_dec = torch.randn(2, 144, 4)
    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    assert output.shape == (2, 96, 7)