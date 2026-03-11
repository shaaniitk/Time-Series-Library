import torch
import torch.nn as nn
from types import SimpleNamespace
import sys
import os
import unittest
from pathlib import Path

# Ensure root path is accessible so `from models.TFT_Nixtla import Model` resolves gracefully
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from models.TFT_Nixtla import Model

def create_dummy_batch(batch_size, seq_len, label_len, pred_len, c_out):
    """Generates standard Dummy TSL batch configurations"""
    x_enc = torch.randn(batch_size, seq_len, c_out)
    x_mark_enc = torch.randn(batch_size, seq_len, 4) # 4 temporal covariates (hour, day, month, year)
    
    x_dec = torch.randn(batch_size, label_len + pred_len, c_out)
    x_mark_dec = torch.randn(batch_size, label_len + pred_len, 4)
    
    y = torch.randn(batch_size, pred_len, c_out)
    
    return x_enc, x_mark_enc, x_dec, x_mark_dec, y

def apply_test(C):
    print(f"\n--- Testing TFT_Nixtla Forward & Gradients [C={C}] ---")
    
    configs = SimpleNamespace(
        task_name='long_term_forecast',
        freq='h',
        seq_len=30,
        label_len=16,
        pred_len=4,
        enc_in=C,
        dec_in=C,
        c_out=C,
        d_model=64,
        n_heads=4,
        dropout=0.1,
        batch_size=8
    )
    
    model = Model(configs)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    x_enc, x_mark_enc, x_dec, x_mark_dec, y = create_dummy_batch(
        configs.batch_size, configs.seq_len, configs.label_len, configs.pred_len, configs.c_out
    )
    
    # Check proper initialization of Cross-Channel Mixer
    if C > 1:
        assert hasattr(model, 'cross_channel_mixer'), "Cross channel mixer missing on Multivariate config"
        assert model.cross_channel_mixer.weight.shape == (C, C)
    else:
        assert not hasattr(model, 'cross_channel_mixer'), "Cross channel mixer should not instantiate on Univariate"
    
    # Check Forward Shape
    optimizer.zero_grad()
    y_hat = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    
    assert y_hat.shape == y.shape, f"Forward shape mismatch: {y_hat.shape} != {y.shape}"
    print(f"Forward Shape OK: {y_hat.shape}")
    
    # Check Gradients inside Mixer specifically
    loss = criterion(y_hat, y)
    loss.backward()
    
    if C > 1:
        assert model.cross_channel_mixer.weight.grad is not None, "Cross channel mixer did not receive gradients!"
        assert torch.sum(torch.abs(model.cross_channel_mixer.weight.grad)) > 0, "Cross channel mixer gradients are zero!"
    
    # Optimization step
    optimizer.step()
    
    loss_new = criterion(model(x_enc, x_mark_enc, x_dec, x_mark_dec), y)
    print(f"Loss optimized: {loss.item():.4f} -> {loss_new.item():.4f}")
    
    # Test fusion_alpha logging presence inside dual attention proxy
    try:
        # traverse to the dual attention module
        dual_attn = model.nixtla_tft.temporal_fusion_decoder.attention
        alpha_val = torch.sigmoid(dual_attn.attention_fusion_logit).item()
        print(f"Fusion Alpha properly exposed: {alpha_val:.4f}")
    except Exception as e:
        print(f"Error accessing fusion alpha: {e}")
        raise e

class TestNixtlaTFTCrossChannel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        
    def test_univariate_c1(self):
        apply_test(C=1)
        
    def test_multivariate_c4(self):
        apply_test(C=4)
        
    def test_multivariate_c10(self):
        apply_test(C=10)

if __name__ == "__main__":
    unittest.main()
