import torch
import pytest
from models.modular_autoformer import ModularAutoformer as Autoformer
from models.BayesianEnhancedAutoformer import Model as BayesianEnhancedAutoformer
from models.HierarchicalEnhancedAutoformer import Model as HierarchicalEnhancedAutoformer
from models.HybridAutoformer import Model as HybridAutoformer
from utils.tools import dotdict

@pytest.fixture
def setup_data():
    configs = dotdict()
    configs.task_name = 'long_term_forecast'
    configs.seq_len = 96
    configs.label_len = 48
    configs.pred_len = 24
    configs.enc_in = 7
    configs.dec_in = 7
    configs.c_out = 7
    configs.d_model = 512
    configs.embed = 'timeF'
    configs.freq = 'h'
    configs.dropout = 0.05
    configs.e_layers = 2
    configs.d_layers = 1
    configs.n_heads = 8
    configs.d_ff = 2048
    configs.moving_avg = 25
    configs.activation = 'gelu'
    configs.norm_type = 'LayerNorm'
    configs.factor = 1
    configs.use_amp = False
    configs.output_attention = False
    configs.attention_type = 'auto'

    x_enc = torch.randn(32, configs.seq_len, configs.enc_in)
    x_mark_enc = torch.randn(32, configs.seq_len, 4)
    x_dec = torch.randn(32, configs.label_len + configs.pred_len, configs.dec_in)
    x_mark_dec = torch.randn(32, configs.label_len + configs.pred_len, 4)

    return configs, x_enc, x_mark_enc, x_dec, x_mark_dec

def test_autoformer_forward(setup_data):
    configs, x_enc, x_mark_enc, x_dec, x_mark_dec = setup_data
    model = Autoformer(configs)
    model.eval()
    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    assert output.shape == (32, configs.pred_len, configs.c_out)

def test_bayesian_enhanced_autoformer_forward(setup_data):
    configs, x_enc, x_mark_enc, x_dec, x_mark_dec = setup_data
    model = BayesianEnhancedAutoformer(configs)
    model.eval()
    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    assert output.shape == (32, configs.pred_len, configs.c_out)

def test_hierarchical_enhanced_autoformer_forward(setup_data):
    configs, x_enc, x_mark_enc, x_dec, x_mark_dec = setup_data
    model = HierarchicalEnhancedAutoformer(configs)
    model.eval()
    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    assert output.shape == (32, configs.pred_len, configs.c_out)

def test_hybrid_autoformer_forward(setup_data):
    configs, x_enc, x_mark_enc, x_dec, x_mark_dec = setup_data
    model = HybridAutoformer(configs)
    model.eval()
    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    assert isinstance(output, dict) or output.shape == (32, configs.pred_len, configs.c_out)