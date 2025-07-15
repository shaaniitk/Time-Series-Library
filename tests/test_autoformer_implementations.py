import torch
import pytest
from models.Autoformer import Model as Autoformer
from models.EnhancedAutoformer import EnhancedAutoformer, LearnableSeriesDecomp
from layers.Attention import get_attention_layer
from utils.tools import dotdict

# Fixture for generating common configs and data
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
    configs.attention_type = 'auto' # Default to auto

    x_enc = torch.randn(32, configs.seq_len, configs.enc_in)
    x_mark_enc = torch.randn(32, configs.seq_len, 4)
    x_dec = torch.randn(32, configs.label_len + configs.pred_len, configs.dec_in)
    x_mark_dec = torch.randn(32, configs.label_len + configs.pred_len, 4)

    return configs, x_enc, x_mark_enc, x_dec, x_mark_dec

# --- Autoformer Tests ---

def test_autoformer_forward_pass(setup_data):
    configs, x_enc, x_mark_enc, x_dec, x_mark_dec = setup_data
    model = Autoformer(configs)
    model.eval()

    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    assert output.shape == (32, configs.pred_len, configs.c_out)

def test_autoformer_decomposition(setup_data):
    configs, x_enc, _, _, _ = setup_data
    # Need to access decomp from the model instance
    from layers.Autoformer_EncDec import series_decomp
    decomp_module = series_decomp(configs.moving_avg)
    seasonal, trend = decomp_module(x_enc)
    assert seasonal.shape == x_enc.shape
    assert trend.shape == x_enc.shape
    assert torch.allclose(seasonal + trend, x_enc, atol=1e-6)

# --- EnhancedAutoformer Tests ---

def test_get_attention_layer(setup_data):
    configs, _, _, _, _ = setup_data
    configs.attention_type = 'full'
    attention_layer = get_attention_layer(configs)
    from layers.Attention import FullAttention
    assert isinstance(attention_layer, FullAttention)

def test_enhanced_autoformer_forward_pass(setup_data):
    configs, x_enc, x_mark_enc, x_dec, x_mark_dec = setup_data
    model = EnhancedAutoformer(configs)
    model.eval()

    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    assert output.shape == (32, configs.pred_len, configs.c_out)

def test_learnable_series_decomp(setup_data):
    configs, x_enc, _, _, _ = setup_data
    decomp_module = LearnableSeriesDecomp(configs.enc_in)
    seasonal, trend = decomp_module(x_enc)
    assert seasonal.shape == x_enc.shape
    assert trend.shape == x_enc.shape
    assert torch.allclose(seasonal + trend, x_enc, atol=1e-6)

    # Test learnability
    optimizer = torch.optim.SGD(decomp_module.parameters(), lr=0.01)
    loss = (seasonal + trend).mean()
    loss.backward()
    for param in decomp_module.parameters():
        assert param.grad is not None

def test_enhanced_autoformer_imputation(setup_data):
    configs, x_enc, x_mark_enc, x_dec, x_mark_dec = setup_data
    configs.task_name = 'imputation'
    model = EnhancedAutoformer(configs)
    model.eval()

    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    assert output.shape == (32, configs.seq_len, configs.c_out)

def test_enhanced_autoformer_anomaly_detection(setup_data):
    configs, x_enc, x_mark_enc, x_dec, x_mark_dec = setup_data
    configs.task_name = 'anomaly_detection'
    model = EnhancedAutoformer(configs)
    model.eval()

    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    assert output.shape == (32, configs.seq_len, configs.c_out)

def test_enhanced_autoformer_quantile_mode(setup_data):
    configs, x_enc, x_mark_enc, x_dec, x_mark_dec = setup_data
    configs.quantile_levels = [0.1, 0.5, 0.9]
    configs.c_out = configs.c_out * len(configs.quantile_levels)
    model = EnhancedAutoformer(configs)
    model.eval()

    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    assert output.shape == (32, configs.pred_len, configs.c_out)