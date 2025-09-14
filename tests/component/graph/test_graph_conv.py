import torch
import pytest
from layers.modular.graph.conv import CrossAttentionGNNConv, PGAT_CrossAttn_Layer

@pytest.fixture
def cross_attention_gnn_conv_model():
    return CrossAttentionGNNConv(d_model=64)

@pytest.fixture
def pgat_cross_attn_layer_model():
    return PGAT_CrossAttn_Layer(d_model=64, n_heads=4, d_ff=128)

def test_cross_attention_gnn_conv_forward(cross_attention_gnn_conv_model):
    x = torch.randn(32, 50, 64)
    adj = torch.randn(32, 50, 50)
    output = cross_attention_gnn_conv_model(x, adj)
    assert output.shape == (32, 50, 64)

def test_pgat_cross_attn_layer_forward(pgat_cross_attn_layer_model):
    s_q = torch.randn(32, 10, 64)
    s_k = torch.randn(32, 20, 64)
    output, attn = pgat_cross_attn_layer_model(s_q, s_k)
    assert output.shape == (32, 10, 64)
    assert attn.shape == (32, 4, 10, 20)