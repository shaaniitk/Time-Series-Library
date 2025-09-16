import re
import pytest
import torch

from layers.modular.attention.multihead_graph_attention import GraphTransformerLayer


@pytest.fixture
def small_layer():
    torch.manual_seed(0)
    # d_model must match test tensors' last dim
    return GraphTransformerLayer(d_model=8, num_heads=2, dropout=0.0)


def make_valid_inputs(d_model=8):
    torch.manual_seed(0)
    x_dict = {
        'wave': torch.randn(3, d_model),
        'transition': torch.randn(2, d_model),
        'target': torch.randn(4, d_model),
    }
    # wave (3) -> transition (2)
    e1 = torch.tensor([[0, 1, 2], [0, 1, 1]], dtype=torch.long)
    # transition (2) -> target (4)
    e2 = torch.tensor([[0, 1, 1], [0, 2, 3]], dtype=torch.long)
    edge_index_dict = {
        ('wave', 'interacts_with', 'transition'): e1,
        ('transition', 'influences', 'target'): e2,
    }
    return x_dict, edge_index_dict


def test_valid_pass_through(small_layer):
    x_dict, edge_index_dict = make_valid_inputs(d_model=8)
    out = small_layer(x_dict, edge_index_dict)
    assert isinstance(out, dict)
    assert set(out.keys()) == {'wave', 'transition', 'target'}
    # Shapes should be preserved per node type
    assert out['wave'].shape == x_dict['wave'].shape
    assert out['transition'].shape == x_dict['transition'].shape
    assert out['target'].shape == x_dict['target'].shape


def test_missing_node_type_raises(small_layer):
    x_dict, edge_index_dict = make_valid_inputs(d_model=8)
    x_dict.pop('target')
    with pytest.raises(ValueError, match=r"Missing node type 'target'"):
        small_layer(x_dict, edge_index_dict)


def test_feature_wrong_rank_raises(small_layer):
    x_dict, edge_index_dict = make_valid_inputs(d_model=8)
    x_dict['wave'] = torch.randn(3, 8, 1)  # 3D instead of 2D
    with pytest.raises(ValueError, match=r"must be 2D \[num_nodes, d_model\]"):
        small_layer(x_dict, edge_index_dict)


def test_feature_d_model_mismatch_raises(small_layer):
    x_dict, edge_index_dict = make_valid_inputs(d_model=8)
    x_dict['transition'] = torch.randn(2, 9)  # last dim != d_model
    with pytest.raises(ValueError, match=r"last dim must equal d_model=8"):
        small_layer(x_dict, edge_index_dict)


def test_edge_wrong_dtype_raises(small_layer):
    x_dict, edge_index_dict = make_valid_inputs(d_model=8)
    edge_index_dict[('wave', 'interacts_with', 'transition')] = edge_index_dict[('wave', 'interacts_with', 'transition')].to(torch.float32)
    with pytest.raises(ValueError, match=r"must be of dtype torch\.long"):
        small_layer(x_dict, edge_index_dict)


def test_edge_wrong_shape_raises(small_layer):
    x_dict, edge_index_dict = make_valid_inputs(d_model=8)
    bad = torch.tensor([[0, 1, 2], [0, 1, 1]]).t().contiguous()  # [E, 2] instead of [2, E]
    edge_index_dict[('wave', 'interacts_with', 'transition')] = bad
    with pytest.raises(ValueError, match=r"must have shape \[2, E\]"):
        small_layer(x_dict, edge_index_dict)


def test_edge_negative_indices_raises(small_layer):
    x_dict, edge_index_dict = make_valid_inputs(d_model=8)
    bad = torch.tensor([[-1, 0], [0, 0]], dtype=torch.long)
    edge_index_dict[('wave', 'interacts_with', 'transition')] = bad
    with pytest.raises(ValueError, match=r"contains negative indices"):
        small_layer(x_dict, edge_index_dict)


def test_edge_out_of_bounds_raises(small_layer):
    x_dict, edge_index_dict = make_valid_inputs(d_model=8)
    # wave has 3 nodes, put 3 which is OOB
    bad = torch.tensor([[3, 1], [0, 1]], dtype=torch.long)
    edge_index_dict[('wave', 'interacts_with', 'transition')] = bad
    with pytest.raises(ValueError, match=r"source index out of bounds"):
        small_layer(x_dict, edge_index_dict)