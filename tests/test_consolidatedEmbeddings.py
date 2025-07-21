"""
Test for unified Embedding module and registry
"""
import pytest
import torch
from utils.modular_components.implementations.embeddings import EMBEDDING_REGISTRY, get_embedding_method
from utils.modular_components.config_schemas import EmbeddingConfig

def test_registry_keys():
    expected = {'temporal', 'value', 'covariate', 'hybrid'}
    assert set(EMBEDDING_REGISTRY.keys()) == expected

def test_temporal_embedding_instantiation():
    config = EmbeddingConfig(d_model=8, dropout=0.1)
    emb = get_embedding_method('temporal', config=config)
    assert emb is not None
    x = torch.randn(2, 5, 8)
    temporal_features = {
        'hour': torch.randint(0, 24, (2, 5)),
        'day': torch.randint(1, 32, (2, 5)),
        'weekday': torch.randint(0, 7, (2, 5)),
        'month': torch.randint(1, 13, (2, 5)),
    }
    out = emb(x, temporal_features=temporal_features)
    assert out.shape == x.shape

def test_value_embedding_instantiation():
    config = EmbeddingConfig(d_model=8, dropout=0.1)
    setattr(config, 'num_features', 1)
    emb = get_embedding_method('value', config=config)
    assert emb is not None
    values = torch.randn(2, 5, 1)
    out = emb(values)
    assert out.shape == (2, 5, 8)

def test_covariate_embedding_instantiation():
    config = EmbeddingConfig(d_model=8, dropout=0.1)
    setattr(config, 'categorical_features', {'cat1': 10})
    setattr(config, 'numerical_features', 2)
    emb = get_embedding_method('covariate', config=config)
    assert emb is not None
    categorical_data = {'cat1': torch.randint(0, 10, (2, 5))}
    numerical_data = torch.randn(2, 5, 2)
    out = emb(categorical_data=categorical_data, numerical_data=numerical_data)
    assert out.shape == (2, 5, 8)

def test_hybrid_embedding_instantiation():
    config = EmbeddingConfig(d_model=8, dropout=0.1)
    setattr(config, 'use_temporal', True)
    setattr(config, 'use_value', True)
    setattr(config, 'use_covariate', True)
    setattr(config, 'temporal_config', {})
    setattr(config, 'value_config', {})
    setattr(config, 'covariate_config', {'categorical_features': {'cat1': 10}, 'numerical_features': 2})
    emb = get_embedding_method('hybrid', config=config)
    assert emb is not None
    values = torch.randn(2, 5, 1)
    temporal_features = {
        'hour': torch.randint(0, 24, (2, 5)),
        'day': torch.randint(1, 32, (2, 5)),
        'weekday': torch.randint(0, 7, (2, 5)),
        'month': torch.randint(1, 13, (2, 5)),
    }
    categorical_data = {'cat1': torch.randint(0, 10, (2, 5))}
    numerical_data = torch.randn(2, 5, 2)
    out = emb(values=values, temporal_features=temporal_features,
              categorical_data=categorical_data, numerical_data=numerical_data)
    assert out.shape == (2, 5, 8)
