import torch
from unified_component_registry import unified_registry
from layers.modular.core.config_schemas import EmbeddingConfig

# List
comps = unified_registry.list_all_components()
print('embeddings:', comps.get('embedding', []))

# Temporal embedding
Temporal = unified_registry.get_component('embedding', 'temporal_embedding')
tcfg = EmbeddingConfig(d_model=16, dropout=0.0)
temporal = Temporal(tcfg)
base = torch.randn(2, 5, 16)
emb = temporal(base)
print('temporal_shape:', tuple(emb.shape))

# Value embedding
Value = unified_registry.get_component('embedding', 'value_embedding')
vcfg = EmbeddingConfig(d_model=16, dropout=0.0)
value = Value(vcfg)
vals = torch.randn(2, 5, 1)
vemb = value(vals)
print('value_shape:', tuple(vemb.shape))
