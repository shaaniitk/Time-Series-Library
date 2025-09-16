import os
import sys
import io
import types
from contextlib import redirect_stdout

# Ensure project root is on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

import torch

# Import modules after setting path
from models.SOTA_Temporal_PGAT import SOTA_Temporal_PGAT, EnhancedPGAT_CrossAttn_Layer, AutoCorrTemporalAttention
from layers.modular.decoder.mixture_density_decoder import MixtureDensityDecoder
import utils.graph_utils as graph_utils


def fake_get_pyg_graph(config, device):
    """Lightweight stand-in for utils.graph_utils.get_pyg_graph that avoids torch_geometric dependency"""
    class Node:
        def __init__(self, num_nodes):
            self.num_nodes = num_nodes
            # topology features should match d_model for projections in conv layers
            self.t = torch.zeros(num_nodes, getattr(config, 'd_model', 64), device=device)

    class Edge:
        def __init__(self, edge_index):
            self.edge_index = edge_index

    class Data:
        def __init__(self):
            num_waves = getattr(config, 'num_waves', 4)
            num_targets = getattr(config, 'num_targets', 4)
            num_transitions = getattr(config, 'num_transitions', 4)
            self._nodes = {
                'wave': Node(num_waves),
                'target': Node(num_targets),
                'transition': Node(num_transitions),
            }
            self._edges = {}
            # wave -> transition (fully connected bipartite)
            wave_idx = torch.arange(num_waves, device=device).repeat_interleave(num_transitions)
            trans_idx_in = torch.arange(num_transitions, device=device).repeat(num_waves)
            # Our MessagePassing expects edge_index[0] = target (i), edge_index[1] = source (j)
            self._edges[('wave', 'interacts_with', 'transition')] = Edge(torch.stack([trans_idx_in, wave_idx]))
            # transition -> target (fully connected bipartite)
            trans_idx_out = torch.arange(num_transitions, device=device).repeat_interleave(num_targets)
            target_idx = torch.arange(num_targets, device=device).repeat(num_transitions)
            self._edges[('transition', 'influences', 'target')] = Edge(torch.stack([target_idx, trans_idx_out]))
        def __getitem__(self, key):
            if isinstance(key, tuple):
                return self._edges[key]
            return self._nodes[key]
    return Data()


def mk_config(**overrides):
    class C:
        pass
    c = C()
    # Defaults aligned with model expectations
    c.d_model = 64
    c.n_heads = 4
    c.dropout = 0.1
    c.seq_len = 8
    c.pred_len = 4
    c.features = 'M'  # multivariate -> 7 features default in model fallback
    # Gates default enabled
    c.enable_dynamic_graph = True
    # Reduce optional modules to avoid noisy skips during smoke
    c.enable_structural_pos_encoding = True
    c.enable_graph_positional_encoding = True
    c.enable_graph_attention = True
    c.use_autocorr_attention = True
    c.use_mixture_density = False
    c.mdn_components = 2
    c.use_adaptive_temporal = True
    for k, v in overrides.items():
        setattr(c, k, v)
    return c


def summarize_runtime(model, stdout_text):
    summary = {}
    # Enhanced PGAT spatial encoder
    summary['enhanced_pgat_enabled_flag'] = getattr(model, 'enhanced_pgat_enabled', False)
    summary['enhanced_pgat_type_used'] = isinstance(getattr(model, 'spatial_encoder', None), EnhancedPGAT_CrossAttn_Layer)

    # Dynamic graph
    summary['dynamic_graph_enabled_cfg'] = getattr(model.config, 'enable_dynamic_graph', False)
    summary['dynamic_graph_modules_inited'] = (model.dynamic_graph is not None and model.adaptive_graph is not None)

    # Structural positional encoding
    summary['struct_pos_encoding_cfg'] = getattr(model.config, 'enable_structural_pos_encoding', False)
    summary['struct_pos_encoding_skipped'] = ('Structural positional encoding skipped' in stdout_text)

    # Graph positional encoding
    summary['graph_pos_encoding_cfg'] = getattr(model.config, 'enable_graph_positional_encoding', False)
    summary['graph_pos_encoding_skipped'] = ('Graph positional encoding skipped' in stdout_text)

    # Graph attention
    summary['graph_attention_cfg'] = getattr(model.config, 'enable_graph_attention', False)
    summary['graph_attention_skipped'] = ('Graph attention skipped' in stdout_text)

    # Temporal encoder autocorr
    summary['autocorr_temporal_used'] = isinstance(getattr(model, 'temporal_encoder', None), AutoCorrTemporalAttention)

    # Mixture density decoder
    summary['mixture_density_used'] = isinstance(getattr(model, 'decoder', None), MixtureDensityDecoder)

    # Enhanced PGAT fallback detection
    summary['enhanced_pgat_fallback_triggered'] = ('Enhanced PGAT unavailable' in stdout_text)

    return summary


def run_one_pass(force_pgat_ctor_error=False):
    # Monkeypatch get_pyg_graph to avoid external deps
    graph_utils.get_pyg_graph = fake_get_pyg_graph

    # Optionally force Enhanced PGAT constructor to raise to verify fallback
    failing_class = None
    if force_pgat_ctor_error:
        class FailingEnhancedPGAT:
            def __init__(self, *args, **kwargs):
                raise RuntimeError('forced ctor failure for fallback smoke test')
        failing_class = FailingEnhancedPGAT

    # Build config and model
    cfg = mk_config()
    # Override the class symbol in the imported model module if requested
    if failing_class is not None:
        import models.SOTA_Temporal_PGAT as PGATModule
        PGATModule.EnhancedPGAT_CrossAttn_Layer = failing_class

    model = SOTA_Temporal_PGAT(cfg, mode='standard')
    model.set_graph_info_storage(True)

    # Create random inputs
    B, F = 2, 7  # 7 features aligns with features='M' default fallback
    wave_len, pred_len = cfg.seq_len, cfg.pred_len
    device = torch.device('cpu')
    wave = torch.randn(B, wave_len, F, device=device)
    target = torch.randn(B, pred_len, F, device=device)

    # Run forward and capture stdout
    f = io.StringIO()
    with redirect_stdout(f):
        out = model(wave, target, graph=None)
    logs = f.getvalue()

    # Summarize
    summary = summarize_runtime(model, logs)

    return out, logs, summary


def main():
    # 1) Normal pass (no forced errors)
    out1, logs1, summary1 = run_one_pass(force_pgat_ctor_error=False)
    # 2) Forced fallback pass (force Enhanced PGAT ctor error)
    out2, logs2, summary2 = run_one_pass(force_pgat_ctor_error=True)

    print('=== Smoke Forward Pass: Normal ===')
    print('Runtime summary:', summary1)
    print('Output shape (normal):', tuple(out1.shape) if hasattr(out1, 'shape') else type(out1))
    if logs1.strip():
        print('Internal logs (normal):')
        print(logs1.strip())

    print('\n=== Smoke Forward Pass: Forced Fallback (Enhanced PGAT ctor error) ===')
    print('Runtime summary:', summary2)
    print('Output shape (forced):', tuple(out2.shape) if hasattr(out2, 'shape') else type(out2))
    if logs2.strip():
        print('Internal logs (forced):')
        print(logs2.strip())


if __name__ == '__main__':
    main()