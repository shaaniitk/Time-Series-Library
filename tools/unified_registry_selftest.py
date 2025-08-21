"""Lightweight self-test to instantiate each registered component with dummy args.
Run: python -m tools.unified_registry_selftest
"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from layers.modular.core import unified_registry, ComponentFamily  # type: ignore
import importlib  # noqa: F401
import traceback  # noqa: F401

# Ensure registrations loaded before listing
import layers.modular.core.register_components  # noqa: F401
print('DEBUG: Families discovered (post-registration):', unified_registry.list())

# Minimal constructor kwargs heuristics
class _DummyAttention:
    output_dim_multiplier = 1
    def __call__(self, *a, **k):
        pass

class _DummyDecomp:
    def forward(self, x):
        return x, x

dummy_attn = _DummyAttention()
dummy_decomp = _DummyDecomp()

DEFAULTS = {
    # Base defaults; some components override below.
    ComponentFamily.ATTENTION: dict(d_model=32, n_heads=4, dropout=0.0, seq_len=24),
    ComponentFamily.DECOMPOSITION: dict(kernel_size=25, input_dim=32, init_kernel_size=25),
    ComponentFamily.ENCODER: dict(num_encoder_layers=1, d_model=32, n_heads=4, d_ff=64, dropout=0.0, activation='gelu', attention_comp=dummy_attn, decomp_comp=dummy_decomp),
    ComponentFamily.DECODER: dict(d_layers=1, d_model=32, c_out=16, n_heads=4, d_ff=64, dropout=0.0, activation='gelu', self_attention_comp=dummy_attn, cross_attention_comp=dummy_attn, decomp_comp=dummy_decomp),
    ComponentFamily.SAMPLING: dict(),
    ComponentFamily.OUTPUT_HEAD: dict(d_model=32, c_out=16),
    ComponentFamily.LOSS: dict(),
    ComponentFamily.FUSION: dict(d_model=32, n_levels=2),
    ComponentFamily.BACKBONE: dict(d_model=32, dropout=0.0, model_name='amazon/chronos-t5-tiny', pretrained=False),
    ComponentFamily.EMBEDDING: dict(d_model=32, max_len=64, dropout=0.0),
    ComponentFamily.PROCESSOR: dict(d_model=32, pred_len=8),
    ComponentFamily.FEEDFORWARD: dict(d_model=32, d_ff=64, dropout=0.0, activation='relu'),
    ComponentFamily.OUTPUT: dict(d_model=32, output_dim=8),
}

results = {}
for fam, entries in unified_registry.list().items():
    for name in entries:
        family_enum = ComponentFamily(fam)
        kwargs = DEFAULTS.get(family_enum, {}).copy()
        # Family-specific tweaks
        if family_enum == ComponentFamily.ATTENTION:
            # Special-case constructor signatures
            if name in {'autocorrelation','cross_resolution_attention','wavelet_attention','meta_learning_adapter', 'adaptive_autocorrelation_layer', 'hierarchical_autocorrelation', 'bayesian_attention', 'bayesian_multi_head_attention', 'variational_attention', 'bayesian_cross_attention', 'adaptive_wavelet_attention', 'multi_scale_wavelet_attention', 'causal_convolution', 'convolutional_attention', 'temporal_conv_net', 'graph_attention_layer', 'multi_graph_attention', 'adaptive_mixture'}:
                kwargs.pop('seq_len', None)
            if name == 'cross_resolution_attention':
                kwargs.setdefault('n_levels', 2)
            if name == 'fourier_block':
                # FourierBlock expects (in_channels, out_channels, n_heads, seq_len, ...)
                kwargs = dict(in_channels=32, out_channels=32, n_heads=4, seq_len=24, modes=8)
            if name == 'fourier_cross_attention':
                kwargs = dict(in_channels=32, out_channels=32, seq_len_q=24, seq_len_kv=24, modes=8, num_heads=4)
            if name == 'wavelet_decomposition':
                kwargs = dict(input_dim=32, levels=2, kernel_size=3)
        elif family_enum == ComponentFamily.DECOMPOSITION:
            if name in {'moving_avg','series_decomp'}:
                kwargs = {'kernel_size': 25}
            elif name == 'learnable_decomp':
                kwargs = {'input_dim': 32, 'init_kernel_size': 25}
            elif name == 'stable_decomp':
                kwargs = {'kernel_size': 25}
            elif name in {'wavelet_decomp','wavelet_hierarchical_decomp'}:
                kwargs = {'seq_len': 32, 'd_model': 32, 'levels': 2}
            elif name in {'learnable_wavelet_decomp','advanced_wavelet_decomp'}:
                kwargs = {'d_model': 32, 'levels': 2}
        elif family_enum == ComponentFamily.ENCODER:
            if name == 'hierarchical_encoder':
                # minimal synthetic hierarchical_config object
                class _HC:
                    n_levels = 2
                    level_configs = None
                kwargs['hierarchical_config'] = _HC()
            elif name == 'graph_encoder':
                # Replace encoder-specific modular args with graph encoder signature
                kwargs = {
                    'num_layers': 1,
                    'd_model': 32,
                    'n_heads': 4,
                    'dropout': 0.0,
                    'activation': 'gelu',
                    'graph_type': 'correlation',
                }
            elif name in {'hybrid_graph_encoder', 'adaptive_graph_encoder'}:
                kwargs = {
                    'num_layers': 1,
                    'd_model': 32,
                    'n_heads': 4,
                    'dropout': 0.0,
                    'activation': 'gelu',
                    'graph_type': 'correlation',
                }
        elif family_enum == ComponentFamily.DECODER:
            # Provide decomp_comp argument names expected in layers
            kwargs['decomp_comp'] = dummy_decomp
        elif family_enum == ComponentFamily.OUTPUT_HEAD:
            if name == 'quantile':
                kwargs['num_quantiles'] = 3
        elif family_enum == ComponentFamily.LOSS:
            if name == 'ps_loss':
                kwargs = {'pred_len': 24}
            elif name in {'quantile_loss','pinball','quantile','multi_quantile'}:
                kwargs = {'quantiles': [0.1,0.5,0.9]}
        try:
            _ = unified_registry.create(family_enum, name, **kwargs)
            results[f"{fam}:{name}"] = 'OK'
        except Exception as e:
            results[f"{fam}:{name}"] = f'FAIL: {e.__class__.__name__}: {e}'

print('Self-test results:')
failures = 0
for k, v in sorted(results.items()):
    print(f'  {k}: {v}')
    if v.startswith('FAIL'): failures += 1
print(f"Summary: {len(results)} attempted, {failures} failures")
if failures:
    raise SystemExit(1)
