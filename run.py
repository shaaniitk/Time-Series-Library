import argparse
import copy
from dataclasses import asdict
import json
import os
import sys

if "MIOPEN_LOG_LEVEL" not in os.environ:
    os.environ["MIOPEN_LOG_LEVEL"] = "3"

if "HSA_OVERRIDE_GFX_VERSION" not in os.environ:
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

import torch
import torch.backends
from utils.print_args import print_args
from utils.tft_config import TFT_PAIRED_ARCHITECTURE_KEYS, apply_tft_profile
from utils.reproducibility import (
    SeedBundle,
    resolve_seed_bundle,
    set_experiment_seed,
    stable_json_hash,
)


def build_parser():
    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn; native TemporalFusionTransformer ignores this knob.')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function, e.g. MSE or Quantile')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', action='store_true', default=True, help='use gpu (default: on)')
    parser.add_argument('--no_use_gpu', action='store_false', dest='use_gpu', help='disable gpu (force cpu)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', action='store_true', default=False,
                        help='enable dtw metric (time consuming; default: off)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--model_init_seed', type=int, default=None,
                        help='Optional base-model initialization seed; omitted values are deterministically derived from --seed and run index.')
    parser.add_argument('--extension_init_seed', type=int, default=None,
                        help='Optional extension initialization seed for paired ablations; omitted values are deterministically derived.')
    parser.add_argument('--data_order_seed', type=int, default=None,
                        help='Optional training-sampler seed; omitted values are deterministically derived.')
    parser.add_argument('--worker_seed', type=int, default=None,
                        help='Optional DataLoader worker seed base; omitted values are deterministically derived.')
    parser.add_argument('--deterministic_mode', type=str, default=None,
                        choices=['off', 'warn', 'strict'],
                        help='Deterministic backend policy. Native TFT v2 defaults to warn; legacy/non-TFT runs default to off.')
    parser.add_argument('--evaluation_policy', type=str, default=None,
                        choices=['validation_only', 'legacy_val_and_test'],
                        help='Whether fitting may inspect test data. Native TFT v2 defaults to validation_only.')
    parser.add_argument('--run_index', type=int, default=0,
                        help='Starting reproducibility-run index; standalone evaluation uses this index to resolve its checkpoint setting.')
    parser.add_argument('--tft_paired_initialization', action='store_true', default=False,
                        help='Construct an explicit reference and copy every shared TFT tensor bitwise into the variant (semantics-v2, extended_safe only).')
    parser.add_argument('--tft_paired_reference_disable', type=str, default='',
                        help='Comma-separated TFT boolean flags to disable in the paired reference, e.g. tft_use_fft_branch.')
    parser.add_argument('--tft_paired_reference_overrides', type=str, default='{}',
                        help='JSON object of additional tft_* values for the paired reference; shared-name shape changes fail.')
    parser.add_argument('--tft_paired_reference_only_pattern', action='append', default=[],
                        help='Repeatable regex allowing state names present only in the paired reference.')
    parser.add_argument('--tft_paired_variant_only_pattern', action='append', default=[],
                        help='Repeatable regex allowing state names present only in the paired variant.')
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    # GCN
    parser.add_argument('--node_dim', type=int, default=10, help='each node embbed to dim dimentions')
    parser.add_argument('--gcn_depth', type=int, default=2, help='')
    parser.add_argument('--gcn_dropout', type=float, default=0.3, help='')
    parser.add_argument('--propalpha', type=float, default=0.3, help='')
    parser.add_argument('--conv_channel', type=int, default=32, help='')
    parser.add_argument('--skip_channel', type=int, default=32, help='')

    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')

    parser.add_argument('--tft_profile', type=str, default='extended_safe',
                        choices=['canonical', 'extended_safe', 'experimental_full'],
                        help='Resolved TFT profile. canonical is the trustworthy reference, extended_safe keeps repaired defaults, experimental_full enables hardened research combinations.')
    parser.add_argument('--tft_extension_semantics_version', type=int, default=2,
                        choices=[1, 2],
                        help='Native TFT extension semantics: 1 replays legacy July-2026 operators; 2 uses repaired/current contracts.')
    parser.add_argument('--tft_allow_legacy_extension_checkpoint', action='store_true', default=False,
                        help='Explicitly allow loading legacy extension weights, only with --tft_extension_semantics_version 1.')
    tft_integration_choices = ['off', 'neutral', 'small_residual', 'legacy']
    for integration_flag, integration_label in (
        ('tft_fft_integration_mode', 'FFT branch'),
        ('tft_cross_attention_integration_mode', 'explicit cross-attention'),
        ('tft_lag_integration_mode', 'lag-attention'),
        ('tft_higher_order_integration_mode', 'higher-order interaction'),
        ('tft_temporal_compression_integration_mode', 'temporal compression'),
        ('tft_graph_integration_mode', 'graph cross-mixing'),
        ('tft_covariate_reattention_integration_mode', 'covariate reattention'),
        ('tft_regime_moe_integration_mode', 'regime MoE'),
        ('tft_dual_attention_integration_mode', 'dual attention'),
        ('tft_vsn_bypass_integration_mode', 'VSN residual bypass'),
    ):
        parser.add_argument(
            f'--{integration_flag}',
            type=str,
            default=None,
            choices=tft_integration_choices,
            help=(
                f'Integration mode for the native TFT {integration_label}; '
                'an enabled semantics-v2 branch defaults to exact-neutral.'
            ),
        )
    parser.add_argument(
        '--tft_extension_residual_shape',
        type=str,
        default='scalar',
        choices=['scalar', 'channel'],
        help='Use one residual strength per extension or one per latent channel.',
    )
    parser.add_argument(
        '--tft_small_residual_init',
        type=float,
        default=1e-3,
        help='Declared nonzero initialization used only by small_residual mode.',
    )
    parser.add_argument(
        '--tft_position_unit',
        type=str,
        default='steps',
        choices=['steps', 'trading_sessions', 'calendar_days'],
        help='Physical unit of native TFT temporal coordinates.',
    )
    parser.add_argument(
        '--tft_position_source',
        type=str,
        default='row_index',
        choices=['row_index', 'explicit_argument', 'known_feature'],
        help='Source of the native TFT temporal-coordinate contract.',
    )
    parser.add_argument(
        '--tft_position_feature_name',
        type=str,
        default='',
        help='Known-feature column used as temporal coordinates when source=known_feature.',
    )
    parser.add_argument(
        '--tft_declared_regular_sampling',
        action='store_true',
        default=False,
        help=(
            'Explicitly declare that adjacent rows are equally spaced. Required '
            'by semantics-v2 when tft_position_source=row_index; do not use for '
            'irregular market-calendar observations.'
        ),
    )

    # TFT strict schema controls
    parser.add_argument('--tft_observed_pos', type=str, default='',
                        help='Comma-separated observed feature indices for TFT when dataset key is not pre-registered.')
    parser.add_argument('--tft_static_pos', type=str, default='',
                        help='Comma-separated static feature indices for TFT.')
    parser.add_argument('--tft_target_pos', type=str, default='',
                        help='Comma-separated source feature indices (in encoder features) mapped to output target channels.')
    parser.add_argument('--tft_use_swiglu', action='store_true', default=False,
                        help='Use SwiGLU instead of GLU for TFT gating networks.')
    parser.add_argument('--tft_full_attention', action='store_true', default=False,
                        help='Use full standard MultiHeadAttention instead of Interpretable Attention in TFT.')
    parser.add_argument('--tft_cross_variable_mixing', action='store_true', default=False,
                        help='Apply Cross-Variable Attention mixing before VSN in TFT.')
    parser.add_argument('--tft_allow_custom_known', action='store_true', default=False,
                        help='Relax rigid known_len timestamp count validations in TFT inputs.')
    parser.add_argument('--tft_known_len', type=int, default=0,
                        help='Required known-future feature width when tft_allow_custom_known=True.')
    parser.add_argument('--tft_known_max_channels', type=int, default=512,
                        help='Maximum supported known-future channels for TFT custom-known embedding.')
    parser.add_argument('--tft_known_feature_names', type=str, default='',
                        help='Comma-separated known-future feature names; required when tft_allow_custom_known=True.')
    parser.add_argument('--tft_vsn_residual_bypass', action='store_true', default=False,
                        help='Enable residual bypass in TFT variable selection networks.')
    parser.add_argument('--tft_dual_attention_fusion', action='store_true', default=False,
                        help='Fuse full and interpretable attention branches in TFT decoder.')
    parser.add_argument('--tft_use_lag_attention', action='store_true', default=False,
                        help='Enable reusable multi-scale lag attention branch in TFT decoder.')
    parser.add_argument('--tft_lag_scales', type=str, default='1,2,4',
                        help='Comma-separated lag scales for TFT lag attention.')
    parser.add_argument('--tft_temporal_backbone', type=str, default='hybrid_tcn_lstm', choices=['lstm', 'gated_tcn', 'hybrid_tcn_lstm'],
                        help='Temporal backbone used before TFT enrichment and attention blocks.')
    parser.add_argument('--tft_temporal_backbone_layers', type=int, default=3,
                        help='Number of TCN layers used by TFT gated_tcn and hybrid_tcn_lstm backbones; plain lstm ignores this knob.')
    parser.add_argument('--tft_temporal_kernel_size', type=int, default=3,
                        help='Kernel size for the TFT gated TCN temporal backbone.')
    parser.add_argument('--tft_temporal_hidden_size', type=int, default=0,
                        help='Hidden size for the TFT gated TCN temporal backbone; 0 uses d_model.')
    parser.add_argument('--tft_use_higher_order', action='store_true', default=False,
                        help='Enable reusable higher-order interaction block in TFT decoder.')
    parser.add_argument('--tft_interaction_order', type=int, default=2,
                        help='Higher-order interaction order for TFT decoder block (2 or 3).')
    parser.add_argument('--tft_interaction_rank', type=int, default=0,
                        help='Low-rank dimension for TFT higher-order interaction block; 0 uses the model default.')
    parser.add_argument('--tft_use_regime_moe', action='store_true', default=False,
                        help='Enable regime-aware sparse MoE in TFT decoder.')
    parser.add_argument('--tft_use_explicit_cross_attention', action='store_true', default=False,
                        help='Enable explicit future-to-history cross-attention in TFT decoder layers.')
    parser.add_argument('--tft_cross_attention_type', type=str, default='full', choices=['full', 'interpretable'],
                        help='Type of explicit TFT cross-attention to use when enabled.')
    parser.add_argument('--tft_attention_position_bias', type=str, default='none', choices=['none', 'rope', 'alibi'],
                        help='Temporal positional biasing strategy for TFT attention blocks.')
    parser.add_argument('--tft_attention_backend', type=str, default='exact', choices=['exact', 'sdpa'],
                        help='Backend for TFT full-attention branches; sdpa falls back to exact when attention weights are requested.')
    parser.add_argument('--tft_attention_dropout', type=float, default=0.0,
                        help='Attention-probability dropout for native TFT attention blocks; output projection dropout remains separate.')
    parser.add_argument('--tft_rope_base', type=float, default=10000.0,
                        help='Base period used when TFT attention positional bias is set to rope.')
    parser.add_argument('--tft_alibi_scale', type=float, default=1.0,
                        help='Scale factor applied to ALiBi slopes when TFT attention positional bias is set to alibi.')
    parser.add_argument('--tft_use_revin', action='store_true', default=False,
                        help='Enable RevIN normalization/denormalization in TFT.')
    parser.add_argument('--tft_revin_affine', action='store_true', default=False,
                        help='Enable learnable affine parameters in TFT RevIN normalization.')
    parser.add_argument('--tft_use_quantile_head', action='store_true', default=False,
                        help='Enable a TFT quantile prediction head in addition to the point forecast head.')
    parser.add_argument('--tft_output_quantiles', type=str, default='0.1,0.5,0.9',
                        help='Comma-separated quantile levels for TFT quantile head, e.g. 0.1,0.5,0.9.')
    parser.add_argument('--tft_output_mode', type=str, default='',
                        help='TFT output mode: point, quantile, or joint. Empty uses a backward-compatible default.')
    parser.add_argument('--tft_point_loss_coeff', type=float, default=1.0,
                        help='Point-loss coefficient for TFT joint output mode.')
    parser.add_argument('--tft_quantile_loss_coeff', type=float, default=1.0,
                        help='Quantile-loss coefficient for TFT joint output mode.')
    parser.add_argument('--tft_num_regimes', type=int, default=4,
                        help='Number of regimes for TFT regime-aware MoE.')
    parser.add_argument('--tft_num_moe_experts', type=int, default=4,
                        help='Number of experts for TFT regime-aware MoE.')
    parser.add_argument('--tft_moe_top_k', type=int, default=2,
                        help='Top-k experts to route to per timestep in TFT MoE.')
    parser.add_argument('--tft_moe_hidden_size', type=int, default=0,
                        help='Hidden size for TFT MoE experts; 0 uses d_model.')
    parser.add_argument('--tft_moe_noise_epsilon', type=float, default=1e-2,
                        help='Noise epsilon for TFT MoE noisy routing.')
    parser.add_argument('--tft_moe_aux_loss_coeff', type=float, default=0.0,
                        help='Coefficient applied to TFT MoE auxiliary routing loss during training/validation.')
    parser.add_argument('--tft_payload_stack_layers', action='store_true', default=False,
                        help='Stack layer-wise TFT decoder interpretation payloads when return_interpretation=True.')
    parser.add_argument('--tft_use_fft_branch', action='store_true', default=False,
                        help='Enable parallel FFT spectral processing branch in TFT temporal backbone.')
    parser.add_argument('--tft_fft_modes', type=int, default=32,
                        help='Number of frequency modes to retain in TFT FFT branch.')
    parser.add_argument('--tft_fft_mode_select', type=str, default='low', choices=['low', 'top_amplitude', 'learned'],
                        help='Frequency mode selection strategy for TFT FFT branch.')
    parser.add_argument('--tft_stochastic_depth_rate', type=float, default=0.0,
                        help='Stochastic depth drop rate for TFT decoder layers (0.0 = disabled).')
    parser.add_argument('--tft_gradient_checkpointing', action='store_true', default=False,
                        help='Enable gradient checkpointing for TFT decoder layers to save memory.')
    parser.add_argument('--tft_use_temporal_compression', action='store_true', default=False,
                        help='Enable learned temporal compression of history before attention (long sequences).')
    parser.add_argument('--tft_tc_stride', type=int, default=2,
                        help='Compression stride for temporal compression (2 = halve history length).')
    parser.add_argument('--tft_tc_threshold', type=int, default=256,
                        help='Minimum history length to activate temporal compression; no-op below this.')
    parser.add_argument('--tft_vsn_n_selection_heads', type=int, default=1,
                        help='Number of selection heads in TFT Variable Selection Networks (1 = original behavior).')
    parser.add_argument('--tft_mlp_quantile_projection', action='store_true', default=False,
                        help='Use a 2-layer MLP instead of single Linear for TFT quantile projection head.')
    parser.add_argument('--tft_quantile_projection_ff_size', type=int, default=0,
                        help='Hidden size for MLP quantile projection; 0 uses d_model.')
    parser.add_argument('--tft_per_target_heads', action='store_true', default=False,
                        help='Use per-target MLP decoder heads instead of shared linear projection in TFT.')
    parser.add_argument('--tft_vsn_per_feature_gating', action='store_true', default=False,
                        help='Use per-feature sigmoid gating instead of per-covariate softmax in TFT VSN.')
    parser.add_argument('--tft_covariate_reattention', action='store_true', default=False,
                        help='Enable covariate-aware cross-attention enrichment in TFT decoder layers.')
    parser.add_argument('--tft_moe_capacity_factor', type=float, default=1.25,
                        help='Expert capacity factor for TFT MoE; each expert handles at most capacity_factor * tokens/num_experts tokens.')
    parser.add_argument('--tft_vsn_low_rank_threshold', type=int, default=64,
                        help='When variable_num >= this threshold, VSN uses low-rank factorization for weight generation.')
    parser.add_argument('--tft_debug_checks', action='store_true', default=False,
                        help='Enable expensive per-layer TFT finiteness/debug checks; shape/schema checks remain always enabled.')
    parser.add_argument('--tft_graph_type', type=str, default='dense', choices=['dense', 'sparse', 'temporal_sparse'],
                        help='Graph learner type for TFT cross-variable mixing: dense (original), sparse (top-k), temporal_sparse (top-k + GRU evolution).')
    parser.add_argument('--tft_graph_top_k', type=int, default=10,
                        help='Max neighbors per node in sparse TFT graph learner.')
    parser.add_argument('--tft_graph_num_layers', type=int, default=2,
                        help='Number of GNN message-passing layers in advanced TFT graph learner.')
    parser.add_argument('--tft_graph_temporal_evolution', action='store_true', default=False,
                        help='Enable GRU-based temporal adjacency evolution in TFT graph learner.')
    parser.add_argument('--tft_graph_edge_features', action='store_true', default=False,
                        help='Enable learnable edge features in TFT graph message passing.')

    # TimeFilter
    parser.add_argument('--alpha', type=float, default=0.1, help='KNN for Graph Construction')
    parser.add_argument('--top_p', type=float, default=0.5, help='Dynamic Routing in MoE')
    parser.add_argument('--pos', type=int, choices=[0, 1], default=1, help='Positional Embedding. Set pos to 0 or 1')
    return parser


def normalize_args(args):
    args = argparse.Namespace(**vars(args))
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    def _parse_int_list(value):
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if value == '':
                return None
            return [int(v.strip()) for v in value.split(',') if v.strip() != '']
        return value

    def _parse_float_list(value):
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if value == '':
                return None
            return [float(v.strip()) for v in value.split(',') if v.strip() != '']
        return value

    def _parse_str_list(value):
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if value == '':
                return None
            return [v.strip() for v in value.split(',') if v.strip() != '']
        return value

    args.tft_observed_pos = _parse_int_list(args.tft_observed_pos)
    args.tft_static_pos = _parse_int_list(args.tft_static_pos)
    args.tft_target_pos = _parse_int_list(args.tft_target_pos)
    args.tft_lag_scales = _parse_int_list(args.tft_lag_scales)
    args.tft_output_quantiles = _parse_float_list(args.tft_output_quantiles)
    args.tft_known_feature_names = _parse_str_list(args.tft_known_feature_names)
    args.tft_paired_reference_disable = (
        _parse_str_list(args.tft_paired_reference_disable) or []
    )
    args.tft_paired_reference_only_pattern = [
        str(value).strip()
        for value in (args.tft_paired_reference_only_pattern or [])
        if str(value).strip()
    ]
    args.tft_paired_variant_only_pattern = [
        str(value).strip()
        for value in (args.tft_paired_variant_only_pattern or [])
        if str(value).strip()
    ]
    if isinstance(args.tft_paired_reference_overrides, str):
        try:
            args.tft_paired_reference_overrides = json.loads(
                args.tft_paired_reference_overrides
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                "tft_paired_reference_overrides must be a valid JSON object."
            ) from exc
    if not isinstance(args.tft_paired_reference_overrides, dict):
        raise ValueError("tft_paired_reference_overrides must resolve to a JSON object.")
    if isinstance(args.tft_output_mode, str) and args.tft_output_mode.strip() == '':
        args.tft_output_mode = None
    if args.tft_interaction_rank == 0:
        args.tft_interaction_rank = None
    if args.tft_temporal_hidden_size == 0:
        args.tft_temporal_hidden_size = None
    if args.tft_moe_hidden_size == 0:
        args.tft_moe_hidden_size = None
    if args.tft_known_len is not None and args.tft_known_len <= 0:
        args.tft_known_len = None

    if args.model == 'TemporalFusionTransformer':
        args = apply_tft_profile(args)

    is_native_tft = args.model == 'TemporalFusionTransformer'
    semantics_version = int(
        getattr(args, 'tft_extension_semantics_version', 1)
    ) if is_native_tft else 1
    if args.deterministic_mode is None:
        args.deterministic_mode = (
            'warn' if is_native_tft and semantics_version == 2 else 'off'
        )
    if args.evaluation_policy is None:
        args.evaluation_policy = (
            'validation_only'
            if is_native_tft and semantics_version == 2
            else 'legacy_val_and_test'
        )
    if int(args.run_index) < 0:
        raise ValueError("run_index must be non-negative.")

    paired_controls_present = bool(
        args.tft_paired_reference_disable
        or args.tft_paired_reference_overrides
        or args.tft_paired_reference_only_pattern
        or args.tft_paired_variant_only_pattern
    )
    if paired_controls_present and not args.tft_paired_initialization:
        raise ValueError(
            "tft_paired_reference_* controls require --tft_paired_initialization."
        )
    if args.tft_paired_initialization:
        if not is_native_tft:
            raise ValueError(
                "--tft_paired_initialization is supported only for the native "
                "TemporalFusionTransformer."
            )
        if semantics_version != 2:
            raise ValueError(
                "--tft_paired_initialization requires "
                "--tft_extension_semantics_version 2."
            )
        if args.tft_profile != 'extended_safe':
            raise ValueError(
                "--tft_paired_initialization currently requires "
                "--tft_profile extended_safe so profile defaults cannot silently "
                "change the reference arm."
            )
        if not (
            args.tft_paired_reference_disable
            or args.tft_paired_reference_overrides
        ):
            raise ValueError(
                "--tft_paired_initialization requires at least one reference "
                "disable flag or override."
            )

        forbidden_reference_keys = {
            'tft_profile',
            'tft_extension_semantics_version',
            'tft_allow_legacy_extension_checkpoint',
            'tft_config_digest',
            'tft_digest_schema',
            'tft_paired_initialization',
            'tft_paired_reference_disable',
            'tft_paired_reference_overrides',
            'tft_paired_reference_only_pattern',
            'tft_paired_variant_only_pattern',
        }
        disable_flags = sorted(set(args.tft_paired_reference_disable))
        if len(disable_flags) != len(args.tft_paired_reference_disable):
            raise ValueError("tft_paired_reference_disable contains duplicate flags.")
        for key in disable_flags:
            if key in forbidden_reference_keys or not key.startswith('tft_'):
                raise ValueError(f"Invalid paired reference disable flag: {key!r}.")
            if not hasattr(args, key):
                raise ValueError(f"Unknown paired reference disable flag: {key!r}.")
            if key not in TFT_PAIRED_ARCHITECTURE_KEYS:
                raise ValueError(
                    f"Paired reference disable {key!r} is not an allowed TFT "
                    "architecture control."
                )
            value = getattr(args, key)
            if not isinstance(value, bool):
                raise ValueError(
                    f"Paired reference disable flag {key!r} must name a boolean option."
                )
            if not value:
                raise ValueError(
                    f"Paired variant does not enable {key!r}; disabling it in the "
                    "reference would not create an ablation."
                )

        overrides = {}
        for key, value in sorted(args.tft_paired_reference_overrides.items()):
            if (
                not isinstance(key, str)
                or key in forbidden_reference_keys
                or not key.startswith('tft_')
            ):
                raise ValueError(f"Invalid paired reference override key: {key!r}.")
            if not hasattr(args, key):
                raise ValueError(f"Unknown paired reference override key: {key!r}.")
            if key not in TFT_PAIRED_ARCHITECTURE_KEYS:
                raise ValueError(
                    f"Paired reference override {key!r} is not an allowed TFT "
                    "architecture control."
                )
            if key in disable_flags:
                raise ValueError(
                    f"Paired reference key {key!r} cannot be both disabled and overridden."
                )
            if getattr(args, key) == value:
                raise ValueError(
                    f"Paired reference override {key!r} equals the variant value; "
                    "it would not create an ablation."
                )
            overrides[key] = value

        args.tft_paired_reference_disable = disable_flags
        args.tft_paired_reference_overrides = overrides

    paired_spec = {
        'schema_version': 1,
        'enabled': bool(args.tft_paired_initialization),
        'reference_disable': sorted(args.tft_paired_reference_disable),
        'reference_overrides': {
            key: args.tft_paired_reference_overrides[key]
            for key in sorted(args.tft_paired_reference_overrides)
        },
        'reference_only_patterns': list(args.tft_paired_reference_only_pattern),
        'variant_only_patterns': list(args.tft_paired_variant_only_pattern),
    }
    args.tft_paired_reference_spec = paired_spec
    args.tft_paired_reference_spec_digest = stable_json_hash(paired_spec)[:12]

    if args.model == 'TemporalFusionTransformer':
        from models.TemporalFusionTransformer import datatype_dict
        if args.data not in datatype_dict and args.tft_observed_pos is None:
            print(
                f"ERROR: Dataset '{args.data}' is not registered for TemporalFusionTransformer. "
                "Provide --tft_observed_pos explicitly."
            )
            print(f"Registered datasets: {list(datatype_dict.keys())}")
            sys.exit(1)
        if args.c_out != args.enc_in and args.tft_target_pos is None:
            if args.c_out == 1 and args.enc_in > 1:
                pass # The model handles this MS fallback automatically
            else:
                print(
                    "ERROR: For TemporalFusionTransformer with c_out != enc_in, "
                    "you must provide --tft_target_pos."
                )
                sys.exit(1)
    return args


def build_setting(args, ii):
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.expand,
        args.d_conv,
        args.factor,
        args.embed,
        args.distil,
        args.des, ii)
    if args.model == 'TemporalFusionTransformer':
        setting = (
            f"{setting}_tp{args.tft_profile}"
            f"_tsv{args.tft_extension_semantics_version}"
            f"_td{args.tft_config_digest}"
        )
        if int(args.tft_extension_semantics_version) == 2:
            run_digest = getattr(args, 'reproducibility_digest', None)
            if not run_digest:
                bundle = resolve_seed_bundle(args, iteration=ii)
                run_digest = stable_json_hash(
                    {
                        "seed_bundle": asdict(bundle),
                        "deterministic_mode": getattr(args, 'deterministic_mode', 'warn'),
                        "evaluation_policy": getattr(args, 'evaluation_policy', 'validation_only'),
                        "paired_reference_spec": getattr(
                            args,
                            'tft_paired_reference_spec',
                            {
                                'schema_version': 1,
                                'enabled': False,
                                'reference_disable': [],
                                'reference_overrides': {},
                                'reference_only_patterns': [],
                                'variant_only_patterns': [],
                            },
                        ),
                    }
                )[:12]
            setting = f"{setting}_rd{run_digest}"
    return setting


def resolve_run_args(base_args, iteration):
    run_args = copy.deepcopy(base_args)
    run_args._seed_roots = {
        'seed': int(getattr(base_args, 'seed', 2021)),
        'model_init_seed': getattr(base_args, 'model_init_seed', None),
        'extension_init_seed': getattr(base_args, 'extension_init_seed', None),
        'data_order_seed': getattr(base_args, 'data_order_seed', None),
        'worker_seed': getattr(base_args, 'worker_seed', None),
        'training_seed': getattr(base_args, 'training_seed', None),
    }
    isolated_v2 = (
        getattr(run_args, 'model', None) == 'TemporalFusionTransformer'
        and int(getattr(run_args, 'tft_extension_semantics_version', 1)) == 2
    )
    if isolated_v2:
        bundle = resolve_seed_bundle(run_args, iteration=iteration)
    else:
        # Historical/native-v1 and non-TFT paths keep one global stream so the
        # explicit v1 matrix replay remains faithful. Iterations receive a
        # deterministic, collision-free base+index schedule.
        base_seed = int(getattr(run_args, 'seed', 2021))
        if base_seed < 0:
            raise ValueError("seed must be non-negative.")
        legacy_seed = base_seed + int(iteration)
        if legacy_seed >= 2**63 - 1:
            raise ValueError("seed + run_index exceeds Torch's portable seed range.")
        bundle = SeedBundle(
            derivation_version=0,
            base_seed=base_seed,
            iteration=int(iteration),
            experiment_seed=legacy_seed,
            model_init_seed=legacy_seed,
            extension_init_seed=legacy_seed,
            data_order_seed=legacy_seed,
            worker_seed=legacy_seed,
            training_seed=legacy_seed,
        )
    run_args._isolated_rng_streams = isolated_v2
    for name, value in asdict(bundle).items():
        if name in {
            'model_init_seed',
            'extension_init_seed',
            'data_order_seed',
            'worker_seed',
        }:
            setattr(run_args, name, value)
    run_args.experiment_seed = bundle.experiment_seed
    run_args.training_seed = bundle.training_seed
    run_args.seed_derivation_version = bundle.derivation_version
    run_args.run_index = int(iteration)
    run_args._seed_bundle = bundle
    run_args.reproducibility_digest = stable_json_hash(
        {
            "seed_bundle": asdict(bundle),
            "deterministic_mode": run_args.deterministic_mode,
            "evaluation_policy": run_args.evaluation_policy,
            "paired_reference_spec": getattr(
                run_args,
                'tft_paired_reference_spec',
                {
                    'schema_version': 1,
                    'enabled': False,
                    'reference_disable': [],
                    'reference_overrides': {},
                    'reference_only_patterns': [],
                    'variant_only_patterns': [],
                },
            ),
        }
    )[:12]
    return run_args, bundle


def _clear_device_cache(args):
    if not args.use_gpu:
        return
    if args.gpu_type == 'mps':
        torch.backends.mps.empty_cache()
    elif args.gpu_type == 'cuda':
        torch.cuda.empty_cache()


def execute_training_runs(args, Exp):
    settings = []
    for offset in range(args.itr):
        iteration = int(args.run_index) + offset
        run_args, bundle = resolve_run_args(args, iteration)
        set_experiment_seed(
            bundle,
            selected_seed=bundle.model_init_seed,
            deterministic_mode=run_args.deterministic_mode,
        )
        print('Args in experiment:')
        print_args(run_args)
        exp = Exp(run_args)
        if run_args._isolated_rng_streams:
            set_experiment_seed(
                bundle,
                selected_seed=bundle.training_seed,
                deterministic_mode=run_args.deterministic_mode,
            )
        setting = build_setting(run_args, iteration)
        settings.append(setting)

        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        if run_args.evaluation_policy == 'legacy_val_and_test':
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
        _clear_device_cache(run_args)
    return settings


def execute_standalone_evaluation(args, Exp):
    iteration = int(args.run_index)
    run_args, bundle = resolve_run_args(args, iteration)
    set_experiment_seed(
        bundle,
        selected_seed=bundle.model_init_seed,
        deterministic_mode=run_args.deterministic_mode,
    )
    print('Args in experiment:')
    print_args(run_args)
    exp = Exp(run_args)
    setting = build_setting(run_args, iteration)

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    _clear_device_cache(run_args)
    return setting


if __name__ == '__main__':

    parser = build_parser()
    args = normalize_args(parser.parse_args())

    if args.task_name == 'long_term_forecast':
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        from exp.exp_imputation import Exp_Imputation
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        from exp.exp_anomaly_detection import Exp_Anomaly_Detection
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        from exp.exp_classification import Exp_Classification
        Exp = Exp_Classification
    elif args.task_name == 'zero_shot_forecast':
        from exp.exp_zero_shot_forecasting import Exp_Zero_Shot_Forecast
        Exp = Exp_Zero_Shot_Forecast
    else:
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        execute_training_runs(args, Exp)
    else:
        execute_standalone_evaluation(args, Exp)
