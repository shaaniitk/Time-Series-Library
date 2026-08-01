from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from utils.tools import EarlyStopping, adjust_learning_rate, combine_primary_and_aux_loss, get_auxiliary_loss, visual
import json
from utils.metrics import metric, quantile_metric
from utils.losses import QuantileLoss
from utils.tft_schema import (
    inverse_transform_selected,
    is_tft_model,
    resolve_tft_schema,
    resolve_target_positions,
    select_tft_truth,
)
from utils.tft_config import (
    TFT_PAIRED_ARCHITECTURE_KEYS,
    TFT_RESULT_METADATA_FILENAME,
    apply_tft_profile,
    validate_tft_v2_artifact_readiness,
    write_tft_semantics_metadata,
)
from utils.tft_checkpoint import load_tft_checkpoint, resolve_tft_checkpoint_path
from utils.reproducibility import (
    atomic_write_json,
    build_paired_models,
    resolve_seed_bundle,
    stable_json_hash,
    state_dict_sha256,
)
import torch
import torch.nn as nn
from torch import optim
import os
import time
import numpy as np
import platform
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single


TFT_REPRODUCIBILITY_FILENAME = "reproducibility_manifest.json"


# These expressions describe state introduced or replaced by a single optional
# TFT switch. They are intentionally narrow: any unlisted name or any shared
# name whose shape/dtype changes makes paired construction fail closed.
_TFT_OPTIONAL_STATE_PATTERNS = {
    'tft_vsn_residual_bypass': {
        'off_only': (),
        'on_only': (
            r'^(?:static_encoder.*|history_vsn|future_vsn)\.(?:residual_projection\.|vsn_bypass_residual_adapter\.residual_strength$)',
        ),
    },
    'tft_use_fft_branch': {
        'off_only': (),
        'on_only': (
            r'^temporal_fusion_decoder\.layers\.\d+\.(?:fft_branch|fft_fusion_gate|fft_residual_adapter)\.',
        ),
    },
    'tft_use_explicit_cross_attention': {
        'off_only': (),
        'on_only': (
            r'^temporal_fusion_decoder\.layers\.\d+\.(?:cross_attention|gate_after_cross_attention|cross_attention_residual_adapter)\.',
        ),
    },
    'tft_use_lag_attention': {
        'off_only': (),
        'on_only': (
            r'^temporal_fusion_decoder\.layers\.\d+\.(?:lag_attention_module\.|lag_residual_adapter\.)',
        ),
    },
    'tft_use_higher_order': {
        'off_only': (),
        'on_only': (
            r'^temporal_fusion_decoder\.layers\.\d+\.(?:higher_order_block|higher_order_residual_adapter)\.',
        ),
    },
    'tft_use_temporal_compression': {
        'off_only': (),
        'on_only': (
            r'^temporal_fusion_decoder\.layers\.\d+\.(?:temporal_compression|compression_residual_adapter)\.',
        ),
    },
    'tft_covariate_reattention': {
        'off_only': (),
        'on_only': (
            r'^temporal_fusion_decoder\.layers\.\d+\.(?:covariate_cross_attention|gate_after_reattention|covariate_reattention_residual_adapter)\.',
        ),
    },
    'tft_cross_variable_mixing': {
        'off_only': (),
        'on_only': (
            r'^(?:static_encoder|history_vsn|future_vsn).*(?:\.cross_mixing\.|\.graph_residual_adapter\.residual_strength$)',
        ),
    },
    'tft_use_regime_moe': {
        'off_only': (),
        'on_only': (
            r'^temporal_fusion_decoder\.layers\.\d+\.(?:regime_moe|regime_moe_residual_adapter)\.',
        ),
    },
    'tft_dual_attention_fusion': {
        'off_only': (),
        'on_only': (
            r'^temporal_fusion_decoder\.layers\.\d+\.(?:dual_attention_module|dual_attention_residual_adapter)\.',
        ),
    },
    'tft_full_attention': {
        'off_only': (
            r'^temporal_fusion_decoder\.layers\.\d+\.attention\.qkv_linears\.',
        ),
        'on_only': (
            r'^temporal_fusion_decoder\.layers\.\d+\.attention\.(?:q_linear|k_linear|v_linear)\.',
        ),
    },
    'tft_use_quantile_head': {
        'off_only': (),
        'on_only': (r'^quantile_projection\.',),
    },
    'tft_per_target_heads': {
        'off_only': (r'^temporal_fusion_decoder\.out_projection\.',),
        'on_only': (r'^temporal_fusion_decoder\.target_heads\.',),
    },
}


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        if is_tft_model(args):
            # A model instance may have cached a resolved schema on a reused
            # Namespace. Re-resolve it from the declared fields so config and
            # pairing identity cannot inherit stale positions/names.
            if hasattr(args, '_tft_resolved_schema'):
                delattr(args, '_tft_resolved_schema')
            args = apply_tft_profile(args)
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self._tft_target_positions = resolve_target_positions(args) if is_tft_model(args) else None
        self._tft_output_mode = str(getattr(args, 'tft_output_mode', 'joint' if getattr(args, 'tft_use_quantile_head', False) else 'point')).lower()
        self._point_loss_coeff = float(getattr(args, 'tft_point_loss_coeff', 1.0))
        self._quantile_loss_coeff = float(getattr(args, 'tft_quantile_loss_coeff', 1.0))
        quantiles = getattr(args, 'tft_output_quantiles', None)
        self._joint_quantile_criterion = QuantileLoss(quantiles) if is_tft_model(args) and self._tft_output_mode in {'quantile', 'joint'} else None
        self._quantile_levels = tuple(self._joint_quantile_criterion.quantiles) if self._joint_quantile_criterion is not None else None
        default_evaluation_policy = (
            'validation_only'
            if is_tft_model(args)
            and int(getattr(args, 'tft_extension_semantics_version', 1)) == 2
            else 'legacy_val_and_test'
        )
        self._evaluation_policy = str(
            getattr(args, 'evaluation_policy', default_evaluation_policy)
            or default_evaluation_policy
        )
        if self._evaluation_policy not in {'validation_only', 'legacy_val_and_test'}:
            raise ValueError(
                "evaluation_policy must be validation_only or legacy_val_and_test."
            )
        model_for_state = self.model.module if hasattr(self.model, 'module') else self.model
        self._initial_state_hash = (
            state_dict_sha256(model_for_state.state_dict())
            if is_tft_model(args)
            else None
        )
        paired_report = getattr(self.args, '_paired_initialization_report', None)
        if bool(getattr(self.args, 'tft_paired_initialization', False)):
            if not isinstance(paired_report, dict):
                raise RuntimeError(
                    "Paired TFT construction did not produce its required state report."
                )
            reported_variant_hash = paired_report.get('variant_state_sha256')
            if reported_variant_hash != self._initial_state_hash:
                raise RuntimeError(
                    "Paired initialization report is not bound to the constructed "
                    "TFT variant state."
                )
            paired_report = dict(paired_report)
            paired_report['bound_initial_state_sha256'] = self._initial_state_hash
            paired_report['binding_verified'] = True
            self._trusted_paired_initialization_report = deepcopy(paired_report)
            self._trusted_paired_initialization_report_digest = stable_json_hash(
                self._trusted_paired_initialization_report
            )
            self.args._paired_initialization_report = deepcopy(paired_report)
        elif paired_report is not None:
            raise RuntimeError(
                "An unpaired TFT construction cannot carry a paired-state report."
            )
        else:
            self._trusted_paired_initialization_report = None
            self._trusted_paired_initialization_report_digest = None
        self._last_reproducibility_manifest = None
        self._validate_tft_objective_config()

    def _validated_paired_initialization_report(self):
        report = getattr(self.args, '_paired_initialization_report', None)
        enabled = bool(getattr(self.args, 'tft_paired_initialization', False))
        if not enabled:
            if report is not None:
                raise RuntimeError(
                    "Unpaired TFT run contains an unexpected paired-state report."
                )
            return None
        if not isinstance(report, dict):
            raise RuntimeError("Paired TFT run is missing its initialization report.")
        trusted_report = getattr(
            self, '_trusted_paired_initialization_report', None
        )
        trusted_digest = getattr(
            self, '_trusted_paired_initialization_report_digest', None
        )
        if not isinstance(trusted_report, dict) or not isinstance(trusted_digest, str):
            raise RuntimeError(
                "Paired TFT run has no trusted construction-time report binding."
            )
        if stable_json_hash(report) != trusted_digest:
            raise RuntimeError(
                "Paired TFT initialization report was mutated after model construction."
            )
        if stable_json_hash(trusted_report) != trusted_digest:
            raise RuntimeError(
                "Trusted paired TFT initialization report failed integrity validation."
            )
        paired_spec = getattr(self.args, 'tft_paired_reference_spec', None)
        paired_spec_digest = getattr(
            self.args, 'tft_paired_reference_spec_digest', None
        )
        current_spec = {
            'schema_version': 1,
            'enabled': True,
            'reference_disable': sorted(
                list(
                    getattr(
                        self.args, 'tft_paired_reference_disable', ()
                    ) or ()
                )
            ),
            'reference_overrides': {
                key: value
                for key, value in sorted(
                    dict(
                        getattr(
                            self.args, 'tft_paired_reference_overrides', {}
                        ) or {}
                    ).items()
                )
            },
            'reference_only_patterns': list(
                getattr(
                    self.args, 'tft_paired_reference_only_pattern', ()
                ) or ()
            ),
            'variant_only_patterns': list(
                getattr(
                    self.args, 'tft_paired_variant_only_pattern', ()
                ) or ()
            ),
        }
        if (
            not isinstance(paired_spec, dict)
            or stable_json_hash(paired_spec)[:12] != paired_spec_digest
            or current_spec != paired_spec
            or trusted_report.get('paired_reference_spec') != paired_spec
            or trusted_report.get('paired_reference_spec_digest')
            != paired_spec_digest
            or trusted_report.get('variant_config_digest')
            != getattr(self.args, 'tft_config_digest', None)
        ):
            raise RuntimeError(
                "Paired TFT reference specification/config binding was mutated "
                "or is inconsistent."
            )
        if (
            trusted_report.get('binding_verified') is not True
            or trusted_report.get('variant_state_sha256') != self._initial_state_hash
            or trusted_report.get('bound_initial_state_sha256')
            != self._initial_state_hash
        ):
            raise RuntimeError(
                "Paired TFT initialization report failed model-state binding validation."
            )
        return deepcopy(trusted_report)

    @staticmethod
    def _loader_reproducibility(loader):
        if loader is None:
            return None
        metadata = getattr(loader, 'reproducibility_metadata', None)
        return dict(metadata) if isinstance(metadata, dict) else {
            'split': None,
            'sampler_policy': type(loader.sampler).__name__,
            'fold_manifest': None,
            'fold_manifest_hash': None,
            'epoch0_order_hash': None,
            'first_batch_index_hash': None,
            'data_order_seed': None,
            'worker_seed': None,
            'metadata_status': 'legacy_or_unavailable',
        }

    def _seed_bundle_payload(self):
        bundle = getattr(self.args, '_seed_bundle', None)
        if bundle is None:
            bundle = resolve_seed_bundle(
                self.args,
                iteration=int(getattr(self.args, 'run_index', 0)),
            )
        return asdict(bundle) if is_dataclass(bundle) else dict(bundle)

    def _build_reproducibility_manifest(
        self,
        setting,
        *,
        train_loader=None,
        validation_loader=None,
        test_loader=None,
        artifact_kind='checkpoint',
        epochs_completed=None,
        final_state_hash=None,
    ):
        loaders = {
            'train': self._loader_reproducibility(train_loader),
            'validation': self._loader_reproducibility(validation_loader),
            'test': self._loader_reproducibility(test_loader),
        }
        fold_hashes = {
            name: metadata.get('fold_manifest_hash')
            for name, metadata in loaders.items()
            if metadata is not None
        }
        paired_report = self._validated_paired_initialization_report()
        runner_managed_seeds = getattr(self.args, '_seed_bundle', None) is not None
        isolated_rng_streams = bool(
            getattr(self.args, '_isolated_rng_streams', False)
        )
        payload = {
            'manifest_schema_version': 1,
            'artifact_kind': artifact_kind,
            'setting': setting,
            'model': getattr(self.args, 'model', None),
            'extension_semantics_version': getattr(
                self.args, 'tft_extension_semantics_version', None
            ),
            'config_digest': getattr(self.args, 'tft_config_digest', None),
            'run_digest': getattr(self.args, 'reproducibility_digest', None),
            'paired_reference_spec': getattr(
                self.args, 'tft_paired_reference_spec', None
            ),
            'paired_reference_spec_digest': getattr(
                self.args, 'tft_paired_reference_spec_digest', None
            ),
            'seed_roots': getattr(self.args, '_seed_roots', None),
            'seed_bundle': self._seed_bundle_payload(),
            'rng_stream_policy': (
                'direct_caller_unverified'
                if not runner_managed_seeds
                else 'isolated_v2'
                if isolated_rng_streams
                else 'legacy_global_v1'
            ),
            'seed_usage': {
                'model_init_seed': (
                    'used_for_model_construction'
                    if runner_managed_seeds
                    else 'direct_caller_unverified'
                ),
                'extension_init_seed': (
                    'used_by_paired_initialization'
                    if isinstance(paired_report, dict)
                    else (
                        'reserved_not_used_in_unpaired_construction'
                        if isolated_rng_streams
                        else 'shared_legacy_global_stream'
                    )
                ),
                'data_order_seed': (
                    'used_by_v2_epoch_sampler_when_available'
                    if isolated_rng_streams
                    else 'shared_legacy_global_stream'
                ),
                'worker_seed': (
                    'used_by_v2_loader_generator_when_available'
                    if isolated_rng_streams
                    else 'shared_legacy_global_stream'
                ),
                'training_seed': (
                    'reset_before_training'
                    if runner_managed_seeds and isolated_rng_streams
                    else 'continued_after_model_construction'
                    if runner_managed_seeds
                    else 'direct_caller_unverified'
                ),
            },
            'deterministic_mode': getattr(self.args, 'deterministic_mode', 'off'),
            'evaluation_policy': self._evaluation_policy,
            'initial_state_hash': self._initial_state_hash,
            'final_state_hash': final_state_hash,
            'paired_initialization': paired_report,
            'paired_initialization_report_digest': (
                self._trusted_paired_initialization_report_digest
                if paired_report is not None
                else None
            ),
            'shared_state_hash': (
                paired_report.get(
                    'shared_state_sha256',
                    paired_report.get('shared_state_hash'),
                )
                if isinstance(paired_report, dict)
                else None
            ),
            'loaders': loaders,
            'fold_manifest_hash': stable_json_hash(fold_hashes),
            'first_batch_index_hash': (
                loaders['train'].get('first_batch_index_hash')
                if loaders['train'] is not None
                else None
            ),
            'epochs_completed': epochs_completed,
            'environment': {
                'python': platform.python_version(),
                'torch': torch.__version__,
                'device': str(self.device),
                'platform': platform.platform(),
            },
            'bitwise_identity_scope': (
                'same recorded software/hardware environment; no cross-driver '
                'or cross-hardware bitwise guarantee'
            ),
        }
        return payload

    def _write_reproducibility_manifest(self, directory, payload):
        path = atomic_write_json(
            os.path.join(directory, TFT_REPRODUCIBILITY_FILENAME),
            payload,
        )
        self._last_reproducibility_manifest = payload
        return path

    def _validate_tft_objective_config(self):
        if self._tft_target_positions is None:
            return
        if self._tft_output_mode not in {'point', 'quantile', 'joint'}:
            raise ValueError("tft_output_mode must be one of: point, quantile, joint.")
        if self._tft_output_mode == 'joint':
            if self._point_loss_coeff <= 0.0 or self._quantile_loss_coeff <= 0.0:
                raise ValueError("tft_output_mode=joint requires positive tft_point_loss_coeff and tft_quantile_loss_coeff.")
        if self._tft_output_mode in {'quantile', 'joint'} and self._joint_quantile_criterion is None:
            raise ValueError(f"tft_output_mode={self._tft_output_mode} requires tft_output_quantiles.")

    def _build_model(self):
        # Never trust a caller-supplied private report. Only this guarded
        # construction path may create one, and __init__ binds it to the exact
        # initial model state before any artifact can be written.
        if hasattr(self.args, '_paired_initialization_report'):
            delattr(self.args, '_paired_initialization_report')

        model_class = self.model_dict[self.args.model]
        if (
            is_tft_model(self.args)
            and bool(getattr(self.args, 'tft_paired_initialization', False))
        ):
            if int(getattr(self.args, 'tft_extension_semantics_version', 1)) != 2:
                raise ValueError("Paired TFT initialization requires semantics version 2.")
            if getattr(self.args, 'tft_profile', None) != 'extended_safe':
                raise ValueError(
                    "Paired TFT initialization requires tft_profile=extended_safe."
                )
            paired_spec = getattr(self.args, 'tft_paired_reference_spec', None)
            paired_spec_digest = getattr(
                self.args, 'tft_paired_reference_spec_digest', None
            )
            if (
                not isinstance(paired_spec, dict)
                or stable_json_hash(paired_spec)[:12] != paired_spec_digest
                or paired_spec.get('enabled') is not True
            ):
                raise ValueError(
                    "Paired TFT initialization requires a normalized, intact "
                    "reference specification."
                )

            disable_flags = list(
                getattr(self.args, 'tft_paired_reference_disable', ()) or ()
            )
            overrides = dict(
                getattr(self.args, 'tft_paired_reference_overrides', {}) or {}
            )
            reference_only_patterns = list(
                getattr(
                    self.args, 'tft_paired_reference_only_pattern', ()
                ) or ()
            )
            variant_only_patterns = list(
                getattr(self.args, 'tft_paired_variant_only_pattern', ()) or ()
            )
            current_spec = {
                'schema_version': 1,
                'enabled': True,
                'reference_disable': sorted(disable_flags),
                'reference_overrides': {
                    key: overrides[key] for key in sorted(overrides)
                },
                'reference_only_patterns': reference_only_patterns,
                'variant_only_patterns': variant_only_patterns,
            }
            if current_spec != paired_spec:
                raise ValueError(
                    "Paired TFT controls no longer match their normalized, "
                    "identity-bound reference specification."
                )
            if len(set(disable_flags)) != len(disable_flags):
                raise ValueError("Paired TFT disable controls contain duplicates.")
            for key in disable_flags:
                if key not in TFT_PAIRED_ARCHITECTURE_KEYS:
                    raise ValueError(
                        f"Paired TFT reference disable {key!r} is not an allowed "
                        "architecture control."
                    )
                if not isinstance(getattr(self.args, key, None), bool):
                    raise ValueError(
                        f"Paired TFT reference disable {key!r} must name a "
                        "boolean architecture control."
                    )
                if not getattr(self.args, key):
                    raise ValueError(
                        f"Paired TFT variant does not enable {key!r}."
                    )
            for key, value in overrides.items():
                if key not in TFT_PAIRED_ARCHITECTURE_KEYS:
                    raise ValueError(
                        f"Paired TFT reference override {key!r} is not an "
                        "allowed architecture control."
                    )
                if key in disable_flags:
                    raise ValueError(
                        f"Paired TFT reference key {key!r} is both disabled and "
                        "overridden."
                    )
                if not hasattr(self.args, key):
                    raise ValueError(
                        f"Unknown paired TFT reference override {key!r}."
                    )
                if getattr(self.args, key) == value:
                    raise ValueError(
                        f"Paired TFT reference override {key!r} does not differ "
                        "from the variant."
                    )

            reference_args = deepcopy(self.args)
            for key in disable_flags:
                setattr(reference_args, key, False)
            for key, value in overrides.items():
                setattr(reference_args, key, deepcopy(value))
            reference_args.tft_paired_initialization = False
            if hasattr(reference_args, '_tft_resolved_schema'):
                delattr(reference_args, '_tft_resolved_schema')
            reference_args = apply_tft_profile(reference_args)
            reference_schema = resolve_tft_schema(reference_args)
            variant_schema = resolve_tft_schema(self.args)
            if reference_schema != variant_schema:
                raise ValueError(
                    "Paired reference and variant must have the identical resolved "
                    "TFT feature/target/known-future schema."
                )
            if reference_args.tft_config_digest == self.args.tft_config_digest:
                raise ValueError(
                    "Paired reference and variant resolve to the same TFT "
                    "architecture digest."
                )

            changed_keys = set(disable_flags) | set(overrides)
            for key in sorted(changed_keys):
                pattern_spec = _TFT_OPTIONAL_STATE_PATTERNS.get(key)
                if pattern_spec is None:
                    continue
                reference_value = bool(getattr(reference_args, key, False))
                variant_value = bool(getattr(self.args, key, False))
                if reference_value == variant_value:
                    continue
                if not reference_value and variant_value:
                    reference_only_patterns.extend(pattern_spec['off_only'])
                    variant_only_patterns.extend(pattern_spec['on_only'])
                else:
                    reference_only_patterns.extend(pattern_spec['on_only'])
                    variant_only_patterns.extend(pattern_spec['off_only'])

            bundle = getattr(self.args, '_seed_bundle', None)
            if bundle is None:
                bundle = resolve_seed_bundle(
                    self.args,
                    iteration=int(getattr(self.args, 'run_index', 0)),
                )
            _, model, report = build_paired_models(
                lambda: model_class(reference_args).float(),
                lambda: model_class(self.args).float(),
                reference_seed=int(bundle.model_init_seed),
                variant_seed=int(bundle.extension_init_seed),
                allowed_reference_only_regexes=tuple(reference_only_patterns),
                allowed_variant_only_regexes=tuple(variant_only_patterns),
            )
            report = dict(report)
            report.update(
                {
                    'reference_config_digest': reference_args.tft_config_digest,
                    'variant_config_digest': self.args.tft_config_digest,
                    'paired_reference_spec': deepcopy(
                        paired_spec
                    ),
                    'paired_reference_spec_digest': getattr(
                        self.args, 'tft_paired_reference_spec_digest', None
                    ),
                    'resolved_reference_only_patterns': sorted(
                        set(reference_only_patterns)
                    ),
                    'resolved_variant_only_patterns': sorted(
                        set(variant_only_patterns)
                    ),
                }
            )
            self.args._paired_initialization_report = report
        else:
            model = model_class(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if str(getattr(self.args, 'loss', 'MSE')).upper() == 'QUANTILE':
            quantiles = getattr(self.args, 'tft_output_quantiles', None)
            if not getattr(self.args, 'tft_use_quantile_head', False):
                raise ValueError("loss=Quantile requires tft_use_quantile_head=True.")
            return QuantileLoss(quantiles)
        if self._tft_output_mode == 'quantile':
            raise ValueError("tft_output_mode=quantile requires loss=Quantile.")
        criterion = nn.MSELoss()
        return criterion

    def _get_aux_loss_coeff(self):
        return float(getattr(self.args, 'tft_moe_aux_loss_coeff', 0.0))

    @staticmethod
    def _cv_squared(x):
        eps = 1e-10
        if x.numel() <= 1:
            return x.new_tensor(0.0)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    @staticmethod
    def _unpack_forecast_batch(batch):
        if not isinstance(batch, (tuple, list)):
            raise TypeError("Forecast batches must be tuples/lists.")
        if len(batch) == 4:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            return (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
                None,
                None,
            )
        if len(batch) == 6:
            (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
                temporal_positions,
                temporal_valid_mask,
            ) = batch
            if not torch.is_tensor(temporal_positions):
                temporal_positions = torch.as_tensor(temporal_positions)
            if not torch.is_tensor(temporal_valid_mask):
                temporal_valid_mask = torch.as_tensor(temporal_valid_mask)
            if temporal_valid_mask.dtype != torch.bool:
                raise TypeError(
                    "TFT temporal_valid_mask batches must have boolean dtype."
                )
            return (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
                temporal_positions,
                temporal_valid_mask,
            )
        raise ValueError(
            "Forecast batches must contain 4 legacy tensors or 6 tensors "
            "including TFT positions and validity."
        )

    def _forward_model(
        self,
        batch_x,
        batch_x_mark,
        dec_inp,
        batch_y_mark,
        temporal_positions=None,
        temporal_valid_mask=None,
    ):
        coordinate_kwargs = {}
        if temporal_positions is not None or temporal_valid_mask is not None:
            if not is_tft_model(self.args):
                raise ValueError(
                    "Temporal coordinate batches are only supported by the native TFT."
                )
            if temporal_positions is None or temporal_valid_mask is None:
                raise ValueError(
                    "Both temporal_positions and temporal_valid_mask are required."
                )
            # Keep these as tensors until DataParallel scatters the batch.  Each
            # replica constructs its immutable TemporalCoordinateContract inside
            # Model.forward, avoiding an unscattered Python dataclass payload.
            coordinate_kwargs = {
                "temporal_positions": temporal_positions,
                "temporal_valid_mask": temporal_valid_mask,
            }
        if self._tft_target_positions is None:
            return self.model(
                batch_x,
                batch_x_mark,
                dec_inp,
                batch_y_mark,
                **coordinate_kwargs,
            )
        return self.model(
            batch_x,
            batch_x_mark,
            dec_inp,
            batch_y_mark,
            return_auxiliary=True,
            **coordinate_kwargs,
        )

    def _extract_outputs_and_aux(self, model_output):
        if self._tft_target_positions is None:
            return model_output, None, get_auxiliary_loss(self.model)

        declared_aux_loss = getattr(model_output, 'moe_aux_loss', None)
        if declared_aux_loss is not None:
            aux_loss = declared_aux_loss
        elif model_output.moe_importance_sum is not None:
            importance_sum = model_output.moe_importance_sum
            load_sum = getattr(model_output, 'moe_load_sum', None)
            if importance_sum.ndim == 1:
                importance_sum = importance_sum.unsqueeze(0)
            importance_sum = importance_sum.sum(dim=0)
            aux_loss = self._cv_squared(importance_sum)
            if load_sum is not None:
                if load_sum.ndim == 1:
                    load_sum = load_sum.unsqueeze(0)
                load_sum = load_sum.sum(dim=0)
                aux_loss = aux_loss + self._cv_squared(load_sum)
        else:
            aux_loss = None
        return model_output.point_full, model_output.quantile_forecast, aux_loss

    @staticmethod
    def _masked_mse(outputs, true, valid_mask):
        expanded_mask = valid_mask.unsqueeze(-1).expand_as(outputs)
        if not bool(expanded_mask.any()):
            raise ValueError("A supervised batch must contain a valid forecast token.")
        safe_outputs = torch.where(
            expanded_mask, outputs, torch.zeros_like(outputs)
        )
        safe_true = torch.where(expanded_mask, true, torch.zeros_like(true))
        return (safe_outputs - safe_true).square().sum() / expanded_mask.sum()

    def _compute_supervised_loss(
        self,
        outputs,
        true,
        criterion,
        quantile_outputs,
        valid_mask=None,
    ):
        masked = valid_mask is not None and not bool(valid_mask.all())
        if self._tft_target_positions is None:
            if isinstance(criterion, QuantileLoss):
                if quantile_outputs is None:
                    raise RuntimeError("Quantile loss selected but model did not populate last_quantile_predictions.")
                return criterion(
                    quantile_outputs,
                    true,
                    valid_mask=valid_mask if masked else None,
                )
            if masked:
                return self._masked_mse(outputs, true, valid_mask)
            return criterion(outputs, true)

        if self._tft_output_mode == 'point':
            if masked:
                return self._masked_mse(outputs, true, valid_mask)
            return criterion(outputs, true)

        if self._tft_output_mode == 'quantile':
            if quantile_outputs is None:
                raise RuntimeError("tft_output_mode=quantile requires model quantile outputs.")
            return criterion(
                quantile_outputs,
                true,
                valid_mask=valid_mask if masked else None,
            )

        if self._point_loss_coeff <= 0.0 or self._quantile_loss_coeff <= 0.0:
            raise ValueError("tft_output_mode=joint requires positive tft_point_loss_coeff and tft_quantile_loss_coeff.")
        if quantile_outputs is None:
            raise RuntimeError("tft_output_mode=joint requires model quantile outputs.")
        point_loss = (
            self._masked_mse(outputs, true, valid_mask)
            if masked
            else nn.MSELoss()(outputs, true)
        )
        quantile_loss = self._joint_quantile_criterion(
            quantile_outputs,
            true,
            valid_mask=valid_mask if masked else None,
        )
        return self._point_loss_coeff * point_loss + self._quantile_loss_coeff * quantile_loss

    def _select_targets_for_loss(self, outputs, batch_y):
        if self._tft_target_positions is None:
            f_dim = -1 if self.args.features == 'MS' else 0
            pred = outputs[:, -self.args.pred_len:, f_dim:]
            true = batch_y[:, -self.args.pred_len:, f_dim:]
        else:
            pred = outputs[:, -self.args.pred_len:, :]
            true = select_tft_truth(batch_y, self.args.pred_len, self._tft_target_positions)
            if pred.shape != true.shape:
                raise RuntimeError(
                    f"TFT prediction/target shape mismatch: pred={tuple(pred.shape)} true={tuple(true.shape)}."
                )
        return pred, true


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                (
                    batch_x,
                    batch_y,
                    batch_x_mark,
                    batch_y_mark,
                    temporal_positions,
                    temporal_valid_mask,
                ) = self._unpack_forecast_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if temporal_positions is not None:
                    temporal_positions = temporal_positions.to(self.device)
                    temporal_valid_mask = temporal_valid_mask.to(self.device)
                    forecast_valid_mask = temporal_valid_mask[:, -self.args.pred_len:]
                else:
                    forecast_valid_mask = None

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        model_output = self._forward_model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark,
                            temporal_positions, temporal_valid_mask,
                        )
                else:
                    model_output = self._forward_model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark,
                        temporal_positions, temporal_valid_mask,
                    )
                outputs, quantile_outputs, aux_loss = self._extract_outputs_and_aux(model_output)
                pred, true = self._select_targets_for_loss(outputs, batch_y)
                true = true.to(self.device)

                pred = pred.detach()
                true = true.detach()

                detached_quantiles = quantile_outputs.detach() if torch.is_tensor(quantile_outputs) else quantile_outputs
                loss = self._compute_supervised_loss(
                    pred,
                    true,
                    criterion,
                    detached_quantiles,
                    valid_mask=forecast_valid_mask,
                )
                loss = combine_primary_and_aux_loss(loss, aux_loss, self._get_aux_loss_coeff())
                if not torch.isfinite(loss):
                    print(f"[vali] Non-finite loss at batch {i}; skipping batch.")
                    continue

                total_loss.append(loss.item())
            total_loss = np.average(total_loss) if total_loss else float('nan')
        self.model.train()
        return total_loss

    def train(self, setting):
        # Reject an unreleased semantics-v2 operator before constructing data
        # loaders, creating an output directory, or taking an optimizer step.
        # Artifact-time validation remains as a second line of defence, but a
        # failed readiness contract must never leave a plausible bare
        # checkpoint behind.
        if is_tft_model(self.args):
            validate_tft_v2_artifact_readiness(self.args)

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data = None
        test_loader = None
        if self._evaluation_policy == 'legacy_val_and_test':
            test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        reproducibility_manifest = None
        if is_tft_model(self.args):
            reproducibility_manifest = self._build_reproducibility_manifest(
                setting,
                train_loader=train_loader,
                validation_loader=vali_loader,
                test_loader=test_loader,
                artifact_kind='checkpoint',
                epochs_completed=0,
            )
            self._write_reproducibility_manifest(path, reproducibility_manifest)

        time_now = time.time()

        train_steps = len(train_loader)
        checkpoint_saved_callback = None
        if is_tft_model(self.args):
            def checkpoint_saved_callback(checkpoint_path):
                # Bind semantics and the exact file hash immediately after
                # every atomic best-checkpoint replacement.  An interrupted
                # later epoch therefore does not leave a valid v2 state dict
                # masquerading as an unversioned legacy artifact.
                write_tft_semantics_metadata(
                    path,
                    self.args,
                    artifact_kind="checkpoint",
                    setting=setting,
                    checkpoint_path=checkpoint_path,
                )

        early_stopping = EarlyStopping(
            patience=self.args.patience,
            verbose=True,
            checkpoint_saved_callback=checkpoint_saved_callback,
        )

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        epochs_completed = 0
        for epoch in range(self.args.train_epochs):
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                (
                    batch_x,
                    batch_y,
                    batch_x_mark,
                    batch_y_mark,
                    temporal_positions,
                    temporal_valid_mask,
                ) = self._unpack_forecast_batch(batch)
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if temporal_positions is not None:
                    temporal_positions = temporal_positions.to(self.device)
                    temporal_valid_mask = temporal_valid_mask.to(self.device)
                    forecast_valid_mask = temporal_valid_mask[:, -self.args.pred_len:]
                else:
                    forecast_valid_mask = None

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        model_output = self._forward_model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark,
                            temporal_positions, temporal_valid_mask,
                        )
                        outputs, quantile_outputs, aux_loss = self._extract_outputs_and_aux(model_output)
                        outputs, batch_y = self._select_targets_for_loss(outputs, batch_y)
                        batch_y = batch_y.to(self.device)
                        loss = self._compute_supervised_loss(
                            outputs,
                            batch_y,
                            criterion,
                            quantile_outputs,
                            valid_mask=forecast_valid_mask,
                        )
                        loss = combine_primary_and_aux_loss(loss, aux_loss, self._get_aux_loss_coeff())
                        if not torch.isfinite(loss):
                            raise RuntimeError(f"Non-finite training loss at epoch {epoch + 1}, batch {i + 1}")
                        train_loss.append(loss.item())
                else:
                    model_output = self._forward_model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark,
                        temporal_positions, temporal_valid_mask,
                    )
                    outputs, quantile_outputs, aux_loss = self._extract_outputs_and_aux(model_output)
                    outputs, batch_y = self._select_targets_for_loss(outputs, batch_y)
                    batch_y = batch_y.to(self.device)
                    loss = self._compute_supervised_loss(
                        outputs,
                        batch_y,
                        criterion,
                        quantile_outputs,
                        valid_mask=forecast_valid_mask,
                    )
                    loss = combine_primary_and_aux_loss(loss, aux_loss, self._get_aux_loss_coeff())
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite training loss at epoch {epoch + 1}, batch {i + 1}")
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            epochs_completed = epoch + 1
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            if test_loader is not None:
                test_loss = self.vali(test_data, test_loader, criterion)
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        if is_tft_model(self.args):
            write_tft_semantics_metadata(
                path,
                self.args,
                artifact_kind="checkpoint",
                setting=setting,
                checkpoint_path=best_model_path,
            )
            load_tft_checkpoint(
                self.model,
                best_model_path,
                self.args,
                map_location=self.device,
                external_load=False,
                expected_setting=setting,
            )
        else:
            self.model.load_state_dict(torch.load(best_model_path))

        if reproducibility_manifest is not None:
            model_for_state = self.model.module if hasattr(self.model, 'module') else self.model
            reproducibility_manifest['epochs_completed'] = epochs_completed
            reproducibility_manifest['final_state_hash'] = state_dict_sha256(
                model_for_state.state_dict()
            )
            self._write_reproducibility_manifest(path, reproducibility_manifest)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        loaded_checkpoint_path = None
        if test:
            print('loading model')
            if is_tft_model(self.args):
                checkpoint_path = resolve_tft_checkpoint_path(
                    self.args.checkpoints,
                    setting,
                    self.args,
                )
                loaded_checkpoint_path = str(checkpoint_path)
                load_tft_checkpoint(
                    self.model,
                    checkpoint_path,
                    self.args,
                    map_location=self.device,
                    external_load=True,
                    expected_setting=checkpoint_path.parent.name,
                )
            else:
                checkpoint_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
                self.model.load_state_dict(torch.load(checkpoint_path))

        preds = []
        trues = []
        quantile_preds = []
        valid_forecast_masks = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                (
                    batch_x,
                    batch_y,
                    batch_x_mark,
                    batch_y_mark,
                    temporal_positions,
                    temporal_valid_mask,
                ) = self._unpack_forecast_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if temporal_positions is not None:
                    temporal_positions = temporal_positions.to(self.device)
                    temporal_valid_mask = temporal_valid_mask.to(self.device)
                    batch_forecast_valid_mask = temporal_valid_mask[
                        :, -self.args.pred_len:
                    ].detach().cpu().numpy()
                else:
                    batch_forecast_valid_mask = None

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        model_output = self._forward_model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark,
                            temporal_positions, temporal_valid_mask,
                        )
                else:
                    model_output = self._forward_model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark,
                        temporal_positions, temporal_valid_mask,
                    )
                outputs, quantile_outputs, _ = self._extract_outputs_and_aux(model_output)

                outputs, batch_y = self._select_targets_for_loss(outputs, batch_y)
                batch_y = batch_y.to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                quantile_outputs_np = quantile_outputs.detach().cpu().numpy() if torch.is_tensor(quantile_outputs) else None
                if test_data.scale and self.args.inverse:
                    if self._tft_target_positions is None:
                        shape = batch_y.shape
                        outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    else:
                        outputs = inverse_transform_selected(outputs, test_data.scaler, self._tft_target_positions)
                        batch_y = inverse_transform_selected(batch_y, test_data.scaler, self._tft_target_positions)
                        if quantile_outputs_np is not None:
                            per_quantile = []
                            for q_idx in range(quantile_outputs_np.shape[2]):
                                restored_q = inverse_transform_selected(
                                    quantile_outputs_np[:, :, q_idx, :],
                                    test_data.scaler,
                                    self._tft_target_positions,
                                )
                                per_quantile.append(restored_q)
                            quantile_outputs_np = np.stack(per_quantile, axis=2)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if batch_forecast_valid_mask is not None:
                    valid_forecast_masks.append(batch_forecast_valid_mask)
                if quantile_outputs_np is not None:
                    quantile_preds.append(quantile_outputs_np)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    plot_source_pos = -1 if self._tft_target_positions is None else self._tft_target_positions[-1]
                    gt = np.concatenate((input[0, :, plot_source_pos], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, plot_source_pos], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        valid_forecast_mask = None
        if valid_forecast_masks:
            valid_forecast_mask = np.concatenate(
                valid_forecast_masks, axis=0
            ).reshape(-1, self.args.pred_len)
            if valid_forecast_mask.shape != preds.shape[:2]:
                raise RuntimeError(
                    "Forecast validity masks do not align with test predictions."
                )
        print('test shape:', preds.shape, trues.shape)

        metric_preds = preds
        metric_trues = trues
        if valid_forecast_mask is not None and not bool(valid_forecast_mask.all()):
            metric_mask = np.broadcast_to(
                valid_forecast_mask[..., None], preds.shape
            )
            metric_preds = preds[metric_mask]
            metric_trues = trues[metric_mask]

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if is_tft_model(self.args):
            write_tft_semantics_metadata(
                folder_path,
                self.args,
                artifact_kind="result",
                filename=TFT_RESULT_METADATA_FILENAME,
                setting=setting,
            )

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                if valid_forecast_mask is None:
                    x = preds[i].reshape(-1, 1)
                    y = trues[i].reshape(-1, 1)
                else:
                    sample_mask = np.broadcast_to(
                        valid_forecast_mask[i, :, None], preds[i].shape
                    )
                    x = preds[i][sample_mask].reshape(-1, 1)
                    y = trues[i][sample_mask].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(metric_preds, metric_trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        quantile_summary = None
        if quantile_preds and self._quantile_levels is not None:
            quantile_preds = np.concatenate(quantile_preds, axis=0)
            quantile_preds = quantile_preds.reshape(-1, quantile_preds.shape[-3], quantile_preds.shape[-2], quantile_preds.shape[-1])
            quantile_metric_preds = quantile_preds
            quantile_metric_trues = trues
            if valid_forecast_mask is not None and not bool(valid_forecast_mask.all()):
                quantile_metric_preds = quantile_preds[valid_forecast_mask][:, None, :, :]
                quantile_metric_trues = trues[valid_forecast_mask][:, None, :]
            quantile_summary = quantile_metric(
                quantile_metric_preds,
                quantile_metric_trues,
                self._quantile_levels,
            )
            f.write(
                ', pinball:{pinball}, coverage:{coverage}, interval_width:{interval_width}, crossing_rate:{crossing_rate}'.format(
                    **quantile_summary
                )
            )
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        if valid_forecast_mask is not None:
            np.save(folder_path + 'valid_mask.npy', valid_forecast_mask)
        if quantile_summary is not None:
            with open(folder_path + 'quantile_metrics.json', 'w', encoding='utf-8') as quantile_file:
                json.dump(quantile_summary, quantile_file, indent=2, sort_keys=True)
            np.save(folder_path + 'quantile_pred.npy', quantile_preds)

        if is_tft_model(self.args):
            if self._last_reproducibility_manifest is None:
                result_reproducibility = self._build_reproducibility_manifest(
                    setting,
                    test_loader=test_loader,
                    artifact_kind='result',
                    final_state_hash=state_dict_sha256(
                        (self.model.module if hasattr(self.model, 'module') else self.model).state_dict()
                    ),
                )
            else:
                result_reproducibility = deepcopy(
                    self._last_reproducibility_manifest
                )
                result_reproducibility['artifact_kind'] = 'result'
                result_reproducibility['setting'] = setting
                result_reproducibility['loaders']['test'] = (
                    self._loader_reproducibility(test_loader)
                )
                fold_hashes = {
                    name: metadata.get('fold_manifest_hash')
                    for name, metadata in result_reproducibility['loaders'].items()
                    if metadata is not None
                }
                result_reproducibility['fold_manifest_hash'] = stable_json_hash(
                    fold_hashes
                )
            result_reproducibility['source_checkpoint_path'] = loaded_checkpoint_path
            self._write_reproducibility_manifest(
                folder_path,
                result_reproducibility,
            )

        return
