import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

from exp.exp_long_term_forecasting import (
    TFT_REPRODUCIBILITY_FILENAME,
    Exp_Long_Term_Forecast,
)
from utils.losses import QuantileLoss
from utils.tft_config import (
    TFT_CHECKPOINT_METADATA_FILENAME,
    TFT_RESULT_METADATA_FILENAME,
    write_tft_semantics_metadata,
)
from utils.tft_interpretation import export_tft_interpretation_summary, summarize_tft_interpretation
from utils.tft_synthetic import make_multiscale_tft_tensors


def set_seed(seed: int = 123):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_tft_args(checkpoint_dir):
    return SimpleNamespace(
        task_name="long_term_forecast",
        is_training=1,
        model_id="tft-exp-smoke",
        model="TemporalFusionTransformer",
        data="custom_tft_exp_smoke",
        root_path=".",
        data_path="unused.csv",
        features="M",
        target="OT",
        freq="h",
        checkpoints=str(checkpoint_dir),
        seq_len=24,
        label_len=12,
        pred_len=4,
        seasonal_patterns="Monthly",
        inverse=False,
        mask_rate=0.25,
        anomaly_ratio=0.25,
        expand=2,
        d_conv=4,
        top_k=5,
        num_kernels=6,
        enc_in=16,
        dec_in=4,
        c_out=4,
        d_model=32,
        n_heads=4,
        e_layers=1,
        d_layers=1,
        d_ff=128,
        moving_avg=25,
        factor=1,
        distil=True,
        dropout=0.1,
        embed="timeF",
        activation="gelu",
        channel_independence=1,
        decomp_method="moving_avg",
        use_norm=1,
        down_sampling_layers=0,
        down_sampling_window=1,
        down_sampling_method=None,
        seg_len=96,
        num_workers=0,
        itr=1,
        train_epochs=2,
        batch_size=4,
        patience=2,
        learning_rate=1e-3,
        des="test",
        loss="MSE",
        lradj="type1",
        use_amp=False,
        use_gpu=False,
        gpu=0,
        gpu_type="cuda",
        use_multi_gpu=False,
        devices="0",
        p_hidden_dims=[128, 128],
        p_hidden_layers=2,
        use_dtw=False,
        augmentation_ratio=0,
        seed=2,
        jitter=False,
        scaling=False,
        permutation=False,
        randompermutation=False,
        magwarp=False,
        timewarp=False,
        windowslice=False,
        windowwarp=False,
        rotation=False,
        spawner=False,
        dtwwarp=False,
        shapedtwwarp=False,
        wdba=False,
        discdtw=False,
        discsdtw=False,
        extra_tag="",
        patch_len=16,
        node_dim=10,
        gcn_depth=2,
        gcn_dropout=0.3,
        propalpha=0.3,
        conv_channel=32,
        skip_channel=32,
        individual=False,
        tft_extension_semantics_version=1,
        tft_allow_legacy_extension_checkpoint=True,
        tft_observed_pos=list(range(16)),
        tft_static_pos=[],
        tft_target_pos=[0, 1, 2, 3],
        tft_use_swiglu=True,
        tft_full_attention=True,
        tft_cross_variable_mixing=True,
        tft_allow_custom_known=True,
        tft_vsn_residual_bypass=True,
        tft_dual_attention_fusion=True,
        tft_use_explicit_cross_attention=True,
        tft_cross_attention_type="interpretable",
        tft_attention_position_bias="rope",
        tft_attention_backend="sdpa",
        tft_rope_base=10000.0,
        tft_alibi_scale=1.0,
        tft_use_revin=True,
        tft_revin_affine=True,
        tft_use_quantile_head=True,
        tft_output_quantiles=[0.1, 0.5, 0.9],
        tft_use_lag_attention=True,
        tft_lag_scales=[1, 2, 4],
        tft_temporal_backbone="hybrid_tcn_lstm",
        tft_temporal_backbone_layers=2,
        tft_temporal_kernel_size=3,
        tft_temporal_hidden_size=32,
        tft_use_higher_order=True,
        tft_interaction_order=2,
        tft_interaction_rank=8,
        tft_use_regime_moe=True,
        tft_num_regimes=3,
        tft_num_moe_experts=4,
        tft_moe_top_k=2,
        tft_moe_hidden_size=32,
        tft_moe_noise_epsilon=1e-2,
        tft_moe_aux_loss_coeff=0.05,
        tft_payload_stack_layers=True,
        tft_known_len=12,
        tft_known_max_channels=32,
        tft_known_feature_names=[f"known_{i}" for i in range(12)],
        alpha=0.1,
        top_p=0.5,
        pos=1,
    )


def make_exp_dataset(args, n_samples=12):
    tensors = make_multiscale_tft_tensors(
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        enc_in=args.enc_in,
        c_out=args.c_out,
        known_len=args.tft_known_len,
        n_samples=n_samples,
        noise_std=0.01,
    )
    return TensorDataset(tensors["x_enc"], tensors["batch_y"], tensors["x_mark_enc"], tensors["x_mark_dec"])


class TinyLongForecastExp(Exp_Long_Term_Forecast):
    def __init__(self, args, datasets):
        self._datasets = datasets
        super(TinyLongForecastExp, self).__init__(args)

    def _get_data(self, flag):
        return self._datasets[flag]


class TestTFTInterpretationAndExp(unittest.TestCase):
    def setUp(self):
        set_seed(123)

    def test_interpretation_summary_and_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_tft_args(tmpdir)
            model = TinyLongForecastExp(args, {}).model
            tensors = make_multiscale_tft_tensors(
                seq_len=args.seq_len,
                label_len=args.label_len,
                pred_len=args.pred_len,
                enc_in=args.enc_in,
                c_out=args.c_out,
                known_len=args.tft_known_len,
                n_samples=2,
                noise_std=0.0,
            )
            x_enc = tensors["x_enc"]
            x_mark_enc = tensors["x_mark_enc"]
            x_dec = tensors["x_dec"]
            x_mark_dec = tensors["x_mark_dec"]
            payload = model(x_enc, x_mark_enc, x_dec, x_mark_dec, return_interpretation=True)

            summary = summarize_tft_interpretation(payload, top_k=2)
            self.assertEqual(summary["prediction_shape"], [2, args.pred_len, args.c_out])
            self.assertEqual(summary["quantile_prediction_shape"], [2, args.pred_len, 3, args.c_out])
            self.assertEqual(summary["position_bias_type"], "rope")
            self.assertEqual(summary["temporal_backbone_type"], "hybrid_tcn_lstm")
            self.assertEqual(summary["attention_backend_config"], "sdpa")
            self.assertEqual(summary["attention_backend_used"], "exact")
            self.assertTrue(summary["has_quantile_predictions"])
            self.assertTrue(summary["has_graph_attention"])
            self.assertTrue(summary["has_cross_attention"])
            self.assertTrue(summary["has_lag_attention"])
            self.assertTrue(summary["has_higher_order"])
            self.assertTrue(summary["has_regime_moe"])
            self.assertEqual(len(summary["history_feature_names"]), len(summary["observed_feature_names"]) + len(summary["known_feature_names"]))
            self.assertEqual(summary["future_feature_names"], args.tft_known_feature_names)
            self.assertEqual(summary["history_vsn_axis"], "history_features")
            self.assertEqual(summary["future_vsn_axis"], "future_known_features")
            self.assertIn("feature_name", summary["top_history_vsn_entries"][0])
            self.assertIn("feature_name", summary["top_future_vsn_entries"][0])
            self.assertFalse(summary["interpretation_flags"]["is_canonical_vsn_attribution"])
            self.assertTrue(summary["interpretation_flags"]["uses_graph_pre_mixing"])
            self.assertTrue(summary["interpretation_flags"]["uses_vsn_bypass"])
            self.assertTrue(summary["interpretation_flags"]["uses_noninterpretable_attention_branch"])
            self.assertTrue(summary["interpretation_flags"]["routing_is_detached"])

            output_path = Path(tmpdir) / "interpretation_summary.json"
            exported = export_tft_interpretation_summary(payload, output_path, top_k=2)
            self.assertEqual(summary["prediction_shape"], exported["prediction_shape"])
            self.assertTrue(output_path.exists())
            on_disk = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["decoder_num_layers"], args.e_layers)
            self.assertEqual(on_disk["future_feature_names"], args.tft_known_feature_names)

    def test_static_interpretation_uses_feature_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_tft_args(tmpdir)
            args.enc_in = 4
            args.c_out = 1
            args.dec_in = 1
            args.tft_feature_names = ["target", "dynamic_aux", "dynamic_aux_2", "static_entity"]
            args.tft_observed_pos = [0, 1, 2]
            args.tft_static_pos = [3]
            args.tft_target_pos = [0]
            model = TinyLongForecastExp(args, {}).model
            x_enc = torch.zeros(2, args.seq_len, args.enc_in)
            x_enc[:, :, 0] = torch.linspace(0.0, 1.0, steps=args.seq_len)
            x_enc[:, :, 1] = torch.linspace(1.0, 2.0, steps=args.seq_len)
            x_enc[:, :, 2] = torch.linspace(2.0, 3.0, steps=args.seq_len)
            x_enc[0, :, 3] = 5.0
            x_enc[1, :, 3] = 50.0
            x_mark_enc = torch.zeros(2, args.seq_len, args.tft_known_len)
            x_dec = torch.zeros(2, args.label_len + args.pred_len, args.c_out)
            x_mark_dec = torch.zeros(2, args.label_len + args.pred_len, args.tft_known_len)

            payload = model(x_enc, x_mark_enc, x_dec, x_mark_dec, return_interpretation=True)
            summary = summarize_tft_interpretation(payload, top_k=2)

            self.assertEqual(summary["static_feature_names"], ["static_entity"])
            self.assertIn("c_s", summary["top_static_vsn_entries"])
            self.assertEqual(summary["top_static_vsn_entries"]["c_s"][0]["feature_name"], "static_entity")
            self.assertIn("c_s", summary["static_graph_attention_mean"])

    def test_canonical_profile_sets_interpretation_flags_and_named_vsn_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_tft_args(tmpdir)
            args.enc_in = 5
            args.c_out = 1
            args.dec_in = 1
            args.tft_feature_names = ["target", "obs_aux", "static_id", "known_hour", "known_holiday"]
            args.tft_observed_pos = [0, 1]
            args.tft_static_pos = [2]
            args.tft_target_pos = [0]
            args.tft_known_len = 2
            args.tft_known_feature_names = ["known_hour", "known_holiday"]
            args.tft_profile = "canonical"
            args.tft_use_swiglu = False
            args.tft_cross_variable_mixing = False
            args.tft_vsn_residual_bypass = False
            args.tft_full_attention = False
            args.tft_dual_attention_fusion = False
            args.tft_use_explicit_cross_attention = False
            args.tft_attention_position_bias = "none"
            args.tft_attention_backend = "exact"
            args.tft_use_revin = False
            args.tft_revin_affine = False
            args.tft_use_higher_order = False
            args.tft_use_regime_moe = False
            args.tft_use_lag_attention = False
            args.tft_use_fft_branch = False
            args.tft_covariate_reattention = False
            args.tft_payload_stack_layers = False
            args.tft_temporal_backbone = "lstm"
            args.tft_temporal_backbone_layers = 1
            model = TinyLongForecastExp(args, {}).model

            x_enc = torch.randn(2, args.seq_len, args.enc_in)
            x_enc[:, :, 2] = x_enc[:, :1, 2]
            x_mark_enc = torch.randn(2, args.seq_len, args.tft_known_len)
            x_dec = torch.zeros(2, args.label_len + args.pred_len, args.c_out)
            x_mark_dec = torch.randn(2, args.label_len + args.pred_len, args.tft_known_len)

            payload = model(x_enc, x_mark_enc, x_dec, x_mark_dec, return_interpretation=True)
            summary = summarize_tft_interpretation(payload, top_k=2)

            self.assertTrue(summary["interpretation_flags"]["is_canonical_vsn_attribution"])
            self.assertFalse(summary["interpretation_flags"]["uses_graph_pre_mixing"])
            self.assertFalse(summary["interpretation_flags"]["uses_vsn_bypass"])
            self.assertFalse(summary["interpretation_flags"]["uses_noninterpretable_attention_branch"])
            self.assertFalse(summary["interpretation_flags"]["routing_is_detached"])
            self.assertEqual(summary["history_feature_names"], ["target", "obs_aux", "known_hour", "known_holiday"])
            self.assertEqual(summary["future_feature_names"], ["known_hour", "known_holiday"])
            self.assertIn(summary["top_history_vsn_entries"][0]["feature_name"], summary["history_feature_names"])
            self.assertIn(summary["top_future_vsn_entries"][0]["feature_name"], summary["future_feature_names"])

    def test_quantile_loss_is_selectable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_tft_args(tmpdir)
            args.loss = "Quantile"
            exp = TinyLongForecastExp(args, {})
            criterion = exp._select_criterion()
            self.assertIsInstance(criterion, QuantileLoss)

    def test_exp_long_term_train_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_tft_args(tmpdir)
            train_ds = make_exp_dataset(args, n_samples=12)
            val_ds = make_exp_dataset(args, n_samples=8)
            test_ds = make_exp_dataset(args, n_samples=8)
            datasets = {
                "train": (train_ds, DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)),
                "val": (val_ds, DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)),
                "test": (test_ds, DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)),
            }
            exp = TinyLongForecastExp(args, datasets)
            trained_model = exp.train("tiny_tft_exp_smoke")
            self.assertIsNotNone(trained_model)
            self.assertTrue(Path(tmpdir, "tiny_tft_exp_smoke", "checkpoint.pth").exists())
            self.assertTrue(
                Path(
                    tmpdir,
                    "tiny_tft_exp_smoke",
                    TFT_CHECKPOINT_METADATA_FILENAME,
                ).exists()
            )
            reproducibility_path = Path(
                tmpdir,
                "tiny_tft_exp_smoke",
                TFT_REPRODUCIBILITY_FILENAME,
            )
            self.assertTrue(reproducibility_path.exists())
            reproducibility = json.loads(
                reproducibility_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                reproducibility["evaluation_policy"],
                "legacy_val_and_test",
            )
            self.assertIsNotNone(reproducibility["initial_state_hash"])
            self.assertIsNotNone(reproducibility["final_state_hash"])
            self.assertIn("data_order_seed", reproducibility["seed_bundle"])
            self.assertIsNotNone(getattr(exp.model, "last_moe_aux_loss", None))

    def test_validation_only_training_never_constructs_test_loader(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_tft_args(tmpdir)
            args.evaluation_policy = "validation_only"
            train_ds = make_exp_dataset(args, n_samples=8)
            val_ds = make_exp_dataset(args, n_samples=4)
            datasets = {
                "train": (train_ds, DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)),
                "val": (val_ds, DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)),
            }
            exp = TinyLongForecastExp(args, datasets)
            exp.train("tiny_tft_validation_only")
            manifest = json.loads(
                Path(
                    tmpdir,
                    "tiny_tft_validation_only",
                    TFT_REPRODUCIBILITY_FILENAME,
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["evaluation_policy"], "validation_only")
            self.assertIsNone(manifest["loaders"]["test"])

    def test_quantile_test_writes_calibration_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_tft_args(tmpdir)
            args.train_epochs = 1
            test_ds = make_exp_dataset(args, n_samples=8)
            test_ds.scale = False
            datasets = {
                "train": (test_ds, DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)),
                "val": (test_ds, DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)),
                "test": (test_ds, DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)),
            }
            exp = TinyLongForecastExp(args, datasets)
            cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                exp.test("tiny_tft_quantile_metrics", test=0)
            finally:
                os.chdir(cwd)

            results_dir = Path(tmpdir) / "results" / "tiny_tft_quantile_metrics"
            quantile_metrics_path = results_dir / "quantile_metrics.json"
            quantile_pred_path = results_dir / "quantile_pred.npy"
            result_metadata_path = results_dir / TFT_RESULT_METADATA_FILENAME
            reproducibility_path = results_dir / TFT_REPRODUCIBILITY_FILENAME
            self.assertTrue(quantile_metrics_path.exists())
            self.assertTrue(quantile_pred_path.exists())
            self.assertTrue(result_metadata_path.exists())
            self.assertTrue(reproducibility_path.exists())
            summary = json.loads(quantile_metrics_path.read_text(encoding="utf-8"))
            self.assertIn("pinball", summary)
            self.assertIn("coverage", summary)
            self.assertIn("interval_width", summary)
            self.assertIn("crossing_rate", summary)

    def test_standalone_test_respects_custom_checkpoint_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_tft_args(tmpdir)
            test_ds = make_exp_dataset(args, n_samples=4)
            test_ds.scale = False
            datasets = {
                "train": (test_ds, DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)),
                "val": (test_ds, DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)),
                "test": (test_ds, DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)),
            }
            exp = TinyLongForecastExp(args, datasets)
            setting = "custom_checkpoint_root"
            checkpoint_dir = Path(args.checkpoints) / setting
            checkpoint_dir.mkdir(parents=True)
            torch.save(exp.model.state_dict(), checkpoint_dir / "checkpoint.pth")
            write_tft_semantics_metadata(
                checkpoint_dir,
                exp.args,
                artifact_kind="checkpoint",
                setting=setting,
            )

            cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                exp.test(setting, test=1)
            finally:
                os.chdir(cwd)

    def test_standalone_test_resolves_historical_unversioned_v1_setting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_tft_args(tmpdir)
            test_ds = make_exp_dataset(args, n_samples=4)
            test_ds.scale = False
            datasets = {
                "train": (test_ds, DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)),
                "val": (test_ds, DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)),
                "test": (test_ds, DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)),
            }
            exp = TinyLongForecastExp(args, datasets)
            versioned_setting = "legacy_matrix_case_tsv1_td0123456789ab"
            historical_setting = "legacy_matrix_case_td0123456789ab"
            checkpoint_dir = Path(args.checkpoints) / historical_setting
            checkpoint_dir.mkdir(parents=True)
            torch.save(exp.model.state_dict(), checkpoint_dir / "checkpoint.pth")

            cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                exp.test(versioned_setting, test=1)
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
