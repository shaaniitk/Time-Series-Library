import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
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
        tft_observed_pos=list(range(16)),
        tft_static_pos=[],
        tft_target_pos=[0, 1, 2, 3],
        tft_use_swiglu=True,
        tft_full_attention=True,
        tft_cross_variable_mixing=True,
        tft_allow_custom_known=True,
        tft_vsn_residual_bypass=True,
        tft_dual_attention_fusion=True,
        tft_use_lag_attention=True,
        tft_lag_scales=[1, 2, 4],
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
            self.assertTrue(summary["has_graph_attention"])
            self.assertTrue(summary["has_lag_attention"])
            self.assertTrue(summary["has_higher_order"])
            self.assertTrue(summary["has_regime_moe"])

            output_path = Path(tmpdir) / "interpretation_summary.json"
            exported = export_tft_interpretation_summary(payload, output_path, top_k=2)
            self.assertEqual(summary["prediction_shape"], exported["prediction_shape"])
            self.assertTrue(output_path.exists())
            on_disk = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["decoder_num_layers"], args.e_layers)

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
            self.assertIsNotNone(getattr(exp.model, "last_moe_aux_loss", None))


if __name__ == "__main__":
    unittest.main()