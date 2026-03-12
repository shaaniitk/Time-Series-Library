import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models import TemporalFusionTransformer as tsl_tft

try:
    from models import TFT_Nixtla as nixtla_tft
    NIXTLA_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover
    nixtla_tft = None
    NIXTLA_IMPORT_ERROR = e


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_tsl_config():
    return SimpleNamespace(
        task_name="long_term_forecast",
        data="custom_tft_comp_test",
        seq_len=24,
        label_len=12,
        pred_len=4,
        enc_in=16,
        dec_in=4,
        c_out=4,
        d_model=48,
        n_heads=4,
        dropout=0.1,
        embed="timeF",
        freq="h",
        e_layers=1,
        tft_use_swiglu=True,
        tft_full_attention=True,
        tft_dual_attention_fusion=True,
        tft_cross_variable_mixing=True,
        tft_vsn_residual_bypass=True,
        tft_allow_custom_known=True,
        tft_known_len=12,
        tft_known_max_channels=64,
        tft_observed_pos=list(range(16)),
        tft_static_pos=[],
        tft_target_pos=[0, 1, 2, 3],
    )


def make_tsl_dataset(cfg, n_samples=48):
    x_enc = torch.randn(n_samples, cfg.seq_len, cfg.enc_in)
    x_mark_enc = torch.randn(n_samples, cfg.seq_len, cfg.tft_known_len)
    x_dec = torch.randn(n_samples, cfg.label_len + cfg.pred_len, cfg.c_out)
    x_mark_dec = torch.randn(n_samples, cfg.label_len + cfg.pred_len, cfg.tft_known_len)

    # Learnable target from both history and known future covariates.
    recent = x_enc[:, -cfg.pred_len:, : cfg.c_out]
    futr = x_mark_dec[:, -cfg.pred_len:, : cfg.c_out]
    y = 0.65 * recent + 0.35 * futr
    y = y + 0.01 * torch.randn_like(y)

    return TensorDataset(x_enc, x_mark_enc, x_dec, x_mark_dec, y)


def train_tsl_once(model, loader, cfg, lr=3e-3, epochs=4):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses = []
    model.train()
    for _ in range(epochs):
        running = 0.0
        for x_enc, x_mark_enc, x_dec, x_mark_dec, y in loader:
            opt.zero_grad()
            out_full = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            pred = out_full[:, -cfg.pred_len :, :]
            loss = criterion(pred, y)
            loss.backward()
            opt.step()
            running += loss.item()
        losses.append(running / len(loader))
    return losses


@unittest.skipIf(nixtla_tft is None, f"Nixtla import failed: {NIXTLA_IMPORT_ERROR}")
class TestTFTComprehensive(unittest.TestCase):
    def setUp(self):
        set_seed(42)

    def test_tsl_component_contracts_and_payload(self):
        cfg = build_tsl_config()

        emb = tsl_tft.TFTCustomKnownEmbedding(cfg.d_model, max_channels=32)
        known = torch.randn(3, 10, 12)
        known_out = emb(known)
        self.assertEqual(tuple(known_out.shape), (3, 10, 12, cfg.d_model))

        vsn = tsl_tft.VariableSelectionNetwork(
            d_model=cfg.d_model,
            variable_num=8,
            dropout=0.1,
            use_swiglu=True,
            cross_variable_mixing=True,
            n_heads=cfg.n_heads,
            residual_bypass=True,
        )
        x = torch.randn(2, cfg.seq_len, 8, cfg.d_model)
        context = torch.randn(2, cfg.d_model)
        selected, weights = vsn(x, context=context, return_weights=True)
        self.assertEqual(tuple(selected.shape), (2, cfg.seq_len, cfg.d_model))
        self.assertEqual(tuple(weights.shape), (2, cfg.seq_len, 8))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-5))

        model = tsl_tft.Model(cfg)
        ds = make_tsl_dataset(cfg, n_samples=8)
        x_enc, x_mark_enc, x_dec, x_mark_dec, _ = ds[0]
        payload = model(
            x_enc.unsqueeze(0),
            x_mark_enc.unsqueeze(0),
            x_dec.unsqueeze(0),
            x_mark_dec.unsqueeze(0),
            return_interpretation=True,
        )
        self.assertIn("predictions", payload)
        self.assertIn("predictions_full", payload)
        self.assertIn("attention_weights", payload)
        self.assertIn("attention_weights_full", payload)
        self.assertIn("attention_fusion_alpha", payload)
        self.assertEqual(tuple(payload["predictions"].shape), (1, cfg.pred_len, cfg.c_out))
        self.assertEqual(tuple(payload["predictions_full"].shape), (1, cfg.seq_len + cfg.pred_len, cfg.c_out))

    def test_tsl_model_learns_structured_signal(self):
        cfg = build_tsl_config()
        model = tsl_tft.Model(cfg)
        ds = make_tsl_dataset(cfg, n_samples=48)
        loader = DataLoader(ds, batch_size=8, shuffle=True)

        losses = train_tsl_once(model, loader, cfg, lr=3e-3, epochs=4)
        self.assertGreater(losses[0], losses[-1], "TSL TFT did not improve training loss.")
        self.assertLess(losses[-1], losses[0] * 0.90, "TSL TFT loss reduction is too weak for a learnable target.")

        # Ensure important gates receive gradients at least once.
        gate_grad = model.history_vsn.residual_gate.grad
        self.assertIsNotNone(gate_grad)
        self.assertGreater(gate_grad.abs().sum().item(), 0.0)

    def test_nixtla_components_and_model_behavior(self):
        cfg = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=24,
            label_len=12,
            pred_len=4,
            enc_in=4,
            dec_in=4,
            c_out=4,
            d_model=32,
            n_heads=4,
            dropout=0.1,
            batch_size=8,
            freq="h",
        )

        # Component contract: dual-attention payload includes both branches and fusion alpha.
        dual = nixtla_tft.DualInterpretableMultiHeadAttention(
            n_head=cfg.n_heads,
            hidden_size=cfg.d_model,
            example_length=cfg.seq_len + cfg.pred_len,
            attn_dropout=cfg.dropout,
            dropout=cfg.dropout,
        )
        x = torch.randn(2, cfg.seq_len + cfg.pred_len, cfg.d_model)
        out, payload = dual(x, mask_future_timesteps=True)
        self.assertEqual(tuple(out.shape), tuple(x.shape))
        self.assertEqual(len(payload), 3)
        int_w, full_w, alpha = payload
        self.assertEqual(int_w.ndim, 4)
        self.assertEqual(full_w.ndim, 4)
        self.assertTrue(0.0 <= float(alpha) <= 1.0)

        model = nixtla_tft.Model(cfg)
        self.assertTrue(hasattr(model, "cross_channel_mixer"))

        # Integrity checks for temporal feature dimensions should trigger ValueError.
        wrong_mark = torch.randn(2, cfg.seq_len, model.time_features_dim + 1)
        x_enc = torch.randn(2, cfg.seq_len, cfg.c_out)
        x_dec = torch.randn(2, cfg.label_len + cfg.pred_len, cfg.c_out)
        x_mark_dec = torch.randn(2, cfg.label_len + cfg.pred_len, model.time_features_dim)
        with self.assertRaises(ValueError):
            _ = model(x_enc, wrong_mark, x_dec, x_mark_dec)

        # Short learnability and gradient-flow check on a structured target.
        x_mark_enc = torch.randn(32, cfg.seq_len, model.time_features_dim)
        x_mark_dec = torch.randn(32, cfg.label_len + cfg.pred_len, model.time_features_dim)
        x_enc = torch.randn(32, cfg.seq_len, cfg.c_out)
        x_dec = torch.randn(32, cfg.label_len + cfg.pred_len, cfg.c_out)

        recent = x_enc[:, -cfg.pred_len :, :]
        futr = x_mark_dec[:, -cfg.pred_len :, : cfg.c_out]
        y = 0.7 * recent + 0.3 * futr

        ds = TensorDataset(x_enc, x_mark_enc, x_dec, x_mark_dec, y)
        loader = DataLoader(ds, batch_size=8, shuffle=True)

        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        criterion = nn.MSELoss()
        losses = []
        model.train()
        for _ in range(4):
            running = 0.0
            for bx, bxm, bd, bdm, by in loader:
                opt.zero_grad()
                pred = model(bx, bxm, bd, bdm)
                loss = criterion(pred, by)
                loss.backward()
                opt.step()
                running += loss.item()
            losses.append(running / len(loader))

        self.assertGreater(losses[0], losses[-1], "Nixtla TFT did not improve training loss.")
        self.assertLess(losses[-1], losses[0] * 0.92, "Nixtla TFT loss reduction is too weak for a learnable target.")

        self.assertIsNotNone(model.cross_channel_mixer.weight.grad)
        self.assertGreater(model.cross_channel_mixer.weight.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
