import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from layers.DynamicGraph import DynamicGraphLearner
from layers.TemporalFusion_layers import (
    GatedDilatedTemporalBackbone,
    HigherOrderInteractionBlock,
    HybridTemporalBackbone,
    InterpretableCrossAttention,
    MultiScaleLagAttention,
    PositionalMultiHeadAttention,
    RegimeAwareSparseMoE,
    build_causal_mask,
)
from models import TemporalFusionTransformer as tsl_tft
from utils.tft_synthetic import make_multiscale_tft_dataset
from utils.losses import QuantileLoss
from utils.tools import combine_primary_and_aux_loss, get_auxiliary_loss

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
        e_layers=2,
        tft_use_swiglu=True,
        tft_full_attention=True,
        tft_dual_attention_fusion=True,
        tft_use_explicit_cross_attention=True,
        tft_cross_attention_type="full",
        tft_attention_position_bias="none",
        tft_attention_backend="exact",
        tft_rope_base=10000.0,
        tft_alibi_scale=1.0,
        tft_use_revin=True,
        tft_revin_affine=True,
        tft_use_quantile_head=False,
        tft_output_quantiles=[0.1, 0.5, 0.9],
        tft_use_lag_attention=True,
        tft_lag_scales=[1, 2, 4],
        tft_temporal_backbone="lstm",
        tft_temporal_backbone_layers=3,
        tft_temporal_kernel_size=3,
        tft_temporal_hidden_size=64,
        tft_use_higher_order=True,
        tft_interaction_order=2,
        tft_interaction_rank=12,
        tft_use_regime_moe=True,
        tft_num_regimes=3,
        tft_num_moe_experts=4,
        tft_moe_top_k=2,
        tft_moe_hidden_size=64,
        tft_moe_noise_epsilon=1e-2,
        tft_moe_aux_loss_coeff=0.05,
        tft_payload_stack_layers=True,
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
    return make_multiscale_tft_dataset(
        seq_len=cfg.seq_len,
        label_len=cfg.label_len,
        pred_len=cfg.pred_len,
        enc_in=cfg.enc_in,
        c_out=cfg.c_out,
        known_len=cfg.tft_known_len,
        n_samples=n_samples,
        noise_std=0.01,
    )


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
            aux_loss = getattr(model, "last_moe_aux_loss", None)
            if torch.is_tensor(aux_loss):
                loss = loss + getattr(cfg, "tft_moe_aux_loss_coeff", 0.0) * aux_loss
            loss.backward()
            opt.step()
            running += loss.item()
        losses.append(running / len(loader))
    return losses


@unittest.skipIf(nixtla_tft is None, f"Nixtla import failed: {NIXTLA_IMPORT_ERROR}")
class TestTFTComprehensive(unittest.TestCase):
    def setUp(self):
        set_seed(42)

    def test_higher_order_interaction_component(self):
        cfg = build_tsl_config()
        interaction_block = HigherOrderInteractionBlock(
            d_model=cfg.d_model,
            interaction_order=cfg.tft_interaction_order,
            interaction_rank=cfg.tft_interaction_rank,
            dropout=0.0,
        )
        x = torch.randn(2, cfg.seq_len + cfg.pred_len, cfg.d_model)
        out, payload = interaction_block(x, return_payload=True)
        self.assertEqual(tuple(out.shape), tuple(x.shape))
        self.assertEqual(tuple(payload["interaction_contribution"].shape), tuple(x.shape))
        self.assertEqual(tuple(payload["interaction_gates"].shape), (2, cfg.seq_len + cfg.pred_len, cfg.tft_interaction_order))
        self.assertTrue(torch.isfinite(payload["interaction_contribution"]).all())
        self.assertTrue(torch.allclose(payload["interaction_gates"].sum(dim=-1), torch.ones_like(payload["interaction_gates"].sum(dim=-1)), atol=1e-6))

    def test_multiscale_lag_attention_component(self):
        cfg = build_tsl_config()
        lag_attention = MultiScaleLagAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            lag_scales=cfg.tft_lag_scales,
            dropout=0.0,
        )
        x = torch.randn(2, cfg.seq_len + cfg.pred_len, cfg.d_model)
        out, payload = lag_attention(x, return_attention=True)
        self.assertEqual(tuple(out.shape), tuple(x.shape))
        self.assertEqual(tuple(payload["lag_attention"].shape), (2, cfg.n_heads, cfg.seq_len + cfg.pred_len, cfg.seq_len + cfg.pred_len, len(cfg.tft_lag_scales)))
        self.assertEqual(tuple(payload["lag_scale_weights"].shape), (len(cfg.tft_lag_scales),))
        self.assertTrue(torch.allclose(payload["lag_scale_weights"].sum(), torch.tensor(1.0), atol=1e-6))

        future_mask = torch.triu(torch.ones(cfg.seq_len + cfg.pred_len, cfg.seq_len + cfg.pred_len, dtype=torch.bool), diagonal=1)
        masked_values = payload["lag_attention"].masked_select(future_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1))
        self.assertTrue(torch.allclose(masked_values, torch.zeros_like(masked_values), atol=1e-6))

    def test_gated_temporal_backbone_component(self):
        cfg = build_tsl_config()
        backbone = GatedDilatedTemporalBackbone(
            d_model=cfg.d_model,
            num_layers=cfg.tft_temporal_backbone_layers,
            kernel_size=cfg.tft_temporal_kernel_size,
            hidden_size=cfg.tft_temporal_hidden_size,
            dropout=0.0,
        )
        x = torch.randn(2, cfg.seq_len + cfg.pred_len, cfg.d_model)
        out = backbone(x)
        self.assertEqual(tuple(out.shape), tuple(x.shape))
        self.assertEqual(len(backbone.blocks), cfg.tft_temporal_backbone_layers)
        self.assertTrue(torch.isfinite(out).all())

    def test_hybrid_temporal_backbone_component(self):
        cfg = build_tsl_config()
        backbone = HybridTemporalBackbone(
            d_model=cfg.d_model,
            num_layers=cfg.tft_temporal_backbone_layers,
            kernel_size=cfg.tft_temporal_kernel_size,
            hidden_size=cfg.tft_temporal_hidden_size,
            dropout=0.0,
        )
        x = torch.randn(2, cfg.seq_len + cfg.pred_len, cfg.d_model)
        c0 = torch.randn(1, 2, cfg.d_model)
        h0 = torch.randn(1, 2, cfg.d_model)
        out, next_state = backbone(x, state=(c0, h0))
        self.assertEqual(tuple(out.shape), tuple(x.shape))
        self.assertEqual(tuple(next_state[0].shape), (1, 2, cfg.d_model))
        self.assertEqual(tuple(next_state[1].shape), (1, 2, cfg.d_model))
        self.assertTrue(torch.isfinite(out).all())

    def test_positional_multihead_attention_component(self):
        cfg = build_tsl_config()
        x = torch.randn(2, cfg.seq_len + cfg.pred_len, cfg.d_model)
        attn_mask = build_causal_mask(x.shape[1], x.device, x.dtype)
        baseline = PositionalMultiHeadAttention(cfg.d_model, cfg.n_heads, dropout=0.0, position_bias_type="none")
        rope = PositionalMultiHeadAttention(
            cfg.d_model,
            cfg.n_heads,
            dropout=0.0,
            position_bias_type="rope",
            rope_base=cfg.tft_rope_base,
        )
        alibi = PositionalMultiHeadAttention(
            cfg.d_model,
            cfg.n_heads,
            dropout=0.0,
            position_bias_type="alibi",
            alibi_scale=cfg.tft_alibi_scale,
        )
        rope.load_state_dict(baseline.state_dict())
        alibi.load_state_dict(baseline.state_dict())

        base_out, base_attn = baseline(x, x, x, return_attention=True, attn_mask=attn_mask)
        rope_out, rope_attn = rope(x, x, x, return_attention=True, attn_mask=attn_mask)
        alibi_out, alibi_attn = alibi(x, x, x, return_attention=True, attn_mask=attn_mask)

        self.assertEqual(tuple(base_out.shape), tuple(x.shape))
        self.assertEqual(tuple(rope_attn.shape), (2, cfg.n_heads, x.shape[1], x.shape[1]))
        self.assertEqual(tuple(alibi_attn.shape), (2, cfg.n_heads, x.shape[1], x.shape[1]))
        future_mask = torch.triu(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool), diagonal=1)
        self.assertTrue(torch.allclose(rope_attn.masked_select(future_mask.unsqueeze(0).unsqueeze(0)), torch.zeros_like(rope_attn.masked_select(future_mask.unsqueeze(0).unsqueeze(0))), atol=1e-6))
        self.assertTrue(torch.allclose(alibi_attn.masked_select(future_mask.unsqueeze(0).unsqueeze(0)), torch.zeros_like(alibi_attn.masked_select(future_mask.unsqueeze(0).unsqueeze(0))), atol=1e-6))
        self.assertFalse(torch.allclose(base_out, rope_out))
        self.assertFalse(torch.allclose(base_out, alibi_out))

    def test_sdpa_attention_backend_matches_exact(self):
        cfg = build_tsl_config()
        x = torch.randn(2, cfg.seq_len + cfg.pred_len, cfg.d_model)
        attn_mask = build_causal_mask(x.shape[1], x.device, x.dtype)
        for position_bias_type in ("none", "rope", "alibi"):
            exact = PositionalMultiHeadAttention(
                cfg.d_model,
                cfg.n_heads,
                dropout=0.0,
                position_bias_type=position_bias_type,
                rope_base=cfg.tft_rope_base,
                alibi_scale=cfg.tft_alibi_scale,
                attention_backend="exact",
            )
            sdpa = PositionalMultiHeadAttention(
                cfg.d_model,
                cfg.n_heads,
                dropout=0.0,
                position_bias_type=position_bias_type,
                rope_base=cfg.tft_rope_base,
                alibi_scale=cfg.tft_alibi_scale,
                attention_backend="sdpa",
            )
            sdpa.load_state_dict(exact.state_dict())

            exact_out = exact(x, x, x, attn_mask=attn_mask)
            sdpa_out = sdpa(x, x, x, attn_mask=attn_mask)
            self.assertTrue(torch.allclose(exact_out, sdpa_out, atol=1e-5, rtol=1e-4))
            self.assertEqual(sdpa.last_attention_backend, "sdpa")

            _, exact_attn = sdpa(x, x, x, return_attention=True, attn_mask=attn_mask)
            self.assertEqual(sdpa.last_attention_backend, "exact")
            self.assertEqual(tuple(exact_attn.shape), (2, cfg.n_heads, x.shape[1], x.shape[1]))

    def test_interpretable_cross_attention_component(self):
        cfg = build_tsl_config()
        cross_attention = InterpretableCrossAttention(cfg.d_model, cfg.n_heads, dropout=0.0)
        query = torch.randn(2, cfg.pred_len, cfg.d_model)
        context = torch.randn(2, cfg.seq_len, cfg.d_model)
        out, attn = cross_attention(query, context, return_attention=True)
        self.assertEqual(tuple(out.shape), (2, cfg.pred_len, cfg.d_model))
        self.assertEqual(tuple(attn.shape), (2, cfg.n_heads, cfg.pred_len, cfg.seq_len))
        self.assertTrue(torch.allclose(attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)), atol=1e-6))

    def test_regime_moe_component(self):
        cfg = build_tsl_config()
        moe = RegimeAwareSparseMoE(
            d_model=cfg.d_model,
            num_experts=cfg.tft_num_moe_experts,
            top_k=cfg.tft_moe_top_k,
            num_regimes=cfg.tft_num_regimes,
            hidden_size=cfg.tft_moe_hidden_size,
            dropout=0.0,
            noise_epsilon=cfg.tft_moe_noise_epsilon,
        )
        x = torch.randn(2, cfg.seq_len + cfg.pred_len, cfg.d_model)
        context = torch.randn(2, cfg.d_model)
        out, aux_loss, payload = moe(x, context=context, return_payload=True)
        self.assertEqual(tuple(out.shape), tuple(x.shape))
        self.assertEqual(tuple(payload["expert_routing"].shape), (2, cfg.seq_len + cfg.pred_len, cfg.tft_num_moe_experts))
        self.assertEqual(tuple(payload["regime_probabilities"].shape), (2, cfg.seq_len + cfg.pred_len, cfg.tft_num_regimes))
        self.assertEqual(tuple(payload["regime_probabilities_pooled"].shape), (2, cfg.tft_num_regimes))
        self.assertTrue(torch.allclose(payload["expert_routing"].sum(dim=-1), torch.ones_like(payload["expert_routing"].sum(dim=-1)), atol=1e-6))
        self.assertTrue(torch.allclose(payload["regime_probabilities"].sum(dim=-1), torch.ones_like(payload["regime_probabilities"].sum(dim=-1)), atol=1e-6))
        self.assertGreater((payload["regime_probabilities"][:, 1:, :] - payload["regime_probabilities"][:, :-1, :]).abs().mean().item(), 0.0)
        self.assertGreaterEqual(float(aux_loss), 0.0)

    def test_auxiliary_loss_helpers(self):
        primary = torch.tensor(2.0)
        aux = torch.tensor(0.5)
        combined = combine_primary_and_aux_loss(primary, aux, coeff=0.2)
        self.assertAlmostEqual(float(combined), 2.1, places=6)

        model_holder = SimpleNamespace(last_moe_aux_loss=aux)
        wrapped = SimpleNamespace(module=model_holder)
        self.assertEqual(float(get_auxiliary_loss(model_holder)), 0.5)
        self.assertEqual(float(get_auxiliary_loss(wrapped)), 0.5)

    def test_quantile_loss(self):
        criterion = QuantileLoss([0.1, 0.5, 0.9])
        forecast = torch.tensor([[[[1.0], [2.0], [3.0]]]])
        target = torch.tensor([[[2.5]]])
        loss = criterion(forecast, target)
        self.assertGreater(float(loss), 0.0)
        self.assertAlmostEqual(float(loss), float(QuantileLoss([0.1, 0.5, 0.9])(forecast, target)), places=6)

    def test_tsl_component_contracts_and_payload(self):
        cfg = build_tsl_config()

        graph = DynamicGraphLearner(cfg.d_model, cfg.n_heads, dropout=0.1, output_attention=True)
        graph_x = torch.randn(2, cfg.seq_len, 8, cfg.d_model)
        graph_out = graph(graph_x, return_attention=False)
        self.assertTrue(torch.is_tensor(graph_out))
        self.assertEqual(tuple(graph_out.shape), tuple(graph_x.shape))
        graph_out_attn, graph_attn = graph(graph_x, return_attention=True)
        self.assertEqual(tuple(graph_out_attn.shape), tuple(graph_x.shape))
        self.assertEqual(tuple(graph_attn.shape), (2, cfg.seq_len, cfg.n_heads, 8, 8))
        self.assertTrue(torch.isfinite(graph_out_attn).all())
        self.assertTrue(torch.isfinite(graph_attn).all())

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
        selected, weight_payload = vsn(x, context=context, return_weights=True)
        weights = weight_payload["selection"]
        graph_weights = weight_payload["graph_attention"]
        self.assertEqual(tuple(selected.shape), (2, cfg.seq_len, cfg.d_model))
        self.assertEqual(tuple(weights.shape), (2, cfg.seq_len, 8))
        self.assertEqual(tuple(graph_weights.shape), (2, cfg.seq_len, cfg.n_heads, 8, 8))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-5))

        model = tsl_tft.Model(cfg)
        self.assertTrue(model.use_revin)
        self.assertIsNotNone(model.revin)
        self.assertIsNone(model.temporal_fusion_decoder.layers[0].position_wise_grn)
        self.assertEqual(model.temporal_fusion_decoder.layers[0].temporal_backbone_type, "lstm")
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
        self.assertIn("attention_branch_weights", payload)
        self.assertIn("cross_attention_weights", payload)
        self.assertIn("lag_attention_weights", payload)
        self.assertIn("lag_scale_weights", payload)
        self.assertIn("interaction_contribution", payload)
        self.assertIn("interaction_gates", payload)
        self.assertIn("expert_routing", payload)
        self.assertIn("regime_probabilities", payload)
        self.assertIn("regime_probabilities_pooled", payload)
        self.assertIn("moe_aux_loss", payload)
        self.assertIn("decoder_layer_payloads", payload)
        self.assertIn("decoder_num_layers", payload)
        self.assertIn("temporal_backbone_type", payload)
        self.assertIn("attention_backend_config", payload)
        self.assertIn("attention_backend_used", payload)
        self.assertIn("history_graph_attention", payload)
        self.assertIn("future_graph_attention", payload)
        self.assertIn("static_graph_attention", payload)
        self.assertEqual(tuple(payload["predictions"].shape), (1, cfg.pred_len, cfg.c_out))
        self.assertEqual(tuple(payload["predictions_full"].shape), (1, cfg.seq_len + cfg.pred_len, cfg.c_out))
        self.assertEqual(tuple(payload["attention_branch_weights"].shape), (3,))
        self.assertEqual(tuple(payload["cross_attention_weights"].shape), (1, cfg.n_heads, cfg.pred_len, cfg.seq_len))
        self.assertEqual(tuple(payload["lag_scale_weights"].shape), (len(cfg.tft_lag_scales),))
        self.assertEqual(payload["lag_attention_weights"].shape[-1], len(cfg.tft_lag_scales))
        self.assertEqual(tuple(payload["interaction_contribution"].shape), (1, cfg.seq_len + cfg.pred_len, cfg.d_model))
        self.assertEqual(tuple(payload["interaction_gates"].shape), (1, cfg.seq_len + cfg.pred_len, cfg.tft_interaction_order))
        self.assertEqual(tuple(payload["expert_routing"].shape), (1, cfg.seq_len + cfg.pred_len, cfg.tft_num_moe_experts))
        self.assertEqual(tuple(payload["regime_probabilities"].shape), (1, cfg.seq_len + cfg.pred_len, cfg.tft_num_regimes))
        self.assertEqual(tuple(payload["regime_probabilities_pooled"].shape), (1, cfg.tft_num_regimes))
        self.assertGreaterEqual(float(payload["moe_aux_loss"]), 0.0)
        self.assertEqual(payload["decoder_num_layers"], cfg.e_layers)
        self.assertEqual(payload["temporal_backbone_type"], "lstm")
        self.assertEqual(payload["attention_backend_config"], "exact")
        self.assertEqual(payload["attention_backend_used"], "exact")
        self.assertIn("attention_branch_weights", payload["decoder_layer_payloads"])
        self.assertIn("cross_attention", payload["decoder_layer_payloads"])
        self.assertEqual(tuple(payload["decoder_layer_payloads"]["attention_branch_weights"].shape), (cfg.e_layers, 3))
        self.assertEqual(tuple(payload["decoder_layer_payloads"]["cross_attention"].shape), (cfg.e_layers, 1, cfg.n_heads, cfg.pred_len, cfg.seq_len))
        self.assertEqual(tuple(payload["decoder_layer_payloads"]["lag_scale_weights"].shape), (cfg.e_layers, len(cfg.tft_lag_scales)))
        self.assertEqual(payload["decoder_layer_payloads"]["expert_routing"].shape[0], cfg.e_layers)
        self.assertEqual(payload["history_graph_attention"].shape[:3], (1, cfg.seq_len, cfg.n_heads))
        self.assertEqual(payload["future_graph_attention"].shape[:3], (1, cfg.pred_len, cfg.n_heads))
        self.assertIsNone(payload["static_graph_attention"])

    def test_tsl_quantile_head_and_interpretable_cross_attention(self):
        cfg = build_tsl_config()
        cfg.tft_cross_attention_type = "interpretable"
        cfg.tft_use_quantile_head = True
        cfg.tft_output_quantiles = [0.1, 0.5, 0.9]
        model = tsl_tft.Model(cfg)
        ds = make_tsl_dataset(cfg, n_samples=4)
        x_enc, x_mark_enc, x_dec, x_mark_dec, _ = ds[0]
        payload = model(
            x_enc.unsqueeze(0),
            x_mark_enc.unsqueeze(0),
            x_dec.unsqueeze(0),
            x_mark_dec.unsqueeze(0),
            return_interpretation=True,
        )
        self.assertEqual(payload["quantiles"], [0.1, 0.5, 0.9])
        self.assertEqual(tuple(payload["quantile_predictions"].shape), (1, cfg.pred_len, 3, cfg.c_out))
        self.assertEqual(tuple(payload["cross_attention_weights"].shape), (1, cfg.n_heads, cfg.pred_len, cfg.seq_len))

    def test_tsl_position_bias_modes(self):
        for position_bias_type, cross_attention_type in (("rope", "interpretable"), ("alibi", "full")):
            cfg = build_tsl_config()
            cfg.tft_attention_position_bias = position_bias_type
            cfg.tft_cross_attention_type = cross_attention_type
            model = tsl_tft.Model(cfg)
            ds = make_tsl_dataset(cfg, n_samples=4)
            x_enc, x_mark_enc, x_dec, x_mark_dec, _ = ds[0]
            payload = model(
                x_enc.unsqueeze(0),
                x_mark_enc.unsqueeze(0),
                x_dec.unsqueeze(0),
                x_mark_dec.unsqueeze(0),
                return_interpretation=True,
            )
            self.assertEqual(payload["position_bias_type"], position_bias_type)
            self.assertEqual(tuple(payload["predictions"].shape), (1, cfg.pred_len, cfg.c_out))
            self.assertEqual(tuple(payload["cross_attention_weights"].shape), (1, cfg.n_heads, cfg.pred_len, cfg.seq_len))
            self.assertEqual(tuple(payload["attention_weights_full"].shape), (1, cfg.n_heads, cfg.seq_len + cfg.pred_len, cfg.seq_len + cfg.pred_len))
            self.assertTrue(torch.isfinite(payload["predictions"]).all())
            self.assertEqual(model.temporal_fusion_decoder.layers[0].cross_attention_type, cross_attention_type)

    def test_tsl_gated_tcn_backbone_mode(self):
        cfg = build_tsl_config()
        cfg.tft_temporal_backbone = "gated_tcn"
        model = tsl_tft.Model(cfg)
        self.assertIsNone(model.temporal_fusion_decoder.layers[0].history_encoder)
        self.assertIsNotNone(model.temporal_fusion_decoder.layers[0].temporal_backbone)
        ds = make_tsl_dataset(cfg, n_samples=4)
        x_enc, x_mark_enc, x_dec, x_mark_dec, _ = ds[0]
        payload = model(
            x_enc.unsqueeze(0),
            x_mark_enc.unsqueeze(0),
            x_dec.unsqueeze(0),
            x_mark_dec.unsqueeze(0),
            return_interpretation=True,
        )
        self.assertEqual(payload["temporal_backbone_type"], "gated_tcn")
        self.assertEqual(tuple(payload["predictions"].shape), (1, cfg.pred_len, cfg.c_out))
        self.assertTrue(torch.isfinite(payload["predictions"]).all())

    def test_tsl_hybrid_backbone_mode(self):
        cfg = build_tsl_config()
        cfg.tft_temporal_backbone = "hybrid_tcn_lstm"
        model = tsl_tft.Model(cfg)
        self.assertIsInstance(model.temporal_fusion_decoder.layers[0].temporal_backbone, HybridTemporalBackbone)
        ds = make_tsl_dataset(cfg, n_samples=4)
        x_enc, x_mark_enc, x_dec, x_mark_dec, _ = ds[0]
        payload = model(
            x_enc.unsqueeze(0),
            x_mark_enc.unsqueeze(0),
            x_dec.unsqueeze(0),
            x_mark_dec.unsqueeze(0),
            return_interpretation=True,
        )
        self.assertEqual(payload["temporal_backbone_type"], "hybrid_tcn_lstm")
        self.assertTrue(torch.isfinite(payload["predictions"]).all())

    def test_tsl_sdpa_attention_backend_mode(self):
        cfg = build_tsl_config()
        cfg.tft_attention_backend = "sdpa"
        model = tsl_tft.Model(cfg)
        ds = make_tsl_dataset(cfg, n_samples=4)
        x_enc, x_mark_enc, x_dec, x_mark_dec, _ = ds[0]

        forward_out = model(
            x_enc.unsqueeze(0),
            x_mark_enc.unsqueeze(0),
            x_dec.unsqueeze(0),
            x_mark_dec.unsqueeze(0),
        )
        self.assertTrue(torch.isfinite(forward_out).all())
        self.assertEqual(model.temporal_fusion_decoder.layers[0].full_attention_module.last_attention_backend, "sdpa")
        self.assertEqual(model.temporal_fusion_decoder.layers[0].cross_attention.last_attention_backend, "sdpa")

        payload = model(
            x_enc.unsqueeze(0),
            x_mark_enc.unsqueeze(0),
            x_dec.unsqueeze(0),
            x_mark_dec.unsqueeze(0),
            return_interpretation=True,
        )
        self.assertEqual(payload["attention_backend_config"], "sdpa")
        self.assertEqual(payload["attention_backend_used"], "exact")
        self.assertEqual(payload["cross_attention_backend_used"], "exact")

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
        self.assertIsNotNone(model.temporal_fusion_decoder.layers[0].attention_fusion_logits.grad)
        self.assertGreater(model.temporal_fusion_decoder.layers[0].attention_fusion_logits.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(model.temporal_fusion_decoder.layers[0].lag_attention_module.scale_logits.grad)
        self.assertGreater(model.temporal_fusion_decoder.layers[0].lag_attention_module.scale_logits.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(model.temporal_fusion_decoder.layers[0].higher_order_block.gate_projection.weight.grad)
        self.assertGreater(model.temporal_fusion_decoder.layers[0].higher_order_block.gate_projection.weight.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(model.temporal_fusion_decoder.layers[0].regime_moe.gate.weight.grad)
        self.assertGreater(model.temporal_fusion_decoder.layers[0].regime_moe.gate.weight.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(model.temporal_fusion_decoder.layers[0].regime_moe.regime_detector[0].weight.grad)
        self.assertGreater(model.temporal_fusion_decoder.layers[0].regime_moe.regime_detector[0].weight.grad.abs().sum().item(), 0.0)
        cross_attention = model.temporal_fusion_decoder.layers[0].cross_attention
        cross_grad = getattr(cross_attention, "q_linear", None)
        cross_grad = cross_grad.weight.grad if cross_grad is not None else cross_attention.in_proj_weight.grad
        self.assertIsNotNone(cross_grad)
        self.assertGreater(cross_grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(model.revin.affine_weight.grad)
        self.assertGreater(model.revin.affine_weight.grad.abs().sum().item(), 0.0)

    def test_tsl_gated_tcn_model_learns_structured_signal(self):
        cfg = build_tsl_config()
        cfg.tft_temporal_backbone = "gated_tcn"
        model = tsl_tft.Model(cfg)
        ds = make_tsl_dataset(cfg, n_samples=48)
        loader = DataLoader(ds, batch_size=8, shuffle=True)

        losses = train_tsl_once(model, loader, cfg, lr=3e-3, epochs=4)
        self.assertGreater(losses[0], losses[-1], "Gated-TCN TFT did not improve training loss.")
        self.assertLess(losses[-1], losses[0] * 0.90, "Gated-TCN TFT loss reduction is too weak for a learnable target.")
        block = model.temporal_fusion_decoder.layers[0].temporal_backbone.blocks[0]
        self.assertIsNotNone(block.filter_conv.conv.weight.grad)
        self.assertGreater(block.filter_conv.conv.weight.grad.abs().sum().item(), 0.0)

    def test_tsl_hybrid_backbone_model_learns_structured_signal(self):
        cfg = build_tsl_config()
        cfg.tft_temporal_backbone = "hybrid_tcn_lstm"
        model = tsl_tft.Model(cfg)
        ds = make_tsl_dataset(cfg, n_samples=48)
        loader = DataLoader(ds, batch_size=8, shuffle=True)

        losses = train_tsl_once(model, loader, cfg, lr=3e-3, epochs=4)
        self.assertGreater(losses[0], losses[-1], "Hybrid TFT did not improve training loss.")
        self.assertLess(losses[-1], losses[0] * 0.90, "Hybrid TFT loss reduction is too weak for a learnable target.")
        backbone = model.temporal_fusion_decoder.layers[0].temporal_backbone
        self.assertIsNotNone(backbone.fusion_gate.weight.grad)
        self.assertGreater(backbone.fusion_gate.weight.grad.abs().sum().item(), 0.0)

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
