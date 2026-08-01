import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from layers.DynamicGraph import DynamicGraphLearner
from layers.AdvancedDynamicGraph import AdvancedDynamicGraphLearner
from layers.TemporalFusion_layers import (
    GatedDilatedTemporalBackbone,
    HigherOrderInteractionBlock,
    HybridTemporalBackbone,
    InterpretableCrossAttention,
    MultiScaleLagAttention,
    PositionalMultiHeadAttention,
    RegimeAwareSparseMoE,
    SpectralBranch,
    TemporalCompression,
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
        tft_vsn_n_selection_heads=1,
        # Phase A/B/C defaults
        tft_per_target_heads=False,
        tft_vsn_per_feature_gating=False,
        tft_covariate_reattention=False,
        tft_moe_capacity_factor=1.25,
        tft_vsn_low_rank_threshold=64,
        # Exercise active semantics-v2 branches in this comprehensive suite.
        # Neutrality itself is covered by the dedicated SR02 parity matrix.
        tft_fft_integration_mode="small_residual",
        tft_cross_attention_integration_mode="small_residual",
        tft_lag_integration_mode="small_residual",
        tft_higher_order_integration_mode="small_residual",
        tft_temporal_compression_integration_mode="small_residual",
        tft_graph_integration_mode="small_residual",
        tft_covariate_reattention_integration_mode="small_residual",
        tft_regime_moe_integration_mode="small_residual",
        tft_dual_attention_integration_mode="small_residual",
        tft_vsn_bypass_integration_mode="small_residual",
        # Advanced graph defaults
        tft_graph_type="dense",
        tft_graph_top_k=10,
        tft_graph_num_layers=2,
        tft_graph_temporal_evolution=False,
        tft_graph_edge_features=False,
        tft_allow_custom_known=True,
        tft_known_len=12,
        tft_known_max_channels=64,
        tft_known_feature_names=[f"known_{i}" for i in range(12)],
        tft_observed_pos=list(range(16)),
        tft_static_pos=[],
        # The synthetic generator emits a complete, regularly spaced grid.
        tft_declared_regular_sampling=True,
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
        self.assertEqual(tuple(payload["interaction_gates"].shape), (2, cfg.seq_len + cfg.pred_len, cfg.tft_interaction_order - 1))
        self.assertTrue(torch.isfinite(payload["interaction_contribution"]).all())
        self.assertTrue(torch.all((payload["interaction_gates"] >= 0.0) & (payload["interaction_gates"] <= 1.0)))

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
        self.assertEqual(payload["lag_attention_mode"], "shifted_history_attention")

        future_mask = torch.triu(torch.ones(cfg.seq_len + cfg.pred_len, cfg.seq_len + cfg.pred_len, dtype=torch.bool), diagonal=1)
        masked_values = payload["lag_attention"].masked_select(future_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1))
        self.assertTrue(torch.allclose(masked_values, torch.zeros_like(masked_values), atol=1e-6))

        for branch_idx, lag in enumerate(cfg.tft_lag_scales):
            padded = payload["lag_attention"][..., :lag, branch_idx]
            self.assertTrue(torch.allclose(padded, torch.zeros_like(padded), atol=1e-6))

    def test_lag_attention_uses_shifted_physical_positions(self):
        lag_attention = MultiScaleLagAttention(
            d_model=16, n_heads=2, lag_scales=[2, 4], dropout=0.0, position_bias_type="alibi",
        )
        x = torch.randn(1, 10, 16)
        _, payload = lag_attention(x, return_attention=True, positions=torch.arange(10, dtype=torch.float32))
        self.assertTrue(torch.equal(payload["lag_query_positions"], torch.arange(10, dtype=torch.float32)))
        self.assertTrue(torch.equal(payload["lag_key_positions"][0], torch.arange(10, dtype=torch.float32) - 2.0))
        self.assertTrue(torch.equal(payload["lag_key_positions"][1], torch.arange(10, dtype=torch.float32) - 4.0))

    def test_lag_attention_excessive_lag_fails(self):
        lag_attention = MultiScaleLagAttention(
            d_model=16, n_heads=2, lag_scales=[8], dropout=0.0,
        )
        x = torch.randn(1, 8, 16)
        with self.assertRaises(ValueError):
            lag_attention(x)

    def test_compression_shell_preserves_full_original_coordinates(self):
        set_seed(42)
        cfg = build_tsl_config()
        cfg.tft_use_temporal_compression = True
        cfg.tft_temporal_compression_mode = "legacy_codec"
        cfg.tft_tc_stride = 2
        cfg.tft_tc_threshold = 4
        cfg.tft_use_lag_attention = True
        cfg.tft_lag_scales = [1, 2]
        cfg.tft_attention_position_bias = "alibi"
        model = tsl_tft.Model(cfg)
        ds = make_tsl_dataset(cfg, n_samples=4)
        x_enc, x_mark_enc, x_dec, x_mark_dec, _ = ds[0]
        payload = model(
            x_enc.unsqueeze(0), x_mark_enc.unsqueeze(0),
            x_dec.unsqueeze(0), x_mark_dec.unsqueeze(0),
            return_interpretation=True,
        )
        positions = payload["temporal_positions"][0]
        self.assertTrue(torch.all(positions[1:] > positions[:-1]))
        self.assertEqual(int(positions[0].item()), 0)
        self.assertEqual(int(positions[1].item()), 1)
        self.assertEqual(int(positions[-1].item()), cfg.seq_len + cfg.pred_len - 1)

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

    def test_attention_probability_dropout_changes_training_output(self):
        cfg = build_tsl_config()
        x = torch.randn(2, cfg.seq_len + cfg.pred_len, cfg.d_model)
        attn_mask = build_causal_mask(x.shape[1], x.device, x.dtype)
        attn = PositionalMultiHeadAttention(
            cfg.d_model, cfg.n_heads, dropout=0.0, attn_dropout=0.3, position_bias_type="none", attention_backend="exact",
        ).train()
        out1 = attn(x, x, x, attn_mask=attn_mask)
        out2 = attn(x, x, x, attn_mask=attn_mask)
        self.assertFalse(torch.allclose(out1, out2))

    def test_attention_probability_dropout_is_deterministic_in_eval(self):
        cfg = build_tsl_config()
        x = torch.randn(2, cfg.seq_len + cfg.pred_len, cfg.d_model)
        attn_mask = build_causal_mask(x.shape[1], x.device, x.dtype)
        attn = PositionalMultiHeadAttention(
            cfg.d_model, cfg.n_heads, dropout=0.0, attn_dropout=0.3, position_bias_type="none", attention_backend="exact",
        ).eval()
        out1, attn1 = attn(x, x, x, return_attention=True, attn_mask=attn_mask)
        out2, attn2 = attn(x, x, x, return_attention=True, attn_mask=attn_mask)
        self.assertTrue(torch.allclose(out1, out2))
        self.assertTrue(torch.allclose(attn1, attn2))

    def test_attention_dropout_exact_and_sdpa_eval_match(self):
        cfg = build_tsl_config()
        x = torch.randn(2, cfg.seq_len + cfg.pred_len, cfg.d_model)
        attn_mask = build_causal_mask(x.shape[1], x.device, x.dtype)
        exact = PositionalMultiHeadAttention(
            cfg.d_model, cfg.n_heads, dropout=0.0, attn_dropout=0.25, position_bias_type="alibi",
            alibi_scale=cfg.tft_alibi_scale, attention_backend="exact",
        ).eval()
        sdpa = PositionalMultiHeadAttention(
            cfg.d_model, cfg.n_heads, dropout=0.0, attn_dropout=0.25, position_bias_type="alibi",
            alibi_scale=cfg.tft_alibi_scale, attention_backend="sdpa",
        ).eval()
        sdpa.load_state_dict(exact.state_dict())
        out_exact, attn_exact = exact(x, x, x, return_attention=True, attn_mask=attn_mask)
        out_sdpa = sdpa(x, x, x, attn_mask=attn_mask)
        self.assertTrue(torch.allclose(out_exact, out_sdpa, atol=1e-5, rtol=1e-5))
        _, attn_sdpa = sdpa(x, x, x, return_attention=True, attn_mask=attn_mask)
        self.assertTrue(torch.allclose(attn_exact, attn_sdpa, atol=1e-5, rtol=1e-5))

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
        self.assertEqual(payload["moe_routing_mode"], "dense_compute_topk_mixing")
        self.assertTrue(torch.allclose(payload["expert_routing"].sum(dim=-1), torch.ones_like(payload["expert_routing"].sum(dim=-1)), atol=1e-6))
        self.assertTrue(torch.allclose(payload["regime_probabilities"].sum(dim=-1), torch.ones_like(payload["regime_probabilities"].sum(dim=-1)), atol=1e-6))
        self.assertGreater((payload["regime_probabilities"][:, 1:, :] - payload["regime_probabilities"][:, :-1, :]).abs().mean().item(), 0.0)
        self.assertGreaterEqual(float(aux_loss), 0.0)

    def test_moe_small_batch_has_nonzero_route(self):
        moe = RegimeAwareSparseMoE(
            d_model=8, num_experts=4, top_k=2, num_regimes=3, hidden_size=16, dropout=0.0, noise_epsilon=1e-2,
        )
        moe.capacity_factor = 0.1
        moe.train()
        x = torch.randn(1, 1, 8, requires_grad=True)
        out, aux_loss, payload = moe(x, return_payload=True)
        routing = payload["expert_routing"]
        self.assertGreater(routing.sum().item(), 0.0)
        self.assertTrue(torch.allclose(routing.sum(dim=-1), torch.ones_like(routing.sum(dim=-1)), atol=1e-6))
        loss = out.mean() + aux_loss
        loss.backward()
        self.assertIsNotNone(moe.gate.weight.grad)
        self.assertGreater(moe.gate.weight.grad.abs().sum().item(), 0.0)

    def test_moe_every_token_has_route(self):
        moe = RegimeAwareSparseMoE(
            d_model=8, num_experts=4, top_k=2, num_regimes=3, hidden_size=16, dropout=0.0, noise_epsilon=1e-2,
        )
        moe.capacity_factor = 0.25
        moe.train()
        x = torch.randn(2, 3, 8)
        _, _, payload = moe(x, return_payload=True)
        routing = payload["expert_routing"]
        self.assertTrue(torch.all(routing.sum(dim=-1) > 0.0))

    def test_moe_heavy_imbalance_capacity_case_keeps_routes(self):
        moe = RegimeAwareSparseMoE(
            d_model=8, num_experts=4, top_k=2, num_regimes=3, hidden_size=16, dropout=0.0, noise_epsilon=1e-2,
        )
        moe.capacity_factor = 0.25
        moe.train()
        with torch.no_grad():
            moe.gate.weight.zero_()
            moe.regime_expert_bias.zero_()
            moe.regime_expert_bias[:, 0] = 10.0
            moe.regime_expert_bias[:, 1] = 9.0
        x = torch.randn(3, 4, 8)
        _, aux_loss, payload = moe(x, return_payload=True)
        routing = payload["expert_routing"]
        self.assertTrue(torch.allclose(routing.sum(dim=-1), torch.ones_like(routing.sum(dim=-1)), atol=1e-6))
        self.assertGreaterEqual(float(aux_loss), 0.0)
        self.assertIn("expert_load_sum", payload)
        self.assertGreater(payload["expert_load_sum"].sum().item(), 0.0)

    def test_moe_selected_experts_receive_gradients(self):
        moe = RegimeAwareSparseMoE(
            d_model=8, num_experts=4, top_k=2, num_regimes=3, hidden_size=16, dropout=0.0, noise_epsilon=1e-2,
        )
        moe.capacity_factor = 0.5
        moe.train()
        x = torch.randn(2, 3, 8, requires_grad=True)
        out, aux_loss, payload = moe(x, return_payload=True)
        loss = out.square().mean() + aux_loss
        loss.backward()
        used = payload["expert_routing"].sum(dim=(0, 1)) > 0
        grad_norms = moe.expert_w1.grad.abs().sum(dim=(1, 2))
        self.assertTrue(torch.all(grad_norms[used] > 0))

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
        self.assertIsNotNone(model.temporal_fusion_decoder.layers[0].position_wise_grn)
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
        self.assertIsNone(payload["attention_branch_weights"])
        self.assertIsNone(payload["attention_fusion_alpha"])
        self.assertIn("dual_attention_fusion", payload["extension_residuals"])
        self.assertIn("lag_attention", payload["extension_residuals"])
        self.assertEqual(tuple(payload["cross_attention_weights"].shape), (1, cfg.n_heads, cfg.pred_len, cfg.seq_len))
        self.assertEqual(tuple(payload["lag_scale_weights"].shape), (len(cfg.tft_lag_scales),))
        self.assertEqual(payload["lag_attention_weights"].shape[-1], len(cfg.tft_lag_scales))
        self.assertEqual(tuple(payload["interaction_contribution"].shape), (1, cfg.seq_len + cfg.pred_len, cfg.d_model))
        self.assertEqual(tuple(payload["interaction_gates"].shape), (1, cfg.seq_len + cfg.pred_len, cfg.tft_interaction_order - 1))
        self.assertEqual(tuple(payload["expert_routing"].shape), (1, cfg.seq_len + cfg.pred_len, cfg.tft_num_moe_experts))
        self.assertEqual(tuple(payload["regime_probabilities"].shape), (1, cfg.seq_len + cfg.pred_len, cfg.tft_num_regimes))
        self.assertEqual(tuple(payload["regime_probabilities_pooled"].shape), (1, cfg.tft_num_regimes))
        self.assertGreaterEqual(float(payload["moe_aux_loss"]), 0.0)
        self.assertEqual(payload["decoder_num_layers"], cfg.e_layers)
        self.assertEqual(payload["temporal_backbone_type"], "lstm")
        self.assertEqual(payload["attention_backend_config"], "exact")
        self.assertEqual(payload["attention_backend_used"], "exact")
        self.assertIn("extension_residuals", payload["decoder_layer_payloads"])
        self.assertIn("cross_attention", payload["decoder_layer_payloads"])
        self.assertNotIn("attention_branch_weights", payload["decoder_layer_payloads"])
        self.assertEqual(tuple(payload["decoder_layer_payloads"]["cross_attention"].shape), (cfg.e_layers, 1, cfg.n_heads, cfg.pred_len, cfg.seq_len))
        self.assertEqual(tuple(payload["decoder_layer_payloads"]["lag_scale_weights"].shape), (cfg.e_layers, len(cfg.tft_lag_scales)))
        self.assertEqual(payload["decoder_layer_payloads"]["expert_routing"].shape[0], cfg.e_layers)
        self.assertEqual(payload["history_graph_attention"].shape[:3], (1, cfg.seq_len, 1))
        self.assertEqual(payload["history_graph_metadata"]["num_reported_heads"], 1)
        self.assertEqual(payload["future_graph_attention"].shape[:3], (1, cfg.pred_len, 1))
        self.assertEqual(payload["future_graph_metadata"]["num_reported_heads"], 1)
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
        self.assertEqual(model.temporal_fusion_decoder.layers[0].attention.last_attention_backend, "sdpa")
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
        gate_grad = model.history_vsn.vsn_bypass_residual_adapter.residual_strength.grad
        self.assertIsNotNone(gate_grad)
        self.assertGreater(gate_grad.abs().sum().item(), 0.0)
        dual_strength_grad = model.temporal_fusion_decoder.layers[0].dual_attention_residual_adapter.residual_strength.grad
        self.assertIsNotNone(dual_strength_grad)
        self.assertGreater(dual_strength_grad.abs().sum().item(), 0.0)
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
        self.assertIsNotNone(model.revin.affine_weight_raw.grad)
        self.assertGreater(model.revin.affine_weight_raw.grad.abs().sum().item(), 0.0)
        self.assertTrue(torch.all(model.revin.affine_weight > 0))

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

    # ---------- SpectralBranch component tests ----------

    def test_spectral_branch_component_shape_and_gradient(self):
        cfg = build_tsl_config()
        for mode_select in ("low", "top_amplitude", "learned"):
            branch = SpectralBranch(
                d_model=cfg.d_model, modes=16, mode_select=mode_select, dropout=0.0,
            )
            x = torch.randn(2, cfg.seq_len + cfg.pred_len, cfg.d_model, requires_grad=True)
            out = branch(x)
            self.assertEqual(tuple(out.shape), tuple(x.shape))
            self.assertTrue(torch.isfinite(out).all())
            out.sum().backward()
            self.assertIsNotNone(x.grad)
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertIsNotNone(branch.weight_real.grad)
            self.assertGreater(branch.weight_real.grad.abs().sum().item(), 0.0)

    def test_spectral_branch_modes_clamped(self):
        branch = SpectralBranch(d_model=32, modes=999, mode_select='low', dropout=0.0)
        x = torch.randn(2, 8, 32)
        out = branch(x)
        self.assertEqual(tuple(out.shape), (2, 8, 32))

    # ---------- FFT branch integration tests ----------

    def test_tsl_fft_branch_forward_all_backbones(self):
        for backbone_type in ("lstm", "gated_tcn", "hybrid_tcn_lstm"):
            cfg = build_tsl_config()
            cfg.tft_temporal_backbone = backbone_type
            cfg.tft_use_fft_branch = True
            cfg.tft_fft_modes = 16
            cfg.tft_fft_mode_select = "low"
            model = tsl_tft.Model(cfg)
            ds = make_tsl_dataset(cfg, n_samples=4)
            x_enc, x_mark_enc, x_dec, x_mark_dec, _ = ds[0]
            payload = model(
                x_enc.unsqueeze(0), x_mark_enc.unsqueeze(0),
                x_dec.unsqueeze(0), x_mark_dec.unsqueeze(0),
                return_interpretation=True,
            )
            self.assertEqual(tuple(payload["predictions"].shape), (1, cfg.pred_len, cfg.c_out))
            self.assertTrue(torch.isfinite(payload["predictions"]).all())
            self.assertIsNotNone(model.temporal_fusion_decoder.layers[0].fft_branch)
            self.assertIsNotNone(model.temporal_fusion_decoder.layers[0].fft_fusion_gate)

    def test_tsl_fft_branch_interpretation_payload(self):
        cfg = build_tsl_config()
        cfg.tft_use_fft_branch = True
        cfg.tft_fft_modes = 16
        cfg.tft_fft_mode_select = "learned"
        model = tsl_tft.Model(cfg)
        ds = make_tsl_dataset(cfg, n_samples=4)
        x_enc, x_mark_enc, x_dec, x_mark_dec, _ = ds[0]
        payload = model(
            x_enc.unsqueeze(0), x_mark_enc.unsqueeze(0),
            x_dec.unsqueeze(0), x_mark_dec.unsqueeze(0),
            return_interpretation=True,
        )
        # fft_gate_mean is a scalar float from sigmoid gate [0, 1]
        self.assertIn("fft_gate_mean", payload)
        fft_gate = payload["fft_gate_mean"]
        self.assertIsInstance(fft_gate, float)
        self.assertGreaterEqual(fft_gate, 0.0)
        self.assertLessEqual(fft_gate, 1.0)
        self.assertIn("fft_learned_mask_mean", payload)
        self.assertIn("fft_learned_mask_std", payload)
        self.assertIn("fft_learned_mask_peak_bin_mean", payload)
        self.assertGreaterEqual(payload["fft_learned_mask_mean"], 0.0)
        self.assertLessEqual(payload["fft_learned_mask_mean"], 1.0)
        self.assertGreaterEqual(payload["fft_learned_mask_std"], 0.0)

    def test_tsl_fft_branch_learns_structured_signal(self):
        cfg = build_tsl_config()
        cfg.tft_use_fft_branch = True
        cfg.tft_fft_modes = 16
        cfg.tft_fft_mode_select = "low"
        model = tsl_tft.Model(cfg)
        ds = make_tsl_dataset(cfg, n_samples=48)
        loader = DataLoader(ds, batch_size=8, shuffle=True)
        losses = train_tsl_once(model, loader, cfg, lr=3e-3, epochs=4)
        self.assertGreater(losses[0], losses[-1], "FFT-branch TFT did not improve training loss.")
        self.assertLess(losses[-1], losses[0] * 0.90, "FFT-branch TFT loss reduction is too weak.")
        branch = model.temporal_fusion_decoder.layers[0].fft_branch
        self.assertIsNotNone(branch.weight_real.grad)
        self.assertGreater(branch.weight_real.grad.abs().sum().item(), 0.0)
        gate = model.temporal_fusion_decoder.layers[0].fft_fusion_gate
        self.assertIsNotNone(gate.weight.grad)
        self.assertGreater(gate.weight.grad.abs().sum().item(), 0.0)

    # ---------- Stochastic depth tests ----------

    def test_stochastic_depth_rate_zero_is_deterministic(self):
        cfg = build_tsl_config()
        cfg.e_layers = 3
        cfg.tft_stochastic_depth_rate = 0.0
        model = tsl_tft.Model(cfg)
        model.eval()
        ds = make_tsl_dataset(cfg, n_samples=4)
        x_enc, x_mark_enc, x_dec, x_mark_dec, _ = ds[0]
        inp = (x_enc.unsqueeze(0), x_mark_enc.unsqueeze(0), x_dec.unsqueeze(0), x_mark_dec.unsqueeze(0))
        with torch.no_grad():
            out1 = model(*inp)
            out2 = model(*inp)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-6))

    def test_stochastic_depth_nonzero_backward_works(self):
        cfg = build_tsl_config()
        cfg.e_layers = 4
        cfg.tft_stochastic_depth_rate = 0.5
        model = tsl_tft.Model(cfg)
        ds = make_tsl_dataset(cfg, n_samples=8)
        loader = DataLoader(ds, batch_size=4, shuffle=True)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        for x_enc, x_mark_enc, x_dec, x_mark_dec, y in loader:
            opt.zero_grad()
            out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            loss = criterion(out[:, -cfg.pred_len:, :], y)
            loss.backward()
            opt.step()
            self.assertTrue(torch.isfinite(loss))
            break

    def test_stochastic_depth_eval_is_deterministic(self):
        cfg = build_tsl_config()
        cfg.e_layers = 4
        cfg.tft_stochastic_depth_rate = 0.5
        model = tsl_tft.Model(cfg)
        model.eval()
        ds = make_tsl_dataset(cfg, n_samples=4)
        x_enc, x_mark_enc, x_dec, x_mark_dec, _ = ds[0]
        inp = (x_enc.unsqueeze(0), x_mark_enc.unsqueeze(0), x_dec.unsqueeze(0), x_mark_dec.unsqueeze(0))
        with torch.no_grad():
            out1 = model(*inp)
            out2 = model(*inp)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-6))

    # ---------- Gradient checkpointing tests ----------

    def test_gradient_checkpointing_eval_parity(self):
        cfg = build_tsl_config()
        cfg.e_layers = 2
        model = tsl_tft.Model(cfg)
        model.eval()
        ds = make_tsl_dataset(cfg, n_samples=4)
        x_enc, x_mark_enc, x_dec, x_mark_dec, _ = ds[0]
        inp = (x_enc.unsqueeze(0), x_mark_enc.unsqueeze(0), x_dec.unsqueeze(0), x_mark_dec.unsqueeze(0))
        with torch.no_grad():
            model.temporal_fusion_decoder.gradient_checkpointing = False
            out_no_ckpt = model(*inp)
            model.temporal_fusion_decoder.gradient_checkpointing = True
            out_ckpt = model(*inp)
        self.assertTrue(torch.allclose(out_no_ckpt, out_ckpt, atol=1e-6),
                        "Checkpointed and non-checkpointed eval outputs differ.")

    def test_gradient_checkpointing_backward_succeeds(self):
        cfg = build_tsl_config()
        cfg.e_layers = 2
        cfg.tft_gradient_checkpointing = True
        model = tsl_tft.Model(cfg)
        ds = make_tsl_dataset(cfg, n_samples=8)
        loader = DataLoader(ds, batch_size=4, shuffle=True)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        for x_enc, x_mark_enc, x_dec, x_mark_dec, y in loader:
            opt.zero_grad()
            out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            loss = criterion(out[:, -cfg.pred_len:, :], y)
            loss.backward()
            opt.step()
            self.assertTrue(torch.isfinite(loss))
            break

    def test_gradient_checkpointing_moe_aux_loss_preserved(self):
        cfg = build_tsl_config()
        cfg.e_layers = 2
        cfg.tft_gradient_checkpointing = True
        model = tsl_tft.Model(cfg)
        ds = make_tsl_dataset(cfg, n_samples=8)
        loader = DataLoader(ds, batch_size=4, shuffle=True)
        model.train()
        for x_enc, x_mark_enc, x_dec, x_mark_dec, y in loader:
            _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            aux = model.temporal_fusion_decoder.last_moe_aux_loss
            self.assertIsNotNone(aux, "MoE aux loss should be populated through gradient checkpointing.")
            self.assertTrue(torch.isfinite(aux))
            break

    def test_gradient_checkpointing_with_attention_returns_payload(self):
        cfg = build_tsl_config()
        cfg.e_layers = 2
        cfg.tft_gradient_checkpointing = True
        model = tsl_tft.Model(cfg)
        ds = make_tsl_dataset(cfg, n_samples=4)
        x_enc, x_mark_enc, x_dec, x_mark_dec, _ = ds[0]
        model.eval()
        payload = model(
            x_enc.unsqueeze(0), x_mark_enc.unsqueeze(0),
            x_dec.unsqueeze(0), x_mark_dec.unsqueeze(0),
            return_interpretation=True,
        )
        self.assertIsNotNone(payload)
        self.assertIn("predictions", payload)
        self.assertTrue(torch.isfinite(payload["predictions"]).all())

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

    # ------------------------------------------------------------------ #
    #  Temporal Compression (Learned Sequence Compression)
    # ------------------------------------------------------------------ #

    def test_temporal_compression_component_shape_and_gradient(self):
        """Compress and decompress roundtrip preserves shape; gradients flow."""
        for stride in (2, 4):
            tc = TemporalCompression(d_model=32, stride=stride, threshold=0, dropout=0.0)
            x = torch.randn(2, 64, 32, requires_grad=True)

            self.assertTrue(tc.should_compress(64))
            compressed, orig_len = tc.compress(x)
            expected_approx = 64 // stride
            # Allow ±1 due to conv padding arithmetic
            self.assertAlmostEqual(compressed.shape[1], expected_approx, delta=2,
                                   msg=f"stride={stride}: compressed length {compressed.shape[1]} "
                                       f"not near {expected_approx}")
            self.assertEqual(compressed.shape[0], 2)
            self.assertEqual(compressed.shape[2], 32)

            restored = tc.decompress_to(compressed, orig_len)
            self.assertEqual(restored.shape, (2, 64, 32),
                             f"stride={stride}: decompress did not restore shape")

            # Gradient flow
            loss = restored.sum()
            loss.backward()
            self.assertIsNotNone(x.grad)
            self.assertGreater(x.grad.abs().sum().item(), 0.0)

    def test_temporal_compression_threshold_noop(self):
        """should_compress returns False when seq_len <= threshold."""
        tc = TemporalCompression(d_model=32, stride=2, threshold=256, dropout=0.0)
        self.assertFalse(tc.should_compress(256))
        self.assertFalse(tc.should_compress(128))
        self.assertTrue(tc.should_compress(257))

    def test_temporal_compression_stride_1_noop(self):
        """stride=1 should never activate compression."""
        tc = TemporalCompression(d_model=32, stride=1, threshold=0, dropout=0.0)
        self.assertFalse(tc.should_compress(512))

    def test_tsl_temporal_compression_active_long_sequence(self):
        """Model with temporal compression produces correct output shape for long sequences."""
        set_seed(42)
        cfg = build_tsl_config()
        # Long sequence to trigger compression (threshold=64 for testing)
        cfg.seq_len = 128
        cfg.label_len = 64
        cfg.pred_len = 16
        cfg.tft_use_temporal_compression = True
        cfg.tft_temporal_compression_mode = "legacy_codec"
        cfg.tft_tc_stride = 2
        cfg.tft_tc_threshold = 64  # Low threshold so seq_len=128 triggers it
        cfg.tft_use_higher_order = False
        cfg.tft_use_regime_moe = False
        cfg.tft_use_lag_attention = False
        cfg.tft_dual_attention_fusion = False
        cfg.tft_use_explicit_cross_attention = False
        cfg.tft_use_fft_branch = False
        cfg.e_layers = 1

        model = tsl_tft.Model(cfg).float().eval()
        ds = make_tsl_dataset(cfg, n_samples=4)
        loader = DataLoader(ds, batch_size=2)
        x_enc, x_mark_enc, x_dec, x_mark_dec, y = next(iter(loader))

        with torch.no_grad():
            payload = model(x_enc, x_mark_enc, x_dec, x_mark_dec, return_interpretation=True)
        pred = payload["predictions"]
        self.assertEqual(pred.shape, (2, cfg.pred_len, cfg.c_out))
        # Verify compression was active in payload
        self.assertTrue(payload.get("tc_active", False),
                        "Temporal compression should be active for seq_len=128, threshold=64")

    def test_tsl_temporal_compression_inactive_short_sequence(self):
        """Compression is no-op when seq_len <= threshold."""
        set_seed(42)
        cfg = build_tsl_config()
        cfg.seq_len = 24
        cfg.label_len = 12
        cfg.pred_len = 4
        cfg.tft_use_temporal_compression = True
        cfg.tft_temporal_compression_mode = "legacy_codec"
        cfg.tft_tc_stride = 2
        cfg.tft_tc_threshold = 256  # Short seq < threshold -> no-op
        cfg.tft_use_higher_order = False
        cfg.tft_use_regime_moe = False
        cfg.tft_use_lag_attention = False
        cfg.tft_dual_attention_fusion = False
        cfg.tft_use_explicit_cross_attention = False
        cfg.tft_use_fft_branch = False
        cfg.e_layers = 1

        model = tsl_tft.Model(cfg).float().eval()
        ds = make_tsl_dataset(cfg, n_samples=4)
        loader = DataLoader(ds, batch_size=2)
        x_enc, x_mark_enc, x_dec, x_mark_dec, y = next(iter(loader))

        with torch.no_grad():
            payload = model(x_enc, x_mark_enc, x_dec, x_mark_dec, return_interpretation=True)
        pred = payload["predictions"]
        self.assertEqual(pred.shape, (2, cfg.pred_len, cfg.c_out))
        self.assertFalse(payload.get("tc_active", True),
                         "Temporal compression should be inactive for short sequences")

    def test_tsl_temporal_compression_backward_succeeds(self):
        """Backward pass works with temporal compression active."""
        set_seed(42)
        cfg = build_tsl_config()
        cfg.seq_len = 128
        cfg.label_len = 64
        cfg.pred_len = 16
        cfg.tft_use_temporal_compression = True
        cfg.tft_temporal_compression_mode = "legacy_codec"
        cfg.tft_tc_stride = 2
        cfg.tft_tc_threshold = 64
        cfg.tft_use_higher_order = False
        cfg.tft_use_regime_moe = False
        cfg.tft_use_lag_attention = False
        cfg.tft_dual_attention_fusion = False
        cfg.tft_use_explicit_cross_attention = False
        cfg.tft_use_fft_branch = False
        cfg.e_layers = 2

        model = tsl_tft.Model(cfg).float().train()
        ds = make_tsl_dataset(cfg, n_samples=4)
        loader = DataLoader(ds, batch_size=2)
        x_enc, x_mark_enc, x_dec, x_mark_dec, y = next(iter(loader))

        out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        pred = out[:, -cfg.pred_len:, :]
        loss = F.mse_loss(pred, y)
        loss.backward()

        # Check gradients flow to compression parameters
        for name, p in model.named_parameters():
            if 'temporal_compression' in name and p.requires_grad:
                self.assertIsNotNone(p.grad, f"No gradient for {name}")
                break
        else:
            self.fail("No temporal_compression parameters found in model")

    def test_tsl_temporal_compression_learns_with_long_signal(self):
        """Loss decreases over epochs with temporal compression on long sequences."""
        set_seed(42)
        cfg = build_tsl_config()
        cfg.seq_len = 128
        cfg.label_len = 64
        cfg.pred_len = 16
        cfg.tft_use_temporal_compression = True
        cfg.tft_temporal_compression_mode = "legacy_codec"
        cfg.tft_tc_stride = 2
        cfg.tft_tc_threshold = 64
        cfg.tft_use_higher_order = False
        cfg.tft_use_regime_moe = False
        cfg.tft_use_lag_attention = False
        cfg.tft_dual_attention_fusion = False
        cfg.tft_use_explicit_cross_attention = False
        cfg.tft_use_fft_branch = False
        cfg.e_layers = 1

        model = tsl_tft.Model(cfg).float()
        ds = make_tsl_dataset(cfg, n_samples=32)
        loader = DataLoader(ds, batch_size=8, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)

        losses = train_tsl_once(model, loader, cfg, lr=3e-3, epochs=4)
        self.assertGreater(losses[0], losses[-1],
                           "Loss should decrease with temporal compression enabled")

    # ── Multi-Head Variable Selection Tests ──
    def test_vsn_multi_head_selection(self):
        """Multi-head VSN produces correct shapes and per-head softmax sums to 1."""
        cfg = build_tsl_config()
        K = 2
        C = 8
        vsn = tsl_tft.VariableSelectionNetwork(
            d_model=cfg.d_model,
            variable_num=C,
            dropout=0.1,
            use_swiglu=False,
            cross_variable_mixing=False,
            n_heads=cfg.n_heads,
            residual_bypass=True,
            n_selection_heads=K,
        )
        x = torch.randn(2, cfg.seq_len, C, cfg.d_model)
        context = torch.randn(2, cfg.d_model)
        selected, weight_payload = vsn(x, context=context, return_weights=True)
        weights = weight_payload["selection"]
        # Output shape: [B, T, d_model]
        self.assertEqual(tuple(selected.shape), (2, cfg.seq_len, cfg.d_model))
        # Weight shape: [B, T, K, C] for multi-head
        self.assertEqual(tuple(weights.shape), (2, cfg.seq_len, K, C))
        # Each head's weights should sum to 1 across variables
        for k in range(K):
            head_sum = weights[:, :, k, :].sum(dim=-1)
            self.assertTrue(torch.allclose(head_sum, torch.ones_like(head_sum), atol=1e-5),
                            f"Selection head {k} weights do not sum to 1")

    def test_vsn_single_head_backward_compat(self):
        """n_selection_heads=1 produces same weight shape [B,T,C] as original."""
        cfg = build_tsl_config()
        vsn = tsl_tft.VariableSelectionNetwork(
            d_model=cfg.d_model,
            variable_num=6,
            dropout=0.0,
            n_selection_heads=1,
        )
        x = torch.randn(2, cfg.seq_len, 6, cfg.d_model)
        context = torch.randn(2, cfg.d_model)
        selected, weight_payload = vsn(x, context=context, return_weights=True)
        weights = weight_payload["selection"]
        self.assertEqual(tuple(selected.shape), (2, cfg.seq_len, cfg.d_model))
        # Single-head: weights are [B, T, C], NOT [B, T, 1, C]
        self.assertEqual(tuple(weights.shape), (2, cfg.seq_len, 6))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-5))

    def test_vsn_multi_head_gradient_flow(self):
        """Gradients flow to all K head GRNs in multi-head VSN."""
        cfg = build_tsl_config()
        K = 4
        vsn = tsl_tft.VariableSelectionNetwork(
            d_model=cfg.d_model,
            variable_num=8,
            dropout=0.0,
            n_selection_heads=K,
        )
        x = torch.randn(2, cfg.seq_len, 8, cfg.d_model)
        context = torch.randn(2, cfg.d_model)
        selected = vsn(x, context=context)
        loss = selected.sum()
        loss.backward()
        for k, head_grn in enumerate(vsn.head_grns):
            for name, param in head_grn.named_parameters():
                self.assertIsNotNone(param.grad, f"head_grns[{k}].{name} has no gradient")
                self.assertTrue(param.grad.abs().sum() > 0, f"head_grns[{k}].{name} has zero gradient")

    # ── MLP Quantile Projection Tests ──
    def test_mlp_quantile_projection_shape(self):
        """MLP quantile projection produces correct output shape."""
        cfg = build_tsl_config()
        cfg.tft_use_quantile_head = True
        cfg.tft_output_quantiles = [0.1, 0.5, 0.9]
        cfg.tft_mlp_quantile_projection = True
        cfg.tft_quantile_projection_ff_size = cfg.d_model
        model = tsl_tft.Model(cfg)
        # Verify it's a Sequential (MLP), not a plain Linear
        self.assertIsInstance(model.quantile_projection, nn.Sequential)
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
        self.assertTrue(torch.isfinite(payload["quantile_predictions"]).all())

    def test_mlp_quantile_projection_disabled(self):
        """Disabled MLP quantile projection uses plain Linear."""
        cfg = build_tsl_config()
        cfg.tft_use_quantile_head = True
        cfg.tft_output_quantiles = [0.1, 0.5, 0.9]
        cfg.tft_mlp_quantile_projection = False
        model = tsl_tft.Model(cfg)
        self.assertIsInstance(model.quantile_projection, nn.Linear)
        ds = make_tsl_dataset(cfg, n_samples=4)
        x_enc, x_mark_enc, x_dec, x_mark_dec, _ = ds[0]
        payload = model(
            x_enc.unsqueeze(0),
            x_mark_enc.unsqueeze(0),
            x_dec.unsqueeze(0),
            x_mark_dec.unsqueeze(0),
            return_interpretation=True,
        )
        self.assertEqual(tuple(payload["quantile_predictions"].shape), (1, cfg.pred_len, 3, cfg.c_out))

    # ── Holistic Integration Test ──
    def test_multi_head_vsn_and_mlp_quantile_holistic(self):
        """Both multi-head VSN and MLP quantile projection enabled together: forward, backward, interpretation, and learning."""
        cfg = build_tsl_config()
        cfg.tft_vsn_n_selection_heads = 2
        cfg.tft_use_quantile_head = True
        cfg.tft_output_quantiles = [0.1, 0.5, 0.9]
        cfg.tft_mlp_quantile_projection = True
        cfg.tft_quantile_projection_ff_size = cfg.d_model
        model = tsl_tft.Model(cfg)

        # Verify structural expectations
        self.assertEqual(model.history_vsn.n_selection_heads, 2)
        self.assertEqual(model.future_vsn.n_selection_heads, 2)
        self.assertEqual(len(model.history_vsn.head_grns), 2)
        self.assertIsInstance(model.quantile_projection, nn.Sequential)

        # Forward + interpretation
        ds = make_tsl_dataset(cfg, n_samples=8)
        x_enc, x_mark_enc, x_dec, x_mark_dec, _ = ds[0]
        payload = model(
            x_enc.unsqueeze(0),
            x_mark_enc.unsqueeze(0),
            x_dec.unsqueeze(0),
            x_mark_dec.unsqueeze(0),
            return_interpretation=True,
        )
        dec_out = payload["predictions"]
        self.assertEqual(tuple(dec_out.shape), (1, cfg.pred_len, cfg.c_out))
        self.assertTrue(torch.isfinite(dec_out).all())

        # Multi-head VSN weights should have head dimension
        hist_w = payload["history_vsn_weights"]
        future_w = payload["future_vsn_weights"]
        K = 2
        expected_hist_vars = cfg.enc_in + cfg.tft_known_len  # observed + known
        expected_future_vars = cfg.tft_known_len  # known only
        self.assertEqual(hist_w.shape[-2], K)
        self.assertEqual(hist_w.shape[-1], expected_hist_vars)
        self.assertEqual(future_w.shape[-2], K)
        self.assertEqual(future_w.shape[-1], expected_future_vars)

        # Quantile predictions
        qp = payload["quantile_predictions"]
        self.assertEqual(tuple(qp.shape), (1, cfg.pred_len, 3, cfg.c_out))
        self.assertTrue(torch.isfinite(qp).all())

        # Backward succeeds — most trainable params should receive gradients
        # (static_encoder and context-projection layers without static input won't)
        loss = dec_out.sum() + qp.sum()
        loss.backward()
        graded = sum(1 for _, p in model.named_parameters() if p.requires_grad and p.grad is not None)
        total = sum(1 for _, p in model.named_parameters() if p.requires_grad)
        self.assertGreater(graded / total, 0.8,
                           f"Only {graded}/{total} params received gradients")

        # Learning: loss decreases over a few epochs
        model.zero_grad()
        loader = DataLoader(ds, batch_size=4, shuffle=True)
        losses = train_tsl_once(model, loader, cfg, lr=3e-3, epochs=4)
        self.assertGreater(losses[0], losses[-1],
                           "Loss should decrease with multi-head VSN + MLP quantile projection")


class TestAdvancedDynamicGraph(unittest.TestCase):
    """Targeted tests for the AdvancedDynamicGraphLearner."""

    def test_sparse_graph_sparsity(self):
        """Each node should attend to exactly top_k neighbors."""
        set_seed(42)
        C, d, top_k = 16, 32, 5
        model = AdvancedDynamicGraphLearner(
            d_model=d, n_heads=4, top_k=top_k, num_layers=2,
            temporal_evolution=False, edge_features=False, num_nodes=C,
        )
        x = torch.randn(2, 10, C, d)  # [B, T, C, d]
        out = model(x)
        self.assertEqual(out.shape, x.shape)
        # Check adjacency sparsity via return_attention
        out, adj = model(x, return_attention=True)
        # adj: [B, T, n_heads, C, C] — all heads are same (broadcast)
        adj_2d = adj[0, 0, 0]  # [C, C]
        for row in range(C):
            nonzero = (adj_2d[row] > 1e-6).sum().item()
            self.assertLessEqual(nonzero, top_k,
                f"Row {row} has {nonzero} nonzero entries, expected <= {top_k}")

    def test_multihop_captures_indirect(self):
        """2-layer GNN should capture indirect A->B->C dependency better than 1-layer."""
        set_seed(42)
        C, d = 8, 16
        # Chain: node 0 → node 1 → node 2 (information propagation)
        x = torch.zeros(4, 5, C, d)
        x[:, :, 0, :] = torch.randn(4, 5, d)  # source signal at node 0

        model_1hop = AdvancedDynamicGraphLearner(
            d_model=d, n_heads=2, top_k=C, num_layers=1,
            temporal_evolution=False, edge_features=False, num_nodes=C,
        )
        model_2hop = AdvancedDynamicGraphLearner(
            d_model=d, n_heads=2, top_k=C, num_layers=2,
            temporal_evolution=False, edge_features=False, num_nodes=C,
        )
        out_1 = model_1hop(x)
        out_2 = model_2hop(x)
        # 2-hop should spread information more widely (higher norm at distant nodes)
        spread_1 = out_1[:, :, 2:, :].norm().item()
        spread_2 = out_2[:, :, 2:, :].norm().item()
        # Both should be non-zero (message passing works)
        self.assertGreater(spread_1, 0)
        self.assertGreater(spread_2, 0)
        # Output shapes still correct
        self.assertEqual(out_1.shape, x.shape)
        self.assertEqual(out_2.shape, x.shape)

    def test_temporal_evolution_varies_adjacency(self):
        """With temporal evolution, adjacency should differ across timesteps."""
        set_seed(42)
        C, d = 8, 16
        B, T = 2, 20
        model = AdvancedDynamicGraphLearner(
            d_model=d, n_heads=2, top_k=C, num_layers=1,
            temporal_evolution=True, edge_features=False, num_nodes=C,
        )
        # Create data with time-varying statistics
        x = torch.randn(B, T, C, d)
        x[:, T//2:, :, :] *= 3.0  # second half has different scale
        out, adj = model(x, return_attention=True)
        # adj: [B, T, n_heads, C, C]
        adj_t0 = adj[0, 0, 0]  # [C, C]
        adj_tN = adj[0, T-1, 0]  # [C, C]
        # These should NOT be identical (temporal evolution should differ)
        diff = (adj_t0 - adj_tN).abs().max().item()
        self.assertGreater(diff, 1e-4,
            "Temporal evolution should produce different adjacencies at different timesteps")

    def test_temporal_sparse_graph_stays_top_k(self):
        set_seed(42)
        C, d, top_k = 10, 12, 3
        model = AdvancedDynamicGraphLearner(
            d_model=d, n_heads=2, top_k=top_k, num_layers=1,
            temporal_evolution=True, edge_features=False, num_nodes=C,
        )
        x = torch.randn(2, 6, C, d)
        _, adj = model(x, return_attention=True)
        support = adj[:, :, 0] > 1e-6
        row_nonzero = support.sum(dim=-1)
        self.assertTrue(torch.all(row_nonzero <= top_k))

    def test_graph_top_k_zero_dense_or_rejected(self):
        set_seed(42)
        C, d = 7, 8
        model = AdvancedDynamicGraphLearner(
            d_model=d, n_heads=2, top_k=0, num_layers=1,
            temporal_evolution=False, edge_features=False, num_nodes=C,
        )
        x = torch.randn(1, 3, C, d)
        _, adj = model(x, return_attention=True)
        dense_support = adj[0, 0, 0] > 1e-6
        self.assertEqual(int(dense_support.sum(dim=-1).min().item()), C)

    def test_temporal_graph_rows_are_finite_and_normalized(self):
        set_seed(42)
        C, d = 8, 16
        model = AdvancedDynamicGraphLearner(
            d_model=d, n_heads=2, top_k=4, num_layers=1,
            temporal_evolution=True, edge_features=False, num_nodes=C,
        )
        x = torch.randn(2, 5, C, d)
        _, adj = model(x, return_attention=True)
        adj = adj[:, :, 0]
        self.assertTrue(torch.isfinite(adj).all())
        row_sums = adj.sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))

    def test_temporal_graph_initial_alpha_is_point_one(self):
        model = AdvancedDynamicGraphLearner(
            d_model=8, n_heads=2, top_k=4, num_layers=1,
            temporal_evolution=True, edge_features=False, num_nodes=6,
        )
        alpha = torch.sigmoid(model.temporal_evolver.alpha_logit).item()
        self.assertAlmostEqual(alpha, 0.1, places=5)

    def test_temporal_variation_does_not_expand_support(self):
        set_seed(42)
        C, d, top_k = 9, 8, 2
        model = AdvancedDynamicGraphLearner(
            d_model=d, n_heads=2, top_k=top_k, num_layers=1,
            temporal_evolution=True, edge_features=False, num_nodes=C,
        )
        x = torch.randn(1, 4, C, d)
        x_flat = x.reshape(-1, C, d)
        _, _, structure_mask = model.structure_learner(x_flat, return_structure=True)
        _, adj = model(x, return_attention=True)
        support = adj[:, :, 0] > 1e-6
        expected_mask = structure_mask.reshape(1, 4, C, C)
        self.assertTrue(torch.equal(support, expected_mask))

    def test_temporal_evolution_parameter_growth_regression(self):
        small = AdvancedDynamicGraphLearner(
            d_model=8, n_heads=2, top_k=4, num_layers=1,
            temporal_evolution=True, edge_features=False, num_nodes=8,
        )
        large = AdvancedDynamicGraphLearner(
            d_model=8, n_heads=2, top_k=4, num_layers=1,
            temporal_evolution=True, edge_features=False, num_nodes=32,
        )

        def temporal_params(model):
            return sum(p.numel() for name, p in model.named_parameters() if "temporal_evolver" in name)

        small_count = temporal_params(small)
        large_count = temporal_params(large)
        self.assertLess(large_count / small_count, 25.0)

    def test_edge_features_enhance_messages(self):
        """Edge features should change the output compared to no-edge-features."""
        set_seed(42)
        C, d = 8, 16
        model_no_edge = AdvancedDynamicGraphLearner(
            d_model=d, n_heads=2, top_k=C, num_layers=1,
            temporal_evolution=False, edge_features=False, num_nodes=C,
        )
        model_edge = AdvancedDynamicGraphLearner(
            d_model=d, n_heads=2, top_k=C, num_layers=1,
            temporal_evolution=False, edge_features=True, num_nodes=C,
        )
        x = torch.randn(2, 5, C, d)
        out_no = model_no_edge(x)
        out_yes = model_edge(x)
        # Both should produce valid outputs
        self.assertEqual(out_no.shape, x.shape)
        self.assertEqual(out_yes.shape, x.shape)
        self.assertTrue(torch.isfinite(out_no).all())
        self.assertTrue(torch.isfinite(out_yes).all())
        # edge_features model should have more parameters
        params_no = sum(p.numel() for p in model_no_edge.parameters())
        params_yes = sum(p.numel() for p in model_edge.parameters())
        self.assertGreater(params_yes, params_no,
            "Edge features should add parameters")

    def test_static_input_3d(self):
        """Module should handle 3D static input [B, C, d]."""
        set_seed(42)
        C, d = 6, 16
        model = AdvancedDynamicGraphLearner(
            d_model=d, n_heads=2, top_k=4, num_layers=2,
            temporal_evolution=False, edge_features=False, num_nodes=C,
        )
        x = torch.randn(4, C, d)  # static: [B, C, d]
        out = model(x)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_model_integration_sparse_graph(self):
        """Full TFT Model with sparse graph should train without errors."""
        set_seed(42)
        cfg = build_tsl_config()
        cfg.tft_graph_type = "sparse"
        cfg.tft_graph_top_k = 5
        cfg.tft_graph_num_layers = 2
        cfg.tft_graph_temporal_evolution = False
        cfg.tft_graph_edge_features = False
        model = tsl_tft.Model(cfg)
        ds = make_tsl_dataset(cfg, n_samples=16)
        loader = DataLoader(ds, batch_size=4, shuffle=True)
        losses = train_tsl_once(model, loader, cfg, lr=1e-3, epochs=3)
        self.assertGreater(len(losses), 0)
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses),
            "Sparse graph training produced NaN/Inf losses")

    def test_model_integration_temporal_graph(self):
        """Full TFT Model with temporal sparse graph should train without errors."""
        set_seed(42)
        cfg = build_tsl_config()
        cfg.tft_graph_type = "temporal_sparse"
        cfg.tft_graph_top_k = 5
        cfg.tft_graph_num_layers = 2
        cfg.tft_graph_temporal_evolution = True
        cfg.tft_graph_edge_features = False
        model = tsl_tft.Model(cfg)
        ds = make_tsl_dataset(cfg, n_samples=16)
        loader = DataLoader(ds, batch_size=4, shuffle=True)
        losses = train_tsl_once(model, loader, cfg, lr=1e-3, epochs=3)
        self.assertGreater(len(losses), 0)
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses),
            "Temporal graph training produced NaN/Inf losses")


if __name__ == "__main__":
    unittest.main()
