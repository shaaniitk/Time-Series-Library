#!/usr/bin/env bash

set -euo pipefail

# Advanced-feature matrix for native TFT on ETTh1 OT-only forecasting.
# This intentionally uses the production run.py / Exp_Long_Term_Forecast path.
#
# Design:
# - hold the dataset/task fixed on the healthiest observed setup so far:
#     features=MS, target=OT, seq_len=96, pred_len=24, c_out=1
# - start from the repaired extended_safe baseline
# - enable one advanced feature family at a time
# - keep model size modest to avoid confounding "feature effect" with obvious overfit
#
# Usage:
#   bash scripts/long_term_forecast/ETT_script/TFT_ETTh1_OT_feature_matrix.sh
#
# Optional:
#   export CUDA_VISIBLE_DEVICES=0

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

model_name=TemporalFusionTransformer

run_case() {
  local case_id="$1"
  shift

  echo
  echo "=================================================================="
  echo "Running case: ${case_id}"
  echo "=================================================================="

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id "${case_id}" \
    --model "${model_name}" \
    --data ETTh1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --features MS \
    --target OT \
    --freq h \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 1 \
    --tft_target_pos 6 \
    --d_model 16 \
    --n_heads 2 \
    --e_layers 1 \
    --d_layers 1 \
    --dropout 0.2 \
    --learning_rate 5e-5 \
    --lradj cosine \
    --train_epochs 30 \
    --batch_size 64 \
    --patience 6 \
    --loss MSE \
    --itr 1 \
    --des "${case_id}" \
    --tft_profile extended_safe \
    --tft_temporal_backbone lstm \
    --tft_temporal_backbone_layers 1 \
    --tft_temporal_kernel_size 3 \
    --tft_attention_dropout 0.08 \
    "$@"
}

# 0. Reference baseline
run_case tft_ot_p24_baseline

# 1. Probabilistic heads / loss modes
run_case tft_ot_p24_joint_quantile \
  --tft_use_quantile_head \
  --tft_output_mode joint

run_case tft_ot_p24_quantile_only \
  --tft_use_quantile_head \
  --tft_output_mode quantile \
  --loss Quantile

# 2. Attention-position/backends
run_case tft_ot_p24_alibi \
  --tft_attention_position_bias alibi

run_case tft_ot_p24_sdpa \
  --tft_attention_backend sdpa

# 3. Explicit cross-attention family
run_case tft_ot_p24_xattn_interp \
  --tft_use_explicit_cross_attention \
  --tft_cross_attention_type interpretable

# 4. Lag attention
run_case tft_ot_p24_lag \
  --tft_use_lag_attention \
  --tft_lag_scales 1,2,4

# 5. Graph/cross-variable mixing
run_case tft_ot_p24_crossmix_sparse \
  --tft_cross_variable_mixing \
  --tft_graph_type sparse \
  --tft_graph_top_k 3 \
  --tft_graph_num_layers 2

# 6. Spectral branch
run_case tft_ot_p24_fft \
  --tft_use_fft_branch \
  --tft_fft_modes 16 \
  --tft_fft_mode_select learned

# 7. Higher-order interactions
run_case tft_ot_p24_higher_order \
  --tft_use_higher_order \
  --tft_interaction_order 2 \
  --tft_interaction_rank 8

# 8. Regime-aware MoE
run_case tft_ot_p24_moe \
  --tft_use_regime_moe \
  --tft_num_regimes 3 \
  --tft_num_moe_experts 4 \
  --tft_moe_top_k 2 \
  --tft_moe_hidden_size 16 \
  --tft_moe_aux_loss_coeff 0.02

# 9. Temporal compression
run_case tft_ot_p24_temporal_compression \
  --tft_use_temporal_compression \
  --tft_tc_stride 2 \
  --tft_tc_threshold 64

# 10. Covariate reattention (requires cross-variable mixing signal path)
run_case tft_ot_p24_covariate_reattention \
  --tft_cross_variable_mixing \
  --tft_graph_type sparse \
  --tft_graph_top_k 3 \
  --tft_covariate_reattention

# 11. Experimental-full profile at the same small capacity, to see whether the
# full hardened extension bundle helps or just adds noise on this task.
run_case tft_ot_p24_experimental_profile \
  --tft_profile experimental_full \
  --d_model 16 \
  --n_heads 2 \
  --dropout 0.25 \
  --learning_rate 3e-5 \
  --tft_moe_hidden_size 16 \
  --tft_interaction_rank 8 \
  --tft_fft_modes 16 \
  --tft_graph_top_k 3 \
  --tft_tc_threshold 64
