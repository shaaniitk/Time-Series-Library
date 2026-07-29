#!/usr/bin/env bash

set -euo pipefail

# Production-path native TFT run for ETTh1 OT-only forecasting.
# This intentionally uses the real run.py / Exp_Long_Term_Forecast path
# rather than the handwritten deep_ett benchmark harness.
#
# Key choices:
# - OT-only target through features=MS, c_out=1, tft_target_pos=6
# - extended_safe TFT profile
# - small native TFT capacity
# - cosine LR schedule instead of run.py's default type1, which halves LR every epoch
# - no RevIN, because OT-only diagnostics showed no benefit from it here

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

model_name=TemporalFusionTransformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ETTh1_OT_stable_prod \
  --model "${model_name}" \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --features MS \
  --target OT \
  --freq h \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
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
  --des OTStableProd \
  --tft_profile extended_safe \
  --tft_temporal_backbone lstm \
  --tft_temporal_backbone_layers 1 \
  --tft_temporal_kernel_size 3 \
  --tft_attention_dropout 0.08
