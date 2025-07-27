# HierarchicalEnhancedAutoformer with Mixture of Experts (MoE): Full Workflow

## 1. Data Input and Preprocessing
- **Inputs:**
  - `x_enc`: Encoder input tensor [B, seq_len, enc_in] (historical targets + covariates)
  - `x_mark_enc`: Encoder time features/covariates [B, seq_len, ...]
  - `x_dec`: Decoder input tensor [B, label_len+pred_len, dec_in] (previous targets + covariates)
  - `x_mark_dec`: Decoder time features/covariates [B, label_len+pred_len, ...]
- **Preprocessing:**
  - Scaling, feature arrangement, and concatenation of covariates/targets are handled outside the model.
  - The model expects all features in a specific order; it does not distinguish covariates/targets internally.

## 2. Embedding and Decomposition
- **Embedding:**
  - `DataEmbedding_wo_pos` is applied to both encoder and decoder inputs, producing `enc_emb` and `dec_emb`.
  - All features (targets + covariates) are embedded into a `d_model`-dimensional space.
- **Decomposition:**
  - `LearnableSeriesDecomp` decomposes the decoder input into seasonal and trend components.
  - `seasonal_arg` and `trend_arg` are constructed for hierarchical processing.

## 3. Hierarchical Encoder and Decoder with MoE
- **Encoder:**
  - Receives multi-resolution features from a wavelet decomposer.
  - Each resolution is processed by a stack of encoder layers.
  - When MoE is enabled, encoder layers use `GatedMoEFFN`, which routes tokens to experts via `Top1Gating`.
  - Auxiliary (load balancing) loss is accumulated from all MoE layers.
- **Decoder:**
  - Similarly structured, with MoE-enabled decoder layers.
  - Processes both seasonal and trend paths; outputs are fused.
  - Accumulates auxiliary loss from all MoE layers.

## 4. Covariate and Target Treatment
- Covariates and targets are both present in the input tensors and time features.
- The model embeds all features together; decomposition operates on the full decoder input.
- The data pipeline ensures correct arrangement; the model is agnostic to feature type after input.
- All downstream layers (including MoE) operate on these combined representations.
- The final projection layer maps the fused output to the required number of target variables.

## 5. Loss Computation
- **Main Loss:** Computed between model output and true targets (e.g., MSE).
- **Auxiliary Loss:** Sum of all MoE load balancing losses from encoder and decoder.
- **Total Loss:** `main_loss + aux_weight * aux_loss` (aux_weight is a small scalar, e.g., 0.01).
- Auxiliary loss encourages balanced expert utilization without overwhelming the main objective.
- During inference, only the main output is returned.

## 6. Backward Propagation
- `backward()` is called on the total loss during training.
- Gradients flow through the entire model, including MoE gating, experts, and all hierarchical layers.
- Auxiliary loss provides a gradient signal to the gating network for balanced expert usage.
- All parameters (experts, router, etc.) are updated via the optimizer.
- Gradient clipping may be applied for stability.

## 7. Special MoE Considerations
- `Top1Gating` assigns each token to a single expert based on the highest gate score.
- Auxiliary loss penalizes imbalanced expert usage (fraction of tokens per expert vs. average softmax probability).
- If experts are underutilized, auxiliary loss increases, encouraging more even distribution.
- `aux_weight` must be tuned to avoid overwhelming the main loss.
- During inference, the router still selects experts, but auxiliary loss is not computed.

## 8. Output and Post-processing
- The final output is produced by projecting the fused seasonal and trend representations to the target dimension (`c_out`).
- During training, the model returns both output and auxiliary loss; during inference, only output.
- Output is typically rescaled and post-processed outside the model to obtain the final forecast.
- The model is agnostic to the arrangement of covariates/targets in the input, relying on the data pipeline for correct formatting.

---

**This document provides a complete, stepwise workflow for HierarchicalEnhancedAutoformer with MoE enabled, covering covariate/target treatment, forward and backward propagation, and all key architectural details.**
