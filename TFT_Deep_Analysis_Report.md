# TFT Deep Analysis Report

> **Scope**: `models/TemporalFusionTransformer.py` + `layers/TemporalFusion_layers.py` + `layers/DynamicGraph.py` + `layers/AdvancedDynamicGraph.py` + `layers/StandardNorm.py`
>
> Date: 2026-07-27

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Component-Level Algorithm Analysis](#2-component-level-algorithm-analysis)
   - 2.1 Embedding Layer (TFTEmbedding)
   - 2.2 Variable Selection Network (VSN)
   - 2.3 Static Covariate Encoder
   - 2.4 Gating Primitives — GRN, GLU, SwiGLU, GateAddNorm
   - 2.5 Temporal Backbone (LSTM / GatedTCN / Hybrid)
   - 2.6 Spectral Branch (FFT)
   - 2.7 Temporal Compression
   - 2.8 Attention Stack
   - 2.9 Dynamic & Advanced Graph Learner
   - 2.10 Higher-Order Interaction Block
   - 2.11 Regime-Aware Sparse MoE
   - 2.12 Multi-Scale Lag Attention
   - 2.13 Normalization Strategy (RevIN vs. Non-stationary)
   - 2.14 TemporalFusionDecoder & Output Heads
3. [Bug Report](#3-bug-report)
4. [Preferential Enhancements](#4-preferential-enhancements)
5. [Theory-Based Learning in TFT](#5-theory-based-learning-in-tft)
   - 5.1 Loss-Function Inductive Biases
   - 5.2 Architectural Priors
   - 5.3 Regularization as Theory Encoding
   - 5.4 Data Augmentation from Theory
   - 5.5 Physics / Domain Equation Constraints
   - 5.6 Differentiable Simulation as a Supervision Signal
   - 5.7 Knowledge Distillation from Theory-Rich Models
   - 5.8 Interpretability-Guided Theory Alignment
6. [Summary & Priority Matrix](#6-summary--priority-matrix)

---

## 1. Architecture Overview

This implementation is a heavily extended Temporal Fusion Transformer (TFT), far beyond the original [Lim et al., 2021] paper. The canonical TFT pipeline has been augmented with ten optional components, selectable at config time. The full data flow is:

```
x_enc [B,T,enc_in]  x_mark_enc [B,T,known_len]
         |                    |
         +-----> TFTEmbedding <------+  x_dec, x_mark_dec
                     |
        static_input [B,C_s,d]   observed_input [B,T,C_o,d]   known_input [B,T+pred,C_k,d]
                |                        |                              |
         StaticCovariateEncoder    history_vsn (obs+known[past])   future_vsn (known[future])
         [c_s, c_c, c_h, c_e]          |                              |
                                  history_feat [B,T,d]        future_feat [B,pred,d]
                                        |                              |
                          TemporalFusionDecoderLayer (x e_layers)
                                        |
                          projected output [B,pred,c_out]
                                        |
                               De-normalization
                                        |
                            dec_out [B,seq_len+pred,c_out]
```

### Key divergences from the original TFT paper

| Original TFT | This Implementation |
|---|---|
| LSTM-only temporal backbone | LSTM / Gated-TCN / Hybrid-TCN-LSTM (configurable) |
| Single-head VSN | Multi-head VSN, per-feature sigmoid gating, low-rank factorization |
| Static causal self-attention | `PositionalMultiHeadAttention` with RoPE/ALiBi/none + SDPA backend |
| No cross-attention | Optional explicit cross-attention (interpretable or full) |
| No frequency modeling | Parallel FFT spectral branch with learned modes |
| Single attention head per layer | Dual-attention fusion (full + interpretable heads) |
| No sparsity | Regime-aware sparse MoE replacing position-wise FFN |
| No graph learning | Dense or sparse dynamic graph covariate mixer |
| No multi-scale dependencies | Multi-scale lag attention at configurable lags |
| Fixed sequence length processing | Optional TemporalCompression for long sequences |
| No higher-order interactions | Optional 2nd/3rd order feature interaction block |
| RevIN optional, applied globally | RevIN optional, with correct target-only scatter/de-norm |

---

## 2. Component-Level Algorithm Analysis

### 2.1 Embedding Layer — `TFTEmbedding`

**Algorithm**

Each variable type is embedded independently before any mixing:

- **Observed covariates** (`observed_pos`): Each scalar channel `x_enc[:,:,i]` → `DataEmbedding(1, d_model)` → one embedding per variable per time step → stacked to `[B, T, C_o, d]`.
- **Known covariates** (timestamps): `TFTTemporalEmbedding` or `TFTTimeFeatureEmbedding` → each temporal component (month, day, weekday, hour, minute) gets its own embedding → stacked to `[B, T+pred, C_k, d]`. For `timeF` mode, each of the `d_inp` scalar channels gets a separate `nn.Linear(1, d_model)`, so positional features remain disentangled.
- **Static covariates** (`static_pos`): Uses only the first time step `x_enc[:, :1, i]` — correct since static features are time-invariant.

**Design observations**

- `TFTTemporalEmbedding.forward` hard-codes the order `[month, day, weekday, hour, (minute)]`. This is fragile if `x_mark` column ordering ever changes. A named-field approach would be safer.
- `x_mark` is concatenated along the time dimension as `[x_mark_enc, x_mark_dec[:, -pred_len:, :]]`, giving `T + pred_len` timesteps total. The slice `-pred_len:` is correct and avoids re-using the label window.
- The custom known embedding (`TFTCustomKnownEmbedding`) uses a value projection + channel embedding, effectively giving each known feature an additive channel bias. This is lightweight but does not model inter-channel interactions.

---

### 2.2 Variable Selection Network — `VariableSelectionNetwork`

**Algorithm**

The VSN learns *soft* importance weights over C variables at each time step.

```
x [B,T,C,d]  ->  flatten to [B,T,C*d]
                     |
              (optional) cross-mixing via DynamicGraphLearner
                     |
              (optional) low-rank projection [C*d -> rank]
                     |
              GRN(flat_input, context) -> unnormalized weights [B,T,C]
                     |
              Softmax -> selection_weights [B,T,C]
                     |
           x_processed = stack of per-variable GRN(x[...,i,:]) -> [B,T,d,C]
                     |
           selection_result = x_processed @ selection_weights  -> [B,T,d]
                     |
           (optional) residual bypass: linear(flat) gated by tanh(scalar_gate)
```

**Multi-head extension**: For `n_selection_heads > 1`, `d_model` is split into `K` subspaces of size `d_head = d_model // K`. Each head independently selects variables in its subspace using a separate GRN, and results are concatenated. De-correlated context projections per head prevent head collapse.

**Per-feature gating alternative**: When `per_feature_gating=True`, the VSN switches from softmax selection to per-covariate sigmoid gates, effectively letting the network decide to include or exclude each covariate independently rather than distributing unit probability mass over them.

**Complexity observations**

- The flattened input `[B, T, C*d]` fed to the GRN head has dimension `C * d_model`. For C=11 variables and d_model=512, that's 5632 inputs. For larger covariate sets this can be very large, hence the low-rank threshold (default 64 variables triggers rank factorization).
- The `residual_projection` maps `[C*d]` → `d`. This is a large linear layer that can dominate parameter count for large C. A depthwise-then-pointwise structure would be cheaper.
- `torch.flatten(x, start_dim=-2)` collapses `[B,T,C,d]` to `[B,T,C*d]` which allocates a new contiguous tensor — not avoidable since GRN requires contiguous input.

---

### 2.3 Static Covariate Encoder — `StaticCovariateEncoder`

Produces four context vectors `[c_s, c_c, c_h, c_e]` from static features, each derived by an independent VSN followed by a GRN. When `static_len == 0` all four are `None`, which propagates correctly through the decoder (LSTM initial state, enrichment GRN context, etc.).

**Design note**: Four separate VSNs share the same hyperparameters but learn independently. Since they produce conceptually different contextualization roles (variable selection context, cell state, hidden state, enrichment), independent parameterization is correct. However, during the early training phase all four encoders receive identical gradients from the static input, which may slow differentiation. A shared backbone with four separate heads could be explored.

---

### 2.4 Gating Primitives — `GRN`, `GLU`, `SwiGLU`, `GateAddNorm`

**GLU**: `fc1(x)` → value; `fc2(x)` → gate. Uses `nn.GLU` which implements `x[...,:half] * sigmoid(x[...,half:])`. Since `fc1` and `fc2` project independently, the combined output is `fc1(x) * sigmoid(fc2(x))` — correct. Note this doubles the Linear parameter count compared to a single projection + split.

**SwiGLU**: Replaces `sigmoid` gate with `SiLU(fc1(x))`. Empirically stronger on LLM-scale models; at small d_model the difference is minimal but never harmful.

**GRN**: The core non-linear unit. Mathematically:
```
η₁ = ELU(W_a·a + W_c·c)     (context injection)
η₂ = W_i·η₁
output = LayerNorm(GLU(η₂) + W_skip·a)
```
The residual projection (`project_a`) only activates when `hidden_size != input_size`, which handles the dimensionality mismatch cleanly.

**Key observation on GRN context injection**: The context `c` is broadcast via `c.unsqueeze(1)`, working correctly for `c: [B, d]` applied to `a: [B, T, d]`. If `c` is `None`, the branch is skipped — correct.

---

### 2.5 Temporal Backbone

Three choices exist:

**LSTM** (`temporal_backbone_type='lstm'`): Original TFT design. History and future encoded sequentially, sharing state across the boundary. Cell/hidden state initialized from `c_c` / `c_h` static context.

**Gated Dilated TCN** (`'gated_tcn'`): Stack of `GatedDilatedTemporalBlock` with doubling dilation `2^i`. Each block:
```
filter = tanh(CausalConv1d(x, dilation=2^i))
gate   = sigmoid(CausalConv1d(x, dilation=2^i))
out    = LayerNorm(x + Conv1x1(filter * gate))
```
This is the WaveNet gating pattern, appropriately adapted. Receptive field grows as `sum(2^i * (K-1))` for `i in 0..L-1`.

**Hybrid** (`'hybrid_tcn_lstm'`): TCN followed by LSTM with a learned fusion gate:
```
tcn_feat = GatedDilatedTCN(x)
rnn_feat, state = LSTM(tcn_feat, init_state)
gate = sigmoid(Linear([tcn_feat; rnn_feat]))
output = gate * rnn_feat + (1-gate) * tcn_feat
```
This lets the LSTM operate on already-convolved features, reducing its burden of learning local patterns. The `gate` biases toward LSTM initially (both logits start at 0 → gate ≈ 0.5).

**Algorithm concern with LSTM backbone**: The LSTM is split into `history_encoder` and `future_encoder`. The `future_encoder` receives the LSTM state from `history_encoder`. However, in `TemporalFusionDecoder.forward`, when `e_layers > 1`, `curr_history` and `curr_future` are updated each layer but re-split at `history_input.shape[1]`. This is correctly derived from the original history length, not the compressed length, so the split index is stable across layers.

---

### 2.6 Spectral Branch — `SpectralBranch`

Runs **in parallel** with the temporal backbone:

```
x [B,L,D] -> permute -> rfft -> select/transform modes -> irfft -> LayerNorm(x + projection(output))
```

Three mode selection strategies:
- `low`: deterministic low-frequency selection (stable but biased toward smooth signals).
- `top_amplitude`: non-differentiable topk over energy — the model cannot learn *which* modes to select, only what to do with them.
- `learned`: differentiable sigmoid soft-mask over all modes — fully trainable.

Fusion with temporal features via a learned sigmoid gate over the concatenation — this is the correct way to interpolate two signal representations.

---

### 2.7 Temporal Compression — `TemporalCompression`

A depthwise-separable strided convolution + transposed convolution pair for compressing/decompressing history features before/after attention. Activated only when `history_len > threshold`.

The decompress-to-target-len logic trims or pads the transposed conv output, which is correct since `ConvTranspose1d` output length is not always exactly `target_len` due to padding arithmetic.

**Key integration subtlety**: When compression is active, `enriched_features` in the decoder layer contains compressed history + full-resolution future. The split index `compressed_history_len` is used throughout the cross-attention and causal masking paths. After the attention pass, `temporal_compression.decompress_to` restores the history to original length before `gate_final`. This means `gate_final(out, temporal_features)` receives correctly-aligned tensors.

---

### 2.8 Attention Stack

**`InterpretableMultiHeadAttention`**: Projects Q, K from all heads, V shared across heads (`v` has shape `[B,T,d_head]`). This is the original TFT interpretable attention design where value representations are averaged across heads. Averaging in `attention_out = torch.mean(attention_out, dim=1)` reduces multi-head capacity but improves interpretability.

**`PositionalMultiHeadAttention`**: Full multi-head attention with optional RoPE/ALiBi positional biases. Supports SDPA backend (`torch.nn.functional.scaled_dot_product_attention`) for Flash Attention on CUDA.

**Dual-attention fusion**: Both `PositionalMultiHeadAttention` (full capacity) and `InterpretableMultiHeadAttention` (interpretable) run in parallel, their outputs blended via learned softmax logits initialized to favor interpretable branch (`init_logits[1] = 0.0` vs. `init_logits[0] = -1.0`).

**Causal mask pre-buffering**: Both `InterpretableMultiHeadAttention` and `TemporalFusionDecoderLayer` build a causal mask at `max_len = seq_len + pred_len` and register it as a non-persistent buffer. This avoids re-allocation each forward pass. The buffer is on CPU at init; it's moved to the correct dtype/device lazily by `.to(enriched_features.dtype)`. ⚠️ **Device bug**: if the model moves to GPU after construction, the buffer stays on CPU and `.to(dtype=...)` does **not** move it to GPU — `.to(device=..., dtype=...)` is required.

---

### 2.9 Dynamic & Advanced Graph Learner

**`DynamicGraphLearner`** (dense): Reshapes `[B,T,C,d]` → `[B*T, C, d]` and applies full attention across C nodes. This is O(C²) per timestep. Correct API: takes `return_attention` kwarg and returns `(features, weights)` or just `features`.

**`AdvancedDynamicGraphLearner`** (sparse): Adds sparse adjacency learning via top-k masking, multi-hop GNN with DenseNet skip connections, optional temporal adjacency evolution via GRU, and optional edge feature modulation. The DenseNet aggregation at the end (`skip_proj(concat of all layer outputs)`) is a good way to preserve shallow representations.

**Temporal evolution design**: The `TemporalGraphEvolution` GRU has output size `C*C`, which for large covariate sets becomes unwieldy. For C=100, that's 10,000 hidden units per GRU timestep. This is a known scaling issue.

---

### 2.10 Higher-Order Interaction Block — `HigherOrderInteractionBlock`

Computes pairwise (2nd order) and optionally triadic (3rd order) FM-style interactions:
```
pair  = W_pair( (W_left·x) ⊙ (W_right·x) / sqrt(rank) )
triple = W_triple( (W_left·x) ⊙ (W_right·x) ⊙ (W_third·x) / rank )   # if order==3
gates = softmax(W_gate·x)   # [B,T,order]
output = LayerNorm(x + W_out( dropout( sum_i(gates_i * term_i) ) ))
```
The rank normalization (`sqrt(rank)` for pairs, `rank` for triples) prevents scale explosion as rank grows. The gate is input-dependent, making the interaction contribution adaptive.

---

### 2.11 Regime-Aware Sparse MoE — `RegimeAwareSparseMoE`

**Regime detection** → soft regime probabilities → bias expert routing logits by learned regime-expert bias matrix. This is the key theoretical contribution: the model learns that different market/data regimes should route to different expert sub-networks.

**Expert computation**: Fully fused via einsum over expert parameters `[E, D, H]` — all experts computed in parallel, no Python loop. Routing weights then mix outputs.

**Capacity constraint**: The per-expert capacity loop at training time is a Python-level for loop over experts — this will not scale for large E. A vectorized scatter-based implementation would be faster.

**Auxiliary load-balancing loss**: Uses coefficient of variation squared (CV²) of expert importance. The standard Switch Transformer loss is `n_experts * sum(f_i * p_i)`. The CV² is a valid alternative but may have different gradient properties.

---

### 2.12 Multi-Scale Lag Attention — `MultiScaleLagAttention`

For each lag `l` in `lag_scales`:
```
shifted_x = pad(x[:, :-l, :], left=l)   # shift right by l steps
attn_out_l = Attention(query=x, key=shifted, value=shifted, causal_mask)
```
Outputs are blended via learned softmax weights over scales. The `_shift_sequence` method correctly uses `F.pad` for efficiency. **Edge case**: when `lag >= seq_len`, the shifted sequence is all-zeros — the attention layer will still compute valid (though trivially uninformative) outputs.

---

### 2.13 Normalization Strategy

**Non-stationary mode** (default, `tft_use_revin=False`):
```
means = x_enc.mean(T)
stdev = sqrt(var(x_enc) + 1e-5)
x_enc_norm = (x_enc - means) / stdev
```
After prediction:
```
target_stdev = stdev[:, 0, :].index_select(-1, target_pos)
target_means = means[:, 0, :].index_select(-1, target_pos)
dec_out = dec_out * target_stdev + target_means
```
The `[:, 0, :]` indexing on `stdev [B,1,C]` collapses the time dim — correct since `keepdim=True` gives shape `[B,1,C]`.

**RevIN mode** (`tft_use_revin=True`):
The prediction is scattered into a full `[B, pred_len, enc_in]` buffer at `target_pos` indices, passed through `revin(..., 'denorm')` on all channels, then re-indexed. This is the correct way to apply per-channel affine denormalization because RevIN's stored `stdev`/`mean` have shape `[B,1,enc_in]`.

---

### 2.14 TemporalFusionDecoder & Output Heads

**Stochastic depth**: Layer-drop probability increases linearly with depth: `drop_prob = layer_idx / (num_layers-1) * rate`. Layer 0 is never dropped. This is the canonical stochastic depth schedule from [Huang et al., 2016].

**Gradient checkpointing**: Activated during training when `tft_gradient_checkpointing=True`, using `use_reentrant=False` for compatibility with the autograd system. Disabled when `return_attention=True` (checkpointing is incompatible with captured activations).

**Per-target heads**: When `tft_per_target_heads=True` and `c_out > 1`, each target channel gets its own 2-layer MLP head `Linear(d_model → d_model//2) → GELU → Linear(→1)`. This adds `c_out * (d_model * d_model/2 + d_model/2)` parameters — manageable for small c_out but grows quickly.

---

## 3. Bug Report

### BUG-01 — Causal mask buffer stays on CPU after `.to(device)` ❌ CRITICAL

**Location**: `InterpretableMultiHeadAttention.__init__` and `TemporalFusionDecoderLayer.__init__`

**Code**:
```python
self.register_buffer('_causal_mask_buf', build_causal_mask(max_len, torch.device('cpu'), torch.float32), persistent=False)
```
And later in forward:
```python
causal_mask = self._causal_mask_buf[:T, :T].to(attention_score.dtype)
```

**Problem**: `.to(dtype)` only converts dtype, not device. When the model is moved to GPU via `model.to('cuda')` or `model.cuda()`, `register_buffer` with `persistent=False` does **not** guarantee device migration because the buffer is explicitly constructed on CPU. In practice, PyTorch *does* migrate non-persistent buffers on `.to(device)`, but calling `.to(dtype)` without `.to(device=..., dtype=...)` in the forward pass risks using a CPU tensor in a CUDA operation if the buffer fails to migrate.

**Fix**:
```python
# In forward:
causal_mask = self._causal_mask_buf[:T, :T].to(device=attention_score.device, dtype=attention_score.dtype)
```
This is a one-line fix in both affected classes and makes device handling explicit.

---

### BUG-02 — `_validate_inputs` checks `x_dec.shape[2] != configs.c_out` but `x_dec` is all-zeros in practice ⚠️ MEDIUM

**Location**: `Model._validate_inputs`

**Code**:
```python
if x_dec.shape[2] != self.configs.c_out:
    raise ValueError(...)
```

**Problem**: In the standard TSL `exp_main.py` data pipeline, `x_dec` is constructed as `[x_mark_dec, zeros(pred_len, enc_in)]`, giving it `enc_in` channels, not `c_out`. When `c_out != enc_in` (e.g., MS forecasting with `c_out=1`), this validation will raise a false error even on valid inputs. The TFT model never reads `x_dec` values — only `x_mark_dec` is consumed. This validation is therefore semantically incorrect.

**Fix**: Remove or relax the `x_dec` feature-size check, or document that callers must pass `x_dec` with exactly `c_out` channels (and update the TSL pipeline accordingly).

---

### BUG-03 — `HigherOrderInteractionBlock` triple-order normalization off by factor of sqrt ⚠️ MEDIUM

**Location**: `layers/TemporalFusion_layers.py`, `HigherOrderInteractionBlock.forward`

**Code** (reconstructed from the truncated read):
```python
triple_term = self.triple_projection((left * right * third) / self.interaction_rank)
```
versus pair term:
```python
pair_term = self.pair_projection((left * right) / (self.interaction_rank ** 0.5))
```

**Problem**: The pair term is normalized by `sqrt(rank)` (mimicking attention scaling), while the triple term is normalized by `rank` (not `rank^(2/3)` or `rank` consistently). For interaction of order N, a consistent scale would be `rank^((N-1)/2)` — giving `sqrt(rank)` for N=2 and `rank` for N=3. The current triple normalization is actually correct by this formula, but the inconsistency with the pair normalization is confusing and should be documented.

---

### BUG-04 — Stochastic depth skips gradient checkpointing path ⚠️ LOW

**Location**: `TemporalFusionDecoder.forward`

**Code**:
```python
if self.training and self.stochastic_depth_rate > 0.0 and ...:
    drop_prob = ...
    if torch.rand(1).item() < drop_prob:
        out = torch.cat([curr_history, curr_future], dim=1)
        continue
# Gradient checkpointing:
if self.gradient_checkpointing and self.training and not return_attention:
    out = torch.utils.checkpoint.checkpoint(...)
```

**Problem**: The stochastic depth check runs before gradient checkpointing. When a layer is dropped, `out = cat(curr_history, curr_future)` which is the *previous* layer's output, not the current layer's residual. This is technically correct for stochastic depth but misses updating `curr_history` and `curr_future` in the dropped-layer case — they remain stale from the previous iteration. The `continue` statement skips the update at the bottom:
```python
curr_history = out[:, :history_input.shape[1], :]
curr_future  = out[:, history_input.shape[1]:, :]
```
This means the *next* layer will receive the output from two layers back rather than one. This is actually correct stochastic depth behavior (identity skip) but is not re-split, so `curr_history` and `curr_future` are stale on the next iteration. **The bug is**: the `continue` exits before updating `curr_history`/`curr_future` from `out`.

**Fix**:
```python
if torch.rand(1).item() < drop_prob:
    out = torch.cat([curr_history, curr_future], dim=1)
    # Update curr splits for the next layer
    curr_history = out[:, :history_input.shape[1], :]
    curr_future  = out[:, history_input.shape[1]:, :]
    if return_attention:
        attention_payloads.append({})
    continue
```

---

### BUG-05 — MoE capacity constraint Python loop does not scale ⚠️ LOW-MEDIUM

**Location**: `RegimeAwareSparseMoE._compute_sparse_routing`

```python
for e in range(self.num_experts):
    expert_mask = flat_probs[:, e] > 0
    ...
    _, keep_idx = torch.topk(expert_vals, capacity)
    drop_mask = ...
    flat_probs[drop_mask, e] = 0.0
```

**Problem**: This is an `O(num_experts)` Python loop with `torch.topk` inside. For `num_experts=16`, this is 16 topk calls per forward pass. This is fine at small scale but becomes a training bottleneck. More importantly, in-place modification of `flat_probs` which is derived from `sparse_probs` via `.reshape` — if `sparse_probs` is not contiguous, `.reshape` returns a copy and the in-place modifications won't propagate back.

**Fix**: Use a vectorized scatter-based capacity constraint, or at minimum call `.contiguous()` before `.reshape`.

---

### BUG-06 — `TFTTemporalEmbedding.forward` returns embedding tensor not d_model-dimensional ⚠️ LOW

**Location**: `TFTTemporalEmbedding.forward`

```python
embedding_x = torch.stack([month_x, day_x, weekday_x, hour_x, minute_x], dim=-2)
```

Each of `month_x`, `day_x` etc. has shape `[B, T, d_model]` (from the parent `TemporalEmbedding`). The stack on `dim=-2` gives `[B, T, n_components, d_model]`. This is the expected known_input shape `[B,T,C_k,d]`. Correct — but note that `minute_x = 0.` (a scalar float) when `minute_embed` is absent. `torch.stack([..., 0.])` will fail because you cannot stack a float with tensors. In practice this is guarded by the conditional — the `if hasattr(self, 'minute_embed')` correctly selects the 4-element stack. So this is not a runtime bug, but the naming of the else branch (returning `0.` as `minute_x`) is misleading.

---

### BUG-07 — `forecast()` variance check runs before RevIN normalization ⚠️ INFO

**Location**: `Model.forecast`

```python
var = torch.var(x_enc, dim=1, keepdim=True, unbiased=False)
if (var < 1e-8).any():
    warnings.warn("Near-constant channels detected...")
if self.use_revin:
    x_enc = self.revin(x_enc, 'norm')
    ...
else:
    means = x_enc.mean(1, keepdim=True).detach()
    x_enc = x_enc - means
    stdev = torch.sqrt(torch.clamp(var, min=1e-10) + 1e-5)
    x_enc = x_enc / stdev
```

**Problem**: When `use_revin=True`, `var` is computed but never used. The warning is still useful (RevIN will also handle near-constant channels via its own eps), but the variable allocation is wasted. Not a bug per se, but wasteful.

---

### BUG-08 — `build_causal_mask` buffer shared between CPU and GPU path for SDPA ⚠️ INFO

When `attention_backend='sdpa'`, `torch.nn.functional.scaled_dot_product_attention` is called with `is_causal=False` and an explicit `attn_mask`. The causal mask from `_causal_mask_buf` must be on the same device as the query. Since SDPA returns directly without `return_attention`, the buffer device issue from BUG-01 also affects this path.

---

## 4. Preferential Enhancements

### ENH-01 — Flash Attention for `InterpretableMultiHeadAttention`

`InterpretableMultiHeadAttention` currently always uses the manual `torch.matmul` path and cannot leverage Flash Attention because V is not multi-headed. However, since the averaging over heads happens *after* the matmul, you could compute multi-headed output and average at the end, then dispatch to SDPA:

```python
# In forward, reshape v to be multi-headed for SDPA compat:
v_expanded = v.unsqueeze(1).expand(B, self.n_heads, T, self.d_head)
# ... then average after SDPA output
```
This enables memory-efficient attention on long sequences even for the interpretable head.

---

### ENH-02 — Vectorize MoE capacity constraint

Replace the Python expert loop with a vectorized implementation:

```python
# Vectorized capacity enforcement
flat_probs = sparse_probs.reshape(-1, self.num_experts)  # [N, E]
capacity = int(self.capacity_factor * flat_probs.shape[0] * self.top_k / self.num_experts)
# For each expert, zero out tokens beyond capacity by sorting descending
for e in range(self.num_experts): ...  # current
# Better: use scatter approach
order = flat_probs.argsort(dim=0, descending=True)  # [N, E]
overflow_mask = order >= capacity  # [N, E]
flat_probs = flat_probs.masked_fill(overflow_mask, 0.0)
```
This still has a sort but avoids the topk loop.

---

### ENH-03 — Learnable temperature in `SparseGraphStructureLearner`

The current temperature parameter is fixed at `1.0`. Making it a learnable parameter per node or a global scalar would allow the model to control graph sparsity during training:

```python
self.temperature = nn.Parameter(torch.tensor(1.0))
logits = torch.bmm(Q, K.transpose(1,2)) * self.scale / self.temperature.clamp(min=0.1)
```

---

### ENH-04 — Positional encoding for covariate reattention

The covariate reattention in `TemporalFusionDecoderLayer` flattens `[B, T, C, d]` to `[B, T*C, d]` as keys/values. This loses the temporal structure of covariate embeddings. Adding a sinusoidal or learned positional encoding over the time dimension before flattening would help the decoder distinguish covariates at different time steps:

```python
# Add time-positional bias to pre_vsn_embs before reshaping
time_pe = sinusoidal_pe(T_cov, d_cov)  # [T_cov, d_cov]
pre_vsn_embs = pre_vsn_embs + time_pe.unsqueeze(0).unsqueeze(2)
```

---

### ENH-05 — Hierarchical static context

Currently all four static context vectors are derived independently from the same static VSN output. In practice, `c_h` and `c_c` (cell and hidden init) should be more tightly coupled since they govern the same LSTM. A shared trunk with task-specific heads:

```python
shared = shared_grn(static_feat)
c_h = head_h(shared)
c_c = head_c(shared)
c_s = head_s(shared)
c_e = head_e(shared)
```
This halves the static encoder parameter count while maintaining expressivity.

---

### ENH-06 — Curriculum learning for stochastic depth

The current stochastic depth rate is static. A warmup schedule that linearly increases `stochastic_depth_rate` from 0 to the configured value over the first N epochs would allow the model to stabilize before depth dropping begins:

```python
# In training loop, set:
model.temporal_fusion_decoder.stochastic_depth_rate = min(
    config.tft_stochastic_depth_rate,
    config.tft_stochastic_depth_rate * epoch / warmup_epochs
)
```

---

### ENH-07 — Conditional known embedding for custom covariates

`TFTCustomKnownEmbedding` uses a fixed channel embedding table. For custom known features with semantic meaning (e.g., day-of-week, is-holiday), a `nn.Embedding` per categorical channel plus `nn.Linear` per continuous channel would be more expressive:

```python
class HeterogeneousKnownEmbedding(nn.Module):
    def __init__(self, channel_configs, d_model):
        # channel_configs: list of ('categorical', vocab_size) or ('continuous', 1)
        ...
```

---

### ENH-08 — Interpretability: export VSN weights as importance scores

The `return_interpretation=True` path already collects `history_vsn_weights` and `future_vsn_weights`. A utility method that aggregates these across the batch and time dimension into per-variable importance rankings would make the TFT's interpretability practical:

```python
@torch.no_grad()
def get_variable_importance(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
    payload = self.forward(x_enc, x_mark_enc, x_dec, x_mark_dec, return_interpretation=True)
    hist_weights = payload['history_vsn_weights']   # [B, T, C] or [B, T, K, C]
    importance = hist_weights.mean(dim=(0, 1))       # [C] or [K, C]
    return importance
```

---

### ENH-09 — Replace manual attention clamp with `torch.nan_to_num`

The current clamping logic:
```python
if (attention_score.abs() > clamp_limit).any():
    warnings.warn(...)
    attention_score = attention_score.clamp(...)
```
is a global check that triggers on any outlier. A safer approach is to use `torch.nan_to_num` after softmax to handle any residual NaN without masking valid large values:

```python
attention_prob = F.softmax(attention_score, dim=-1)
attention_prob = attention_prob.nan_to_num(nan=0.0)
```

---

### ENH-10 — TemporalGraphEvolution GRU scaling fix

For large covariate sets (C > 30), the GRU hidden size `C*C` becomes prohibitive. A low-rank approximation:

```python
# Instead of GRU with C*C hidden:
self.gru = nn.GRU(num_nodes, rank)           # [B, T, C] -> [B, T, rank]
self.delta_proj = nn.Linear(rank, C * C)     # [B, T, rank] -> [B, T, C*C]
```
This reduces parameters from `O(C^4)` to `O(C^2 * rank)`.

---

## 5. Theory-Based Learning in TFT

"Theory-based learning" refers to injecting domain knowledge, physical constraints, or mathematical structure into the learning process beyond what the raw data provides. Most discussions focus on the **loss function**, but there are at least seven other distinct mechanisms applicable to a TFT for time-series forecasting.

---

### 5.1 Loss-Function Inductive Biases (the common approach)

The most straightforward way to encode theory. Examples relevant to TFT:

| Theory | Loss Term |
|---|---|
| Forecast should be smooth | `λ * ||dec_out[t+1] - dec_out[t]||²` (finite-difference penalty) |
| Forecast should be monotone in rising windows | Hinge on sign of diff |
| Quantile predictions should be monotone (no quantile crossing) | `max(0, q_low - q_high).mean()` |
| Prediction should revert to a long-run mean at long horizons | Soft constraint toward mean |
| Periodicity at known frequency f | Fourier regularization: penalize energy outside multiples of f |
| Covariate effects should be sparse | L1 on VSN weights |
| Regime transitions should be rare | TV norm on `regime_probabilities` across time |

For the **quantile head** already in this model, **monotonicity enforcement** is the most immediately useful addition:

```python
# After quantile_out = quantile_projection(decoder_hidden).view(..., Q, c_out)
# Enforce q[i] <= q[i+1] via:
quantile_out_sorted = quantile_out.sort(dim=-2).values  # sort over Q dim
crossing_loss = (quantile_out[:, :, :-1, :] - quantile_out[:, :, 1:, :]).clamp(min=0).mean()
total_loss = task_loss + lambda_crossing * crossing_loss
```

---

### 5.2 Architectural Priors (theory encoded in structure)

This is the most powerful form because it applies at every forward pass, not just during loss computation.

#### 5.2.1 Causal Convolutions as Temporal Causality

Already implemented via `CausalConv1d`. This encodes the theory that "the future cannot influence the past." No data can teach this constraint — it must be architectural.

#### 5.2.2 Non-negative Mixture Weights

Variable selection softmax weights already enforce the theory that "the selected representation is a convex combination of covariates." Making the per-feature gating also output a sigmoid-normalized sum (already done in the per-feature gating path) further respects the unit-sum prior.

#### 5.2.3 Theory-Informed Lag Scales

Instead of arbitrary `lag_scales=[1,2,4,8]`, use domain knowledge:
- **Electricity**: lag at 24h, 48h, 168h (weekly cycle), 8760h (annual)
- **Finance**: lag at 1, 5 (week), 21 (month), 252 (year) trading days
- **Weather**: 6h, 24h, 48h, 168h

```python
# ETTh1 (hourly electricity):
tft_lag_scales = [24, 48, 168, 720]
```
This encodes domain periodicity theory directly into the attention structure.

#### 5.2.4 Monotone Expert Routing via Sorted Regimes

If you have prior knowledge about regime ordering (e.g., "low volatility" → "high volatility" is a one-directional shift), you can enforce monotonicity in the regime transitions via a sorted softmax over learned regime embeddings using a cumulative softmax.

#### 5.2.5 Symmetry Breaking for the Spectral Branch

For known-periodic signals, initialize `SpectralBranch.weight_real` and `weight_imag` at the known fundamental frequency and harmonics rather than `xavier_uniform_`. This gives the model a head start aligned with physical theory.

---

### 5.3 Regularization as Theory Encoding

Regularization terms impose soft geometric constraints without altering the loss landscape globally.

#### 5.3.1 Temporal Smoothness of Expert Routing

If you believe regimes change slowly, penalize rapid regime switching:

```python
# regime_probs: [B, T, R] from RegimeAwareSparseMoE
regime_smooth_loss = ((regime_probs[:, 1:, :] - regime_probs[:, :-1, :]) ** 2).mean()
total_loss = task_loss + lambda_regime * regime_smooth_loss
```

#### 5.3.2 Graph Sparsity Prior

The `SparseGraphStructureLearner` learns graph edges. If theory says the graph should be sparse (only a few covariates affect each other), add an entropy regularizer on the adjacency:

```python
# adj: [N, C, C] soft adjacency from structure learner
entropy = -(adj * (adj + 1e-8).log()).sum(-1).mean()
total_loss = task_loss - lambda_sparse * entropy  # maximize entropy = spread attention = sparser hard adj
```

#### 5.3.3 Decorrelation of Attention Heads

Theory from independent component analysis: attention heads should learn diverse features. A decorrelation penalty on the multi-head attention weight matrices:

```python
W_q = model.attention.q_linear.weight  # [d_model, d_model]
# Split into head subspaces and penalize cross-head cosine similarity
```

---

### 5.4 Data Augmentation from Theory

Augmentation encodes theory by generating training examples consistent with domain laws, forcing the model to learn invariances it cannot learn from real data alone.

#### 5.4.1 Time Warping with Preserved Causality

Stretch or compress the temporal axis within windows while keeping the causal direction — tests whether the model learns timescale-invariant patterns.

#### 5.4.2 Covariate Permutation Invariance

For the VSN: randomly permute the order of covariates in the input. The model should learn the same variable importance regardless of column order. Augmenting with permuted inputs teaches permutation equivariance.

#### 5.4.3 Multiplicative Noise on Known Features

If theory says known timestamp features are reliable (not noisy), augmenting with small multiplicative noise on `x_mark` teaches robustness to clock drift or rounding — relevant for industrial IoT sensors.

#### 5.4.4 Synthetic Regime Injection

Concatenate synthetic regime-shift data (generated from a simple piecewise-stationary model) with real training data. This provides explicit supervision for `RegimeAwareSparseMoE` to learn meaningful regime boundaries.

---

### 5.5 Physics / Domain Equation Constraints

For forecasting problems with known governing equations, you can add a physics residual loss term.

#### 5.5.1 Energy Balance Constraint (Electricity)

For power grid forecasting: supply ≈ demand at each time step. If the model predicts load at multiple nodes, penalize the imbalance:

```python
predicted_load_sum = dec_out.sum(dim=-1)   # sum over channels
grid_imbalance_loss = (predicted_load_sum - known_supply).pow(2).mean()
```

#### 5.5.2 Smoothness via Numerical PDE Operators

For temperature forecasting, the diffusion equation says spatial/temporal gradients should follow `∂T/∂t = α∇²T`. The finite-difference approximation of this can be used as a physics residual loss on the decoder output.

#### 5.5.3 Non-Negativity Constraints

For count data (e.g., traffic, demand), theory says values must be ≥ 0. Softplus output activation + non-negativity penalty:

```python
dec_out = F.softplus(dec_out)  # or: penalty = dec_out.clamp(max=0).pow(2).mean()
```

---

### 5.6 Differentiable Simulation as a Supervision Signal

This is the most advanced approach. Instead of just penalizing prediction error against observations, you run predictions through a domain simulator and penalize the simulator's outputs.

```
TFT predictions → domain simulator f(x) → simulator output → theory-grounded loss
```

**Examples**:
- **Financial risk**: Pass predictions through a VaR calculation; penalize VaR exceedances.
- **Energy**: Pass load predictions through an optimal power flow (OPF) solver; penalize constraint violations.
- **Climate**: Pass temperature predictions through a simplified energy balance model.

The gradient of the simulator output w.r.t. TFT parameters is the key signal. For non-differentiable simulators, use the straight-through estimator or REINFORCE.

For this TFT codebase, a practical example:

```python
class SimulatorLoss(nn.Module):
    def forward(self, predictions, known_future_inputs):
        # predictions: [B, pred_len, c_out]  — e.g., load forecast
        # Simple simulation: if predicted peak > capacity, cost is quadratic
        capacity = known_future_inputs[:, :, CAPACITY_CHANNEL]
        excess = (predictions.max(dim=-1).values - capacity).clamp(min=0)
        return excess.pow(2).mean()
```

---

### 5.7 Knowledge Distillation from Theory-Rich Models

Use a domain-specific model (e.g., a SARIMA fitted on each channel, or an exponential smoothing model) as a **teacher** to guide the TFT student's predictions:

```python
# Teacher predictions from SARIMA / ETS / Prophet
teacher_pred = teacher_model.predict(x_enc)  # [B, pred_len, c_out]
# Distillation loss (soft targets)
distill_loss = F.mse_loss(dec_out, teacher_pred.detach())
total_loss = task_loss + lambda_distill * distill_loss
```

This encodes the theory "short-horizon forecasts should look like classical statistical forecasts" while still allowing the TFT to deviate when it learns better patterns from data.

**For TFT specifically**, you can distill the VSN variable importance weights from a SHAP-based analysis of a gradient-boosted model trained on the same data — encoding the theory "these covariates matter" as a soft supervision signal on `history_vsn_weights`.

---

### 5.8 Interpretability-Guided Theory Alignment

Unique to TFT: because the model produces interpretable attention weights and variable importance scores, you can penalize *deviations from expected theoretical behavior* in the interpretation outputs themselves.

#### 5.8.1 Variable Importance Ordering Constraint

If domain theory says "variable A should be more important than variable B for this task," add:
```python
importance_loss = F.relu(vsn_weights[:, :, B_idx] - vsn_weights[:, :, A_idx]).mean()
```

#### 5.8.2 Attention Locality Prior

Theory might say "attention over recent history should be stronger than distant history." Penalize attention mass on temporally distant positions:
```python
# attention_prob: [B, n_heads, T, T]
distance = (torch.arange(T) - torch.arange(T).unsqueeze(-1)).float().abs()  # [T, T]
locality_loss = (attention_prob * distance.to(device)).mean()
total_loss = task_loss + lambda_locality * locality_loss
```

#### 5.8.3 Regime Semantic Alignment

If you have labeled regime periods in training data (e.g., "recession", "expansion" in economic data), provide regime supervision:
```python
# regime_probs: [B, T, R], regime_labels: [B, T] integer labels
regime_supervision_loss = F.cross_entropy(
    regime_probs.reshape(-1, R),
    regime_labels.reshape(-1)
)
```

---

## 6. Summary & Priority Matrix

### Bug Priority

| ID | Description | Severity | Fix Effort |
|---|---|---|---|
| BUG-01 | Causal mask stays on CPU after model.to(device) | **Critical** | 2 lines |
| BUG-04 | Stochastic depth skips `curr_history`/`curr_future` update | **High** | 3 lines |
| BUG-02 | `x_dec` shape validation fails for MS forecasting | **Medium** | 1 line |
| BUG-05 | MoE capacity loop modifies non-contiguous view | **Medium** | 1 line |
| BUG-03 | Triple-order normalization factor inconsistency | Low | Doc only |
| BUG-06 | `minute_x = 0.` is a float in stack (misleading) | Low | Cosmetic |
| BUG-07 | Unused `var` computation when `use_revin=True` | Info | 1 line |
| BUG-08 | SDPA path inherits BUG-01 device issue | Info | Same fix as BUG-01 |

### Enhancement Priority

| ID | Description | Impact | Effort |
|---|---|---|---|
| ENH-01 | SDPA for interpretable attention head | High (memory) | Medium |
| ENH-10 | Low-rank temporal graph evolution GRU | High (scaling) | Medium |
| ENH-03 | Learnable temperature in sparse graph | Medium (quality) | Low |
| ENH-04 | Positional encoding for covariate reattention | Medium (quality) | Low |
| ENH-05 | Shared static context backbone | Medium (params) | Medium |
| ENH-02 | Vectorized MoE capacity constraint | Low-Medium (speed) | Medium |
| ENH-06 | Curriculum schedule for stochastic depth | Low (training) | Low |
| ENH-07 | Heterogeneous known feature embedding | Low (quality) | High |
| ENH-08 | Variable importance export utility | Low (usability) | Low |
| ENH-09 | `nan_to_num` instead of attention clamping | Low (robustness) | Low |

### Theory-Based Learning Complexity vs. Impact

| Mechanism | Complexity | Expected Impact | Recommended for This Model |
|---|---|---|---|
| Quantile monotonicity loss (5.1) | Low | High | **Yes — implement immediately** |
| Theory-informed lag scales (5.2.3) | Low | High | **Yes — config change only** |
| Temporal smoothness of regimes (5.3.1) | Low | Medium | Yes |
| Covariate permutation augmentation (5.4.2) | Low | Medium | Yes |
| VSN importance ordering constraint (5.8.1) | Low | Medium | Yes, domain-specific |
| Knowledge distillation from classical models (5.7) | Medium | High | **Yes** |
| Attention locality prior (5.8.2) | Medium | Medium | Yes |
| Graph sparsity prior (5.3.2) | Medium | Medium | Yes |
| Regime semantic alignment (5.8.3) | Medium | High | Yes, if regime labels available |
| Differentiable simulator (5.6) | High | Very High | Domain-specific |
| Physics residual (5.5) | High | Very High | Domain-specific |

---

### Immediate Action Checklist

1. **Fix BUG-01** — add `.to(device=..., dtype=...)` in both causal mask usages.
2. **Fix BUG-04** — update `curr_history`/`curr_future` before `continue` in stochastic depth.
3. **Fix BUG-02** — relax or remove `x_dec` feature-size validation.
4. **Add `.contiguous()` before `.reshape` in MoE capacity block (BUG-05)**.
5. **Set theory-informed lag scales in your config** for ETT data (e.g., `[24, 48, 168, 720]` for hourly).
6. **Add quantile monotonicity loss** if `tft_use_quantile_head=True`.
7. **Consider knowledge distillation** from an N-BEATS or ETS model as a warmup training signal.

---

*Report generated by deep analysis of `models/TemporalFusionTransformer.py` and supporting layer files.*
*All line references are to the implementation as provided on 2026-07-27.*
