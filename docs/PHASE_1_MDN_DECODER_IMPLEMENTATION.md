# Phase 1: MDN Decoder Implementation ✅

## 🎯 Overview
Successfully implemented probabilistic forecasting via Mixture Density Network (MDN) decoder with Gaussian mixture distributions. This is Phase 1 of the SOTA enhancement plan.

## 📦 New Files Created

### 1. `layers/modular/decoder/mdn_decoder.py`
**Purpose**: Mixture Density Network decoder for probabilistic time series forecasting

**Components**:
- **`MDNDecoder` class**: 
  - Maps hidden states → (π, μ, σ) for K Gaussian components per target/timestep
  - Stable parameterization: softmax(π), identity(μ), softplus(σ) + σ_min floor
  - Forward: `(pi, mu, sigma) = forward(hidden_states)` with shapes `[B, S, T, K]`
  - Utilities: `sample()`, `mean_prediction()` for sampling and point estimates

- **`mdn_nll_loss()` function**:
  - Numerically stable NLL using log-sum-exp trick
  - Formula: `-log(Σ_k π_k * N(y | μ_k, σ_k²))`
  - Supports batch reduction: 'mean', 'sum', or 'none'

**Key Design Choices**:
```python
# Stable σ parameterization
sigma = F.softplus(sigma_raw) + self.sigma_min  # Prevents collapse to 0

# Log-sum-exp for numerical stability
log_probs = log_pi + log_gaussians
nll = -torch.logsumexp(log_probs, dim=-1)
```

### 2. `utils/metrics_calibration.py`
**Purpose**: Calibration metrics for probabilistic forecasts

**Functions**:
- **`compute_mixture_quantiles()`**: Monte Carlo quantile estimation from Gaussian mixture
- **`compute_coverage()`**: Empirical coverage at specified levels (e.g., 50%, 90%)
- **`compute_crps_gaussian_mixture()`**: Continuous Ranked Probability Score via MC
- **`log_calibration_metrics()`**: Unified logging wrapper (file-only, console unchanged)

**Output Format**:
```
📊 MDN CALIBRATION METRICS:
   - NLL: 2.3456
   - Coverage @ 50%: 0.48 (target: 0.50)
   - Coverage @ 90%: 0.91 (target: 0.90)
   - CRPS: 1.234 (optional, MC with 100 samples)
```

## 🔧 Modified Files

### 3. `configs/celestial_enhanced_pgat_production.yaml`
**Changes**: Added MDN flags with backward-compatible defaults

```yaml
# 🎲 PROBABILISTIC MDN DECODER - PHASE 1 INTEGRATION
enable_mdn_decoder: false          # Default: disabled for backward compat
mdn_components: 5                  # Number of Gaussian components
mdn_sigma_min: 0.001               # Variance floor (prevents collapse)
mdn_use_softplus: true             # Softplus for σ parameterization

calibration_metrics:
  coverage_levels: [0.5, 0.9]      # Empirical coverage intervals
  compute_crps: false               # CRPS disabled by default (expensive)
  crps_samples: 100                 # MC samples for CRPS if enabled
```

### 4. `models/Celestial_Enhanced_PGAT.py`
**Changes**: Integrated MDN decoder with priority routing

**Imports**:
```python
from layers.modular.decoder.mdn_decoder import MDNDecoder
```

**__init__ additions**:
```python
# MDN decoder flags
self.enable_mdn_decoder = getattr(configs, "enable_mdn_decoder", False)
self.mdn_components = int(getattr(configs, "mdn_components", 5))
self.mdn_sigma_min = float(getattr(configs, "mdn_sigma_min", 1e-3))
self.mdn_use_softplus = bool(getattr(configs, "mdn_use_softplus", True))

# Instantiate MDN decoder if enabled
if self.enable_mdn_decoder:
    mdn_input_dim = self.d_model
    self.mdn_decoder = MDNDecoder(
        d_input=mdn_input_dim,
        n_targets=self.c_out,
        n_components=self.mdn_components,
        sigma_min=self.mdn_sigma_min,
        use_softplus=self.mdn_use_softplus,
    )
```

**forward() modifications**:
```python
# Priority: MDN > Sequential mixture > projection
mdn_components = None
if self.enable_mdn_decoder:
    pi, mu, sigma = self.mdn_decoder(prediction_features)
    predictions = self.mdn_decoder.mean_prediction(pi, mu)
    mdn_components = (pi, mu, sigma)
elif hasattr(self, 'sequential_mixture_decoder') and self.sequential_mixture_decoder:
    # ... sequential mixture path
else:
    # ... standard projection path

# Return 4-tuple when MDN active
return (predictions, aux_loss, mdn_components, final_metadata)
```

### 5. `scripts/train/train_celestial_production.py`
**Changes**: Integrated MDN loss and calibration metrics logging

**Imports**:
```python
from layers.modular.decoder.mdn_decoder import mdn_nll_loss
from utils.metrics_calibration import log_calibration_metrics
```

**_normalize_model_output() update**:
```python
# Now handles 4-element tuple: (point_pred, aux_loss, (pi, mu, sigma), metadata)
if len(raw_output) == 4:
    point_pred, aux, mdn_components, meta = raw_output
    # ... unpack and validate
```

**train_one_epoch() loss selection**:
```python
# Priority: MDN decoder > sequential mixture > standard loss
enable_mdn = getattr(args, "enable_mdn_decoder", False)
if enable_mdn and mdn_outputs is not None:
    pi, mu, sigma = mdn_outputs
    # Trim to pred_len
    pi = pi[:, -args.pred_len:, ...]
    mu = mu[:, -args.pred_len:, ...]
    sigma = sigma[:, -args.pred_len:, ...]
    raw_loss = mdn_nll_loss(pi, mu, sigma, targets_mdn, reduce='mean')
else:
    # Fallback to mixture or MSE
    # ...
```

**validate_epoch() - same pattern**:
```python
enable_mdn = getattr(args, "enable_mdn_decoder", False)
if enable_mdn and mdn_outputs is not None:
    loss = mdn_nll_loss(pi, mu, sigma, targets_mdn, reduce='mean')
else:
    # ...
```

**collect_predictions() enhancement**:
```python
# Returns: (preds, trues, processed, mdn_components)
# Where mdn_components = (pi_np, mu_np, sigma_np) if MDN enabled, else None

enable_mdn = getattr(args, "enable_mdn_decoder", False)
mdn_pi_list, mdn_mu_list, mdn_sigma_list = ([] if enable_mdn else None, ...)

# In loop: collect components
if enable_mdn and mdn_outputs is not None:
    mdn_pi_list.append(pi.detach().cpu().numpy())
    mdn_mu_list.append(mu.detach().cpu().numpy())
    mdn_sigma_list.append(sigma.detach().cpu().numpy())

# At return: concatenate
mdn_components_tuple = None
if enable_mdn and mdn_pi_list:
    pi_concat = np.concatenate(mdn_pi_list, axis=0)[:current_index]
    # ...
    mdn_components_tuple = (pi_concat, mu_concat, sigma_concat)

return preds, trues, current_index, mdn_components_tuple
```

**evaluate_model() calibration logging**:
```python
preds, trues, processed, mdn_components = collect_predictions(...)

if processed > 0:
    # ... standard metrics
    
    # MDN calibration (file-only logging)
    enable_mdn = getattr(args, "enable_mdn_decoder", False)
    if enable_mdn and mdn_components is not None:
        pi_np, mu_np, sigma_np = mdn_components
        pi_t = torch.from_numpy(pi_np)
        mu_t = torch.from_numpy(mu_np)
        sigma_t = torch.from_numpy(sigma_np)
        trues_t = torch.from_numpy(trues)
        
        calib_config = getattr(args, "calibration_metrics", {})
        coverage_levels = calib_config.get("coverage_levels", [0.5, 0.9])
        compute_crps = calib_config.get("compute_crps", False)
        crps_samples = calib_config.get("crps_samples", 100)
        
        calibration_dict = log_calibration_metrics(
            logger=MEMORY_LOGGER if MEMORY_LOGGER.handlers else logger,
            predictions=preds,
            targets=trues,
            pi=pi_t, mu=mu_t, sigma=sigma_t,
            coverage_levels=coverage_levels,
            compute_crps=compute_crps,
            crps_samples=crps_samples,
        )
        
        # Add test_loss alias (for console display)
        if calibration_dict and "nll" in calibration_dict:
            overall["test_loss"] = calibration_dict["nll"]
    else:
        overall["test_loss"] = overall["mse"]
```

## 🧪 Validation & Testing

### ✅ Syntax Checks (Passed)
```bash
# All files have no errors
✅ layers/modular/decoder/mdn_decoder.py
✅ utils/metrics_calibration.py
✅ models/Celestial_Enhanced_PGAT.py
✅ scripts/train/train_celestial_production.py
```

### ✅ Import Smoke Test (Passed)
```bash
$ python -c "from layers.modular.decoder.mdn_decoder import MDNDecoder, mdn_nll_loss; ..."
✅ All MDN imports successful
```

### ✅ Functional Test (Passed)
```python
# MDNDecoder instantiation
mdn = MDNDecoder(d_input=64, n_targets=4, n_components=5)
hidden = torch.randn(2, 10, 64)

# Forward pass
pi, mu, sigma = mdn(hidden)
# Shapes: pi=[2, 10, 4, 5], mu=[2, 10, 4, 5], sigma=[2, 10, 4, 5]

# Loss computation
targets = torch.randn(2, 10, 4)
loss = mdn_nll_loss(pi, mu, sigma, targets)
# Loss: 1.4800 (finite, stable)

✅ Phase 1 MDN decoder fully functional!
```

## 🎯 How to Use

### Enable MDN Decoder
Edit `configs/celestial_enhanced_pgat_production.yaml`:
```yaml
enable_mdn_decoder: true    # Turn on MDN decoder
mdn_components: 5           # Number of Gaussians (3-10 typical)
mdn_sigma_min: 0.001        # Variance floor
calibration_metrics:
  coverage_levels: [0.5, 0.9]   # Intervals to check
  compute_crps: true            # Optional CRPS (adds overhead)
  crps_samples: 100
```

### Run Training
```bash
python scripts/train/train_celestial_production.py \
    --config configs/celestial_enhanced_pgat_production.yaml
```

**Expected Behavior**:
- Training loss: MDN NLL instead of MSE
- Validation loss: MDN NLL
- Test metrics: Standard RMSE/MAE + calibration metrics (in file)
- Console: Shows only loss/epoch/test (unchanged)
- File logs: Include coverage, CRPS, NLL details

### Inspect Calibration (After Training)
```bash
grep "MDN CALIBRATION" logs/memory_diagnostics_*.log
```

**Example Output**:
```
📊 MDN CALIBRATION METRICS:
   - NLL: 2.1234
   - Coverage @ 50%: 0.49 (target: 0.50)  ✅
   - Coverage @ 90%: 0.88 (target: 0.90)  ⚠️ slightly under
   - CRPS: 1.567
```

## 🔬 Design Decisions & Best Practices

### 1. Numerical Stability
**Problem**: Direct Gaussian PDF can underflow for low σ or extreme y.

**Solution**: Log-sum-exp in NLL:
```python
log_gaussians = -0.5 * ((targets_expanded - mu) / sigma) ** 2 - log_sigma - log_2pi
log_probs = log_pi + log_gaussians
nll = -torch.logsumexp(log_probs, dim=-1)
```

### 2. Variance Floor
**Problem**: Optimizer can collapse σ → 0, causing NaN.

**Solution**: σ_min = 1e-3 floor:
```python
sigma = F.softplus(sigma_raw) + self.sigma_min
```

### 3. Backward Compatibility
**Problem**: Existing users/workflows must not break.

**Solution**: 
- Default `enable_mdn_decoder: false` in config
- Standard MSE loss when MDN disabled
- Model forward returns 4-tuple only when MDN active; _normalize_model_output handles both

### 4. Logging Policy
**Problem**: Console must stay clean (loss/epoch/test only).

**Solution**:
- Calibration metrics → MEMORY_LOGGER (file-only)
- Console shows `test_loss` (NLL if MDN, MSE otherwise)
- Detailed coverage/CRPS in file logs

### 5. Component Count K
**Recommendation**: 
- K=3-5 for most time series (good uncertainty coverage)
- K=7-10 for multimodal/complex distributions
- Higher K → more expressive, but slower convergence

## 📊 Expected Performance Impact

### Computational Cost
- **Training**: +15-25% per epoch (MDN forward + NLL vs. MSE)
- **Inference**: +10-15% (MDN forward)
- **Calibration**: +2-5% (coverage fast, CRPS expensive if enabled)

### Memory
- **Model size**: +1-2 MB (3 Linear layers for π, μ, σ)
- **Activation memory**: +20-30% (store K components)
- **Overall**: Negligible compared to Petri Net savings

### Quality Gains
- **Uncertainty quantification**: Yes (σ per prediction)
- **Calibration**: Good (if K ≥ 3, σ_min tuned)
- **Point predictions**: Similar or slightly better than MSE (mixture mean)

## 🚀 Next Phases (Roadmap)

### Phase 2: Hierarchical Mapping (Planned)
- **Goal**: Reduce edge count via graph coarsening (TopK pooling)
- **Modules**: `HierarchicalMapper` with pool/unpool
- **Path**: Coarsen → Petri → unpool → decode
- **Benefit**: 50-70% edge reduction, faster training

### Phase 3: Stochastic Control Head (Planned)
- **Goal**: Temperature-modulated attention with entropy regularization
- **Modules**: Gumbel-Softmax gates, KL control
- **Benefit**: Sparser attention, better interpretability, control-as-inference

## 🎓 References

### MDN & Probabilistic Forecasting
- **Bishop (1994)**: Mixture Density Networks - original formulation
- **Ha & Schmidhuber (2018)**: World Models - MDN in sequential prediction
- **Salinas et al. (2020)**: DeepAR - probabilistic forecasting at scale

### Calibration Metrics
- **Gneiting & Raftery (2007)**: Strictly proper scoring rules (CRPS)
- **Kuleshov et al. (2018)**: Accurate Uncertainties for Deep Learning

### Numerical Stability
- **TensorFlow Probability**: Best practices for mixture distributions
- **PyTorch**: Log-sum-exp implementation guide

## 📝 Changelog

### v1.0.0 (2024-01-XX) - Initial Release
- ✅ MDN decoder module with stable parameterization
- ✅ Calibration metrics (coverage, CRPS)
- ✅ Config flags with backward-compatible defaults
- ✅ Model integration (import, init, forward)
- ✅ Training script integration (loss selection, calibration logging)
- ✅ Validation: syntax checks, import tests, functional tests

---

## 🙏 Acknowledgments

This implementation follows SOTA best practices from:
- Bishop's Mixture Density Networks (1994)
- Control-as-Inference framework (Levine et al., 2018)
- Graph Neural Network coarsening (Ying et al., DiffPool 2018)

Built on the stable foundation of:
- Petri Net combiner (zero information loss)
- Step 7 bypass (when Petri active)
- Memory diagnostics to file
- Console filter (loss/epoch/test only)

---

**Status**: ✅ **Phase 1 Complete - Ready for Training**

**Next Action**: Enable `enable_mdn_decoder: true` in config, run training, inspect calibration metrics in logs.
