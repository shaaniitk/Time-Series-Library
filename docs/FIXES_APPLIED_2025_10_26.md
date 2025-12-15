# Fixes Applied: Celestial Fusion Gate Gradient Collapse

**Date**: 2025-10-26  
**Issue**: Zero gradients in celestial projection layers from batch 0  
**Root Causes Identified**: Gate initialization, embedding scale, information bottleneck

---

## Fixes Implemented

### **Fix 1: Gate Initialization (CRITICAL)**
**File**: `models/Celestial_Enhanced_PGAT.py` line ~329

**Problem**:
- Gate randomly initialized → often starts at 0.1-0.2 (closed)
- Once closed, gradients vanish → gate never learns to open
- Vicious cycle: closed gate → no learning → stays closed

**Solution**:
```python
# Initialize final layer bias to zero
# This makes Sigmoid(0) = 0.5 (balanced start)
with torch.no_grad():
    self.celestial_fusion_gate[-2].bias.zero_()
```

**Expected Impact**:
- Gate starts at ~0.5 (balanced)
- Celestial features have equal chance to prove value
- Gradients flow from batch 0
- Model can explore if celestial helps

---

### **Fix 2: Celestial Embedding Scale (CRITICAL)**
**File**: `layers/modular/graph/celestial_body_nodes.py` line ~55

**Problem**:
- Embeddings initialized with scale 0.02 (very small)
- Tiny embeddings → tiny keys/values → uniform attention
- Uniform attention → no gradients to projections
- Vicious cycle: small embeddings → weak signal → no learning

**Solution**:
```python
# Increase from 0.02 → 0.1 (5× larger)
self.body_embeddings = nn.Parameter(
    torch.randn(self.num_bodies, d_model) * 0.1
)
self.aspect_embeddings = nn.Parameter(
    torch.randn(self.num_aspects, d_model) * 0.1
)
```

**Expected Impact**:
- Stronger initial celestial signals
- Attention can be selective (not uniform)
- Gradients strong enough to learn
- Faster convergence

---

### **Fix 3: Fusion Dimension Increase (MODERATE)**
**File**: `configs/celestial_enhanced_pgat_production_RECOVERY.yaml`

**Problem**:
- Previous: fusion_dim = 64 (default)
- Compression: 416D → 64D (84% information loss!)
- Attention operates on compressed space
- Can't represent complex celestial relationships

**Solution**:
```yaml
celestial_fusion_dim: 256  # New: 38% compression (was 84%)
```

**Expected Impact**:
- Richer attention representations
- Better celestial feature preservation
- More expressive fused outputs
- Higher memory usage (+50MB expected)

---

### **Fix 4: Diagnostic Logging (MONITORING)**
**File**: `models/Celestial_Enhanced_PGAT.py` line ~1453

**Added**:
```python
# Log every 100 forward passes during training
if self.training and self._gate_log_counter % 100 == 0:
    self.logger.info(
        "[CELESTIAL FUSION] Gate stats | "
        f"mean={fusion_gate.mean():.6f} std={fusion_gate.std():.6f} "
        f"min={fusion_gate.min():.6f} max={fusion_gate.max():.6f} | "
        f"celestial_features_std={celestial_features.std():.6f} "
        f"enc_out_std={enc_out.std():.6f} "
        f"fused_output_std={fused_output.std():.6f} "
        f"influence_norm={celestial_influence.norm():.4f}"
    )
```

**Expected Output** (healthy training):
```
[CELESTIAL FUSION] Gate stats | mean=0.500000 std=0.120000 min=0.200000 max=0.800000 | 
celestial_features_std=0.450000 enc_out_std=0.520000 fused_output_std=0.380000 influence_norm=12.5000
```

**Warning Signs**:
- `mean < 0.1` → Gate collapsed (bad)
- `std < 0.01` → Gate stuck (bad)
- `celestial_features_std < 0.05` → Embeddings too small (bad)
- `influence_norm < 0.1` → No celestial contribution (bad)

---

## What to Watch During Training

### **First 10 Batches** (Critical Warmup)
✅ **Good signs**:
- Gate mean stays 0.4-0.6
- Gate std > 0.1 (varying across samples)
- Celestial features std > 0.2
- influence_norm > 5.0
- Celestial projection gradients > 0.001

❌ **Bad signs** (revert or adjust):
- Gate mean drops below 0.2 (closing)
- Gate std < 0.05 (stuck)
- Celestial features std < 0.1 (weak signal)
- All celestial gradients still 0.00000000

### **After 100 Batches** (Learning Phase)
✅ **Good signs**:
- Gate mean adjusts (0.3-0.7 is fine, model is learning)
- y_pred std increases from 0.03 → 0.1+
- Train loss decreases
- Celestial projection gradients active (0.01-0.5 range)

❌ **Bad signs**:
- Gate mean → 0.0 or 1.0 (saturated)
- y_pred std still < 0.05 (mode collapse persists)
- Celestial gradients zero or decreasing

### **After 1 Epoch** (Validation)
✅ **Success criteria**:
- y_pred std > 0.15 (no more mode collapse)
- Train loss < 1.0
- Val loss < 1.0 
- Gate mean 0.2-0.8 (adaptive, not stuck)
- Celestial projection grad_norm > 0.01

---

## Training Command

```bash
# Activate environment
source /home/kalki/Documents/workspace/Time-Series-Library/tsl-env/bin/activate

# Run training with RECOVERY config (now with fixes)
python scripts/train/train_celestial_production.py \
  --config configs/celestial_enhanced_pgat_production_RECOVERY.yaml \
  2>&1 | tee training_with_fixes_$(date +%Y%m%d_%H%M%S).log
```

**Monitor in real-time** (separate terminal):
```bash
# Watch gate statistics
tail -f training_with_fixes_*.log | grep "CELESTIAL FUSION"

# Watch gradient norms
tail -f training_with_fixes_*.log | grep "celestial.*grad_norm"
```

---

## Expected Outcomes

### **Scenario 1: Fixes Work** ✅
- Gate stays open (mean 0.4-0.6) for first 500 batches
- Celestial projections receive gradients (0.01-0.5)
- y_pred variance increases (0.03 → 0.2+)
- Model explores celestial features
- **Outcome**: We'll see if celestial features actually help with MSE loss
  - If they help: loss decreases, gate stays open
  - If they don't help: gate closes gradually over epochs (expected behavior)

### **Scenario 2: Gate Still Collapses** ❌
**If gate mean drops to <0.1 within 100 batches**:
- Problem is NOT initialization (we fixed that)
- Problem is likely **MSE loss penalizing variability**
- Next step: Switch to MDN with NLL loss
- Or: Use auxiliary loss to encourage gate openness

### **Scenario 3: Gate Opens But Mode Collapse Persists** ⚠️
**If gate mean stays 0.4+ but y_pred std still <0.1**:
- Celestial features are active but not diverse enough
- Need to check: Are all 13 celestial bodies being used?
- Next step: Add attention entropy loss to encourage selectivity

---

## Rollback Plan

If training crashes or performance degrades:

```bash
# Revert Fix 1 (gate init)
cd /home/kalki/Documents/workspace/Time-Series-Library
git diff models/Celestial_Enhanced_PGAT.py

# Revert Fix 2 (embedding scale)
git diff layers/modular/graph/celestial_body_nodes.py

# Revert Fix 3 (fusion dim)
git diff configs/celestial_enhanced_pgat_production_RECOVERY.yaml

# If needed:
git checkout models/Celestial_Enhanced_PGAT.py
git checkout layers/modular/graph/celestial_body_nodes.py
git checkout configs/celestial_enhanced_pgat_production_RECOVERY.yaml
```

---

## Next Steps Based on Results

### **If Gate Opens & Loss Improves**
→ Continue training, monitor for overfitting
→ Consider adding more regularization if validation loss spikes

### **If Gate Opens But Loss Doesn't Improve**
→ Celestial features may not correlate with targets (this is fine!)
→ Gate will naturally close over epochs (model learning to ignore them)
→ Consider removing celestial components if they never help

### **If Gate Still Collapses Immediately**
→ Try Fix 4: Switch to MDN with NLL loss
→ Or: Add gate regularization loss to penalize closing

### **If Training Crashes (OOM)**
→ Reduce fusion_dim from 256 → 128
→ Or: Reduce batch_size from 12 → 8
→ Or: Enable gradient checkpointing

---

## Key Insight: Covariate-Target Correlation

**Your Question**: "Could model not finding relationship between covariate and target cause this?"

**Answer**: 
- **No** for immediate collapse (batch 0-10)
  - Even with zero true correlation, random exploration should produce gradients
  - Instant zero gradients = initialization problem, not learning problem
  
- **Yes** for gradual collapse (batch 100-1000)
  - If celestial features don't help minimize loss, gate SHOULD close
  - This is correct behavior (model pruning useless features)
  - But it should happen gradually, not instantly

**Current Hypothesis**:
1. **Instant collapse** (what you saw) = poor initialization (now fixed)
2. **Gradual collapse** (what might happen now) = celestial features don't help with MSE
3. **If gradual collapse happens**: This is actually the model working correctly!
   - It's saying "these celestial features don't predict OHLC prices"
   - **Not a bug, but a valid scientific result**

**The Real Test**:
- After these fixes, if gate opens initially but closes after 1000 batches
- **That means**: Celestial bodies don't actually correlate with financial markets (at least not for MSE-based prediction)
- **This would be valuable information** (disproving astrological trading hypothesis)

---

## Summary

**What we fixed**: Structural issues preventing learning from starting
**What we'll discover**: Whether celestial features actually help predict financial data
**What we expect**: Gate to stay open long enough to fairly test the hypothesis

Let's see what happens! 🚀
