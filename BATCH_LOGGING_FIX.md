# Batch Logging Fix - Per-Batch Loss Display

## Issue Found

**Location:** `scripts/train/train_celestial_production.py`, line 884

**Problem:** The per-batch logging was displaying `raw_loss` instead of `loss`, showing 3× inflated values during training.

## Before Fix

```python
logger.info(
    "Batch %s/%s | loss=%0.6f elapsed=%0.1fs",
    batch_index,
    len(train_loader),
    raw_loss.detach().item(),  # ← WRONG: Shows 3× inflated value
    elapsed,
)
```

**Impact:**
- Console output showed: `Batch 0/562 | loss=2.749450 elapsed=1.2s` (3× too high)
- This made it appear that training loss was much higher than actual
- **However:** The actual training was correct! Only the display was wrong.

## After Fix

```python
logger.info(
    "Batch %s/%s | loss=%0.6f elapsed=%0.1fs",
    batch_index,
    len(train_loader),
    loss.detach().item(),  # ✓ CORRECT: Shows scaled loss that optimizer sees
    elapsed,
)
```

**Now displays:**
- Console output: `Batch 0/562 | loss=0.916483 elapsed=1.2s` (correct value)
- This matches what the optimizer actually optimizes
- Consistent with epoch average loss reporting

## What Was Already Correct

✅ **Training accumulation** (line 843): `train_loss += loss.detach().item()` - Uses scaled loss  
✅ **Epoch average calculation**: `avg_train_loss = train_loss / max(train_batches, 1)` - Correct  
✅ **Gradient computation** (line 785): `loss.backward()` - Uses scaled loss  
✅ **Optimizer step**: Works on accumulated gradients from scaled loss  

## What This Means

### Before Fix:
- **Displayed per-batch loss:** ~0.98 (but showing 3× inflated `raw_loss`)
- **Actual training loss:** ~0.33 (the scaled loss that optimizer sees)
- **Epoch average:** ~0.33 (calculated from accumulated scaled losses) ✓ Correct

### After Fix:
- **Displayed per-batch loss:** ~0.33 (correctly shows scaled loss)
- **Actual training loss:** ~0.33 (the scaled loss that optimizer sees)
- **Epoch average:** ~0.33 (calculated from accumulated scaled losses)
- **Everything aligned!** 🎯

## Diagnostic Logging (Unchanged)

The detailed diagnostic logging in `training_diagnostic.log` was already correct and explicitly shows both values:

```
raw_loss (full batch loss): 2.74945
loss (scaled for backward): 0.91648
loss/raw_loss ratio: 0.333 (should be ~1/3)
```

This diagnostic output is intentionally verbose and correctly labeled, so it was left unchanged.

## Memory Logging (Unchanged)

The memory diagnostic logging at line 868-873 also correctly labels the raw_loss:

```python
{
    "raw_loss": raw_loss_value,  # Explicitly labeled
    "optimizer_step": int(is_cycle_end or is_final_partial),
    "accumulated_batches": batch_index + 1,
}
```

This is also intentional and correctly labeled for debugging purposes.

## Summary

**What was broken:** Console logging display only (cosmetic issue)  
**What was working:** All actual training mechanics (optimizer, gradients, accumulation)  
**What was fixed:** Changed line 884 from `raw_loss.detach().item()` to `loss.detach().item()`  

**Result:** Console output now accurately reflects the loss value that the optimizer is minimizing.

---

**Fixed by:** Beast Mode AI Analysis  
**Date:** October 25, 2025  
**File Modified:** `scripts/train/train_celestial_production.py` (1 line)
