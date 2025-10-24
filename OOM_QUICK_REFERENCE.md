# OOM Debugging - Quick Reference Card

## 🚀 Fastest Way to Debug Your OOM Issue

### 1-Minute Setup
```bash
# Run this one command
python patch_training_for_oom_debug.py
# Press 'y' when prompted
python run_training_with_oom_debug.py
```

### When Training Crashes with OOM

You'll see detailed output like this:

```
🚨 BACKWARD PASS FAILED - OOM ERROR!
================================================================================
Epoch: 3, Batch: 147/500
Error: CUDA out of memory. Tried to allocate 2.34 GiB...

CUDA Memory State at Failure:
  Current Allocated: 14523.2MB
  Current Reserved: 15360.0MB
  Max Allocated: 14891.5MB
  GPU Total Memory: 16384.0MB
  Memory Utilization: 90.9%

Batch Information:
  Batch size: 32
  Sequence length: 512
  Input tensor size: 2048.0MB
  
SUGGESTED ACTIONS:
1. Reduce batch size in config
2. Increase gradient_accumulation_steps
3. Enable mixed precision training
```

### 2. Analyze the Diagnostics

```bash
python analyze_memory_diagnostics.py
```

This shows:
- **Where** memory blows up (forward? backward?)
- **Why** it's happening (gradients? activations?)
- **How** to fix it (specific recommendations)

### 3. Apply the Fix

Most common fixes (in order of effectiveness):

#### Option A: Reduce Batch Size (Fastest)
```yaml
# In your config file
batch_size: 16  # was 32
```

#### Option B: Use Gradient Accumulation
```yaml
batch_size: 16
gradient_accumulation_steps: 2
# Effective batch size = 16 * 2 = 32 (same as before)
```

#### Option C: Enable Mixed Precision
```yaml
mixed_precision: true
use_amp: true
```

## 📊 Understanding the Output

### Good Backward Pass:
```
[DEBUG] Batch 15 PRE-BACKWARD  | Allocated: 2450.3MB
[DEBUG] Batch 15 POST-BACKWARD | Allocated: 2937.8MB (+487.5MB)
```
✅ Growth: ~500MB (gradients) - **This is normal**

### Problematic Backward Pass:
```
[DEBUG] Batch 15 PRE-BACKWARD  | Allocated: 8450.3MB
[DEBUG] Batch 15 POST-BACKWARD | Allocated: 13921.7MB (+5471.4MB)
```
❌ Growth: >5GB - **This is excessive!**

**What it means:**
- Forward pass created huge computational graph
- Backward is trying to compute gradients for all of it
- Not enough memory for gradient storage

**Fix:** Reduce model size or use gradient checkpointing

## 🔍 Key Metrics to Watch

### Memory Allocation
- **Good:** Stable around 60-80% of GPU memory
- **Warning:** Above 85% consistently
- **Critical:** Above 95% or fluctuating wildly

### Gradient Norms
- **Good:** 0.1 - 10.0
- **Warning:** 10.0 - 100.0
- **Critical:** >100 (may indicate gradient explosion)

### NaN/Inf Gradients
- **Good:** Always 0
- **Bad:** Any value > 0 indicates training instability

## 🛠️ Emergency Fixes

### If you need to train RIGHT NOW:

```python
# Add to your training config
batch_size: 8              # Reduce drastically
gradient_accumulation_steps: 4  # Maintain effective batch
mixed_precision: true      # Halves memory usage
clip_grad_norm: 1.0       # Prevents gradient explosion
```

### If training is unstable (NaN loss):

```yaml
learning_rate: 0.0001      # Lower learning rate
warmup_epochs: 5           # Gradual warmup
clip_grad_norm: 1.0       # Clip gradients
```

## 📁 Files Created

After running debugging, you'll have:

1. **backward_memory_diagnostics.json**
   - Detailed snapshots of every training step
   - Memory usage at each stage
   - Gradient statistics

2. **Training logs**
   - Real-time memory tracking
   - Batch-by-batch progress
   - Error details if crash occurs

3. **Analyzer output**
   - Summary of findings
   - Specific recommendations
   - Comparison tables

## 🎯 Decision Tree

```
OOM during backward?
├─ YES → Check memory growth
│  ├─ >2GB growth → Reduce batch size (50%)
│  ├─ Gradual increase → Memory leak, check optimizer.zero_grad()
│  └─ Spikes randomly → Unstable gradients, enable clipping
│
└─ NO, OOM during forward → Check model size
   ├─ Large model → Enable gradient checkpointing
   ├─ Long sequences → Reduce seq_len or use chunked attention
   └─ Large batch → Reduce batch_size
```

## 💡 Pro Tips

1. **Test with small batch first**
   ```bash
   # Find maximum safe batch size
   python test_max_batch_size.py
   ```

2. **Monitor during training**
   ```python
   # Add to training loop
   if batch_idx % 50 == 0:
       print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
   ```

3. **Use PyTorch's built-in tools**
   ```bash
   # Profile memory usage
   python -m torch.utils.bottleneck your_training_script.py
   ```

4. **Incremental debugging**
   - Start with batch_size=1 (should never OOM)
   - Double until OOM
   - Use 75% of that value

## 📞 Still Having Issues?

If none of the above works:

1. **Check your data loader**
   ```python
   # Ensure data isn't accumulating
   pin_memory=True
   num_workers=0  # Try this first
   persistent_workers=False
   ```

2. **Verify model architecture**
   ```python
   # Print model summary
   from torchinfo import summary
   summary(model, input_size=(batch_size, seq_len, input_dim))
   ```

3. **Look for custom layers**
   - Custom attention mechanisms
   - Complex loss functions
   - Data augmentation in forward pass

4. **Check for hidden batch dimensions**
   ```python
   # Some layers expand batches internally
   # MoE (Mixture of Experts) layers
   # Multi-head attention
   ```

## ✅ Verification Checklist

Before asking for help, verify:

- [ ] Ran training with debugging enabled
- [ ] Analyzed diagnostics with analyzer script
- [ ] Tried reducing batch_size by 50%
- [ ] Enabled gradient accumulation
- [ ] Tried mixed precision training
- [ ] Checked for NaN/Inf in gradients
- [ ] Verified optimizer.zero_grad() is called
- [ ] Looked at model parameter count
- [ ] Checked GPU memory capacity

## 🎓 Learn More

- **Full guide:** `BACKWARD_OOM_DEBUGGING_GUIDE.md`
- **Tools reference:** See individual script docstrings
- **PyTorch docs:** https://pytorch.org/docs/stable/notes/cuda.html

---

**Remember:** OOM is usually fixable with batch size reduction + gradient accumulation!

Good luck! 🚀
