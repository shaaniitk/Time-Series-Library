# Backward Pass OOM Debugging Guide

## Overview

This guide helps you diagnose and fix Out-of-Memory (OOM) errors that occur during the backward pass of neural network training.

## Quick Start

### Option 1: Automated Debugging (Recommended)

```bash
# Run training with enhanced debugging
python patch_training_for_oom_debug.py
# Follow the prompts to create wrapper
python run_training_with_oom_debug.py
```

### Option 2: Manual Integration

Add debugging to your existing training script:

```python
from utils.backward_memory_debugger import create_backward_debugger

# Create debugger
debugger = create_backward_debugger(device, checkpoint_dir)

# In training loop, before backward:
debugger.capture_pre_backward(batch_idx, epoch)

# Backward pass
loss.backward()

# After backward:
debugger.capture_post_backward(batch_idx, epoch, model)

# At end of epoch:
debugger.save_diagnostics()
debugger.print_summary()
```

### Option 3: Use Enhanced Training Script

```python
from scripts.train.train_with_backward_debugging import enhanced_train_epoch_with_debugging

# Replace your train_epoch call with this function
avg_loss, batches = enhanced_train_epoch_with_debugging(
    model, train_loader, optimizer, criterion, scaler, use_amp,
    gradient_accumulation_steps, device, epoch, args, logger,
    target_scaler, target_indices, total_train_batches,
    full_cycles, remainder_batches, epoch_start,
    sequential_mixture_loss_cls, mixture_loss_cls, checkpoint_dir
)
```

## Tools Provided

### 1. Backward Memory Debugger (`utils/backward_memory_debugger.py`)

**Core functionality:**
- Captures memory snapshots at critical points:
  - Pre-forward pass
  - Post-forward / Pre-backward
  - Pre-backward
  - Post-backward
  - Post-optimizer step
- Tracks:
  - CUDA allocated/reserved memory
  - Active tensors and gradient tensors
  - Gradient statistics (norm, NaN, Inf)
  - Memory growth between stages
- Detects memory leaks automatically
- Saves detailed JSON diagnostics

**Key Methods:**
```python
debugger.capture_pre_backward(batch_idx, epoch)
debugger.capture_post_backward(batch_idx, epoch, model)
debugger.save_diagnostics()
debugger.print_summary()
debugger.find_memory_leaks()
```

### 2. Enhanced Training Script (`scripts/train/train_with_backward_debugging.py`)

**Features:**
- Drop-in replacement for standard training loop
- Automatic memory tracking every N batches
- Detailed error reporting on OOM
- Gradient health monitoring (NaN/Inf detection)
- Memory leak detection during training
- Comprehensive diagnostics output

**Debug frequency:**
- First epoch: Every batch
- Epochs 1-5: Every 10 batches
- After epoch 5: Every 25 batches

### 3. Patching Tools (`patch_training_for_oom_debug.py`)

**Provides:**
- Manual patch instructions with code snippets
- Automated wrapper generation
- Monkey-patching utilities
- Quick integration guides

### 4. Memory Diagnostics Analyzer (`analyze_memory_diagnostics.py`)

**Analyzes:**
- Memory growth patterns across training stages
- Backward pass specific behavior
- Memory leaks
- Tensor accumulation
- Gradient health (NaN/Inf)

**Usage:**
```bash
# Analyze most recent diagnostics
python analyze_memory_diagnostics.py

# Analyze specific file
python analyze_memory_diagnostics.py checkpoints/my_model/backward_memory_diagnostics.json
```

**Output includes:**
- Memory usage statistics by stage
- Backward pass growth analysis
- Problematic batch identification
- Leak detection
- Tensor accumulation trends
- **Actionable recommendations**

### 5. Helper Tools (`enable_backward_debugging.py`)

**Provides:**
- Monkey-patch functions
- Backward hooks for gradient monitoring
- Wrapper functions for backward() calls
- Usage instructions and examples

## Debugging Workflow

### Step 1: Enable Debugging

Choose one of the integration methods above to add debugging to your training.

### Step 2: Run Training

Start your training. You'll see output like:

```
[DEBUG] Batch 15 PRE-BACKWARD | Allocated: 2450.3MB, Reserved: 2600.0MB
[DEBUG] Batch 15 POST-BACKWARD | Allocated: 3821.7MB (+1371.4MB), Reserved: 3900.0MB (+1300.0MB)
  Parameters with gradients: 15234/15234
  Total gradient memory: 487.2MB
```

### Step 3: Analyze Results

If training completes or crashes, analyze the diagnostics:

```bash
python analyze_memory_diagnostics.py
```

This will show:
- Where memory grows most
- Which batches cause issues
- Whether there are leaks
- Specific recommendations

### Step 4: Apply Fixes

Based on analyzer recommendations, common fixes include:

#### Fix 1: Reduce Batch Size
```yaml
# In your config
batch_size: 16  # was 32
```

#### Fix 2: Increase Gradient Accumulation
```yaml
batch_size: 16  # reduced
gradient_accumulation_steps: 2  # added/increased
# Effective batch size remains 32
```

#### Fix 3: Enable Mixed Precision
```yaml
mixed_precision: true
use_amp: true
```

#### Fix 4: Add Gradient Checkpointing
```python
from torch.utils.checkpoint import checkpoint

# In model forward:
def forward(self, x):
    # Wrap memory-intensive layers
    x = checkpoint(self.encoder, x)
    return x
```

#### Fix 5: Clip Gradients
```yaml
clip_grad_norm: 1.0  # or lower if needed
```

#### Fix 6: Reduce Model Size
```yaml
d_model: 256  # was 512
n_heads: 4    # was 8
e_layers: 2   # was 3
```

## Understanding OOM Errors

### Common Causes

1. **Backward Pass Growth**
   - Symptom: Memory spikes during `loss.backward()`
   - Cause: Large computational graph
   - Fix: Gradient checkpointing, smaller batch

2. **Gradient Accumulation**
   - Symptom: Memory increases with each batch
   - Cause: Gradients not being cleared
   - Fix: Ensure `optimizer.zero_grad(set_to_none=True)`

3. **Memory Leaks**
   - Symptom: Monotonic memory growth
   - Cause: Tensors retained unnecessarily
   - Fix: Use `.detach()`, check for global variables

4. **Large Activations**
   - Symptom: High memory in forward pass
   - Cause: Model stores all intermediate activations
   - Fix: Gradient checkpointing, reduce sequence length

5. **Gradient Explosion**
   - Symptom: NaN/Inf gradients, OOM
   - Cause: Unstable training, high learning rate
   - Fix: Gradient clipping, lower learning rate

### Memory Breakdown

Typical memory usage in training:

```
Model Parameters:     500 MB  (fixed)
Optimizer States:     1000 MB (Adam uses 2x params)
Forward Activations:  1500 MB (depends on batch size)
Backward Gradients:   1500 MB (same as activations)
-------------------------------------------------
Total:                4500 MB minimum
```

For a 16GB GPU:
- Safe batch size: Use ~12GB max (leave 4GB buffer)
- If OOM at 12GB usage: Problem is likely gradient accumulation or leaks

## Diagnostic Output Explanation

### Pre-Backward Snapshot
```json
{
  "stage": "pre_backward",
  "cuda_allocated_mb": 2450.3,
  "cuda_reserved_mb": 2600.0,
  "num_tensors": 1523,
  "num_tensors_with_grad": 245,
  "num_tensors_in_grad_graph": 892
}
```
- `allocated`: Actual memory used by tensors
- `reserved`: Memory reserved by CUDA
- `tensors_in_grad_graph`: Tensors that will need gradients

### Post-Backward Snapshot
```json
{
  "stage": "post_backward",
  "cuda_allocated_mb": 3821.7,
  "params_with_grad": 15234,
  "total_gradient_norm": 12.4567,
  "params_with_nan_grad": 0,
  "params_with_inf_grad": 0,
  "gradient_memory_mb": 487.2
}
```
- `params_with_grad`: Should equal total trainable params
- `gradient_norm`: Typical range 0.1-100, >1000 is concerning
- `nan_grad/inf_grad`: Should always be 0

### Memory Leak Indicators

```json
{
  "type": "monotonic_memory_growth",
  "description": "Memory consistently increasing after optimizer steps",
  "growth_rate_mb_per_step": 15.2,
  "recent_values": [2400, 2415, 2430, 2445, 2460]
}
```
- `growth_rate`: Should be near 0
- Growing values indicate leak

## Advanced Debugging

### Custom Hooks

Add gradient hooks for specific layers:

```python
def gradient_hook(name):
    def hook(grad):
        print(f"{name} gradient norm: {grad.norm().item():.4f}")
        return grad
    return hook

for name, param in model.named_parameters():
    if 'attention' in name:
        param.register_hook(gradient_hook(name))
```

### Memory Profiling

Use PyTorch profiler for detailed analysis:

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             profile_memory=True) as prof:
    loss.backward()

print(prof.key_averages().table(sort_by="cuda_memory_usage"))
```

### Gradient Checkpointing Example

```python
from torch.utils.checkpoint import checkpoint

class MyModel(nn.Module):
    def forward(self, x):
        # Regular forward for small layers
        x = self.embedding(x)
        
        # Checkpoint expensive layers
        x = checkpoint(self.large_encoder, x)
        x = checkpoint(self.transformer_layers, x)
        
        # Regular forward for output
        x = self.output_head(x)
        return x
```

## Troubleshooting

### Issue: Debugger not capturing data

**Solution:**
```python
# Ensure debugger is created before training
debugger = create_backward_debugger(device, checkpoint_dir)

# Call capture methods at right points
debugger.capture_pre_backward(batch_idx, epoch)
loss.backward()
debugger.capture_post_backward(batch_idx, epoch, model)
```

### Issue: JSON file not created

**Solution:**
```python
# Explicitly save at end of epoch
debugger.save_diagnostics()

# Or specify path
debugger.save_diagnostics(Path("my_diagnostics.json"))
```

### Issue: OOM still occurs with debugging

**Solution:**
- Debugging adds minimal overhead (~10-50MB)
- If debugging itself causes OOM, your batch size is too large
- Reduce batch size first, then enable debugging

### Issue: Can't interpret analyzer output

**Solution:**
- Look for "CRITICAL" findings first
- Read recommendations section
- Focus on largest memory growth transitions
- Check for NaN/Inf gradients

## Best Practices

1. **Always debug on a subset first**
   ```python
   if epoch == 0 and batch_idx < 10:
       debugger.capture_pre_backward(batch_idx, epoch)
   ```

2. **Save diagnostics periodically**
   ```python
   if batch_idx % 100 == 0:
       debugger.save_diagnostics()
   ```

3. **Monitor gradient norms**
   ```python
   total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   if total_norm > 100:
       logger.warning(f"Large gradient norm: {total_norm}")
   ```

4. **Clear cache on errors**
   ```python
   except RuntimeError as e:
       if "out of memory" in str(e):
           torch.cuda.empty_cache()
           # Retry with smaller batch or skip
   ```

5. **Use context managers**
   ```python
   with torch.cuda.amp.autocast():
       outputs = model(inputs)
   ```

## FAQ

**Q: Does debugging slow down training?**
A: Minimal impact (~1-3%). Memory snapshots are fast.

**Q: Can I use this with DDP/distributed training?**
A: Yes, but each process creates separate diagnostics.

**Q: What if I don't have CUDA?**
A: Tools work on CPU, but focus on CPU memory instead.

**Q: How much memory does the debugger use?**
A: ~10-50MB depending on snapshot frequency.

**Q: Can I debug inference OOM?**
A: Yes, use the same tools but call capture methods during inference.

## Support

If you're still experiencing OOM after following this guide:

1. Share your diagnostics JSON file
2. Include model architecture summary
3. Provide config file (batch size, model dimensions)
4. Share analyzer output

## Summary

**Quick checklist for OOM debugging:**
- [ ] Enable backward debugging
- [ ] Run training (even if it crashes)
- [ ] Analyze diagnostics with analyzer script
- [ ] Apply top 3 recommendations
- [ ] Re-run with changes
- [ ] Repeat until stable

Most OOMs can be fixed by:
1. Reducing batch size (50% reduction)
2. Enabling gradient accumulation
3. Using mixed precision
4. Adding gradient checkpointing to large layers

Good luck! 🚀
