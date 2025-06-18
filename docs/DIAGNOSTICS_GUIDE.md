# 🔧 TimesNet Training Diagnostics & Troubleshooting Guide

## 🚨 **If Training is Too Slow - START HERE**

### 🎯 **Quick Fix Steps (30 seconds)**

1. **Run Emergency Test**:
   ```bash
   python quick_diagnostic_test.py
   ```

2. **If that works, open the Light Config Notebook**:
   ```python
   # In TimesNet_Light_Config.ipynb
   emergency_debug_mode()  # Should complete in <30 seconds
   ```

3. **If emergency mode works, try Ultra-Fast**:
   ```python
   switch_to_ultra_fast()  # Should complete in <2 minutes
   ```

## 📊 **Available Diagnostic Tools**

### 1. 🧪 **Quick Diagnostic Script** (`quick_diagnostic_test.py`)
- **Purpose**: Verify basic system functionality
- **Runtime**: 10-30 seconds
- **Tests**: CUDA, PyTorch, model creation, forward pass
- **Usage**: `python quick_diagnostic_test.py`

### 2. 🚨 **Emergency Debug Mode** (In notebook)
- **Purpose**: Most minimal possible training test
- **Runtime**: 10-30 seconds  
- **Config**: 1 epoch, 8 parameters in d_model, 1 prediction step
- **Usage**: `emergency_debug_mode()`

### 3. ⚡ **Ultra-Fast Configuration** (In notebook)
- **Purpose**: Quick experimentation with tiny model
- **Runtime**: 30 seconds - 2 minutes
- **Config**: 3 epochs, 16 d_model, 3 prediction steps
- **Usage**: `switch_to_ultra_fast()`

### 4. 📊 **Data Loading Speed Test** (In notebook)
- **Purpose**: Isolate data loading performance issues
- **Tests**: Data manager creation, dataset creation, batch loading
- **Usage**: Run the "Data Loading Speed Test" cell

### 5. ⚡ **Micro-Benchmark** (In notebook)
- **Purpose**: 10-batch speed test for immediate feedback
- **Runtime**: 30 seconds - 2 minutes
- **Usage**: Run the "Micro-Benchmark" cell

### 6. 🔍 **Comprehensive Diagnostics** (In notebook)
- **Purpose**: Detailed bottleneck analysis
- **Breaks down**: Data loading, device transfer, forward pass, backward pass, etc.
- **Usage**: Run the "Comprehensive Performance Diagnostics" cell

## 🎯 **Performance Targets**

| Configuration | Target Time/Batch | Total Training Time | Use Case |
|---------------|-------------------|-------------------|----------|
| 🚨 **Emergency** | <0.1s | <30 seconds | System verification |
| ⚡ **Ultra-Fast** | <0.5s | <2 minutes | Quick experiments |
| 💡 **Light** | <1.0s | <10 minutes | Light training |
| ⚖️ **Medium** | <2.0s | <25 minutes | Balanced training |

## ⚠️ **Common Issues & Solutions**

### Problem: First batch never completes
**Symptoms**: Training hangs on first batch
**Solutions**:
1. ✅ Run `emergency_debug_mode()`
2. ✅ Check CUDA availability: `torch.cuda.is_available()`
3. ✅ Set `num_workers = 0` in DataLoader
4. ✅ Reduce batch size to 8 or 16
5. ✅ Switch to CPU: `device = 'cpu'`

### Problem: Very slow training (>10 min for light config)
**Symptoms**: Each batch takes >2 seconds
**Solutions**:
1. ✅ Use `switch_to_debugging()` first
2. ✅ Reduce model size: `d_model = 8`, `e_layers = 1`
3. ✅ Disable AMP: `use_amp = False`
4. ✅ Check GPU utilization: `nvidia-smi`
5. ✅ Reduce sequence length: `seq_len = 12`

### Problem: Out of memory errors
**Symptoms**: CUDA out of memory
**Solutions**:
1. ✅ Reduce `batch_size` to 8 or 16
2. ✅ Reduce `d_model` to 16 or 8
3. ✅ Reduce `seq_len` to 50 or 25
4. ✅ Clear GPU cache: `torch.cuda.empty_cache()`
5. ✅ Switch to CPU if GPU is too small

### Problem: Data loading is slow
**Symptoms**: Data loading test shows >0.1s per batch
**Solutions**:
1. ✅ Set `num_workers = 0`
2. ✅ Check if data file is corrupted
3. ✅ Move data to faster storage (SSD vs HDD)
4. ✅ Reduce batch size

## 🛠️ **Configuration Troubleshooting**

### Ultra-Fast Configuration Parameters
```python
# If ultra-fast is still too slow, try these minimal settings:
seq_len = 12              # Minimal sequence length
pred_len = 1              # Single prediction step
d_model = 8               # Smallest model dimension
e_layers = 1              # Single encoder layer
n_heads = 1               # Single attention head
batch_size = 8            # Small batch size
train_epochs = 1          # Single epoch
num_workers = 0           # No multiprocessing
use_amp = False           # Disable automatic mixed precision
```

### Emergency Configuration Parameters
```python
# Absolute minimum for system testing:
seq_len = 12
pred_len = 1
label_len = 2
d_model = 8
d_ff = 16
n_heads = 1
e_layers = 1
d_layers = 1
top_k = 1
num_kernels = 1
batch_size = 8
train_epochs = 1
```

## 📋 **Diagnostic Workflow**

1. **🧪 System Check**: Run `python quick_diagnostic_test.py`
2. **🚨 Emergency Test**: If system check passes, try `emergency_debug_mode()`
3. **⚡ Speed Test**: If emergency works, try `switch_to_ultra_fast()`
4. **📊 Data Test**: If still slow, run data loading speed test
5. **🔍 Full Diagnostic**: Run comprehensive diagnostics for detailed analysis

## 💡 **Advanced Troubleshooting**

### Check GPU Utilization
```bash
# Monitor GPU usage during training
nvidia-smi -l 1
```

### Check CPU Usage
```bash
# Monitor CPU usage
htop  # Linux/Mac
taskmgr  # Windows
```

### Memory Profiling
```python
# Add to notebook for memory profiling
import torch.profiler
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, 
               torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    # Run training step here
    pass
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## 🎯 **Success Criteria**

- ✅ **Emergency mode completes in <30 seconds**
- ✅ **Ultra-fast mode completes in <2 minutes**  
- ✅ **Light mode completes in <10 minutes**
- ✅ **No out of memory errors**
- ✅ **GPU utilization >80% (if using GPU)**

If emergency mode fails, there's a fundamental system issue (CUDA installation, data corruption, insufficient resources).

## 📞 **Getting Help**

If none of these diagnostics help:

1. **Share diagnostic outputs**: Run all diagnostic tools and share results
2. **System specs**: GPU model, RAM, PyTorch version, CUDA version
3. **Error messages**: Full error traces from diagnostic tools
4. **Configuration used**: Which config failed (emergency, ultra-fast, light)

The diagnostic tools will pinpoint exactly where the bottleneck is occurring!
