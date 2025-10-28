# Debugging Celestial Production Training Script

## 🎯 Quick Start Debugging

### Recommended Debug Order:
1. **Start with**: "Debug: Celestial Memory Fix" - Tests memory issues specifically
2. **Then try**: "Debug: Celestial Production (Simplified)" - Tests full pipeline step by step  
3. **Finally**: "Debug: Celestial Production Training" - Full production script

## VS Code Debug Configurations Added

I've added several debug configurations to your `.vscode/launch.json`:

### 1. Debug: Celestial Production Training
- **Purpose**: Debug the full production training script
- **File**: `scripts/train/train_celestial_production.py`
- **Environment**: Sets PYTHONPATH and CUDA device
- **Usage**: Select this configuration and press F5 to start debugging

### 2. Debug: Celestial Production (Memory Profiling)
- **Purpose**: Debug with memory optimization settings
- **File**: `scripts/train/train_celestial_production.py`
- **Environment**: Includes `PYTORCH_CUDA_ALLOC_CONF` for memory management
- **Usage**: Use when investigating memory issues

### 3. Debug: Celestial Production (Simplified)
- **Purpose**: Debug with a simplified test script
- **File**: `scripts/train/debug_celestial_production.py`
- **Usage**: Tests full pipeline step by step

### 4. Debug: Celestial Memory Fix
- **Purpose**: Specifically targets the memory explosion issue
- **File**: `scripts/train/debug_celestial_memory_fix.py`
- **Config**: Uses `configs/celestial_enhanced_pgat_production_debug.yaml`
- **Usage**: **START HERE** - Tests memory issues with optimized settings

## Debugging Strategy

### Step 1: Start with Simplified Debug
1. Open VS Code
2. Select "Debug: Celestial Production (Simplified)" from the debug dropdown
3. Press F5 to start debugging
4. This will test each component individually:
   - Configuration loading
   - Data loading
   - Model initialization
   - Forward pass
   - Loss computation
   - Training step

### Step 2: Set Breakpoints
Common places to set breakpoints:
- Line 89: After config loading
- Line 115: After data loading
- Line 140: After model initialization
- Line 165: Before forward pass
- Line 180: After forward pass
- Line 200: Before loss computation

### Step 3: Monitor Memory Usage
The debug script includes memory monitoring functions:
- `debug_memory_usage()`: Shows GPU memory allocation
- `debug_tensor_shapes()`: Shows tensor dimensions

### Step 4: Common Issues to Check

#### Memory Issues
- **Symptom**: CUDA out of memory errors
- **Debug**: Use "Memory Profiling" configuration
- **Solutions**: 
  - Reduce batch_size in config
  - Reduce seq_len or d_model
  - Enable gradient checkpointing

#### Shape Mismatches
- **Symptom**: RuntimeError about tensor shapes
- **Debug**: Check `debug_tensor_shapes()` output
- **Solutions**:
  - Verify c_out matches target dimensions
  - Check pred_len and label_len settings

#### Configuration Issues
- **Symptom**: AttributeError or missing config values
- **Debug**: Check config loading section
- **Solutions**:
  - Verify config file exists
  - Check all required fields are present

#### Model Initialization Issues
- **Symptom**: Model fails to initialize
- **Debug**: Step through model creation
- **Solutions**:
  - Check input dimensions (enc_in, dec_in)
  - Verify celestial system settings

## Key Files to Monitor

### Configuration
- `configs/celestial_enhanced_pgat_production.yaml`

### Model Files
- `models/Celestial_Enhanced_PGAT.py`
- `layers/modular/graph/celestial_graph_combiner.py`

### Data Files
- `data/prepared_financial_data.csv`
- `data_provider/data_factory.py`

## Debug Commands

### In VS Code Debug Console
```python
# Check tensor shapes
print(f"batch_x shape: {batch_x.shape}")
print(f"outputs shape: {outputs.shape if isinstance(outputs, torch.Tensor) else [o.shape for o in outputs]}")

# Check memory
torch.cuda.memory_summary()

# Check model parameters
sum(p.numel() for p in model.parameters())

# Check config values
vars(args)
```

### Memory Debugging
```python
# Clear cache
torch.cuda.empty_cache()

# Monitor memory
torch.cuda.memory_allocated() / 1024**2  # MB
torch.cuda.memory_reserved() / 1024**2   # MB
```

## Troubleshooting Common Errors

### 1. "No module named 'models.Celestial_Enhanced_PGAT'"
- **Cause**: PYTHONPATH not set correctly
- **Solution**: Debug configurations include PYTHONPATH setup

### 2. "CUDA out of memory"
- **Cause**: Model or batch too large for GPU
- **Solutions**:
  - Reduce batch_size from 16 to 8 or 4
  - Reduce seq_len from 250 to 128
  - Reduce d_model from 130 to 64

### 3. "Config file not found"
- **Cause**: Working directory or file path issue
- **Solution**: Debug configurations set correct working directory

### 4. "Shape mismatch in loss computation"
- **Cause**: Model output doesn't match expected dimensions
- **Solution**: Check c_out and target_wave_indices settings

## Next Steps After Debugging

1. **If simplified debug works**: Move to full production script
2. **If memory issues**: Adjust configuration parameters
3. **If shape issues**: Review model architecture and data pipeline
4. **If training works**: Monitor convergence and metrics

## Quick Test Command
```bash
# Run simplified debug directly
cd /path/to/project
python scripts/train/debug_celestial_production.py
```

This will give you immediate feedback on what's working and what needs attention.