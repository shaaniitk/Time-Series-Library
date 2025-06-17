# Enhanced Autoformer Mode-Aware Training Guide

## Overview

This guide explains how to use the Enhanced Autoformer variants with mode-aware loss functions for different forecasting scenarios (M, MS, S modes). **NEW: Now with fully dynamic configuration that automatically adapts to any dataset!**

## 🎉 NEW: Dynamic Configuration System

**No more hardcoded dimensions!** The system now automatically detects your dataset's characteristics and generates appropriate configurations.

### Key Features:
- ✅ **Automatic dimension detection** - Reads any CSV and determines number of features/targets
- ✅ **Dynamic config generation** - Creates configs that match your exact data
- ✅ **Interactive selection** - Guides you through choosing the right complexity and mode
- ✅ **Validation system** - Checks existing configs against your data
- ✅ **Template system** - Flexible base templates for any dataset

### Quick Start with Dynamic System:

#### Option 1: Interactive Config Selection
```bash
# Analyzes your data and creates personalized config
python select_dynamic_config.py
```

#### Option 2: Auto-Generate All Configs
```bash
# Generate configs for any dataset
python generate_dynamic_configs.py data/your_dataset.csv

# Creates 15 configs: 5 complexity levels × 3 modes
# All with correct dimensions for YOUR data!
```

#### Option 3: Training with Dynamic Script
```bash
# Train with automatic dimension detection
python train_dynamic_autoformer.py \
    --config config_enhanced_autoformer_MS_medium_auto.yaml \
    --model_type enhanced \
    --auto_fix  # Automatically fixes dimension mismatches
```

## Forecasting Modes

### 🎯 Mode M (Multivariate)
- **Input**: ALL features from your dataset (auto-detected)
- **Output**: ALL features 
- **Use case**: Full ecosystem forecasting
- **Config**: `enc_in=N_total, c_out=N_total` (auto-set)

### 🎯 Mode MS (Multivariate-to-Multi-target)
- **Input**: ALL features (rich context)
- **Output**: Target features only (auto-detected)
- **Use case**: Target prediction with full context
- **Config**: `enc_in=N_total, c_out=N_targets` (auto-set)

### 🎯 Mode S (Target-only)
- **Input**: Target features only (auto-detected)
- **Output**: Target features only
- **Use case**: Pure target dynamics
- **Config**: `enc_in=N_targets, c_out=N_targets` (auto-set)

## Model Variants

### 1. Enhanced Autoformer
- Adaptive auto-correlation
- Learnable decomposition
- Multi-scale correlation analysis

### 2. Bayesian Enhanced Autoformer
- Uncertainty quantification
- Bayesian layers with KL divergence
- Probabilistic predictions

### 3. Hierarchical Enhanced Autoformer
- Multi-resolution wavelet processing
- Cross-resolution attention
- Hierarchical decomposition

## Mode-Aware Loss Functions

### Key Features
- **Automatic feature slicing**: No manual slicing needed in training code
- **Mode-specific handling**: Different logic for M, MS, S modes
- **Consistent scaling**: Proper handling of scaled/unscaled data
- **Component tracking**: Detailed loss breakdown for debugging

### Usage Examples

```python
from utils.enhanced_losses import create_enhanced_loss

# Create mode-aware loss for different model types
loss_enhanced = create_enhanced_loss(model_type='enhanced', mode='MS')
loss_bayesian = create_enhanced_loss(model_type='bayesian', mode='S') 
loss_hierarchical = create_enhanced_loss(model_type='hierarchical', mode='M')

# In training loop - no manual slicing needed!
outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
loss = loss_function(outputs[:, -pred_len:, :], batch_y[:, -pred_len:, :])
```

## Dynamic Configuration Files

### Auto-Generated Configs:
Your dataset will automatically generate 15 configurations:

**Ultra-Light Configs** (Quick prototyping):
- `config_enhanced_autoformer_M_ultralight_auto.yaml`
- `config_enhanced_autoformer_MS_ultralight_auto.yaml` 
- `config_enhanced_autoformer_S_ultralight_auto.yaml`

**Light Configs** (Development):
- `config_enhanced_autoformer_M_light_auto.yaml`
- `config_enhanced_autoformer_MS_light_auto.yaml`
- `config_enhanced_autoformer_S_light_auto.yaml`

**Medium Configs** (Balanced performance):
- `config_enhanced_autoformer_M_medium_auto.yaml`
- `config_enhanced_autoformer_MS_medium_auto.yaml`
- `config_enhanced_autoformer_S_medium_auto.yaml`

**Heavy Configs** (High accuracy):
- `config_enhanced_autoformer_M_heavy_auto.yaml`
- `config_enhanced_autoformer_MS_heavy_auto.yaml`
- `config_enhanced_autoformer_S_heavy_auto.yaml`

**Very Heavy Configs** (Maximum capacity):
- `config_enhanced_autoformer_M_veryheavy_auto.yaml`
- `config_enhanced_autoformer_MS_veryheavy_auto.yaml`
- `config_enhanced_autoformer_S_veryheavy_auto.yaml`

### Example Auto-Generated Config Structure:
```yaml
# Auto-generated for YOUR dataset
enc_in: 118      # ← Automatically detected from your data
dec_in: 4        # ← Based on your target columns  
c_out: 4         # ← Matches your forecasting mode

_data_analysis:  # ← Metadata about your dataset
  n_total_features: 118
  n_targets: 4
  n_covariates: 114
  target_columns: ['log_Open', 'log_High', 'log_Low', 'log_Close']
  mode_description: "Multi-target: 118 → 4"
```

## New Training Scripts

### 1. Dynamic Training Script
```bash
# Trains any model with any dataset
python train_dynamic_autoformer.py \
    --config config_enhanced_autoformer_MS_medium_auto.yaml \
    --model_type enhanced \
    --validate_data \  # Validates config against data
    --auto_fix         # Fixes dimension mismatches automatically
```

### 2. Data Analysis Utility
```bash
# Analyze any dataset
python utils/data_analysis.py data/your_dataset.csv

# Output:
# Total features: 118
# Target features: 4 ['log_Open', 'log_High', 'log_Low', 'log_Close']  
# Covariate features: 114
# Mode configurations automatically determined!
```

## Scaling Behavior

### ✅ Training Loss
- Both predictions and ground truth are **already scaled**
- **No additional scaling needed** - direct comparison
- Mode-aware loss handles feature selection automatically

### ✅ Validation/Test Loss  
- Model predictions are **scaled** (trained on scaled data)
- Ground truth from dataset is **unscaled** (to avoid data leakage)
- **Mode-aware loss scales ground truth internally** before comparison

## Architecture Compatibility

All three forecasting modes are **fully supported** by the Autoformer architecture:

```python
# Encoder input: configs.enc_in (flexible)
# Decoder output: configs.c_out (via projection layer)
# Mode-aware loss: handles feature slicing automatically
```

## Performance Comparison

### Expected Computational Complexity (Medium Config):
1. **S Mode**: Fastest (4 features only)
2. **MS Mode**: Medium (118→4 features)  
3. **M Mode**: Slowest (118→118 features)

### Expected Forecasting Performance:
1. **MS Mode**: Best balance (rich input, focused output)
2. **M Mode**: Maximum information but potentially noisy
3. **S Mode**: Pure price dynamics, good for technical analysis

## File Structure

```
Time-Series-Library/
├── config_enhanced_autoformer_M_medium.yaml     # M mode config
├── config_enhanced_autoformer_MS_medium.yaml    # MS mode config  
├── config_enhanced_autoformer_S_medium.yaml     # S mode config
├── train_configurable_autoformer.py             # Main trainer
├── train_all_enhanced_autoformers.py            # Batch trainer
├── test_mode_aware_losses.py                    # Loss function tests
├── utils/enhanced_losses.py                     # Mode-aware loss functions
└── models/
    ├── EnhancedAutoformer.py                    # Enhanced variant
    ├── BayesianEnhancedAutoformer.py            # Bayesian variant
    └── HierarchicalEnhancedAutoformer.py        # Hierarchical variant
```

## Next Steps

1. **Test individual configs**:
   ```bash
   python train_configurable_autoformer.py --config config_enhanced_autoformer_MS_medium.yaml --model_type enhanced
   ```

2. **Run comprehensive training**:
   ```bash
   python train_all_enhanced_autoformers.py
   ```

3. **Compare results** across modes and model types

4. **Adjust complexity** by modifying config files (d_model, layers, seq_len)

## Benefits

✅ **No manual feature slicing** in training code  
✅ **Consistent scaling behavior** across modes  
✅ **Automatic mode detection** and appropriate loss computation  
✅ **Flexible architecture** supporting all input/output combinations  
✅ **Comprehensive testing** with detailed diagnostics  
✅ **Easy configuration** via YAML files  
✅ **Dynamic configuration** that adapts to any dataset  

The enhanced autoformers with mode-aware loss functions provide a robust, flexible framework for financial time series forecasting across different modeling scenarios.
