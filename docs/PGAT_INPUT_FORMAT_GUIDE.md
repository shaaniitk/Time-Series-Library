# PGAT Input Format Guide

## 🎯 **Understanding SOTA_Temporal_PGAT Input Requirements**

The SOTA_Temporal_PGAT has a specific input format that's different from standard time series models. Understanding this is crucial for proper usage.

## 📊 **Input Format Overview**

### **Standard Time Series Model (e.g., Transformer)**

```python
# Standard approach
x_enc: [batch, seq_len, features]     # Historical data
x_dec: [batch, pred_len, features]    # Decoder input (optional)
# Output: [batch, pred_len, targets]
```

### **SOTA_Temporal_PGAT Format**

```python
# PGAT approach
wave_window: [batch, seq_len, ALL_features]    # Historical data
target_window: [batch, pred_len, ALL_features] # Future data (same feature dim!)
graph: [nodes, nodes]                          # Graph structure
# Output: [batch, pred_len, targets_only]
```

## 🔑 **Key Difference: Feature Dimensions**

### **Critical Rule:**

**Both `wave_window` and `target_window` MUST have the same feature dimension.**

### **Why?**

PGAT concatenates both windows along the time dimension:

```python
# Inside PGAT forward()
combined_input = torch.cat([wave_window, target_window], dim=1)
# Result: [batch, seq_len + pred_len, features]
```

## 📝 **Correct Input Setup**

### **Example: 2 Covariates + 2 Targets**

```python
# Configuration
enc_in = 4      # 2 covariates + 2 targets
c_out = 2       # Only targets in output
seq_len = 24    # Historical length
pred_len = 6    # Prediction length

# Correct input shapes
wave_window = torch.randn(batch, 24, 4)    # Historical: all features
target_window = torch.randn(batch, 6, 4)   # Future: all features
graph = torch.randn(10, 10)                # Graph structure

# Output shape
output = model(wave_window, target_window, graph)
# output.shape = [batch, 6, 2]  # Only targets predicted
```

### **What Each Window Contains**

#### **wave_window: Historical Data**

```python
# Shape: [batch, seq_len, enc_in]
# Content: [covariate_0, covariate_1, target_0, target_1]
# Time: t-seq_len to t-1
```

#### **target_window: Future Data**

```python
# Shape: [batch, pred_len, enc_in]  # Same feature dim as wave_window!
# Content: [covariate_0, covariate_1, target_0, target_1]
# Time: t to t+pred_len-1
# Note: During inference, future covariates are known, targets are unknown
```

## 🔧 **Implementation Examples**

### **Training Data Preparation**

```python
def prepare_pgat_data(df, seq_len, pred_len):
    """Prepare data for PGAT training."""

    # Assume df has columns: ['covariate_0', 'covariate_1', 'target_0', 'target_1']
    all_features = ['covariate_0', 'covariate_1', 'target_0', 'target_1']
    target_features = ['target_0', 'target_1']

    wave_windows = []
    target_windows = []

    for i in range(len(df) - seq_len - pred_len + 1):
        # Historical window: all features
        wave_window = df[all_features].iloc[i:i+seq_len].values

        # Future window: all features (including future covariates)
        target_window = df[all_features].iloc[i+seq_len:i+seq_len+pred_len].values

        wave_windows.append(wave_window)
        target_windows.append(target_window)

    return np.array(wave_windows), np.array(target_windows)
```

### **Inference Data Preparation**

```python
def prepare_pgat_inference(df, seq_len, pred_len, future_covariates):
    """Prepare data for PGAT inference."""

    # Historical data: all features
    wave_window = df[all_features].iloc[-seq_len:].values

    # Future data: known covariates + placeholder targets
    target_window = np.zeros((pred_len, len(all_features)))
    target_window[:, :2] = future_covariates  # Known future covariates
    target_window[:, 2:] = 0  # Unknown targets (will be predicted)

    return wave_window, target_window
```

## ❌ **Common Mistakes**

### **Mistake 1: Different Feature Dimensions**

```python
# WRONG ❌
wave_window = torch.randn(batch, 24, 4)    # 4 features
target_window = torch.randn(batch, 6, 2)   # 2 features - MISMATCH!

# Error: RuntimeError: Sizes of tensors must match except in dimension 1
```

### **Mistake 2: Only Targets in target_window**

```python
# WRONG ❌
wave_window = torch.randn(batch, 24, 4)    # All features
target_window = torch.randn(batch, 6, 2)   # Only targets

# Should be:
target_window = torch.randn(batch, 6, 4)   # All features ✅
```

### **Mistake 3: Confusing Input vs Output Dimensions**

```python
# Configuration
enc_in = 4      # Input features (covariates + targets)
c_out = 2       # Output features (targets only)

# WRONG ❌: Using c_out for target_window
target_window = torch.randn(batch, pred_len, c_out)  # Wrong!

# CORRECT ✅: Using enc_in for both windows
wave_window = torch.randn(batch, seq_len, enc_in)
target_window = torch.randn(batch, pred_len, enc_in)
```

## 🎯 **Quick Fix for Dimension Errors**

If you get dimension mismatch errors:

1. **Check shapes:**

   ```python
   print(f"wave_window: {wave_window.shape}")
   print(f"target_window: {target_window.shape}")
   # Both should have same last dimension!
   ```

2. **Fix target_window:**

   ```python
   # If target_window has wrong dimension
   if target_window.shape[-1] != wave_window.shape[-1]:
       # Pad with zeros or known future covariates
       batch, pred_len, wrong_dim = target_window.shape
       correct_dim = wave_window.shape[-1]

       # Create correctly sized tensor
       fixed_target_window = torch.zeros(batch, pred_len, correct_dim)
       fixed_target_window[:, :, -wrong_dim:] = target_window  # Put targets at end
       target_window = fixed_target_window
   ```

## 📚 **Summary**

**Key Points:**

1. **Same Feature Dimension**: Both windows must have `enc_in` features
2. **Different Content**: Historical vs future data, but same structure
3. **Output Dimension**: Model outputs only `c_out` target predictions
4. **Concatenation**: PGAT concatenates windows along time dimension

**Remember:**

- `enc_in` = total input features (covariates + targets)
- `c_out` = output features (targets only)
- Both input windows use `enc_in`, output uses `c_out`

This format allows PGAT to process the complete temporal sequence while maintaining the sophisticated graph-based relationships between different types of features.
