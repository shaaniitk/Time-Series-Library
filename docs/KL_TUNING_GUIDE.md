# 🎯 KL Loss Tuning Guide for Bayesian Time Series Models

## Overview

KL (Kullback-Leibler) loss in Bayesian neural networks regularizes the learned parameter distributions by measuring their deviation from prior distributions. Proper KL tuning is crucial for:
- **Preventing overfitting** through regularization
- **Enabling uncertainty quantification** 
- **Balancing model complexity** vs. fit quality

## 🚀 Quick Start

### 1. Estimate Initial KL Weight
```python
from utils.kl_tuning import suggest_kl_weight

# Estimate from first few training batches
initial_data_loss = 0.5  # Your measured data loss
suggested_weight = suggest_kl_weight(initial_data_loss, target_percentage=0.1)
model.kl_weight = suggested_weight
```

### 2. Setup Adaptive KL Tuning
```python
from utils.kl_tuning import KLTuner

kl_tuner = KLTuner(
    model=model,
    target_kl_percentage=0.1,  # 10% of total loss
    min_weight=1e-6,
    max_weight=5e-2
)
```

### 3. Update During Training
```python
# In your training loop
for epoch in range(num_epochs):
    # ... training code ...
    
    # Update KL weight adaptively
    new_weight, kl_contribution = kl_tuner.update_kl_weight(
        epoch=epoch,
        data_loss=avg_data_loss,
        kl_loss=avg_kl_loss,
        method='adaptive'
    )
    
    print(f"Epoch {epoch}: KL weight={new_weight:.2e}, KL%={kl_contribution*100:.1f}%")

# Save tuning history
kl_tuner.plot_kl_tuning_history('kl_history.png')
```

## 🎛️ KL Tuning Methods

### 1. **Adaptive Tuning** (Recommended)
- **Purpose**: Automatically maintains target KL percentage
- **Best for**: Most scenarios, stable training
- **Settings**: `target_kl_percentage=0.1` (10%)

```python
kl_tuner.update_kl_weight(epoch, data_loss, kl_loss, method='adaptive')
```

### 2. **Annealing Schedules**
- **Purpose**: Predetermined weight reduction over time
- **Best for**: When you want decreasing regularization
- **Options**: `linear`, `cosine`, `exponential`, `cyclical`

```python
kl_tuner.update_kl_weight(
    epoch, data_loss, kl_loss, 
    method='annealing',
    total_epochs=100,
    schedule_type='cosine'
)
```

### 3. **Fixed Weight**
- **Purpose**: Constant KL weight throughout training
- **Best for**: Baseline comparisons, simple scenarios

```python
kl_tuner.update_kl_weight(epoch, data_loss, kl_loss, method='fixed')
```

## 📊 Target KL Contribution Guidelines

| KL % | Use Case | Description |
|------|----------|-------------|
| 0-2% | Minimal regularization | Focus on fit quality |
| 5-10% | **Standard choice** | Balanced regularization |
| 10-15% | Medium regularization | Good uncertainty estimates |
| 15-25% | Heavy regularization | Prevent overfitting |
| >25% | Usually too high | May cause underfitting |

## 🔧 Troubleshooting

### Problem: KL Loss Explodes
**Symptoms**: KL loss grows exponentially, training unstable
**Solutions**:
- Lower `prior_std` (e.g., from 0.1 to 0.01)
- Reduce initial KL weight
- Check layer initialization
- Use gradient clipping

### Problem: KL Too Stable/Low
**Symptoms**: KL contribution stays <3%, no regularization effect
**Solutions**:
- Increase learning rate
- Increase `prior_std`
- Check if Bayesian layers are properly enabled
- Increase KL weight manually

### Problem: Erratic KL Percentage
**Symptoms**: KL% fluctuates wildly between epochs
**Solutions**:
- Use longer smoothing window in adaptive tuning
- Reduce learning rate
- Increase batch size for more stable estimates

### Problem: Model Underfits
**Symptoms**: High training loss, poor convergence
**Solutions**:
- Reduce target KL percentage
- Lower KL weight
- Check if priors are too restrictive

### Problem: Model Overfits
**Symptoms**: Large gap between train/val loss
**Solutions**:
- Increase target KL percentage
- Use more aggressive annealing
- Add other regularization techniques

## 🎨 Visualization and Monitoring

### Key Metrics to Track
1. **KL Contribution %**: Should stay near target
2. **KL Weight Evolution**: Should adapt smoothly
3. **Data Loss vs KL Loss**: Both should decrease
4. **Total Loss**: Overall training progress

### Interpretation
- **Healthy KL tuning**: Smooth KL% around target, decreasing total loss
- **Too aggressive**: KL% jumps around, training instability
- **Too conservative**: KL% well below target, possible overfitting

## 🧪 Advanced Techniques

### 1. Multi-Stage KL Tuning
```python
# Stage 1: High KL for exploration
if epoch < 10:
    target_pct = 0.20
# Stage 2: Medium KL for refinement  
elif epoch < 25:
    target_pct = 0.10
# Stage 3: Low KL for fine-tuning
else:
    target_pct = 0.05

kl_tuner.target_kl_percentage = target_pct
```

### 2. Loss-Dependent KL Adjustment
```python
# Increase KL weight if validation loss increases
if val_loss > best_val_loss:
    model.kl_weight *= 1.1  # Increase regularization
else:
    model.kl_weight *= 0.95  # Decrease regularization
```

### 3. Quantile-Specific KL Tuning
For models with both quantile and KL loss:
```python
# Balance quantile loss and KL loss
quantile_weight = 1.0
kl_weight = model.kl_weight

total_loss = quantile_weight * quantile_loss + kl_weight * kl_loss
```

## 📈 Integration with Existing Code

### With BayesianEnhancedAutoformer
```python
# Model setup
configs.bayesian_layers = True
configs.kl_weight = 0.01  # Initial weight
model = BayesianEnhancedAutoformer(configs)

# KL tuner setup
kl_tuner = KLTuner(model, target_kl_percentage=0.1)

# Training loop
for epoch in range(num_epochs):
    # ... forward/backward pass ...
    
    data_loss = criterion(outputs, targets)
    kl_loss = model.kl_loss()
    
    # Update KL weight
    new_weight, kl_pct = kl_tuner.update_kl_weight(
        epoch, data_loss.item(), kl_loss.item(), 'adaptive'
    )
    
    total_loss = data_loss + model.kl_weight * kl_loss
    total_loss.backward()
```

### Command Line Usage
```bash
# Basic training with adaptive KL tuning
python train_bayesian_with_kl_tuning.py \
    --model BayesianEnhancedAutoformer \
    --kl_tuning_method adaptive \
    --target_kl_percentage 0.1

# Ablation study
python train_bayesian_with_kl_tuning.py \
    --des ablation \
    --model BayesianEnhancedAutoformer

# Custom KL settings
python train_bayesian_with_kl_tuning.py \
    --kl_tuning_method annealing \
    --annealing_schedule cosine \
    --max_kl_weight 0.05
```

## 🎯 Best Practices Summary

1. **Start with suggested weight**: Use `suggest_kl_weight()` for initialization
2. **Use adaptive tuning**: Works well for most scenarios
3. **Target 10% KL**: Good starting point for most models
4. **Monitor trends**: Watch KL% over time, not just absolute values
5. **Visualize history**: Use plotting tools to diagnose issues
6. **Adjust based on symptoms**: Follow troubleshooting guide
7. **Validate on held-out data**: Ensure good generalization
8. **Save tuning history**: For reproducibility and analysis

## 📁 File Structure

```
utils/
├── kl_tuning.py              # Main KL tuning utilities
├── enhanced_losses.py        # Enhanced loss functions
└── bayesian_losses.py        # Bayesian-specific losses

examples/
├── kl_tuning_demo.py         # Basic demonstration
├── practical_kl_tuning.py    # Practical integration
└── train_bayesian_with_kl_tuning.py  # Full training script

models/
└── BayesianEnhancedAutoformer.py  # Model with KL support
```

## 🔗 Related Topics

- **Quantile Loss**: Combines well with KL loss for full uncertainty
- **Prior Selection**: Choice of priors affects KL behavior  
- **Uncertainty Calibration**: Proper KL tuning improves calibration
- **Model Selection**: Use KL contribution for model comparison

---

*This guide provides comprehensive coverage of KL loss tuning for Bayesian time series models. For specific implementation details, refer to the provided code examples and utilities.*
