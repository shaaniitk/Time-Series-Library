# Production Training with Comprehensive Diagnostics

Complete guide for running GPU training with full diagnostic logging for gradient analysis and component improvement.

---

## Quick Start

### 1. Simple Command (Recommended for GPU)

```bash
python -X utf8 scripts/train/train_with_diagnostics.py \
    --config configs/celestial_enhanced_pgat_production.yaml \
    --enable_gradient_diagnostics \
    --enable_attention_logging \
    --enable_component_profiling \
    --log_interval 10 \
    --gradient_log_interval 50 \
    --attention_log_interval 100 \
    --checkpoint_interval 500
```

### 2. Full Command with Custom Experiment Name

```bash
EXPERIMENT_NAME="hierarchical_fusion_c2t_$(date +%Y%m%d_%H%M%S)"

python -X utf8 scripts/train/train_with_diagnostics.py \
    --config configs/celestial_enhanced_pgat_production.yaml \
    --experiment_name "${EXPERIMENT_NAME}" \
    --log_dir "logs/${EXPERIMENT_NAME}" \
    --checkpoint_dir "checkpoints/${EXPERIMENT_NAME}" \
    --enable_gradient_diagnostics \
    --enable_attention_logging \
    --enable_component_profiling \
    --log_interval 10 \
    --gradient_log_interval 50 \
    --attention_log_interval 100 \
    --checkpoint_interval 500 \
    2>&1 | tee "logs/${EXPERIMENT_NAME}/training.log"
```

### 3. Using Shell Script (Linux/Mac)

```bash
chmod +x run_production_with_diagnostics.sh
./run_production_with_diagnostics.sh
```

### 4. Windows PowerShell

```powershell
$EXPERIMENT_NAME = "hierarchical_fusion_c2t_$(Get-Date -Format 'yyyyMMdd_HHmmss')"

python -X utf8 scripts/train/train_with_diagnostics.py `
    --config configs/celestial_enhanced_pgat_production.yaml `
    --experiment_name $EXPERIMENT_NAME `
    --log_dir "logs/$EXPERIMENT_NAME" `
    --checkpoint_dir "checkpoints/$EXPERIMENT_NAME" `
    --enable_gradient_diagnostics `
    --enable_attention_logging `
    --enable_component_profiling `
    --log_interval 10 `
    --gradient_log_interval 50 `
    --attention_log_interval 100 `
    --checkpoint_interval 500 `
    2>&1 | Tee-Object "logs/$EXPERIMENT_NAME/training.log"
```

---

## What Gets Logged

### 1. Gradient Diagnostics (`--enable_gradient_diagnostics`)

**Location**: `logs/<experiment>/gradients/`

**Files**:
- `gradients_step_XXXXXX.json`: Detailed gradient stats per component
- `history.json`: Complete gradient evolution timeline

**Metrics per component**:
- Total gradient norm
- Mean, std, max, min gradient values
- Per-parameter gradient statistics
- Component-wise gradient distribution

**Components tracked**:
- `fusion_cross_attention`: Hierarchical fusion module
- `hierarchical_fusion_proj`: Fusion projection layer
- `celestial_to_target_attention`: C→T attention mechanism
- `wave_patching_composer`: Wave multi-scale patching
- `target_patching_composer`: Target multi-scale patching
- `graph_combiner`: Graph meta-controller
- `stochastic_learner`: Stochastic graph learner
- `decoder`: Output decoder

### 2. Attention Logging (`--enable_attention_logging`)

**Location**: `logs/<experiment>/attention/`

**Files**:
- `attention_step_XXXXXX.json`: Attention weights per step

**Metrics**:
- **Fusion attention**:
  - Attention weights over all (scale, timestep) pairs
  - Attention entropy (distribution measure)
  - Scale-specific attention patterns
- **C→T attention**:
  - Per-target attention to celestial bodies
  - Top-K celestial influences per asset
  - Attention statistics (mean, std, max)

### 3. Component Profiling (`--enable_component_profiling`)

**Location**: `logs/<experiment>/components/`

**Files**:
- `profile_step_XXXXXX.json`: Component-level metrics

**Metrics**:
- Parameter counts per component
- Trainable vs frozen parameters
- Enhanced config information
- Internal model logs
- Component utilization

### 4. TensorBoard Logs

**Location**: `logs/<experiment>/tensorboard/`

**Metrics**:
- Training loss curves
- Validation loss curves
- Component-wise gradient norms
- Learning rate schedule
- Custom scalars

### 5. Checkpoints

**Location**: `checkpoints/<experiment>/`

**Files**:
- `checkpoint_step_XXXXXX.pth`: Periodic checkpoints
- `best_model.pth`: Best validation performance
- `final_model.pth`: Final trained model

---

## Command-Line Arguments

### Required
- `--config`: Path to YAML config file

### Experiment Configuration
- `--experiment_name`: Name for this experiment (default: auto-generated timestamp)
- `--log_dir`: Directory for logs (default: `logs/diagnostics`)
- `--checkpoint_dir`: Directory for checkpoints (default: `checkpoints/diagnostics`)

### Diagnostic Flags
- `--enable_gradient_diagnostics`: Enable gradient flow analysis
- `--enable_attention_logging`: Enable attention weight logging
- `--enable_component_profiling`: Enable component performance profiling

### Logging Intervals
- `--log_interval`: Batch interval for basic logging (default: 10)
- `--gradient_log_interval`: Step interval for gradient logging (default: 50)
- `--attention_log_interval`: Step interval for attention logging (default: 100)
- `--checkpoint_interval`: Step interval for checkpoints (default: 500)

---

## Analyzing Results

### 1. Run Analysis Script

```bash
python scripts/analysis/analyze_training_diagnostics.py \
    --log_dir logs/<your_experiment_name>
```

### 2. What You Get

**Generated Files** (in `logs/<experiment>/analysis/`):
- `gradient_norms.csv`: Gradient evolution data
- `gradient_flow.png`: Gradient flow visualization
- `fusion_attention_entropy.png`: Attention distribution evolution
- `recommendations.json`: Automated improvement suggestions

**Console Output**:
- Gradient flow analysis per component
- Vanishing/exploding gradient warnings
- Attention pattern analysis
- Parameter distribution breakdown
- Actionable improvement recommendations

### 3. TensorBoard Visualization

```bash
tensorboard --logdir logs/<your_experiment_name>/tensorboard
```

Access at: `http://localhost:6006`

---

## Improvement Workflow

### Step 1: Train with Diagnostics

```bash
python -X utf8 scripts/train/train_with_diagnostics.py \
    --config configs/celestial_enhanced_pgat_production.yaml \
    --enable_gradient_diagnostics \
    --enable_attention_logging \
    --enable_component_profiling
```

### Step 2: Analyze Results

```bash
python scripts/analysis/analyze_training_diagnostics.py \
    --log_dir logs/<experiment_name>
```

### Step 3: Review Recommendations

Open `logs/<experiment_name>/analysis/recommendations.json`:

```json
[
  {
    "component": "fusion_cross_attention",
    "issue": "Vanishing gradients",
    "severity": "HIGH",
    "suggestion": "Consider: (1) Increase learning rate for fusion_cross_attention, (2) Add skip connections, (3) Use different initialization"
  }
]
```

### Step 4: Apply Improvements

Based on recommendations, modify config or code:

**Example 1: Adjust learning rate per component**
```yaml
# In config.yaml
component_learning_rates:
  fusion_cross_attention: 0.0005  # Increased from 0.0001
  celestial_to_target_attention: 0.0003
  default: 0.0001
```

**Example 2: Modify gradient clipping**
```yaml
# In config.yaml
gradient_clip_norm: 2.0  # Increased from 1.0
```

**Example 3: Add normalization**
```python
# In model code
self.fusion_norm = nn.LayerNorm(d_model)
refined_temporal = self.fusion_norm(refined_temporal)
```

### Step 5: Re-train and Compare

```bash
python -X utf8 scripts/train/train_with_diagnostics.py \
    --config configs/celestial_enhanced_pgat_production_improved.yaml \
    --experiment_name "improved_$(date +%Y%m%d_%H%M%S)" \
    --enable_gradient_diagnostics \
    --enable_attention_logging \
    --enable_component_profiling
```

---

## Key Metrics to Monitor

### 1. Gradient Health
- ✅ **Healthy**: Mean gradient norm 0.001 - 1.0
- ⚠️ **Vanishing**: Mean gradient norm < 1e-5
- ⚠️ **Exploding**: Max gradient norm > 100

### 2. Attention Patterns
- ✅ **Healthy**: Entropy increases over training (learning to attend)
- ⚠️ **Issue**: Entropy stays constant (not learning)
- ✅ **Distributed**: Attention spread across multiple sources
- ⚠️ **Collapsed**: Attention focused on single source

### 3. Component Balance
- ✅ **Balanced**: All components show similar gradient magnitudes
- ⚠️ **Imbalanced**: Some components have 10x+ different gradients

### 4. Training Dynamics
- ✅ **Healthy**: Smooth loss decrease
- ⚠️ **Unstable**: Large loss spikes
- ⚠️ **Plateaued**: No improvement for many epochs

---

## Example Analysis Output

```
================================================================================
GRADIENT ANALYSIS
================================================================================
Found 100 gradient snapshots
Saved gradient norms to: logs/.../analysis/gradient_norms.csv
Saved gradient plot to: logs/.../analysis/gradient_flow.png

Gradient Flow Analysis:

fusion_cross_attention:
  Mean norm: 0.045231
  Std norm: 0.012456
  Max norm: 0.089432
  Min norm: 0.015678

celestial_to_target_attention:
  Mean norm: 0.032145
  Std norm: 0.008234
  Max norm: 0.056789
  Min norm: 0.012345

================================================================================
ATTENTION ANALYSIS
================================================================================
Found 50 attention snapshots
Saved fusion attention plot to: logs/.../analysis/fusion_attention_entropy.png

Fusion Attention Analysis:
  Mean entropy: 4.2341
  Entropy trend: Increasing

C→T Attention Analysis:

Bitcoin:
  Mean attention: 0.0769
  Attention spread (std): 0.0234
  Max attention: 0.1523
  Top 5 celestial bodies: [3, 7, 1, 9, 4]

================================================================================
COMPONENT ANALYSIS
================================================================================
Parameter Breakdown:

fusion_cross_attention:
  Total parameters: 264,192
  Trainable parameters: 264,192
  Percentage of total: 12.34%

celestial_to_target_attention:
  Total parameters: 131,328
  Trainable parameters: 131,328
  Percentage of total: 6.13%

Total Model Parameters: 2,145,280
Total Trainable Parameters: 2,145,280

================================================================================
IMPROVEMENT RECOMMENDATIONS
================================================================================

✅ No major issues detected. Training appears healthy!

================================================================================
ANALYSIS COMPLETE
================================================================================
```

---

## Troubleshooting

### Issue: Out of Memory on GPU

**Solution 1**: Reduce batch size
```yaml
batch_size: 8  # Reduce from 12
```

**Solution 2**: Reduce logging frequency
```bash
--gradient_log_interval 200  # Increase from 50
--attention_log_interval 500  # Increase from 100
```

### Issue: Training Very Slow

**Solution**: Disable some diagnostics
```bash
# Only enable gradient diagnostics (fastest)
--enable_gradient_diagnostics
```

### Issue: Vanishing Gradients Detected

**Solution**: Adjust learning rate or architecture
```yaml
learning_rate: 0.0005  # Increase from 0.0001
```

### Issue: Exploding Gradients

**Solution**: Increase gradient clipping
```yaml
gradient_clip_norm: 2.0  # Increase from 1.0
```

---

## Best Practices

1. **Start with full diagnostics**: Enable all flags for first run
2. **Analyze early**: Check diagnostics after 1-2 epochs
3. **Iterate quickly**: Apply improvements and re-run
4. **Compare experiments**: Use TensorBoard to compare runs
5. **Monitor GPU usage**: Use `nvidia-smi` to check utilization
6. **Save everything**: Disk is cheap, diagnostic data is valuable
7. **Document changes**: Keep notes on what you tried

---

## Additional Resources

- **Config files**: `configs/celestial_enhanced_pgat_production.yaml`
- **Training script**: `scripts/train/train_with_diagnostics.py`
- **Analysis script**: `scripts/analysis/analyze_training_diagnostics.py`
- **Validation report**: `FUSION_C2T_VALIDATION_REPORT.md`
- **Status summary**: `FUSION_C2T_STATUS.txt`

---

**Generated**: October 26, 2025  
**For**: GPU Production Training with Diagnostic Analysis
