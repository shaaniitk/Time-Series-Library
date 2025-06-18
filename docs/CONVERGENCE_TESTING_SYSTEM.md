#!/usr/bin/env python3
"""
Convergence Testing System Documentation

This document describes the enhanced time series forecasting library's
convergence testing system using synthetic data generation.

## Overview

The convergence testing system allows you to:
1. Generate synthetic datasets with known mathematical relationships
2. Test model convergence on deterministic patterns
3. Validate uncertainty quantification with quantile regression
4. Debug model architectures before using real data

## System Components

### 1. Synthetic Data Generator (`utils/synthetic_data_generator.py`)
- Generates sin/cos patterns with configurable complexity
- Creates multivariate datasets with known relationships
- Supports different noise levels and frequencies
- Compatible with existing data loaders

### 2. Smart Dimension Manager (`utils/dimension_manager.py`)
- Automatically detects correct model dimensions
- Eliminates manual dimension configuration
- Works with both real and synthetic data
- Updates all config files automatically

### 3. Enhanced Training Script (`scripts/train/train_dynamic_autoformer.py`)
- Supports convergence test mode with `--synthetic_data`
- Automatic dimension correction
- Quantile regression and KL loss support
- Seamless integration with synthetic data

## Quick Start

### Basic Convergence Test
```bash
# Test basic model convergence with synthetic sin/cos data
python scripts/train/train_dynamic_autoformer.py \
    --config config/config_enhanced_autoformer_MS_ultralight.yaml \
    --model_type enhanced \
    --synthetic_data \
    --synthetic_type sincos \
    --synthetic_n_points 1000 \
    --synthetic_noise 0.01
```

### Quantile Regression Test
```bash
# Test uncertainty quantification with 7 quantiles
python scripts/train/train_dynamic_autoformer.py \
    --config config/config_enhanced_autoformer_MS_light.yaml \
    --model_type enhanced \
    --synthetic_data \
    --quantile_mode \
    --num_quantiles 7
```

### Bayesian Model Test
```bash
# Test Bayesian model with KL loss on synthetic data
python scripts/train/train_dynamic_autoformer.py \
    --config config/config_enhanced_autoformer_MS_medium.yaml \
    --model_type bayesian \
    --synthetic_data \
    --quantile_mode \
    --num_quantiles 5 \
    --enable_kl \
    --kl_anneal
```

## Synthetic Data Types

### 1. Sin/Cos Basic (`sincos`)
- **Mathematical relationships**: 
  - t1 = sin(X - X1)
  - t2 = sin(X1 - X2) 
  - t3 = sin(X2 - X)
- **Features**: 3 covariates → 3 targets
- **Use case**: Basic convergence testing, debugging

### 2. Complex Synthetic (`complex`)
- **Configurable complexity**: simple, medium, complex
- **Multiple harmonics**: 1-5 harmonic components
- **Optional trends and seasonality**
- **Variable features**: 10+ covariates, 3+ targets
- **Use case**: Advanced testing, realistic scenarios

## Configuration Options

### Synthetic Data Parameters
- `--synthetic_type`: sincos, complex
- `--synthetic_n_points`: Number of data points (500-5000)
- `--synthetic_noise`: Noise level (0.0-0.5)
- `--synthetic_complexity`: simple, medium, complex

### Model Configuration
- **Ultralight**: 32 dim, 1+1 layers, 30 epochs - Fast testing
- **Light**: 64 dim, 2+1 layers, 50 epochs - Standard testing  
- **Medium**: 128 dim, 2+1 layers, 100 epochs - Thorough testing
- **Heavy**: 256 dim, 2+2 layers, 200 epochs - Full testing

## Expected Results

### Convergence Criteria
With synthetic data, expect:
- **Loss convergence**: Within 10-50 epochs for sin/cos
- **Low final loss**: < 0.1 for noise levels < 0.05
- **Stable training**: No divergence or oscillation

### Performance Benchmarks
| Config | Sin/Cos (0.01 noise) | Complex Medium | Time |
|--------|----------------------|----------------|------|
| Ultralight | MSE < 0.05 | MSE < 0.2 | 2-5 min |
| Light | MSE < 0.02 | MSE < 0.1 | 5-10 min |
| Medium | MSE < 0.01 | MSE < 0.05 | 10-20 min |

## Quantile Regression Results

With 7 quantiles on synthetic data:
- **Prediction intervals**: 90%, 60%, 30% coverage
- **Calibration**: Should match theoretical coverage
- **Uncertainty**: Higher for noisy regions, lower for clean patterns

## Troubleshooting

### Common Issues

1. **Insufficient data**: Use `n_points >= 10 * seq_len`
2. **Dimension mismatch**: Smart dimension manager auto-fixes
3. **Memory issues**: Reduce `n_points` or use ultralight config
4. **Poor convergence**: Reduce noise level or increase epochs

### Debug Steps

1. **Check data generation**:
   ```python
   from utils.synthetic_data_generator import generate_sincos_basic
   data = generate_sincos_basic(n_points=1000, noise_level=0.01)
   print(data['metadata'])
   ```

2. **Verify dimensions**:
   ```python
   from utils.dimension_manager import smart_dimension_setup
   dm = smart_dimension_setup("data/temp_synthetic_sincos.csv")
   print(dm.get_dimensions_for_mode('MS'))
   ```

3. **Monitor training**:
   ```bash
   # Watch log files
   tail -f checkpoints/*/train.log
   ```

## Advanced Usage

### Custom Mathematical Relationships
```python
# Create custom synthetic data
from utils.synthetic_data_generator import generate_complex_synthetic
custom_data = generate_complex_synthetic(
    n_points=2000,
    n_features=8,
    n_targets=4,
    complexity='complex',
    include_trend=True,
    noise_level=0.1
)
```

### Batch Testing
```bash
# Test multiple configurations
for config in ultralight light medium; do
    python scripts/train/train_dynamic_autoformer.py \
        --config config/config_enhanced_autoformer_MS_${config}.yaml \
        --synthetic_data \
        --synthetic_n_points 1000
done
```

## Integration with Real Data

After convergence testing:

1. **Switch to real data**: Remove `--synthetic_data` flag
2. **Keep same config**: Dimensions auto-adjust
3. **Monitor convergence**: Compare with synthetic performance
4. **Adjust hyperparameters**: Based on synthetic results

## Best Practices

### Development Workflow
1. **Start with synthetic**: Test new models/features
2. **Validate convergence**: Ensure basic functionality
3. **Test uncertainty**: Use quantile regression
4. **Scale to real data**: Apply lessons learned

### Performance Optimization
1. **Use ultralight config**: For rapid prototyping
2. **Low noise synthetic**: For debugging
3. **Progressive complexity**: Simple → complex synthetic → real
4. **Dimension validation**: Always use smart dimension manager

## Example Output

```
🔬 Convergence test mode enabled:
   Synthetic data type: sincos
   Data points: 1000
   Noise level: 0.01

✅ Synthetic data prepared:
   Mathematical relationships: 3
   Updated dimensions: enc_in=6, dec_in=3, c_out=3

📊 Final Training Configuration:
   Model: EnhancedAutoformer (enhanced)
   Architecture: 6 → 3
   Model parameters: 29,456

🎯 Training Results:
   Epochs: 30
   Final Loss: 0.0234
   Convergence: ✅ Achieved
```

This system provides a robust foundation for testing model convergence,
debugging architectures, and validating uncertainty quantification
before deploying on real-world datasets.
