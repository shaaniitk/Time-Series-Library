# HF Enhanced Models Documentation

## Overview

The HF Enhanced Models provide advanced time series forecasting capabilities while leveraging the stability and proven performance of Hugging Face transformers. This implementation solves the key challenge of adding sophisticated features (Bayesian uncertainty, hierarchical processing, quantile regression) to HF models without modifying their core architecture.

## Key Innovation: External Extension Strategy

Instead of modifying HF model internals, we use a **layered wrapper approach**:

1. **HF Backbone**: Unchanged and stable (HFEnhancedAutoformer)
2. **External Extensions**: Advanced features through separate processors
3. **Loss Integration**: Leverage existing loss infrastructure from `losses.py` and `bayesian_losses.py`

This ensures:
- ✅ HF model stability and proven performance
- ✅ Advanced features without backbone modifications  
- ✅ Full compatibility with existing loss functions
- ✅ Modular design for flexible feature combinations

## Architecture Differences: HF vs Existing Models

| Aspect | Existing Models | HF Enhanced Models |
|--------|----------------|-------------------|
| **Loss Functions** | Built-in support for 20+ loss types | **Same support** via external integration |
| **Covariates** | Native DimensionManager integration | **Same integration** via wrapper approach |
| **Bayesian Features** | Direct Bayesian layer replacement | **External layer conversion** (deterministic → Bayesian) |
| **Hierarchical Processing** | Built-in multi-scale decomposition | **External processors** with wavelet support |
| **Quantile Regression** | Native quantile expansion | **External quantile expansion** |
| **Parameter Count** | 8.4M - 15.2M parameters | **8.4M + extensions** (modular sizing) |
| **Stability** | Custom architecture risks | **Proven HF stability** + external features |

## Available Models

### 1. HFEnhancedAutoformer (Standard)
```python
from models.HFEnhancedAutoformer import HFEnhancedAutoformer

model = HFEnhancedAutoformer(config)
```
- **Features**: Basic HF transformer with input/output projections
- **Parameters**: ~8.4M
- **Use Case**: Baseline forecasting with HF stability

### 2. HFBayesianEnhancedAutoformer
```python
from models.HFAdvancedFactory import HFBayesianEnhancedAutoformer

config.use_bayesian = True
config.uncertainty_method = 'bayesian'  # or 'dropout'
config.n_samples = 10
model = HFBayesianEnhancedAutoformer(config)
```
- **Features**: Uncertainty quantification, confidence intervals
- **Technical Solution**: External layer conversion (deterministic → Bayesian)
- **Parameters**: ~8.4M + Bayesian extensions
- **Use Case**: Risk-aware forecasting, uncertainty bounds

### 3. HFHierarchicalEnhancedAutoformer
```python
from models.HFAdvancedFactory import HFHierarchicalEnhancedAutoformer

config.use_hierarchical = True
config.use_wavelet = True
config.hierarchy_levels = [1, 2, 4]
model = HFHierarchicalEnhancedAutoformer(config)
```
- **Features**: Multi-scale processing, wavelet decomposition
- **Technical Solution**: External hierarchical processors
- **Parameters**: ~8.4M + hierarchical extensions  
- **Use Case**: Complex temporal patterns, multi-resolution analysis

### 4. HFQuantileEnhancedAutoformer
```python
from models.HFAdvancedFactory import HFQuantileEnhancedAutoformer

config.use_quantile = True
config.quantile_levels = [0.1, 0.5, 0.9]
config.loss_function = 'pinball'
model = HFQuantileEnhancedAutoformer(config)
```
- **Features**: Quantile regression, distribution forecasting
- **Technical Solution**: External quantile expansion layer
- **Parameters**: ~8.4M + quantile extensions
- **Use Case**: Risk assessment, prediction intervals

### 5. HFFullEnhancedAutoformer
```python
from models.HFAdvancedFactory import HFFullEnhancedAutoformer

config.use_bayesian = True
config.use_hierarchical = True  
config.use_quantile = True
model = HFFullEnhancedAutoformer(config)
```
- **Features**: All advanced capabilities combined
- **Technical Solution**: Layered external processors
- **Parameters**: ~8.4M + all extensions
- **Use Case**: Maximum sophistication, research applications

## Technical Challenges Solved

### Challenge 1: Bayesian Layers in HF Models
**Problem**: HF transformers have deterministic weights, but Bayesian methods need weight distributions.

**Solution**: External layer conversion approach
```python
# Convert specific HF layers to Bayesian equivalents
bayesian_layers = convert_to_bayesian(hf_backbone, ['output_projection'])

# Sampling during forward pass
for sample in range(n_samples):
    prediction = hf_backbone(x)  # Each call samples different weights
    predictions.append(prediction)
```

### Challenge 2: Hierarchical Processing Integration
**Problem**: HF models have fixed architecture, but hierarchical processing needs multi-scale decomposition.

**Solution**: External preprocessing and postprocessing
```python
# Preprocessing: Apply wavelet decomposition
processed_input = wavelet_processor(x_enc)

# HF backbone (unchanged)
backbone_output = hf_backbone(processed_input, ...)

# Postprocessing: Hierarchical aggregation
final_output = hierarchical_processor(backbone_output)
```

### Challenge 3: Loss Function Integration
**Problem**: HF models need to work with existing sophisticated loss infrastructure.

**Solution**: External loss management with wrapper integration
```python
# Use existing loss infrastructure
base_loss_fn = get_loss_function('pinball', config)
bayesian_loss_fn = create_bayesian_loss(base_loss_fn, config)

# Compute loss with all components
total_loss = bayesian_loss_fn(predictions, targets) + kl_loss + reg_loss
```

## Configuration Guide

### Basic Configuration
```python
config = Namespace(
    # Data dimensions
    seq_len=96,           # Input sequence length
    pred_len=24,          # Prediction horizon  
    enc_in=7,             # Number of input features
    dec_in=7,             # Number of decoder features
    c_out=7,              # Number of output features
    
    # Model architecture
    d_model=512,          # Hidden dimension
    n_heads=8,            # Attention heads
    e_layers=2,           # Encoder layers
    d_layers=1,           # Decoder layers
    dropout=0.1,          # Dropout rate
    activation='gelu',    # Activation function
)
```

### Bayesian Configuration
```python
# Add to basic config
config.use_bayesian = True
config.uncertainty_method = 'bayesian'      # 'bayesian' or 'dropout'
config.n_samples = 10                       # Monte Carlo samples
config.kl_weight = 1e-5                     # KL divergence weight
config.bayesian_layers = ['output_projection']  # Layers to convert
```

### Hierarchical Configuration
```python
# Add to basic config
config.use_hierarchical = True
config.use_wavelet = True
config.hierarchy_levels = [1, 2, 4]         # Multi-scale levels
config.aggregation_method = 'adaptive'      # 'adaptive', 'concat', 'residual'
config.wavelet_type = 'db4'                 # Wavelet type
config.decomposition_levels = 3             # Wavelet decomposition depth
```

### Quantile Configuration
```python
# Add to basic config
config.use_quantile = True
config.quantile_levels = [0.1, 0.5, 0.9]   # Quantile levels
config.loss_function = 'pinball'            # Use pinball loss
```

### Loss Function Options
All existing loss functions from `utils/losses.py` are supported:
- `'mse'` - Mean Squared Error
- `'mae'` - Mean Absolute Error  
- `'huber'` - Huber Loss
- `'pinball'` - Pinball Loss (for quantiles)
- `'dtw'` - Dynamic Time Warping Loss
- `'mase'` - Mean Absolute Scaled Error
- `'smape'` - Symmetric Mean Absolute Percentage Error
- And 15+ more advanced loss functions

## Usage Examples

### Example 1: Basic Usage
```python
from models.HFEnhancedAutoformer import HFEnhancedAutoformer

# Create model
model = HFEnhancedAutoformer(config)

# Forward pass
output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
# output shape: [batch_size, pred_len, c_out]
```

### Example 2: Bayesian Uncertainty
```python
from models.HFAdvancedFactory import create_hf_bayesian_model

# Configure for Bayesian
config.use_bayesian = True
config.uncertainty_method = 'bayesian'
config.n_samples = 10

# Create model
model = create_hf_bayesian_model(config)

# Forward with uncertainty
model.eval()
with torch.no_grad():
    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    # output is dict with:
    # - 'prediction': mean prediction
    # - 'uncertainty': standard deviation  
    # - 'confidence_intervals': 68%, 95%, 99% intervals

print(f"Mean prediction: {output['prediction'].shape}")
print(f"Uncertainty: {output['uncertainty'].shape}")
print(f"68% confidence interval: {output['confidence_intervals']['68%']}")
```

### Example 3: Training with Loss Integration
```python
from models.HFAdvancedFactory import create_hf_full_model

# Create full model
config.use_bayesian = True
config.use_quantile = True
config.loss_function = 'pinball'
model = create_hf_full_model(config)

# Training loop
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for batch in dataloader:
    x_enc, x_mark_enc, x_dec, x_mark_dec, targets = batch
    
    # Forward pass
    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    
    # Compute loss (leverages existing infrastructure)
    total_loss, components = model.compute_loss(output, targets, x_enc)
    # components: {'base_loss': ..., 'kl_loss': ..., 'regularization_loss': ...}
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

### Example 4: Auto Model Selection
```python
from models.HFAdvancedFactory import create_hf_model_from_config

# Auto-detect features from config
config.use_bayesian = True
config.quantile_levels = [0.1, 0.5, 0.9]

# Automatically creates HFBayesianQuantileModel
model = create_hf_model_from_config(config, model_type='auto')
```

## Integration with Existing Codebase

### With experiment runners:
```python
# In exp/exp_basic.py - already integrated
self.model_dict = {
    'HFEnhancedAutoformer': HFEnhancedAutoformer,
    'HFBayesianEnhancedAutoformer': HFBayesianEnhancedAutoformer,
    'HFHierarchicalEnhancedAutoformer': HFHierarchicalEnhancedAutoformer,
    'HFQuantileEnhancedAutoformer': HFQuantileEnhancedAutoformer,
    'HFFullEnhancedAutoformer': HFFullEnhancedAutoformer,
    # ... other models
}
```

### With configuration files:
```yaml
# Add to config YAML
model: HFBayesianEnhancedAutoformer
use_bayesian: true
uncertainty_method: bayesian
n_samples: 10
loss_function: mse
```

### With loss functions:
```python
# Automatic integration with existing loss infrastructure
from utils.losses import get_loss_function
from utils.bayesian_losses import create_bayesian_loss

# Works with all existing losses
loss_fn = get_loss_function('pinball', config)  # From losses.py
bayesian_loss = create_bayesian_loss(loss_fn, config)  # From bayesian_losses.py
```

## Performance Characteristics

### Memory Usage
- **Standard HF**: ~8.4M parameters, baseline memory
- **+ Bayesian**: +~1-2M parameters (depending on converted layers)  
- **+ Hierarchical**: +~2-4M parameters (depending on hierarchy levels)
- **+ Quantile**: +~0.5M parameters (quantile expansion layer)
- **Full Model**: ~12-15M parameters total

### Computational Overhead
- **Standard HF**: Baseline performance
- **+ Bayesian**: +N×forward_time (N = n_samples)
- **+ Hierarchical**: +~20-30% (wavelet processing)
- **+ Quantile**: +~10% (quantile expansion)
- **Full Model**: Combined overhead, but still efficient

### Training Stability
- **HF Backbone**: Proven stability from Hugging Face
- **External Extensions**: No impact on core stability
- **Loss Integration**: Leverages battle-tested loss functions
- **Gradient Flow**: Clean gradients through modular design

## Testing and Validation

### Run Tests
```bash
# Run comprehensive test suite
python test_hf_enhanced_models.py

# Run usage examples
python examples_hf_enhanced_models.py
```

### Test Coverage
- ✅ Standard HF model functionality
- ✅ Bayesian uncertainty quantification
- ✅ Hierarchical multi-scale processing  
- ✅ Quantile regression capabilities
- ✅ Full model with all features
- ✅ Loss function integration (20+ loss types)
- ✅ Training workflow compatibility
- ✅ Auto model selection

## Troubleshooting

### Common Issues

**Issue**: ImportError for HF models
```python
# Solution: Check imports in exp/exp_basic.py
from models.HFAdvancedFactory import HFBayesianEnhancedAutoformer
```

**Issue**: Dimension mismatch with quantile targets
```python
# Solution: Expand targets for quantile regression
n_quantiles = len(config.quantile_levels)
quantile_targets = base_targets.repeat(1, 1, n_quantiles)
```

**Issue**: Bayesian layer conversion fails
```python
# Solution: Check layer names in config
config.bayesian_layers = ['output_projection']  # Valid layer names only
```

**Issue**: Wavelet processing errors
```python
# Solution: Install PyWavelets and check wavelet type
pip install PyWavelets
config.wavelet_type = 'db4'  # Use supported wavelets
```

### Debug Information
```python
# Get model information
model = create_hf_full_model(config)
info = model.get_model_info()

print(f"Total parameters: {info['parameters']['total']:,}")
print(f"Extensions: {info['extensions']}")
print(f"Capabilities: {info['capabilities']}")

# Check extension-specific info
if 'bayesian_info' in info:
    print(f"Bayesian config: {info['bayesian_info']}")
```

## Future Extensions

The modular design allows easy addition of new features:

1. **New Uncertainty Methods**: Add to BayesianExtension
2. **New Hierarchical Processors**: Add to HierarchicalExtensions  
3. **New Loss Functions**: Leverage existing infrastructure
4. **New HF Backbones**: Swap in different HF models
5. **Ensemble Methods**: Combine multiple HF models

## Conclusion

The HF Enhanced Models successfully solve the challenge of adding sophisticated time series forecasting capabilities to Hugging Face transformers while:

- 🔒 **Preserving HF stability**: Core backbone unchanged
- 🎯 **Leveraging existing infrastructure**: Full integration with losses.py and bayesian_losses.py  
- 🧩 **Modular design**: Mix and match features as needed
- 📈 **Advanced capabilities**: Bayesian uncertainty, hierarchical processing, quantile regression
- 🔄 **Training compatibility**: Works with existing experiment runners and workflows

This approach provides the best of both worlds: proven HF performance with cutting-edge time series forecasting features.
