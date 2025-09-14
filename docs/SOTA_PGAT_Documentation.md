# SOTA Temporal PGAT: State-of-the-Art Probabilistic Graph Attention Network

## Overview

The SOTA Temporal PGAT is an advanced time series forecasting model that combines probabilistic graph attention networks with state-of-the-art temporal modeling techniques. This implementation incorporates multiple cutting-edge components to achieve superior performance in uncertainty quantification and temporal pattern recognition.

## Architecture Components

### 1. Mixture Density Network (MDN) Decoder

**Location**: `layers/modular/decoder/mixture_density_decoder.py`

**Purpose**: Replaces single Gaussian probabilistic decoding with a mixture of K Gaussian components for enhanced uncertainty quantification.

**Key Features**:
- Predicts K Gaussian components (means, standard deviations, mixing coefficients)
- Mixture Negative Log-Likelihood loss for training
- Sampling and prediction summary methods
- Configurable number of mixture components

**Usage**:
```python
from layers.modular.decoder.mixture_density_decoder import MixtureDensityDecoder

decoder = MixtureDensityDecoder(
    d_model=512,
    pred_len=96,
    num_components=5  # Number of Gaussian components
)

# Forward pass
mixture_params = decoder(encoded_features)
means, log_stds, log_weights = mixture_params

# Sample from mixture
samples = decoder.sample(mixture_params, num_samples=100)

# Get prediction summary
pred_mean, pred_std = decoder.prediction_summary(mixture_params)
```

### 2. AutoCorrelation Temporal Attention

**Location**: `layers/modular/attention/autocorr_temporal_attention.py`

**Purpose**: Replaces standard multi-head attention with autocorrelation-based attention mechanism for improved temporal pattern recognition.

**Key Features**:
- Autocorrelation computation in frequency domain using FFT
- Time delay aggregation for temporal dependencies
- Multi-scale temporal pattern recognition
- Efficient O(L log L) complexity

**Usage**:
```python
from layers.modular.attention.autocorr_temporal_attention import AutoCorrTemporalAttention

attention = AutoCorrTemporalAttention(
    d_model=512,
    num_heads=8,
    factor=3,  # Top-k autocorrelations to use
    dropout=0.1
)

# Forward pass
output = attention(queries, keys, values, history)
```

### 3. Structural Positional Encoding

**Location**: `layers/modular/embedding/structural_positional_encoding.py`

**Purpose**: Enhances model's understanding of graph structure using Laplacian eigenmaps.

**Key Features**:
- Laplacian eigenmap computation for structural encoding
- Graph-aware positional information
- Configurable number of eigenvectors
- Integration with initial embeddings

**Usage**:
```python
from layers.modular.embedding.structural_positional_encoding import StructuralPositionalEncoding

struct_encoding = StructuralPositionalEncoding(
    d_model=512,
    num_eigenvectors=16,
    max_nodes=1000
)

# Apply structural encoding
encoded_features = struct_encoding(node_features, adjacency_matrix)
```

### 4. Enhanced Temporal Encoding

**Location**: `layers/modular/embedding/enhanced_temporal_encoding.py`

**Purpose**: Provides sophisticated temporal positional encoding combining sinusoidal and adaptive components.

**Key Features**:
- Sinusoidal temporal encoding for periodic patterns
- Adaptive temporal encoding for non-stationary patterns
- Multi-scale temporal representations
- Learnable temporal parameters

**Usage**:
```python
from layers.modular.embedding.enhanced_temporal_encoding import EnhancedTemporalEncoding

temporal_encoding = EnhancedTemporalEncoding(
    d_model=512,
    max_len=1000,
    num_scales=4
)

# Apply temporal encoding
encoded_sequence = temporal_encoding(input_sequence)
```

### 5. Dynamic Edge Weight PGAT Layer

**Location**: `layers/modular/graph/enhanced_pgat_layer.py`

**Purpose**: Implements learnable dynamic edge weights for adaptive graph attention.

**Key Features**:
- Dynamic edge weight computation based on node features
- Multi-head attention with learnable edge importance
- Structural and feature-based weight combination
- Adaptive residual connections

**Usage**:
```python
from layers.modular.graph.enhanced_pgat_layer import EnhancedPGAT_CrossAttn_Layer

pgat_layer = EnhancedPGAT_CrossAttn_Layer(
    d_model=512,
    num_heads=8,
    use_dynamic_weights=True
)

# Forward pass
x_updated, t_updated = pgat_layer(x_dict, t_dict, edge_index_dict)

# Get edge weights for analysis
edge_weights = pgat_layer.get_edge_weights(x_dict, edge_index_dict)
```

## Model Configuration

### Required Configuration Parameters

```python
class SOTAConfig:
    # Model dimensions
    d_model: int = 512
    n_heads: int = 8
    
    # Sequence lengths
    seq_len: int = 96
    pred_len: int = 96
    
    # MDN parameters
    use_mixture_decoder: bool = True
    num_mixture_components: int = 5
    
    # Attention parameters
    use_autocorr_attention: bool = True
    autocorr_factor: int = 3
    
    # Encoding parameters
    use_structural_encoding: bool = True
    num_eigenvectors: int = 16
    use_enhanced_temporal_encoding: bool = True
    
    # Graph parameters
    use_dynamic_edge_weights: bool = True
    
    # Other parameters
    dropout: float = 0.1
    activation: str = 'gelu'
```

### Optional Configuration Parameters

```python
# Advanced MDN settings
config.mdn_hidden_dim = 256
config.mdn_num_layers = 2

# AutoCorrelation settings
config.autocorr_dropout = 0.1
config.autocorr_activation = 'relu'

# Structural encoding settings
config.struct_encoding_dropout = 0.1
config.max_nodes = 1000

# Temporal encoding settings
config.temporal_encoding_scales = 4
config.max_temporal_len = 1000

# Dynamic edge weight settings
config.edge_weight_dropout = 0.1
config.edge_adaptation_factor = 1.0
```

## Training and Usage

### Basic Usage

```python
import torch
from models.SOTA_Temporal_PGAT import SOTA_Temporal_PGAT
from layers.modular.decoder.mixture_density_decoder import MixtureNLLLoss

# Initialize model
model = SOTA_Temporal_PGAT(config)

# Initialize loss function
criterion = MixtureNLLLoss()

# Training loop
for batch in dataloader:
    # Forward pass
    mixture_params = model(batch)
    
    # Compute loss
    loss = criterion(mixture_params, batch['target'])
    
    # Backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Advanced Usage with Uncertainty Quantification

```python
# Prediction with uncertainty
with torch.no_grad():
    mixture_params = model(test_batch)
    
    # Get prediction summary
    pred_mean, pred_std = model.decoder.prediction_summary(mixture_params)
    
    # Sample multiple predictions
    samples = model.decoder.sample(mixture_params, num_samples=100)
    
    # Compute prediction intervals
    lower_bound = torch.quantile(samples, 0.025, dim=0)
    upper_bound = torch.quantile(samples, 0.975, dim=0)
```

### Graph Analysis

```python
# Analyze dynamic edge weights
with torch.no_grad():
    # Forward pass to compute edge weights
    _ = model(batch)
    
    # Get edge weights from spatial encoder
    if hasattr(model.spatial_encoder, 'get_edge_weights'):
        edge_weights = model.spatial_encoder.get_edge_weights(
            model.last_x_dict, model.last_edge_index_dict
        )
        
        # Analyze edge importance
        for edge_type, weights in edge_weights.items():
            print(f"Edge type {edge_type}: mean weight = {weights.mean():.4f}")
```

## Performance Considerations

### Memory Usage
- MDN decoder increases memory usage by factor of K (number of components)
- AutoCorrelation attention has O(L log L) complexity vs O(L²) for standard attention
- Structural encoding requires eigendecomposition (computed once per graph)
- Dynamic edge weights add computational overhead but improve model expressiveness

### Computational Complexity
- **Standard Attention**: O(L² × d)
- **AutoCorrelation Attention**: O(L log L × d)
- **MDN Decoder**: O(K × d × pred_len)
- **Structural Encoding**: O(N³) for eigendecomposition (one-time cost)
- **Dynamic Edge Weights**: O(E × d × H) where E is number of edges, H is number of heads

### Optimization Tips

1. **Batch Size**: Use larger batch sizes to amortize graph computation costs
2. **Mixed Precision**: Enable mixed precision training to reduce memory usage
3. **Gradient Checkpointing**: Use for very deep models to trade compute for memory
4. **Component Selection**: Disable unused components via configuration flags

```python
# Memory-efficient configuration
config.use_mixture_decoder = True
config.num_mixture_components = 3  # Reduce from 5
config.use_autocorr_attention = True  # More efficient than standard attention
config.use_structural_encoding = False  # Disable if graph structure is simple
config.use_dynamic_edge_weights = False  # Disable for faster training
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Reduce number of mixture components
   - Disable structural encoding for large graphs
   - Use gradient checkpointing

2. **NaN Loss Values**
   - Check for numerical instability in mixture weights
   - Ensure proper initialization of edge weight parameters
   - Verify input data normalization

3. **Slow Training**
   - Disable dynamic edge weights during initial training
   - Use smaller number of eigenvectors for structural encoding
   - Reduce autocorrelation factor

4. **Poor Convergence**
   - Ensure proper learning rate scheduling
   - Check gradient clipping settings
   - Verify loss function implementation

### Debugging Tools

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Monitor gradient norms
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm()
        print(f"{name}: grad_norm = {grad_norm:.6f}")

# Check mixture parameters
mixture_params = model(batch)
means, log_stds, log_weights = mixture_params
print(f"Means range: [{means.min():.4f}, {means.max():.4f}]")
print(f"Log stds range: [{log_stds.min():.4f}, {log_stds.max():.4f}]")
print(f"Log weights range: [{log_weights.min():.4f}, {log_weights.max():.4f}]")
```

## References

1. **Mixture Density Networks**: Bishop, C. M. (1994). Mixture density networks.
2. **AutoCorrelation Attention**: Wu, H., et al. (2021). Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting.
3. **Graph Attention Networks**: Veličković, P., et al. (2017). Graph attention networks.
4. **Laplacian Eigenmaps**: Belkin, M., & Niyogi, P. (2003). Laplacian eigenmaps for dimensionality reduction and data representation.
5. **Probabilistic Time Series Forecasting**: Salinas, D., et al. (2020). DeepAR: Probabilistic forecasting with autoregressive recurrent neural networks.

## License

This implementation is part of the Time-Series-Library project. Please refer to the main project license for usage terms.