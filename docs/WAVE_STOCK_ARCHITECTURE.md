# Multi-Modal Wave-Stock Prediction Architecture

## Problem Definition

**Objective**: Predict stock returns for 14 days using 10 Hilbert-transformed wave signals
**Data Structure**:
- **Targets**: Stock returns (1 variable)
- **Covariates**: 10 waves × 4 variables = 40 features
  - Each wave: [r, cos(θ), sin(θ), dθ/dt]
- **Frequency**: Daily
- **Prediction Horizon**: 14 days (multi-step)
- **Task Type**: Classification (directional accuracy)

## Architecture Overview

### Dual-Stream Processing Pipeline

```
┌─────────────────┐    ┌──────────────────────────────┐
│   Stock Returns │    │        Wave Data             │
│   [B, L, 1]     │    │   [B, L, 40] (10×4)         │
└─────────┬───────┘    └──────────────┬───────────────┘
          │                           │
          ▼                           ▼
┌─────────────────┐    ┌──────────────────────────────┐
│ Wavelet Decomp  │    │   Wave Group Processing      │
│ + Embedding     │    │   (Graph + Cross Attention)  │
│ + Hierarchical  │    │   + MoE                      │
│ Encoder         │    │                              │
└─────────┬───────┘    └──────────────┬───────────────┘
          │                           │
          └─────────┬─────────────────┘
                    ▼
          ┌─────────────────┐
          │ Hierarchical    │
          │ Fusion Network  │
          └─────────┬───────┘
                    ▼
          ┌─────────────────┐
          │ Bayesian Decoder│
          │ + Quantile Head │
          └─────────────────┘
```

## Component Specifications

### 1. Target Stream (Stock Returns)

**Decomposition**: `WaveletHierarchicalDecomposition`
- **Levels**: 3 (daily, weekly, monthly patterns)
- **Wavelet Type**: Learnable
- **Purpose**: Capture multi-scale market dynamics

**Embedding**: `PatchEmbedding`
- **Patch Length**: 5 days
- **Stride**: 1 day
- **Purpose**: Local temporal pattern capture

**Encoder**: `HierarchicalEncoder`
- **Attention**: `HierarchicalAutoCorrelation`
- **Layers**: 3
- **Purpose**: Multi-scale temporal modeling

### 2. Covariate Stream (Wave Data)

**Wave Grouping**: Reshape [B, L, 40] → [B, L, 10, 4]

**Intra-Wave Processing**:
- **Component**: `GraphAttentionLayer`
- **Graph Construction**: Correlation-based adjacency for 4 variables per wave
- **Purpose**: Model phase-amplitude relationships within each wave

**Cross-Wave Processing**:
- **Component**: `MultiGraphAttention`
- **Graph Construction**: Wave-to-wave correlation matrix (10×10)
- **Purpose**: Discover inter-wave dependencies

**Expert Processing**:
- **Component**: `GatedMoEFFN`
- **Experts**: 6 (different wave-market regimes)
- **Gating**: Based on wave energy and market volatility
- **Purpose**: Specialized processing for different market conditions

### 3. Fusion Network

**Component**: `HierarchicalFusion`
- **Input Dimensions**: 
  - Target features: [B, L, d_model]
  - Wave features: [B, L, d_model]
- **Fusion Strategy**: Attention-weighted combination
- **Output**: Unified feature representation [B, L, d_model]

### 4. Prediction Head

**Decoder**: `EnhancedDecoder` with Bayesian components
- **Multi-step**: Autoregressive for 14-day horizon
- **Uncertainty**: Variational layers for confidence intervals

**Output Head**: `QuantileOutputHead`
- **Quantiles**: [0.1, 0.25, 0.5, 0.75, 0.9]
- **Classification**: 3 classes (Up, Down, Neutral)
- **Loss**: Combined quantile + cross-entropy

## Implementation Details

### Model Configuration

```python
config = {
    # Data dimensions
    'seq_len': 60,           # 2 months lookback
    'pred_len': 14,          # 2 weeks prediction
    'enc_in': 1,             # Stock returns
    'covariate_in': 40,      # 10 waves × 4 variables
    'c_out': 3,              # 3 classes (Up/Down/Neutral)
    
    # Model architecture
    'd_model': 128,
    'd_ff': 256,
    'n_heads': 8,
    'e_layers': 3,
    'dropout': 0.1,
    
    # Wave-specific
    'n_waves': 10,
    'wave_features': 4,
    'moe_experts': 6,
    
    # Decomposition
    'wavelet_levels': 3,
    'patch_len': 5,
    
    # Training
    'batch_size': 32,
    'learning_rate': 1e-4,
    'epochs': 100
}
```

### Loss Function

```python
total_loss = (
    0.6 * classification_loss +     # Cross-entropy for direction
    0.3 * quantile_loss +          # Uncertainty quantification
    0.1 * moe_aux_loss             # Load balancing
)
```

### Graph Construction Strategy

**Intra-Wave Graphs** (4×4 per wave):
- Correlation between [r, cos(θ), sin(θ), dθ/dt]
- Threshold: 0.3 for edge creation
- Purpose: Phase-amplitude coupling

**Inter-Wave Graphs** (10×10):
- Cross-correlation between wave energies
- Dynamic adjacency based on rolling correlation
- Purpose: Wave interaction patterns

## Expected Performance Benefits

1. **Specialized Processing**: Separate streams optimize for different data types
2. **Relationship Discovery**: Graph attention reveals hidden wave dependencies
3. **Regime Adaptation**: MoE learns different market-wave relationships
4. **Uncertainty Quantification**: Bayesian components provide confidence measures
5. **Multi-Scale Modeling**: Captures both short-term and long-term patterns

## Evaluation Metrics

- **Primary**: Directional Accuracy (Up/Down/Neutral)
- **Secondary**: 
  - Sharpe Ratio of trading strategy
  - Calibration of uncertainty estimates
  - Hit rate at different confidence levels

## Implementation Priority

1. **Phase 1**: Basic dual-stream architecture
2. **Phase 2**: Add graph attention mechanisms
3. **Phase 3**: Integrate MoE and Bayesian components
4. **Phase 4**: Hyperparameter optimization and evaluation