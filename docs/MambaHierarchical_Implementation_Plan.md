# MambaHierarchical Model Implementation Plan

## Architecture Overview

### Target Variables Path:
1. Target variables → **Wavelet Decomposition** 
2. → **Mamba Block** 
3. → **Multi-Head Attention** 
4. → **Single Context Vector (Targets)**

### Covariates Path (Hilbert-transformed data):
1. 40 covariates (grouped into 10 families of 4)
2. Each family → **Separate Mamba Block** (10 Mamba blocks)
3. 40 context-aware vectors → **Hierarchical Attention Process**
4. → **Single Context Vector (Covariates)**

### Fusion Block:
1. **Dual Cross-Attention** (Target context ↔ Covariate context)
2. → **Mixture of Experts (MoE)** (optional)
3. → **Output: num_targets**

## Implementation Components

### Components to Import/Use:
- `layers.modular.decomposition.wavelet_decomposition` - Wavelet decomposition
- `layers.modular.attention.enhanced_autocorrelation` - Multi-head attention
- `layers.modular.fusion.hierarchical_fusion` - For hierarchical attention
- `layers.GatedMoEFFN` - Mixture of Experts
- `layers.Embed.DataEmbedding_wo_pos` - Input embedding
- `mamba_ssm.Mamba` - Mamba blocks

### Components to Implement:
1. `MambaBlock` - Wrapper for Mamba with proper input/output handling
2. `DualCrossAttention` - Cross attention between target and covariate contexts
3. `CovariateHierarchicalProcessor` - Handles 10 families of 4 covariates each
4. `TargetProcessor` - Handles wavelet → mamba → attention for targets
5. `MambaHierarchical` - Main model class

## Implementation Status

### ✅ Completed:
- [x] Implementation plan document created
- [x] MambaBlock wrapper (`layers/MambaBlock.py`)
  - Base MambaBlock with input/output handling
  - TargetMambaBlock with trend decomposition
  - CovariateMambaBlock with family attention
- [x] TargetProcessor (`layers/TargetProcessor.py`)
  - Wavelet decomposition integration
  - Mamba processing for each decomposition level
  - Multi-head attention for component fusion
- [x] CovariateProcessor (`layers/CovariateProcessor.py`)
  - **UPDATED**: Flexible family-based covariate grouping (no hardcoded families)
  - Individual Mamba blocks per family (variable sizes supported)
  - Hierarchical attention fusion
  - **NEW**: Custom family configuration support
- [x] DualCrossAttention (`layers/DualCrossAttention.py`)
  - Bidirectional cross-attention
  - Gated fusion mechanism
  - Simplified version for fallback
- [x] Main model structure (`models/MambaHierarchical.py`)
  - Complete integration of all components
  - Configuration parameter handling
  - Error handling and fallbacks
  - **NEW**: Comprehensive logging dumps throughout pipeline
  - **NEW**: Flexible covariate family configuration

### 🔄 In Progress:
- [ ] Testing and validation

### 🚀 Major Improvements Added:
- [x] **EnhancedTargetProcessor** (`layers/EnhancedTargetProcessor.py`)
  - Explicit trend-seasonal decomposition using learnable components
  - Separate Mamba processing for trend and seasonal components
  - Cross-attention between trend and seasonal contexts
  - Proper wavelet decomposition integration
- [x] **SequentialDecoder** (`layers/SequentialDecoder.py`)
  - Autoregressive sequential generation
  - Separate trend and seasonal prediction heads
  - Proper temporal dynamics modeling
  - Support for future covariate integration
- [x] **ImprovedMambaHierarchical** (`models/ImprovedMambaHierarchical.py`)
  - Integration of all improvements
  - Proper decoder input utilization
  - Enhanced context fusion with MoE
  - Comprehensive decomposition analysis methods

### ⏳ Pending:
- [ ] Configuration examples for improved model
- [ ] Training scripts with trend-seasonal loss
- [ ] Performance benchmarking
- [ ] Ablation studies

## File Structure:
```
models/
└── MambaHierarchical.py          # Main model only

layers/
├── MambaBlock.py                 # Mamba wrapper component
├── TargetProcessor.py            # Target processing pipeline
├── CovariateProcessor.py         # Covariate processing pipeline
└── DualCrossAttention.py         # Cross attention fusion
```

## Configuration Parameters:
```python
configs = {
    'num_targets': 4,              # Number of target variables
    'num_covariates': 40,          # Total covariates
    'covariate_family_size': 4,    # Covariates per family
    'wavelet_type': 'db4',         # Wavelet type for decomposition
    'wavelet_levels': 3,           # Decomposition levels
    'use_moe': True,               # Enable Mixture of Experts
    'num_experts': 8,              # Number of MoE experts
    'd_model': 128,                # Model dimension
    'mamba_d_state': 64,           # Mamba state dimension
    'mamba_d_conv': 4,             # Mamba convolution dimension
    'mamba_expand': 2,             # Mamba expansion factor
}
```

## Implementation Details:

### Component Descriptions:

#### 1. MambaBlock (`layers/MambaBlock.py`)
- **Base MambaBlock**: Wrapper around mamba_ssm.Mamba with proper I/O handling
- **TargetMambaBlock**: Specialized for targets with optional trend decomposition
- **CovariateMambaBlock**: Optimized for covariate families with family attention
- **Features**: Input normalization, residual connections, configurable projections

#### 2. TargetProcessor (`layers/TargetProcessor.py`)
- **Pipeline**: Wavelet Decomposition → Mamba Blocks → Multi-Head Attention
- **Wavelet Integration**: Uses modular wavelet decomposition with fallback
- **Multi-level Processing**: Separate Mamba block for each decomposition level
- **Context Generation**: Aggregates multi-level outputs into single context vector

#### 3. CovariateProcessor (`layers/CovariateProcessor.py`)
- **Family Grouping**: Splits 40 covariates into 10 families of 4
- **Parallel Processing**: Individual Mamba block per family
- **Hierarchical Fusion**: Uses modular hierarchical fusion with fallback
- **Future Support**: Handles future covariate data (Hilbert-transformed)

#### 4. DualCrossAttention (`layers/DualCrossAttention.py`)
- **Bidirectional**: Target↔Covariate cross-attention
- **Gated Fusion**: Learnable fusion of attended contexts
- **Residual & Norm**: Layer normalization and residual connections
- **Fallback**: SimpleDualCrossAttention for reduced complexity

#### 5. MambaHierarchical (`models/MambaHierarchical.py`)
- **Main Architecture**: Integrates all components
- **Input Handling**: Splits features into targets/covariates
- **MoE Integration**: Optional Mixture of Experts
- **Error Resilience**: Comprehensive fallback mechanisms

### Key Features:
- **Modular Design**: Each component is independent and reusable
- **Error Handling**: Graceful degradation with fallback mechanisms
- **Logging**: Comprehensive debug logging throughout
- **Configuration**: Extensive configuration options with sensible defaults
- **Compatibility**: Works with existing Time-Series-Library infrastructure

## Implementation Notes:
- Keep all components modular and reusable
- Add proper error handling and input validation
- Include comprehensive logging for debugging
- Support both training and inference modes
- Handle variable sequence lengths gracefully