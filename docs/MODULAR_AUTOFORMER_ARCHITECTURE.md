# Modular Autoformer Architecture - Complete Framework Documentation

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [HF Integration Framework](#hf-integration-framework)
4. [Component Registry System](#component-registry-system)
5. [Framework Interactions](#framework-interactions)
6. [Custom vs HF Models](#custom-vs-hf-models)
7. [Adding New Models](#adding-new-models)
8. [Adding New Components](#adding-new-components)
9. [Testing and Validation](#testing-and-validation)
10. [Best Practices](#best-practices)

## Architecture Overview

The Modular Autoformer Architecture implements a unified framework that supports both custom GCLI-based modular autoformers and HuggingFace (HF) autoformer models through a single interface. The architecture follows the GCLI (General Component Library Interface) recommendations for modular, extensible, and maintainable code.

### Key Design Principles

1. **Modular Design**: Each component (attention, decomposition, encoder, etc.) is a self-contained, replaceable module
2. **Type Safety**: Pydantic schemas ensure configuration validation and type safety
3. **Unified Interface**: Single factory pattern supports both custom and HF models
4. **Component Registry**: Centralized registry manages all available components
5. **"Dumb Assembler"**: Simple assembly logic without complex business rules
6. **Separation of Concerns**: Clear boundaries between components and configurations

### Architecture Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                            │
│  ┌─────────────────────┐  ┌─────────────────────────────────┐   │
│  │   Custom Models     │  │      HF Models                 │   │
│  │  ModularAutoformer  │  │  HFEnhancedAutoformer          │   │
│  └─────────────────────┘  └─────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                    Unified Factory Layer                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │            UnifiedAutoformerFactory                     │   │
│  │  - Framework Detection    - Config Completion           │   │
│  │  - Model Creation        - Interface Wrapping          │   │
│  └─────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                    Component Layer                             │
│  ┌───────────┐ ┌────────────┐ ┌─────────┐ ┌──────────────┐    │
│  │Attention  │ │Decomp.     │ │Encoder  │ │Decoder       │    │
│  │Components │ │Components  │ │Components│ │Components    │    │
│  └───────────┘ └────────────┘ └─────────┘ └──────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│                    Registry & Schema Layer                     │
│  ┌─────────────────────┐  ┌─────────────────────────────────┐   │
│  │  Component Registry │  │     Pydantic Schemas           │   │
│  │  - Type Management  │  │  - Configuration Validation    │   │
│  │  - Metadata Storage │  │  - Type Safety                 │   │
│  └─────────────────────┘  └─────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

The framework consists of **7 component types**, each serving a specific purpose in the autoformer architecture. **All components are fully modular and interchangeable**, allowing for flexible architecture composition through the component registry system.

### Component Modularity Principles

1. **Interface Standardization**: All components inherit from `ModularComponent` with standardized initialization and forward methods
2. **Registry Management**: Components are registered with metadata for dynamic discovery and validation
3. **Configuration Validation**: Pydantic schemas ensure type safety and parameter validation
4. **Dependency Injection**: Components receive dependencies through the assembler pattern
5. **Runtime Swapping**: Components can be swapped at runtime through configuration changes

### 1. Attention Components

**Purpose**: Handle attention mechanisms for temporal dependencies

**Available Types** (11 total):
- `AUTOCORRELATION`: Standard autocorrelation attention for temporal dependencies
- `ADAPTIVE_AUTOCORRELATION`: Enhanced autocorrelation with adaptive window selection and multi-scale analysis
- `FOURIER_ATTENTION`: Fourier-based attention for capturing periodic patterns in frequency domain
- `FOURIER_BLOCK`: Fourier block for frequency domain representation learning with learnable modes
- `CROSS_RESOLUTION`: Multi-resolution attention for hierarchical temporal processing
- `MULTI_HEAD`: Traditional multi-head attention mechanism
- `SPARSE`: Sparse attention for long sequences
- `LOG_SPARSE`: Logarithmic sparse attention pattern
- `PROB_SPARSE`: Probabilistic sparse attention selection

**Implementation**: `configs/concrete_components.py`

```python
class AutoCorrelationAttention(AttentionComponent):
    def __init__(self, config: AttentionConfig, **kwargs):
        super().__init__(config, **kwargs)
        # Component initialization
        
    def forward(self, queries, keys, values, attn_mask=None):
        # Attention computation
        return output, attention_weights
```

### 2. Decomposition Components

**Purpose**: Separate time series into trend and seasonal components

**Available Types** (4 total):
- `MOVING_AVG`: Moving average decomposition for trend extraction
- `LEARNABLE_DECOMP`: Learnable decomposition with trainable trend/seasonal separation
- `WAVELET_DECOMP`: Wavelet-based hierarchical decomposition for multi-scale analysis
- `ADVANCED_WAVELET`: Advanced learnable wavelet decomposition with multi-level filtering and reconstruction weights

**Implementation**:
```python
class LearnableDecomposition(DecompositionComponent):
    def forward(self, x):
        # x: [B, L, D]
        seasonal, trend = self.decompose(x)
        return seasonal, trend
```

### 3. Encoder Components

**Purpose**: Encode input sequences with attention and decomposition

**Available Types** (5 total):
- `STANDARD_ENCODER`: Basic transformer encoder with attention and feedforward layers
- `ENHANCED_ENCODER`: Enhanced encoder with adaptive features and improved normalization
- `HIERARCHICAL_ENCODER`: Multi-level hierarchical encoder for multi-scale processing
- `TEMPORAL_CONV_ENCODER`: Temporal Convolutional Network encoder for causal sequence modeling
- `META_LEARNING_ADAPTER`: Meta-learning adapter for quick adaptation to new time series patterns

### 4. Decoder Components

**Purpose**: Decode and generate predictions

**Available Types** (3 total):
- `STANDARD_DECODER`: Basic transformer decoder with cross-attention capabilities
- `ENHANCED_DECODER`: Enhanced decoder with adaptive features and improved prediction
- `HIERARCHICAL_DECODER`: Multi-level hierarchical decoder (planned for future implementation)

### 5. Sampling Components

**Purpose**: Handle uncertainty quantification and sampling strategies

**Available Types** (4 total):
- `DETERMINISTIC`: Standard deterministic prediction without uncertainty
- `BAYESIAN`: Bayesian sampling for uncertainty quantification with dropout
- `MONTE_CARLO`: Monte Carlo sampling methods for robust uncertainty estimation
- `ADAPTIVE_MIXTURE`: Adaptive mixture of experts for different time series patterns with gating networks

### 6. Output Head Components

**Purpose**: Final projection to output dimensions

**Available Types** (3 total):
- `STANDARD_HEAD`: Basic linear projection to output dimensions
- `QUANTILE`: Multi-quantile prediction head for uncertainty bounds and confidence intervals
- `BAYESIAN_HEAD`: Bayesian linear head with weight uncertainty quantification for robust predictions

### 7. Loss Components

**Purpose**: Loss function computation and optimization

**Available Types** (8 total):
- `MSE`: Mean squared error for standard regression
- `MAE`: Mean absolute error for robust regression  
- `QUANTILE_LOSS`: Quantile loss for single quantile prediction
- `BAYESIAN_MSE`: Bayesian MSE with KL divergence regularization
- `BAYESIAN_QUANTILE`: Bayesian quantile loss for uncertainty quantification
- `FOCAL_LOSS`: Focal loss for handling imbalanced time series data
- `ADAPTIVE_AUTOFORMER_LOSS`: Adaptive loss with learnable trend/seasonal component weighting
- `ADAPTIVE_LOSS_WEIGHTING`: Multi-task adaptive loss weighting for complex optimization

**Implementation**: Each loss component handles different prediction scenarios:

```python
class MSELoss(LossComponent):
    """Standard MSE loss for deterministic predictions"""
    def forward(self, predictions, targets, **kwargs):
        return nn.MSELoss()(predictions, targets)

class BayesianQuantileLoss(LossComponent):
    """Bayesian quantile loss with KL divergence"""
    def forward(self, predictions, targets, **kwargs):
        # Quantile loss + KL divergence from Bayesian layers
        quantile_loss = self._compute_quantile_loss(predictions, targets)
        kl_loss = self._compute_kl_divergence()
        return quantile_loss + self.kl_weight * kl_loss
```

## HF Integration Framework

The framework seamlessly integrates HuggingFace autoformer models through a unified interface that abstracts the differences between custom and HF implementations.

### HF Model Types

**Location**: `models/HFAutoformerSuite.py`, `models/unified_autoformer_factory.py`

1. **HFEnhancedAutoformer**: Basic HF enhanced autoformer with improved attention mechanisms
2. **HFBayesianAutoformer**: HF Bayesian autoformer with uncertainty quantification capabilities
3. **HFHierarchicalAutoformer**: HF hierarchical autoformer for multi-scale temporal modeling
4. **HFQuantileAutoformer**: HF quantile autoformer for probabilistic forecasting
5. **HFEnhancedAutoformerAdvanced**: Advanced HF enhanced model with additional optimizations
6. **HFBayesianAutoformerProduction**: Production-ready HF Bayesian model with robust uncertainty

## Complete Component File Examples

### Custom Component Implementation (`configs/concrete_components.py`)

```python
# Example: Complete Attention Component
class AutoCorrelationAttention(AttentionComponent):
    """AutoCorrelation attention component with GCLI compliance"""
    
    def __init__(self, config: AttentionConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="AutoCorrelation",
            component_type=ComponentType.AUTOCORRELATION,
            required_params=['d_model', 'n_heads'],
            optional_params=['dropout', 'factor'],
            description="AutoCorrelation mechanism for time series modeling"
        )
    
    def _initialize_component(self, **kwargs):
        """Initialize component with configuration validation"""
        self.d_model = self.config.d_model
        self.n_heads = self.config.n_heads
        self.dropout = nn.Dropout(self.config.dropout)
        self.factor = self.config.factor
        
        # Core projection layers
        self.query_projection = nn.Linear(self.d_model, self.d_model)
        self.key_projection = nn.Linear(self.d_model, self.d_model)
        self.value_projection = nn.Linear(self.d_model, self.d_model)
        self.out_projection = nn.Linear(self.d_model, self.d_model)
    
    def forward(self, queries, keys, values, attn_mask=None):
        """Modular forward pass with standardized interface"""
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        # Project and reshape
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        
        # AutoCorrelation mechanism
        scale = 1. / math.sqrt(queries.shape[-1])
        scores = torch.einsum("blhd,bshd->bhls", queries, keys) * scale
        
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.einsum("bhls,bshd->blhd", attn, values)
        out = out.contiguous().view(B, L, -1)
        
        return self.out_projection(out), attn

# Example: Complete Loss Component
class BayesianQuantileLoss(LossComponent):
    """Bayesian quantile loss with KL divergence regularization"""
    
    def __init__(self, config: LossConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="BayesianQuantileLoss",
            component_type=ComponentType.BAYESIAN_QUANTILE,
            required_params=['quantiles', 'prior_scale', 'kl_weight'],
            optional_params=['reduction'],
            description="Bayesian quantile loss for uncertainty quantification"
        )
    
    def _initialize_component(self, **kwargs):
        """Initialize with quantile and Bayesian parameters"""
        self.quantiles = torch.tensor(self.config.quantiles, dtype=torch.float32)
        self.prior_scale = self.config.prior_scale
        self.kl_weight = self.config.kl_weight
        self.reduction = getattr(self.config, 'reduction', 'mean')
        
    def forward(self, predictions, targets, model=None, **kwargs):
        """Compute combined quantile and KL divergence loss"""
        # Predictions shape: [B, T, C*Q] where Q is number of quantiles
        batch_size, seq_len, combined_dim = predictions.shape
        num_quantiles = len(self.quantiles)
        num_features = combined_dim // num_quantiles
        
        # Reshape predictions: [B, T, C, Q]
        pred_quantiles = predictions.view(batch_size, seq_len, num_features, num_quantiles)
        
        # Compute quantile loss
        quantile_loss = 0.0
        for i, tau in enumerate(self.quantiles):
            pred_q = pred_quantiles[:, :, :, i]  # [B, T, C]
            residual = targets - pred_q
            loss_q = torch.where(residual >= 0, 
                               tau * residual, 
                               (tau - 1) * residual)
            quantile_loss += loss_q.mean()
        
        # Add KL divergence if model has Bayesian layers
        kl_loss = 0.0
        if model is not None and hasattr(model, 'get_kl_divergence'):
            kl_loss = model.get_kl_divergence()
        
        total_loss = quantile_loss + self.kl_weight * kl_loss
        
        return total_loss if self.reduction == 'mean' else total_loss.sum()
```

### Complete Model Implementation (`models/modular_autoformer.py`)

```python
class ModularAutoformer(BaseTimeSeriesForecaster):
    """
    Complete GCLI-compliant Modular Autoformer implementation
    
    This model uses the "dumb assembler" pattern with modular components
    managed through the component registry system.
    """
    
    def __init__(self, configs):
        super().__init__(configs)
        self.framework_type = 'custom'
        
        # Initialize GCLI assembler
        self.assembler = ModularAssembler()
        
        # Store configuration
        self.configs = configs
        self.task_name = configs.task_name
        
        # Component assembly through registry
        self.components = self._assemble_components()
        
        # Extract assembled components
        self.enc_embedding = self.components['enc_embedding']
        self.dec_embedding = self.components['dec_embedding'] 
        self.encoder = self.components['encoder']
        self.decoder = self.components['decoder']
        self.output_head = self.components['output_head']
        self.loss_fn = self.components['loss']
        
        # Optional components
        self.sampling_component = self.components.get('sampling')
        
        # Model metadata
        self.model_type = getattr(configs, 'model_variant', 'modular_autoformer')
        
    def _assemble_components(self):
        """Assemble all components using GCLI assembler pattern"""
        try:
            return self.assembler.assemble(self.configs)
        except Exception as e:
            logger.error(f"Component assembly failed: {e}")
            raise RuntimeError(f"Failed to assemble modular autoformer: {e}")
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                mask=None, **kwargs):
        """
        Modular forward pass using assembled components
        
        Args:
            x_enc: Encoder input [B, L, D]
            x_mark_enc: Encoder time features [B, L, F] 
            x_dec: Decoder input [B, T, D]
            x_mark_dec: Decoder time features [B, T, F]
            mask: Optional attention mask
            
        Returns:
            Predictions tensor [B, T, C] or [B, T, C*Q] for quantile models
        """
        # Embedding layers
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        
        # Encoder processing
        enc_out, enc_attns = self.encoder(enc_out, attn_mask=mask)
        
        # Decoder processing with cross-attention
        dec_out, trend = self.decoder(dec_out, enc_out, 
                                    x_mask=mask, cross_mask=None)
        
        # Output head projection
        predictions = self.output_head(dec_out)
        
        # Handle different prediction modes
        if self.task_name == 'long_term_forecast':
            # Return predictions in format [B, pred_len, c_out]
            return predictions[:, -self.configs.pred_len:, :]
        else:
            return predictions
    
    def predict_with_uncertainty(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                               **kwargs):
        """Unified uncertainty prediction interface"""
        if self.sampling_component and hasattr(self.sampling_component, 'forward'):
            # Use Bayesian sampling for uncertainty
            def model_fn():
                return self.forward(x_enc, x_mark_enc, x_dec, x_mark_dec, **kwargs)
            
            return self.sampling_component.forward(model_fn, **kwargs)
        else:
            # Deterministic prediction
            prediction = self.forward(x_enc, x_mark_enc, x_dec, x_mark_dec, **kwargs)
            return {
                'prediction': prediction,
                'uncertainty': None
            }
    
    def supports_uncertainty(self):
        """Check if model supports uncertainty quantification"""
        return (self.sampling_component is not None and 
                self.sampling_component.metadata.component_type in 
                [ComponentType.BAYESIAN, ComponentType.MONTE_CARLO])
    
    def get_component_info(self):
        """Get information about assembled components"""
        return {
            component_name: {
                'type': component.metadata.component_type.value,
                'name': component.metadata.name,
                'description': component.metadata.description
            }
            for component_name, component in self.components.items()
            if hasattr(component, 'metadata')
        }
```

### HF Model Integration (`models/HFAutoformerSuite.py`)

```python
class HFEnhancedAutoformer(nn.Module, HFFrameworkMixin):
    """
    HuggingFace Enhanced Autoformer with unified interface
    
    Provides drop-in replacement for custom ModularAutoformer with
    production-ready HF optimizations and consistent APIs.
    """
    
    def __init__(self, configs):
        super().__init__()
        self.framework_type = 'hf'
        self.model_type = 'hf_enhanced'
        self.configs = configs
        
        # Initialize HF-specific components
        self._initialize_hf_components()
        
        # Uncertainty support
        self.supports_bayesian = False
        self.uncertainty_method = 'none'
        
    def _initialize_hf_components(self):
        """Initialize HF-optimized autoformer components"""
        from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer
        from layers.SelfAttention_Family import AutoCorrelationLayer
        from layers.Embed import DataEmbedding
        
        # Embedding layers
        self.enc_embedding = DataEmbedding(
            self.configs.enc_in, self.configs.d_model, 
            self.configs.embed, self.configs.freq, self.configs.dropout
        )
        self.dec_embedding = DataEmbedding(
            self.configs.dec_in, self.configs.d_model,
            self.configs.embed, self.configs.freq, self.configs.dropout
        )
        
        # HF-optimized encoder/decoder
        self.encoder = self._build_hf_encoder()
        self.decoder = self._build_hf_decoder()
        
        # Output projection
        self.projection = nn.Linear(self.configs.d_model, self.configs.c_out, bias=True)
        
    def _build_hf_encoder(self):
        """Build HF-optimized encoder"""
        return Encoder([
            EncoderLayer(
                AutoCorrelationLayer(
                    correlation=self._build_autocorr(),
                    d_model=self.configs.d_model,
                    n_heads=self.configs.n_heads
                ),
                self.configs.d_model,
                self.configs.d_ff,
                dropout=self.configs.dropout,
                activation=self.configs.activation
            ) for _ in range(self.configs.e_layers)
        ], norm_layer=torch.nn.LayerNorm(self.configs.d_model))
    
    def _build_hf_decoder(self):
        """Build HF-optimized decoder"""
        return Decoder([
            DecoderLayer(
                self_attention=AutoCorrelationLayer(
                    correlation=self._build_autocorr(),
                    d_model=self.configs.d_model,
                    n_heads=self.configs.n_heads
                ),
                cross_attention=AutoCorrelationLayer(
                    correlation=self._build_autocorr(),
                    d_model=self.configs.d_model,
                    n_heads=self.configs.n_heads
                ),
                d_model=self.configs.d_model,
                d_ff=self.configs.d_ff,
                dropout=self.configs.dropout,
                activation=self.configs.activation,
            ) for _ in range(self.configs.d_layers)
        ], norm_layer=torch.nn.LayerNorm(self.configs.d_model),
           projection=nn.Linear(self.configs.d_model, self.configs.c_out, bias=True))
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """HF autoformer forward pass with unified interface"""
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        
        # Encoding
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        
        # Decoding with trend decomposition
        dec_out = self.decoder(dec_out, enc_out, 
                             x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                             trend=None)
        
        return dec_out  # [B, L, D]
    
    def supports_uncertainty(self):
        """HF model uncertainty support"""
        return self.supports_bayesian
    
    def get_model_info(self):
        """Get HF model information"""
        return {
            'framework_type': self.framework_type,
            'model_type': self.model_type,
            'hf_optimized': True,
            'supports_uncertainty': self.supports_uncertainty(),
            'uncertainty_method': self.uncertainty_method,
            'component_count': 'HF-integrated'
        }
```

### HF Integration Architecture

```python
class UnifiedAutoformerFactory:
    HF_MODELS = {
        'hf_enhanced': HFEnhancedAutoformer,
        'hf_bayesian': HFBayesianAutoformer,
        'hf_hierarchical': HFHierarchicalAutoformer,
        'hf_quantile': HFQuantileAutoformer,
        'hf_enhanced_advanced': HFEnhancedAutoformerAdvanced,
        'hf_bayesian_production': HFBayesianAutoformerProduction
    }
    
    @classmethod
    def create_model(cls, model_type: str, config: Union[Namespace, Dict],
                    framework_preference: str = 'auto'):
        # Unified model creation logic
```

### Config Completion for HF Models

HF models require specific configuration parameters that may not be present in basic configs:

```python
def _ensure_hf_config_completeness(cls, config: Namespace):
    hf_defaults = {
        'embed': 'timeF',
        'freq': 'h',
        'dropout': 0.1,
        'activation': 'gelu',
        'factor': 1,
        'output_attention': False,
        'use_amp': False,
        'task_name': 'long_term_forecast',
        # ... additional defaults
    }
    # Add missing parameters
```

## Component Registry System

The component registry manages all available components and their metadata, enabling dynamic component discovery and validation.

### Complete Component Inventory

The framework currently supports **38 distinct component implementations** across 7 component types:

#### Attention Components (11 types)
- `AUTOCORRELATION`: Standard autocorrelation attention
- `ADAPTIVE_AUTOCORRELATION`: Enhanced autocorrelation with adaptive features  
- `FOURIER_ATTENTION`: Fourier-based attention for periodic patterns
- `FOURIER_BLOCK`: Fourier block for frequency domain representation learning
- `CROSS_RESOLUTION`: Multi-resolution attention for hierarchical processing
- `MULTI_HEAD`: Traditional multi-head attention
- `SPARSE`: Sparse attention for long sequences
- `LOG_SPARSE`: Logarithmic sparse attention pattern
- `PROB_SPARSE`: Probabilistic sparse attention selection

#### Decomposition Components (4 types)
- `MOVING_AVG`: Moving average decomposition
- `LEARNABLE_DECOMP`: Learnable decomposition with trainable parameters
- `WAVELET_DECOMP`: Wavelet-based hierarchical decomposition
- `ADVANCED_WAVELET`: Advanced learnable wavelet decomposition with multi-level filtering

#### Encoder Components (5 types)  
- `STANDARD_ENCODER`: Basic transformer encoder
- `ENHANCED_ENCODER`: Enhanced encoder with adaptive features
- `HIERARCHICAL_ENCODER`: Multi-level hierarchical encoder
- `TEMPORAL_CONV_ENCODER`: Temporal Convolutional Network encoder for causal modeling
- `META_LEARNING_ADAPTER`: Meta-learning adapter for quick pattern adaptation

#### Decoder Components (3 types)
- `STANDARD_DECODER`: Basic transformer decoder
- `ENHANCED_DECODER`: Enhanced decoder with adaptive features  
- `HIERARCHICAL_DECODER`: Multi-level hierarchical decoder (planned)

#### Sampling Components (4 types)
- `DETERMINISTIC`: Standard deterministic prediction
- `BAYESIAN`: Bayesian sampling for uncertainty quantification
- `MONTE_CARLO`: Monte Carlo sampling methods
- `ADAPTIVE_MIXTURE`: Adaptive mixture of experts with gating networks

#### Output Head Components (3 types)
- `STANDARD_HEAD`: Basic linear projection
- `QUANTILE`: Multi-quantile prediction head
- `BAYESIAN_HEAD`: Bayesian linear head with weight uncertainty

#### Loss Components (8 types)
- `MSE`: Mean squared error for standard regression
- `MAE`: Mean absolute error for robust regression
- `QUANTILE_LOSS`: Quantile loss for probabilistic forecasting
- `BAYESIAN_MSE`: Bayesian MSE with KL divergence regularization
- `BAYESIAN_QUANTILE`: Bayesian quantile loss for uncertainty quantification
- `FOCAL_LOSS`: Focal loss for handling imbalanced data
- `ADAPTIVE_AUTOFORMER_LOSS`: Adaptive loss with trend/seasonal weighting
- `ADAPTIVE_LOSS_WEIGHTING`: Multi-task adaptive loss weighting

### Registry Structure

**Location**: `configs/modular_components.py`

```python
class ComponentRegistry:
    def __init__(self):
        self._components: Dict[ComponentType, Type[ModularComponent]] = {}
        self._metadata: Dict[ComponentType, ComponentMetadata] = {}
    
    def register_component(self, component_type: ComponentType, 
                          component_class: Type[ModularComponent],
                          metadata: ComponentMetadata):
        # Registration logic
        
    def get_component(self, component_type: ComponentType):
        # Component retrieval
```

### Component Metadata

Each component includes rich metadata for validation and discovery:

```python
@dataclass
class ComponentMetadata:
    name: str
    component_type: ComponentType
    required_params: List[str]
    optional_params: List[str] = field(default_factory=list)
    description: str = ""
    version: str = "1.0.0"
    compatibility: List[ComponentType] = field(default_factory=list)
```

## Framework Interactions

### 1. Model Creation Flow

```
User Request
     ↓
UnifiedAutoformerFactory.create_model()
     ↓
Framework Detection (Custom vs HF)
     ↓
Config Validation & Completion
     ↓
Model Creation
     ↓ (if Custom)
ModularAssembler.assemble()
     ↓
Component Registry Lookup
     ↓
Component Instantiation
     ↓
Model Assembly
     ↓
UnifiedModelInterface Wrapping
```

### 2. Component Assembly Process

For custom models, the `ModularAssembler` handles component assembly:

```python
class ModularAssembler:
    def assemble(self, config: AutoformerConfig) -> ModularAutoformer:
        # 1. Create components from configuration
        attention = self._create_attention_component(config.attention)
        decomp = self._create_decomposition_component(config.decomposition)
        encoder = self._create_encoder_component(config.encoder, attention, decomp)
        decoder = self._create_decoder_component(config.decoder, attention, decomp)
        
        # 2. Assemble complete model
        return ModularAutoformer(config, encoder, decoder, ...)
```

### 3. Configuration Flow

```
User Config (Dict/Namespace)
     ↓
Pydantic Schema Validation
     ↓
Component Configuration Creation
     ↓
Parameter Validation
     ↓
Component Instantiation
```

## Custom vs HF Models

### Custom Models (GCLI-based)

**Characteristics**:
- Modular component-based architecture
- Full control over each component
- Pydantic configuration validation
- Component registry management
- Type-safe assembly

**Architecture**:
```python
class ModularAutoformer(BaseTimeSeriesForecaster):
    def __init__(self, config):
        # Component assembly through ModularAssembler
        self.assembler = ModularAssembler()
        self.encoder, self.decoder, ... = self.assembler.assemble(config)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Forward pass through assembled components
```

**Advantages**:
- Maximum flexibility and customization
- Component-level testing and validation
- Easy to extend with new components
- Type safety and validation

**Use Cases**:
- Research and experimentation
- Custom component development
- Fine-grained control over architecture

### HF Models

**Characteristics**:
- Pre-built HuggingFace model implementations
- Production-ready and optimized
- Consistent with HF ecosystem
- Automatic config completion

**Architecture**:
```python
class HFEnhancedAutoformer(BaseTimeSeriesForecaster, HFFrameworkMixin):
    def __init__(self, configs):
        super().__init__(configs)
        self.framework_type = 'hf'
        # HF-specific initialization
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # HF model forward pass
```

**Advantages**:
- Battle-tested implementations
- Optimized performance
- Consistent APIs
- Production stability

**Use Cases**:
- Production deployments
- Standard forecasting tasks
- When stability is prioritized over flexibility

### Unified Interface

Both model types are wrapped in a `UnifiedModelInterface`:

```python
class UnifiedModelInterface:
    def __init__(self, model: BaseTimeSeriesForecaster):
        self.model = model
        self.framework_type = getattr(model, 'framework_type', 'unknown')
        
    def predict(self, x_enc, x_mark_enc, x_dec, x_mark_dec, **kwargs):
        return self.model(x_enc, x_mark_enc, x_dec, x_mark_dec, **kwargs)
        
    def predict_with_uncertainty(self, ...):
        # Unified uncertainty prediction interface
```

## Adding New Models

### Adding a Custom Model

#### 1. Define Model Configuration

Create a new configuration class in `configs/schemas.py`:

```python
@dataclass
class YourCustomConfig(AutoformerConfig):
    custom_param1: int = 64
    custom_param2: str = "default_value"
    
    # Component configurations
    attention: AttentionConfig = field(default_factory=lambda: AttentionConfig(
        component_type=ComponentType.YOUR_ATTENTION,
        d_model=512,
        n_heads=8
    ))
```

#### 2. Create Model Implementation

Implement your model in `models/your_custom_model.py`:

```python
class YourCustomAutoformer(BaseTimeSeriesForecaster):
    def __init__(self, config: YourCustomConfig):
        super().__init__(config)
        self.assembler = ModularAssembler()
        self.components = self.assembler.assemble(config)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Your custom forward logic
        return predictions
```

#### 3. Register with Factory

Add to `UnifiedAutoformerFactory` in `models/unified_autoformer_factory.py`:

```python
CUSTOM_MODELS = {
    # ... existing models
    'your_custom': 'modular',
}

def _create_custom_model(cls, model_type: str, config: Namespace):
    if model_type == 'your_custom':
        return YourCustomAutoformer(config)
    # ... existing logic
```

#### 4. Add Configuration Template

Create a configuration template in `configs/model_configs/`:

```python
# your_custom_config.py
def get_your_custom_config():
    return YourCustomConfig(
        # Default parameters
        seq_len=96,
        pred_len=24,
        # Custom parameters
        custom_param1=64,
        custom_param2="optimized"
    )
```

### Adding an HF Model

#### 1. Implement HF Model

Create your HF model in `models/HFYourModel.py`:

```python
class HFYourModel(BaseTimeSeriesForecaster, HFFrameworkMixin):
    def __init__(self, configs):
        super().__init__(configs)
        self.framework_type = 'hf'
        self.model_type = 'hf_your_model'
        
        # Initialize HF components
        self.hf_backbone = self._initialize_hf_backbone()
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # HF model implementation
        return predictions
```

#### 2. Register with Factory

Add to `UnifiedAutoformerFactory`:

```python
from models.HFYourModel import HFYourModel

HF_MODELS = {
    # ... existing models
    'hf_your_model': HFYourModel,
}

hf_type_mapping = {
    # ... existing mappings
    'your_model': 'hf_your_model',
}
```

#### 3. Update Imports

Add import in `unified_autoformer_factory.py`:

```python
from models.HFYourModel import HFYourModel
```

## Adding New Components

### 1. Define Component Type

Add new component type to `configs/schemas.py`:

```python
class ComponentType(str, Enum):
    # ... existing types
    YOUR_NEW_COMPONENT = "your_new_component"
```

### 2. Create Component Configuration

Add configuration schema:

```python
@dataclass
class YourComponentConfig(BaseConfig):
    component_type: ComponentType = ComponentType.YOUR_NEW_COMPONENT
    your_param1: int = 32
    your_param2: float = 0.1
```

### 3. Create Base Component Class

Add base class in `configs/modular_components.py`:

```python
class YourComponent(ModularComponent):
    """Base class for your component type"""
    
    def __init__(self, config: YourComponentConfig, **kwargs):
        super().__init__(config, **kwargs)
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """Component-specific forward method"""
        pass
```

### 4. Implement Concrete Component

Create implementation in `configs/concrete_components.py`:

```python
class YourConcreteComponent(YourComponent):
    def __init__(self, config: YourComponentConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.metadata = ComponentMetadata(
            name="YourConcreteComponent",
            component_type=ComponentType.YOUR_NEW_COMPONENT,
            required_params=['your_param1'],
            optional_params=['your_param2'],
            description="Description of your component"
        )
    
    def _initialize_component(self, **kwargs):
        # Component initialization
        self.your_param1 = self.config.your_param1
        self.layer = nn.Linear(self.your_param1, self.your_param1)
    
    def forward(self, x, **kwargs):
        # Component logic
        return self.layer(x)
```

### 5. Register Component

Add registration in `concrete_components.py`:

```python
def register_concrete_components():
    # ... existing registrations
    
    component_registry.register_component(
        ComponentType.YOUR_NEW_COMPONENT,
        YourConcreteComponent,
        ComponentMetadata(
            name="YourConcreteComponent",
            component_type=ComponentType.YOUR_NEW_COMPONENT,
            required_params=['your_param1'],
            optional_params=['your_param2'],
            description="Description of your component"
        )
    )
```

### 6. Update Assembler

Modify `ModularAssembler` in `configs/modular_components.py`:

```python
class ModularAssembler:
    def _create_your_component(self, config: YourComponentConfig, **kwargs):
        """Create your component from configuration"""
        component_class = component_registry.get_component(config.component_type)
        return component_class(config, **kwargs)
    
    def assemble(self, config: AutoformerConfig):
        # ... existing assembly logic
        your_component = self._create_your_component(config.your_component)
        # Integrate into model assembly
```

### 7. Update Model Integration

Modify models to use the new component:

```python
class ModularAutoformer(BaseTimeSeriesForecaster):
    def __init__(self, config):
        # ... existing initialization
        self.your_component = assembler.create_your_component(config.your_component)
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # ... existing forward logic
        x = self.your_component(x, **kwargs)
        # Continue with forward pass
```

## Testing and Validation

### Component Testing Requirements

#### 1. Unit Tests

Create component tests in `tests/components/test_your_component.py`:

```python
class TestYourComponent(unittest.TestCase):
    def setUp(self):
        self.config = YourComponentConfig(
            your_param1=32,
            your_param2=0.1
        )
        self.component = YourConcreteComponent(self.config)
    
    def test_initialization(self):
        """Test component initialization"""
        self.assertEqual(self.component.your_param1, 32)
        self.assertIsNotNone(self.component.layer)
    
    def test_forward_pass(self):
        """Test forward pass functionality"""
        x = torch.randn(2, 96, 32)
        output = self.component(x)
        self.assertEqual(output.shape, x.shape)
    
    def test_parameter_validation(self):
        """Test parameter validation"""
        # Test invalid parameters
        with self.assertRaises(ValidationError):
            invalid_config = YourComponentConfig(your_param1=-1)
```

#### 2. Integration Tests

Add integration tests in `tests/integration/test_your_component_integration.py`:

```python
def test_component_integration():
    """Test component integration with complete model"""
    config = AutoformerConfig(
        your_component=YourComponentConfig(your_param1=64)
    )
    model = ModularAutoformer(config)
    
    # Test forward pass
    x_enc = torch.randn(2, 96, 7)
    x_mark_enc = torch.randn(2, 96, 4)
    x_dec = torch.randn(2, 72, 7)
    x_mark_dec = torch.randn(2, 72, 4)
    
    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    assert output.shape == (2, 24, 7)
```

#### 3. Registry Tests

Test component registration:

```python
def test_component_registry():
    """Test component registry functionality"""
    # Test registration
    assert ComponentType.YOUR_NEW_COMPONENT in component_registry._components
    
    # Test retrieval
    component_class = component_registry.get_component(ComponentType.YOUR_NEW_COMPONENT)
    assert component_class == YourConcreteComponent
    
    # Test metadata
    metadata = component_registry.get_metadata(ComponentType.YOUR_NEW_COMPONENT)
    assert metadata.name == "YourConcreteComponent"
```

### Required Updates for New Components

#### 1. Component Registry
- Register component type and implementation
- Add metadata with parameter requirements
- Update registry initialization

#### 2. Configuration Schemas
- Add component type to `ComponentType` enum
- Create component-specific configuration schema
- Update main configuration to include new component

#### 3. Assembler
- Add assembly method for new component
- Integrate component into model assembly process
- Handle component dependencies

#### 4. Model Integration
- Update models to use new component
- Modify forward pass to incorporate component
- Handle component interactions

#### 5. Testing Suite
- Create comprehensive unit tests
- Add integration tests
- Test registry functionality
- Validate component interactions

#### 6. Documentation
- Update component documentation
- Add usage examples
- Document parameter requirements
- Update architecture diagrams

### Dimension and Compatibility Management

#### 1. Dimension Validation

Components must validate input/output dimensions:

```python
class YourConcreteComponent(YourComponent):
    def _validate_dimensions(self, x):
        """Validate input dimensions"""
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input, got {x.dim()}D")
        
        batch_size, seq_len, d_model = x.shape
        if d_model != self.expected_dim:
            raise ValueError(f"Expected d_model={self.expected_dim}, got {d_model}")
    
    def forward(self, x, **kwargs):
        self._validate_dimensions(x)
        # Forward logic
```

#### 2. Cross-Component Validation

Implement compatibility checks:

```python
class ComponentCompatibilityValidator:
    @staticmethod
    def validate_compatibility(config: AutoformerConfig):
        """Validate component compatibility"""
        # Check attention-encoder compatibility
        if config.attention.d_model != config.encoder.d_model:
            raise ValueError("Attention and encoder d_model mismatch")
        
        # Check encoder-decoder compatibility
        if config.encoder.d_model != config.decoder.d_model:
            raise ValueError("Encoder and decoder d_model mismatch")
        
        # Add your component validations
        if hasattr(config, 'your_component'):
            if config.your_component.your_param1 != config.encoder.d_model:
                raise ValueError("Your component dimension mismatch")
```

#### 3. Runtime Validation

Add runtime checks in the assembler:

```python
def assemble(self, config: AutoformerConfig):
    # Validate configuration
    ComponentCompatibilityValidator.validate_compatibility(config)
    
    # Create components
    components = self._create_all_components(config)
    
    # Validate component interactions
    self._validate_component_interactions(components)
    
    return ModularAutoformer(config, **components)
```

## Best Practices

### 1. Component Design

- **Single Responsibility**: Each component should have a single, well-defined purpose
- **Interface Consistency**: Follow the `ModularComponent` interface
- **Parameter Validation**: Use Pydantic for type safety and validation
- **Documentation**: Include comprehensive docstrings and metadata

### 2. Configuration Management

- **Type Safety**: Use Pydantic dataclasses for all configurations
- **Default Values**: Provide sensible defaults for optional parameters
- **Validation**: Implement custom validators for complex constraints
- **Versioning**: Track configuration schema versions

### 3. Testing Strategy

- **Unit Tests**: Test each component in isolation
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete model workflows
- **Regression Tests**: Ensure backward compatibility

### 4. Performance Considerations

- **Memory Efficiency**: Monitor memory usage in components
- **Computational Efficiency**: Profile critical paths
- **Batching**: Ensure components handle batched inputs correctly
- **GPU Compatibility**: Test on both CPU and GPU

### 5. Error Handling

- **Graceful Degradation**: Handle component failures gracefully
- **Informative Errors**: Provide clear error messages
- **Validation**: Validate inputs and configurations early
- **Logging**: Include appropriate logging for debugging

### 6. Documentation

- **API Documentation**: Document all public methods and parameters
- **Usage Examples**: Provide practical usage examples
- **Architecture Diagrams**: Maintain visual documentation
- **Migration Guides**: Document breaking changes and migrations

This comprehensive framework provides a robust foundation for building, extending, and maintaining modular autoformer architectures with seamless HF integration. The modular design ensures flexibility while maintaining type safety and validation throughout the system.
