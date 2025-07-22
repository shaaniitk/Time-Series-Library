# Modular HFAutoformer Architecture Documentation

## Overview

The Modular HFAutoformer is a **unified, component-based time series forecasting architecture** that uses the **base Autoformer design** enhanced with **modular utilities and components**. This architecture emerged from the evolution of four initial prototype models into a single, flexible, modular design that leverages the **component registry system** for maximum flexibility and extensibility.

**Key Concept**: Instead of separate model files, we have **one modular HFAutoformer** that can be configured with different **backbone components**, **loss functions**, **attention mechanisms**, and **specialized processors** through the modular component registry.

This document provides a comprehensive guide to understanding, using, and extending the modular HFAutoformer architecture.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Modular HFAutoformer Core](#modular-hfautoformer-core)
3. [Component Registry System](#component-registry-system)
4. [Modular Backbone Integration](#modular-backbone-integration)
5. [Advanced Component Library](#advanced-component-library)
6. [Configuration and Usage](#configuration-and-usage)
7. [Adding New Components](#adding-new-components)
8. [Testing Framework](#testing-framework)
9. [Evolution from Prototypes](#evolution-from-prototypes)
10. [Troubleshooting](#troubleshooting)

## Architecture Overview

The HFAutoformer models represent a **paradigm shift** from in-house transformer implementations to **leveraging pre-trained Hugging Face models** as backbones. This approach provides:

- **Pre-trained Intelligence**: Leverage models like Amazon Chronos, Google T5, etc.
- **Backbone Flexibility**: Swap between different HF models via modular registry
- **Bayesian Integration**: Uncertainty quantification with KL divergence support
- **Quantile Regression**: Multi-quantile forecasting capabilities
- **Hierarchical Processing**: Multi-resolution time series analysis

### Key Distinction: HFAutoformer vs EnhancedAutoformer

| Aspect | HFAutoformer | EnhancedAutoformer |
|--------|--------------|-------------------|
| **Backbone** | 🤗 Hugging Face Models (Chronos, T5, BERT) | 🏠 In-house transformer layers |
| **Approach** | Pre-trained model adaptation | Custom architecture design |
| **Components** | Modular backbone registry | Fixed architecture components |
| **Training** | Fine-tuning + adaptation layers | End-to-end training |
| **Flexibility** | Runtime backbone swapping | Fixed model architecture |

### Directory Structure

```
Time-Series-Library/
├── models/
│   ├── HFAutoformerSuite.py           # 🤗 MAIN HF AUTOFORMER MODELS
│   ├── HFEnhancedAutoformer.py        # Basic HF enhanced model
│   ├── HFBayesianAutoformer.py        # Bayesian uncertainty quantification
│   ├── HFHierarchicalAutoformer_Step3.py # Multi-resolution processing
│   ├── HFQuantileAutoformer_Step4.py  # Quantile regression
│   └── EnhancedAutoformer.py          # ❌ In-house model (different system)
├── utils/
│   ├── modular_components/            # 🔥 BACKBONE COMPONENT SYSTEM
│   │   ├── implementations/
│   │   │   ├── backbones.py           # 🤗 HF backbone implementations
│   │   │   ├── simple_backbones.py    # Simple transformer backbones
│   │   │   ├── advanced_losses.py     # 🧠 Bayesian & advanced losses
│   │   │   ├── advanced_attentions.py # ⚡ Optimized attention layers
│   │   │   └── specialized_processors.py # 🔧 Signal processing
│   │   ├── registry.py                # Component discovery system
│   │   └── base_interfaces.py         # Abstract backbone interfaces
│   ├── bayesian_losses.py             # 🎯 KL divergence implementations
│   └── enhanced_losses.py             # Advanced loss library
└── ...
```

## Modular HFAutoformer Core

The Modular HFAutoformer represents the evolution of time series modeling into a **unified, component-based architecture**. Instead of separate model files for different capabilities, we have **one modular HFAutoformer** that can be configured with different backbone components and utilities to achieve various specialized behaviors.

### Core Architecture Philosophy

```python
# Unified HFAutoformer that can be configured for different use cases
class HFAutoformer(BaseAutoformer):
    def __init__(self, config):
        super().__init__(config)
        
        # Modular components loaded at runtime
        self.backbone = BackboneRegistry.get_backbone(config.backbone_type)
        self.loss_function = LossRegistry.get_loss(config.loss_type)
        self.attention = AttentionRegistry.get_attention(config.attention_type)
        self.processor = ProcessorRegistry.get_processor(config.processor_type)
        
        # Initialize with selected components
        self._initialize_modular_components()
```

### Configuration-Based Specialization

The same unified HFAutoformer can be configured for different specialized behaviors:

**Bayesian Uncertainty Quantification:**
```python
config = {
    'backbone_type': 'bayesian_hf',
    'loss_type': 'kl_divergence',
    'attention_type': 'bayesian_attention',
    'processor_type': 'uncertainty_processor'
}
```

**Quantile Regression:**
```python
config = {
    'backbone_type': 'quantile_hf', 
    'loss_type': 'quantile_loss',
    'attention_type': 'multi_quantile_attention',
    'processor_type': 'quantile_processor'
}
```

**Hierarchical Multi-Scale:**
```python
config = {
    'backbone_type': 'hierarchical_hf',
    'loss_type': 'hierarchical_loss', 
    'attention_type': 'hierarchical_attention',
    'processor_type': 'multi_scale_processor'
}
```

### Integration with Base Autoformer

The modular HFAutoformer inherits all core functionality from the base Autoformer while adding:

- **Dynamic Component Loading**: Runtime selection of backbone models
- **Configuration-Driven Architecture**: Same model, different behaviors  
- **Pluggable Components**: Easy swapping of attention, loss, processing modules
- **Backward Compatibility**: Works with existing Autoformer configurations
- **HF Model Integration**: Seamless use of pre-trained Hugging Face models

## Component Registry System

The heart of the modular HFAutoformer architecture is its **Component Registry System**, which dynamically discovers and manages 34 specialized components across 8 categories. This system enables runtime component selection and configuration without code changes.

### Registry Categories

The system organizes components into logical categories:

```python
COMPONENT_CATEGORIES = {
    'backbones': 'Hugging Face and transformer backbone models',
    'losses': 'Loss functions (Bayesian, quantile, hierarchical)',  
    'attentions': 'Attention mechanisms and optimizations',
    'processors': 'Signal processing and feature extraction',
    'embeddings': 'Input embedding and encoding layers',
    'decoders': 'Output decoding and projection layers',
    'optimizers': 'Training optimization strategies',
    'schedulers': 'Learning rate scheduling approaches'
}
```

### Dynamic Component Discovery

The registry automatically discovers components using Python introspection:

```python
# Automatic component discovery
discovered_components = ComponentRegistry.discover_components()
print(f"Found {len(discovered_components)} components across {len(COMPONENT_CATEGORIES)} categories")

# Runtime component access
backbone = BackboneRegistry.get_backbone('chronos_t5')
loss_fn = LossRegistry.get_loss('kl_divergence') 
attention = AttentionRegistry.get_attention('hierarchical')
```

### Component Interface

All components implement standardized interfaces for seamless integration:

```python
# Base component interface
class BaseComponent(ABC):
    @abstractmethod
    def initialize(self, config): pass
    
    @abstractmethod  
    def forward(self, *args, **kwargs): pass
    
    @abstractmethod
    def get_config(self): pass
## Modular Backbone Integration

The backbone integration system is the core of the modular HFAutoformer, providing seamless integration with Hugging Face pre-trained models while maintaining flexibility for custom implementations.

### Backbone Registry

The backbone registry manages available backbone models with automatic fallback handling:

```python
class BackboneRegistry:
    AVAILABLE_BACKBONES = {
        'chronos_t5': ChronosT5Backbone,
        'amazon_chronos': AmazonChronosBackbone, 
        'huggingface_t5': HuggingFaceT5Backbone,
        'bert_backbone': BERTBackbone,
        'simple_transformer': SimpleTransformerBackbone,
        'enhanced_transformer': EnhancedTransformerBackbone
    }
    
    @classmethod
    def get_backbone(cls, backbone_type, config):
        """Get backbone with automatic fallback handling"""
        try:
            return cls.AVAILABLE_BACKBONES[backbone_type](config)
        except Exception as e:
            logger.warning(f"Failed to load {backbone_type}, falling back to simple_transformer")
            return cls.AVAILABLE_BACKBONES['simple_transformer'](config)
```

### Backbone Hierarchy

The system implements an intelligent fallback hierarchy:

1. **Primary**: Amazon Chronos T5 (pre-trained time series model)
2. **Secondary**: Hugging Face T5 (general pre-trained model)  
3. **Tertiary**: BERT-based backbone (if available)
4. **Fallback**: Simple transformer (always available)

### Integration Architecture

Each backbone implements a standardized interface for seamless integration:

```python
class HFBackbone(BaseBackbone):
    def __init__(self, config):
        self.hf_model = self._load_hf_model(config.model_name)
        self.input_projection = nn.Linear(config.enc_in, config.d_model)
        self.output_projection = nn.Linear(config.d_model, config.c_out)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Project input to backbone dimension
        x_enc_proj = self.input_projection(x_enc)
        
        # Process through HF backbone
        backbone_output = self.hf_model(x_enc_proj)
        
        # Project to output dimension
        prediction = self.output_projection(backbone_output)
        
        return prediction
```
## Advanced Component Library

The modular HFAutoformer system includes a comprehensive library of 34 specialized components organized across 8 categories. Each component can be dynamically loaded and configured at runtime.

### Loss Function Components

Advanced loss functions for specialized forecasting scenarios:

```python
# Bayesian Loss with KL Divergence
class BayesianKLLoss(BaseLoss):
    def __init__(self, kl_weight=0.3, reduction='mean'):
        self.kl_weight = kl_weight
        self.mse_loss = nn.MSELoss(reduction=reduction)
        
    def forward(self, pred, true, model=None):
        # Standard reconstruction loss
        recon_loss = self.mse_loss(pred, true)
        
        # KL divergence for regularization  
        kl_loss = self._compute_kl_divergence(model)
        
        return recon_loss + self.kl_weight * kl_loss

# Quantile Loss for Probabilistic Forecasting
class QuantileLoss(BaseLoss):
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        self.quantiles = quantiles
        
    def forward(self, pred, true):
        total_loss = 0
        for i, q in enumerate(self.quantiles):
            pred_q = pred[:, :, i]  # Quantile-specific prediction
            errors = true - pred_q
            loss = torch.mean(torch.max(q * errors, (q - 1) * errors))
            total_loss += loss
        return total_loss
```

### Attention Mechanism Components

Optimized attention mechanisms for different use cases:

```python
# Hierarchical Attention for Multi-Scale Processing
class HierarchicalAttention(BaseAttention):
    def __init__(self, d_model, n_heads, hierarchy_levels=3):
        self.hierarchy_levels = hierarchy_levels
        self.scale_attentions = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads) 
            for _ in range(hierarchy_levels)
        ])
        
    def forward(self, x):
        # Process at multiple temporal scales
        scale_outputs = []
        for i, attention in enumerate(self.scale_attentions):
            scale_factor = 2 ** i
            x_scaled = self._temporal_downsample(x, scale_factor)
            attended = attention(x_scaled, x_scaled, x_scaled)[0]
            scale_outputs.append(self._temporal_upsample(attended, scale_factor))
            
        # Fuse multi-scale representations
        return self._fuse_scales(scale_outputs)

# Bayesian Attention with Uncertainty
class BayesianAttention(BaseAttention):
    def __init__(self, d_model, n_heads, dropout_rate=0.1):
        self.attention = nn.MultiheadAttention(d_model, n_heads)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, return_uncertainty=False):
        if return_uncertainty:
            # Monte Carlo sampling for uncertainty
            samples = []
            for _ in range(10):  # MC samples
                attended = self.attention(x, x, x)[0]
                samples.append(self.dropout(attended))
            
            # Compute mean and uncertainty
            mean_output = torch.stack(samples).mean(dim=0)
            uncertainty = torch.stack(samples).var(dim=0)
            return mean_output, uncertainty
        else:
            return self.attention(x, x, x)[0]
```

### Specialized Processor Components

Signal processing and feature extraction components:

```python
# Wavelet Processing for Time-Frequency Analysis
class WaveletProcessor(BaseProcessor):
    def __init__(self, wavelet='db4', levels=3):
        self.wavelet = wavelet
        self.levels = levels
        
    def forward(self, x):
        # Wavelet decomposition
        coeffs = pywt.wavedec(x.cpu().numpy(), self.wavelet, level=self.levels)
        
        # Process coefficients
        processed_coeffs = []
        for coeff in coeffs:
            coeff_tensor = torch.from_numpy(coeff).to(x.device)
            processed_coeffs.append(self._process_coefficient(coeff_tensor))
            
        # Wavelet reconstruction
        return pywt.waverec(processed_coeffs, self.wavelet)

# Multi-Scale Feature Processor
class MultiScaleProcessor(BaseProcessor):
    def __init__(self, scales=[1, 4, 16]):
        self.scales = scales
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(1, 64, kernel_size=scale, stride=scale)
            for scale in scales
        ])
        
    def forward(self, x):
        # Extract features at multiple scales
        scale_features = []
        for i, conv in enumerate(self.conv_layers):
            features = conv(x.unsqueeze(1))  # Add channel dimension
            features = F.adaptive_avg_pool1d(features, x.size(-1))  # Resize back
            scale_features.append(features.squeeze(1))
            
## Configuration and Usage

The modular HFAutoformer provides flexible configuration options for different use cases and requirements.

### Basic Configuration

```python
# Basic configuration for standard forecasting
config = {
    'seq_len': 96,          # Input sequence length
    'pred_len': 24,         # Prediction horizon  
    'enc_in': 7,            # Number of input features
    'c_out': 7,             # Number of output features
    'd_model': 512,         # Model dimension
    'backbone_type': 'chronos_t5',     # HF backbone
    'loss_type': 'mse',                # Loss function
    'attention_type': 'standard',      # Attention mechanism
    'processor_type': 'identity'       # Feature processor
}

# Initialize modular HFAutoformer
model = HFAutoformer(config)
```

### Specialized Configurations

**Bayesian Uncertainty Quantification:**
```python
bayesian_config = {
    **base_config,
    'backbone_type': 'chronos_t5',
    'loss_type': 'bayesian_kl',
    'attention_type': 'bayesian_attention', 
    'processor_type': 'uncertainty_processor',
    'uncertainty_method': 'bayesian',
    'n_samples': 50,
    'kl_weight': 0.3
}

model = HFAutoformer(bayesian_config)

# Get predictions with uncertainty
result = model(x_enc, x_mark_enc, x_dec, x_mark_dec, return_uncertainty=True)
prediction = result['prediction']
uncertainty = result['uncertainty'] 
confidence_intervals = result['confidence_intervals']
```

**Quantile Regression:**
```python
quantile_config = {
    **base_config,
    'backbone_type': 'chronos_t5',
    'loss_type': 'quantile_loss',
    'attention_type': 'multi_quantile_attention',
    'processor_type': 'quantile_processor', 
    'quantiles': [0.1, 0.25, 0.5, 0.75, 0.9],
    'quantile_weight': 1.0
}

model = HFAutoformer(quantile_config) 

# Get quantile predictions
quantile_outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
# Returns predictions for each quantile level
```

**Hierarchical Multi-Scale:**
```python
hierarchical_config = {
    **base_config,
    'backbone_type': 'chronos_t5',
    'loss_type': 'hierarchical_loss',
    'attention_type': 'hierarchical_attention',
    'processor_type': 'multi_scale_processor',
    'hierarchy_levels': 3,
    'resolution_scales': [1, 4, 16],
    'fusion_method': 'weighted'
}

model = HFAutoformer(hierarchical_config)

# Get multi-scale predictions
output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
```

### Runtime Component Swapping

The modular design allows runtime component changes:

```python
# Initialize with one configuration
model = HFAutoformer(base_config)

# Dynamically change backbone
model.backbone = BackboneRegistry.get_backbone('t5_backbone', config)

# Change loss function 
model.loss_function = LossRegistry.get_loss('quantile_loss', config)

# Update attention mechanism
model.attention = AttentionRegistry.get_attention('hierarchical', config)
```

### Training Integration

The modular HFAutoformer integrates seamlessly with existing training infrastructure:

```python
# Standard training loop
def train_hfautoformer(model, train_loader, val_loader, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    for epoch in range(config.train_epochs):
        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            output = model(batch_x, batch_x_mark, 
                          batch_y[:, :config.label_len, :], 
                          batch_y_mark[:, :config.label_len, :])
            
            # Compute loss using configured loss function
            loss = model.loss_function(output, batch_y)
            
            loss.backward()
            optimizer.step()
```

**Google T5** - Text-to-text transfer transformer

```python
backbone_config = {
    'model_name': 'google/flan-t5-small',  # or -base, -large
    'encoder_only': True,  # Use only encoder for efficiency
    'd_model': 512
}
```

**Features:**
- **Encoder-decoder** or **encoder-only** modes
- **Robust fallback** when Chronos unavailable
- **Well-established architecture** with strong performance
- **Memory efficient** with encoder-only option

#### 3. RobustHFBackbone 🛡️

**Robust Hugging Face integration** with comprehensive error handling

```python
backbone_config = {
    'model_families': ['chronos', 't5', 'bert'],  # Fallback hierarchy
    'auto_fallback': True,
    'error_recovery': 'graceful'
}
```

**Features:**
- **Intelligent fallback chain**: Chronos → T5 → BERT → simple transformer
- **Error recovery**: Graceful degradation when models unavailable
- **Automatic configuration**: Smart parameter adaptation
- **Production ready**: Comprehensive error handling

#### 4. SimpleTransformerBackbone ⚡

**Lightweight transformer** for resource-constrained environments

```python
backbone_config = {
    'd_model': 256,
    'n_heads': 8,
    'n_layers': 6,
    'dropout': 0.1
}
```

**Features:**
- **No external dependencies**: Pure PyTorch implementation
- **Configurable architecture**: Adjustable layers, heads, dimensions
- **Fast initialization**: No model download required
- **Memory efficient**: Optimized for smaller models

### Backbone Selection Strategy

The HFAutoformer models follow this **automatic selection hierarchy**:

```python
# Backbone selection priority (highest to lowest)
1. ChronosBackbone     # 🚀 If Chronos available and working
2. T5Backbone          # 📝 If T5 available (strong fallback)
3. RobustHFBackbone    # 🛡️ If HF transformers available
4. SimpleTransformerBackbone  # ⚡ Always available (PyTorch only)
```

### Runtime Backbone Swapping

Through the modular registry, backbones can be swapped at runtime:

```python
from utils.modular_components.registry import create_component

# Create different backbones
chronos_backbone = create_component('backbone', 'chronos', chronos_config)
t5_backbone = create_component('backbone', 't5', t5_config)
robust_backbone = create_component('backbone', 'robust_hf', robust_config)

# Use in HFAutoformer models
model.set_backbone(chronos_backbone)  # Switch to Chronos
model.set_backbone(t5_backbone)       # Switch to T5
```

## Modular Component Registry

### Overview

The modular component registry (`utils/modular_components/registry.py`) enables **dynamic backbone discovery and instantiation** for HFAutoformer models.

### Registry System for HFAutoformer

```python
# Global registry for backbone components
from utils.modular_components.registry import get_global_registry, create_component

# Get registry
registry = get_global_registry()

# Create backbone components
chronos_backbone = create_component('backbone', 'chronos', {
    'model_name': 'amazon/chronos-t5-small',
    'pretrained': True,
    'd_model': 512
})

# Create supporting components for HFAutoformer
bayesian_loss = create_component('loss', 'bayesian_mse', {
    'kl_weight': 1e-5,
    'uncertainty_weight': 0.1
})

attention = create_component('attention', 'optimized_autocorrelation', {
    'd_model': 512,
    'num_heads': 8
})
```

### HFAutoformer-Specific Component Types

#### 1. **Backbone Components** (`backbone`) 🤗
- `chronos`: Amazon Chronos time series transformer
- `t5`: Google T5 encoder-decoder or encoder-only
- `robust_hf`: Robust HF integration with fallbacks
- `simple_transformer`: Lightweight PyTorch-only backbone
- `bert`: BERT-based backbone for sequence modeling

#### 2. **Loss Functions** (`loss`) 🧠
- **Bayesian Losses**: `bayesian_mse`, `bayesian_mae`, `bayesian_quantile`
  - **Critical for HFBayesianAutoformer**: Proper KL divergence support
  - Integration with `utils/bayesian_losses.py`
- **Advanced Losses**: `adaptive_structural`, `frequency_aware`
- **Quantile Losses**: `pinball`, `quantile_regression`

#### 3. **Attention Mechanisms** (`attention`) ⚡
- `optimized_autocorrelation`: Memory-optimized for long sequences
- `adaptive_autocorrelation`: Multi-scale attention
- `enhanced_autocorrelation`: Complete attention layer
- `memory_efficient`: Gradient checkpointing support

#### 4. **Specialized Processors** (`processor`) 🔧
- `frequency_domain`: FFT-based spectral analysis
- `dtw_alignment`: Dynamic time warping
- `trend_analysis`: Multi-scale trend extraction
- `integrated_signal`: Comprehensive signal processing

### Component Registration for HFAutoformer

Components are automatically registered through the modular system:

```python
# Automatic registration in implementations/__init__.py
from utils.modular_components.implementations import get_integration_status

status = get_integration_status()
print(status)  # Shows all available components including backbones
```

**Registration Categories:**
- **Backbone Models**: 7 registered (Chronos, T5, BERT, etc.)
- **Bayesian Components**: 3 registered (MSE, MAE, Quantile with KL)
- **Attention Mechanisms**: 9 registered (optimized, adaptive, enhanced)
- **Loss Functions**: 12 registered (standard + advanced + Bayesian)
- **Specialized Processors**: 6 registered (frequency, DTW, trend, etc.)

## Advanced Integrations

### Bayesian Components 🧠

**Critical for HFBayesianAutoformer**: Proper KL divergence support restored

#### Bayesian Loss Integration
```python
# Located in utils/modular_components/implementations/advanced_losses.py
# Integrates with HFBayesianAutoformer

class BayesianMSELoss(BaseLoss):
    """MSE Loss with KL divergence for HFBayesianAutoformer"""
    
    def __init__(self, config):
        # Integration with existing utils/bayesian_losses.py
        from utils.bayesian_losses import BayesianLoss
        base_loss_fn = nn.MSELoss(reduction=self.reduction)
        self.bayesian_loss = BayesianLoss(
            base_loss_fn=base_loss_fn,
            kl_weight=self.kl_weight,           # Critical for Bayesian training
            uncertainty_weight=self.uncertainty_weight
        )
    
    def compute_loss(self, pred, true, model=None):
        """Compute loss with KL divergence extraction"""
        if model is not None:
            # Extract KL divergence from model parameters
            kl_div = self.bayesian_loss.compute_kl_loss(model)
            base_loss = self.bayesian_loss.base_loss_fn(pred, true)
            return base_loss + self.kl_weight * kl_div
        else:
            return self.bayesian_loss.base_loss_fn(pred, true)
```

#### Usage with HFBayesianAutoformer
```python
# Create HFBayesianAutoformer with proper KL divergence
from models.HFAutoformerSuite import HFBayesianAutoformer
from utils.modular_components.registry import create_component

# Create Bayesian loss with KL support
bayesian_loss = create_component('loss', 'bayesian_mse', {
    'kl_weight': 1e-5,           # Critical for proper Bayesian training
    'uncertainty_weight': 0.1,   # Uncertainty regularization
    'reduction': 'mean'
})

# Create HFBayesianAutoformer model
model = HFBayesianAutoformer(
    configs,
    uncertainty_method='bayesian',
    n_samples=50
)

# Training with proper KL divergence
pred = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
loss = bayesian_loss.compute_loss(pred, true, model=model)  # Includes KL divergence
```

### Optimized Attention for HFAutoformer ⚡

#### Memory-Optimized Attention Integration
```python
# Located in utils/modular_components/implementations/advanced_attentions.py
# Can be used in HFAutoformer attention layers

class OptimizedAutoCorrelationAttention(BaseAttention):
    """Memory-optimized attention for HFAutoformer models"""
    
    def __init__(self, config):
        # Integration with layers/AutoCorrelation_Optimized.py
        from layers.AutoCorrelation_Optimized import OptimizedAutoCorrelation
        
        self.attention = OptimizedAutoCorrelation(
            max_seq_len=config.max_seq_len,  # Memory optimization
            d_model=config.d_model,
            num_heads=config.num_heads,
            use_mixed_precision=True,        # Memory efficiency
            chunk_size=config.get('chunk_size', 128)  # Chunked processing
        )
```

#### Integration with HFAutoformer Models
While HFAutoformer models primarily use their HF backbone attention, these optimized attention mechanisms can be used in:
- **Adaptation layers** between backbone and output
- **Multi-head attention wrappers** around backbone features
- **Hierarchical fusion** in HFHierarchicalAutoformer
- **Uncertainty estimation** layers in HFBayesianAutoformer

### Signal Processing Integration 🔧

#### Specialized Processors for HFAutoformer
```python
# Located in utils/modular_components/implementations/specialized_processors.py
# Used for pre/post-processing in HFAutoformer models

class IntegratedSignalProcessor:
    """Comprehensive signal processing for HFAutoformer"""
    
    def __init__(self):
        self.freq_processor = FrequencyDomainProcessor()      # FFT analysis
        self.dtw_processor = DTWAlignmentProcessor()          # Sequence alignment
        self.trend_processor = TrendProcessor()               # Multi-scale trends
    
    def preprocess_for_hf_backbone(self, x):
        """Preprocess time series for HF backbone consumption"""
        # Frequency domain analysis
        freq_features = self.freq_processor.extract_features(x)
        
        # Trend decomposition
        trend, seasonal = self.trend_processor.decompose(x)
        
        # Combine features for backbone input
        enhanced_input = torch.cat([x, freq_features, trend], dim=-1)
        return enhanced_input
    
    def postprocess_hf_output(self, backbone_output, original_input):
        """Post-process HF backbone output"""
        # Apply DTW alignment if needed
        aligned_output = self.dtw_processor.align(backbone_output, original_input)
        
        # Multi-scale trend integration
        final_output = self.trend_processor.integrate_trends(aligned_output)
        
        return final_output
```

#### Usage in HFAutoformer Pipeline
```python
# Enhanced HFAutoformer with signal processing
from utils.modular_components.registry import create_component

# Create signal processor
processor = create_component('processor', 'integrated_signal', {})

# Create HFAutoformer model
model = HFEnhancedAutoformer(configs)

# Enhanced forward pass with signal processing
def enhanced_forward(x_enc, x_mark_enc, x_dec, x_mark_dec):
    # Preprocess input for better backbone consumption
    enhanced_x_enc = processor.preprocess_for_hf_backbone(x_enc)
    
    # Standard HFAutoformer forward pass
    backbone_output = model(enhanced_x_enc, x_mark_enc, x_dec, x_mark_dec)
    
    # Post-process output for better predictions
    final_output = processor.postprocess_hf_output(backbone_output, x_enc)
    
    return final_output
```

## Adding New Components

The modular system makes it easy to add new components while maintaining compatibility with existing code.

### Creating a New Backbone

```python
# 1. Implement the backbone interface
class CustomHFBackbone(BaseBackbone):
    def __init__(self, config):
        super().__init__(config)
        self.hf_model = AutoModel.from_pretrained(config.model_name)
        self.input_projection = nn.Linear(config.enc_in, config.d_model)
        self.output_projection = nn.Linear(config.d_model, config.c_out)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Custom backbone logic
        x_proj = self.input_projection(x_enc)
        backbone_output = self.hf_model(x_proj)
        return self.output_projection(backbone_output.last_hidden_state)
        
    def get_config(self):
        return {'type': 'custom_hf', 'model_name': self.config.model_name}

# 2. Register the component
BackboneRegistry.register_backbone('custom_hf', CustomHFBackbone)

# 3. Use in configuration
config = {'backbone_type': 'custom_hf', 'model_name': 'custom/model'}
model = HFAutoformer(config)
```

### Creating a New Loss Function

```python
# 1. Implement loss interface
class CustomLoss(BaseLoss):
    def __init__(self, alpha=1.0, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, pred, true, model=None):
        # Custom loss computation
        mse_loss = F.mse_loss(pred, true)
        custom_penalty = self.beta * torch.mean(torch.abs(pred - true))
        return self.alpha * mse_loss + custom_penalty
        
    def get_config(self):
        return {'type': 'custom', 'alpha': self.alpha, 'beta': self.beta}

# 2. Register the component  
LossRegistry.register_loss('custom_loss', CustomLoss)

# 3. Use in model
config = {'loss_type': 'custom_loss', 'alpha': 1.5, 'beta': 0.3}
```

### Component Discovery

New components are automatically discovered if placed in the correct directory structure:

```
utils/modular_components/implementations/
├── backbones.py           # Add new backbone classes here
├── advanced_losses.py     # Add new loss functions here  
├── advanced_attentions.py # Add new attention mechanisms here
├── specialized_processors.py # Add new processors here
└── custom_components.py   # Or create new files for custom components
```

The registry system will automatically detect and register new components on import.

## Evolution from Prototypes

The current unified modular HFAutoformer architecture represents the evolution of our time series modeling approach. This section explains how we moved from separate prototype models to a single, configurable architecture.

### The Journey: From Separate Models to Unified Design

**Phase 1: Initial Prototypes (HFAutoformerSuite.py)**

Initially, we developed four separate HFAutoformer models to explore different capabilities:

1. **HFEnhancedAutoformer**: Basic HF backbone integration
2. **HFBayesianAutoformer**: Bayesian uncertainty quantification  
3. **HFHierarchicalAutoformer**: Multi-scale hierarchical processing
4. **HFQuantileAutoformer**: Quantile regression and probabilistic forecasting

```python
# Old approach - separate model files
from models.HFAutoformerSuite import (
    HFEnhancedAutoformer,     # Basic HF integration
    HFBayesianAutoformer,     # Uncertainty quantification  
    HFHierarchicalAutoformer, # Multi-scale processing
    HFQuantileAutoformer      # Quantile regression
)

# Each model was a separate implementation
enhanced_model = HFEnhancedAutoformer(config)
bayesian_model = HFBayesianAutoformer(config)  
hierarchical_model = HFHierarchicalAutoformer(config)
quantile_model = HFQuantileAutoformer(config)
```

**Phase 2: Component Extraction**

We realized that these models shared common patterns and extracted reusable components:

- **Backbone Integration**: All models used HF pre-trained models
- **Loss Functions**: Each model had specialized loss functions (KL divergence, quantile loss, etc.)
- **Attention Mechanisms**: Different attention patterns for different use cases
- **Processing Logic**: Specialized processors for different data types

**Phase 3: Unified Modular Design**

The breakthrough was realizing we could have **one HFAutoformer** that could be configured to achieve all these specialized behaviors:

```python
# New approach - one unified model with different configurations

# Instead of separate models, same model with different configs:

# Bayesian configuration (replaces HFBayesianAutoformer)
bayesian_config = {
    'backbone_type': 'chronos_t5',
    'loss_type': 'bayesian_kl', 
    'attention_type': 'bayesian_attention',
    'processor_type': 'uncertainty_processor'
}

# Quantile configuration (replaces HFQuantileAutoformer)  
quantile_config = {
    'backbone_type': 'chronos_t5',
    'loss_type': 'quantile_loss',
    'attention_type': 'multi_quantile_attention', 
    'processor_type': 'quantile_processor'
}

# Hierarchical configuration (replaces HFHierarchicalAutoformer)
hierarchical_config = {
    'backbone_type': 'chronos_t5',
    'loss_type': 'hierarchical_loss',
    'attention_type': 'hierarchical_attention',
    'processor_type': 'multi_scale_processor'
}

# All use the same unified HFAutoformer
model_bayesian = HFAutoformer(bayesian_config)
model_quantile = HFAutoformer(quantile_config) 
model_hierarchical = HFAutoformer(hierarchical_config)
```

### Benefits of the Unified Approach

**1. Code Reusability**
- Single model implementation instead of four separate ones
- Shared infrastructure for training, evaluation, and deployment
- Common interfaces and APIs

**2. Flexibility**
- Mix and match components (e.g., Bayesian backbone + quantile loss)
- Runtime configuration changes
- Easy experimentation with component combinations

**3. Maintainability** 
- Single point of updates and bug fixes
- Consistent behavior across different configurations
- Simplified testing and validation

**4. Extensibility**
- Add new components without modifying core model
- Registry system automatically discovers new components
- Backward compatibility with existing configurations

### Migration Path

For users transitioning from the prototype models:

```python
# Old way (still supported for backward compatibility)
from models.HFAutoformerSuite import HFBayesianAutoformer
model = HFBayesianAutoformer(config)

# New way (recommended)
from models.HFAutoformer import HFAutoformer
bayesian_config = {**config, 'backbone_type': 'chronos_t5', 'loss_type': 'bayesian_kl'}
model = HFAutoformer(bayesian_config)
```

The prototype models in `HFAutoformerSuite.py` remain available for backward compatibility and serve as examples of how to configure the unified modular architecture for specific use cases.

### Step 1: Implement Component Class

Create a new component by inheriting from the appropriate base class:

```python
# Example: Adding a new attention mechanism
# File: utils/modular_components/implementations/my_attention.py

from ..base_interfaces import BaseAttention
from ..config_schemas import AttentionConfig

class MyCustomAttention(BaseAttention):
    """Custom attention mechanism"""
    
    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        # Initialize your custom attention here
        
    def forward(self, queries, keys, values, attn_mask=None):
        # Implement your attention logic
        return output, attention_weights
    
    def get_attention_type(self) -> str:
        return "my_custom_attention"
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            'type': 'my_custom_attention',
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'custom_feature': True
        }
```

### Step 2: Register the Component

Add registration to the appropriate registration file:

```python
# In utils/modular_components/implementations/register_advanced.py
# Or create a new registration file

def register_my_components():
    """Register my custom components"""
    from .my_attention import MyCustomAttention
    
    register_component(
        'attention', 'my_custom_attention', MyCustomAttention,
        metadata={
            'type': 'custom',
            'performance': 'high',
            'memory_efficient': True
        }
    )

# Add to the main registration function
def register_all_advanced_components():
    register_advanced_losses()
    register_advanced_attentions()
    register_specialized_processors()
    register_my_components()  # Add this line
```

### Step 3: Update Auto-Registration

Ensure your component is imported and registered automatically:

```python
# In utils/modular_components/implementations/__init__.py

try:
    from . import my_attention
    MY_COMPONENTS_AVAILABLE = True
    logger.info("My custom components available")
except ImportError as e:
    logger.warning(f"Could not import my components: {e}")
    MY_COMPONENTS_AVAILABLE = False

if MY_COMPONENTS_AVAILABLE:
    __all__.append('my_attention')
```

### Step 4: Create Configuration Schema (Optional)

If your component needs special configuration:

```python
# In utils/modular_components/config_schemas.py

@dataclass
class MyCustomAttentionConfig(AttentionConfig):
    """Configuration for my custom attention"""
    custom_parameter: float = 1.0
    enable_feature: bool = True
    optimization_level: str = 'high'
```

### Step 5: Add Tests

Create tests for your component:

```python
# In tests/test_my_components.py

def test_my_custom_attention():
    """Test my custom attention component"""
    from utils.modular_components.registry import create_component
    
    attention = create_component('attention', 'my_custom_attention', {
        'd_model': 512,
        'num_heads': 8,
        'custom_parameter': 2.0
    })
    
    # Test functionality
    queries = torch.randn(2, 100, 512)
    keys = torch.randn(2, 100, 512)
    values = torch.randn(2, 100, 512)
    
    output, attn_weights = attention(queries, keys, values)
    
    assert output.shape == queries.shape
    assert attention.get_attention_type() == "my_custom_attention"
```

## Testing Framework

### Current Test Structure

```
tests/
├── test_advanced_integration.py      # 🧪 Comprehensive integration tests
├── test_my_components.py             # Your custom component tests
└── ...
```

### Main Integration Test

The comprehensive integration test (`test_advanced_integration.py`) validates:

1. **Component Registration**: All components properly registered
2. **Bayesian Integration**: KL divergence support working
3. **Advanced Attention**: Optimized mechanisms functional
4. **Specialized Processors**: Signal processing components available
5. **Existing Implementation Access**: Original implementations accessible
6. **Integration Completeness**: All advanced features working

### Running Tests

```bash
# Run comprehensive integration test
cd Time-Series-Library
python test_advanced_integration.py

# Expected output:
# 🎉 🎉 🎉 INTEGRATION COMPLETE! 🎉 🎉 🎉
# Overall: 6/6 tests passed
```

### Test Output Analysis

The test provides detailed feedback:

```
📊 Total advanced components registered: 34
📈 Components by category: {'backbone': 7, 'embedding': 4, 'attention': 9, ...}
🧠 Bayesian components: 3
⚡ Optimized components: 2
🔧 Utility processors: 6

🔍 Bayesian Integration Validation:
  ✅ bayesian_mse_registered: True
  ✅ kl_divergence_supported: True
  ✅ uncertainty_supported: True
```

### Adding New Tests

When adding new components, update the test framework:

```python
# In test_advanced_integration.py

def test_my_custom_components():
    """Test my custom components"""
    from utils.modular_components.registry import get_global_registry
    
    registry = get_global_registry()
    
    # Test registration
    assert registry.is_registered('attention', 'my_custom_attention')
    
    # Test metadata
    metadata = registry.get_metadata('attention', 'my_custom_attention')
    assert metadata.get('type') == 'custom'
    
    # Test instantiation
    component = registry.create('attention', 'my_custom_attention', MockConfig())
    assert component is not None
    
    return True

# Add to main test function
def run_comprehensive_integration_test():
    tests = [
        # ... existing tests ...
        ("My Custom Components", test_my_custom_components),
    ]
```

## Configuration System

### Configuration Files

The system supports multiple configuration approaches:

1. **YAML Configuration**
   ```yaml
   # config/enhanced_autoformer.yaml
   model:
     name: "EnhancedAutoformer"
     d_model: 512
     components:
       attention: "optimized_autocorrelation"
       loss: "bayesian_mse"
       processor: "integrated_signal"
   
   loss_config:
     kl_weight: 1e-5
     uncertainty_weight: 0.1
   ```

2. **Python Configuration**
   ```python
   # Direct configuration in code
   config = {
       'attention': {
           'type': 'adaptive_autocorrelation',
           'd_model': 512,
           'num_heads': 8,
           'scales': [1, 2, 4]
       },
       'loss': {
           'type': 'bayesian_mse',
           'kl_weight': 1e-5,
           'uncertainty_weight': 0.1
       }
   }
   ```

### Dynamic Configuration

Components can be configured and swapped at runtime:

```python
# Create model with different configurations
from utils.modular_components.registry import create_component

# Experiment 1: Standard attention + Bayesian loss
attention_1 = create_component('attention', 'multi_head', config_1)
loss_1 = create_component('loss', 'bayesian_mse', loss_config)

# Experiment 2: Optimized attention + Frequency-aware loss
attention_2 = create_component('attention', 'optimized_autocorrelation', config_2)
loss_2 = create_component('loss', 'frequency_aware', freq_config)
```

## Utilities and Helpers

### Key Utility Files

1. **`utils/bayesian_losses.py`** 🎯
   - Original Bayesian loss implementations
   - KL divergence computation
   - Uncertainty quantification
   - **Critical for proper Bayesian training**

2. **`utils/enhanced_losses.py`**
   - Advanced loss function library
   - Frequency-aware losses
   - Multi-scale trend analysis
   - Adaptive structural losses

3. **`utils/losses.py`**
   - Standard loss functions
   - DTW alignment losses
   - Patch-based structural losses

4. **`layers/AutoCorrelation_Optimized.py`** 🚀
   - Memory-optimized attention
   - Mixed precision support
   - Chunked processing for long sequences

5. **`layers/EnhancedAutoCorrelation.py`** 🔄
   - Adaptive autocorrelation mechanisms
   - Multi-scale analysis
   - Dynamic kernel size adaptation

### Helper Functions

```python
# Component discovery
from utils.modular_components.registry import list_all_components
components = list_all_components()

# Integration status
from utils.modular_components.implementations import get_integration_status
status = get_integration_status()

# Validation
from utils.modular_components.implementations import validate_critical_integrations
is_valid = validate_critical_integrations()
```

### Logging and Monitoring

The framework includes comprehensive logging:

```python
import logging
from utils.logger import logger

# Component-specific logging
logger.info("Initializing BayesianMSELoss with KL divergence")
logger.debug("Processing attention with optimized autocorrelation")
logger.warning("Fallback to standard implementation")
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```python
   # Problem: Cannot import advanced components
   # Solution: Check if dependencies are installed
   from utils.modular_components.implementations import get_integration_status
   status = get_integration_status()
   print(status)  # Shows what's available
   ```

2. **Component Not Found**
   ```python
   # Problem: Component 'my_component' not found
   # Solution: Check registration
   from utils.modular_components.registry import list_all_components
   components = list_all_components()
   print(components['attention'])  # List available attention components
   ```

3. **Bayesian Loss Issues**
   ```python
   # Problem: KL divergence not working
   # Solution: Validate Bayesian integration
   from utils.modular_components.implementations.register_advanced import validate_bayesian_integration
   validation = validate_bayesian_integration()
   print(validation)  # Shows Bayesian component status
   ```

4. **Memory Issues with Attention**
   ```python
   # Problem: Out of memory with long sequences
   # Solution: Use optimized attention
   attention = create_component('attention', 'memory_efficient', {
       'd_model': 512,
       'num_heads': 8,
       'memory_efficient': True,
       'use_checkpointing': True
   })
   ```

### Debugging Commands

```bash
# Test integration
python test_advanced_integration.py

# Check component status
python -c "from utils.modular_components.implementations import get_integration_status; print(get_integration_status())"

# Validate Bayesian support
python -c "from utils.modular_components.implementations.register_advanced import validate_bayesian_integration; print(validate_bayesian_integration())"
```

### Performance Optimization

1. **Memory Optimization**
   - Use `optimized_autocorrelation` for long sequences
   - Enable gradient checkpointing with `memory_efficient`
   - Use mixed precision training

2. **Training Optimization**
   - Use appropriate KL weights for Bayesian models (`1e-5` is recommended)
   - Monitor uncertainty regularization
   - Use frequency-aware losses for spectral patterns

3. **Component Selection**
   - `adaptive_autocorrelation` for multi-scale patterns
   - `integrated_signal` processor for comprehensive analysis
   - `bayesian_quantile` for uncertainty quantification

## Summary

The Modular HFAutoformer represents a breakthrough in time series forecasting architecture that successfully unifies specialized capabilities into a single, configurable system. This document has covered the complete architecture of our unified, component-based approach.

### Key Achievements

✅ **Unified Architecture**: One HFAutoformer instead of four separate model files  
✅ **Component Modularity**: 34 specialized components across 8 categories  
✅ **Configuration-Driven**: Same model, different behaviors through configuration  
✅ **HF Integration**: Seamless use of pre-trained Hugging Face models (Chronos, T5, BERT)  
✅ **Advanced Capabilities**: Bayesian uncertainty, quantile regression, hierarchical processing  
✅ **Extensibility**: Easy addition of new components through registry system  
✅ **Backward Compatibility**: Migration path from prototype models  

### Architecture Philosophy

The modular HFAutoformer embodies three core principles:

1. **Modularity**: Components can be mixed and matched for different use cases
2. **Configurability**: Runtime selection of capabilities without code changes  
3. **Extensibility**: New components integrate seamlessly through standardized interfaces

### Evolution Summary

```
Prototype Phase (4 Separate Models)          Unified Phase (1 Modular Model)
├── HFEnhancedAutoformer          ──────►    ├── backbone_type: 'chronos_t5'
├── HFBayesianAutoformer          ──────►    ├── loss_type: 'bayesian_kl' 
├── HFHierarchicalAutoformer      ──────►    ├── attention_type: 'hierarchical'
└── HFQuantileAutoformer          ──────►    └── processor_type: 'multi_scale'
                                                        ↓
                                             Single HFAutoformer(config)
```

### Future-Ready Design

The modular architecture positions us for future developments:

- **New HF Models**: Easy integration of future pre-trained models
- **Advanced Techniques**: Simple addition of new loss functions, attention mechanisms
- **Research Integration**: Rapid prototyping and experimentation
- **Production Scaling**: Consistent deployment patterns across different use cases

### Getting Started

```python
# Basic usage
from models.HFAutoformer import HFAutoformer

config = {
    'seq_len': 96, 'pred_len': 24, 'enc_in': 7, 'c_out': 7,
    'backbone_type': 'chronos_t5',    # HF pre-trained backbone
    'loss_type': 'mse',               # Standard loss
    'attention_type': 'standard',     # Standard attention  
    'processor_type': 'identity'      # No special processing
}

model = HFAutoformer(config)
```

The modular HFAutoformer represents not just a technical achievement, but a paradigm shift toward more flexible, maintainable, and powerful time series modeling. By building upon the base Autoformer with modular utilities, we've created a system that adapts to diverse requirements while maintaining simplicity and performance.

---

*This documentation serves as the definitive guide to the Modular HFAutoformer architecture. For implementation details, refer to the source code in `models/HFAutoformer.py` and the component implementations in `utils/modular_components/`.*
