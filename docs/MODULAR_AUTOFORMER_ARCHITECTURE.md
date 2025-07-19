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

The framework consists of several component types, each serving a specific purpose in the autoformer architecture:

### 1. Attention Components

**Purpose**: Handle attention mechanisms for temporal dependencies

**Available Types**:
- `AUTOCORRELATION`: Standard autocorrelation attention
- `ADAPTIVE_AUTOCORRELATION`: Adaptive autocorrelation with dynamic factors
- `CROSS_RESOLUTION`: Multi-resolution attention for hierarchical processing

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

**Available Types**:
- `MOVING_AVG`: Moving average decomposition
- `LEARNABLE_DECOMP`: Learnable decomposition with trainable parameters
- `WAVELET_DECOMP`: Wavelet-based decomposition for hierarchical analysis

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

**Available Types**:
- `STANDARD_ENCODER`: Basic transformer encoder
- `ENHANCED_ENCODER`: Enhanced encoder with adaptive features
- `HIERARCHICAL_ENCODER`: Multi-level hierarchical encoder

### 4. Decoder Components

**Purpose**: Decode and generate predictions

**Available Types**:
- `STANDARD_DECODER`: Basic transformer decoder
- `ENHANCED_DECODER`: Enhanced decoder with adaptive features

### 5. Sampling Components

**Purpose**: Handle uncertainty quantification and sampling strategies

**Available Types**:
- `DETERMINISTIC`: Standard deterministic prediction
- `BAYESIAN`: Bayesian sampling for uncertainty quantification

### 6. Output Head Components

**Purpose**: Final projection to output dimensions

**Available Types**:
- `STANDARD_HEAD`: Basic linear projection
- `QUANTILE`: Multi-quantile prediction head

### 7. Loss Components

**Purpose**: Loss function computation

**Available Types**:
- `MSE`: Mean squared error
- `BAYESIAN`: Bayesian loss with KL divergence

## HF Integration Framework

The framework seamlessly integrates HuggingFace autoformer models through a unified interface that abstracts the differences between custom and HF implementations.

### HF Model Types

**Location**: `models/HFAutoformerSuite.py`, `models/unified_autoformer_factory.py`

1. **HFEnhancedAutoformer**: Basic HF enhanced autoformer
2. **HFBayesianAutoformer**: HF Bayesian autoformer with uncertainty
3. **HFHierarchicalAutoformer**: HF hierarchical autoformer
4. **HFQuantileAutoformer**: HF quantile autoformer
5. **HFEnhancedAutoformerAdvanced**: Advanced HF enhanced model
6. **HFBayesianAutoformerProduction**: Production-ready HF Bayesian model

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
