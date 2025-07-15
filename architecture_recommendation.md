# Unified Modular Architecture for Time Series Models

## Current State Analysis
- **In-house models**: Autoformer.py + 4 Enhanced versions + 2 Fixed versions
- **HF models**: Multiple HF* models leveraging pre-trained backbones
- **Modular components**: Base interfaces, factories, implementations in utils/

## Recommended Architecture

### 1. **Unified Base Architecture**
```
models/
├── base/
│   ├── BaseTimeSeriesModel.py          # Common interface for all models
│   ├── InHouseModelBase.py             # Base for custom models
│   └── HFModelBase.py                  # Base for HF-backed models
├── inhouse/
│   ├── Autoformer.py                   # Original
│   ├── AutoformerFixed.py              # Stable version
│   ├── EnhancedAutoformer.py           # Enhanced features
│   └── variants/                       # Other enhanced versions
├── hf_models/
│   ├── HFAutoformerSuite.py            # HF-backed Autoformer
│   ├── HFBayesianAutoformer.py         # HF + Bayesian
│   └── adapters/                       # HF adaptation layers
└── factory/
    └── ModelFactory.py                 # Unified model creation
```

### 2. **Component Integration Strategy**

#### A. **Shared Component Layer**
```python
# All models use same modular components
utils/modular_components/
├── interfaces/          # BaseBackbone, BaseEmbedding, etc.
├── implementations/     # Concrete implementations
├── adapters/           # Bridge HF ↔ In-house
└── factories/          # Component creation
```

#### B. **Model Type Abstraction**
```python
class BaseTimeSeriesModel(ABC):
    def __init__(self, config):
        self.backbone = self._create_backbone(config)
        self.embedding = self._create_embedding(config)
        self.processor = self._create_processor(config)
        self.output_head = self._create_output(config)
    
    @abstractmethod
    def _create_backbone(self, config): pass
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Unified forward pass logic
        embedded = self.embedding.embed_sequence(x_enc, x_mark_enc)
        processed = self.backbone.forward(embedded)
        output = self.processor.process_sequence(processed)
        return self.output_head.generate_output(output)
```

### 3. **Backbone Abstraction**

#### A. **In-House Backbone**
```python
class AutoformerBackbone(BaseBackbone):
    def __init__(self, config):
        # Traditional Autoformer encoder/decoder
        self.encoder = Encoder([...])
        self.decoder = Decoder([...])
    
    def forward(self, x):
        return self.decoder(self.encoder(x))
```

#### B. **HF Backbone**
```python
class HFBackbone(BaseBackbone):
    def __init__(self, config):
        # Load pre-trained HF model
        self.hf_model = AutoModel.from_pretrained(config.model_name)
        self.adapter = TimeSeriesAdapter(config)
    
    def forward(self, x):
        hf_output = self.hf_model(x)
        return self.adapter.adapt_output(hf_output)
```

### 4. **Unified Factory Pattern**

```python
class UnifiedModelFactory:
    def create_model(self, model_type: str, config):
        if model_type.startswith('hf_'):
            return self._create_hf_model(model_type, config)
        else:
            return self._create_inhouse_model(model_type, config)
    
    def _create_hf_model(self, model_type, config):
        backbone = HFBackbone(config)
        return HFModelBase(backbone, config)
    
    def _create_inhouse_model(self, model_type, config):
        backbone = AutoformerBackbone(config)
        return InHouseModelBase(backbone, config)
```

### 5. **Configuration Schema**

```python
@dataclass
class UnifiedModelConfig:
    # Model type selection
    model_family: str = "autoformer"  # autoformer, transformer, etc.
    model_variant: str = "enhanced"   # base, enhanced, fixed, etc.
    backbone_type: str = "inhouse"    # inhouse, hf_chronos, hf_t5, etc.
    
    # HF-specific configs
    hf_model_name: Optional[str] = None
    use_pretrained: bool = True
    
    # Common configs
    d_model: int = 512
    seq_len: int = 96
    pred_len: int = 24
    # ... other configs
```

### 6. **Adapter Pattern for HF Integration**

```python
class HFTimeSeriesAdapter:
    """Adapts HF models for time series tasks"""
    
    def __init__(self, hf_model, ts_config):
        self.hf_model = hf_model
        self.input_adapter = InputAdapter(ts_config)
        self.output_adapter = OutputAdapter(ts_config)
    
    def forward(self, ts_data):
        # Convert time series → HF format
        hf_input = self.input_adapter.adapt_input(ts_data)
        
        # Process through HF model
        hf_output = self.hf_model(hf_input)
        
        # Convert HF output → time series format
        return self.output_adapter.adapt_output(hf_output)
```

## Implementation Strategy

### Phase 1: **Unified Base Classes**
1. Create `BaseTimeSeriesModel` interface
2. Refactor existing models to inherit from base
3. Create `InHouseModelBase` and `HFModelBase`

### Phase 2: **Component Integration**
1. Ensure all models use modular components
2. Create adapters for HF ↔ component bridge
3. Implement unified factory

### Phase 3: **Testing Integration**
1. Update test suite for unified architecture
2. Add compatibility tests between model types
3. Performance benchmarking

## Benefits

1. **Unified Interface**: Same API for all model types
2. **Component Reuse**: Share embeddings, processors, outputs
3. **Easy Extension**: Add new models by implementing base interface
4. **HF Integration**: Seamless use of pre-trained models
5. **Maintainability**: Clear separation of concerns
6. **Testing**: Single test framework for all models

## Migration Path

1. **Keep existing models working** during transition
2. **Gradual migration** to unified architecture
3. **Backward compatibility** through factory patterns
4. **Progressive enhancement** of modular components

This architecture allows you to:
- Keep your existing in-house models
- Integrate HF models seamlessly  
- Share components between model types
- Maintain clean separation between model families
- Scale to new model types easily