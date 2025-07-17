# Modular Autoformer Architecture: Component Flow and Extension Guide

## Overview

In our modular design, encoder and decoder functionalities are handled through a component registry system that allows dynamic loading and composition of different architectural components. This guide explains how the system works and where to add new functionality like Mixture of Experts (MoE).

## Current Architecture Flow

### 1. Component Registration and Discovery

```
ModularAutoformer
    ↓
ConfigurationEngine (loads YAML config)
    ↓
ComponentRegistry (manages all components)
    ↓
Component Factories (creates instances)
    ↓
Assembled Model (composed from components)
```

### 2. Component Types in Registry

Based on the existing registry structure:

```python
registry._components = {
    'backbone': {},      # Core model architectures
    'embedding': {},     # Input embedding layers
    'attention': {},     # Attention mechanisms (AutoCorrelation, etc.)
    'processor': {},     # Processing layers (could include encoder/decoder)
    'feedforward': {},   # FFN layers
    'loss': {},         # Loss functions
    'output': {},       # Output projections
    'adapter': {}       # Adaptation layers
}
```

## Encoder/Decoder Component Flow

### Current Implementation Structure

Based on the analysis, here's how encoder/decoder components flow in the modular design:

```
┌─────────────────────────────────────────────────────────────┐
│                    ModularAutoformer                        │
├─────────────────────────────────────────────────────────────┤
│  1. Configuration Loading (YAML)                           │
│  2. Component Resolution (Registry Lookup)                 │
│  3. Component Assembly (Factory Creation)                  │
│  4. Forward Pass Orchestration                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Component Assembly                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Decomposition│  │ Attention   │  │ Embedding   │        │
│  │ Component   │  │ Component   │  │ Component   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Encoder     │  │ Decoder     │  │ Sampling    │        │
│  │ Component   │  │ Component   │  │ Component   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## Adding Mixture of Experts (MoE) to Decoder

### File Structure for New MoE Component

To add MoE functionality to your decoder, you would add files in this structure:

```
utils/modular_components/implementations/
├── feedforward.py (existing - extend this)
├── processors.py (existing - extend this)
└── mixture_of_experts.py (NEW - create this)

components/decoder/ (NEW directory structure from migration plan)
├── decoder_registry.py
├── standard_decoder.py
├── enhanced_decoder.py
└── moe_decoder.py (NEW - your MoE implementation)
```

### Step-by-Step Implementation Guide

#### 1. Create the MoE Component (`utils/modular_components/implementations/mixture_of_experts.py`)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_interfaces import BaseComponent, ComponentType
from typing import Dict, Any, Optional

class MixtureOfExpertsFFN(BaseComponent):
    """
    Mixture of Experts Feed-Forward Network for enhanced decoder capability
    """
    
    component_type = ComponentType.FEEDFORWARD
    
    def __init__(self, 
                 d_model: int,
                 d_ff: int,
                 num_experts: int = 8,
                 top_k: int = 2,
                 dropout: float = 0.1,
                 activation: str = "relu",
                 **kwargs):
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router/Gating network
        self.router = nn.Linear(d_model, num_experts)
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU() if activation == "relu" else nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout)
            ) for _ in range(num_experts)
        ])
        
        # Load balancing loss coefficient
        self.load_balance_coeff = 0.01
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # [batch_size * seq_len, d_model]
        
        # Router computation
        router_logits = self.router(x_flat)  # [batch_size * seq_len, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-k expert selection
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = F.softmax(top_k_probs, dim=-1)  # Renormalize
        
        # Expert computation
        output = torch.zeros_like(x_flat)
        
        for i in range(self.top_k):
            expert_indices = top_k_indices[:, i]
            expert_probs = top_k_probs[:, i].unsqueeze(-1)
            
            for expert_idx in range(self.num_experts):
                mask = (expert_indices == expert_idx)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[expert_idx](expert_input)
                    output[mask] += expert_probs[mask] * expert_output
        
        # Reshape back
        output = output.view(batch_size, seq_len, d_model)
        
        # Load balancing loss (for training)
        if self.training:
            self.load_balance_loss = self._compute_load_balance_loss(router_probs)
        
        return output
    
    def _compute_load_balance_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        """Compute load balancing loss to encourage equal expert utilization"""
        expert_usage = router_probs.mean(dim=0)  # Average usage per expert
        ideal_usage = 1.0 / self.num_experts
        return self.load_balance_coeff * torch.sum((expert_usage - ideal_usage) ** 2)

    def get_config(self) -> Dict[str, Any]:
        return {
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'num_experts': self.num_experts,
            'top_k': self.top_k
        }

# Register the component
def register_moe_components(registry):
    """Register MoE components with the registry"""
    registry.register(
        component_type='feedforward',
        component_name='mixture_of_experts',
        component_class=MixtureOfExpertsFFN,
        metadata={
            'description': 'Mixture of Experts Feed-Forward Network',
            'parameters': ['d_model', 'd_ff', 'num_experts', 'top_k'],
            'use_cases': ['large_models', 'capacity_scaling', 'specialization']
        }
    )
```

#### 2. Create Enhanced Decoder with MoE (`components/decoder/moe_decoder.py`)

```python
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from ..base_interfaces import BaseComponent, ComponentType
from utils.modular_components.registry import ComponentRegistry

class MoEDecoderLayer(BaseComponent):
    """
    Enhanced Autoformer decoder layer with Mixture of Experts
    """
    
    component_type = ComponentType.PROCESSOR
    
    def __init__(self,
                 d_model: int,
                 c_out: int,
                 attention_config: Dict[str, Any],
                 moe_config: Dict[str, Any],
                 decomposition_config: Dict[str, Any],
                 dropout: float = 0.1,
                 registry: Optional[ComponentRegistry] = None,
                 **kwargs):
        super().__init__()
        
        self.d_model = d_model
        self.c_out = c_out
        
        # Get components from registry
        if registry is None:
            from utils.modular_components import get_global_registry
            registry = get_global_registry()
        
        # Self and cross attention
        attention_class = registry.get('attention', attention_config['type'])
        self.self_attention = attention_class(**attention_config['params'])
        self.cross_attention = attention_class(**attention_config['params'])
        
        # MoE Feed-forward network
        moe_class = registry.get('feedforward', moe_config['type'])
        self.moe_ffn = moe_class(d_model=d_model, **moe_config['params'])
        
        # Series decomposition
        decomp_class = registry.get('processor', decomposition_config['type'])
        self.decomp1 = decomp_class(**decomposition_config['params'])
        self.decomp2 = decomp_class(**decomposition_config['params'])
        self.decomp3 = decomp_class(**decomposition_config['params'])
        
        # Projection layer
        self.projection = nn.Conv1d(
            in_channels=d_model,
            out_channels=c_out,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='circular',
            bias=False
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # Self attention
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x, trend1 = self.decomp1(x)
        
        # Cross attention
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        x, trend2 = self.decomp2(x)
        
        # MoE Feed-forward
        moe_output = self.moe_ffn(x)
        x, trend3 = self.decomp3(x + moe_output)
        
        # Trend aggregation and projection
        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        
        return x, residual_trend

class MoEDecoder(BaseComponent):
    """
    Complete decoder with MoE layers
    """
    
    component_type = ComponentType.PROCESSOR
    
    def __init__(self, 
                 layers_config: Dict[str, Any],
                 norm_layer_config: Optional[Dict[str, Any]] = None,
                 **kwargs):
        super().__init__()
        
        # Create decoder layers
        self.layers = nn.ModuleList([
            MoEDecoderLayer(**layer_config) 
            for layer_config in layers_config
        ])
        
        # Optional normalization
        if norm_layer_config:
            norm_class = registry.get('processor', norm_layer_config['type'])
            self.norm = norm_class(**norm_layer_config['params'])
        else:
            self.norm = None
    
    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            if trend is not None:
                trend = trend + residual_trend
            else:
                trend = residual_trend
        
        if self.norm is not None:
            x = self.norm(x)
        
        return x, trend

# Registration function
def register_moe_decoder_components(registry):
    """Register MoE decoder components"""
    registry.register(
        component_type='processor',
        component_name='moe_decoder_layer',
        component_class=MoEDecoderLayer,
        metadata={
            'description': 'Decoder layer with Mixture of Experts',
            'parameters': ['d_model', 'c_out', 'attention_config', 'moe_config'],
            'use_cases': ['large_scale_forecasting', 'multi_domain_adaptation']
        }
    )
    
    registry.register(
        component_type='processor',
        component_name='moe_decoder',
        component_class=MoEDecoder,
        metadata={
            'description': 'Complete decoder with MoE capability',
            'parameters': ['layers_config', 'norm_layer_config'],
            'use_cases': ['complex_forecasting', 'multi_task_learning']
        }
    )
```

#### 3. Configuration File (`configs/autoformer/moe_enhanced.yaml`)

```yaml
# configs/autoformer/moe_enhanced.yaml
model_type: "ModularAutoformer"
components:
  decomposition:
    type: "LearnableDecomposition"
    params:
      d_model: 512
      adaptive_kernel: true
  
  attention:
    type: "EnhancedAutoCorrelation"
    params:
      factor: 1
      attention_dropout: 0.1
  
  encoder:
    type: "EnhancedEncoder"
    params:
      layers: 2
      d_model: 512
  
  decoder:
    type: "moe_decoder"
    params:
      layers_config:
        - d_model: 512
          c_out: 7  # number of output features
          attention_config:
            type: "EnhancedAutoCorrelation"
            params:
              factor: 1
              attention_dropout: 0.1
          moe_config:
            type: "mixture_of_experts"
            params:
              d_ff: 2048
              num_experts: 8
              top_k: 2
              dropout: 0.1
          decomposition_config:
            type: "LearnableDecomposition"
            params:
              d_model: 512
        - d_model: 512
          c_out: 7
          attention_config:
            type: "EnhancedAutoCorrelation"
            params:
              factor: 1
              attention_dropout: 0.1
          moe_config:
            type: "mixture_of_experts"
            params:
              d_ff: 2048
              num_experts: 8
              top_k: 2
              dropout: 0.1
          decomposition_config:
            type: "LearnableDecomposition"
            params:
              d_model: 512
  
  sampling:
    type: "DeterministicSampling"

# Training configuration
training:
  include_moe_losses: true  # Include load balancing losses
  moe_loss_weight: 0.01
```

#### 4. Registration and Usage

Add to your component registration (`utils/modular_components/implementations/__init__.py`):

```python
# Add to existing registrations
def register_all_components(registry):
    # ... existing registrations ...
    
    # Register MoE components
    from .mixture_of_experts import register_moe_components
    from ..implementations.moe_decoder import register_moe_decoder_components
    
    register_moe_components(registry)
    register_moe_decoder_components(registry)
```

#### 5. Usage in Model

```python
# Example usage in training script
from utils.modular_components import get_global_registry
from models.modular_autoformer import ModularAutoformer

# Load configuration
config = yaml.load(open('configs/autoformer/moe_enhanced.yaml'))

# Create model with MoE decoder
model = ModularAutoformer(config)

# Training loop with MoE losses
for batch in dataloader:
    output = model(batch)
    
    # Get standard loss
    loss = criterion(output, target)
    
    # Add MoE load balancing losses
    if config['training']['include_moe_losses']:
        moe_loss = 0
        for name, module in model.named_modules():
            if hasattr(module, 'load_balance_loss'):
                moe_loss += module.load_balance_loss
        
        loss += config['training']['moe_loss_weight'] * moe_loss
    
    loss.backward()
    optimizer.step()
```

## Summary

To add Mixture of Experts to your decoder:

1. **Create the MoE component** in `utils/modular_components/implementations/mixture_of_experts.py`
2. **Create the enhanced decoder** in `components/decoder/moe_decoder.py` (following migration plan)
3. **Register components** in the component registry
4. **Create configuration** in `configs/autoformer/moe_enhanced.yaml`
5. **Use via configuration** - no code changes needed in the main model

This modular approach allows you to:
- Mix and match components easily
- Configure different MoE setups via YAML
- Combine with other enhancements (Bayesian, Hierarchical, etc.)
- Test different expert configurations without code changes
- Maintain backward compatibility with existing models

The key is that **all functionality is added through the component registry system** - you don't modify existing model files, you create new components and configure them via YAML.
