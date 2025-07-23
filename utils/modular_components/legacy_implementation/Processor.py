"""
Unified Processor Implementations for Modular Framework
"""
from utils.modular_components.base_interfaces import BaseComponent
from utils.modular_components.config_schemas import ProcessorConfig
from typing import Any

PROCESSOR_REGISTRY = {}

def register_processor(name: str, cls: type):
    PROCESSOR_REGISTRY[name] = cls

def create_processor(name: str, config: ProcessorConfig, **kwargs) -> BaseComponent:
    if name not in PROCESSOR_REGISTRY:
        raise ValueError(f"Processor '{name}' not found in registry.")
    return PROCESSOR_REGISTRY[name](config, **kwargs)

# Example processor implementations
class Seq2SeqProcessor(BaseComponent):
    def __init__(self, config: ProcessorConfig, **kwargs):
        super().__init__()
        self.config = config
    def forward(self, x: Any) -> Any:
        return x

class EncoderOnlyProcessor(BaseComponent):
    def __init__(self, config: ProcessorConfig, **kwargs):
        super().__init__()
        self.config = config
    def forward(self, x: Any) -> Any:
        return x

class HierarchicalProcessor(BaseComponent):
    """
    Hierarchical processing for modular framework, adapted from HFHierarchicalExtensions.
    Provides multi-scale temporal modeling for time series.
    """
    def __init__(self, config: ProcessorConfig, **kwargs):
        super().__init__()
        self.config = config
        # Hierarchical configuration
        self.hierarchy_levels = getattr(config, 'hierarchy_levels', [1, 2, 4])
        self.aggregation_method = getattr(config, 'aggregation_method', 'adaptive')
        self.level_weights_learnable = getattr(config, 'level_weights_learnable', True)
        # Model dimensions
        self.seq_len = getattr(config, 'seq_len', 96)
        self.pred_len = getattr(config, 'pred_len', 24)
        self.enc_in = getattr(config, 'enc_in', 1)
        self.d_model = getattr(config, 'd_model', 512)
        # Level processors
        self._init_level_processors()
        # Aggregation mechanism
        self._init_aggregation()

    def _init_level_processors(self):
        """Initialize processors for each hierarchy level"""
        import torch.nn as nn
        self.level_processors = nn.ModuleDict()
        for level in self.hierarchy_levels:
            self.level_processors[f'level_{level}'] = nn.Sequential(
                nn.Linear(self.enc_in, self.d_model),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.d_model, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.enc_in)
            )

    def _init_aggregation(self):
        import torch
        import torch.nn as nn
        n_levels = len(self.hierarchy_levels)
        if self.aggregation_method == 'adaptive':
            self.level_weights = nn.Parameter(torch.ones(n_levels) / n_levels)
            self.level_attention = nn.Sequential(
                nn.Linear(self.enc_in * n_levels, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, n_levels),
                nn.Softmax(dim=-1)
            )
        elif self.aggregation_method == 'concat':
            self.concat_projection = nn.Linear(self.enc_in * n_levels, self.enc_in)
        elif self.aggregation_method == 'residual':
            self.residual_projections = nn.ModuleList([
                nn.Linear(self.enc_in, self.enc_in) for _ in range(n_levels - 1)
            ])

    def forward(self, x, target_level=None):
        import torch
        import torch.nn.functional as F
        if target_level is not None:
            return self._single_level_forward(x, target_level)
        else:
            return self._multi_level_forward(x)

    def _single_level_forward(self, x, level):
        downsampled = self._downsample(x, level)
        processor_name = f'level_{level}'
        if processor_name in self.level_processors:
            processed = self.level_processors[processor_name](downsampled)
        else:
            processed = downsampled
        upsampled = self._upsample(processed, level, x.shape[1])
        return upsampled

    def _multi_level_forward(self, x):
        import torch
        import torch.nn.functional as F
        level_outputs = []
        for level in self.hierarchy_levels:
            level_output = self._single_level_forward(x, level)
            level_outputs.append(level_output)
        if self.aggregation_method == 'adaptive':
            return self._adaptive_aggregation(level_outputs, x)
        elif self.aggregation_method == 'concat':
            return self._concat_aggregation(level_outputs)
        elif self.aggregation_method == 'residual':
            return self._residual_aggregation(level_outputs, x)
        else:
            weights = F.softmax(self.level_weights, dim=0)
            aggregated = sum(w * output for w, output in zip(weights, level_outputs))
            return aggregated

    def _downsample(self, x, factor):
        if factor == 1:
            return x
        downsampled = x[:, ::factor, :]
        if downsampled.shape[1] < 1:
            downsampled = x[:, :1, :]
        return downsampled

    def _upsample(self, x, factor, target_length):
        import torch.nn.functional as F
        if factor == 1 and x.shape[1] == target_length:
            return x
        x_permuted = x.permute(0, 2, 1)
        upsampled = F.interpolate(x_permuted, size=target_length, mode='linear', align_corners=False)
        upsampled = upsampled.permute(0, 2, 1)
        return upsampled

    def _adaptive_aggregation(self, level_outputs, original_x):
        import torch
        concat_outputs = torch.cat(level_outputs, dim=-1)
        attention_weights = self.level_attention(concat_outputs)
        aggregated = torch.zeros_like(level_outputs[0])
        for i, level_output in enumerate(level_outputs):
            weight = attention_weights[:, :, i:i+1]
            aggregated += weight * level_output
        return aggregated

    def _concat_aggregation(self, level_outputs):
        import torch
        concatenated = torch.cat(level_outputs, dim=-1)
        projected = self.concat_projection(concatenated)
        return projected

    def _residual_aggregation(self, level_outputs, original_x):
        result = level_outputs[0]
        for i in range(1, len(level_outputs)):
            residual = self.residual_projections[i-1](result)
            result = residual + level_outputs[i]
        return result

    def get_hierarchical_info(self):
        return {
            'hierarchy_levels': self.hierarchy_levels,
            'aggregation_method': self.aggregation_method,
            'level_weights_learnable': self.level_weights_learnable,
            'n_level_processors': len(self.level_processors)
        }

class AutoregressiveProcessor(BaseComponent):
    def __init__(self, config: ProcessorConfig, **kwargs):
        super().__init__()
        self.config = config
    def forward(self, x: Any) -> Any:
        return x

# Register all processor types
register_processor('seq2seq', Seq2SeqProcessor)
register_processor('encoder_only', EncoderOnlyProcessor)
register_processor('hierarchical', HierarchicalProcessor)
register_processor('autoregressive', AutoregressiveProcessor)
