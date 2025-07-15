import torch
import torch.nn as nn
from typing import Dict, Any


def apply_gradient_clipping(model: nn.Module, max_norm: float = 1.0):
    """Apply gradient clipping to prevent exploding gradients"""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def initialize_weights(model: nn.Module):
    """Initialize model weights properly"""
    for name, param in model.named_parameters():
        if 'weight' in name:
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.uniform_(param, -0.1, 0.1)
        elif 'bias' in name:
            nn.init.constant_(param, 0)


def validate_model_inputs(x_enc: torch.Tensor, expected_features: int):
    """Validate model inputs"""
    if x_enc.dim() != 3:
        raise ValueError(f"Expected 3D input tensor, got {x_enc.dim()}D")
    
    if x_enc.size(-1) != expected_features:
        raise ValueError(f"Expected {expected_features} features, got {x_enc.size(-1)}")
    
    if torch.isnan(x_enc).any():
        raise ValueError("Input contains NaN values")
    
    if torch.isinf(x_enc).any():
        raise ValueError("Input contains infinite values")


class ModelProfiler:
    """Simple model profiler for performance monitoring"""
    
    def __init__(self):
        self.times = {}
        self.memory = {}
    
    def start_timer(self, name: str):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.times[name] = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if self.times[name]:
            self.times[name].record()
    
    def end_timer(self, name: str) -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            if name in self.times and self.times[name]:
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()
                torch.cuda.synchronize()
                elapsed = self.times[name].elapsed_time(end_event)
                return elapsed
        return 0.0
    
    def get_memory_usage(self) -> Dict[str, float]:
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
                'cached': torch.cuda.memory_reserved() / 1024**2,      # MB
            }
        return {'allocated': 0.0, 'cached': 0.0}