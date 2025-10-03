"""Memory-efficient gradient tracker with bounded history."""

from __future__ import annotations
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple
import torch

@dataclass
class BoundedParameterStats:
    """Memory-efficient parameter statistics with circular buffer."""
    
    max_history: int = 100
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = 0.0
    last: float = 0.0
    history: deque = field(default_factory=lambda: deque(maxlen=100))

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        self.last = value
        self.history.append(value)  # Automatically bounded by deque

    def variance(self) -> float:
        if self.count <= 1:
            return 0.0
        return self.m2 / (self.count - 1)

@dataclass
class MemoryEfficientGradientTracker:
    """Memory-efficient gradient tracker with automatic cleanup."""
    
    max_global_norms: int = 1000
    max_memory_history: int = 500
    cleanup_interval: int = 100
    
    def __init__(self, max_global_norms: int = 1000):
        self.max_global_norms = max_global_norms
        self.global_norms = deque(maxlen=max_global_norms)
        self.per_parameter: Dict[str, BoundedParameterStats] = {}
        self.rss_history_mb = deque(maxlen=self.max_memory_history)
        self.cuda_allocated_mb = deque(maxlen=self.max_memory_history)
        self.step_count = 0
        self.vanishing_threshold = 1e-8
        self.vanishing_steps: List[int] = []

    def log_step(self, named_parameters: Iterable[Tuple[str, torch.nn.Parameter]]) -> None:
        """Log gradient step with memory efficiency."""
        total = 0.0
        param_count = 0
        
        for name, param in named_parameters:
            if param.grad is None:
                continue
            
            norm = param.grad.norm(p=2).item()
            total += norm * norm
            param_count += 1
            
            # Only track top parameters to limit memory
            if param_count <= 50:  # Limit tracked parameters
                if name not in self.per_parameter:
                    self.per_parameter[name] = BoundedParameterStats()
                self.per_parameter[name].update(norm)

        if total == 0.0:
            self.vanishing_steps.append(self.step_count)
            self.global_norms.append(0.0)
        else:
            global_norm = math.sqrt(total)
            if global_norm < self.vanishing_threshold:
                self.vanishing_steps.append(self.step_count)
            self.global_norms.append(global_norm)

        self.step_count += 1
        
        # Periodic cleanup
        if self.step_count % self.cleanup_interval == 0:
            self._cleanup_memory()

    def _cleanup_memory(self) -> None:
        """Periodic memory cleanup."""
        # Limit vanishing steps history
        if len(self.vanishing_steps) > 100:
            self.vanishing_steps = self.vanishing_steps[-50:]
        
        # Remove least important parameters if too many
        if len(self.per_parameter) > 100:
            # Keep only parameters with highest variance (most informative)
            sorted_params = sorted(
                self.per_parameter.items(),
                key=lambda x: x[1].variance(),
                reverse=True
            )
            self.per_parameter = dict(sorted_params[:50])

    def get_memory_usage_mb(self) -> float:
        """Estimate current memory usage of tracker."""
        # Rough estimation
        global_norms_mb = len(self.global_norms) * 8 / (1024 * 1024)  # 8 bytes per float
        per_param_mb = len(self.per_parameter) * 0.001  # Rough estimate
        return global_norms_mb + per_param_mb