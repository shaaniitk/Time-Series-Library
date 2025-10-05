#!/usr/bin/env python3
"""
Layer-wise Gradient Tracker

Instead of tracking 8M+ individual parameters, this tracks gradients at the layer level:
- Computes scalar metrics (norm, mean, std) for each layer's gradients
- Tracks gradient flow between layers
- Identifies vanishing/exploding gradient issues
- Much more memory efficient and interpretable

Key Metrics:
- Layer gradient norm (L2 norm of all parameters in layer)
- Layer gradient mean/std
- Gradient flow ratio between consecutive layers
- Vanishing gradient detection (norm < threshold)
- Exploding gradient detection (norm > threshold)
"""

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class LayerGradientStats:
    """Statistics for a single layer's gradients over time."""
    
    name: str
    parameter_count: int = 0
    
    # Running statistics
    norm_history: deque = field(init=False)
    mean_history: deque = field(init=False)
    std_history: deque = field(init=False)
    
    # Summary statistics
    norm_min: float = float('inf')
    norm_max: float = 0.0
    norm_sum: float = 0.0
    count: int = 0
    
    # Gradient health indicators
    vanishing_count: int = 0
    exploding_count: int = 0
    
    def __post_init__(self):
        self.norm_history = deque(maxlen=1000)  # Keep last 1000 values
        self.mean_history = deque(maxlen=1000)
        self.std_history = deque(maxlen=1000)
    
    def update(self, gradients: List[torch.Tensor], vanishing_threshold: float = 1e-7, exploding_threshold: float = 10.0):
        """Update statistics with new gradient information."""
        if not gradients:
            return
        
        # Concatenate all gradients in this layer
        all_grads = torch.cat([g.flatten() for g in gradients if g is not None])
        
        if len(all_grads) == 0:
            return
        
        # Compute metrics
        norm = all_grads.norm(p=2).item()
        mean = all_grads.mean().item()
        std = all_grads.std().item()
        
        # Update histories
        self.norm_history.append(norm)
        self.mean_history.append(mean)
        self.std_history.append(std)
        
        # Update summary statistics
        self.norm_min = min(self.norm_min, norm)
        self.norm_max = max(self.norm_max, norm)
        self.norm_sum += norm
        self.count += 1
        
        # Check for gradient health issues
        if norm < vanishing_threshold:
            self.vanishing_count += 1
        elif norm > exploding_threshold:
            self.exploding_count += 1
    
    def get_summary(self) -> Dict:
        """Get summary statistics for this layer."""
        if self.count == 0:
            return {"name": self.name, "parameter_count": self.parameter_count, "no_data": True}
        
        norm_mean = self.norm_sum / self.count if self.count > 0 else 0.0
        
        return {
            "name": self.name,
            "parameter_count": self.parameter_count,
            "gradient_norm": {
                "min": float(self.norm_min) if self.norm_min != float('inf') else 0.0,
                "max": float(self.norm_max),
                "mean": float(norm_mean),
                "last": float(self.norm_history[-1]) if self.norm_history else 0.0,
            },
            "gradient_health": {
                "vanishing_episodes": self.vanishing_count,
                "exploding_episodes": self.exploding_count,
                "vanishing_rate": self.vanishing_count / self.count if self.count > 0 else 0.0,
                "exploding_rate": self.exploding_count / self.count if self.count > 0 else 0.0,
            },
            "updates": self.count,
        }


@dataclass
class LayerWiseGradientTracker:
    """
    Efficient layer-wise gradient tracking system.
    
    Instead of tracking 8M+ parameters individually, this groups parameters by layer
    and computes meaningful scalar metrics for each layer.
    """
    
    # Configuration
    vanishing_threshold: float = 1e-7
    exploding_threshold: float = 10.0
    tracking_frequency: int = 1  # Track every N steps (can be 1 since it's efficient)
    max_history: int = 1000
    
    # State
    step_count: int = 0
    layer_stats: Dict[str, LayerGradientStats] = field(default_factory=dict)
    global_gradient_norm: deque = field(init=False)
    gradient_flow_ratios: Dict[str, deque] = field(init=False)
    
    # Layer organization
    layer_groups: Dict[str, List[str]] = field(init=False)
    layer_order: List[str] = field(init=False)
    
    def __post_init__(self):
        self.global_gradient_norm = deque(maxlen=self.max_history)
        self.gradient_flow_ratios = defaultdict(lambda: deque(maxlen=self.max_history))
        self.layer_groups = {}
        self.layer_order = []
    
    def initialize_from_model(self, model: nn.Module):
        """Initialize layer tracking from model structure."""
        print("🔍 Analyzing model structure for layer-wise gradient tracking...")
        
        # Group parameters by layer/module
        layer_params = defaultdict(list)
        layer_param_counts = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Extract layer name (everything before the last dot)
                if '.' in name:
                    layer_name = '.'.join(name.split('.')[:-1])
                else:
                    layer_name = name
                
                layer_params[layer_name].append(name)
                
                if layer_name not in layer_param_counts:
                    layer_param_counts[layer_name] = 0
                layer_param_counts[layer_name] += param.numel()
        
        # Create layer statistics objects
        for layer_name, param_names in layer_params.items():
            self.layer_stats[layer_name] = LayerGradientStats(
                name=layer_name,
                parameter_count=layer_param_counts[layer_name]
            )
            self.layer_groups[layer_name] = param_names
        
        # Create a logical order for layers (for gradient flow analysis)
        self.layer_order = sorted(layer_params.keys())
        
        print(f"✅ Layer-wise gradient tracking initialized:")
        print(f"   - Total layers: {len(self.layer_stats)}")
        print(f"   - Total parameters: {sum(layer_param_counts.values()):,}")
        print(f"   - Tracking frequency: every {self.tracking_frequency} steps")
        
        # Print layer summary
        for layer_name in self.layer_order[:10]:  # Show first 10 layers
            count = layer_param_counts[layer_name]
            print(f"     {layer_name}: {count:,} parameters")
        
        if len(self.layer_order) > 10:
            print(f"     ... and {len(self.layer_order) - 10} more layers")
    
    def should_track_this_step(self) -> bool:
        """Determine if we should track gradients on this step."""
        return self.step_count % self.tracking_frequency == 0
    
    def track_gradients(self, model: nn.Module):
        """Track gradients for all layers in the model."""
        self.step_count += 1
        
        if not self.should_track_this_step():
            return
        
        # Collect gradients by layer
        layer_gradients = defaultdict(list)
        global_norm_squared = 0.0
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Determine layer name
                if '.' in name:
                    layer_name = '.'.join(name.split('.')[:-1])
                else:
                    layer_name = name
                
                layer_gradients[layer_name].append(param.grad)
                global_norm_squared += param.grad.norm(p=2).item() ** 2
        
        # Update layer statistics
        layer_norms = {}
        for layer_name, gradients in layer_gradients.items():
            if layer_name in self.layer_stats:
                self.layer_stats[layer_name].update(
                    gradients, 
                    self.vanishing_threshold, 
                    self.exploding_threshold
                )
                
                # Store layer norm for gradient flow analysis
                if gradients:
                    all_grads = torch.cat([g.flatten() for g in gradients])
                    layer_norms[layer_name] = all_grads.norm(p=2).item()
        
        # Update global gradient norm
        global_norm = math.sqrt(global_norm_squared) if global_norm_squared > 0 else 0.0
        self.global_gradient_norm.append(global_norm)
        
        # Compute gradient flow ratios between consecutive layers
        self._compute_gradient_flow_ratios(layer_norms)
    
    def _compute_gradient_flow_ratios(self, layer_norms: Dict[str, float]):
        """Compute gradient flow ratios between consecutive layers."""
        for i in range(len(self.layer_order) - 1):
            current_layer = self.layer_order[i]
            next_layer = self.layer_order[i + 1]
            
            if current_layer in layer_norms and next_layer in layer_norms:
                current_norm = layer_norms[current_layer]
                next_norm = layer_norms[next_layer]
                
                if current_norm > 0:
                    ratio = next_norm / current_norm
                    flow_key = f"{current_layer} -> {next_layer}"
                    self.gradient_flow_ratios[flow_key].append(ratio)
    
    def get_summary(self) -> Dict:
        """Get comprehensive summary of gradient tracking results."""
        summary = {
            "tracking_info": {
                "total_steps": self.step_count,
                "tracked_steps": len(self.global_gradient_norm),
                "tracking_frequency": self.tracking_frequency,
                "total_layers": len(self.layer_stats),
            },
            "global_gradient_norm": {},
            "layer_statistics": {},
            "gradient_flow": {},
            "gradient_health": {
                "layers_with_vanishing": 0,
                "layers_with_exploding": 0,
                "total_vanishing_episodes": 0,
                "total_exploding_episodes": 0,
            }
        }
        
        # Global gradient norm statistics
        if self.global_gradient_norm:
            norms = list(self.global_gradient_norm)
            summary["global_gradient_norm"] = {
                "min": float(min(norms)),
                "max": float(max(norms)),
                "mean": float(sum(norms) / len(norms)),
                "last": float(norms[-1]),
            }
        
        # Layer-wise statistics
        for layer_name, stats in self.layer_stats.items():
            layer_summary = stats.get_summary()
            summary["layer_statistics"][layer_name] = layer_summary
            
            # Update health counters
            if not layer_summary.get("no_data", False):
                health = layer_summary["gradient_health"]
                if health["vanishing_episodes"] > 0:
                    summary["gradient_health"]["layers_with_vanishing"] += 1
                if health["exploding_episodes"] > 0:
                    summary["gradient_health"]["layers_with_exploding"] += 1
                
                summary["gradient_health"]["total_vanishing_episodes"] += health["vanishing_episodes"]
                summary["gradient_health"]["total_exploding_episodes"] += health["exploding_episodes"]
        
        # Gradient flow statistics
        for flow_key, ratios in self.gradient_flow_ratios.items():
            if ratios:
                ratio_list = list(ratios)
                summary["gradient_flow"][flow_key] = {
                    "mean_ratio": float(sum(ratio_list) / len(ratio_list)),
                    "min_ratio": float(min(ratio_list)),
                    "max_ratio": float(max(ratio_list)),
                    "last_ratio": float(ratio_list[-1]),
                    "samples": len(ratio_list),
                }
        
        return summary
    
    def get_problematic_layers(self) -> Dict[str, List[str]]:
        """Identify layers with gradient problems."""
        problems = {
            "vanishing_gradients": [],
            "exploding_gradients": [],
            "no_gradients": [],
        }
        
        for layer_name, stats in self.layer_stats.items():
            summary = stats.get_summary()
            
            if summary.get("no_data", False):
                problems["no_gradients"].append(layer_name)
                continue
            
            health = summary["gradient_health"]
            
            # Consider a layer problematic if >10% of updates have issues
            if health["vanishing_rate"] > 0.1:
                problems["vanishing_gradients"].append(layer_name)
            
            if health["exploding_rate"] > 0.1:
                problems["exploding_gradients"].append(layer_name)
        
        return problems
    
    def print_summary(self):
        """Print a human-readable summary of gradient tracking results."""
        summary = self.get_summary()
        problems = self.get_problematic_layers()
        
        print("\n" + "="*60)
        print("📊 LAYER-WISE GRADIENT TRACKING SUMMARY")
        print("="*60)
        
        # Basic info
        info = summary["tracking_info"]
        print(f"Total Steps: {info['total_steps']}")
        print(f"Tracked Steps: {info['tracked_steps']}")
        print(f"Layers Analyzed: {info['total_layers']}")
        
        # Global gradient norm
        if summary["global_gradient_norm"]:
            global_norm = summary["global_gradient_norm"]
            print(f"\n🌐 Global Gradient Norm:")
            print(f"   Min: {global_norm['min']:.2e}")
            print(f"   Max: {global_norm['max']:.2e}")
            print(f"   Mean: {global_norm['mean']:.2e}")
            print(f"   Last: {global_norm['last']:.2e}")
        
        # Gradient health
        health = summary["gradient_health"]
        print(f"\n🏥 Gradient Health:")
        print(f"   Layers with vanishing gradients: {health['layers_with_vanishing']}")
        print(f"   Layers with exploding gradients: {health['layers_with_exploding']}")
        print(f"   Total vanishing episodes: {health['total_vanishing_episodes']}")
        print(f"   Total exploding episodes: {health['total_exploding_episodes']}")
        
        # Problematic layers
        if any(problems.values()):
            print(f"\n⚠️  Problematic Layers:")
            for problem_type, layers in problems.items():
                if layers:
                    print(f"   {problem_type}: {len(layers)} layers")
                    for layer in layers[:5]:  # Show first 5
                        print(f"     - {layer}")
                    if len(layers) > 5:
                        print(f"     ... and {len(layers) - 5} more")
        
        # Top layers by gradient norm
        layer_stats = summary["layer_statistics"]
        layers_with_data = [(name, stats) for name, stats in layer_stats.items() 
                           if not stats.get("no_data", False)]
        
        if layers_with_data:
            # Sort by last gradient norm
            layers_by_norm = sorted(layers_with_data, 
                                  key=lambda x: x[1]["gradient_norm"]["last"], 
                                  reverse=True)
            
            print(f"\n🔥 Top Layers by Gradient Norm:")
            for i, (layer_name, stats) in enumerate(layers_by_norm[:10]):
                norm = stats["gradient_norm"]["last"]
                param_count = stats["parameter_count"]
                print(f"   {i+1:2d}. {layer_name}: {norm:.2e} ({param_count:,} params)")
        
        print("="*60)


def create_layer_wise_tracker(vanishing_threshold: float = 1e-7, 
                             exploding_threshold: float = 10.0,
                             tracking_frequency: int = 1) -> LayerWiseGradientTracker:
    """Create a layer-wise gradient tracker with specified parameters."""
    return LayerWiseGradientTracker(
        vanishing_threshold=vanishing_threshold,
        exploding_threshold=exploding_threshold,
        tracking_frequency=tracking_frequency,
    )