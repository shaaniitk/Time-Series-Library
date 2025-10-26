"""
Comprehensive Fusion Diagnostics for Celestial Enhanced PGAT

This module provides utilities to detect and diagnose magnitude imbalance issues
at all fusion points in the model architecture.
"""

import logging
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class FusionDiagnostics:
    """Monitors and diagnoses fusion points for magnitude imbalance."""
    
    def __init__(self, enabled: bool = True, log_first_n_batches: int = 10):
        """
        Initialize fusion diagnostics.
        
        Args:
            enabled: Whether diagnostics are active
            log_first_n_batches: Number of initial batches to log
        """
        self.enabled = enabled
        self.log_first_n_batches = log_first_n_batches
        self.batch_count = 0
        self.fusion_stats: Dict[str, List[Dict[str, float]]] = {}
        
    def reset(self):
        """Reset diagnostic state."""
        self.batch_count = 0
        self.fusion_stats.clear()
        
    def log_fusion_point(
        self,
        name: str,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        fusion_result: Optional[torch.Tensor] = None,
        gate: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Log statistics for a fusion point.
        
        Args:
            name: Identifier for this fusion point
            tensor_a: First input tensor
            tensor_b: Second input tensor
            fusion_result: Result of fusion (optional)
            gate: Gate values if using gated fusion (optional)
        """
        if not self.enabled or self.batch_count >= self.log_first_n_batches:
            return
            
        with torch.no_grad():
            # Compute norms
            norm_a = torch.linalg.vector_norm(tensor_a, dim=-1).mean().item()
            norm_b = torch.linalg.vector_norm(tensor_b, dim=-1).mean().item()
            
            # Compute means and stds
            mean_a = tensor_a.mean().item()
            std_a = tensor_a.std(unbiased=False).item()
            mean_b = tensor_b.mean().item()
            std_b = tensor_b.std(unbiased=False).item()
            
            # Magnitude ratio
            ratio = norm_a / (norm_b + 1e-8)
            
            stats = {
                'batch': self.batch_count,
                'norm_a': norm_a,
                'norm_b': norm_b,
                'ratio': ratio,
                'mean_a': mean_a,
                'std_a': std_a,
                'mean_b': mean_b,
                'std_b': std_b,
            }
            
            # Add gate statistics if available
            if gate is not None:
                stats['gate_mean'] = gate.mean().item()
                stats['gate_std'] = gate.std(unbiased=False).item()
                stats['gate_min'] = gate.min().item()
                stats['gate_max'] = gate.max().item()
            
            # Add fusion result statistics if available
            if fusion_result is not None:
                stats['result_norm'] = torch.linalg.vector_norm(fusion_result, dim=-1).mean().item()
                stats['result_mean'] = fusion_result.mean().item()
                stats['result_std'] = fusion_result.std(unbiased=False).item()
            
            # Store stats
            if name not in self.fusion_stats:
                self.fusion_stats[name] = []
            self.fusion_stats[name].append(stats)
            
            # Log if there's significant imbalance
            if ratio > 10.0 or ratio < 0.1:
                logger.warning(
                    "⚠️ MAGNITUDE IMBALANCE at %s | norm_a=%.4f norm_b=%.4f ratio=%.2fx",
                    name, norm_a, norm_b, ratio
                )
            else:
                logger.debug(
                    "✓ Fusion point %s | norm_a=%.4f norm_b=%.4f ratio=%.2fx",
                    name, norm_a, norm_b, ratio
                )
    
    def increment_batch(self):
        """Increment batch counter."""
        self.batch_count += 1
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics across all batches.
        
        Returns:
            Dictionary mapping fusion point names to average statistics
        """
        summary = {}
        for name, stats_list in self.fusion_stats.items():
            if not stats_list:
                continue
                
            # Average across batches
            avg_stats = {}
            keys = stats_list[0].keys()
            for key in keys:
                if key == 'batch':
                    continue
                values = [s[key] for s in stats_list]
                avg_stats[f'{key}_mean'] = sum(values) / len(values)
                avg_stats[f'{key}_max'] = max(values)
                avg_stats[f'{key}_min'] = min(values)
            
            summary[name] = avg_stats
        
        return summary
    
    def print_summary(self):
        """Print summary of all fusion points."""
        if not self.fusion_stats:
            logger.info("No fusion diagnostics collected")
            return
            
        logger.info("=" * 80)
        logger.info("FUSION DIAGNOSTICS SUMMARY")
        logger.info("=" * 80)
        
        summary = self.get_summary()
        for name, stats in summary.items():
            logger.info(f"\n📊 {name}:")
            logger.info(f"  Norm ratio: {stats.get('ratio_mean', 0):.2f}x (range: {stats.get('ratio_min', 0):.2f}-{stats.get('ratio_max', 0):.2f})")
            logger.info(f"  Norm A: {stats.get('norm_a_mean', 0):.4f}")
            logger.info(f"  Norm B: {stats.get('norm_b_mean', 0):.4f}")
            
            if 'gate_mean_mean' in stats:
                logger.info(f"  Gate: {stats['gate_mean_mean']:.4f} ± {stats['gate_std_mean']:.4f}")
            
            # Flag issues
            ratio_mean = stats.get('ratio_mean', 1.0)
            if ratio_mean > 5.0 or ratio_mean < 0.2:
                logger.warning(f"  ⚠️ IMBALANCE DETECTED - Consider adding normalization!")
        
        logger.info("=" * 80)


def compare_fusion_strategies(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    gate: Optional[torch.Tensor] = None,
    name: str = "fusion_point",
) -> Dict[str, torch.Tensor]:
    """
    Compare different fusion strategies empirically.
    
    Args:
        tensor_a: First tensor (e.g., enc_out)
        tensor_b: Second tensor (e.g., fused_output)
        gate: Optional gate values
        name: Name for logging
    
    Returns:
        Dictionary mapping strategy names to fusion results
    """
    results = {}
    
    with torch.no_grad():
        # Strategy 1: Simple addition
        results['addition'] = tensor_a + tensor_b
        
        # Strategy 2: Interpolation (if gate provided)
        if gate is not None:
            results['interpolation'] = (1 - gate) * tensor_a + gate * tensor_b
            results['gated_addition'] = tensor_a + gate * tensor_b
        
        # Strategy 3: Element-wise multiplication
        results['multiplication'] = tensor_a * tensor_b
        
        # Strategy 4: Concatenation (dimension changes, just for reference)
        results['concatenation'] = torch.cat([tensor_a, tensor_b], dim=-1)
        
        # Log statistics
        logger.debug(f"Fusion strategy comparison for {name}:")
        for strategy, result in results.items():
            if result.shape == tensor_a.shape:  # Only comparable if same shape
                norm = torch.linalg.vector_norm(result, dim=-1).mean().item()
                logger.debug(f"  {strategy}: norm={norm:.4f}")
    
    return results


class NormMonitor:
    """Monitors norms of all layer outputs."""
    
    def __init__(self, model: nn.Module, enabled: bool = True):
        """
        Initialize norm monitor.
        
        Args:
            model: Model to monitor
            enabled: Whether monitoring is active
        """
        self.enabled = enabled
        self.model = model
        self.hooks = []
        self.norms: Dict[str, List[float]] = {}
        
        if enabled:
            self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on all layers."""
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    with torch.no_grad():
                        norm = torch.linalg.vector_norm(output, dim=-1).mean().item()
                        if name not in self.norms:
                            self.norms[name] = []
                        self.norms[name].append(norm)
                        
                        # Flag extreme values
                        if norm > 100.0:
                            logger.warning(f"⚠️ Large norm at {name}: {norm:.2f}")
                        elif norm < 0.01:
                            logger.warning(f"⚠️ Small norm at {name}: {norm:.4f}")
            return hook
        
        # Register hooks on key layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm, nn.MultiheadAttention)):
                handle = module.register_forward_hook(hook_fn(name))
                self.hooks.append(handle)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
    
    def get_summary(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Get summary of norms.
        
        Returns:
            Dictionary mapping layer names to (mean, min, max) norms
        """
        summary = {}
        for name, norms_list in self.norms.items():
            if norms_list:
                summary[name] = (
                    sum(norms_list) / len(norms_list),  # mean
                    min(norms_list),  # min
                    max(norms_list),  # max
                )
        return summary
    
    def print_summary(self, top_k: int = 20):
        """Print summary of largest and smallest norms."""
        if not self.norms:
            logger.info("No norm statistics collected")
            return
        
        summary = self.get_summary()
        sorted_by_mean = sorted(summary.items(), key=lambda x: x[1][0], reverse=True)
        
        logger.info("=" * 80)
        logger.info(f"TOP {top_k} LARGEST NORMS:")
        for name, (mean, min_val, max_val) in sorted_by_mean[:top_k]:
            logger.info(f"  {name}: {mean:.4f} (range: {min_val:.4f}-{max_val:.4f})")
        
        logger.info(f"\nTOP {top_k} SMALLEST NORMS:")
        for name, (mean, min_val, max_val) in reversed(sorted_by_mean[-top_k:]):
            logger.info(f"  {name}: {mean:.4f} (range: {min_val:.4f}-{max_val:.4f})")
        logger.info("=" * 80)
