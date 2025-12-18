# models/celestial_modules/diagnostics.py

import logging
import torch
from typing import Any, Dict, Optional
from utils.fusion_diagnostics import FusionDiagnostics

class ModelDiagnostics:
    """Diagnostics system for the Celestial Enhanced PGAT model."""
    
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize fusion diagnostics
        if config.enable_fusion_diagnostics:
            self.fusion_diagnostics = FusionDiagnostics(
                enabled=True,
                log_first_n_batches=config.fusion_diag_batches
            )
        else:
            self.fusion_diagnostics = None
    
    def log_fusion_point(self, name: str, tensor_a: torch.Tensor, tensor_b: torch.Tensor, 
                        fusion_result: torch.Tensor, gate: Optional[torch.Tensor] = None):
        """Log a fusion point for diagnostics."""
        if self.fusion_diagnostics is not None:
            self.fusion_diagnostics.log_fusion_point(
                name=name,
                tensor_a=tensor_a,
                tensor_b=tensor_b,
                fusion_result=fusion_result,
                gate=gate
            )
    
    def print_fusion_diagnostics_summary(self):
        """Print summary of fusion diagnostics."""
        if self.fusion_diagnostics is not None:
            self.fusion_diagnostics.print_summary()
        else:
            self.logger.info("Fusion diagnostics not enabled")
    
    def increment_fusion_diagnostics_batch(self):
        """Increment the batch counter for fusion diagnostics."""
        if self.fusion_diagnostics is not None:
            self.fusion_diagnostics.increment_batch()
    
    def collect_wave_metadata(self, x_enc, celestial_features, adjacency_matrix, 
                             phase_metadata, target_waves, target_metadata) -> Dict[str, Any]:
        """Collect comprehensive wave processing metadata."""
        if not self.config.collect_diagnostics:
            return {}
            
        from .utils import ModelUtils
        
        wave_metadata = ModelUtils.move_to_cpu({
            'original_targets': target_waves,
            'phase_metadata': phase_metadata,
            'target_metadata': target_metadata,
            'adjacency_matrix_stats': {
                'mean': adjacency_matrix.mean().item() if adjacency_matrix is not None else 0.0,
                'std': adjacency_matrix.std().item() if adjacency_matrix is not None else 0.0,
            },
            'celestial_features_shape': celestial_features.shape if celestial_features is not None else None,
        })
        
        return wave_metadata
    
    def collect_celestial_metadata(self, celestial_results) -> Dict[str, Any]:
        """Collect celestial graph processing metadata."""
        if not self.config.collect_diagnostics:
            return {}
            
        from .utils import ModelUtils
        return ModelUtils.move_to_cpu(celestial_results.get('metadata', {}))
    
    def collect_fusion_metadata(self, fusion_metadata) -> Dict[str, Any]:
        """Collect fusion processing metadata."""
        if not self.config.collect_diagnostics:
            return {}
            
        from .utils import ModelUtils
        return ModelUtils.move_to_cpu(fusion_metadata)
    
    def log_gradient_stats(self, model: torch.nn.Module) -> Dict[str, float]:
        """
        Log gradient statistics for the entire model.
        Useful for detecting vanishing/exploding gradients.
        
        Returns:
            Dict with 'total_grad_norm', 'max_grad_norm', and per-component norms.
        """
        if not self.config.collect_diagnostics:
             return {}
        
        # DEBUG: Verify we are here
        # print("DEBUG: Inside log_gradient_stats", flush=True)

        total_norm = 0.0
        max_norm = 0.0
        stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                max_norm = max(max_norm, param_norm)
                
                # Log critical components specifically
                if 'graph_module' in name or 'decoder_module' in name:
                    # Keep hierarchy simple
                    short_name = name.split('.')[0] + "_" + name.split('.')[-1]
                    stats[f"grad_norm/{short_name}"] = param_norm
        
        total_norm = total_norm ** 0.5
        stats['grad_norm/total'] = total_norm
        stats['grad_norm/max'] = max_norm
        
        # Log to logger info if values are suspicious
        if total_norm < 1e-6:
             self.logger.warning(f"⚠️ Vanishing Gradient detected! Total Norm: {total_norm:.2e}")
        elif total_norm > 10.0:
             self.logger.warning(f"⚠️ Exploding Gradient detected! Total Norm: {total_norm:.2f}")
             
        return stats

    def check_tensor_health(self, name: str, tensor: torch.Tensor) -> Dict[str, float]:
        """
        Check tensor for anomalies (NaN, Inf) and health stats (Mean, Std).
        
        Args:
            name: Label for the tensor
            tensor: The tensor to check
        """
        if not self.config.collect_diagnostics:
            return {}

        if tensor is None:
            return {}
            
        # Anomaly Detection
        if torch.isnan(tensor).any():
            self.logger.error(f"❌ NaN detected in {name}!")
        if torch.isinf(tensor).any():
            self.logger.error(f"❌ Inf detected in {name}!")
            
        stats = {
            f"{name}/mean": tensor.mean().item(),
            f"{name}/std": tensor.std().item(),
            f"{name}/scale": tensor.abs().mean().item()
        }
        return stats
        
    def prepare_final_metadata(self, wave_metadata: Dict[str, Any], 
                              celestial_metadata: Dict[str, Any],
                              fusion_metadata: Dict[str, Any],
                              celestial_target_diagnostics: Optional[Dict[str, Any]] = None,
                              **kwargs) -> Optional[Dict[str, Any]]:
        """Prepare final metadata dictionary for return."""
        if not self.config.collect_diagnostics:
            return None
            
        from .utils import ModelUtils
        
        final_metadata = {
            **wave_metadata,
            'celestial_metadata': celestial_metadata,
            'fusion_metadata': fusion_metadata,
            'celestial_target_diagnostics': ModelUtils.move_to_cpu(celestial_target_diagnostics) if celestial_target_diagnostics else {},
            **kwargs
        }
        
        return final_metadata