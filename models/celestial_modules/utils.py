# models/celestial_modules/utils.py

import logging
import os
import torch
from typing import Any, Dict, Optional

class ModelUtils:
    """Utility methods for the Celestial Enhanced PGAT model."""
    
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Setup memory logger if debugging is enabled
        if config.enable_memory_debug or config.enable_memory_diagnostics:
            self.memory_logger = logging.getLogger("scripts.train.train_celestial_production.memory")
        else:
            self.memory_logger = None
    
    def log_info(self, message: str, *args: Any) -> None:
        """Log informational messages when verbose logging is enabled."""
        if self.config.verbose_logging:
            self.logger.info(message, *args)

    def log_debug(self, message: str, *args: Any) -> None:
        """Log debug diagnostics when diagnostics are requested."""
        if self.config.collect_diagnostics or self.config.verbose_logging:
            self.logger.debug(message, *args)

    def debug_memory(self, stage: str) -> None:
        """Emit lightweight memory diagnostics when requested."""
        if not (self.config.enable_memory_debug or self.config.collect_diagnostics or self.config.verbose_logging):
            return
        try:
            import psutil
            process = psutil.Process(os.getpid())
            cpu_mb = process.memory_info().rss / (1024 * 1024)
            self.logger.debug("MODEL_DEBUG [%s]: CPU=%0.1fMB", stage, cpu_mb)
        except Exception:
            self.logger.debug("MODEL_DEBUG [%s]: memory diagnostic unavailable", stage)

    def print_memory_debug(self, stage, extra_info=""):
        """Emit memory diagnostics to the dedicated memory logger (file), not stdout."""
        if not (self.config.enable_memory_debug or self.config.enable_memory_diagnostics):
            return
        if self.memory_logger is None:
            return
            
        import torch
        import gc
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
            
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                max_allocated = torch.cuda.max_memory_allocated() / 1024**3
                self.memory_logger.debug(
                    "MEMORY [%s] %s | GPU_Allocated=%.2fGB GPU_Reserved=%.2fGB GPU_MaxAllocated=%.2fGB",
                    stage, extra_info, allocated, reserved, max_allocated,
                )
            else:
                import psutil
                process = psutil.Process(os.getpid())
                cpu_gb = process.memory_info().rss / 1024**3
                self.memory_logger.debug(
                    "MEMORY [%s] %s | CPU=%.2fGB",
                    stage, extra_info, cpu_gb,
                )
        except Exception:
            self.memory_logger.debug("MEMORY [%s] %s | unavailable", stage, extra_info)

    @staticmethod
    def move_to_cpu(value: Any) -> Any:
        """Recursively detach tensors and move them to CPU for lightweight diagnostics."""
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: ModelUtils.move_to_cpu(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            converted = [ModelUtils.move_to_cpu(item) for item in value]
            return tuple(converted) if isinstance(value, tuple) else converted
        return value

    def log_configuration_summary(self) -> None:
        """Emit a concise configuration summary when verbose logging is enabled."""
        if not self.config.verbose_logging:
            return
        self.logger.info(
            (
                "Initializing Celestial Enhanced PGAT | seq_len=%s pred_len=%s d_model=%s "
                "celestial_bodies=%s wave_aggregation=%s mixture_decoder=%s stochastic_learner=%s "
                "hierarchical_mapping=%s"
            ),
            self.config.seq_len,
            self.config.pred_len,
            self.config.d_model,
            self.config.num_celestial_bodies,
            self.config.aggregate_waves_to_celestial,
            self.config.use_mixture_decoder,
            self.config.use_stochastic_learner,
            self.config.use_hierarchical_mapping,
        )
        if self.config.aggregate_waves_to_celestial:
            self.logger.info(
                "Phase-aware aggregation configured | input_waves=%s target_wave_indices=%s",
                self.config.num_input_waves,
                self.config.target_wave_indices,
            )

    def validate_adjacency_dimensions(self, astronomical_adj: torch.Tensor, 
                                    learned_adj: torch.Tensor, 
                                    dynamic_adj: torch.Tensor, 
                                    enc_out: torch.Tensor) -> None:
        """Validate that all adjacency matrices have consistent 4D dimensions."""
        batch_size, seq_len, d_model = enc_out.shape
        expected_shape = (batch_size, seq_len, self.config.num_celestial_bodies, self.config.num_celestial_bodies)
        
        matrices = {
            'astronomical_adj': astronomical_adj,
            'learned_adj': learned_adj, 
            'dynamic_adj': dynamic_adj
        }
        
        for name, matrix in matrices.items():
            if matrix.shape != expected_shape:
                self.logger.error(
                    "Adjacency matrix dimension mismatch: %s has shape %s, expected %s",
                    name, matrix.shape, expected_shape
                )
                raise ValueError(
                    f"Adjacency matrix {name} has incorrect dimensions: "
                    f"got {matrix.shape}, expected {expected_shape}"
                )
        
        self.log_debug("✅ All adjacency matrices validated: %s", expected_shape)

    def normalize_adj(self, adj_matrix):
        """Normalize adjacency matrix for stable fusion. Handles 4D dynamic matrices only."""
        # Add self-loops for stability
        identity = torch.eye(adj_matrix.size(-1), device=adj_matrix.device, dtype=torch.float32)
        identity = identity.unsqueeze(0).unsqueeze(0)  # Expand to [1, 1, nodes, nodes]
        
        adj_with_self_loops = adj_matrix + identity.expand_as(adj_matrix)
        
        # Row normalization (convert to stochastic matrix)
        row_sums = adj_with_self_loops.sum(dim=-1, keepdim=True)
        normalized = adj_with_self_loops / (row_sums + 1e-8)
        
        return normalized

    def get_point_prediction(self, forward_output):
        """Extracts a single point prediction from the model's output, handling the probabilistic case."""
        if self.config.use_mixture_decoder and isinstance(forward_output, tuple):
            means, _, log_weights = forward_output
            with torch.no_grad():
                if means.dim() == 4:  # [batch, pred_len, num_targets, num_components]
                    # Calculate weighted average of component means
                    weights = torch.softmax(log_weights, dim=-1).unsqueeze(2).expand_as(means)
                    predictions = (means * weights).sum(dim=-1)
                else:  # Univariate case
                    weights = torch.softmax(log_weights, dim=-1)
                    predictions = (means * weights).sum(dim=-1, keepdim=True)
                
                # Fallback to ensure correct output shape if something goes wrong
                if predictions.size(-1) != self.config.c_out:
                    return means[..., 0] if means.dim() == 4 else means[..., 0].unsqueeze(-1)
                return predictions
        return forward_output  # Output is already a point prediction