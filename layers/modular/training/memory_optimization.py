"""
Memory Optimization Components

Modular components for memory-efficient training of large time series models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Callable
import math
from torch.utils.checkpoint import checkpoint


class GradientCheckpointing(nn.Module):
    """Gradient checkpointing wrapper for memory efficiency."""
    
    def __init__(self, module: nn.Module, segments: int = 2):
        super().__init__()
        self.module = module
        self.segments = segments
        
    def forward(self, *args, **kwargs):
        """Forward with gradient checkpointing."""
        if self.training:
            return checkpoint(self.module, *args, **kwargs)
        else:
            return self.module(*args, **kwargs)


class MixedPrecisionTraining:
    """Mixed precision training utilities."""
    
    def __init__(self, enabled: bool = True, loss_scale: float = 2**16):
        self.enabled = enabled
        self.loss_scale = loss_scale
        self.scaler = torch.cuda.amp.GradScaler() if enabled and torch.cuda.is_available() else None
        
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision training."""
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss
    
    def step_optimizer(self, optimizer: torch.optim.Optimizer):
        """Step optimizer with gradient scaling."""
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def autocast_context(self):
        """Get autocast context manager."""
        if self.enabled and torch.cuda.is_available():
            return torch.cuda.amp.autocast()
        else:
            return torch.no_grad() if not self.training else torch.enable_grad()


class MemoryEfficientAttention(nn.Module):
    """Memory-efficient attention implementation."""
    
    def __init__(self, d_model: int, num_heads: int, chunk_size: int = 512):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Memory-efficient attention forward pass."""
        batch_size, seq_len, d_model = query.shape
        
        # Project to Q, K, V
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Chunked attention computation
        if seq_len > self.chunk_size:
            output = self._chunked_attention(Q, K, V)
        else:
            output = self._standard_attention(Q, K, V)
        
        # Output projection
        output = output.view(batch_size, seq_len, d_model)
        return self.out_proj(output)
    
    def _standard_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Standard attention computation."""
        # Q, K, V: [batch, seq_len, num_heads, head_dim]
        Q = Q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        
        return output.transpose(1, 2)  # [batch, seq_len, num_heads, head_dim]
    
    def _chunked_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Chunked attention computation for memory efficiency."""
        batch_size, seq_len, num_heads, head_dim = Q.shape
        
        # Initialize output
        output = torch.zeros_like(Q)
        
        # Process in chunks
        for i in range(0, seq_len, self.chunk_size):
            end_i = min(i + self.chunk_size, seq_len)
            Q_chunk = Q[:, i:end_i]  # [batch, chunk_size, num_heads, head_dim]
            
            # Compute attention for this chunk against all keys
            Q_chunk = Q_chunk.transpose(1, 2)  # [batch, num_heads, chunk_size, head_dim]
            K_full = K.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
            V_full = V.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
            
            scores = torch.matmul(Q_chunk, K_full.transpose(-2, -1)) / math.sqrt(head_dim)
            attn_weights = F.softmax(scores, dim=-1)
            chunk_output = torch.matmul(attn_weights, V_full)
            
            output[:, i:end_i] = chunk_output.transpose(1, 2)
        
        return output


class MemoryMonitor:
    """Monitor and log memory usage during training."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.memory_history = []
        
    def log_memory(self, step: str):
        """Log current memory usage."""
        if self.device == 'cuda' and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            
            self.memory_history.append({
                'step': step,
                'allocated_gb': allocated,
                'reserved_gb': reserved
            })
            
            print(f"Memory at {step}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    def get_peak_memory(self) -> Dict[str, float]:
        """Get peak memory usage."""
        if not self.memory_history:
            return {'allocated_gb': 0.0, 'reserved_gb': 0.0}
        
        peak_allocated = max(entry['allocated_gb'] for entry in self.memory_history)
        peak_reserved = max(entry['reserved_gb'] for entry in self.memory_history)
        
        return {
            'peak_allocated_gb': peak_allocated,
            'peak_reserved_gb': peak_reserved
        }
    
    def clear_cache(self):
        """Clear GPU cache."""
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()


class AdaptiveBatchSizing:
    """Adaptive batch sizing based on memory constraints."""
    
    def __init__(self, initial_batch_size: int = 32, max_batch_size: int = 128,
                 memory_threshold: float = 0.8):
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.memory_threshold = memory_threshold
        self.current_batch_size = initial_batch_size
        self.oom_count = 0
        
    def adjust_batch_size(self, oom_occurred: bool = False) -> int:
        """Adjust batch size based on memory usage."""
        if oom_occurred:
            self.oom_count += 1
            # Reduce batch size by 25%
            self.current_batch_size = max(1, int(self.current_batch_size * 0.75))
            print(f"OOM detected. Reducing batch size to {self.current_batch_size}")
        else:
            # Try to increase batch size if no OOM for a while
            if self.oom_count == 0 and self.current_batch_size < self.max_batch_size:
                # Check current memory usage
                if torch.cuda.is_available():
                    memory_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                    if memory_usage < self.memory_threshold:
                        # Increase batch size by 10%
                        self.current_batch_size = min(self.max_batch_size, 
                                                    int(self.current_batch_size * 1.1))
                        print(f"Increasing batch size to {self.current_batch_size}")
        
        return self.current_batch_size
    
    def reset_oom_count(self):
        """Reset OOM counter."""
        self.oom_count = 0


class MemoryOptimizedTrainer:
    """Trainer with comprehensive memory optimizations."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        
        # Initialize memory optimization components
        self.mixed_precision = MixedPrecisionTraining(
            enabled=config.get('use_mixed_precision', True)
        )
        
        self.memory_monitor = MemoryMonitor(
            device=config.get('device', 'cuda')
        )
        
        self.adaptive_batching = AdaptiveBatchSizing(
            initial_batch_size=config.get('batch_size', 32),
            max_batch_size=config.get('max_batch_size', 128),
            memory_threshold=config.get('memory_threshold', 0.8)
        )
        
        # Apply gradient checkpointing if requested
        if config.get('use_gradient_checkpointing', False):
            self._apply_gradient_checkpointing()
    
    def _apply_gradient_checkpointing(self):
        """Apply gradient checkpointing to model components."""
        # Apply to attention layers
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or 'moe' in name.lower():
                # Wrap in gradient checkpointing
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent = dict(self.model.named_modules())[parent_name]
                    setattr(parent, child_name, GradientCheckpointing(module))
    
    def train_step(self, batch: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """Memory-optimized training step."""
        self.memory_monitor.log_memory('step_start')
        
        try:
            # Forward pass with mixed precision
            with self.mixed_precision.autocast_context():
                if isinstance(batch, dict):
                    output = self.model(**batch)
                else:
                    output = self.model(*batch)
                
                # Compute loss
                if isinstance(output, tuple):
                    predictions, moe_info = output
                    loss = self._compute_loss(predictions, batch.get('targets'))
                    
                    # Add MoE load balancing losses
                    if moe_info:
                        for expert_type, info in moe_info.items():
                            if 'load_balancing_loss' in info:
                                loss += 0.01 * info['load_balancing_loss']
                else:
                    loss = self._compute_loss(output, batch.get('targets'))
            
            # Backward pass with gradient scaling
            optimizer.zero_grad()
            scaled_loss = self.mixed_precision.scale_loss(loss)
            scaled_loss.backward()
            
            # Optimizer step with gradient scaling
            self.mixed_precision.step_optimizer(optimizer)
            
            self.memory_monitor.log_memory('step_end')
            
            # Reset OOM counter on successful step
            self.adaptive_batching.reset_oom_count()
            
            return {
                'loss': loss.item(),
                'batch_size': self.adaptive_batching.current_batch_size,
                'memory_usage': self.memory_monitor.get_peak_memory()
            }
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM Error: {e}")
                
                # Clear cache and adjust batch size
                self.memory_monitor.clear_cache()
                new_batch_size = self.adaptive_batching.adjust_batch_size(oom_occurred=True)
                
                return {
                    'loss': float('inf'),
                    'oom_occurred': True,
                    'new_batch_size': new_batch_size,
                    'error': str(e)
                }
            else:
                raise e
    
    def _compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss function."""
        if targets is None:
            # If no targets provided, use dummy loss for testing
            return predictions.mean() * 0
        
        return F.mse_loss(predictions, targets)
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory usage summary."""
        return {
            'peak_memory': self.memory_monitor.get_peak_memory(),
            'current_batch_size': self.adaptive_batching.current_batch_size,
            'oom_count': self.adaptive_batching.oom_count,
            'mixed_precision_enabled': self.mixed_precision.enabled,
            'memory_history': self.memory_monitor.memory_history[-10:]  # Last 10 entries
        }