"""
Memory Debugging Utilities for Celestial Enhanced PGAT

Use these utilities to debug memory issues during training.
"""

import torch
import psutil
import os
from typing import Dict, Any
import logging

class MemoryDebugger:
    """Utility class for debugging memory usage during training."""
    
    def __init__(self, log_file: str = "memory_debug.log"):
        self.process = psutil.Process(os.getpid())
        self.checkpoints = {}
        
        # Setup logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def checkpoint(self, name: str, **tensors):
        """Create a memory checkpoint with tensor information."""
        stats = self._get_memory_stats()
        
        # Log tensor shapes and memory usage
        self.logger.info(f"=== CHECKPOINT: {name} ===")
        self.logger.info(f"CPU Memory: {stats['cpu_mb']:.1f}MB")
        if stats['cuda_allocated_mb'] is not None:
            self.logger.info(f"CUDA Allocated: {stats['cuda_allocated_mb']:.1f}MB")
            self.logger.info(f"CUDA Reserved: {stats['cuda_reserved_mb']:.1f}MB")
        
        # Log tensor information
        for tensor_name, tensor in tensors.items():
            if isinstance(tensor, torch.Tensor):
                size_mb = tensor.numel() * tensor.element_size() / (1024 ** 2)
                self.logger.info(f"  {tensor_name}: {tensor.shape} | {size_mb:.1f}MB | {tensor.device}")
            else:
                self.logger.info(f"  {tensor_name}: {type(tensor)} | {tensor}")
        
        self.checkpoints[name] = stats
        return stats
    
    def compare(self, checkpoint1: str, checkpoint2: str):
        """Compare memory usage between two checkpoints."""
        if checkpoint1 not in self.checkpoints or checkpoint2 not in self.checkpoints:
            self.logger.error("One or both checkpoints not found")
            return
        
        stats1 = self.checkpoints[checkpoint1]
        stats2 = self.checkpoints[checkpoint2]
        
        cpu_diff = stats2['cpu_mb'] - stats1['cpu_mb']
        self.logger.info(f"Memory difference {checkpoint1} -> {checkpoint2}:")
        self.logger.info(f"  CPU: {cpu_diff:+.1f}MB")
        
        if stats1.get('cuda_allocated_mb') and stats2.get('cuda_allocated_mb'):
            cuda_diff = stats2['cuda_allocated_mb'] - stats1['cuda_allocated_mb']
            self.logger.info(f"  CUDA: {cuda_diff:+.1f}MB")
    
    def _get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        memory_info = self.process.memory_info()
        stats = {
            'cpu_mb': memory_info.rss / (1024 ** 2)
        }
        
        if torch.cuda.is_available():
            stats.update({
                'cuda_allocated_mb': torch.cuda.memory_allocated() / (1024 ** 2),
                'cuda_reserved_mb': torch.cuda.memory_reserved() / (1024 ** 2)
            })
        else:
            stats.update({
                'cuda_allocated_mb': None,
                'cuda_reserved_mb': None
            })
        
        return stats
    
    def assert_memory_limit(self, limit_gb: float, stage: str):
        """Assert that memory usage is below a certain limit."""
        stats = self._get_memory_stats()
        
        if stats.get('cuda_allocated_mb'):
            current_gb = stats['cuda_allocated_mb'] / 1024
            assert current_gb < limit_gb, f"Memory limit exceeded at {stage}: {current_gb:.2f}GB > {limit_gb}GB"
        else:
            current_gb = stats['cpu_mb'] / 1024
            assert current_gb < limit_gb, f"Memory limit exceeded at {stage}: {current_gb:.2f}GB > {limit_gb}GB"


def debug_tensor_shapes(**tensors):
    """Debug function to print tensor shapes and memory usage."""
    print("=== TENSOR SHAPES DEBUG ===")
    total_elements = 0
    
    for name, tensor in tensors.items():
        if isinstance(tensor, torch.Tensor):
            elements = tensor.numel()
            size_mb = elements * tensor.element_size() / (1024 ** 2)
            total_elements += elements
            print(f"{name:20s}: {str(tensor.shape):25s} | {size_mb:8.1f}MB | {tensor.device}")
        else:
            print(f"{name:20s}: {str(type(tensor)):25s} | {tensor}")
    
    total_mb = total_elements * 4 / (1024 ** 2)  # Assume float32
    print(f"{'TOTAL':20s}: {total_elements:,} elements | ~{total_mb:.1f}MB")
    print("=" * 70)


def debug_model_forward(model, *args, **kwargs):
    """Debug wrapper for model forward pass."""
    debugger = MemoryDebugger()
    
    try:
        debugger.checkpoint("before_forward", 
                          input_args=[arg.shape if isinstance(arg, torch.Tensor) else type(arg) for arg in args])
        
        output = model(*args, **kwargs)
        
        debugger.checkpoint("after_forward",
                          output_shape=output.shape if isinstance(output, torch.Tensor) else type(output))
        
        debugger.compare("before_forward", "after_forward")
        
        return output
        
    except Exception as e:
        debugger.checkpoint("error_state")
        debugger.logger.error(f"Model forward failed: {e}")
        raise


# Quick debugging functions
def quick_memory_check(stage: str):
    """Quick memory check with print output."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"MEMORY [{stage}]: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    else:
        process = psutil.Process(os.getpid())
        cpu_gb = process.memory_info().rss / (1024 ** 3)
        print(f"MEMORY [{stage}]: {cpu_gb:.2f}GB CPU")


def debug_celestial_combiner(combiner, *args, **kwargs):
    """Specific debugging for celestial graph combiner."""
    print("=== DEBUGGING CELESTIAL COMBINER ===")
    
    # Debug inputs
    if len(args) >= 4:
        astronomical_edges, learned_edges, attention_edges, enc_out = args[:4]
        debug_tensor_shapes(
            astronomical_edges=astronomical_edges,
            learned_edges=learned_edges,
            attention_edges=attention_edges,
            enc_out=enc_out
        )
    
    quick_memory_check("before_combiner")
    
    try:
        output = combiner(*args, **kwargs)
        quick_memory_check("after_combiner")
        
        if isinstance(output, tuple):
            combined_edges, metadata = output
            debug_tensor_shapes(combined_edges=combined_edges)
        else:
            debug_tensor_shapes(output=output)
        
        return output
        
    except Exception as e:
        quick_memory_check("combiner_error")
        print(f"Celestial combiner failed: {e}")
        raise