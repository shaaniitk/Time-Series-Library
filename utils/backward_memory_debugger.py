"""
Enhanced Memory Debugging for Backward Pass
Comprehensive memory tracking during neural network backward propagation
"""

import torch
import gc
import sys
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json
from pathlib import Path

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class BackwardMemoryDebugger:
    """
    Tracks memory consumption during backward pass with extreme granularity.
    
    This debugger helps identify:
    - Memory spikes during backward()
    - Gradient accumulation issues
    - Intermediate activation retention
    - Memory leaks in custom autograd functions
    """
    
    def __init__(self, device: torch.device, log_file: Optional[Path] = None):
        self.device = device
        self.log_file = log_file
        self.snapshots: List[Dict[str, Any]] = []
        self.gradient_stats: List[Dict[str, Any]] = []
        
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
    def capture_pre_forward(self, batch_idx: int, epoch: int) -> Dict[str, Any]:
        """Capture memory state before forward pass."""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "stage": "pre_forward",
            "batch_idx": batch_idx,
            "epoch": epoch,
        }
        
        snapshot.update(self._get_memory_stats())
        snapshot.update(self._get_tensor_stats())
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def capture_post_forward(self, batch_idx: int, epoch: int, 
                            outputs: Any, loss: torch.Tensor) -> Dict[str, Any]:
        """Capture memory state after forward pass and loss computation."""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "stage": "post_forward_pre_backward",
            "batch_idx": batch_idx,
            "epoch": epoch,
            "loss_value": float(loss.item()) if loss is not None else None,
        }
        
        # Analyze output structure
        snapshot.update(self._analyze_outputs(outputs))
        snapshot.update(self._get_memory_stats())
        snapshot.update(self._get_tensor_stats())
        
        # Track loss tensor
        if loss is not None:
            snapshot["loss_requires_grad"] = loss.requires_grad
            snapshot["loss_grad_fn"] = str(loss.grad_fn) if loss.grad_fn else None
            snapshot["loss_size"] = list(loss.shape) if hasattr(loss, 'shape') else None
            
        self.snapshots.append(snapshot)
        return snapshot
    
    def capture_pre_backward(self, batch_idx: int, epoch: int) -> Dict[str, Any]:
        """Capture memory state immediately before backward()."""
        # Force synchronization for accurate timing
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "stage": "pre_backward",
            "batch_idx": batch_idx,
            "epoch": epoch,
        }
        
        snapshot.update(self._get_memory_stats())
        snapshot.update(self._get_tensor_stats())
        snapshot.update(self._get_gradient_stats())
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def capture_post_backward(self, batch_idx: int, epoch: int, 
                             model: torch.nn.Module) -> Dict[str, Any]:
        """Capture memory state after backward() completes."""
        # Force synchronization
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "stage": "post_backward",
            "batch_idx": batch_idx,
            "epoch": epoch,
        }
        
        snapshot.update(self._get_memory_stats())
        snapshot.update(self._get_tensor_stats())
        snapshot.update(self._get_gradient_stats())
        snapshot.update(self._analyze_model_gradients(model))
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def capture_post_optimizer_step(self, batch_idx: int, epoch: int) -> Dict[str, Any]:
        """Capture memory state after optimizer step."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "stage": "post_optimizer_step",
            "batch_idx": batch_idx,
            "epoch": epoch,
        }
        
        snapshot.update(self._get_memory_stats())
        snapshot.update(self._get_tensor_stats())
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def _get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        stats = {}
        
        # CUDA memory
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.synchronize(self.device)
                stats.update({
                    "cuda_allocated_mb": round(torch.cuda.memory_allocated(self.device) / (1024**2), 2),
                    "cuda_reserved_mb": round(torch.cuda.memory_reserved(self.device) / (1024**2), 2),
                    "cuda_max_allocated_mb": round(torch.cuda.max_memory_allocated(self.device) / (1024**2), 2),
                    "cuda_max_reserved_mb": round(torch.cuda.max_memory_reserved(self.device) / (1024**2), 2),
                    "cuda_cached_mb": round(torch.cuda.memory_reserved(self.device) / (1024**2), 2),
                })
                
                # Get memory fragmentation info
                try:
                    memory_stats = torch.cuda.memory_stats(self.device)
                    stats["cuda_num_alloc_retries"] = memory_stats.get("num_alloc_retries", 0)
                    stats["cuda_num_ooms"] = memory_stats.get("num_ooms", 0)
                except:
                    pass
                    
            except Exception as e:
                stats["cuda_memory_error"] = str(e)
        
        # CPU memory
        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                mem_info = process.memory_info()
                stats.update({
                    "process_rss_mb": round(mem_info.rss / (1024**2), 2),
                    "process_vms_mb": round(mem_info.vms / (1024**2), 2),
                })
                
                vm = psutil.virtual_memory()
                stats.update({
                    "system_available_mb": round(vm.available / (1024**2), 2),
                    "system_percent_used": vm.percent,
                })
            except Exception as e:
                stats["cpu_memory_error"] = str(e)
        
        return stats
    
    def _get_tensor_stats(self) -> Dict[str, Any]:
        """Get statistics about active tensors."""
        stats = {
            "num_tensors": 0,
            "num_tensors_with_grad": 0,
            "num_tensors_in_grad_graph": 0,
            "total_tensor_memory_mb": 0.0,
        }
        
        try:
            for obj in gc.get_objects():
                if torch.is_tensor(obj):
                    stats["num_tensors"] += 1
                    
                    if obj.requires_grad:
                        stats["num_tensors_with_grad"] += 1
                    
                    if hasattr(obj, 'grad_fn') and obj.grad_fn is not None:
                        stats["num_tensors_in_grad_graph"] += 1
                    
                    # Estimate memory
                    if hasattr(obj, 'element_size') and hasattr(obj, 'numel'):
                        stats["total_tensor_memory_mb"] += (obj.element_size() * obj.numel()) / (1024**2)
        except Exception as e:
            stats["tensor_stats_error"] = str(e)
        
        stats["total_tensor_memory_mb"] = round(stats["total_tensor_memory_mb"], 2)
        return stats
    
    def _get_gradient_stats(self) -> Dict[str, Any]:
        """Get statistics about gradients."""
        stats = {
            "num_gradient_tensors": 0,
            "gradient_memory_mb": 0.0,
        }
        
        try:
            for obj in gc.get_objects():
                if torch.is_tensor(obj):
                    if hasattr(obj, 'grad') and obj.grad is not None:
                        stats["num_gradient_tensors"] += 1
                        grad = obj.grad
                        if hasattr(grad, 'element_size') and hasattr(grad, 'numel'):
                            stats["gradient_memory_mb"] += (grad.element_size() * grad.numel()) / (1024**2)
        except Exception as e:
            stats["gradient_stats_error"] = str(e)
        
        stats["gradient_memory_mb"] = round(stats["gradient_memory_mb"], 2)
        return stats
    
    def _analyze_outputs(self, outputs: Any) -> Dict[str, Any]:
        """Analyze model outputs structure."""
        analysis = {
            "output_type": type(outputs).__name__,
        }
        
        if isinstance(outputs, torch.Tensor):
            analysis.update({
                "output_shape": list(outputs.shape),
                "output_dtype": str(outputs.dtype),
                "output_requires_grad": outputs.requires_grad,
                "output_grad_fn": str(outputs.grad_fn) if outputs.grad_fn else None,
                "output_memory_mb": round((outputs.element_size() * outputs.numel()) / (1024**2), 2),
            })
        elif isinstance(outputs, (tuple, list)):
            analysis["output_length"] = len(outputs)
            if len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
                main_output = outputs[0]
                analysis.update({
                    "main_output_shape": list(main_output.shape),
                    "main_output_dtype": str(main_output.dtype),
                    "main_output_requires_grad": main_output.requires_grad,
                    "main_output_memory_mb": round((main_output.element_size() * main_output.numel()) / (1024**2), 2),
                })
        
        return analysis
    
    def _analyze_model_gradients(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Analyze gradients in the model."""
        analysis = {
            "params_with_grad": 0,
            "params_without_grad": 0,
            "total_gradient_norm": 0.0,
            "max_gradient_norm": 0.0,
            "gradient_memory_mb": 0.0,
            "params_with_nan_grad": 0,
            "params_with_inf_grad": 0,
        }
        
        try:
            total_norm = 0.0
            max_norm = 0.0
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    analysis["params_with_grad"] += 1
                    
                    # Calculate gradient norm
                    param_norm = param.grad.data.norm(2).item()
                    total_norm += param_norm ** 2
                    max_norm = max(max_norm, param_norm)
                    
                    # Check for NaN/Inf
                    if torch.isnan(param.grad).any():
                        analysis["params_with_nan_grad"] += 1
                    if torch.isinf(param.grad).any():
                        analysis["params_with_inf_grad"] += 1
                    
                    # Memory
                    if hasattr(param.grad, 'element_size') and hasattr(param.grad, 'numel'):
                        analysis["gradient_memory_mb"] += (param.grad.element_size() * param.grad.numel()) / (1024**2)
                else:
                    analysis["params_without_grad"] += 1
            
            analysis["total_gradient_norm"] = round(total_norm ** 0.5, 4)
            analysis["max_gradient_norm"] = round(max_norm, 4)
            analysis["gradient_memory_mb"] = round(analysis["gradient_memory_mb"], 2)
            
        except Exception as e:
            analysis["gradient_analysis_error"] = str(e)
        
        return analysis
    
    def analyze_memory_growth(self) -> Dict[str, Any]:
        """Analyze memory growth patterns across snapshots."""
        if len(self.snapshots) < 2:
            return {"error": "Not enough snapshots for analysis"}
        
        analysis = {
            "snapshots_analyzed": len(self.snapshots),
            "stages": [],
        }
        
        # Group by stage
        stages = {}
        for snapshot in self.snapshots:
            stage = snapshot.get("stage", "unknown")
            if stage not in stages:
                stages[stage] = []
            stages[stage].append(snapshot)
        
        # Analyze each stage
        for stage, snapshots_list in stages.items():
            if not snapshots_list:
                continue
                
            cuda_allocated = [s.get("cuda_allocated_mb", 0) for s in snapshots_list]
            cuda_reserved = [s.get("cuda_reserved_mb", 0) for s in snapshots_list]
            
            stage_analysis = {
                "stage": stage,
                "occurrences": len(snapshots_list),
                "cuda_allocated_avg": round(sum(cuda_allocated) / len(cuda_allocated), 2) if cuda_allocated else 0,
                "cuda_allocated_max": max(cuda_allocated) if cuda_allocated else 0,
                "cuda_reserved_avg": round(sum(cuda_reserved) / len(cuda_reserved), 2) if cuda_reserved else 0,
                "cuda_reserved_max": max(cuda_reserved) if cuda_reserved else 0,
            }
            
            analysis["stages"].append(stage_analysis)
        
        # Calculate growth between consecutive stages
        if len(self.snapshots) >= 2:
            growth_patterns = []
            prev = self.snapshots[0]
            
            for curr in self.snapshots[1:]:
                if prev.get("batch_idx") == curr.get("batch_idx"):
                    growth = {
                        "from_stage": prev.get("stage"),
                        "to_stage": curr.get("stage"),
                        "cuda_allocated_growth_mb": round(
                            curr.get("cuda_allocated_mb", 0) - prev.get("cuda_allocated_mb", 0), 2
                        ),
                        "cuda_reserved_growth_mb": round(
                            curr.get("cuda_reserved_mb", 0) - prev.get("cuda_reserved_mb", 0), 2
                        ),
                        "tensor_count_growth": curr.get("num_tensors", 0) - prev.get("num_tensors", 0),
                    }
                    growth_patterns.append(growth)
                prev = curr
            
            analysis["growth_patterns"] = growth_patterns
        
        return analysis
    
    def find_memory_leaks(self) -> List[Dict[str, Any]]:
        """Identify potential memory leaks."""
        leaks = []
        
        # Check for monotonically increasing memory
        if len(self.snapshots) >= 10:
            post_optimizer_snapshots = [s for s in self.snapshots if s.get("stage") == "post_optimizer_step"]
            
            if len(post_optimizer_snapshots) >= 5:
                allocated = [s.get("cuda_allocated_mb", 0) for s in post_optimizer_snapshots[-5:]]
                
                # Check if consistently increasing
                increasing = all(allocated[i] < allocated[i+1] for i in range(len(allocated)-1))
                
                if increasing:
                    leaks.append({
                        "type": "monotonic_memory_growth",
                        "description": "Memory consistently increasing after optimizer steps",
                        "growth_rate_mb_per_step": round((allocated[-1] - allocated[0]) / len(allocated), 2),
                        "recent_values": allocated,
                    })
        
        # Check for tensors not being freed
        tensor_counts = [s.get("num_tensors", 0) for s in self.snapshots[-10:]]
        if tensor_counts and max(tensor_counts) - min(tensor_counts) > 100:
            leaks.append({
                "type": "tensor_accumulation",
                "description": "Large variation in tensor count suggests tensors not being freed",
                "min_tensors": min(tensor_counts),
                "max_tensors": max(tensor_counts),
                "variation": max(tensor_counts) - min(tensor_counts),
            })
        
        return leaks
    
    def save_diagnostics(self, filepath: Optional[Path] = None) -> None:
        """Save comprehensive diagnostics to file."""
        if filepath is None:
            filepath = self.log_file
        
        if filepath is None:
            print("No filepath specified for saving diagnostics")
            return
        
        diagnostics = {
            "timestamp": datetime.now().isoformat(),
            "device": str(self.device),
            "total_snapshots": len(self.snapshots),
            "snapshots": self.snapshots,
            "memory_growth_analysis": self.analyze_memory_growth(),
            "potential_leaks": self.find_memory_leaks(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(diagnostics, f, indent=2)
        
        print(f"Memory diagnostics saved to {filepath}")
    
    def print_summary(self) -> None:
        """Print a summary of memory usage patterns."""
        if not self.snapshots:
            print("No snapshots captured yet")
            return
        
        print("\n" + "="*80)
        print("BACKWARD PASS MEMORY DEBUGGING SUMMARY")
        print("="*80)
        
        # Latest snapshot
        if self.snapshots:
            latest = self.snapshots[-1]
            print(f"\nLatest snapshot ({latest.get('stage', 'unknown')}):")
            print(f"  CUDA Allocated: {latest.get('cuda_allocated_mb', 'N/A')} MB")
            print(f"  CUDA Reserved: {latest.get('cuda_reserved_mb', 'N/A')} MB")
            print(f"  CUDA Max Allocated: {latest.get('cuda_max_allocated_mb', 'N/A')} MB")
            print(f"  Active Tensors: {latest.get('num_tensors', 'N/A')}")
            print(f"  Tensors with Gradients: {latest.get('num_tensors_with_grad', 'N/A')}")
        
        # Growth analysis
        growth = self.analyze_memory_growth()
        if "stages" in growth:
            print("\nMemory usage by stage:")
            for stage_info in growth["stages"]:
                print(f"  {stage_info['stage']}:")
                print(f"    Avg CUDA Allocated: {stage_info['cuda_allocated_avg']} MB")
                print(f"    Max CUDA Allocated: {stage_info['cuda_allocated_max']} MB")
        
        # Leaks
        leaks = self.find_memory_leaks()
        if leaks:
            print("\n⚠️  POTENTIAL MEMORY LEAKS DETECTED:")
            for leak in leaks:
                print(f"  - {leak['type']}: {leak['description']}")
                if 'growth_rate_mb_per_step' in leak:
                    print(f"    Growth rate: {leak['growth_rate_mb_per_step']} MB/step")
        
        print("="*80 + "\n")


def create_backward_debugger(device: torch.device, 
                            checkpoint_dir: Path) -> BackwardMemoryDebugger:
    """Factory function to create a backward memory debugger."""
    log_file = checkpoint_dir / "backward_memory_diagnostics.json"
    return BackwardMemoryDebugger(device=device, log_file=log_file)
