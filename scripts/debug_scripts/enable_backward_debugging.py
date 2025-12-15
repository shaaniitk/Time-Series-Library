"""
Quick Script to Enable Enhanced Backward Pass Debugging

This script provides an easy way to enable detailed backward pass debugging
in the production training script without modifying the original code.

Usage:
    python enable_backward_debugging.py
    
This will create a wrapper that:
1. Imports the backward debugger
2. Adds detailed memory tracking around backward()
3. Saves comprehensive diagnostics to help identify OOM causes
"""

import torch
import torch.nn as nn
from pathlib import Path
import logging
import sys

# Add project root
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.backward_memory_debugger import create_backward_debugger

LOGGER = logging.getLogger(__name__)


def patch_backward_with_debugging(model: nn.Module, device: torch.device, checkpoint_dir: Path):
    """
    Monkey-patch backward pass to add debugging.
    
    This wraps the model's forward pass to track computational graph creation
    and monitors memory during backward().
    """
    
    # Create debugger
    debugger = create_backward_debugger(device, checkpoint_dir)
    
    # Store original forward
    original_forward = model.forward
    
    # Counter for tracking batches
    batch_counter = {"count": 0, "epoch": 0}
    
    def debug_forward(*args, **kwargs):
        """Wrapped forward with debugging."""
        batch_idx = batch_counter["count"]
        epoch = batch_counter["epoch"]
        
        # Pre-forward snapshot
        if batch_idx % 10 == 0:  # Debug every 10th batch
            debugger.capture_pre_forward(batch_idx, epoch)
        
        # Original forward pass
        outputs = original_forward(*args, **kwargs)
        
        # Post-forward snapshot
        if batch_idx % 10 == 0:
            # Create dummy loss for snapshot (won't be used for actual training)
            if isinstance(outputs, tuple):
                dummy_loss = outputs[0].mean() if isinstance(outputs[0], torch.Tensor) else None
            else:
                dummy_loss = outputs.mean() if isinstance(outputs, torch.Tensor) else None
            
            if dummy_loss is not None:
                debugger.capture_post_forward(batch_idx, epoch, outputs, dummy_loss)
        
        batch_counter["count"] += 1
        return outputs
    
    # Replace forward
    model.forward = debug_forward
    
    # Return debugger for external use
    return debugger, batch_counter


def create_debugging_hooks(model: nn.Module, logger: logging.Logger):
    """
    Create PyTorch hooks to monitor backward pass.
    
    These hooks will:
    - Track gradient computation for each layer
    - Detect NaN/Inf gradients
    - Monitor memory growth during backward
    """
    
    hooks = []
    gradient_stats = {}
    
    def create_backward_hook(name: str):
        """Create a backward hook for a specific parameter."""
        
        def hook_fn(grad):
            """Hook function called during backward."""
            if grad is not None:
                # Track gradient statistics
                gradient_stats[name] = {
                    "norm": grad.norm().item(),
                    "mean": grad.mean().item(),
                    "std": grad.std().item(),
                    "has_nan": torch.isnan(grad).any().item(),
                    "has_inf": torch.isinf(grad).any().item(),
                    "shape": list(grad.shape),
                    "memory_mb": (grad.element_size() * grad.numel()) / (1024**2),
                }
                
                # Alert on issues
                if torch.isnan(grad).any():
                    logger.warning(f"⚠️  NaN gradient detected in {name}")
                if torch.isinf(grad).any():
                    logger.warning(f"⚠️  Inf gradient detected in {name}")
                if grad.norm().item() > 1000:
                    logger.warning(f"⚠️  Large gradient norm in {name}: {grad.norm().item():.2f}")
            
            return grad
        
        return hook_fn
    
    # Register hooks for all parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            hook = param.register_hook(create_backward_hook(name))
            hooks.append(hook)
    
    logger.info(f"Registered {len(hooks)} backward hooks for gradient monitoring")
    
    return hooks, gradient_stats


def wrap_backward_with_memory_tracking(
    loss_tensor: torch.Tensor,
    device: torch.device,
    logger: logging.Logger,
    batch_idx: int,
    epoch: int,
):
    """
    Wrap backward() call with detailed memory tracking.
    
    Usage:
        Instead of: loss.backward()
        Use: wrap_backward_with_memory_tracking(loss, device, logger, batch_idx, epoch)
    """
    
    # Pre-backward memory state
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        pre_allocated = torch.cuda.memory_allocated(device) / (1024**2)
        pre_reserved = torch.cuda.memory_reserved(device) / (1024**2)
        
        logger.debug(
            f"[Batch {batch_idx}] PRE-BACKWARD | "
            f"Allocated: {pre_allocated:.1f}MB, Reserved: {pre_reserved:.1f}MB"
        )
    
    # Perform backward
    try:
        loss_tensor.backward()
        
        # Post-backward memory state
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            post_allocated = torch.cuda.memory_allocated(device) / (1024**2)
            post_reserved = torch.cuda.memory_reserved(device) / (1024**2)
            
            growth_allocated = post_allocated - pre_allocated
            growth_reserved = post_reserved - pre_reserved
            
            logger.debug(
                f"[Batch {batch_idx}] POST-BACKWARD | "
                f"Allocated: {post_allocated:.1f}MB (+{growth_allocated:.1f}MB), "
                f"Reserved: {post_reserved:.1f}MB (+{growth_reserved:.1f}MB)"
            )
            
            # Alert on large growth
            if growth_allocated > 500:  # More than 500MB growth
                logger.warning(
                    f"⚠️  Large memory growth during backward at batch {batch_idx}: "
                    f"{growth_allocated:.1f}MB"
                )
    
    except RuntimeError as e:
        # Capture OOM details
        if "out of memory" in str(e).lower():
            logger.error("🚨 OOM ERROR DURING BACKWARD PASS!")
            logger.error(f"   Batch: {batch_idx}, Epoch: {epoch}")
            logger.error(f"   Error: {e}")
            
            if device.type == "cuda":
                logger.error(f"   Memory allocated: {torch.cuda.memory_allocated(device) / (1024**2):.1f}MB")
                logger.error(f"   Memory reserved: {torch.cuda.memory_reserved(device) / (1024**2):.1f}MB")
                logger.error(f"   Max memory allocated: {torch.cuda.max_memory_allocated(device) / (1024**2):.1f}MB")
                
                # Print CUDA memory summary
                try:
                    logger.error("\nCUDA Memory Summary:")
                    logger.error(torch.cuda.memory_summary(device=device))
                except:
                    pass
        
        # Re-raise
        raise


def print_usage_instructions():
    """Print instructions for using the backward debugging tools."""
    
    print("\n" + "="*80)
    print("BACKWARD PASS DEBUGGING - USAGE INSTRUCTIONS")
    print("="*80)
    
    print("\nOption 1: Use the enhanced training script")
    print("-" * 80)
    print("1. Import the enhanced training function:")
    print("   from scripts.train.train_with_backward_debugging import enhanced_train_epoch_with_debugging")
    print()
    print("2. Replace your train_epoch call with enhanced_train_epoch_with_debugging")
    print("   This will automatically add detailed memory tracking.")
    
    print("\nOption 2: Add debugging to existing code")
    print("-" * 80)
    print("1. Create a debugger:")
    print("   from utils.backward_memory_debugger import create_backward_debugger")
    print("   debugger = create_backward_debugger(device, checkpoint_dir)")
    print()
    print("2. Wrap critical sections:")
    print("   debugger.capture_pre_backward(batch_idx, epoch)")
    print("   loss.backward()")
    print("   debugger.capture_post_backward(batch_idx, epoch, model)")
    print()
    print("3. Save diagnostics:")
    print("   debugger.save_diagnostics()")
    print("   debugger.print_summary()")
    
    print("\nOption 3: Use monkey-patching (quick & dirty)")
    print("-" * 80)
    print("1. In your training script, add:")
    print("   from enable_backward_debugging import patch_backward_with_debugging")
    print("   debugger, counter = patch_backward_with_debugging(model, device, checkpoint_dir)")
    print()
    print("2. Update epoch counter when epochs change:")
    print("   counter['epoch'] = epoch")
    
    print("\nOption 4: Use backward hooks")
    print("-" * 80)
    print("1. Add gradient monitoring hooks:")
    print("   from enable_backward_debugging import create_debugging_hooks")
    print("   hooks, grad_stats = create_debugging_hooks(model, logger)")
    print()
    print("2. Hooks will automatically log gradient issues during training")
    
    print("\nOption 5: Wrap individual backward() calls")
    print("-" * 80)
    print("Replace:")
    print("   loss.backward()")
    print("With:")
    print("   from enable_backward_debugging import wrap_backward_with_memory_tracking")
    print("   wrap_backward_with_memory_tracking(loss, device, logger, batch_idx, epoch)")
    
    print("\n" + "="*80)
    print("After training, check the generated files:")
    print("  - backward_memory_diagnostics.json  (detailed snapshots)")
    print("  - Training logs will contain detailed memory progression")
    print("="*80 + "\n")


if __name__ == "__main__":
    print_usage_instructions()
    
    print("\nTo integrate with your training:")
    print("1. The easiest way is to use the enhanced training script")
    print("2. Or add debugging hooks/wrappers as shown above")
    print("\nFor immediate testing, run your training script with these tools imported.")
