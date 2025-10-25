"""
IMMEDIATE OOM DEBUGGING PATCH
Apply this to your running training to get detailed backward pass memory tracking

This script modifies the train_epoch function in train_celestial_production.py
to add comprehensive memory debugging around the backward pass.

Usage:
    1. Stop your current training
    2. Run: python patch_training_for_oom_debug.py
    3. Restart training: python scripts/train/train_celestial_production.py
"""

import sys
from pathlib import Path

# Ensure we can import from project
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_patched_backward_section():
    """
    Creates the patched code for the backward pass section with detailed debugging.
    """
    
    patched_code = '''
            # ============================================================
            # ENHANCED BACKWARD DEBUGGING SECTION
            # ============================================================
            
            # Pre-backward memory snapshot
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(device)
                pre_backward_allocated = torch.cuda.memory_allocated(device) / (1024**2)
                pre_backward_reserved = torch.cuda.memory_reserved(device) / (1024**2)
                
                if batch_index % 10 == 0 or batch_index < 5:
                    logger.info(
                        f"[DEBUG] Batch {batch_index} PRE-BACKWARD | "
                        f"Allocated: {pre_backward_allocated:.1f}MB, "
                        f"Reserved: {pre_backward_reserved:.1f}MB"
                    )
                    
                    # Log computational graph info
                    if hasattr(loss, 'grad_fn'):
                        logger.debug(f"  Loss grad_fn: {loss.grad_fn}")
                        logger.debug(f"  Loss requires_grad: {loss.requires_grad}")
                        logger.debug(f"  Loss size: {loss.shape if hasattr(loss, 'shape') else 'scalar'}")
            
            # Perform backward with error handling
            try:
                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                    
                # Post-backward memory snapshot
                if device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                    post_backward_allocated = torch.cuda.memory_allocated(device) / (1024**2)
                    post_backward_reserved = torch.cuda.memory_reserved(device) / (1024**2)
                    
                    backward_growth_allocated = post_backward_allocated - pre_backward_allocated
                    backward_growth_reserved = post_backward_reserved - pre_backward_reserved
                    
                    if batch_index % 10 == 0 or batch_index < 5:
                        logger.info(
                            f"[DEBUG] Batch {batch_index} POST-BACKWARD | "
                            f"Allocated: {post_backward_allocated:.1f}MB (+{backward_growth_allocated:.1f}MB), "
                            f"Reserved: {post_backward_reserved:.1f}MB (+{backward_growth_reserved:.1f}MB)"
                        )
                        
                        # Check gradients
                        num_params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
                        total_params = sum(1 for p in model.parameters())
                        logger.debug(f"  Parameters with gradients: {num_params_with_grad}/{total_params}")
                        
                        # Check for NaN/Inf gradients
                        nan_grads = sum(1 for p in model.parameters() if p.grad is not None and torch.isnan(p.grad).any())
                        inf_grads = sum(1 for p in model.parameters() if p.grad is not None and torch.isinf(p.grad).any())
                        
                        if nan_grads > 0:
                            logger.warning(f"  ⚠️  NaN gradients in {nan_grads} parameters!")
                        if inf_grads > 0:
                            logger.warning(f"  ⚠️  Inf gradients in {inf_grads} parameters!")
                        
                        # Alert on excessive memory growth
                        if backward_growth_allocated > 1000:
                            logger.warning(
                                f"  ⚠️  LARGE MEMORY GROWTH: {backward_growth_allocated:.1f}MB during backward!"
                            )
                        
                        # Calculate total gradient memory
                        grad_memory = sum(
                            p.grad.element_size() * p.grad.numel() / (1024**2)
                            for p in model.parameters()
                            if p.grad is not None
                        )
                        logger.debug(f"  Total gradient memory: {grad_memory:.1f}MB")
                    
            except RuntimeError as backward_error:
                # CAPTURE OOM ERROR DETAILS
                logger.error("="*80)
                logger.error("🚨 BACKWARD PASS FAILED - OOM ERROR!")
                logger.error("="*80)
                logger.error(f"Epoch: {epoch + 1}, Batch: {batch_index}/{len(train_loader)}")
                logger.error(f"Error: {backward_error}")
                logger.error("")
                
                if device.type == "cuda" and torch.cuda.is_available():
                    try:
                        current_allocated = torch.cuda.memory_allocated(device) / (1024**2)
                        current_reserved = torch.cuda.memory_reserved(device) / (1024**2)
                        max_allocated = torch.cuda.max_memory_allocated(device) / (1024**2)
                        max_reserved = torch.cuda.max_memory_reserved(device) / (1024**2)
                        
                        logger.error("CUDA Memory State at Failure:")
                        logger.error(f"  Current Allocated: {current_allocated:.1f}MB")
                        logger.error(f"  Current Reserved: {current_reserved:.1f}MB")
                        logger.error(f"  Max Allocated: {max_allocated:.1f}MB")
                        logger.error(f"  Max Reserved: {max_reserved:.1f}MB")
                        logger.error("")
                        
                        # Try to get device properties
                        try:
                            props = torch.cuda.get_device_properties(device)
                            total_memory = props.total_memory / (1024**2)
                            logger.error(f"  GPU Total Memory: {total_memory:.1f}MB")
                            logger.error(f"  Memory Utilization: {(max_allocated / total_memory * 100):.1f}%")
                        except:
                            pass
                        
                        logger.error("")
                        logger.error("Batch Information:")
                        logger.error(f"  Batch size: {batch_x.size(0)}")
                        logger.error(f"  Sequence length: {batch_x.size(1)}")
                        logger.error(f"  Input features: {batch_x.size(2)}")
                        logger.error(f"  Input tensor size: {batch_x.element_size() * batch_x.numel() / (1024**2):.1f}MB")
                        logger.error(f"  Target tensor size: {batch_y.element_size() * batch_y.numel() / (1024**2):.1f}MB")
                        logger.error("")
                        
                        logger.error("Model Output Information:")
                        if 'outputs_tensor' in locals():
                            logger.error(f"  Output shape: {outputs_tensor.shape}")
                            logger.error(f"  Output size: {outputs_tensor.element_size() * outputs_tensor.numel() / (1024**2):.1f}MB")
                        logger.error(f"  Loss value: {raw_loss.item():.6f}")
                        logger.error(f"  Scaled loss: {loss.item():.6f}")
                        logger.error("")
                        
                        # Print memory summary
                        logger.error("CUDA Memory Summary:")
                        logger.error(torch.cuda.memory_summary(device=device, abbreviated=True))
                        
                    except Exception as summary_error:
                        logger.error(f"Failed to capture memory summary: {summary_error}")
                
                logger.error("="*80)
                logger.error("SUGGESTED ACTIONS:")
                logger.error("1. Reduce batch size in config")
                logger.error("2. Increase gradient_accumulation_steps")
                logger.error("3. Enable mixed precision training (use_amp=True)")
                logger.error("4. Use gradient checkpointing")
                logger.error("5. Reduce model size (d_model, n_heads, e_layers)")
                logger.error("="*80)
                
                # Clean up and raise
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                raise backward_error
    '''
    
    return patched_code


def show_instructions():
    """Show instructions for using this patch."""
    
    print("\n" + "="*80)
    print("BACKWARD PASS OOM DEBUGGING - MANUAL PATCH INSTRUCTIONS")
    print("="*80)
    print()
    print("This patch adds detailed debugging around the backward() call in your training.")
    print()
    print("STEP 1: Locate the backward pass in train_celestial_production.py")
    print("-" * 80)
    print("Find these lines (around line 763):")
    print()
    print("    if use_amp and scaler is not None:")
    print("        scaler.scale(loss).backward()")
    print("    else:")
    print("        loss.backward()")
    print()
    print("STEP 2: Replace with enhanced debugging version")
    print("-" * 80)
    print("Replace those lines with the code shown below.")
    print()
    print("="*80)
    print("ENHANCED CODE TO INSERT:")
    print("="*80)
    print(create_patched_backward_section())
    print("="*80)
    print()
    print("STEP 3: Run training and collect diagnostics")
    print("-" * 80)
    print("When you run training, you'll see detailed output like:")
    print()
    print("  [DEBUG] Batch 15 PRE-BACKWARD | Allocated: 2450.3MB, Reserved: 2600.0MB")
    print("  [DEBUG] Batch 15 POST-BACKWARD | Allocated: 3821.7MB (+1371.4MB), Reserved: 3900.0MB (+1300.0MB)")
    print()
    print("If OOM occurs, you'll see a detailed error report showing:")
    print("  - Exact memory state when OOM happened")
    print("  - Batch and tensor sizes")
    print("  - Memory growth during backward")
    print("  - Suggested actions to fix")
    print()
    print("="*80)
    print()
    print("ALTERNATIVE: Use automated tools")
    print("-" * 80)
    print("Instead of manual patching, you can:")
    print()
    print("1. Import the enhanced training function:")
    print("   from scripts.train.train_with_backward_debugging import enhanced_train_epoch_with_debugging")
    print()
    print("2. Use backward debugging tools:")
    print("   from utils.backward_memory_debugger import create_backward_debugger")
    print()
    print("3. Apply monkey patch:")
    print("   from enable_backward_debugging import patch_backward_with_debugging")
    print()
    print("="*80)
    print()


def create_quick_debug_wrapper():
    """Create a wrapper script that can be run immediately."""
    
    wrapper_code = """#!/usr/bin/env python3
'''
Quick OOM Debug Wrapper
Wraps existing training with memory debugging
'''

import sys
import torch
from pathlib import Path

# Import original training function
from scripts.train import train_celestial_production

# Patch torch.Tensor.backward to add debugging
original_backward = torch.Tensor.backward

def debug_backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None):
    '''Wrapped backward with debugging.'''
    device = self.device
    
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        pre = torch.cuda.memory_allocated(device) / (1024**2)
        
        # Call original
        original_backward(self, gradient, retain_graph, create_graph, inputs)
        
        torch.cuda.synchronize(device)
        post = torch.cuda.memory_allocated(device) / (1024**2)
        growth = post - pre
        
        if growth > 100:  # Alert on >100MB growth
            print(f"  [BACKWARD DEBUG] Memory growth: {growth:.1f}MB (Pre: {pre:.1f}MB, Post: {post:.1f}MB)")
    else:
        original_backward(self, gradient, retain_graph, create_graph, inputs)

# Apply patch
torch.Tensor.backward = debug_backward

# Run original training
if __name__ == "__main__":
    train_celestial_production.train_celestial_pgat_production()
"""
    
    wrapper_path = Path("run_training_with_oom_debug.py")
    wrapper_path.write_text(wrapper_code)
    
    print(f"\n✅ Created wrapper script: {wrapper_path}")
    print(f"   Run with: python {wrapper_path}")
    print()


if __name__ == "__main__":
    show_instructions()
    
    # Ask if user wants to create wrapper
    print("\n" + "="*80)
    response = input("Create automated debug wrapper? (y/n): ").strip().lower()
    if response == 'y':
        create_quick_debug_wrapper()
    
    print("\nDone! Follow the instructions above to enable backward debugging.")
    print("="*80 + "\n")
