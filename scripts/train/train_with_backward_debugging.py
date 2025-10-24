"""
Training Script with Enhanced Backward Pass Memory Debugging

This script augments the production training with granular memory tracking
specifically focused on identifying OOM issues during backward pass.

Key Enhancements:
1. Pre/post backward memory snapshots
2. Gradient accumulation state tracking
3. Intermediate activation monitoring
4. Automatic memory leak detection
5. Detailed tensor lifecycle tracking
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
from typing import Any, Dict, Optional
import logging

# Import the backward memory debugger
from utils.backward_memory_debugger import BackwardMemoryDebugger, create_backward_debugger

LOGGER = logging.getLogger(__name__)


def enhanced_train_epoch_with_debugging(
    model: nn.Module,
    train_loader: Any,
    optimizer: Any,
    criterion: nn.Module,
    scaler: Optional[Any],
    use_amp: bool,
    gradient_accumulation_steps: int,
    device: torch.device,
    epoch: int,
    args: Any,
    logger: logging.Logger,
    target_scaler: Any,
    target_indices: Any,
    total_train_batches: int,
    full_cycles: int,
    remainder_batches: int,
    epoch_start: float,
    sequential_mixture_loss_cls: Optional[type],
    mixture_loss_cls: Optional[type],
    checkpoint_dir: Path,
) -> tuple:
    """
    Enhanced training epoch with comprehensive backward pass debugging.
    
    This function wraps the training loop with detailed memory tracking
    at each critical point during forward and backward propagation.
    """
    
    # Create backward debugger
    backward_debugger = create_backward_debugger(device, checkpoint_dir)
    
    model.train()
    train_loss = 0.0
    train_batches = 0
    optimizer.zero_grad(set_to_none=True)
    
    logger.info(f"🔍 Enhanced backward debugging enabled for epoch {epoch + 1}")
    logger.info(f"   Memory snapshots will be saved to: {checkpoint_dir / 'backward_memory_diagnostics.json'}")
    
    # Determine how frequently to do detailed debugging
    # More frequent early on, less frequent later
    if epoch == 0:
        debug_frequency = 1  # Every batch in first epoch
    elif epoch < 5:
        debug_frequency = 10  # Every 10 batches
    else:
        debug_frequency = 25  # Every 25 batches
    
    log_interval = max(1, int(getattr(args, "log_interval", 10)))
    
    for batch_index, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        do_detailed_debug = (batch_index % debug_frequency == 0) or (batch_index < 3)
        
        try:
            # ============================================================
            # STAGE 1: PRE-FORWARD - Capture initial state
            # ============================================================
            if do_detailed_debug:
                pre_forward_snapshot = backward_debugger.capture_pre_forward(batch_index, epoch)
                logger.debug(
                    f"[Batch {batch_index}] PRE-FORWARD | "
                    f"CUDA: {pre_forward_snapshot.get('cuda_allocated_mb', 0):.1f}MB allocated, "
                    f"Tensors: {pre_forward_snapshot.get('num_tensors', 0)}"
                )
            
            # Move data to device
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            # Create decoder input
            from scripts.train.train_celestial_production import _create_enhanced_decoder_input
            dec_inp = _create_enhanced_decoder_input(batch_y, args, logger).float().to(device)
            
            # ============================================================
            # STAGE 2: FORWARD PASS
            # ============================================================
            if use_amp:
                import torch.cuda.amp as torch_amp
                with torch_amp.autocast():
                    outputs_raw = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                outputs_raw = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            # Normalize outputs
            from scripts.train.train_celestial_production import _normalize_model_output, scale_targets_for_loss
            outputs_tensor, aux_loss, mdn_outputs, _ = _normalize_model_output(outputs_raw)
            
            # Compute loss
            c_out_evaluation = len(target_indices)
            y_true_for_loss = scale_targets_for_loss(
                batch_y[:, -args.pred_len:, :], target_scaler, target_indices, device
            )
            
            is_seq_mixture = sequential_mixture_loss_cls is not None and isinstance(criterion, sequential_mixture_loss_cls)
            is_mixture = mixture_loss_cls is not None and isinstance(criterion, mixture_loss_cls)
            
            if is_seq_mixture or is_mixture or mdn_outputs is not None:
                if mdn_outputs is None:
                    raise ValueError("MixtureNLLLoss requires model to return MDN outputs")
                means_t, stds_t, weights_t = mdn_outputs
                if means_t.size(1) > args.pred_len:
                    means_t = means_t[:, -args.pred_len:, ...]
                    stds_t = stds_t[:, -args.pred_len:, ...]
                    weights_t = weights_t[:, -args.pred_len:, ...]
                targets_t = (
                    y_true_for_loss.squeeze(-1)
                    if y_true_for_loss.dim() == 3 and y_true_for_loss.size(-1) == 1
                    else y_true_for_loss
                )
                raw_loss = criterion((means_t, stds_t, weights_t), targets_t)
            else:
                y_pred_for_loss = outputs_tensor[:, -args.pred_len:, :c_out_evaluation]
                raw_loss = criterion(y_pred_for_loss, y_true_for_loss)
            
            if aux_loss:
                raw_loss = raw_loss + aux_loss
            
            if getattr(model, "use_stochastic_learner", False):
                reg_loss = model.get_regularization_loss()
                reg_weight = float(getattr(args, "reg_loss_weight", 0.0005))
                raw_loss = raw_loss + (reg_loss * reg_weight)
            
            # ============================================================
            # STAGE 3: POST-FORWARD - Before backward
            # ============================================================
            if do_detailed_debug:
                post_forward_snapshot = backward_debugger.capture_post_forward(
                    batch_index, epoch, outputs_raw, raw_loss
                )
                
                # Calculate memory growth during forward
                forward_mem_growth = (
                    post_forward_snapshot.get('cuda_allocated_mb', 0) - 
                    pre_forward_snapshot.get('cuda_allocated_mb', 0)
                )
                
                logger.debug(
                    f"[Batch {batch_index}] POST-FORWARD | "
                    f"CUDA: {post_forward_snapshot.get('cuda_allocated_mb', 0):.1f}MB (+{forward_mem_growth:.1f}MB), "
                    f"Loss: {post_forward_snapshot.get('loss_value', 0):.6f}, "
                    f"Output shape: {post_forward_snapshot.get('output_shape', 'N/A')}"
                )
            
            # Scale loss for gradient accumulation
            cycle_index = batch_index // gradient_accumulation_steps
            effective_cycle = gradient_accumulation_steps
            if remainder_batches and cycle_index >= full_cycles:
                effective_cycle = remainder_batches
            
            loss = raw_loss / effective_cycle
            
            # ============================================================
            # STAGE 4: PRE-BACKWARD - Critical snapshot
            # ============================================================
            if do_detailed_debug:
                pre_backward_snapshot = backward_debugger.capture_pre_backward(batch_index, epoch)
                logger.debug(
                    f"[Batch {batch_index}] PRE-BACKWARD | "
                    f"CUDA: {pre_backward_snapshot.get('cuda_allocated_mb', 0):.1f}MB, "
                    f"Tensors in grad graph: {pre_backward_snapshot.get('num_tensors_in_grad_graph', 0)}, "
                    f"Gradient tensors: {pre_backward_snapshot.get('num_gradient_tensors', 0)}"
                )
            
            # ============================================================
            # STAGE 5: BACKWARD PASS - THE CRITICAL OPERATION
            # ============================================================
            try:
                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                    
            except RuntimeError as backward_error:
                # Capture error state
                logger.error(f"🚨 BACKWARD PASS FAILED at batch {batch_index}!")
                logger.error(f"   Error: {backward_error}")
                
                if do_detailed_debug:
                    error_snapshot = backward_debugger.capture_post_backward(batch_index, epoch, model)
                    logger.error(
                        f"   Memory at error: {error_snapshot.get('cuda_allocated_mb', 0):.1f}MB"
                    )
                    logger.error(
                        f"   Max allocated: {error_snapshot.get('cuda_max_allocated_mb', 0):.1f}MB"
                    )
                
                # Save diagnostics immediately
                backward_debugger.save_diagnostics()
                backward_debugger.print_summary()
                
                # Re-raise the error
                raise
            
            # ============================================================
            # STAGE 6: POST-BACKWARD - Analyze gradient computation
            # ============================================================
            if do_detailed_debug:
                post_backward_snapshot = backward_debugger.capture_post_backward(
                    batch_index, epoch, model
                )
                
                # Calculate memory growth during backward
                backward_mem_growth = (
                    post_backward_snapshot.get('cuda_allocated_mb', 0) - 
                    pre_backward_snapshot.get('cuda_allocated_mb', 0)
                )
                
                logger.debug(
                    f"[Batch {batch_index}] POST-BACKWARD | "
                    f"CUDA: {post_backward_snapshot.get('cuda_allocated_mb', 0):.1f}MB (+{backward_mem_growth:.1f}MB), "
                    f"Params with grad: {post_backward_snapshot.get('params_with_grad', 0)}, "
                    f"Total grad norm: {post_backward_snapshot.get('total_gradient_norm', 0):.4f}, "
                    f"NaN grads: {post_backward_snapshot.get('params_with_nan_grad', 0)}, "
                    f"Inf grads: {post_backward_snapshot.get('params_with_inf_grad', 0)}"
                )
                
                # Alert on concerning patterns
                if backward_mem_growth > 1000:  # More than 1GB growth
                    logger.warning(
                        f"⚠️  Large memory growth during backward: {backward_mem_growth:.1f}MB"
                    )
                
                if post_backward_snapshot.get('params_with_nan_grad', 0) > 0:
                    logger.warning(
                        f"⚠️  NaN gradients detected in {post_backward_snapshot.get('params_with_nan_grad', 0)} parameters"
                    )
            
            # ============================================================
            # STAGE 7: OPTIMIZER STEP (if accumulation complete)
            # ============================================================
            is_cycle_end = (batch_index + 1) % gradient_accumulation_steps == 0
            is_final_partial = (
                remainder_batches and cycle_index >= full_cycles and batch_index == total_train_batches - 1
            )
            
            if is_cycle_end or is_final_partial:
                # Gradient clipping
                if hasattr(args, "clip_grad_norm"):
                    clip_value = float(getattr(args, "clip_grad_norm", 0.0))
                    if use_amp and scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                
                # Optimizer step
                if use_amp and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                # Zero gradients for next accumulation cycle
                if batch_index != total_train_batches - 1:
                    optimizer.zero_grad(set_to_none=True)
                
                # ============================================================
                # STAGE 8: POST-OPTIMIZER - Check for memory cleanup
                # ============================================================
                if do_detailed_debug:
                    post_optimizer_snapshot = backward_debugger.capture_post_optimizer_step(
                        batch_index, epoch
                    )
                    
                    logger.debug(
                        f"[Batch {batch_index}] POST-OPTIMIZER | "
                        f"CUDA: {post_optimizer_snapshot.get('cuda_allocated_mb', 0):.1f}MB, "
                        f"Tensors: {post_optimizer_snapshot.get('num_tensors', 0)}"
                    )
            
            # Update training metrics
            train_loss += raw_loss.detach().item()
            train_batches += 1
            
            # Regular logging
            if batch_index % log_interval == 0:
                import time
                elapsed = time.time() - epoch_start
                logger.info(
                    f"Batch {batch_index}/{len(train_loader)} | "
                    f"loss={raw_loss.detach().item():.6f} | "
                    f"elapsed={elapsed:.1f}s"
                )
            
            # Periodic memory leak check
            if batch_index > 0 and batch_index % 50 == 0:
                leaks = backward_debugger.find_memory_leaks()
                if leaks:
                    logger.warning(f"⚠️  Potential memory leaks detected at batch {batch_index}:")
                    for leak in leaks:
                        logger.warning(f"   - {leak['type']}: {leak['description']}")
        
        except Exception as exc:
            logger.exception(f"Training step failed at batch {batch_index}: {exc}")
            
            # Save diagnostics on any error
            backward_debugger.save_diagnostics()
            backward_debugger.print_summary()
            
            # Clean up and continue
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
    
    # End of epoch - save final diagnostics
    backward_debugger.save_diagnostics()
    backward_debugger.print_summary()
    
    avg_train_loss = train_loss / max(train_batches, 1)
    return avg_train_loss, train_batches


# Additional utility functions for memory analysis
def analyze_model_memory_footprint(model: nn.Module, device: torch.device, logger: logging.Logger) -> Dict[str, Any]:
    """Analyze the memory footprint of different model components."""
    
    memory_breakdown = {
        "total_params": 0,
        "total_memory_mb": 0.0,
        "layer_breakdown": [],
    }
    
    logger.info("Analyzing model memory footprint...")
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            num_params = sum(p.numel() for p in module.parameters())
            param_memory = sum(p.numel() * p.element_size() for p in module.parameters()) / (1024**2)
            
            if num_params > 0:
                layer_info = {
                    "name": name,
                    "type": type(module).__name__,
                    "num_params": num_params,
                    "memory_mb": round(param_memory, 2),
                }
                memory_breakdown["layer_breakdown"].append(layer_info)
                memory_breakdown["total_params"] += num_params
                memory_breakdown["total_memory_mb"] += param_memory
    
    # Sort by memory usage
    memory_breakdown["layer_breakdown"].sort(key=lambda x: x["memory_mb"], reverse=True)
    memory_breakdown["total_memory_mb"] = round(memory_breakdown["total_memory_mb"], 2)
    
    # Log top memory consumers
    logger.info(f"Total model parameters: {memory_breakdown['total_params']:,}")
    logger.info(f"Total model memory: {memory_breakdown['total_memory_mb']:.2f} MB")
    logger.info("Top 10 memory-consuming layers:")
    for i, layer in enumerate(memory_breakdown["layer_breakdown"][:10], 1):
        logger.info(f"  {i}. {layer['name']} ({layer['type']}): {layer['memory_mb']:.2f} MB")
    
    return memory_breakdown


if __name__ == "__main__":
    print("This module provides enhanced training functions with backward debugging.")
    print("Import and use enhanced_train_epoch_with_debugging() in your training script.")
