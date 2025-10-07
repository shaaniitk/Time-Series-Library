#!/usr/bin/env python3
"""
Debug Training Script with Detailed Logging
Logs values at each stage to verify model training is working correctly
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import logging
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from data_provider.data_factory import data_provider
from models.Enhanced_SOTA_PGAT import Enhanced_SOTA_PGAT
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import yaml

# Setup detailed logging
def setup_logging():
    """Setup comprehensive logging for debug analysis"""
    log_dir = Path("logs/debug_training")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"debug_training_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__), log_file

def log_tensor_stats(tensor, name, logger):
    """Log comprehensive tensor statistics"""
    if tensor is None:
        logger.debug(f"{name}: None")
        return
    
    if isinstance(tensor, (list, tuple)):
        logger.debug(f"{name}: List/Tuple with {len(tensor)} elements")
        for i, t in enumerate(tensor[:3]):  # Log first 3 elements
            if torch.is_tensor(t):
                log_tensor_stats(t, f"{name}[{i}]", logger)
        return
    
    if not torch.is_tensor(tensor):
        logger.debug(f"{name}: {type(tensor)} = {tensor}")
        return
    
    stats = {
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype),
        'device': str(tensor.device),
        'requires_grad': tensor.requires_grad,
        'mean': float(tensor.mean().item()) if tensor.numel() > 0 else 0.0,
        'std': float(tensor.std().item()) if tensor.numel() > 0 else 0.0,
        'min': float(tensor.min().item()) if tensor.numel() > 0 else 0.0,
        'max': float(tensor.max().item()) if tensor.numel() > 0 else 0.0,
        'has_nan': bool(torch.isnan(tensor).any()),
        'has_inf': bool(torch.isinf(tensor).any()),
        'zero_fraction': float((tensor == 0).float().mean()) if tensor.numel() > 0 else 0.0
    }
    
    logger.debug(f"{name}: {stats}")
    return stats

def log_model_parameters(model, logger):
    """Log model parameter statistics"""
    logger.info("=== MODEL PARAMETER ANALYSIS ===")
    
    total_params = 0
    trainable_params = 0
    param_stats = {}
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        if param.requires_grad:
            trainable_params += param_count
            
        stats = log_tensor_stats(param, f"param_{name}", logger)
        param_stats[name] = stats
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    return param_stats

def log_gradients(model, logger, epoch, batch_idx):
    """Log gradient statistics"""
    logger.debug(f"=== GRADIENT ANALYSIS - Epoch {epoch}, Batch {batch_idx} ===")
    
    grad_stats = {}
    total_grad_norm = 0.0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm ** 2
            
            stats = log_tensor_stats(param.grad, f"grad_{name}", logger)
            grad_stats[name] = stats
        else:
            logger.debug(f"grad_{name}: None (no gradient)")
    
    total_grad_norm = total_grad_norm ** 0.5
    logger.info(f"Total gradient norm: {total_grad_norm:.6f}")
    
    return grad_stats, total_grad_norm

def debug_forward_pass(model, batch_x, batch_y, logger, epoch, batch_idx):
    """Debug a single forward pass with detailed logging"""
    logger.debug(f"=== FORWARD PASS DEBUG - Epoch {epoch}, Batch {batch_idx} ===")
    
    # Log input data
    log_tensor_stats(batch_x, "input_batch_x", logger)
    log_tensor_stats(batch_y, "input_batch_y", logger)
    
    # Enhanced PGAT requires both wave_window and target_window
    # For forecasting, we use batch_x as wave_window and create target_window from it
    wave_window = batch_x
    # Create target_window by taking the last part of the sequence
    target_window = batch_x[:, -batch_y.shape[1]:, :]  # Use last pred_len timesteps
    
    log_tensor_stats(wave_window, "wave_window", logger)
    log_tensor_stats(target_window, "target_window", logger)
    
    # Hook to capture intermediate outputs
    intermediate_outputs = {}
    
    def create_hook(name):
        def hook(module, input, output):
            intermediate_outputs[name] = output
            log_tensor_stats(output, f"intermediate_{name}", logger)
        return hook
    
    # Register hooks on key components
    hooks = []
    if hasattr(model, 'enc_embedding'):
        hooks.append(model.enc_embedding.register_forward_hook(create_hook('enc_embedding')))
    if hasattr(model, 'encoder'):
        hooks.append(model.encoder.register_forward_hook(create_hook('encoder')))
    if hasattr(model, 'decoder'):
        hooks.append(model.decoder.register_forward_hook(create_hook('decoder')))
    if hasattr(model, 'projection'):
        hooks.append(model.projection.register_forward_hook(create_hook('projection')))
    
    try:
        # Forward pass
        logger.debug("Starting forward pass...")
        outputs = model(wave_window, target_window)
        log_tensor_stats(outputs, "model_output", logger)
        
        # Calculate loss with shape handling
        criterion = nn.MSELoss()
        
        # Debug shape mismatch
        logger.info(f"Output shape: {outputs.shape}")
        logger.info(f"Target shape: {batch_y.shape}")
        
        # Handle shape mismatch - adjust target to match output
        if outputs.shape != batch_y.shape:
            logger.info(f"Shape mismatch detected. Adjusting target...")
            logger.info(f"Output dimensions: batch={outputs.shape[0]}, seq_len={outputs.shape[1]}, features={outputs.shape[2]}")
            logger.info(f"Target dimensions: batch={batch_y.shape[0]}, seq_len={batch_y.shape[1]}, features={batch_y.shape[2]}")
            
            # Handle sequence length mismatch first
            if outputs.shape[1] != batch_y.shape[1]:
                logger.info(f"Sequence length mismatch: output={outputs.shape[1]}, target={batch_y.shape[1]}")
                # Take the last pred_len timesteps from target to match output
                pred_len = outputs.shape[1]
                batch_y = batch_y[:, -pred_len:, :]
                logger.info(f"Adjusted target sequence length to: {batch_y.shape}")
            
            # Handle feature dimension mismatch
            if outputs.shape[-1] != batch_y.shape[-1]:
                c_out = outputs.shape[-1]  # Use actual output feature count
                if batch_y.shape[-1] > c_out:
                    # Take the last c_out features (target features)
                    batch_y = batch_y[:, :, -c_out:]
                    logger.info(f"Adjusted target features to: {batch_y.shape}")
                elif batch_y.shape[-1] < c_out:
                    # Pad target if needed
                    padding_size = c_out - batch_y.shape[-1]
                    padding = torch.zeros(batch_y.shape[0], batch_y.shape[1], padding_size, device=batch_y.device)
                    batch_y = torch.cat([batch_y, padding], dim=-1)
                    logger.info(f"Padded target features to: {batch_y.shape}")
            
            logger.info(f"Final shapes - Output: {outputs.shape}, Target: {batch_y.shape}")
        
        loss = criterion(outputs, batch_y)
        logger.info(f"Loss: {loss.item():.6f}")
        
        return outputs, loss, intermediate_outputs
        
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()

def run_debug_training():
    """Run training with comprehensive debug logging"""
    logger, log_file = setup_logging()
    logger.info("Starting debug training session")
    logger.info(f"Log file: {log_file}")
    
    # Load configuration
    config_path = "configs/enhanced_sota_pgat_simplified.yaml"
    logger.info(f"Loading config from: {config_path}")
    
    with open(config_path, 'r') as f:
        args = yaml.safe_load(f)
    
    # Convert to namespace for compatibility
    class Args:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    args = Args(**args)
    logger.info(f"Configuration loaded: {vars(args)}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("=== DATA LOADING ===")
    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')
    
    logger.info(f"Train dataset size: {len(train_data)}")
    logger.info(f"Validation dataset size: {len(vali_data)}")
    logger.info(f"Test dataset size: {len(test_data)}")
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(vali_loader)}")
    
    # Initialize model
    logger.info("=== MODEL INITIALIZATION ===")
    model = Enhanced_SOTA_PGAT(args).to(device)
    logger.info(f"Model created and moved to {device}")
    
    # Log model parameters
    param_stats = log_model_parameters(model, logger)
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Training epochs: {args.train_epochs}")
    
    # Training metrics tracking
    training_metrics = {
        'epoch_losses': [],
        'batch_losses': [],
        'gradient_norms': [],
        'learning_rates': [],
        'validation_losses': []
    }
    
    # Training loop
    logger.info("=== STARTING TRAINING ===")
    
    for epoch in range(args.train_epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"EPOCH {epoch + 1}/{args.train_epochs}")
        logger.info(f"{'='*50}")
        
        model.train()
        train_loss = []
        epoch_gradient_norms = []
        
        for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            # Move to device
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            
            # Log every 10th batch in detail
            if batch_idx % 10 == 0:
                logger.info(f"\n--- Batch {batch_idx + 1}/{len(train_loader)} ---")
                
                # Debug forward pass
                outputs, loss, intermediates = debug_forward_pass(
                    model, batch_x, batch_y, logger, epoch + 1, batch_idx + 1
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Log gradients
                grad_stats, grad_norm = log_gradients(model, logger, epoch + 1, batch_idx + 1)
                epoch_gradient_norms.append(grad_norm)
                
                # Optimizer step
                optimizer.step()
                
                train_loss.append(loss.item())
                training_metrics['batch_losses'].append({
                    'epoch': epoch + 1,
                    'batch': batch_idx + 1,
                    'loss': loss.item(),
                    'gradient_norm': grad_norm
                })
                
                logger.info(f"Batch {batch_idx + 1} - Loss: {loss.item():.6f}, Grad Norm: {grad_norm:.6f}")
            
            else:
                # Regular training step without detailed logging
                wave_window = batch_x
                target_window = batch_x[:, -batch_y.shape[1]:, :]
                outputs = model(wave_window, target_window)
                
                # Handle shape mismatch for regular steps too
                if outputs.shape != batch_y.shape:
                    # Handle sequence length mismatch
                    if outputs.shape[1] != batch_y.shape[1]:
                        pred_len = outputs.shape[1]
                        batch_y = batch_y[:, -pred_len:, :]
                    # Handle feature dimension mismatch
                    c_out = outputs.shape[-1]
                    if batch_y.shape[-1] > c_out:
                        batch_y = batch_y[:, :, -c_out:]
                
                loss = criterion(outputs, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss.append(loss.item())
        
        # Epoch summary
        avg_train_loss = np.mean(train_loss)
        avg_grad_norm = np.mean(epoch_gradient_norms) if epoch_gradient_norms else 0.0
        
        logger.info(f"\nEpoch {epoch + 1} Training Summary:")
        logger.info(f"Average Loss: {avg_train_loss:.6f}")
        logger.info(f"Average Gradient Norm: {avg_grad_norm:.6f}")
        logger.info(f"Batches processed: {len(train_loss)}")
        
        training_metrics['epoch_losses'].append(avg_train_loss)
        training_metrics['gradient_norms'].append(avg_grad_norm)
        training_metrics['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Validation
        logger.info("\n=== VALIDATION ===")
        model.eval()
        vali_loss = []
        
        with torch.no_grad():
            for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                
                wave_window = batch_x
                target_window = batch_x[:, -batch_y.shape[1]:, :]
                outputs = model(wave_window, target_window)
                
                # Handle shape mismatch
                if outputs.shape != batch_y.shape:
                    # Handle sequence length mismatch
                    if outputs.shape[1] != batch_y.shape[1]:
                        pred_len = outputs.shape[1]
                        batch_y = batch_y[:, -pred_len:, :]
                    # Handle feature dimension mismatch
                    c_out = outputs.shape[-1]
                    if batch_y.shape[-1] > c_out:
                        batch_y = batch_y[:, :, -c_out:]
                
                loss = criterion(outputs, batch_y)
                vali_loss.append(loss.item())
                
                if batch_idx == 0:  # Log first validation batch
                    log_tensor_stats(outputs, "validation_output", logger)
                    logger.info(f"Validation batch 1 loss: {loss.item():.6f}")
        
        avg_vali_loss = np.mean(vali_loss)
        training_metrics['validation_losses'].append(avg_vali_loss)
        
        logger.info(f"Validation Loss: {avg_vali_loss:.6f}")
        
        # Early stopping check
        early_stopping(avg_vali_loss, model, args.checkpoints)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break
        
        # Learning rate adjustment
        adjust_learning_rate(optimizer, epoch + 1, args)
        new_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Learning rate adjusted to: {new_lr}")
    
    # Save training metrics
    metrics_file = f"logs/debug_training/training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(metrics_file, 'w') as f:
        json.dump(training_metrics, f, indent=2)
    
    logger.info(f"Training metrics saved to: {metrics_file}")
    
    # Final test
    logger.info("\n=== FINAL TEST ===")
    model.eval()
    test_loss = []
    
    with torch.no_grad():
        for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            
            wave_window = batch_x
            target_window = batch_x[:, -batch_y.shape[1]:, :]
            outputs = model(wave_window, target_window)
            
            # Handle shape mismatch
            if outputs.shape != batch_y.shape:
                # Handle sequence length mismatch
                if outputs.shape[1] != batch_y.shape[1]:
                    pred_len = outputs.shape[1]
                    batch_y = batch_y[:, -pred_len:, :]
                # Handle feature dimension mismatch
                c_out = outputs.shape[-1]
                if batch_y.shape[-1] > c_out:
                    batch_y = batch_y[:, :, -c_out:]
            
            loss = criterion(outputs, batch_y)
            test_loss.append(loss.item())
            
            if batch_idx == 0:  # Log first test batch
                log_tensor_stats(outputs, "test_output", logger)
                logger.info(f"Test batch 1 loss: {loss.item():.6f}")
    
    avg_test_loss = np.mean(test_loss)
    logger.info(f"Final Test Loss: {avg_test_loss:.6f}")
    
    logger.info("=== TRAINING COMPLETE ===")
    logger.info(f"Final Results:")
    logger.info(f"  Training Loss: {training_metrics['epoch_losses'][-1]:.6f}")
    logger.info(f"  Validation Loss: {training_metrics['validation_losses'][-1]:.6f}")
    logger.info(f"  Test Loss: {avg_test_loss:.6f}")
    logger.info(f"  Total Epochs: {len(training_metrics['epoch_losses'])}")
    
    return training_metrics, log_file

if __name__ == "__main__":
    try:
        metrics, log_file = run_debug_training()
        print(f"\nDebug training completed successfully!")
        print(f"Detailed logs saved to: {log_file}")
        print(f"Training metrics available in logs/debug_training/")
    except Exception as e:
        print(f"Error during debug training: {e}")
        import traceback
        traceback.print_exc()