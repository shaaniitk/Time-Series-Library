#!/usr/bin/env python3
"""
Memory-Optimized Training Script for Celestial Enhanced PGAT

This script addresses the >64GB memory issues by:
1. Using the memory-optimized model and configuration
2. Implementing aggressive memory management
3. Adding memory monitoring and cleanup
4. Using gradient checkpointing and mixed precision
"""

import os
import sys
import gc
import psutil
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import time
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from data_provider.data_factory import data_provider
from models.Celestial_Enhanced_PGAT_Memory_Optimized import Model
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor and manage memory usage during training."""
    
    def __init__(self, max_memory_gb: float = 32.0):
        self.max_memory_gb = max_memory_gb
        self.max_memory_bytes = max_memory_gb * 1024 ** 3
        self.process = psutil.Process(os.getpid())
    
    def get_memory_usage(self) -> dict:
        """Get current memory usage statistics."""
        memory_info = self.process.memory_info()
        
        stats = {
            'rss_gb': memory_info.rss / (1024 ** 3),
            'vms_gb': memory_info.vms / (1024 ** 3),
            'percent': self.process.memory_percent()
        }
        
        if torch.cuda.is_available():
            stats['cuda_allocated_gb'] = torch.cuda.memory_allocated() / (1024 ** 3)
            stats['cuda_reserved_gb'] = torch.cuda.memory_reserved() / (1024 ** 3)
        
        return stats
    
    def check_memory_limit(self) -> bool:
        """Check if memory usage exceeds limit."""
        memory_info = self.process.memory_info()
        return memory_info.rss > self.max_memory_bytes
    
    def cleanup_memory(self):
        """Aggressive memory cleanup."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def log_memory_stats(self, stage: str):
        """Log current memory statistics."""
        stats = self.get_memory_usage()
        logger.info(
            f"Memory [{stage}]: RSS={stats['rss_gb']:.2f}GB "
            f"VMS={stats['vms_gb']:.2f}GB "
            f"Percent={stats['percent']:.1f}%"
        )
        
        if 'cuda_allocated_gb' in stats:
            logger.info(
                f"CUDA [{stage}]: Allocated={stats['cuda_allocated_gb']:.2f}GB "
                f"Reserved={stats['cuda_reserved_gb']:.2f}GB"
            )


def load_config():
    """Load memory-optimized configuration."""
    config_path = "configs/celestial_enhanced_pgat_memory_optimized.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert to object for attribute access
    class Config:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    return Config(config)


def prepare_data(args, memory_monitor):
    """Prepare data loaders with memory monitoring."""
    memory_monitor.log_memory_stats("before_data_loading")
    
    # Load data
    train_data, train_loader = data_provider(args, flag='train')
    vali_data, vali_loader = data_provider(args, flag='val')
    test_data, test_loader = data_provider(args, flag='test')
    
    memory_monitor.log_memory_stats("after_data_loading")
    memory_monitor.cleanup_memory()
    
    logger.info(f"Data loaded - Train: {len(train_loader)}, Val: {len(vali_loader)}, Test: {len(test_loader)}")
    
    return (train_data, train_loader), (vali_data, vali_loader), (test_data, test_loader)


def create_model(args, device, memory_monitor):
    """Create memory-optimized model."""
    memory_monitor.log_memory_stats("before_model_creation")
    
    # Create model
    model = Model(args).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model created - Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    memory_monitor.log_memory_stats("after_model_creation")
    memory_monitor.cleanup_memory()
    
    return model


def train_epoch(model, train_loader, optimizer, criterion, scaler, device, 
                memory_monitor, clear_cache_every_n_batches=10):
    """Memory-optimized training epoch."""
    model.train()
    train_loss = 0.0
    batch_count = 0
    
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        # Move to device
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)
        
        # Prepare decoder input
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
        
        # Forward pass with mixed precision
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                loss = criterion(outputs, batch_y[:, -args.pred_len:, :])
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            loss = criterion(outputs, batch_y[:, -args.pred_len:, :])
            loss.backward()
            optimizer.step()
        
        train_loss += loss.item()
        batch_count += 1
        
        # Memory management
        if i % clear_cache_every_n_batches == 0:
            memory_monitor.cleanup_memory()
            
            # Check memory limit
            if memory_monitor.check_memory_limit():
                logger.warning(f"Memory limit exceeded at batch {i}, cleaning up...")
                memory_monitor.cleanup_memory()
        
        # Log progress
        if i % 50 == 0:
            logger.info(f"Batch {i}/{len(train_loader)}, Loss: {loss.item():.6f}")
            memory_monitor.log_memory_stats(f"batch_{i}")
    
    return train_loss / batch_count


def validate_epoch(model, vali_loader, criterion, device, memory_monitor):
    """Memory-optimized validation epoch."""
    model.eval()
    val_loss = 0.0
    batch_count = 0
    
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            loss = criterion(outputs, batch_y[:, -args.pred_len:, :])
            
            val_loss += loss.item()
            batch_count += 1
            
            # Periodic cleanup
            if i % 20 == 0:
                memory_monitor.cleanup_memory()
    
    return val_loss / batch_count


def evaluate_model(model, test_loader, device, args, memory_monitor):
    """Memory-optimized model evaluation."""
    model.eval()
    preds = []
    trues = []
    
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            pred = outputs.detach().cpu().numpy()
            true = batch_y[:, -args.pred_len:, :].detach().cpu().numpy()
            
            preds.append(pred)
            trues.append(true)
            
            # Cleanup every 10 batches
            if i % 10 == 0:
                memory_monitor.cleanup_memory()
    
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    
    # Calculate metrics
    mae, mse, rmse, mape, mspe = metric(preds, trues)
    
    logger.info(f"Test Results - MAE: {mae:.6f}, MSE: {mse:.6f}, RMSE: {rmse:.6f}")
    logger.info(f"Test Results - MAPE: {mape:.6f}, MSPE: {mspe:.6f}")
    
    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape, 'mspe': mspe}


def main():
    """Main training function with memory optimization."""
    # Load configuration
    args = load_config()
    
    # Add required attributes
    args.task_name = 'long_term_forecast'
    args.model_name = 'Celestial_Enhanced_PGAT_Memory_Optimized'
    args.data_name = 'custom'
    args.checkpoints = './checkpoints/'
    args.inverse = False
    args.cols = None
    args.num_workers = 0  # Disable multiprocessing to save memory
    args.itr = 1
    args.train_only = False
    args.do_predict = False
    args.model_id = f"celestial_memory_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize memory monitor
    max_memory_gb = getattr(args, 'max_memory_usage_gb', 32)
    memory_monitor = MemoryMonitor(max_memory_gb=max_memory_gb)
    
    logger.info("Starting Memory-Optimized Celestial Enhanced PGAT Training")
    logger.info(f"Memory limit: {max_memory_gb}GB")
    memory_monitor.log_memory_stats("startup")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(2024)
    np.random.seed(2024)
    
    # Prepare data
    (train_data, train_loader), (vali_data, vali_loader), (test_data, test_loader) = prepare_data(args, memory_monitor)
    
    # Create model
    model = create_model(args, device, memory_monitor)
    
    # Setup training components
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=getattr(args, 'weight_decay', 0.0001))
    criterion = nn.MSELoss()
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision and device.type == 'cuda' else None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoints) / args.model_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Training for {args.train_epochs} epochs")
    memory_monitor.log_memory_stats("training_start")
    
    # Training loop
    best_val_loss = float('inf')
    training_start_time = time.time()
    
    for epoch in range(args.train_epochs):
        epoch_start_time = time.time()
        
        logger.info(f"Epoch {epoch + 1}/{args.train_epochs}")
        memory_monitor.log_memory_stats(f"epoch_{epoch + 1}_start")
        
        # Training
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device, 
            memory_monitor, getattr(args, 'clear_cache_every_n_batches', 10)
        )
        
        # Validation
        val_loss = validate_epoch(model, vali_loader, criterion, device, memory_monitor)
        
        # Learning rate adjustment
        adjust_learning_rate(optimizer, epoch, args)
        
        epoch_time = time.time() - epoch_start_time
        logger.info(
            f"Epoch {epoch + 1} completed in {epoch_time:.2f}s - "
            f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
        )
        
        memory_monitor.log_memory_stats(f"epoch_{epoch + 1}_end")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_dir / 'best_model.pth')
            logger.info(f"New best model saved with val_loss: {val_loss:.6f}")
        
        # Early stopping
        early_stopping(val_loss, model, str(checkpoint_dir))
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break
        
        # Aggressive memory cleanup after each epoch
        memory_monitor.cleanup_memory()
    
    total_training_time = time.time() - training_start_time
    logger.info(f"Training completed in {total_training_time / 3600:.2f} hours")
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(checkpoint_dir / 'best_model.pth'))
    logger.info("Best model loaded for evaluation")
    
    # Final evaluation
    memory_monitor.log_memory_stats("evaluation_start")
    test_results = evaluate_model(model, test_loader, device, args, memory_monitor)
    memory_monitor.log_memory_stats("evaluation_end")
    
    # Save results
    results = {
        'model': 'Celestial_Enhanced_PGAT_Memory_Optimized',
        'config': args.__dict__,
        'training_time_hours': total_training_time / 3600,
        'best_val_loss': best_val_loss,
        'test_results': test_results,
        'memory_limit_gb': max_memory_gb
    }
    
    import json
    with open(checkpoint_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("Training completed successfully!")
    logger.info(f"Results saved to {checkpoint_dir}")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        exit(1)