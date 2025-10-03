"""Memory-optimized PGAT training script with automatic memory management."""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any

import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train.memory_efficient_gradient_tracker import MemoryEfficientGradientTracker
from models.MemoryOptimizedPGAT import MemoryOptimizedPGAT
from models.SimplifiedPGAT import SimplifiedPGAT
from data_provider.data_factory import setup_financial_forecasting_data
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

# Apply scaler patch to fix tensor reshaping issues
from utils.scaler_manager_patch import apply_scaler_patch
apply_scaler_patch()

# Import the original PGAT model to ensure it's available
try:
    from models.SOTA_Temporal_PGAT import SOTA_Temporal_PGAT
except ImportError:
    pass


class MemoryOptimizedExperiment(Exp_Long_Term_Forecast):
    """Memory-optimized experiment with automatic batch size scaling."""
    
    def __init__(self, args: SimpleNamespace):
        self.gradient_tracker = MemoryEfficientGradientTracker(max_global_norms=500)
        self.initial_batch_size = args.batch_size
        self.min_batch_size = max(1, args.batch_size // 8)
        self.memory_threshold_mb = getattr(args, 'memory_threshold_mb', 8000)  # 8GB default
        
        super().__init__(args)
    
    def _build_model(self):
        """Override to use memory-optimized models."""
        logger = logging.getLogger(__name__)
        
        if self.args.model == "SimplifiedPGAT":
            logger.info("Using SimplifiedPGAT model")
            model = SimplifiedPGAT(self.args, mode='probabilistic')
        elif self.args.model == "MemoryOptimizedPGAT":
            logger.info("Attempting to use MemoryOptimizedPGAT model")
            try:
                model = MemoryOptimizedPGAT(self.args, mode='probabilistic')
                logger.info("Successfully created MemoryOptimizedPGAT")
            except Exception as e:
                logger.warning(f"Failed to create MemoryOptimizedPGAT: {e}, falling back to SimplifiedPGAT")
                model = SimplifiedPGAT(self.args, mode='probabilistic')
        else:
            logger.info(f"Using default model for {self.args.model}")
            # Fallback to parent implementation
            model = super()._build_model()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        return model
        
    def _select_optimizer(self):
        """Override to add gradient tracking."""
        optimizer = super()._select_optimizer()
        original_step = optimizer.step
        
        def tracked_step(*args, **kwargs):
            # Track gradients efficiently
            self.gradient_tracker.log_step(self.model.named_parameters())
            
            # Memory cleanup every 50 steps
            if self.gradient_tracker.step_count % 50 == 0:
                self._cleanup_memory()
                
            return original_step(*args, **kwargs)
        
        optimizer.step = tracked_step
        return optimizer
    
    def _cleanup_memory(self):
        """Periodic memory cleanup."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Clear model cache if it exists
        if hasattr(self.model, 'clear_cache'):
            self.model.clear_cache()
    
    def _check_memory_usage(self) -> bool:
        """Check if memory usage is within limits."""
        if torch.cuda.is_available():
            allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            return allocated_mb < self.memory_threshold_mb
        return True
    
    def _reduce_batch_size(self, current_batch_size: int) -> int:
        """Reduce batch size when OOM occurs."""
        new_batch_size = max(self.min_batch_size, current_batch_size // 2)
        logging.warning(f"Reducing batch size from {current_batch_size} to {new_batch_size}")
        return new_batch_size
    
    def train(self, setting: str):
        """Memory-efficient training with automatic batch size adjustment."""
        try:
            super().train(setting)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logging.error(f"OOM Error: {e}")
                
                # Reduce batch size and retry
                new_batch_size = self._reduce_batch_size(self.args.batch_size)
                if new_batch_size >= self.min_batch_size:
                    logging.info(f"Retrying with batch size {new_batch_size}")
                    self.args.batch_size = new_batch_size
                    
                    # Recreate data loaders with new batch size
                    self._recreate_data_loaders()
                    
                    # Clear memory and retry
                    self._cleanup_memory()
                    super().train(setting)
                else:
                    raise RuntimeError("Cannot reduce batch size further") from e
            else:
                raise
    
    def _recreate_data_loaders(self):
        """Recreate data loaders with new batch size."""
        train_loader, vali_loader, test_loader, scaler_manager, dim_manager = setup_financial_forecasting_data(self.args)
        self.args.train_loader = train_loader
        self.args.vali_loader = vali_loader
        self.args.test_loader = test_loader


def setup_memory_optimized_config(args: SimpleNamespace) -> None:
    """Configure memory optimization settings."""
    # Enable memory optimizations
    args.use_gradient_checkpointing = True
    args.use_sparse_attention = True
    args.chunk_size = 32
    args.memory_threshold_mb = 8000  # 8GB threshold
    
    # Mixed precision training
    args.use_amp = True
    
    # Reduce model complexity if needed
    if not hasattr(args, 'd_model'):
        args.d_model = 256  # Smaller model for memory efficiency
    if not hasattr(args, 'n_heads'):
        args.n_heads = 4   # Fewer attention heads
    
    # Batch size optimization
    if not hasattr(args, 'batch_size'):
        args.batch_size = 16  # Conservative batch size
    
    # Use memory-optimized model
    args.model = "MemoryOptimizedPGAT"


def main():
    """Main training function with memory optimization."""
    parser = argparse.ArgumentParser(description="Memory-optimized PGAT training")
    parser.add_argument("--config", type=Path, default="configs/memory_optimized_pgat.yaml")
    parser.add_argument("--batch-size", type=int, default=16, help="Initial batch size")
    parser.add_argument("--memory-threshold", type=int, default=8000, help="Memory threshold in MB")
    parser.add_argument("--enable-amp", action="store_true", help="Enable automatic mixed precision")
    parser.add_argument("--chunk-size", type=int, default=32, help="Sequence chunk size for processing")
    
    cli_args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # Load config from YAML if it exists, otherwise use defaults
        if cli_args.config.exists():
            import yaml
            with open(cli_args.config, 'r') as f:
                config_dict = yaml.safe_load(f)
            args = SimpleNamespace(**config_dict)
        else:
            args = SimpleNamespace()
        
        # Override with CLI args and set required defaults
        args.config = str(cli_args.config)
        args.batch_size = cli_args.batch_size
        args.memory_threshold_mb = cli_args.memory_threshold
        args.use_amp = cli_args.enable_amp
        args.chunk_size = cli_args.chunk_size
        
        # Set all required attributes for data pipeline
        if not hasattr(args, 'task_name'):
            args.task_name = "long_term_forecast"
        if not hasattr(args, 'seq_len'):
            args.seq_len = 96
        if not hasattr(args, 'pred_len'):
            args.pred_len = 24
        if not hasattr(args, 'label_len'):
            args.label_len = 48
        if not hasattr(args, 'd_model'):
            args.d_model = 256
        if not hasattr(args, 'n_heads'):
            args.n_heads = 4
        if not hasattr(args, 'root_path'):
            args.root_path = "data"
        if not hasattr(args, 'data_path'):
            args.data_path = "synthetic_multi_wave.csv"
        if not hasattr(args, 'target'):
            args.target = "target_0,target_1,target_2"
        if not hasattr(args, 'features'):
            args.features = "M"
        if not hasattr(args, 'freq'):
            args.freq = "H"
        if not hasattr(args, 'train_epochs'):
            args.train_epochs = 10
        if not hasattr(args, 'learning_rate'):
            args.learning_rate = 1e-4
        if not hasattr(args, 'patience'):
            args.patience = 3
        if not hasattr(args, 'num_workers'):
            args.num_workers = 2
        if not hasattr(args, 'validation_length'):
            args.validation_length = 150
        if not hasattr(args, 'test_length'):
            args.test_length = 120
        
        # Critical missing attributes that caused the error
        if not hasattr(args, 'loss'):
            args.loss = "mse"
        if not hasattr(args, 'embed'):
            args.embed = "timeF"
        if not hasattr(args, 'dropout'):
            args.dropout = 0.1
        if not hasattr(args, 'd_ff'):
            args.d_ff = 512
        if not hasattr(args, 'activation'):
            args.activation = "gelu"
        if not hasattr(args, 'factor'):
            args.factor = 1
        if not hasattr(args, 'distil'):
            args.distil = True
        if not hasattr(args, 'output_attention'):
            args.output_attention = False
        # Override GPU settings based on actual availability
        cuda_available = torch.cuda.is_available()
        if not cuda_available:
            logger.warning("CUDA not available, forcing CPU mode")
            args.use_gpu = False
            args.gpu_type = "cpu"
        else:
            if not hasattr(args, 'use_gpu'):
                args.use_gpu = True
            if not hasattr(args, 'gpu_type'):
                args.gpu_type = "cuda"
        
        if not hasattr(args, 'gpu'):
            args.gpu = 0
        if not hasattr(args, 'use_multi_gpu'):
            args.use_multi_gpu = False
        if not hasattr(args, 'devices'):
            args.devices = "0"
        if not hasattr(args, 'is_training'):
            args.is_training = 1
        if not hasattr(args, 'model_id'):
            args.model_id = "memory_optimized_pgat"
        if not hasattr(args, 'checkpoints'):
            args.checkpoints = "./checkpoints/"
        if not hasattr(args, 'des'):
            args.des = "memory_optimized"
        if not hasattr(args, 'itr'):
            args.itr = 1
        if not hasattr(args, 'lradj'):
            args.lradj = "type1"
        
        # Apply memory optimizations
        setup_memory_optimized_config(args)
        
        # Force SimplifiedPGAT model for stability
        args.model = "SimplifiedPGAT"
        logger.info("Forced model to SimplifiedPGAT for memory efficiency")
        
        # Disable AMP for CPU mode
        if not cuda_available:
            args.use_amp = False
            logger.info("Disabled mixed precision training for CPU mode")
        
        # Setup data pipeline
        logger.info("Setting up data pipeline...")
        train_loader, vali_loader, test_loader, scaler_manager, dim_manager = setup_financial_forecasting_data(args)
        
        args.train_loader = train_loader
        args.vali_loader = vali_loader  
        args.test_loader = test_loader
        args.scaler_manager = scaler_manager
        args.dim_manager = dim_manager
        
        # Update dimensions
        args.enc_in = dim_manager.enc_in
        args.dec_in = dim_manager.dec_in
        args.c_out = dim_manager.c_out_model
        
        logger.info(f"Dimension manager settings: enc_in={args.enc_in}, dec_in={args.dec_in}, c_out={args.c_out}")
        logger.info(f"Target features: {dim_manager.target_features}")
        logger.info(f"c_out_evaluation: {dim_manager.c_out_evaluation}")
        
        # Create memory-optimized experiment
        experiment = MemoryOptimizedExperiment(args)
        
        # Training
        setting = f"memory_optimized_pgat_{args.batch_size}_{args.d_model}"
        logger.info(f"Starting training: {setting}")
        
        experiment.train(setting)
        logger.info("Training completed successfully")
        
        # Test
        experiment.test(setting, test=1)
        logger.info("Testing completed successfully")
        
        # Print memory statistics
        if hasattr(experiment.model, 'get_memory_stats'):
            stats = experiment.model.get_memory_stats()
            logger.info(f"Final memory stats: {stats}")
            
        tracker_memory = experiment.gradient_tracker.get_memory_usage_mb()
        logger.info(f"Gradient tracker memory usage: {tracker_memory:.2f} MB")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()