"""
Training Script for Enhanced Autoformer with Advanced Features

This script demonstrates how to train the Enhanced Autoformer with:
- Adaptive loss functions
- Curriculum learning
- Enhanced monitoring and logging
- Memory optimization
"""

import os
import sys
import argparse
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

# Suppress warnings
warnings.filterwarnings('ignore')

# Imports
from models.EnhancedAutoformer import Model as EnhancedAutoformer
from utils.enhanced_losses import AdaptiveAutoformerLoss, CurriculumLossScheduler, create_enhanced_loss
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from utils.logger import logger, set_log_level
import logging


class EnhancedAutoformerTrainer:
    """
    Enhanced training framework for Autoformer with advanced features.
    """
    
    def __init__(self, args):
        logger.info("Initializing EnhancedAutoformerTrainer")
        self.args = args
        self.device = self._get_device()
        
        # Setup model
        self.model = self._build_model()
        
        # Setup data
        self.train_data, self.train_loader = self._get_data('train')
        self.val_data, self.val_loader = self._get_data('val')
        self.test_data, self.test_loader = self._get_data('test')
        
        # Setup training components
        self.optimizer = self._select_optimizer()
        self.criterion = self._select_criterion()
        self.early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        
        # Setup curriculum learning if enabled
        if args.use_curriculum:
            self.curriculum = CurriculumLossScheduler(
                start_seq_len=args.curriculum_start_len,
                target_seq_len=args.seq_len,
                curriculum_epochs=args.curriculum_epochs,
                loss_fn=self.criterion
            )
        else:
            self.curriculum = None
            
        # Training metrics tracking
        self.training_metrics = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'epoch_times': [],
            'memory_usage': []
        }
        
    def _get_device(self):
        """Setup device for training."""
        if self.args.use_gpu and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.args.gpu}')
            logger.info(f"Using GPU: {device}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU")
        return device
        
    def _build_model(self):
        """Build Enhanced Autoformer model."""
        logger.info("Building Enhanced Autoformer model")
        
        model = EnhancedAutoformer(self.args).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            
        model = model.to(self.device)
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        return model
        
    def _get_data(self, flag):
        """Get data loader for specified split."""
        data_set, data_loader = data_provider(self.args, flag)
        logger.info(f"{flag} data: {len(data_set)} samples, {len(data_loader)} batches")
        return data_set, data_loader
        
    def _select_optimizer(self):
        """Select optimizer with enhanced scheduling."""
        if self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
            
        logger.info(f"Using optimizer: {self.args.optimizer}")
        return optimizer
        
    def _select_criterion(self):
        """Select enhanced loss function."""
        if self.args.loss_type == 'adaptive':
            criterion = AdaptiveAutoformerLoss(
                base_loss=self.args.base_loss,
                adaptive_weights=True,
                moving_avg=self.args.moving_avg
            )
            logger.info("Using adaptive autoformer loss")
        else:
            criterion = nn.MSELoss()
            logger.info("Using standard MSE loss")
            
        return criterion
        
    def train_epoch(self, epoch):
        """Train for one epoch with enhanced features."""
        self.model.train()
        
        train_loss = []
        epoch_start_time = time.time()
        
        # Track memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_start = torch.cuda.memory_allocated()
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.train_loader):
            
            # Apply curriculum learning if enabled
            if self.curriculum:
                batch_x, batch_y, batch_x_mark, batch_y_mark = self.curriculum.apply_curriculum(
                    batch_x, batch_y, batch_x_mark, batch_y_mark, epoch
                )
            
            # Move to device
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)
            
            # Prepare decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
            
            # Compute loss
            if hasattr(self.criterion, 'forward'):
                if self.curriculum:
                    loss = self.curriculum.compute_loss(outputs, batch_y, epoch)
                else:
                    loss = self.criterion(outputs, batch_y)
            else:
                loss = self.criterion(outputs, batch_y)
            
            train_loss.append(loss.item())
            
            # Backward pass with gradient clipping
            loss.backward()
            if self.args.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            
            # Logging
            if i % self.args.log_interval == 0:
                logger.info(f'Epoch {epoch} [{i}/{len(self.train_loader)}] Loss: {loss.item():.6f}')
        
        # Calculate epoch metrics
        epoch_time = time.time() - epoch_start_time
        avg_train_loss = np.average(train_loss)
        
        # Memory usage
        if torch.cuda.is_available():
            memory_end = torch.cuda.memory_allocated()
            memory_used = (memory_end - memory_start) / (1024**2)  # MB
            self.training_metrics['memory_usage'].append(memory_used)
        
        # Store metrics
        self.training_metrics['train_losses'].append(avg_train_loss)
        self.training_metrics['epoch_times'].append(epoch_time)
        self.training_metrics['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
        
        logger.info(f'Epoch {epoch} completed in {epoch_time:.2f}s, avg loss: {avg_train_loss:.6f}')
        
        return avg_train_loss
        
    def validate_epoch(self, epoch):
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.val_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # Prepare decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # Forward pass
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                # Compute loss
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                
                # Scale the ground truth to match model outputs (model outputs are scaled)
                if hasattr(self.val_data, 'target_scaler') and self.val_data.target_scaler is not None:
                    # Ground truth needs to be scaled to match model predictions
                    true_np = true.numpy()
                    true_scaled_np = self.val_data.target_scaler.transform(
                        true_np.reshape(-1, true_np.shape[-1])
                    ).reshape(true_np.shape)
                    true = torch.from_numpy(true_scaled_np).float()
                
                loss = nn.MSELoss()(pred, true)
                total_loss.append(loss.item())
        
        avg_val_loss = np.average(total_loss)
        self.training_metrics['val_losses'].append(avg_val_loss)
        
        logger.info(f'Validation loss: {avg_val_loss:.6f}')
        return avg_val_loss
        
    def test(self):
        """Test the model and compute metrics."""
        logger.info("Testing model")
        self.model.eval()
        
        preds = []
        trues = []
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # Prepare decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # Forward pass
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                
                # Scale the ground truth to match model outputs (model outputs are scaled)
                if hasattr(self.test_data, 'target_scaler') and self.test_data.target_scaler is not None:
                    # Ground truth needs to be scaled to match model predictions for consistent metrics
                    true = self.test_data.target_scaler.transform(
                        true.reshape(-1, true.shape[-1])
                    ).reshape(true.shape)
                
                preds.append(pred)
                trues.append(true)
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        # Calculate metrics
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        
        logger.info(f'Test Results - MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}')
        logger.info(f'Test Results - MAPE: {mape:.6f}, MSPE: {mspe:.6f}')
        
        return {
            'mse': mse, 'mae': mae, 'rmse': rmse, 'mape': mape, 'mspe': mspe,
            'predictions': preds, 'true_values': trues
        }
        
    def train(self):
        """Main training loop."""
        logger.info("Starting Enhanced Autoformer training")
        
        # NOTE: Scaling consistency fix
        # - Model predictions are in scaled space (trained on scaled data)
        # - Validation/test ground truth is unscaled (to avoid data leakage)
        # - We scale ground truth during loss/metric computation to match predictions
        
        best_val_loss = float('inf')
        training_start_time = time.time()
        
        for epoch in range(self.args.train_epochs):
            epoch_start = time.time()
            
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_loss = self.validate_epoch(epoch)
            
            # Learning rate adjustment
            adjust_learning_rate(self.optimizer, epoch + 1, self.args)
            
            # Early stopping
            self.early_stopping(val_loss, self.model, self.args.checkpoints)
            if self.early_stopping.early_stop:
                logger.info("Early stopping triggered")
                break
                
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('best_model.pth')
                
            epoch_time = time.time() - epoch_start
            logger.info(f'Epoch {epoch+1}/{self.args.train_epochs} completed in {epoch_time:.2f}s')
            
        # Final testing
        total_training_time = time.time() - training_start_time
        logger.info(f'Training completed in {total_training_time:.2f}s')
        
        # Load best model and test
        self.load_model('best_model.pth')
        test_results = self.test()
        
        # Save training metrics
        self.save_training_metrics()
        
        return {
            'best_val_loss': best_val_loss,
            'test_results': test_results,
            'training_metrics': self.training_metrics,
            'total_training_time': total_training_time
        }
        
    def save_model(self, filename):
        """Save model checkpoint."""
        if not os.path.exists(self.args.checkpoints):
            os.makedirs(self.args.checkpoints)
        
        path = os.path.join(self.args.checkpoints, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'args': self.args,
            'training_metrics': self.training_metrics
        }, path)
        logger.info(f"Model saved to {path}")
        
    def load_model(self, filename):
        """Load model checkpoint."""
        path = os.path.join(self.args.checkpoints, filename)
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"Model loaded from {path}")
        else:
            logger.warning(f"Checkpoint not found: {path}")
            
    def save_training_metrics(self):
        """Save training metrics to file."""
        metrics_path = os.path.join(self.args.checkpoints, 'training_metrics.json')
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {}
        for key, value in self.training_metrics.items():
            if isinstance(value, list):
                serializable_metrics[key] = value
            else:
                serializable_metrics[key] = value.tolist() if hasattr(value, 'tolist') else value
        
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        logger.info(f"Training metrics saved to {metrics_path}")


def create_args():
    """Create argument parser for enhanced training."""
    parser = argparse.ArgumentParser(description='Enhanced Autoformer Training')
    
    # Basic model parameters
    parser.add_argument('--model', type=str, default='EnhancedAutoformer')
    parser.add_argument('--data', type=str, default='ETTh1')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    
    # Sequence parameters
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=24)
    
    # Model architecture
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--dec_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    
    # Training parameters
    parser.add_argument('--train_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--lradj', type=str, default='type1')
    
    # Enhanced features
    parser.add_argument('--loss_type', type=str, default='adaptive', choices=['mse', 'adaptive'])
    parser.add_argument('--base_loss', type=str, default='mse')
    parser.add_argument('--use_curriculum', action='store_true')
    parser.add_argument('--curriculum_start_len', type=int, default=24)
    parser.add_argument('--curriculum_epochs', type=int, default=50)
    parser.add_argument('--use_grad_clip', action='store_true')
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    
    # Hardware settings
    parser.add_argument('--use_gpu', action='store_true', default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true')
    parser.add_argument('--device_ids', type=str, default='0,1,2,3')
    
    # Logging and saving
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--log_level', type=str, default='INFO')
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = create_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level.upper())
    set_log_level(log_level)
    
    # Create checkpoint directory
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create trainer and start training
    trainer = EnhancedAutoformerTrainer(args)
    
    logger.info("="*60)
    logger.info("ENHANCED AUTOFORMER TRAINING")
    logger.info("="*60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.data}")
    logger.info(f"Sequence length: {args.seq_len}")
    logger.info(f"Prediction length: {args.pred_len}")
    logger.info(f"Loss type: {args.loss_type}")
    logger.info(f"Curriculum learning: {args.use_curriculum}")
    logger.info("="*60)
    
    # Start training
    results = trainer.train()
    
    # Print final results
    logger.info("="*60)
    logger.info("TRAINING COMPLETED")
    logger.info("="*60)
    logger.info(f"Best validation loss: {results['best_val_loss']:.6f}")
    logger.info(f"Test MSE: {results['test_results']['mse']:.6f}")
    logger.info(f"Test MAE: {results['test_results']['mae']:.6f}")
    logger.info(f"Total training time: {results['total_training_time']:.2f}s")
    logger.info("="*60)


if __name__ == '__main__':
    main()
