#!/usr/bin/env python3

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from data_provider.data_loader import Dataset_Custom
from models import Autoformer
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('TSLib')

class FinancialAutoformerTrainer:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        
        # Initialize data loaders
        self.train_loader = self._get_data_loader('train')
        self.val_loader = self._get_data_loader('val')
        self.test_loader = self._get_data_loader('test')
        
        # Initialize model
        self.model = self._build_model().to(self.device)
        
        # Initialize optimizer and criterion
        self.optimizer = self._select_optimizer()
        self.criterion = self._select_criterion()
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        
        logger.info(f"Autoformer trainer initialized on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def _acquire_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.args.gpu}')
            logger.info(f'Using GPU: cuda:{self.args.gpu}')
        else:
            device = torch.device('cpu')
            logger.info('Using CPU')
        return device

    def _get_data_loader(self, flag):
        """Get data loader for train/val/test"""
        # Set validation and test lengths for border calculation (similar to TimesNet)
        self.args.validation_length = self.args.val_len
        self.args.test_length = self.args.test_len
        
        dataset = Dataset_Custom(
            args=self.args,
            root_path=self.args.root_path,
            data_path=self.args.data_path,
            flag=flag,
            size=[self.args.seq_len, self.args.label_len, self.args.pred_len],
            features=self.args.features,
            target=self.args.target,
            scale=True,
            timeenc=1,
            freq=self.args.freq
        )
        
        batch_size = self.args.batch_size if flag == 'train' else 1
        shuffle = True if flag == 'train' else False
        
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.args.num_workers,
            drop_last=True
        )
        
        logger.info(f'{flag} dataset size: {len(dataset)}')
        return data_loader

    def _build_model(self):
        """Build Autoformer model"""
        model = Autoformer.Model(self.args)
        
        if self.args.use_multi_gpu and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            
        return model

    def _select_optimizer(self):
        """Select optimizer"""
        model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        """Select loss criterion"""
        criterion = nn.MSELoss()
        return criterion

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        epoch_start_time = time.time()
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.train_loader):
            batch_start_time = time.time()
            self.optimizer.zero_grad()
            
            # Move to device
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)
            
            # Autoformer decoder input - PROPER future covariate handling
            # Historical period: real targets + real covariates
            dec_inp_historical = batch_y[:, :self.args.label_len, :].float().to(self.device)
            
            # Future period: zero targets + REAL future covariates (like TimesNet!)
            future_targets_zero = torch.zeros_like(batch_y[:, -self.args.pred_len:, :4]).float().to(self.device)
            future_covariates = batch_y[:, -self.args.pred_len:, 4:].float().to(self.device)
            dec_inp_future = torch.cat([future_targets_zero, future_covariates], dim=-1)
            
            # Combine historical + future for decoder input
            dec_inp = torch.cat([dec_inp_historical, dec_inp_future], dim=1).float().to(self.device)
            
            # Forward pass - Autoformer handles decomposition internally
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            # Calculate loss only on target features (first 4 columns: OHLC)
            # Since c_out=118, extract first 4 columns for OHLC targets from LAST pred_len steps
            target_outputs = outputs[:, -self.args.pred_len:, :4]  # OHLC targets (LAST pred_len steps)
            target_y = batch_y[:, -self.args.pred_len:, :4]  # OHLC ground truth
            loss = self.criterion(target_outputs, target_y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Progress tracking
            batch_time = time.time() - batch_start_time
            total_batches = len(self.train_loader)
            if i % 10 == 0 or i == total_batches - 1:
                progress_pct = (i + 1) / total_batches * 100
                avg_loss_so_far = total_loss / num_batches
                elapsed_time = time.time() - epoch_start_time
                estimated_total_time = elapsed_time / (i + 1) * total_batches
                remaining_time = estimated_total_time - elapsed_time
                print(f"  Batch {i+1:3d}/{total_batches} ({progress_pct:5.1f}%) - "
                      f"Loss: {loss.item():.6f} (Avg: {avg_loss_so_far:.6f}) - "
                      f"Batch Time: {batch_time:.2f}s - Remaining: {remaining_time:.1f}s")
        
        avg_loss = total_loss / num_batches
        return avg_loss

    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in self.val_loader:
                # Move to device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # Autoformer decoder input - with future covariates (like training & TimesNet)
                # Historical period: real targets + real historical covariates  
                dec_inp_historical = batch_y[:, :self.args.label_len, :].float().to(self.device)
                
                # Future period: zero targets + REAL future covariates (like TimesNet!)
                future_targets_zero = torch.zeros_like(batch_y[:, -self.args.pred_len:, :4]).float().to(self.device)
                future_covariates = batch_y[:, -self.args.pred_len:, 4:].float().to(self.device)
                dec_inp_future = torch.cat([future_targets_zero, future_covariates], dim=-1)
                
                # Combine historical + future for decoder input
                dec_inp = torch.cat([dec_inp_historical, dec_inp_future], dim=1).float().to(self.device)
                
                # Forward pass
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # Get predictions and ground truth (extract first 4 columns for OHLC)
                target_outputs = outputs[:, -self.args.pred_len:, :4]  # SCALED OHLC predictions (LAST pred_len steps)
                target_y_unscaled = batch_y[:, -self.args.pred_len:, :4]  # UNSCALED ground truth
                
                # Scale the ground truth to match model outputs
                target_y_unscaled_np = target_y_unscaled.detach().cpu().numpy()
                target_y_scaled_np = self.train_loader.dataset.target_scaler.transform(
                    target_y_unscaled_np.reshape(-1, 4)
                ).reshape(target_y_unscaled_np.shape)
                target_y_scaled = torch.from_numpy(target_y_scaled_np).float().to(self.device)
                
                # Calculate loss on scaled data
                loss = self.criterion(target_outputs, target_y_scaled)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss

    def test(self):
        """Test the model and calculate metrics"""
        logger.info("Testing model")
        self.model.eval()
        
        preds = []
        trues = []
        
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in self.test_loader:
                # Move to device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # Autoformer decoder input - with future covariates (like training & TimesNet)
                # Historical period: real targets + real historical covariates  
                dec_inp_historical = batch_y[:, :self.args.label_len, :].float().to(self.device)
                
                # Future period: zero targets + REAL future covariates (like TimesNet!)
                future_targets_zero = torch.zeros_like(batch_y[:, -self.args.pred_len:, :4]).float().to(self.device)
                future_covariates = batch_y[:, -self.args.pred_len:, 4:].float().to(self.device)
                dec_inp_future = torch.cat([future_targets_zero, future_covariates], dim=-1)
                
                # Combine historical + future for decoder input
                dec_inp = torch.cat([dec_inp_historical, dec_inp_future], dim=1).float().to(self.device)
                
                # Forward pass
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # Get predictions and ground truth (extract first 4 columns for OHLC)
                pred_scaled = outputs[:, -self.args.pred_len:, :4].detach().cpu().numpy()  # OHLC predictions (LAST pred_len steps)
                true_unscaled = batch_y[:, -self.args.pred_len:, :4].detach().cpu().numpy()  # UNSCALED ground truth
                
                # Scale the ground truth for consistent loss calculation
                true_scaled = self.test_loader.dataset.target_scaler.transform(
                    true_unscaled.reshape(-1, 4)
                ).reshape(true_unscaled.shape)
                
                preds.append(pred_scaled)
                trues.append(true_scaled)
        
        # Concatenate all predictions and true values
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        # Calculate metrics on scaled data
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        
        logger.info(f"Test Results (Scaled Space) - MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}")
        logger.info(f"Test Results (Scaled Space) - MAPE: {mape:.6f}, MSPE: {mspe:.6f}")
        
        return {
            'mse': mse, 'mae': mae, 'rmse': rmse, 'mape': mape, 'mspe': mspe,
            'predictions': preds, 'true_values': trues
        }

    def train(self):
        """Main training loop"""
        logger.info("Starting Autoformer training")
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        # Create checkpoint directory
        checkpoint_dir = os.path.join('checkpoints', self.args.model_id)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(self.args.train_epochs):
            print(f"\n=== Epoch {epoch+1}/{self.args.train_epochs} ===")
            logger.info(f"Starting Epoch {epoch+1}/{self.args.train_epochs}")
            start_time = time.time()
            
            # Training
            print("Running training...")
            train_loss = self.train_epoch()
            train_losses.append(train_loss)
            
            # Validation
            print("Running validation...")
            val_start_time = time.time()
            val_loss = self.validate_epoch()
            val_losses.append(val_loss)
            val_time = time.time() - val_start_time
            
            epoch_time = time.time() - start_time
            
            # Progress summary
            print(f"✓ Epoch {epoch+1} Complete - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            print(f"  Total Time: {epoch_time:.2f}s (Val: {val_time:.2f}s)")
            logger.info(f"Epoch {epoch+1}/{self.args.train_epochs} - "
                       f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                       f"Time: {epoch_time:.2f}s")
            
            # Early stopping check
            self.early_stopping(val_loss, self.model, checkpoint_dir)
            if self.early_stopping.early_stop:
                print(f"⚠️ Early stopping triggered at epoch {epoch+1}")
                logger.info("Early stopping triggered")
                break
            
            # Save best model indicator
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"🎯 New best validation loss: {val_loss:.6f}")
            
            # Adjust learning rate
            adjust_learning_rate(self.optimizer, epoch + 1, self.args)
        
        # Load best model
        best_model_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            logger.info("Loaded best model for testing")
        
        return self.model

    def predict_production(self):
        """Generate production predictions for future business days"""
        logger.info(f"Generating production predictions for {self.args.prod_len} future business days")
        
        logger.info("Production prediction with Autoformer:")
        logger.info("  - Can utilize future covariates properly!")
        logger.info("  - Decoder input: Zero targets + Real future covariates")
        logger.info("  - Output: Scaled predictions (use inverse_transform_targets() for business use)")
        
        # TODO: Implement actual prediction logic
        # Autoformer will be able to use your 114 covariate features effectively
        
        return None

def get_args():
    """Get command line arguments"""
    parser = argparse.ArgumentParser(description='Autoformer Financial Forecasting')
    
    # Basic config
    parser.add_argument('--model_id', type=str, default='autoformer_financial', help='model id')
    parser.add_argument('--model', type=str, default='Autoformer', help='model name')
    parser.add_argument('--task_name', type=str, default='long_term_forecast', help='task name')
    
    # Data config
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='data', help='root path of data file')
    parser.add_argument('--data_path', type=str, default='prepared_financial_data.csv', help='data file')
    parser.add_argument('--features', type=str, default='M', help='forecasting task [M, S, MS]')
    parser.add_argument('--target', type=str, default='log_Close', help='target feature')
    parser.add_argument('--freq', type=str, default='b', help='freq for time features encoding')
    
    # Model config - EXTENDED SEQUENCES
    parser.add_argument('--seq_len', type=int, default=200, help='input sequence length (HEAVY: 2x longer)')
    parser.add_argument('--label_len', type=int, default=50, help='start token length (HEAVY: extended)')
    parser.add_argument('--pred_len', type=int, default=50, help='prediction sequence length (HEAVY: 2x longer)')
    parser.add_argument('--val_len', type=int, default=50, help='validation length')
    parser.add_argument('--test_len', type=int, default=50, help='test length')
    parser.add_argument('--prod_len', type=int, default=10, help='production prediction length')
    
    # Autoformer specific config - VERY HEAVY CONFIGURATION 🔥
    parser.add_argument('--enc_in', type=int, default=118, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=118, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=118, help='output size (set to 118 for multivariate mode)')
    parser.add_argument('--d_model', type=int, default=768, help='dimension of model (HEAVY: 1.5x bigger)')
    parser.add_argument('--n_heads', type=int, default=12, help='num of heads (HEAVY: 1.5x more)')
    parser.add_argument('--e_layers', type=int, default=4, help='num of encoder layers (HEAVY: 2x more)')
    parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers (HEAVY: 2x more)')
    parser.add_argument('--d_ff', type=int, default=3072, help='dimension of fcn (HEAVY: 1.5x bigger)')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention')
    
    # Training config
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--patience', type=int, default=7, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='type1', help='learning rate adjustment')
    
    # GPU config
    parser.add_argument('--use_gpu', action='store_true', default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    
    args = parser.parse_args()
    
    # Process device ids for multi-gpu
    args.device_ids = [int(id_) for id_ in args.devices.split(',')]
    
    return args

def main():
    """Main training function"""
    args = get_args()
    
    logger.info("="*50)
    logger.info("AUTOFORMER FINANCIAL FORECASTING")
    logger.info("="*50)
    logger.info(f"Model: {args.model}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Sequence Length: {args.seq_len}")
    logger.info(f"Prediction Length: {args.pred_len}")
    logger.info(f"Features: {args.features} (enc_in: {args.enc_in})")
    logger.info("="*50)
    
    # Initialize trainer
    trainer = FinancialAutoformerTrainer(args)
    
    # Train model
    model = trainer.train()
    
    # Test model
    test_results = trainer.test()
    
    logger.info("="*50)
    logger.info("TRAINING COMPLETED")
    logger.info(f"Final Test MSE: {test_results['mse']:.6f}")
    logger.info(f"Final Test MAE: {test_results['mae']:.6f}")
    logger.info("="*50)

if __name__ == "__main__":
    main()