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
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
        
        for batch_x, batch_y, batch_x_mark, batch_y_mark in self.train_loader:
            self.optimizer.zero_grad()
            
            # Move to device
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)
            
            # Autoformer decoder input - uses standard approach, decomposition is internal
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            
            # Forward pass - Autoformer handles decomposition internally
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            # Calculate loss only on target features (first 4 columns: OHLC)
            # Since c_out=118, extract first 4 columns for OHLC targets
            target_outputs = outputs[:, :, :4]  # First 4 features are OHLC targets
            target_y = batch_y[:, -self.args.pred_len:, :4]  # OHLC ground truth
            loss = self.criterion(target_outputs, target_y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
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
                
                # Autoformer decoder input - standard approach
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # Forward pass
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # Get predictions and ground truth (extract first 4 columns for OHLC)
                target_outputs = outputs[:, :, :4]  # SCALED OHLC predictions
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
                
                # Autoformer decoder input - standard approach
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # Forward pass
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # Get predictions and ground truth (extract first 4 columns for OHLC)
                pred_scaled = outputs[:, :, :4].detach().cpu().numpy()  # OHLC predictions (scaled)
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
            start_time = time.time()
            
            # Training
            train_loss = self.train_epoch()
            train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate_epoch()
            val_losses.append(val_loss)
            
            epoch_time = time.time() - start_time
            
            logger.info(f"Epoch {epoch+1}/{self.args.train_epochs} - "
                       f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                       f"Time: {epoch_time:.2f}s")
            
            # Early stopping check
            self.early_stopping(val_loss, self.model, checkpoint_dir)
            if self.early_stopping.early_stop:
                logger.info("Early stopping triggered")
                break
            
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
    
    # Model config
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
    parser.add_argument('--val_len', type=int, default=24, help='validation length')
    parser.add_argument('--test_len', type=int, default=24, help='test length')
    parser.add_argument('--prod_len', type=int, default=5, help='production prediction length')
    
    # Autoformer specific config
    parser.add_argument('--enc_in', type=int, default=118, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=118, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=118, help='output size (set to 118 for multivariate mode)')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
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