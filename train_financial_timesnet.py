#!/usr/bin/env python3
"""
Financial TimesNet Training Script

Train TimesNet model on prepared financial data with the following configuration:
- Targets: 4 (log_Open, log_High, log_Low, log_Close)
- Covariates: 114 (87 dynamic + 26 static + 1 time_delta)
- Sequence length: 500, Prediction length: 10
- Validation/Test length: 10 each
- Production length: 10 (future business days beyond data)
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.TimesNet import Model as TimesNet
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.logger import logger
from data_provider.data_loader import Dataset_Custom
from torch.utils.data import DataLoader


class FinancialTimesNetTrainer:
    """Trainer for TimesNet model on financial data"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Set random seeds for reproducibility
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        
        # Initialize model and data
        self._setup_data()
        self._setup_model()
        
    def _setup_data(self):
        """Setup data loaders for train/val/test/production"""
        logger.info("Setting up data loaders")
        
        # Load prepared financial data
        data_path = os.path.join(self.args.root_path, 'prepared_financial_data.csv')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Prepared data not found: {data_path}")
        
        df = pd.read_csv(data_path)
        logger.info(f"Loaded financial data: {df.shape}")
        
        # Verify data structure
        target_cols = ['log_Open', 'log_High', 'log_Low', 'log_Close']
        if not all(col in df.columns for col in target_cols):
            raise ValueError(f"Missing target columns. Expected: {target_cols}")
        
        # Calculate data splits
        total_len = len(df)
        train_end = total_len - self.args.val_len - self.args.test_len
        val_end = total_len - self.args.test_len
        
        logger.info(f"Data splits: Train[0:{train_end}], Val[{train_end}:{val_end}], Test[{val_end}:{total_len}]")
        logger.info(f"Production will use last {self.args.seq_len} points for {self.args.prod_len} future predictions")
        
        # Store data splits info
        self.data_info = {
            'total_len': total_len,
            'train_end': train_end,
            'val_end': val_end,
            'test_end': total_len,
            'target_cols': target_cols,
            'data_path': data_path
        }
          # Create data loaders
        self.train_loader = self._create_data_loader('train')
        self.val_loader = self._create_data_loader('val')
        self.test_loader = self._create_data_loader('test')
        
        logger.info(f"Train loader: {len(self.train_loader)} batches")
        logger.info(f"Val loader: {len(self.val_loader)} batches")
        logger.info(f"Test loader: {len(self.test_loader)} batches")
    
    def _create_data_loader(self, flag):
        """Create data loader for specific split"""
        # Set validation and test lengths for border calculation
        self.args.validation_length = self.args.val_len
        self.args.test_length = self.args.test_len
        
        # Create dataset
        dataset = Dataset_Custom(
            args=self.args,
            root_path=self.args.root_path,
            data_path=self.args.data_path,
            flag=flag,
            size=[self.args.seq_len, self.args.label_len, self.args.pred_len],
            features=self.args.features,
            target=self.args.target,
            scale=True,
            timeenc=1 if self.args.embed == 'timeF' else 0,
            freq=self.args.freq
        )
        
        # Create data loader
        shuffle = (flag == 'train')
        data_loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            num_workers=self.args.num_workers,
            drop_last=False
        )
        
        return data_loader
    
    def _setup_model(self):
        """Initialize TimesNet model"""
        logger.info("Setting up TimesNet model")
        
        # Create model
        self.model = TimesNet(self.args).to(self.device)
        
        # Setup optimizer and criterion
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.criterion = torch.nn.MSELoss()
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        # Model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    def train_epoch(self):
        """Train for one epoch with progress tracking"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        total_batches = len(self.train_loader)
        epoch_start_time = time.time()
        print(f"Training on {total_batches} batches...")
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.train_loader):
            # Move to device
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)
            
            # Prepare decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
              # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            # Calculate loss (only on target columns - first 4 features)
            target_outputs = outputs[:, -self.args.pred_len:, :4]  # First 4 features are targets
            target_y = batch_y[:, -self.args.pred_len:, :4]  # First 4 features are targets
            loss = self.criterion(target_outputs, target_y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
              # Show progress every 10 batches or at key milestones
            total_batches = len(self.train_loader)
            if i % 10 == 0 or i == total_batches - 1:
                progress_pct = (i + 1) / total_batches * 100
                avg_loss_so_far = total_loss / num_batches
                elapsed_time = time.time() - epoch_start_time
                estimated_total_time = elapsed_time / (i + 1) * total_batches
                remaining_time = estimated_total_time - elapsed_time
                print(f"  Batch {i+1:3d}/{total_batches} ({progress_pct:5.1f}%) - "
                      f"Loss: {loss.item():.6f} (Avg: {avg_loss_so_far:.6f}) - "
                      f"Remaining: {remaining_time:.1f}s")
        
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
                
                # Prepare decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # Forward pass
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                  # Calculate loss (only on target columns - first 4 features)
                target_outputs = outputs[:, -self.args.pred_len:, :4]  # First 4 features are targets
                target_y = batch_y[:, -self.args.pred_len:, :4]  # First 4 features are targets
                loss = self.criterion(target_outputs, target_y)
                
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
                
                # Prepare decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                  # Forward pass
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # Get predictions and true values (only target columns)
                pred = outputs[:, -self.args.pred_len:, :4].detach().cpu().numpy()  # First 4 features are targets
                true = batch_y[:, -self.args.pred_len:, :4].detach().cpu().numpy()  # First 4 features are targets
                
                preds.append(pred)
                trues.append(true)
        
        # Concatenate all predictions and true values
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        # Calculate metrics
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        
        logger.info(f"Test Results - MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}")
        logger.info(f"Test Results - MAPE: {mape:.6f}, MSPE: {mspe:.6f}")
        
        return {
            'mse': mse, 'mae': mae, 'rmse': rmse, 'mape': mape, 'mspe': mspe,
            'predictions': preds, 'true_values': trues
        }
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training")
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(self.args.train_epochs):
            print(f"\n=== Epoch {epoch+1}/{self.args.train_epochs} ===")
            logger.info(f"Starting Epoch {epoch+1}/{self.args.train_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            train_losses.append(train_loss)
            
            # Validate
            print("Running validation...")
            val_loss = self.validate_epoch()
            val_losses.append(val_loss)
            
            # Log progress
            print(f"✓ Epoch {epoch+1} Complete - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            logger.info(f"Epoch {epoch+1}/{self.args.train_epochs} - "
                       f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Adjust learning rate
            adjust_learning_rate(self.optimizer, epoch + 1, self.args)
            
            # Early stopping check
            self.early_stopping(val_loss, self.model, self.args.checkpoints)
            if self.early_stopping.early_stop:
                logger.info("Early stopping triggered")
                break
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('best_model.pth')
        
        # Load best model for testing
        self.load_model('best_model.pth')
        
        # Test model
        test_results = self.test()
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'test_results': test_results
        }
    
    def save_model(self, filename):
        """Save model checkpoint"""
        if not os.path.exists(self.args.checkpoints):
            os.makedirs(self.args.checkpoints)
        
        path = os.path.join(self.args.checkpoints, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'args': self.args
        }, path)
        logger.debug(f"Model saved to {path}")
    
    def load_model(self, filename):
        """Load model checkpoint"""
        path = os.path.join(self.args.checkpoints, filename)
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"Model loaded from {path}")
        else:
            logger.warning(f"Checkpoint not found: {path}")
    
    def predict_production(self):
        """Generate production predictions for future business days"""
        logger.info(f"Generating production predictions for {self.args.prod_len} future business days")
        
        # Load the latest data for prediction
        data_path = os.path.join(self.args.root_path, 'prepared_financial_data.csv')
        df = pd.read_csv(data_path)
        
        # Get the last sequence_length points
        last_data = df.tail(self.args.seq_len).copy()
        
        # Prepare the data for prediction (this would need to be implemented
        # similar to the data loader but for a single sequence)
        logger.info("Production prediction feature to be implemented")
        logger.info(f"Would predict {self.args.prod_len} steps into the future")
        logger.info(f"Using last {self.args.seq_len} data points as input")
        
        return None


def create_args():
    """Create argument parser with financial data specific settings"""
    parser = argparse.ArgumentParser(description='TimesNet Financial Forecasting')
    
    # Data settings
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='prepared_financial_data.csv', help='data file')
    parser.add_argument('--features', type=str, default='M', help='forecasting task [M, S, MS]')
    parser.add_argument('--target', type=str, default='log_Close', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='b', help='freq for time features encoding [b=business day]')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')    # Forecasting task (BALANCED MEDIUM WEIGHT VERSION)
    parser.add_argument('--seq_len', type=int, default=100, help='input sequence length (balanced)')
    parser.add_argument('--label_len', type=int, default=10, help='start token length (balanced)')
    parser.add_argument('--pred_len', type=int, default=10, help='prediction sequence length (balanced)')
    parser.add_argument('--val_len', type=int, default=10, help='validation length (balanced)')
    parser.add_argument('--test_len', type=int, default=10, help='test length (balanced)')
    parser.add_argument('--prod_len', type=int, default=10, help='production prediction length (balanced)')
    
    # Model define (BALANCED MEDIUM WEIGHT VERSION)
    parser.add_argument('--top_k', type=int, default=5, help='TimesNet kernel size (balanced)')
    parser.add_argument('--num_kernels', type=int, default=5, help='for Inception (balanced)')
    parser.add_argument('--enc_in', type=int, default=118, help='encoder input size (114 covariates + 4 targets)')
    parser.add_argument('--dec_in', type=int, default=118, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=118, help='output size (match enc_in to avoid dimension mismatch)')
    parser.add_argument('--d_model', type=int, default=64, help='dimension of model (balanced)')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads (balanced)')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers (balanced)')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn (balanced)')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding [timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    
    # Optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data (medium weight)')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training')
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    
    # Task specific
    parser.add_argument('--task_name', type=str, default='long_term_forecast', help='task name')
    
    args = parser.parse_args()
    return args


def main():
    """Main training function"""
    logger.info("Starting Financial TimesNet Training")
    
    # Create arguments
    args = create_args()
    
    # Create experiment name
    args.model = 'TimesNet'
    setting = f'{args.model}_{args.data}_{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_fc{args.factor}_eb{args.embed}_dt{args.des}'
    
    logger.info(f"Experiment: {setting}")
    logger.info(f"Data configuration:")
    logger.info(f"  - Sequence length: {args.seq_len}")
    logger.info(f"  - Prediction length: {args.pred_len}")
    logger.info(f"  - Input features: {args.enc_in} (4 targets + 114 covariates)")
    logger.info(f"  - Output features: {args.c_out} (4 targets)")
    logger.info(f"  - Model dimension: {args.d_model}")
    logger.info(f"  - Feed-forward dimension: {args.d_ff}")
    logger.info(f"  - Number of layers: {args.e_layers}")
    logger.info(f"  - Batch size: {args.batch_size}")
    
    # Update checkpoints path with experiment name
    args.checkpoints = os.path.join('./checkpoints/', setting)
    
    try:
        # Create trainer and run training
        trainer = FinancialTimesNetTrainer(args)
        results = trainer.train()
        
        logger.info("Training completed successfully!")
        logger.info(f"Best validation loss: {results['best_val_loss']:.6f}")
        logger.info(f"Final test MSE: {results['test_results']['mse']:.6f}")
        
        # Generate production predictions
        trainer.predict_production()
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == '__main__':
    main()
