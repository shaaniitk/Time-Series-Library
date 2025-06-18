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
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import yaml, fallback to json if not available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("⚠️ PyYAML not installed. Using JSON config format instead.")
    print("💡 Install with: pip install PyYAML")

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
        
    # Update the train_epoch method around lines 175-182

    # Revert train_epoch method to original decoder input

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        epoch_start_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
                        # Move to device
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)
            
            # Prepare decoder input - REVERTED to original TimesNet approach
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            
            # Forward pass
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            # Calculate loss only on target features (first 4 columns)
            target_outputs = outputs[:, -self.args.pred_len:, :self.args.c_out]
            target_y = batch_y[:, -self.args.pred_len:, :self.args.c_out]
            loss = self.criterion(target_outputs, target_y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1

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

    # Update the validate_epoch method around lines 220-240

    # Revert validate_epoch method to original decoder input

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
                
                # Prepare decoder input - REVERTED to original TimesNet approach
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # Forward pass
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # Get predictions (scaled) and ground truth (unscaled)
                model_outputs_scaled = outputs[:, -self.args.pred_len:, :self.args.c_out]      # SCALED predictions
                
                # batch_y from loader for val/test has UNCALED targets (first N) and SCALED covariates (rest)
                batch_y_segment_for_loss = batch_y[:, -self.args.pred_len:, :self.args.c_out]

                # Determine how many actual target features the target_scaler was fitted on
                # This is typically 4 for M/MS modes, and 1 for S mode.
                num_features_in_target_scaler = self.train_loader.dataset.target_scaler.n_features_in_

                # Number of target features we need to extract and scale from batch_y_segment_for_loss
                # This is min(features model outputs, features scaler was fit on)
                num_targets_to_process = min(self.args.c_out, num_features_in_target_scaler)

                scaled_gt_parts = []
                if num_targets_to_process > 0:
                    # Extract unscaled targets from batch_y
                    gt_targets_unscaled_np = batch_y_segment_for_loss[:, :, :num_targets_to_process].detach().cpu().numpy()
                    
                    # Scale them using the target_scaler (input must match n_features_in_)
                    # If num_targets_to_process < num_features_in_target_scaler, this implies we are in 'M' mode
                    # and c_out is small, or 'S' mode where num_targets_to_process is 1.
                    # The scaler expects input with `num_features_in_target_scaler` columns.
                    # We must provide that many, then slice if num_targets_to_process is smaller.
                    # However, simpler: Dataset_Custom ensures target_scaler is fit on 1 (S) or 4 (M/MS) features.
                    # So, gt_targets_unscaled_np should have `num_targets_to_process` columns, and this should match
                    # what the scaler expects for those specific targets.
                    gt_targets_scaled_np = self.train_loader.dataset.target_scaler.transform(
                        gt_targets_unscaled_np.reshape(-1, num_targets_to_process) 
                    ).reshape(gt_targets_unscaled_np.shape)
                    scaled_gt_parts.append(torch.from_numpy(gt_targets_scaled_np).float().to(self.device))

                if self.args.c_out > num_targets_to_process: # Covariates part (already scaled)
                    scaled_gt_parts.append(batch_y_segment_for_loss[:, :, num_targets_to_process:])
                
                ground_truth_final_scaled = torch.cat(scaled_gt_parts, dim=-1) if len(scaled_gt_parts) > 0 else batch_y_segment_for_loss
                loss = self.criterion(model_outputs_scaled, ground_truth_final_scaled)
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
                
                # Prepare decoder input - REVERTED to original TimesNet approach
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # Forward pass
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # Get predictions (scaled) and ground truth (unscaled)
                pred_scaled = outputs[:, -self.args.pred_len:, :self.args.c_out].detach().cpu().numpy()
                batch_y_segment_for_loss_np = batch_y[:, -self.args.pred_len:, :self.args.c_out].detach().cpu().numpy()

                num_features_in_target_scaler = self.test_loader.dataset.target_scaler.n_features_in_
                num_targets_to_process = min(self.args.c_out, num_features_in_target_scaler)

                scaled_gt_parts_np = []
                if num_targets_to_process > 0:
                    gt_targets_unscaled_np_slice = batch_y_segment_for_loss_np[:, :, :num_targets_to_process]
                    gt_targets_scaled_np_slice = self.test_loader.dataset.target_scaler.transform(
                        gt_targets_unscaled_np_slice.reshape(-1, num_targets_to_process)
                    ).reshape(gt_targets_unscaled_np_slice.shape)
                    scaled_gt_parts_np.append(gt_targets_scaled_np_slice)
                
                if self.args.c_out > num_targets_to_process: # Covariates part (already scaled)
                    scaled_gt_parts_np.append(batch_y_segment_for_loss_np[:, :, num_targets_to_process:])

                if len(scaled_gt_parts_np) > 0:
                    true_values_final_scaled_np = np.concatenate(scaled_gt_parts_np, axis=-1)
                else:
                    # This case should ideally not be hit if c_out > 0
                    true_values_final_scaled_np = batch_y_segment_for_loss_np 

                preds.append(pred_scaled)
                trues.append(true_values_final_scaled_np) # Use the correctly constructed scaled ground truth
        
        # Concatenate all predictions and true values (both in scaled space)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        # Calculate metrics on scaled data (consistent with training/validation)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        
        logger.info(f"Test Results (Scaled Space) - MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}")
        logger.info(f"Test Results (Scaled Space) - MAPE: {mape:.6f}, MSPE: {mspe:.6f}")
        
        return {
            'mse': mse, 'mae': mae, 'rmse': rmse, 'mape': mape, 'mspe': mspe,
            'predictions': preds, 'true_values': trues
        }
    
        
    # Fix the train method around line 340-350

    def train(self):
        """Main training loop"""
        logger.info("Starting training")
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        # Create checkpoint directory with proper model_id handling
        model_id = getattr(self.args, 'model_id', 'timesnet_financial')
        checkpoint_dir = os.path.join('checkpoints', model_id)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
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
            
            # Early stopping check - use checkpoint_dir instead of self.args.checkpoints
            self.early_stopping(val_loss, self.model, checkpoint_dir)
            if self.early_stopping.early_stop:
                logger.info("Early stopping triggered")
                break
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Create checkpoints directory if it doesn't exist
                os.makedirs(checkpoint_dir, exist_ok=True)
                self.save_model(os.path.join(checkpoint_dir, 'best_model.pth'))
        
        # Load best model for testing
        self.load_model(os.path.join(checkpoint_dir, 'best_model.pth'))
        
        # Test model
        test_results = self.test()
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'test_results': test_results
        }
    
    # Fix the save_model and load_model methods

    def save_model(self, full_path):
        """Save model checkpoint"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'args': self.args
        }, full_path)
        logger.debug(f"Model saved to {full_path}")

    def load_model(self, full_path):
        """Load model checkpoint"""
        if os.path.exists(full_path):
            checkpoint = torch.load(full_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"Model loaded from {full_path}")
        else:
            logger.warning(f"Checkpoint not found: {full_path}")
    
    # Update the predict_production method to include inverse transform

    def predict_production(self):
        """Generate production predictions for future business days"""
        logger.info(f"Generating production predictions for {self.args.prod_len} future business days")

        # This method needs to be implemented with real prediction logictime 
        # For now, showing the correct decoder input pattern

        logger.info("Production prediction - Decoder input pattern:")
        logger.info("  Historical period: Real targets + real covariates")  
        logger.info("  Future period: Zero targets + REAL future covariates")
        logger.info("  Output: Scaled predictions (use inverse_transform_targets() for business use)")

        # Example implementation structure:
        """
        # Load latest data
        data_path = os.path.join(self.args.root_path, 'prepared_financial_data.csv')
        df = pd.read_csv(data_path)

        # Get sequence for prediction (last seq_len + label_len + pred_len points)
        sequence_data = df.tail(self.args.seq_len + self.args.label_len + self.args.pred_len)

        # Prepare batch_x (historical features)
        batch_x = sequence_data.iloc[:self.args.seq_len, 1:].values  # Remove date column
        batch_x = torch.from_numpy(batch_x).float().unsqueeze(0).to(self.device)

        # Prepare batch_y (label + prediction period) 
        batch_y = sequence_data.iloc[self.args.seq_len:, 1:].values  # Remove date column
        batch_y = torch.from_numpy(batch_y).float().unsqueeze(0).to(self.device)

        # Prepare time features (batch_x_mark, batch_y_mark)
        # ... (similar to batch_x/batch_y but for time features)

        # Prepare decoder input - SAME PATTERN AS TRAINING/TEST
        dec_inp_historical = batch_y[:, :self.args.label_len, :].float().to(self.device)

        # Future period: Zero targets, keep covariates  
        future_targets_zero = torch.zeros_like(batch_y[:, -self.args.pred_len:, :4]).float().to(self.device)
        future_covariates = batch_y[:, -self.args.pred_len:, 4:].float().to(self.device)
        dec_inp_future = torch.cat([future_targets_zero, future_covariates], dim=-1)

        # Combine historical + future
        dec_inp = torch.cat([dec_inp_historical, dec_inp_future], dim=1).float().to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        # Extract target predictions (scaled)
        predictions_scaled = outputs[:, -self.args.pred_len:, :4].detach().cpu().numpy()

        # Inverse transform to original scale for business use
        predictions_original = self.train_loader.dataset.inverse_transform_targets(predictions_scaled)

        return predictions_original
        """

        return None
        
    def quick_diagnostic(self):
        """Run quick diagnostic to check for obvious performance issues"""
        logger.info("🔍 Running Quick Performance Diagnostic...")
        
        # Test data loading
        start_time = time.time()
        train_data, train_loader = self.data_provider("train")
        data_time = time.time() - start_time
        logger.info(f"   Data loading: {data_time:.3f}s")
        
        # Test model creation
        start_time = time.time()
        model = self.build_model()
        model_time = time.time() - start_time
        logger.info(f"   Model creation: {model_time:.3f}s")
        
        # Test single batch
        model.train()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate)
        
        batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(train_loader))
        
        start_time = time.time()
        # Move to device
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        
        # Prepare decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        
        # Calculate loss (only on target columns - first 4 features)
        target_outputs = outputs[:, -self.args.pred_len:, :4]
        target_y = batch_y[:, -self.args.pred_len:, :4]
        loss = criterion(target_outputs, target_y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        batch_time = time.time() - start_time
        logger.info(f"   Single batch training: {batch_time:.3f}s")
        
        # Estimate total time
        num_batches = len(train_loader)
        estimated_time = batch_time * num_batches * self.args.train_epochs / 60
        logger.info(f"   Estimated total training time: {estimated_time:.1f} minutes")
        
        if batch_time > 2.0:
            logger.warning("⚠️ Training is very slow! Consider:")
            logger.warning("   - Reducing model size (d_model, layers)")
            logger.warning("   - Reducing batch size")
            logger.warning("   - Using CPU if GPU is not available")
            return False
        elif batch_time > 0.5:
            logger.warning("🟡 Training speed is moderate")
            return True
        else:
            logger.info("✅ Training speed looks good!")
            return True
    
    # ...existing code...
def load_config(config_path='config/config.yaml'):
    """
    Load configuration from file
    Supports both YAML and JSON formats
    """
    # Try different config file locations and formats
    config_files_to_try = [
        config_path,
        'config/config.yaml',
        'config.yaml',  # fallback for backward compatibility
        'config/config.yml', 
        'config/config.json',
        'config/timesnet_config.yaml',
        'timesnet_config.yaml'  # fallback
    ]
    
    config = None
    config_file_used = None
    
    for config_file in config_files_to_try:
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    if config_file.endswith(('.yaml', '.yml')) and YAML_AVAILABLE:
                        config = yaml.safe_load(f)
                    else:
                        config = json.load(f)
                config_file_used = config_file
                break
            except Exception as e:
                print(f"⚠️ Error loading config file {config_file}: {e}")
                continue
    
    if config is None:
        print("⚠️ No config file found. Using default settings.")
        print("💡 Create config/config.yaml or config.yaml to customize settings")
        return {}
    
    print(f"✅ Loaded configuration from: {config_file_used}")
    return config

# Fix the create_args function around line 600-650

def create_args(config_path='config/config.yaml'):
    """Create argument parser with config file support"""
    parser = argparse.ArgumentParser(description='TimesNet Financial Forecasting')
    
    # Config file option
    parser.add_argument('--config', type=str, default=config_path, 
                    help='Path to configuration file (YAML or JSON)')
    
    # Parse config file argument first
    temp_args, _ = parser.parse_known_args()
    
    # Load configuration from file
    config = load_config(temp_args.config)
    
    # Basic config - ADD MISSING model_id
    parser.add_argument('--model_id', type=str, default=config.get('model_id', 'timesnet_financial'), 
                    help='model identifier')
    
    # Data settings - FIX ROOT PATH
    parser.add_argument('--data', type=str, default=config.get('data', 'custom'), 
                    help='dataset type')
    parser.add_argument('--root_path', type=str, default=config.get('root_path', 'data'), 
                    help='root path of the data file')  # Changed from './data/' to 'data'
    parser.add_argument('--data_path', type=str, default=config.get('data_path', 'prepared_financial_data.csv'), 
                    help='data file')
    
    # ... rest of arguments stay the same ...
    parser.add_argument('--features', type=str, default=config.get('features', 'M'), 
                    help='forecasting task [M, S, MS]')
    parser.add_argument('--target', type=str, default=config.get('target', 'log_Close'), 
                    help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default=config.get('freq', 'b'), 
                    help='freq for time features encoding [b=business day]')
    parser.add_argument('--checkpoints', type=str, default=config.get('checkpoints', './checkpoints/'), 
                    help='location of model checkpoints')
    
    # Forecasting task
    parser.add_argument('--seq_len', type=int, default=config.get('seq_len', 100), 
                    help='input sequence length')
    parser.add_argument('--label_len', type=int, default=config.get('label_len', 10), 
                    help='start token length')
    parser.add_argument('--pred_len', type=int, default=config.get('pred_len', 10), 
                    help='prediction sequence length')
    parser.add_argument('--val_len', type=int, default=config.get('val_len', 10), 
                    help='validation length')
    parser.add_argument('--test_len', type=int, default=config.get('test_len', 10), 
                    help='test length')
    parser.add_argument('--prod_len', type=int, default=config.get('prod_len', 10), 
                    help='production prediction length')
    
    # Model define
    parser.add_argument('--top_k', type=int, default=config.get('top_k', 5), 
                    help='TimesNet kernel size')
    parser.add_argument('--num_kernels', type=int, default=config.get('num_kernels', 5), 
                    help='for Inception')
    parser.add_argument('--enc_in', type=int, default=config.get('enc_in', 118), 
                    help='encoder input size (114 covariates + 4 targets)')
    parser.add_argument('--dec_in', type=int, default=config.get('dec_in', 118), 
                    help='decoder input size')
    parser.add_argument('--c_out', type=int, default=config.get('c_out', 118), 
                    help='output size')
    parser.add_argument('--d_model', type=int, default=config.get('d_model', 64), 
                    help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=config.get('n_heads', 4), 
                    help='num of heads')
    parser.add_argument('--e_layers', type=int, default=config.get('e_layers', 2), 
                    help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=config.get('d_layers', 1), 
                    help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=config.get('d_ff', 128), 
                    help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=config.get('moving_avg', 25), 
                    help='window size of moving average')
    parser.add_argument('--factor', type=int, default=config.get('factor', 1), 
                    help='attn factor')
    parser.add_argument('--distil', action='store_false', 
                    default=not config.get('distil', False),
                    help='whether to use distilling in encoder')
    parser.add_argument('--dropout', type=float, default=config.get('dropout', 0.1), 
                    help='dropout')
    parser.add_argument('--embed', type=str, default=config.get('embed', 'timeF'), 
                    help='time features encoding [timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default=config.get('activation', 'gelu'), 
                    help='activation')
    parser.add_argument('--output_attention', action='store_true', 
                    default=config.get('output_attention', False),
                    help='whether to output attention in encoder')
    
    # Optimization
    parser.add_argument('--num_workers', type=int, default=config.get('num_workers', 10), 
                    help='data loader num workers')
    parser.add_argument('--itr', type=int, default=config.get('itr', 1), 
                    help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=config.get('train_epochs', 100), 
                    help='train epochs')
    parser.add_argument('--batch_size', type=int, default=config.get('batch_size', 32), 
                    help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=config.get('patience', 10), 
                    help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=config.get('learning_rate', 0.0001), 
                    help='optimizer learning rate')
    parser.add_argument('--des', type=str, default=config.get('des', 'financial_forecast'), 
                    help='exp description')
    parser.add_argument('--loss', type=str, default=config.get('loss', 'MSE'), 
                    help='loss function')
    parser.add_argument('--lradj', type=str, default=config.get('lradj', 'type1'), 
                    help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', 
                    default=config.get('use_amp', False),
                    help='use automatic mixed precision training')
    parser.add_argument('--seed', type=int, default=config.get('seed', 2024), 
                    help='random seed')
    # Task specific
    parser.add_argument('--task_name', type=str, default=config.get('task_name', 'long_term_forecast'), 
                    help='task name')
    
    args = parser.parse_args()      # Automatic c_out detection based on features mode
    if args.features == 'M':
        # Full multivariate forecasting - predict all features
        args.c_out = args.enc_in  # Match input size (118)
        logger.info(f"🎯 Multivariate mode: c_out automatically set to {args.c_out} (all features)")
    elif args.features == 'MS':
        # Multivariate-to-MULTI-target mode
        # Determine c_out from the number of specified target columns
        num_targets = len(args.target.split(','))
        if num_targets == 0:
            raise ValueError("For 'MS' mode, 'target' must be a comma-separated list of target column names.")
        args.c_out = num_targets
        logger.info(f"🎯 Multivariate-to-Multi-target mode: c_out set to {args.c_out} (targets: {args.target})")
    else:
        # Univariate (Targets-Only) forecasting ('S' mode):
        # Uses only the columns specified in args.target as input and predicts all of them.
        # enc_in is already dynamically set to the number of columns in args.target.
        args.c_out = args.enc_in  # Predict all input target features
        logger.info(f"🎯 Univariate (Targets-Only) mode ('S'): Using only target columns as input. c_out set to enc_in ({args.c_out}) for targets: {args.target}")
    
    # Display configuration summary
    print("\n🔧 CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"📁 Data: {args.data_path} (features: {args.features})")
    print(f"📏 Sequences: {args.seq_len} → {args.pred_len} (label: {args.label_len})")
    print(f"🧠 Model: d_model={args.d_model}, layers={args.e_layers}, heads={args.n_heads}")
    print(f"⚡ Training: epochs={args.train_epochs}, batch={args.batch_size}, lr={args.learning_rate}")
    print(f"🎯 TimesNet: top_k={args.top_k}, kernels={args.num_kernels}")
    print("=" * 60)
    
    return args


# Fix the main function around line 740-760

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
    
    # Set model_id if not already set
    if not hasattr(args, 'model_id'):
        args.model_id = setting
    
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
