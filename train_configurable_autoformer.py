#!/usr/bin/env python3
"""
Enhanced Autoformer Training Script with Config Support
Supports all three forecasting modes: M, MS, S
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.EnhancedAutoformer import Model as EnhancedAutoformer
from models.BayesianEnhancedAutoformer import Model as BayesianEnhancedAutoformer  
from models.HierarchicalEnhancedAutoformer import Model as HierarchicalEnhancedAutoformer
from utils.logger import logger
from data_provider.data_loader import Dataset_Custom
from torch.utils.data import DataLoader
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric


class ConfigurableAutoformerTrainer:
    """Configurable trainer for all Enhanced Autoformer variants with M/MS/S mode support"""
    
    def __init__(self, config_path, model_type='enhanced'):
        """
        Initialize trainer with config file
        
        Args:
            config_path: Path to YAML config file
            model_type: 'enhanced', 'bayesian', or 'hierarchical'
        """
        self.config_path = config_path
        self.model_type = model_type
        self.load_config()
        self.setup_environment()
        self.create_data_loaders()
        self.build_model()
        
    def load_config(self):
        """Load configuration from YAML file"""
        logger.info(f"Loading config from: {self.config_path}")
        
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Convert to argparse-like object for compatibility
        self.args = argparse.Namespace(**config)
        
        # Validate mode configuration
        self.validate_mode_config()
        
        logger.info(f"Loaded config for model: {self.args.model_id}")
        logger.info(f"Forecasting mode: {self.args.features}")
        logger.info(f"Input features: {self.args.enc_in}, Output features: {self.args.c_out}")
        
    def validate_mode_config(self):
        """Validate configuration for the specified forecasting mode"""
        mode = self.args.features
        
        if mode == 'M':
            # Multivariate: All features → All features
            assert self.args.enc_in == 118, f"M mode should have enc_in=118, got {self.args.enc_in}"
            assert self.args.c_out == 118, f"M mode should have c_out=118, got {self.args.c_out}"
            logger.info("✅ M Mode: All features → All features")
            
        elif mode == 'MS':
            # Multivariate to Multi-target: All features → Target features
            assert self.args.enc_in == 118, f"MS mode should have enc_in=118, got {self.args.enc_in}"
            assert self.args.c_out == 4, f"MS mode should have c_out=4, got {self.args.c_out}"
            logger.info("✅ MS Mode: All features → Target features")
            
        elif mode == 'S':
            # Target-only: Target features → Target features
            assert self.args.enc_in == 4, f"S mode should have enc_in=4, got {self.args.enc_in}"
            assert self.args.c_out == 4, f"S mode should have c_out=4, got {self.args.c_out}"
            logger.info("✅ S Mode: Target features → Target features")
            
        else:
            raise ValueError(f"Unsupported forecasting mode: {mode}. Use M, MS, or S")
    
    def setup_environment(self):
        """Setup training environment"""
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Set random seeds
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.args.seed)
            torch.cuda.manual_seed_all(self.args.seed)
    
    def create_data_loaders(self):
        """Create data loaders based on forecasting mode"""
        logger.info(f"Creating data loaders for mode: {self.args.features}")
        
        # Set validation and test lengths for border calculation
        self.args.validation_length = self.args.val_len
        self.args.test_length = self.args.test_len
        
        # Create datasets
        self.train_data = Dataset_Custom(
            args=self.args,
            root_path=self.args.root_path,
            data_path=self.args.data_path,
            flag='train',
            size=[self.args.seq_len, self.args.label_len, self.args.pred_len],
            features=self.args.features,
            target=self.args.target,
            scale=True,
            timeenc=1,
            freq=self.args.freq
        )
        
        self.val_data = Dataset_Custom(
            args=self.args,
            root_path=self.args.root_path,
            data_path=self.args.data_path,
            flag='val',
            size=[self.args.seq_len, self.args.label_len, self.args.pred_len],
            features=self.args.features,
            target=self.args.target,
            scale=True,
            timeenc=1,
            freq=self.args.freq
        )
        
        self.test_data = Dataset_Custom(
            args=self.args,
            root_path=self.args.root_path,
            data_path=self.args.data_path,
            flag='test',
            size=[self.args.seq_len, self.args.label_len, self.args.pred_len],
            features=self.args.features,
            target=self.args.target,
            scale=True,
            timeenc=1,
            freq=self.args.freq
        )
        
        # Create data loaders
        self.train_loader = DataLoader(self.train_data, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, drop_last=True)
        self.val_loader = DataLoader(self.val_data, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, drop_last=True)
        self.test_loader = DataLoader(self.test_data, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, drop_last=False)
        
        logger.info(f"Data loaders created - Train: {len(self.train_loader)}, Val: {len(self.val_loader)}, Test: {len(self.test_loader)}")
    
    def build_model(self):
        """Build the specified model type"""
        logger.info(f"Building {self.model_type} model")
        
        if self.model_type == 'enhanced':
            self.model = EnhancedAutoformer(self.args)
        elif self.model_type == 'bayesian':
            self.model = BayesianEnhancedAutoformer(self.args)
        elif self.model_type == 'hierarchical':
            self.model = HierarchicalEnhancedAutoformer(self.args)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model = self.model.to(self.device)
        
        # Setup optimizer and criterion
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.criterion = torch.nn.MSELoss()
        
        # Early stopping
        checkpoint_dir = os.path.join(self.args.checkpoints, self.args.model_id)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        # Model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            
            # Move to device
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)
            
            # Prepare decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            
            # Forward pass
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            # Handle different output modes
            if self.args.features == 'M':
                # M mode: predict all features
                pred = outputs[:, -self.args.pred_len:, :]
                true = batch_y[:, -self.args.pred_len:, :]
            elif self.args.features == 'MS':
                # MS mode: predict target features only
                pred = outputs[:, -self.args.pred_len:, :4]  # First 4 features are targets
                true = batch_y[:, -self.args.pred_len:, :4]
            else:  # S mode
                # S mode: predict target features only (input already filtered)
                pred = outputs[:, -self.args.pred_len:, :]
                true = batch_y[:, -self.args.pred_len:, :]
            
            loss = self.criterion(pred, true)
            loss.backward()
            
            if hasattr(self.args, 'use_grad_clip') and self.args.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            
            self.optimizer.step()
            total_loss += loss.item()
            
            if i % self.args.log_interval == 0:
                logger.info(f'Batch {i}/{len(self.train_loader)} - Loss: {loss.item():.6f}')
        
        return total_loss / len(self.train_loader)
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in self.val_loader:
                # Move to device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # Prepare decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # Forward pass
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # Handle different output modes  
                if self.args.features == 'M':
                    pred = outputs[:, -self.args.pred_len:, :].detach().cpu()
                    true = batch_y[:, -self.args.pred_len:, :]
                elif self.args.features == 'MS':
                    pred = outputs[:, -self.args.pred_len:, :4].detach().cpu()  # First 4 features
                    true = batch_y[:, -self.args.pred_len:, :4]
                else:  # S mode
                    pred = outputs[:, -self.args.pred_len:, :].detach().cpu()
                    true = batch_y[:, -self.args.pred_len:, :]
                
                # Scale ground truth to match model outputs (validation data is unscaled)
                if hasattr(self.val_data, 'target_scaler') and self.val_data.target_scaler is not None:
                    true_np = true.numpy()
                    if self.args.features == 'M':
                        # For M mode, need to scale all features
                        true_scaled_np = self.val_data.scaler.transform(
                            true_np.reshape(-1, true_np.shape[-1])
                        ).reshape(true_np.shape)
                    else:
                        # For MS and S modes, scale only target features
                        true_scaled_np = self.val_data.target_scaler.transform(
                            true_np.reshape(-1, true_np.shape[-1])
                        ).reshape(true_np.shape)
                    true = torch.from_numpy(true_scaled_np).float()
                
                loss = self.criterion(pred, true)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def test(self):
        """Test the model"""
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
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # Forward pass
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # Handle different output modes
                if self.args.features == 'M':
                    pred = outputs[:, -self.args.pred_len:, :].detach().cpu().numpy()
                    true = batch_y[:, -self.args.pred_len:, :].detach().cpu().numpy()
                elif self.args.features == 'MS':
                    pred = outputs[:, -self.args.pred_len:, :4].detach().cpu().numpy()  # First 4 features
                    true = batch_y[:, -self.args.pred_len:, :4].detach().cpu().numpy()
                else:  # S mode
                    pred = outputs[:, -self.args.pred_len:, :].detach().cpu().numpy()
                    true = batch_y[:, -self.args.pred_len:, :].detach().cpu().numpy()
                
                # Scale ground truth to match predictions
                if hasattr(self.test_data, 'target_scaler') and self.test_data.target_scaler is not None:
                    if self.args.features == 'M':
                        true = self.test_data.scaler.transform(
                            true.reshape(-1, true.shape[-1])
                        ).reshape(true.shape)
                    else:
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
        
        return {'mse': mse, 'mae': mae, 'rmse': rmse, 'mape': mape, 'mspe': mspe}
    
    def train(self):
        """Main training loop"""
        logger.info(f"Starting training for {self.model_type} model in {self.args.features} mode")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.args.train_epochs):
            logger.info(f"Epoch {epoch+1}/{self.args.train_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate_epoch()
            
            logger.info(f"Epoch {epoch+1} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping
            checkpoint_dir = os.path.join(self.args.checkpoints, self.args.model_id)
            self.early_stopping(val_loss, self.model, checkpoint_dir)
            
            if self.early_stopping.early_stop:
                logger.info("Early stopping")
                break
                
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
            
            # Adjust learning rate
            adjust_learning_rate(self.optimizer, epoch + 1, self.args)
        
        # Test
        test_results = self.test()
        
        logger.info(f"Training completed. Best val loss: {best_val_loss:.6f}")
        return test_results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Configurable Enhanced Autoformer Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--model_type', type=str, default='enhanced', 
                       choices=['enhanced', 'bayesian', 'hierarchical'],
                       help='Type of enhanced autoformer to train')
    
    args = parser.parse_args()
    
    # Create trainer and train
    trainer = ConfigurableAutoformerTrainer(args.config, args.model_type)
    results = trainer.train()
    
    print(f"\n🎯 Final Results for {trainer.model_type} model in {trainer.args.features} mode:")
    print(f"   MSE: {results['mse']:.6f}")
    print(f"   MAE: {results['mae']:.6f}")
    print(f"   RMSE: {results['rmse']:.6f}")


if __name__ == '__main__':
    main()
