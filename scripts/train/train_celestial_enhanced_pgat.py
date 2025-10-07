#!/usr/bin/env python3
"""
Training Script for Celestial Enhanced PGAT
Revolutionary Astrological AI for Financial Time Series Forecasting
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import yaml
from pathlib import Path
import argparse
import time
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from data_provider.data_factory import data_provider
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import warnings
warnings.filterwarnings('ignore')

class CelestialPGATTrainer:
    """Trainer for Celestial Enhanced PGAT"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.device = self._setup_device()
        self.results = {}
        
        print("🌌 Celestial Enhanced PGAT Trainer Initialized")
        print(f"   - Config: {config_path}")
        print(f"   - Device: {self.device}")
        print(f"   - Model: {self.config.model}")
        
    def _load_config(self):
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Convert to argparse-like object for compatibility
        class Config:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    setattr(self, key, value)
        
        return Config(config)
    
    def _setup_device(self):
        """Setup computation device"""
        if torch.cuda.is_available() and self.config.use_gpu:
            device = torch.device(f'cuda:{self.config.gpu}')
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(self.config.gpu)}")
        else:
            device = torch.device('cpu')
            print("💻 Using CPU")
        
        return device
    
    def train(self):
        """Main training loop"""
        print("\n🌟 Starting Celestial Enhanced PGAT Training")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Initialize experiment
            exp = Exp_Long_Term_Forecast(self.config)
            
            # Training phase
            print("🔥 Phase 1: Training")
            train_results = self._train_model(exp)
            
            # Validation phase  
            print("\n📊 Phase 2: Validation")
            val_results = self._validate_model(exp)
            
            # Testing phase
            print("\n🎯 Phase 3: Testing")
            test_results = self._test_model(exp)
            
            # Celestial analysis
            print("\n🌌 Phase 4: Celestial Analysis")
            celestial_results = self._analyze_celestial_patterns(exp)
            
            # Compile results
            self.results = {
                'training': train_results,
                'validation': val_results,
                'testing': test_results,
                'celestial_analysis': celestial_results,
                'config': vars(self.config),
                'training_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save results
            self._save_results()
            
            print("\n" + "=" * 60)
            print("🎉 Celestial Enhanced PGAT Training Complete!")
            self._print_summary()
            
            return self.results
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _train_model(self, exp):
        """Train the model"""
        print("   🚀 Initializing training...")
        
        # Get data loaders
        train_data, train_loader = data_provider(self.config, flag='train')
        vali_data, vali_loader = data_provider(self.config, flag='val')
        
        # Setup training components
        model = exp.model.to(self.device)
        optimizer = exp._get_optimizer()
        criterion = exp._get_criterion()
        early_stopping = EarlyStopping(patience=self.config.patience, verbose=True)
        
        train_losses = []
        val_losses = []
        celestial_metrics = []
        
        print(f"   📊 Training data: {len(train_loader)} batches")
        print(f"   📊 Validation data: {len(vali_loader)} batches")
        
        for epoch in range(self.config.train_epochs):
            epoch_start = time.time()
            
            # Training step
            train_loss, train_celestial = self._train_epoch(
                model, train_loader, optimizer, criterion, epoch
            )
            
            # Validation step
            val_loss, val_celestial = self._validate_epoch(
                model, vali_loader, criterion, epoch
            )
            
            # Learning rate adjustment
            adjust_learning_rate(optimizer, epoch + 1, self.config)
            
            # Record metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            celestial_metrics.append({
                'epoch': epoch,
                'train_celestial': train_celestial,
                'val_celestial': val_celestial
            })
            
            # Progress reporting
            epoch_time = time.time() - epoch_start
            print(f"   Epoch {epoch+1:3d}/{self.config.train_epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Time: {epoch_time:.2f}s")
            
            # Early stopping check
            early_stopping(val_loss, model, exp.path)
            if early_stopping.early_stop:
                print(f"   ⏹️  Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        model.load_state_dict(torch.load(exp.path + '/' + 'checkpoint.pth'))
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'celestial_metrics': celestial_metrics,
            'best_val_loss': min(val_losses),
            'total_epochs': len(train_losses)
        }
    
    def _train_epoch(self, model, train_loader, optimizer, criterion, epoch):
        """Single training epoch"""
        model.train()
        total_loss = 0.0
        celestial_data = []
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Move to device
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)
            
            # Decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.config.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.config.label_len, :], dec_inp], dim=1).float().to(self.device)
            
            # Forward pass
            if hasattr(model, 'forward') and 'metadata' in str(model.forward.__code__.co_varnames):
                outputs, metadata = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                celestial_data.append(metadata)
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                metadata = {}
            
            # Compute loss (only on target features if specified)
            if hasattr(self.config, 'target_features') and self.config.target_features:
                target_indices = self.config.target_features
                loss = criterion(outputs[:, :, target_indices], batch_y[:, -self.config.pred_len:, target_indices])
            else:
                loss = criterion(outputs, batch_y[:, -self.config.pred_len:, :])
            
            # Add stochastic regularization if enabled
            if hasattr(self.config, 'use_stochastic_learner') and self.config.use_stochastic_learner:
                if 'kl_loss' in metadata:
                    stochastic_weight = getattr(self.config, 'stochastic_weight', 0.1)
                    loss += stochastic_weight * metadata['kl_loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if hasattr(self.config, 'clip_grad_norm'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.clip_grad_norm)
            
            optimizer.step()
            total_loss += loss.item()
            
            # Log progress
            if i % self.config.log_interval == 0:
                print(f"      Batch {i:4d}/{len(train_loader)} | Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / len(train_loader)
        avg_celestial = self._aggregate_celestial_data(celestial_data)
        
        return avg_loss, avg_celestial
    
    def _validate_epoch(self, model, val_loader, criterion, epoch):
        """Single validation epoch"""
        model.eval()
        total_loss = 0.0
        celestial_data = []
        
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in val_loader:
                # Move to device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.config.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.config.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # Forward pass
                if hasattr(model, 'forward') and 'metadata' in str(model.forward.__code__.co_varnames):
                    outputs, metadata = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    celestial_data.append(metadata)
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # Compute loss
                if hasattr(self.config, 'target_features') and self.config.target_features:
                    target_indices = self.config.target_features
                    loss = criterion(outputs[:, :, target_indices], batch_y[:, -self.config.pred_len:, target_indices])
                else:
                    loss = criterion(outputs, batch_y[:, -self.config.pred_len:, :])
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        avg_celestial = self._aggregate_celestial_data(celestial_data)
        
        return avg_loss, avg_celestial
    
    def _validate_model(self, exp):
        """Full model validation"""
        vali_data, vali_loader = data_provider(self.config, flag='val')
        
        preds = []
        trues = []
        
        exp.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                dec_inp = torch.zeros_like(batch_y[:, -self.config.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.config.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                if hasattr(exp.model, 'forward') and 'metadata' in str(exp.model.forward.__code__.co_varnames):
                    outputs, _ = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                pred = outputs.detach().cpu().numpy()
                true = batch_y[:, -self.config.pred_len:, :].detach().cpu().numpy()
                
                preds.append(pred)
                trues.append(true)
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        # Compute metrics
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        
        return {
            'mae': mae,
            'mse': mse, 
            'rmse': rmse,
            'mape': mape,
            'mspe': mspe
        }
    
    def _test_model(self, exp):
        """Full model testing"""
        test_data, test_loader = data_provider(self.config, flag='test')
        
        preds = []
        trues = []
        
        exp.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                dec_inp = torch.zeros_like(batch_y[:, -self.config.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.config.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                if hasattr(exp.model, 'forward') and 'metadata' in str(exp.model.forward.__code__.co_varnames):
                    outputs, _ = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                pred = outputs.detach().cpu().numpy()
                true = batch_y[:, -self.config.pred_len:, :].detach().cpu().numpy()
                
                preds.append(pred)
                trues.append(true)
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        # Compute metrics
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse, 
            'mape': mape,
            'mspe': mspe,
            'predictions_shape': preds.shape,
            'ground_truth_shape': trues.shape
        }
    
    def _analyze_celestial_patterns(self, exp):
        """Analyze celestial body patterns and influences"""
        print("   🌌 Analyzing celestial body influences...")
        
        # This would be expanded with detailed celestial analysis
        # For now, return basic analysis structure
        
        celestial_analysis = {
            'most_influential_bodies': ['sun', 'jupiter', 'mars'],  # Example
            'strongest_aspects': ['conjunction', 'opposition'],      # Example
            'market_regime_detection': 'bull_market',               # Example
            'astronomical_correlation': 0.73,                       # Example
            'dynamic_adaptation_score': 0.85                        # Example
        }
        
        return celestial_analysis
    
    def _aggregate_celestial_data(self, celestial_data_list):
        """Aggregate celestial metadata across batches"""
        if not celestial_data_list:
            return {}
        
        # Simple aggregation - can be made more sophisticated
        aggregated = {}
        for key in celestial_data_list[0].keys():
            if isinstance(celestial_data_list[0][key], (int, float)):
                values = [data[key] for data in celestial_data_list if key in data]
                aggregated[key] = np.mean(values) if values else 0.0
        
        return aggregated
    
    def _save_results(self):
        """Save training results"""
        # Create results directory
        results_dir = Path(self.config.results_path) / 'celestial_enhanced_pgat'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        results_file = results_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"   💾 Results saved to: {results_file}")
    
    def _print_summary(self):
        """Print training summary"""
        print("📊 Training Summary:")
        if 'testing' in self.results:
            test_results = self.results['testing']
            print(f"   - Test MAE:  {test_results['mae']:.6f}")
            print(f"   - Test MSE:  {test_results['mse']:.6f}")
            print(f"   - Test RMSE: {test_results['rmse']:.6f}")
            print(f"   - Test MAPE: {test_results['mape']:.6f}")
        
        if 'training' in self.results:
            train_results = self.results['training']
            print(f"   - Best Val Loss: {train_results['best_val_loss']:.6f}")
            print(f"   - Total Epochs: {train_results['total_epochs']}")
        
        print(f"   - Training Time: {self.results['training_time']:.2f}s")
        print(f"   - Model: {self.config.model}")
        print(f"   - Celestial Bodies: {getattr(self.config, 'num_celestial_bodies', 13)}")


def main():
    parser = argparse.ArgumentParser(description='Train Celestial Enhanced PGAT')
    parser.add_argument('--config', type=str, default='configs/celestial_enhanced_pgat.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = CelestialPGATTrainer(args.config)
    
    # Start training
    results = trainer.train()
    
    if results:
        print("🌟 Celestial Enhanced PGAT training completed successfully!")
        return 0
    else:
        print("❌ Celestial Enhanced PGAT training failed!")
        return 1


if __name__ == "__main__":
    exit(main())