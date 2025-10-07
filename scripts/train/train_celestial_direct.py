#!/usr/bin/env python3
"""
Direct Training Script for Celestial Enhanced PGAT
Simplified approach without complex experiment framework
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
from models.Celestial_Enhanced_PGAT import Model
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import warnings
warnings.filterwarnings('ignore')

class SimpleConfig:
    """Simple configuration class"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

def train_celestial_pgat():
    """Direct training function"""
    print("🌌 Starting Direct Celestial Enhanced PGAT Training")
    print("=" * 60)
    
    # Load configuration
    config_path = "configs/celestial_enhanced_pgat.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create config object
    args = SimpleConfig(config_dict)
    
    # Add missing required attributes
    args.task_name = 'long_term_forecast'
    args.model_name = 'Celestial_Enhanced_PGAT'
    args.data_name = 'custom'
    args.checkpoints = './checkpoints/'
    args.inverse = False
    args.cols = None
    args.num_workers = 10
    args.itr = 1
    args.train_only = False
    args.do_predict = False
    
    # Wave aggregation settings
    args.aggregate_waves_to_celestial = True
    args.wave_to_celestial_mapping = True
    args.celestial_node_features = 13
    args.target_wave_indices = [0, 1, 2, 3]  # OHLC indices
    
    print(f"📊 Configuration loaded:")
    print(f"   - Model: {args.model}")
    print(f"   - Data: {args.data}")
    print(f"   - Sequence length: {args.seq_len}")
    print(f"   - Prediction length: {args.pred_len}")
    print(f"   - Model dimension: {args.d_model}")
    
    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # Get data loaders
    print("📂 Loading data...")
    train_data, train_loader = data_provider(args, flag='train')
    vali_data, vali_loader = data_provider(args, flag='val')
    test_data, test_loader = data_provider(args, flag='test')
    
    print(f"   - Training batches: {len(train_loader)}")
    print(f"   - Validation batches: {len(vali_loader)}")
    print(f"   - Test batches: {len(test_loader)}")
    
    # Initialize model
    print("🏗️  Initializing Celestial Enhanced PGAT...")
    try:
        model = Model(args).to(device)
        print(f"✅ Model initialized successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Setup training components
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=getattr(args, 'weight_decay', 0.0001))
    
    # Use proper loss function based on model configuration
    if getattr(model, 'use_mixture_decoder', False):
        # Import the proper mixture loss function that handles multivariate targets
        from layers.modular.decoder.mixture_density_decoder import MixtureNLLLoss
        criterion = MixtureNLLLoss(multivariate_mode='independent')
        print("🎯 Using Gaussian Mixture NLL Loss for probabilistic predictions")
    else:
        criterion = nn.MSELoss()
        print("📊 Using MSE Loss for deterministic predictions")
    
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoints) / args.model_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("\n🔥 Starting training loop...")
    train_losses = []
    val_losses = []
    
    for epoch in range(args.train_epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Move to device
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            # Prepare decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            
            try:
                # Forward pass
                outputs, metadata = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # Determine the correct ground truth for the loss function
                if (model.aggregate_waves_to_celestial and 
                    'original_targets' in metadata and 
                    metadata['original_targets'] is not None):
                    # Use the specific targets isolated during wave aggregation
                    true_targets = metadata['original_targets'][:, -args.pred_len:, :]
                    print(f"🎯 Using wave aggregation targets: {true_targets.shape}")
                else:
                    # Use the standard batch_y targets
                    true_targets = batch_y[:, -args.pred_len:, :]
                
                # Compute loss based on model type
                if getattr(model, 'use_mixture_decoder', False) and isinstance(outputs, dict):
                    # Mixture density network output
                    means = outputs['means']
                    log_stds = outputs['log_stds'] 
                    log_weights = outputs['log_weights']
                    loss = criterion((means, log_stds, log_weights), true_targets)
                elif getattr(model, 'use_mixture_decoder', False) and isinstance(outputs, tuple):
                    # Tuple output (means, log_stds, log_weights)
                    means, log_stds, log_weights = outputs
                    loss = criterion((means, log_stds, log_weights), true_targets)
                else:
                    # Standard deterministic output
                    if hasattr(args, 'target_features') and args.target_features:
                        target_indices = args.target_features
                        loss = criterion(outputs[:, :, target_indices], true_targets[:, :, target_indices])
                    else:
                        loss = criterion(outputs, true_targets)
                
                # Add regularization loss from stochastic graph learner
                if getattr(model, 'use_stochastic_learner', False):
                    reg_loss = model.get_regularization_loss()
                    reg_weight = getattr(args, 'reg_loss_weight', 0.1)
                    loss += reg_loss * reg_weight
                    
                    if i % 100 == 0:  # Log regularization loss occasionally
                        print(f"         📊 Regularization loss: {reg_loss.item():.6f}")
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if hasattr(args, 'clip_grad_norm'):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
                
                # Log progress
                if i % 50 == 0:
                    print(f"      Epoch {epoch+1:3d} | Batch {i:4d}/{len(train_loader)} | Loss: {loss.item():.6f}")
                    
                    # Print celestial metadata
                    if metadata and 'celestial_metadata' in metadata:
                        celestial_meta = metadata['celestial_metadata']
                        if 'most_active_body' in celestial_meta:
                            body_names = celestial_meta.get('body_names', [])
                            if celestial_meta['most_active_body'] < len(body_names):
                                active_body = body_names[celestial_meta['most_active_body']]
                                print(f"         🌌 Most active celestial body: {active_body}")
                
            except Exception as e:
                print(f"❌ Training step failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        avg_train_loss = train_loss / max(train_batches, 1)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
                
                try:
                    outputs, metadata = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                    # Determine correct ground truth (same logic as training)
                    if (model.aggregate_waves_to_celestial and 
                        'original_targets' in metadata and 
                        metadata['original_targets'] is not None):
                        true_targets = metadata['original_targets'][:, -args.pred_len:, :]
                    else:
                        true_targets = batch_y[:, -args.pred_len:, :]
                    
                    # Compute loss based on model type
                    if getattr(model, 'use_mixture_decoder', False) and isinstance(outputs, dict):
                        means = outputs['means']
                        log_stds = outputs['log_stds']
                        log_weights = outputs['log_weights']
                        loss = criterion((means, log_stds, log_weights), true_targets)
                    elif getattr(model, 'use_mixture_decoder', False) and isinstance(outputs, tuple):
                        means, log_stds, log_weights = outputs
                        loss = criterion((means, log_stds, log_weights), true_targets)
                    else:
                        if hasattr(args, 'target_features') and args.target_features:
                            target_indices = args.target_features
                            loss = criterion(outputs[:, :, target_indices], true_targets[:, :, target_indices])
                        else:
                            loss = criterion(outputs, true_targets)
                    
                    val_loss += loss.item()
                    val_batches += 1
                    
                except Exception as e:
                    print(f"⚠️  Validation step warning: {e}")
                    continue
        
        avg_val_loss = val_loss / max(val_batches, 1)
        val_losses.append(avg_val_loss)
        
        # Learning rate adjustment
        adjust_learning_rate(optimizer, epoch + 1, args)
        
        # Progress report
        epoch_time = time.time() - epoch_start
        print(f"   Epoch {epoch+1:3d}/{args.train_epochs} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} | "
              f"Time: {epoch_time:.2f}s")
        
        # Early stopping
        early_stopping(avg_val_loss, model, str(checkpoint_dir))
        if early_stopping.early_stop:
            print(f"   ⏹️  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    best_model_path = checkpoint_dir / 'checkpoint.pth'
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path))
        print("✅ Best model loaded")
    
    # Final evaluation
    print("\n📊 Final Evaluation...")
    model.eval()
    
    # Pre-allocate arrays for better memory efficiency
    num_test_samples = len(test_data)
    preds = np.zeros((num_test_samples, args.pred_len, args.c_out))
    trues = np.zeros((num_test_samples, args.pred_len, args.c_out))
    current_index = 0
    
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            
            try:
                outputs, metadata = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # Handle mixture decoder outputs - extract point predictions
                if getattr(model, 'use_mixture_decoder', False):
                    if isinstance(outputs, dict):
                        # Extract point prediction from mixture
                        from utils.mixture_loss import extract_point_prediction
                        pred_tensor = extract_point_prediction(
                            outputs['means'], outputs['log_stds'], outputs['log_weights']
                        )
                    elif isinstance(outputs, tuple):
                        means, log_stds, log_weights = outputs
                        from utils.mixture_loss import extract_point_prediction
                        pred_tensor = extract_point_prediction(means, log_stds, log_weights)
                    else:
                        pred_tensor = outputs
                else:
                    pred_tensor = outputs
                
                # Determine correct ground truth
                if (model.aggregate_waves_to_celestial and 
                    'original_targets' in metadata and 
                    metadata['original_targets'] is not None):
                    true_tensor = metadata['original_targets'][:, -args.pred_len:, :]
                else:
                    true_tensor = batch_y[:, -args.pred_len:, :]
                
                # Convert to numpy
                pred = pred_tensor.detach().cpu().numpy()
                true = true_tensor.detach().cpu().numpy()
                
                # Calculate the slice to fill
                batch_size = pred.shape[0]
                start_index = current_index
                end_index = start_index + batch_size
                
                # Fill the pre-allocated arrays directly
                if end_index <= num_test_samples:
                    preds[start_index:end_index, :, :] = pred
                    trues[start_index:end_index, :, :] = true
                    current_index = end_index
                
            except Exception as e:
                print(f"⚠️  Test step warning: {e}")
                continue
    
    # Trim arrays to actual size
    if current_index < num_test_samples:
        preds = preds[:current_index]
        trues = trues[:current_index]
    
    if current_index > 0:
        
        # Compute metrics
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        
        print("\n🎯 Final Results:")
        print(f"   - MAE:  {mae:.6f}")
        print(f"   - MSE:  {mse:.6f}")
        print(f"   - RMSE: {rmse:.6f}")
        print(f"   - MAPE: {mape:.6f}")
        print(f"   - MSPE: {mspe:.6f}")
        
        # Save results
        results = {
            'model': 'Celestial_Enhanced_PGAT',
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape),
            'mspe': float(mspe),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'config': config_dict,
            'timestamp': datetime.now().isoformat()
        }
        
        results_file = checkpoint_dir / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {results_file}")
        
        print("\n" + "=" * 60)
        print("🎉 Celestial Enhanced PGAT Training Complete!")
        print("🌌 The Astrological AI has learned the patterns of the cosmos!")
        
        return True
    else:
        print("❌ No valid predictions generated")
        return False

def main():
    """Main function"""
    try:
        success = train_celestial_pgat()
        return 0 if success else 1
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())