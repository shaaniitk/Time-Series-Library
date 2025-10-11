#!/usr/bin/env python3
"""
Direct Training Script for Celestial Enhanced PGAT
Simplified approach without complex experiment framework
"""

import os
import sys
import io

# Ensure stdout/stderr can emit UTF-8 on Windows to avoid UnicodeEncodeError
try:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    else:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    else:
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except Exception:
    # Non-fatal; fallback to default platform encoding
    pass
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
    
    # Wave aggregation settings (respect config toggles; provide sane defaults)
    args.aggregate_waves_to_celestial = getattr(args, 'aggregate_waves_to_celestial', False)
    args.wave_to_celestial_mapping = getattr(args, 'wave_to_celestial_mapping', False)
    args.celestial_node_features = getattr(args, 'celestial_node_features', 13)
    args.target_wave_indices = getattr(args, 'target_wave_indices', [0, 1, 2, 3])
    
    print(f"📊 Configuration loaded:")
    print(f"   - Model: {args.model}")
    print(f"   - Data: {args.data}")
    print(f"   - Sequence length: {args.seq_len}")
    print(f"   - Prediction length: {args.pred_len}")
    print(f"   - Model dimension: {args.d_model}")
    
    # Setup device
    args.use_gpu = False  # Force CPU usage
    device = torch.device('cpu')
    print(f"🚀 Using device: {device} (GPU disabled due to environment issues)")
    
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
        # Import the sequential mixture loss function for better temporal modeling
        from layers.modular.decoder.sequential_mixture_decoder import SequentialMixtureNLLLoss
        criterion = SequentialMixtureNLLLoss(reduction='mean')
        print("🎯 Using Gaussian Mixture NLL Loss for probabilistic predictions")
    else:
        criterion = nn.MSELoss()
        print("📊 Using MSE Loss for deterministic predictions")
    
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoints) / args.model_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Auto-detect target indices from CSV header for slicing ---
    # Allows specifying target names in config (e.g., "log_Open,log_Close")
    # and ensures loss uses the correct target slices of batch_y
    target_indices = None
    try:
        if hasattr(args, 'target') and args.target:
            # Load header from CSV to map target names to indices (excluding 'date')
            import pandas as pd
            csv_path = os.path.join(args.root_path, args.data_path)
            print(f"\n📁 Data file: {csv_path}")
            df_head = pd.read_csv(csv_path, nrows=1)
            feature_cols = [c for c in df_head.columns if c != 'date']
            print(f"🧾 Columns ({len(df_head.columns)}): {list(df_head.columns)}")
            # Try to estimate row count without loading full CSV
            try:
                with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                    row_count = sum(1 for _ in f) - 1  # exclude header
                print(f"📏 CSV approx rows: {row_count}")
            except Exception as e_row:
                print(f"⚠️  Could not estimate row count: {e_row}")
            # Auto-set enc/dec input dims from CSV
            args.enc_in = len(feature_cols)
            args.dec_in = len(feature_cols)
            print(f"📐 Auto-set enc_in/dec_in from CSV: {args.enc_in}")
            if isinstance(args.target, str):
                target_names = [t.strip() for t in args.target.split(',')]
            else:
                target_names = list(args.target)
            name_to_index = {name: idx for idx, name in enumerate(feature_cols)}
            target_indices = [name_to_index[n] for n in target_names if n in name_to_index]
            # Fallback: if not found, use first c_out features
            if not target_indices:
                target_indices = list(range(getattr(args, 'c_out', 1)))
            print(f"🎯 Target indices resolved: {target_indices}")
    except Exception as e:
        print(f"⚠️  Failed to auto-detect target indices: {e}. Using first c_out features.")
        target_indices = list(range(getattr(args, 'c_out', 1)))

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
                    # Ensure BOTH time and channel dimensions align with ground truth
                    # Slice model outputs to last pred_len timesteps
                    out_time = outputs[:, -args.pred_len:, :] if outputs.shape[1] >= args.pred_len else outputs
                    if target_indices is not None:
                        # If model outputs more channels than targets, slice channels
                        out_slice = out_time[:, :, :len(target_indices)] if out_time.shape[-1] > len(target_indices) else out_time
                        gt_slice = true_targets[:, :, target_indices]
                        
                        # CRITICAL FIX: Scale targets to match scaled predictions (like standard framework)
                        if hasattr(train_data, 'scaler') and train_data.scaler is not None:
                            try:
                                # Scale the targets to match scaled predictions
                                gt_slice_np = gt_slice.cpu().numpy()
                                gt_slice_scaled_np = train_data.scaler.transform(
                                    gt_slice_np.reshape(-1, gt_slice_np.shape[-1])
                                ).reshape(gt_slice_np.shape)
                                gt_slice_scaled = torch.from_numpy(gt_slice_scaled_np).float().to(device)
                                loss = criterion(out_slice, gt_slice_scaled)
                            except Exception as e:
                                print(f"⚠️  Target scaling failed: {e}, using unscaled targets")
                                loss = criterion(out_slice, gt_slice)
                        else:
                            loss = criterion(out_slice, gt_slice)
                    else:
                        # Scale all targets if no specific indices
                        if hasattr(train_data, 'scaler') and train_data.scaler is not None:
                            try:
                                true_targets_np = true_targets.cpu().numpy()
                                true_targets_scaled_np = train_data.scaler.transform(
                                    true_targets_np.reshape(-1, true_targets_np.shape[-1])
                                ).reshape(true_targets_np.shape)
                                true_targets_scaled = torch.from_numpy(true_targets_scaled_np).float().to(device)
                                loss = criterion(out_time, true_targets_scaled)
                            except Exception as e:
                                print(f"⚠️  Target scaling failed: {e}, using unscaled targets")
                                loss = criterion(out_time, true_targets)
                        else:
                            loss = criterion(out_time, true_targets)
                
                # Add regularization loss from stochastic graph learner
                if getattr(model, 'use_stochastic_learner', False):
                    reg_loss = model.get_regularization_loss()
                    reg_weight = getattr(args, 'reg_loss_weight', 0.1)
                    loss += reg_loss * reg_weight
                    
                    if i % 100 == 0:  # Log regularization loss occasionally
                        print(f"         📊 Regularization loss: {reg_loss.item():.6f}")
                
                # Backward pass
                loss.backward()
                
                # Check gradient flow (debug)
                if i == 0:  # Only check first batch to avoid spam
                    total_grad_norm = 0.0
                    param_count = 0
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.data.norm(2)
                            total_grad_norm += grad_norm.item() ** 2
                            param_count += 1
                    total_grad_norm = total_grad_norm ** (1. / 2)
                    print(f"         🔍 Total gradient norm: {total_grad_norm:.6f} ({param_count} params)")
                    
                    # Check if parameters are changing
                    if not hasattr(model, '_first_param_value'):
                        model._first_param_value = next(model.parameters()).clone().detach()
                    else:
                        first_param = next(model.parameters())
                        param_change = (first_param - model._first_param_value).abs().mean().item()
                        print(f"         📈 Parameter change: {param_change:.8f}")
                        model._first_param_value = first_param.clone().detach()
                
                # Gradient clipping
                if hasattr(args, 'clip_grad_norm'):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
                
                # Log progress
                if i % 50 == 0:
                    print(f"      Epoch {epoch+1:3d} | Batch {i:4d}/{len(train_loader)} | Loss: {loss.item():.6f}")
                    
                    # Debug: Check if outputs are changing
                    if isinstance(outputs, dict) and 'point_prediction' in outputs:
                        output_mean = outputs['point_prediction'].mean().item()
                        output_std = outputs['point_prediction'].std().item()
                        print(f"         📊 Output stats: mean={output_mean:.6f}, std={output_std:.6f}")
                    elif isinstance(outputs, torch.Tensor):
                        output_mean = outputs.mean().item()
                        output_std = outputs.std().item()
                        print(f"         📊 Output stats: mean={output_mean:.6f}, std={output_std:.6f}")
                    
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
                        # Deterministic loss
                        # Slice model outputs to last pred_len timesteps to match ground truth
                        out_time = outputs[:, -args.pred_len:, :] if outputs.shape[1] >= args.pred_len else outputs
                        if target_indices is not None:
                            out_slice = out_time[:, :, :len(target_indices)] if out_time.shape[-1] > len(target_indices) else out_time
                            gt_slice = true_targets[:, :, target_indices]
                            
                            # CRITICAL FIX: Scale targets for validation loss too
                            if hasattr(vali_data, 'scaler') and vali_data.scaler is not None:
                                try:
                                    gt_slice_np = gt_slice.cpu().numpy()
                                    gt_slice_scaled_np = vali_data.scaler.transform(
                                        gt_slice_np.reshape(-1, gt_slice_np.shape[-1])
                                    ).reshape(gt_slice_np.shape)
                                    gt_slice_scaled = torch.from_numpy(gt_slice_scaled_np).float().to(device)
                                    loss = criterion(out_slice, gt_slice_scaled)
                                except Exception:
                                    loss = criterion(out_slice, gt_slice)
                            else:
                                loss = criterion(out_slice, gt_slice)
                        else:
                            # Scale all targets if no specific indices
                            if hasattr(vali_data, 'scaler') and vali_data.scaler is not None:
                                try:
                                    true_targets_np = true_targets.cpu().numpy()
                                    true_targets_scaled_np = vali_data.scaler.transform(
                                        true_targets_np.reshape(-1, true_targets_np.shape[-1])
                                    ).reshape(true_targets_np.shape)
                                    true_targets_scaled = torch.from_numpy(true_targets_scaled_np).float().to(device)
                                    loss = criterion(out_time, true_targets_scaled)
                                except Exception:
                                    loss = criterion(out_time, true_targets)
                            else:
                                loss = criterion(out_time, true_targets)
                    
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
    # Evaluation arrays sized to number of targets
    eval_c = len(target_indices) if target_indices is not None else args.c_out
    preds = np.zeros((num_test_samples, args.pred_len, eval_c))
    trues = np.zeros((num_test_samples, args.pred_len, eval_c))
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
                
                # Align time and target channels to evaluation dimensions
                # Ensure predictions and ground truth match (args.pred_len, eval_c)
                pred_aligned = pred_tensor[:, -args.pred_len:, :]
                true_aligned = true_tensor[:, -args.pred_len:, :]
                if target_indices is not None:
                    pred_aligned = pred_aligned[:, :, target_indices]
                    true_aligned = true_aligned[:, :, target_indices]

                # SCALING NOTE: For metrics computation, we can use either scaled or unscaled data
                # Both predictions and ground truth should be on the same scale
                # Here we use scaled data (consistent with loss computation)
                
                # Convert to numpy
                pred = pred_aligned.detach().cpu().numpy()  # Scaled predictions
                true = true_aligned.detach().cpu().numpy()  # Unscaled ground truth
                
                # OPTIONAL: Scale ground truth for consistent metrics computation
                if hasattr(test_data, 'scaler') and test_data.scaler is not None:
                    try:
                        # Scale ground truth to match scaled predictions for consistent metrics
                        true_scaled = test_data.scaler.transform(
                            true.reshape(-1, true.shape[-1])
                        ).reshape(true.shape)
                        true = true_scaled  # Use scaled ground truth for metrics
                    except Exception as e:
                        print(f"⚠️  Ground truth scaling for metrics failed: {e}, using unscaled")
                
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