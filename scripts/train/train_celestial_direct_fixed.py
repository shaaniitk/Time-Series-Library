#!/usr/bin/env python3
"""
Direct Training Script for Celestial Enhanced PGAT - FIXED VERSION
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

def scale_targets_properly(targets, target_indices, data_scaler, device):
    """
    Properly scale target values using the correct columns from the main scaler
    
    Args:
        targets: [batch, seq_len, n_targets] Target tensor
        target_indices: List of column indices for targets
        data_scaler: The fitted scaler from training data
        device: PyTorch device
    
    Returns:
        Scaled targets tensor
    """
    try:
        targets_np = targets.cpu().numpy()
        batch_size, seq_len, n_targets = targets_np.shape
        
        # Create full feature array with zeros, then fill target positions
        full_features = np.zeros((batch_size * seq_len, data_scaler.n_features_in_))
        full_features[:, target_indices] = targets_np.reshape(-1, n_targets)
        
        # Scale the full array and extract only target columns
        scaled_full = data_scaler.transform(full_features)
        targets_scaled_np = scaled_full[:, target_indices].reshape(targets_np.shape)
        
        return torch.from_numpy(targets_scaled_np).float().to(device)
        
    except Exception as e:
        print(f"⚠️  Target scaling failed: {e}, using unscaled targets")
        return targets

def train_celestial_pgat():
    """Direct training function"""
    print("🌌 Starting Direct Celestial Enhanced PGAT Training (FIXED VERSION)")
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
    
    # FIXED: Respect config target_wave_indices instead of hardcoding
    # NOTE: target_wave_indices specifies which of the 118 input waves to extract as targets
    # For OHLC prediction: [0, 1, 2, 3] = [log_Open, log_High, log_Low, log_Close]
    args.target_wave_indices = getattr(args, 'target_wave_indices', [0, 1, 2, 3])  # Default to OHLC
    
    print(f"📊 Configuration loaded:")
    print(f"   - Model: {getattr(args, 'model_name', getattr(args, 'model', 'Celestial_Enhanced_PGAT'))}")
    print(f"   - Data: {getattr(args, 'data_name', getattr(args, 'data', 'custom'))}")
    print(f"   - Sequence length: {getattr(args, 'seq_len', 'Unknown')}")
    print(f"   - Prediction length: {getattr(args, 'pred_len', 'Unknown')}")
    print(f"   - Model dimension: {getattr(args, 'd_model', 'Unknown')}")
    print(f"   - Target columns: {getattr(args, 'target', 'Not specified')}")
    print(f"   - Target wave indices: {getattr(args, 'target_wave_indices', [])} (OHLC: 0=Open, 1=High, 2=Low, 3=Close)")
    print(f"   - Number of outputs: {getattr(args, 'c_out', 'Unknown')}")
    print(f"   - Wave aggregation enabled: {getattr(args, 'aggregate_waves_to_celestial', False)}")
    
    # Setup device
    args.use_gpu = False  # Force CPU usage
    device = torch.device('cpu')
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
                target_indices = list(range(getattr(args, 'c_out', 4)))  # Default to 4 for OHLC
                
            print(f"🎯 Target indices resolved: {target_indices}")
            print(f"🎯 Target names: {target_names}")
            
            # Validate OHLC configuration
            if len(target_indices) != args.c_out:
                print(f"⚠️  Warning: Number of target indices ({len(target_indices)}) != c_out ({args.c_out})")
                print(f"   Adjusting c_out to match target indices")
                args.c_out = len(target_indices)
                
    except Exception as e:
        print(f"⚠️  Failed to auto-detect target indices: {e}. Using first c_out features.")
        target_indices = list(range(getattr(args, 'c_out', 4)))  # Default to 4 for OHLC

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
                    
                    # FIXED: Apply proper target slicing and scaling for mixture decoder
                    if target_indices is not None:
                        gt_slice = true_targets[:, :, target_indices]
                    else:
                        gt_slice = true_targets
                    
                    # FIXED: Scale targets using proper column mapping
                    if hasattr(train_data, 'scaler') and train_data.scaler is not None and target_indices is not None:
                        gt_slice_scaled = scale_targets_properly(gt_slice, target_indices, train_data.scaler, device)
                        loss = criterion((means, log_stds, log_weights), gt_slice_scaled)
                    else:
                        loss = criterion((means, log_stds, log_weights), gt_slice)
                        
                elif getattr(model, 'use_mixture_decoder', False) and isinstance(outputs, tuple):
                    # Tuple output (means, log_stds, log_weights)
                    means, log_stds, log_weights = outputs
                    
                    # FIXED: Apply proper target slicing and scaling for mixture decoder
                    if target_indices is not None:
                        gt_slice = true_targets[:, :, target_indices]
                    else:
                        gt_slice = true_targets
                    
                    # FIXED: Scale targets using proper column mapping
                    if hasattr(train_data, 'scaler') and train_data.scaler is not None and target_indices is not None:
                        gt_slice_scaled = scale_targets_properly(gt_slice, target_indices, train_data.scaler, device)
                        loss = criterion((means, log_stds, log_weights), gt_slice_scaled)
                    else:
                        loss = criterion((means, log_stds, log_weights), gt_slice)
                else:
                    # Standard deterministic output
                    out_time = outputs[:, -args.pred_len:, :] if outputs.shape[1] >= args.pred_len else outputs
                    if target_indices is not None:
                        # FIXED: Correct channel indexing
                        if out_time.shape[-1] >= len(target_indices):
                            out_slice = out_time[:, :, :len(target_indices)]
                        else:
                            out_slice = out_time
                        gt_slice = true_targets[:, :, target_indices]
                        
                        # FIXED: Scale targets using proper column mapping
                        if hasattr(train_data, 'scaler') and train_data.scaler is not None:
                            gt_slice_scaled = scale_targets_properly(gt_slice, target_indices, train_data.scaler, device)
                            loss = criterion(out_slice, gt_slice_scaled)
                        else:
                            loss = criterion(out_slice, gt_slice)
                    else:
                        # Scale all targets if no specific indices
                        if hasattr(train_data, 'scaler') and train_data.scaler is not None:
                            gt_scaled = scale_targets_properly(true_targets, list(range(true_targets.shape[-1])), train_data.scaler, device)
                            loss = criterion(out_time, gt_scaled)
                        else:
                            loss = criterion(out_time, true_targets)
                
                # Add regularization loss from stochastic graph learner
                if getattr(model, 'use_stochastic_learner', False):
                    reg_loss = model.get_regularization_loss()
                    reg_weight = getattr(args, 'reg_loss_weight', 0.1)
                    loss += reg_loss * reg_weight
                
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
                    
                    # Compute loss based on model type (same as training)
                    if getattr(model, 'use_mixture_decoder', False) and isinstance(outputs, dict):
                        means = outputs['means']
                        log_stds = outputs['log_stds']
                        log_weights = outputs['log_weights']
                        
                        if target_indices is not None:
                            gt_slice = true_targets[:, :, target_indices]
                        else:
                            gt_slice = true_targets
                        
                        if hasattr(vali_data, 'scaler') and vali_data.scaler is not None and target_indices is not None:
                            gt_slice_scaled = scale_targets_properly(gt_slice, target_indices, vali_data.scaler, device)
                            loss = criterion((means, log_stds, log_weights), gt_slice_scaled)
                        else:
                            loss = criterion((means, log_stds, log_weights), gt_slice)
                            
                    elif getattr(model, 'use_mixture_decoder', False) and isinstance(outputs, tuple):
                        means, log_stds, log_weights = outputs
                        
                        if target_indices is not None:
                            gt_slice = true_targets[:, :, target_indices]
                        else:
                            gt_slice = true_targets
                        
                        if hasattr(vali_data, 'scaler') and vali_data.scaler is not None and target_indices is not None:
                            gt_slice_scaled = scale_targets_properly(gt_slice, target_indices, vali_data.scaler, device)
                            loss = criterion((means, log_stds, log_weights), gt_slice_scaled)
                        else:
                            loss = criterion((means, log_stds, log_weights), gt_slice)
                    else:
                        # Deterministic loss
                        out_time = outputs[:, -args.pred_len:, :] if outputs.shape[1] >= args.pred_len else outputs
                        if target_indices is not None:
                            if out_time.shape[-1] >= len(target_indices):
                                out_slice = out_time[:, :, :len(target_indices)]
                            else:
                                out_slice = out_time
                            gt_slice = true_targets[:, :, target_indices]
                            
                            if hasattr(vali_data, 'scaler') and vali_data.scaler is not None:
                                gt_slice_scaled = scale_targets_properly(gt_slice, target_indices, vali_data.scaler, device)
                                loss = criterion(out_slice, gt_slice_scaled)
                            else:
                                loss = criterion(out_slice, gt_slice)
                        else:
                            if hasattr(vali_data, 'scaler') and vali_data.scaler is not None:
                                gt_scaled = scale_targets_properly(true_targets, list(range(true_targets.shape[-1])), vali_data.scaler, device)
                                loss = criterion(out_time, gt_scaled)
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
        # FIXED: Device-safe checkpoint loading
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("✅ Best model loaded")
    
    print("\n🎉 Celestial Enhanced PGAT OHLC Training Complete!")
    print("🌌 The Astrological AI has learned to predict OHLC patterns from the cosmos!")
    
    return True

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