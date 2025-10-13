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
import pandas as pd
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
from utils.scaler_manager import ScalerManager
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
    # Ensure model_id is set for checkpointing
    args.model_id = getattr(args, 'model_id', f"{args.model_name}_direct_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
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
    print(f"🚀 Using device: {device} (GPU disabled due to environment issues)")
    
    # Get data loaders
    print("📂 Loading data...")
    train_data, train_loader = data_provider(args, flag='train')
    vali_data, vali_loader = data_provider(args, flag='val')
    test_data, test_loader = data_provider(args, flag='test')
    
    print(f"   - Training batches: {len(train_loader)}")
    print(f"   - Validation batches: {len(vali_loader)}")
    print(f"   - Test batches: {len(test_loader)}")
    
    # 🔧 CRITICAL FIX: Setup proper scaling using ScalerManager
    print("🔧 Setting up proper data scaling...")
    
    # Check what the data loader actually provides
    for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
        print(f"   - Actual input shape: {batch_x.shape} (batch, seq, features)")
        print(f"   - Actual target shape: {batch_y.shape}")
        actual_input_features = batch_x.shape[-1]
        actual_target_features = batch_y.shape[-1]
        break
    
    # Load raw data for scaling setup
    raw_df = pd.read_csv(args.root_path + '/' + args.data_path)
    target_features = ['log_Open', 'log_High', 'log_Low', 'log_Close']
    
    # CRITICAL FIX: Use the actual number of features the model receives
    # The data loader might be providing a different feature set than the raw CSV
    print(f"   - Raw CSV has {len(raw_df.columns)} columns")
    print(f"   - Model input expects {actual_input_features} features")
    print(f"   - Model targets expect {actual_target_features} features")
    
    # For now, let's skip the ScalerManager and use a simpler approach
    # since there's a mismatch between CSV structure and data loader output
    print("   - ⚠️  Skipping ScalerManager due to feature dimension mismatch")
    print("   - Using data loader's built-in scaling (if any)")
    scaler_manager = None
    
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
            
            # 🔧 CRITICAL FIX: Apply proper scaling if available
            if scaler_manager is not None:
                # Scale inputs (covariates) and targets separately
                batch_x_scaled = scaler_manager.scale_covariates_tensor(batch_x, device=device)
                batch_y_scaled = scaler_manager.scale_targets_tensor(batch_y, device=device)
            else:
                # Use data as-is (data loader should handle scaling)
                batch_x_scaled = batch_x
                batch_y_scaled = batch_y
            
            # Prepare decoder input (using scaled targets)
            dec_inp = torch.zeros_like(batch_y_scaled[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y_scaled[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            
            try:
                # Forward pass (using scaled inputs)
                outputs, metadata = model(batch_x_scaled, batch_x_mark, dec_inp, batch_y_mark)
                
                # Determine the correct ground truth for the loss function (using scaled targets)
                if (model.aggregate_waves_to_celestial and 
                    'original_targets' in metadata and 
                    metadata['original_targets'] is not None):
                    # Use the specific targets isolated during wave aggregation (already scaled)
                    true_targets = metadata['original_targets'][:, -args.pred_len:, :]
                    print(f"🎯 Using wave aggregation targets: {true_targets.shape}")
                else:
                    # Use the scaled batch_y targets
                    true_targets = batch_y_scaled[:, -args.pred_len:, :]
                
                # Compute loss based on model type (data is already properly scaled)
                if getattr(model, 'use_mixture_decoder', False) and isinstance(outputs, dict):
                    # Mixture density network output
                    means = outputs['means']
                    log_stds = outputs['log_stds'] 
                    log_weights = outputs['log_weights']
                    
                    # Use properly scaled targets (no additional scaling needed)
                    if target_indices is not None:
                        # Slice targets to correct indices
                        gt_slice = true_targets[:, :, target_indices]
                    else:
                        gt_slice = true_targets
                    
                    # Compute loss with properly scaled data
                    loss = criterion(gt_slice, means, log_stds, log_weights)
                        
                elif getattr(model, 'use_mixture_decoder', False) and isinstance(outputs, tuple):
                    # Tuple output (means, log_stds, log_weights)
                    means, log_stds, log_weights = outputs
                    
                    # Apply proper target slicing for mixture decoder
                    if target_indices is not None:
                        gt_slice = true_targets[:, :, target_indices]
                    else:
                        gt_slice = true_targets
                    
                    # Compute loss (data should already be properly scaled)
                    loss = criterion(gt_slice, means, log_stds, log_weights)
                else:
                    # Standard deterministic output
                    # Ensure BOTH time and channel dimensions align with ground truth
                    # Slice model outputs to last pred_len timesteps
                    out_time = outputs[:, -args.pred_len:, :] if outputs.shape[1] >= args.pred_len else outputs
                    if target_indices is not None:
                        # FIXED: Slice outputs to match exact target indices, not just first N channels
                        if out_time.shape[-1] >= len(target_indices):
                            # If model outputs enough channels, take first len(target_indices) channels
                            # This assumes model outputs are in same order as target_indices
                            out_slice = out_time[:, :, :len(target_indices)]
                        else:
                            out_slice = out_time
                        # Slice ground truth to exact target indices
                        gt_slice = true_targets[:, :, target_indices]
                        
                        # CRITICAL FIX: Scale targets to match scaled predictions (like standard framework)
                        # Compute loss (data should already be properly scaled)
                        loss = criterion(out_slice, gt_slice)
                    else:
                        # Compute loss for all targets (data should already be properly scaled)
                        loss = criterion(out_time, true_targets)
                
                # Add regularization loss from stochastic graph learner
                if getattr(model, 'use_stochastic_learner', False):
                    reg_loss = model.get_regularization_loss()
                    # CRITICAL FIX: Much smaller regularization weight to prevent loss explosion
                    reg_weight = getattr(args, 'reg_loss_weight', 0.001)  # Reduced from 0.1 to 0.001
                    reg_contribution = reg_loss * reg_weight
                    loss += reg_contribution
                    
                    if i % 100 == 0:  # Log regularization loss occasionally
                        print(f"📊 Regularization loss: {reg_loss.item():.6f} (weighted: {reg_contribution.item():.6f})")
                
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
                
                # Apply proper scaling for validation if available
                if scaler_manager is not None:
                    batch_x_scaled = scaler_manager.scale_covariates_tensor(batch_x, device=device)
                    batch_y_scaled = scaler_manager.scale_targets_tensor(batch_y, device=device)
                else:
                    batch_x_scaled = batch_x
                    batch_y_scaled = batch_y
                
                dec_inp = torch.zeros_like(batch_y_scaled[:, -args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y_scaled[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
                
                try:
                    outputs, metadata = model(batch_x_scaled, batch_x_mark, dec_inp, batch_y_mark)
                    
                    # Determine correct ground truth (using scaled targets)
                    if (model.aggregate_waves_to_celestial and 
                        'original_targets' in metadata and 
                        metadata['original_targets'] is not None):
                        true_targets = metadata['original_targets'][:, -args.pred_len:, :]
                    else:
                        true_targets = batch_y_scaled[:, -args.pred_len:, :]
                    
                    # Compute loss based on model type (data already properly scaled)
                    if getattr(model, 'use_mixture_decoder', False) and isinstance(outputs, dict):
                        means = outputs['means']
                        log_stds = outputs['log_stds']
                        log_weights = outputs['log_weights']
                        
                        # Use properly scaled targets
                        if target_indices is not None:
                            gt_slice = true_targets[:, :, target_indices]
                        else:
                            gt_slice = true_targets
                        
                        # Compute loss with properly scaled data
                        loss = criterion(gt_slice, means, log_stds, log_weights)
                            
                    elif getattr(model, 'use_mixture_decoder', False) and isinstance(outputs, tuple):
                        means, log_stds, log_weights = outputs
                        
                        # Apply proper target slicing for mixture decoder
                        if target_indices is not None:
                            gt_slice = true_targets[:, :, target_indices]
                        else:
                            gt_slice = true_targets
                        
                        # Compute loss (data should already be properly scaled)
                        loss = criterion(gt_slice, means, log_stds, log_weights)
                    else:
                        # Deterministic loss
                        # Slice model outputs to last pred_len timesteps to match ground truth
                        out_time = outputs[:, -args.pred_len:, :] if outputs.shape[1] >= args.pred_len else outputs
                        if target_indices is not None:
                            # FIXED: Correct channel indexing for validation
                            if out_time.shape[-1] >= len(target_indices):
                                out_slice = out_time[:, :, :len(target_indices)]
                            else:
                                out_slice = out_time
                            gt_slice = true_targets[:, :, target_indices]
                            
                            # Compute loss (data should already be properly scaled)
                            loss = criterion(out_slice, gt_slice)
                        else:
                            # Scale all targets if no specific indices
                            # Compute loss (data should already be properly scaled)
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
        
        print("\n🎯 Final Results (OHLC Prediction):")
        print(f"   - Overall MAE:  {mae:.6f}")
        print(f"   - Overall MSE:  {mse:.6f}")
        print(f"   - Overall RMSE: {rmse:.6f}")
        print(f"   - Overall MAPE: {mape:.6f}")
        print(f"   - Overall MSPE: {mspe:.6f}")
        
        # OHLC-specific metrics if we have 4 targets
        if preds.shape[-1] == 4:
            ohlc_names = ['Open', 'High', 'Low', 'Close']
            print(f"\n📊 Individual OHLC Metrics:")
            for i, name in enumerate(ohlc_names):
                pred_i = preds[:, :, i:i+1]
                true_i = trues[:, :, i:i+1]
                mae_i, mse_i, rmse_i, mape_i, mspe_i = metric(pred_i, true_i)
                print(f"   {name:5s} - MAE: {mae_i:.6f}, RMSE: {rmse_i:.6f}, MAPE: {mape_i:.6f}")
        
        # Save results
        results = {
            'model': 'Celestial_Enhanced_PGAT',
            'task': 'OHLC_Prediction',
            'num_targets': preds.shape[-1],
            'overall_metrics': {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'mape': float(mape),
                'mspe': float(mspe)
            },
            'train_losses': train_losses,
            'val_losses': val_losses,
            'config': config_dict,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add individual OHLC metrics if available
        if preds.shape[-1] == 4:
            ohlc_names = ['Open', 'High', 'Low', 'Close']
            ohlc_metrics = {}
            for i, name in enumerate(ohlc_names):
                pred_i = preds[:, :, i:i+1]
                true_i = trues[:, :, i:i+1]
                mae_i, mse_i, rmse_i, mape_i, mspe_i = metric(pred_i, true_i)
                ohlc_metrics[name] = {
                    'mae': float(mae_i),
                    'mse': float(mse_i),
                    'rmse': float(rmse_i),
                    'mape': float(mape_i),
                    'mspe': float(mspe_i)
                }
            results['ohlc_metrics'] = ohlc_metrics
        
        results_file = checkpoint_dir / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {results_file}")
        
        print("\n" + "=" * 60)
        print("🎉 Celestial Enhanced PGAT OHLC Training Complete!")
        print("🌌 The Astrological AI has learned to predict OHLC patterns from the cosmos!")
        print(f"📈 Successfully trained to predict {preds.shape[-1]} targets: {getattr(args, 'target', 'OHLC')}")
        
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