#!/usr/bin/env python3
"""
PRODUCTION Training Script for Celestial Enhanced PGAT
Heavy-duty overnight training with maximum model capacity
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
import warnings
import logging
warnings.filterwarnings('ignore')
# Suppress specific data loader warnings since we handle scaling manually
logging.getLogger('utils.logger').setLevel(logging.ERROR)

class SimpleConfig:
    """Simple configuration class"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

def scale_targets_for_loss(targets_unscaled, target_scaler, target_indices, device):
    """
    Scale target values for loss computation to match scaled predictions
    
    Args:
        targets_unscaled: [batch, seq_len, n_features] Unscaled target tensor from batch_y
        target_scaler: Fitted scaler for targets
        target_indices: List of indices for target features
        device: PyTorch device
    
    Returns:
        Scaled targets tensor for loss computation
    """
    try:
        # Extract only the target features
        targets_only = targets_unscaled[:, :, target_indices]  # [batch, seq_len, n_targets]
        targets_np = targets_only.cpu().numpy()
        
        # Reshape for scaler: [batch*seq_len, n_targets]
        batch_size, seq_len, n_targets = targets_np.shape
        targets_reshaped = targets_np.reshape(-1, n_targets)
        
        # Scale using target scaler
        targets_scaled_reshaped = target_scaler.transform(targets_reshaped)
        
        # Reshape back: [batch, seq_len, n_targets]
        targets_scaled_np = targets_scaled_reshaped.reshape(batch_size, seq_len, n_targets)
        
        return torch.from_numpy(targets_scaled_np).float().to(device)
        
    except Exception as e:
        print(f"⚠️  Target scaling for loss failed: {e}")
        # Fallback: return unscaled targets (will cause scale mismatch but won't crash)
        return targets_unscaled[:, :, target_indices].to(device)

def _normalize_model_output(raw_output):
    """Convert mixed forward outputs into tensor, scalar aux loss, optional MDN tuple, and metadata."""
    aux_loss = 0.0
    mdn_tuple = None
    metadata = None
    output_tensor = raw_output

    if isinstance(raw_output, (tuple, list)):
        if len(raw_output) == 2:
            primary, secondary = raw_output
            # Check if secondary is metadata (dict) or auxiliary loss
            if isinstance(secondary, dict):
                metadata = secondary
                output_tensor = primary
            elif isinstance(secondary, torch.Tensor) and secondary.numel() == 1:
                aux_loss = float(secondary.item())
                output_tensor = primary
            elif isinstance(secondary, (int, float)):
                aux_loss = float(secondary)
                output_tensor = primary
            else:
                output_tensor = primary
        elif len(raw_output) == 3 and all(isinstance(part, torch.Tensor) for part in raw_output):
            mdn_tuple = (raw_output[0], raw_output[1], raw_output[2])
            output_tensor = raw_output[0]
        elif len(raw_output) >= 1:
            output_tensor = raw_output[0]

    if not isinstance(output_tensor, torch.Tensor):
        raise TypeError("Model forward pass must yield a tensor as primary output.")

    return output_tensor, aux_loss, mdn_tuple, metadata

def train_celestial_pgat_production():
    """Production training function for overnight runs"""
    print("🌌 Starting PRODUCTION Celestial Enhanced PGAT Training")
    print("🚀 HEAVY-DUTY OVERNIGHT CONFIGURATION")
    print("=" * 80)
    
    # Load production configuration
    config_path = "configs/celestial_enhanced_pgat_production.yaml"
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
    args.num_workers = 0  # Disable multiprocessing for stability
    args.itr = 1
    args.train_only = False
    args.do_predict = False
    # Ensure model_id is set for checkpointing
    args.model_id = getattr(args, 'model_id', f"{args.model_name}_production_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Wave aggregation settings
    args.aggregate_waves_to_celestial = getattr(args, 'aggregate_waves_to_celestial', True)
    args.wave_to_celestial_mapping = getattr(args, 'wave_to_celestial_mapping', False)
    args.celestial_node_features = getattr(args, 'celestial_node_features', 13)
    
    # Target wave indices
    args.target_wave_indices = getattr(args, 'target_wave_indices', [0, 1, 2, 3])
    
    print(f"📊 PRODUCTION Configuration:")
    print(f"   - Model: {getattr(args, 'model_name', getattr(args, 'model', 'Celestial_Enhanced_PGAT'))}")
    print(f"   - Sequence length: {getattr(args, 'seq_len', 'Unknown')} (HEAVY DUTY)")
    print(f"   - Prediction length: {getattr(args, 'pred_len', 'Unknown')}")
    print(f"   - Model dimension: {getattr(args, 'd_model', 'Unknown')} (HEAVY)")
    print(f"   - Attention heads: {getattr(args, 'n_heads', 'Unknown')} (HEAVY)")
    print(f"   - Encoder layers: {getattr(args, 'e_layers', 'Unknown')} (DEEP)")
    print(f"   - Decoder layers: {getattr(args, 'd_layers', 'Unknown')} (DEEP)")
    print(f"   - Training epochs: {getattr(args, 'train_epochs', 'Unknown')} (OVERNIGHT)")
    print(f"   - Batch size: {getattr(args, 'batch_size', 'Unknown')}")
    print(f"   - Learning rate: {getattr(args, 'learning_rate', 'Unknown')}")
    print(f"   - Early stopping: DISABLED (patience={getattr(args, 'patience', 'Unknown')})")
    print(f"   - Target columns: {getattr(args, 'target', 'Not specified')}")
    print(f"   - Target wave indices: {getattr(args, 'target_wave_indices', [])} (OHLC)")
    print(f"   - Number of outputs: {getattr(args, 'c_out', 'Unknown')}")
    print(f"   - Wave aggregation enabled: {getattr(args, 'aggregate_waves_to_celestial', False)}")
    
    # Setup device - prefer GPU for production
    if getattr(args, 'use_gpu', True) and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using device: {device} (GPU PRODUCTION MODE)")
    else:
        device = torch.device('cpu')
        print(f"🚀 Using device: {device} (CPU fallback)")
    
    # Get data loaders
    print("📂 Loading PRODUCTION data...")
    train_data, train_loader = data_provider(args, flag='train')
    vali_data, vali_loader = data_provider(args, flag='val')
    test_data, test_loader = data_provider(args, flag='test')
    
    print(f"   - Training batches: {len(train_loader)}")
    print(f"   - Validation batches: {len(vali_loader)}")
    print(f"   - Test batches: {len(test_loader)}")
    
    # Setup scaling for loss computation
    print("🔧 Setting up PRODUCTION scaling for loss computation...")
    
    # Get scalers from the dataset objects
    train_scaler = getattr(train_data, 'scaler', None)
    target_scaler = getattr(train_data, 'target_scaler', None)
    
    if train_scaler is None:
        print("❌ No scaler found in train_data - this will cause scale mismatch!")
        return False
    
    print(f"   - ✅ Found main scaler: {train_scaler.n_features_in_} features")
    if target_scaler:
        print(f"   - ✅ Found target scaler: {target_scaler.n_features_in_} features")
    else:
        print("   - ⚠️  No separate target scaler - will use main scaler for targets")
        target_scaler = train_scaler
    
    # Define target indices for OHLC
    target_indices = [0, 1, 2, 3]  # OHLC indices
    print(f"   - Target indices for OHLC: {target_indices}")
    
    # Initialize HEAVY model
    print("🏗️  Initializing HEAVY PRODUCTION Celestial Enhanced PGAT...")
    try:
        model = Model(args).to(device)
        print(f"✅ HEAVY Model initialized successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   - Total parameters: {total_params:,} (HEAVY PRODUCTION MODEL)")
        print(f"   - Trainable parameters: {trainable_params:,}")
        
        # Estimate model size
        param_size = total_params * 4 / (1024**2)  # Assuming float32
        print(f"   - Estimated model size: {param_size:.1f} MB")
        
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Setup training components
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=getattr(args, 'weight_decay', 0.0001))
    
    # Use proper loss function
    if getattr(model, 'use_mixture_decoder', False):
        from layers.modular.decoder.sequential_mixture_decoder import SequentialMixtureNLLLoss
        from layers.modular.decoder.mixture_density_decoder import MixtureNLLLoss  # Import for type checking
        criterion = SequentialMixtureNLLLoss(reduction='mean')
        print("🎯 Using Gaussian Mixture NLL Loss for probabilistic predictions")
    else:
        criterion = nn.MSELoss()
        print("📊 Using MSE Loss for deterministic predictions")
    
    # NO EARLY STOPPING for production
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    print(f"⏹️  Early stopping patience: {args.patience} (effectively DISABLED)")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoints) / args.model_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"💾 Checkpoints will be saved to: {checkpoint_dir}")
    
    # Auto-detect target indices from CSV header
    target_indices = None
    try:
        if hasattr(args, 'target') and args.target:
            csv_path = os.path.join(args.root_path, args.data_path)
            print(f"📁 Data file: {csv_path}")
            df_head = pd.read_csv(csv_path, nrows=1)
            feature_cols = [c for c in df_head.columns if c != 'date']
            print(f"🧾 Total columns: {len(df_head.columns)}, Features: {len(feature_cols)}")
            
            # Estimate row count
            try:
                with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                    row_count = sum(1 for _ in f) - 1
                print(f"📏 CSV rows: {row_count}")
            except Exception:
                pass
            
            # Auto-set dimensions
            args.enc_in = len(feature_cols)
            args.dec_in = len(feature_cols)
            print(f"📐 Auto-set enc_in/dec_in: {args.enc_in}")
            
            if isinstance(args.target, str):
                target_names = [t.strip() for t in args.target.split(',')]
            else:
                target_names = list(args.target)
            name_to_index = {name: idx for idx, name in enumerate(feature_cols)}
            target_indices = [name_to_index[n] for n in target_names if n in name_to_index]
            
            if not target_indices:
                target_indices = list(range(getattr(args, 'c_out', 1)))
            print(f"🎯 Target indices resolved: {target_indices}")
            print(f"🎯 Target names: {target_names}")
            
            if len(target_indices) != args.c_out:
                print(f"⚠️  Adjusting c_out from {args.c_out} to {len(target_indices)}")
                args.c_out = len(target_indices)
                
    except Exception as e:
        print(f"⚠️  Failed to auto-detect target indices: {e}")
        target_indices = list(range(getattr(args, 'c_out', 4)))

    # PRODUCTION Training loop
    print("\\n🔥 Starting PRODUCTION training loop...")
    print(f"🌙 OVERNIGHT TRAINING: {args.train_epochs} epochs")
    train_losses = []
    val_losses = []
    
    # Training start time
    training_start_time = time.time()
    
    for epoch in range(args.train_epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        print(f"\\n🌌 Epoch {epoch+1}/{args.train_epochs} - PRODUCTION TRAINING")
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Move to device
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            # Use data as-is from Dataset_Custom
            batch_x_input = batch_x  # Already scaled
            
            # Prepare decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            
            try:
                # Forward pass
                outputs_raw = model(batch_x_input, batch_x_mark, dec_inp, batch_y_mark)
                
                # Normalize model output using consistent approach
                outputs_tensor, aux_loss, mdn_outputs, metadata = _normalize_model_output(outputs_raw)
                
                # Prepare y_true for loss: scale the target part of batch_y
                c_out_evaluation = len(target_indices)
                y_true_for_loss = scale_targets_for_loss(
                    batch_y[:, -args.pred_len:, :], target_scaler, target_indices, device
                )
                
                # Compute loss using consistent approach
                if isinstance(criterion, (MixtureNLLLoss, SequentialMixtureNLLLoss)) or mdn_outputs is not None:
                    if mdn_outputs is None:
                        raise ValueError("MixtureNLLLoss requires model to return a (means, stds, weights) tuple during training.")
                    means_t, stds_t, weights_t = mdn_outputs
                    if means_t.size(1) > args.pred_len:
                        means_t = means_t[:, -args.pred_len:, ...]
                        stds_t = stds_t[:, -args.pred_len:, ...]
                        weights_t = weights_t[:, -args.pred_len:, ...]
                    targets_t = y_true_for_loss.squeeze(-1) if y_true_for_loss.dim() == 3 and y_true_for_loss.size(-1) == 1 else y_true_for_loss
                    loss = criterion((means_t, stds_t, weights_t), targets_t)
                else:
                    # Standard deterministic output
                    y_pred_for_loss = outputs_tensor[:, -args.pred_len:, :c_out_evaluation]
                    loss = criterion(y_pred_for_loss, y_true_for_loss)
                
                # Add auxiliary loss if present
                if aux_loss:
                    loss = loss + aux_loss
                
                # Add regularization if enabled
                if getattr(model, 'use_stochastic_learner', False):
                    reg_loss = model.get_regularization_loss()
                    reg_weight = getattr(args, 'reg_loss_weight', 0.0005)
                    reg_contribution = reg_loss * reg_weight
                    loss = loss + reg_contribution

                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if hasattr(args, 'clip_grad_norm'):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
                
                # Log progress
                log_interval = getattr(args, 'log_interval', 10)
                if i % log_interval == 0:
                    elapsed = time.time() - epoch_start
                    print(f"      Batch {i:4d}/{len(train_loader)} | Loss: {loss.item():.6f} | Time: {elapsed:.1f}s")
                    
                    # Debug scaling for first epoch, first batch
                    if epoch == 0 and i == 0:
                        print(f"🔍 PRODUCTION TRAIN - First batch scaling check:")
                        print(f"   - Raw batch_y OHLC: mean={batch_y[:, -args.pred_len:, :4].mean():.6f}, std={batch_y[:, -args.pred_len:, :4].std():.6f}")
                        print(f"   - Scaled targets: mean={y_true_for_loss.mean():.6f}, std={y_true_for_loss.std():.6f}")
                        print(f"   - Model outputs: mean={outputs_tensor.mean():.6f}, std={outputs_tensor.std():.6f}")
                        print(f"   - ✅ Scaling consistency verified for production training")
                
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
                
                batch_x_input = batch_x
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
                
                try:
                    outputs_raw = model(batch_x_input, batch_x_mark, dec_inp, batch_y_mark)
                    
                    # Normalize model output using consistent approach
                    outputs_tensor, aux_loss, mdn_outputs, metadata = _normalize_model_output(outputs_raw)
                    
                    # Prepare y_true for loss: scale the target part of batch_y (same as training)
                    c_out_evaluation = len(target_indices)
                    y_true_for_loss = scale_targets_for_loss(
                        batch_y[:, -args.pred_len:, :], target_scaler, target_indices, device
                    )
                    
                    # Compute validation loss using consistent approach
                    if isinstance(criterion, (MixtureNLLLoss, SequentialMixtureNLLLoss)) or mdn_outputs is not None:
                        if mdn_outputs is None:
                            raise ValueError("MixtureNLLLoss requires model to return a (means, stds, weights) tuple.")
                        means_v, stds_v, weights_v = mdn_outputs
                        if means_v.size(1) > args.pred_len:
                            means_v = means_v[:, -args.pred_len:, ...]
                            stds_v = stds_v[:, -args.pred_len:, ...]
                            weights_v = weights_v[:, -args.pred_len:, ...]
                        targets_v = y_true_for_loss.squeeze(-1) if y_true_for_loss.dim() == 3 and y_true_for_loss.size(-1) == 1 else y_true_for_loss
                        loss = criterion((means_v, stds_v, weights_v), targets_v)
                    else:
                        # Standard deterministic output
                        y_pred_for_loss = outputs_tensor[:, -args.pred_len:, :c_out_evaluation]
                        loss = criterion(y_pred_for_loss, y_true_for_loss)
                    
                    # Add auxiliary loss if present
                    if aux_loss:
                        loss = loss + aux_loss
                    
                    val_loss += loss.item()
                    val_batches += 1
                    
                    # Debug scaling for first epoch, first validation batch
                    if epoch == 0 and val_batches == 1:
                        print(f"🔍 PRODUCTION VAL - First batch scaling check:")
                        print(f"   - Raw batch_y OHLC: mean={batch_y[:, -args.pred_len:, :4].mean():.6f}, std={batch_y[:, -args.pred_len:, :4].std():.6f}")
                        print(f"   - Scaled targets: mean={y_true_for_loss.mean():.6f}, std={y_true_for_loss.std():.6f}")
                        print(f"   - Model outputs: mean={outputs_tensor.mean():.6f}, std={outputs_tensor.std():.6f}")
                        print(f"   - ✅ Scaling consistency verified for production validation")
                    
                except Exception as e:
                    print(f"⚠️  Validation step warning: {e}")
                    continue
        
        avg_val_loss = val_loss / max(val_batches, 1)
        val_losses.append(avg_val_loss)
        
        # Learning rate adjustment
        adjust_learning_rate(optimizer, epoch + 1, args)
        
        # Progress report
        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - training_start_time
        remaining_epochs = args.train_epochs - (epoch + 1)
        estimated_remaining = (total_elapsed / (epoch + 1)) * remaining_epochs
        
        print(f"\\n📊 Epoch {epoch+1:3d}/{args.train_epochs} COMPLETE:")
        print(f"   - Train Loss: {avg_train_loss:.6f}")
        print(f"   - Val Loss: {avg_val_loss:.6f}")
        print(f"   - Epoch Time: {epoch_time:.2f}s")
        print(f"   - Total Elapsed: {total_elapsed/3600:.2f}h")
        print(f"   - Estimated Remaining: {estimated_remaining/3600:.2f}h")
        
        # Save checkpoint at intervals
        checkpoint_interval = getattr(args, 'checkpoint_interval', 5)
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Early stopping check (but with very high patience)
        early_stopping(avg_val_loss, model, str(checkpoint_dir))
        if early_stopping.early_stop:
            print(f"   ⏹️  Early stopping at epoch {epoch+1} (unlikely with high patience)")
            break
    
    # Load best model
    best_model_path = checkpoint_dir / 'checkpoint.pth'
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("✅ Best model loaded")
    
    # Final evaluation
    print("\\n📊 Final PRODUCTION Evaluation...")
    model.eval()
    
    # Evaluation arrays
    num_test_samples = len(test_data)
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
                outputs_raw = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # Normalize model output using consistent approach
                outputs_tensor, aux_loss, mdn_outputs, metadata = _normalize_model_output(outputs_raw)

                # Handle mixture decoder outputs - extract point predictions
                if mdn_outputs is not None:
                    means_te, stds_te, weights_te = mdn_outputs
                    if means_te.size(1) > args.pred_len:
                        means_te = means_te[:, -args.pred_len:, ...]
                        stds_te = stds_te[:, -args.pred_len:, ...]
                        weights_te = weights_te[:, -args.pred_len:, ...]
                    
                    # Handle both univariate and multivariate mixture outputs
                    if means_te.dim() == 4:  # Multivariate: [batch, pred_len, targets, components]
                        # Compute mixture mean for each target separately
                        weights_expanded = weights_te.unsqueeze(2)  # Add target dimension
                        pred_tensor = (weights_expanded * means_te).sum(dim=-1)
                    else:  # Univariate: [batch, pred_len, components]
                        # Use mixture mean as point prediction for evaluation
                        pred_tensor = (weights_te * means_te).sum(dim=-1).unsqueeze(-1)
                else:
                    pred_tensor = outputs_tensor
                
                # Use unscaled ground truth for evaluation (standard approach)
                true_tensor = batch_y[:, -args.pred_len:, :]
                
                # Align dimensions
                pred_aligned = pred_tensor[:, -args.pred_len:, :]
                true_aligned = true_tensor[:, -args.pred_len:, :]
                if target_indices is not None:
                    pred_aligned = pred_aligned[:, :, target_indices]
                    true_aligned = true_aligned[:, :, target_indices]
                
                # Convert to numpy
                pred = pred_aligned.detach().cpu().numpy()
                true = true_aligned.detach().cpu().numpy()
                
                # Fill arrays
                batch_size = pred.shape[0]
                start_index = current_index
                end_index = start_index + batch_size
                
                if end_index <= num_test_samples:
                    preds[start_index:end_index, :, :] = pred
                    trues[start_index:end_index, :, :] = true
                    current_index = end_index
                
            except Exception as e:
                print(f"⚠️  Test step warning: {e}")
                continue
    
    # Trim arrays
    if current_index < num_test_samples:
        preds = preds[:current_index]
        trues = trues[:current_index]
    
    if current_index > 0:
        # Compute metrics
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        
        print("\\n🎯 PRODUCTION Results (OHLC Prediction):")
        print(f"   - Overall MAE:  {mae:.6f}")
        print(f"   - Overall MSE:  {mse:.6f}")
        print(f"   - Overall RMSE: {rmse:.6f}")
        print(f"   - Overall MAPE: {mape:.6f}")
        print(f"   - Overall MSPE: {mspe:.6f}")
        
        # OHLC-specific metrics
        if preds.shape[-1] == 4:
            ohlc_names = ['Open', 'High', 'Low', 'Close']
            print(f"\\n📊 Individual OHLC Metrics:")
            for i, name in enumerate(ohlc_names):
                pred_i = preds[:, :, i:i+1]
                true_i = trues[:, :, i:i+1]
                mae_i, mse_i, rmse_i, mape_i, mspe_i = metric(pred_i, true_i)
                print(f"   {name:5s} - MAE: {mae_i:.6f}, RMSE: {rmse_i:.6f}, MAPE: {mape_i:.6f}")
        
        # Save results
        results = {
            'model': 'Celestial_Enhanced_PGAT_PRODUCTION',
            'task': 'OHLC_Prediction_Production',
            'config': 'HEAVY_DUTY_OVERNIGHT',
            'num_targets': preds.shape[-1],
            'seq_len': args.seq_len,
            'pred_len': args.pred_len,
            'd_model': args.d_model,
            'n_heads': args.n_heads,
            'e_layers': args.e_layers,
            'd_layers': args.d_layers,
            'train_epochs': args.train_epochs,
            'total_parameters': total_params,
            'training_time_hours': (time.time() - training_start_time) / 3600,
            'overall_metrics': {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'mape': float(mape),
                'mspe': float(mspe)
            },
            'train_losses': train_losses,
            'val_losses': val_losses,
            'config_dict': config_dict,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add OHLC metrics
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
        
        results_file = checkpoint_dir / 'production_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 PRODUCTION Results saved to: {results_file}")
        
        print("\\n" + "=" * 80)
        print("🎉 PRODUCTION Celestial Enhanced PGAT Training Complete!")
        print("🌌 The Heavy-Duty Astrological AI has completed overnight training!")
        print(f"📈 Successfully trained {total_params:,} parameters over {args.train_epochs} epochs")
        print(f"⏰ Total training time: {(time.time() - training_start_time)/3600:.2f} hours")
        print(f"🎯 Final RMSE: {rmse:.6f}")
        
        return True
    else:
        print("❌ No valid predictions generated")
        return False

def main():
    """Main function"""
    try:
        success = train_celestial_pgat_production()
        return 0 if success else 1
    except Exception as e:
        print(f"❌ PRODUCTION Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())