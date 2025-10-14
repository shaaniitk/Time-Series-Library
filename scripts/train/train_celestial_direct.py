#!/usr/bin/env python3
"""
Direct Training Script for Celestial Enhanced PGAT
Simplified approach without complex experiment framework
"""

import os
import sys
import io
from contextlib import nullcontext
from typing import Any, Dict, Sequence

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
import time
from datetime import datetime
import json
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from data_provider.data_factory import data_provider
from models.Celestial_Enhanced_PGAT import Model
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from layers.modular.decoder.mixture_density_decoder import MixtureNLLLoss  # Import for type checking
from layers.modular.decoder.sequential_mixture_decoder import SequentialMixtureNLLLoss  # Import for type checking
import warnings
import logging
warnings.filterwarnings('ignore')
# Suppress specific data loader warnings since we handle scaling manually
logging.getLogger('utils.logger').setLevel(logging.ERROR)

class SimpleConfig:
    """Simple configuration class."""

    def __init__(self, config_dict: Dict[str, Any]) -> None:
        for key, value in config_dict.items():
            setattr(self, key, value)


def _parse_gpu_ids(devices: str) -> Sequence[int]:
    """Parse a comma-delimited device string into numeric GPU identifiers."""

    return [int(token.strip()) for token in devices.split(',') if token.strip().isdigit()]


def select_device(args: SimpleConfig) -> torch.device:
    """Resolve the training device based on configuration flags and availability."""

    requested_gpu = int(getattr(args, "gpu", 0))
    use_gpu = bool(getattr(args, "use_gpu", True))
    multi_gpu = bool(getattr(args, "use_multi_gpu", False))

    if use_gpu and torch.cuda.is_available():
        available = torch.cuda.device_count()
        if multi_gpu:
            device_ids = _parse_gpu_ids(str(getattr(args, "devices", "0")))
            if not device_ids:
                device_ids = list(range(available)) if available > 0 else [0]
            primary = device_ids[0]
            torch.cuda.set_device(primary)
            args.gpu_ids = device_ids  # type: ignore[attr-defined]
            device = torch.device(f"cuda:{primary}")
        else:
            if available == 0:
                device = torch.device("cuda:0")
            else:
                primary = requested_gpu if requested_gpu < available else 0
                torch.cuda.set_device(primary)
                device = torch.device(f"cuda:{primary}")
        args.use_gpu = True
        return device

    if use_gpu and not torch.cuda.is_available():
        print("⚠️  GPU requested but CUDA is not available; falling back to CPU.")
    args.use_gpu = False
    return torch.device("cpu")


def should_enable_amp(args: SimpleConfig, device: torch.device) -> bool:
    """Determine whether mixed precision should be used for training."""

    return bool(getattr(args, "mixed_precision", False) and device.type == "cuda")

def scale_targets_for_loss(
    targets_unscaled: Tensor,
    target_scaler: Any,
    target_indices: Sequence[int],
    device: torch.device,
) -> Tensor:
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

def train_celestial_pgat() -> bool:
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
    args.num_workers = 0  # Disable multiprocessing to avoid timeout issues
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
    device = select_device(args)
    args.device = str(device)
    amp_enabled = should_enable_amp(args, device)
    if device.type == "cuda":
        cuda_name = torch.cuda.get_device_name(device)
        print(f"🚀 Using device: {device} ({cuda_name})")
        if amp_enabled:
            print("   - Mixed precision: enabled")
    else:
        print(f"🚀 Using device: {device} (CUDA unavailable)")
    
    # Get data loaders
    print("📂 Loading data...")
    train_data, train_loader = data_provider(args, flag='train')
    vali_data, vali_loader = data_provider(args, flag='val')
    test_data, test_loader = data_provider(args, flag='test')
    
    print(f"   - Training batches: {len(train_loader)}")
    print(f"   - Validation batches: {len(vali_loader)}")
    print(f"   - Test batches: {len(test_loader)}")
    
    # 🔧 SCALING FIX: Dataset_Custom handles scaling, but we need scalers for loss computation
    # data_x = scaled features, data_y = unscaled features
    # For loss computation, we need to scale targets to match scaled predictions
    print("🔧 Setting up scaling for loss computation...")
    print("   - batch_x will be pre-scaled by Dataset_Custom")
    print("   - batch_y will be unscaled from Dataset_Custom")
    print("   - For loss: we'll scale targets to match scaled predictions")
    
    # Get scalers from the dataset objects (Dataset_Custom stores them)
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
    
    # Define target indices for OHLC (first 4 features in the CSV)
    target_indices = [0, 1, 2, 3]  # OHLC indices
    print(f"   - Target indices for OHLC: {target_indices}")
    
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
        criterion = SequentialMixtureNLLLoss(reduction='mean')
        print("🎯 Using Gaussian Mixture NLL Loss for probabilistic predictions")
    else:
        criterion = nn.MSELoss()
        print("📊 Using MSE Loss for deterministic predictions")
    
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    scaler = GradScaler(enabled=amp_enabled)
    
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

    if target_indices is None:
        target_indices = list(range(getattr(args, 'c_out', 4)))

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

            # ✅ FIXED: Use data as-is from Dataset_Custom (no additional scaling needed)
            # batch_x is already scaled by Dataset_Custom
            # batch_y is unscaled (correct for ground truth)

            batch_x_input = batch_x  # Already scaled by Dataset_Custom
            # Prepare decoder input using original data (match encoder dimensions)
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()  # [32, 24, 118]
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)  # [32, 48, 118]

            try:
                base_loss = None
                reg_loss = None
                reg_contribution = None
                with (autocast(device_type="cuda", dtype=torch.float16) if amp_enabled else nullcontext()):
                    # Forward pass (batch_x_input is already properly scaled by Dataset_Custom)
                    outputs_raw = model(batch_x_input, batch_x_mark, dec_inp, batch_y_mark)

                # Normalize model output using the same approach as exp_long_term_forecasting.py
                outputs_tensor, aux_loss, mdn_outputs, metadata = _normalize_model_output(outputs_raw)

                # Prepare y_true for loss: scale the target part of batch_y
                c_out_evaluation = len(target_indices)
                y_true_targets_unscaled = batch_y[:, -args.pred_len:, :c_out_evaluation]
                y_true_for_loss = scale_targets_for_loss(
                    batch_y[:, -args.pred_len:, :], target_scaler, target_indices, device
                )

                # Debug: Print stats for first batch only
                if i == 0:
                    print(f"🔍 TRAIN - Model I/O shapes: input={batch_x_input.shape}, output={outputs_tensor.shape}")
                    print(f"🔍 TRAIN - Loss computation: pred_shape={outputs_tensor[:, -args.pred_len:, :c_out_evaluation].shape}, target_shape={y_true_for_loss.shape}")
                    print(f"📊 TRAIN - Raw batch_y stats (all features): mean={batch_y.mean():.6f}, std={batch_y.std():.6f}")
                    print(f"📊 TRAIN - OHLC unscaled: mean={batch_y[:, -args.pred_len:, :4].mean():.6f}, std={batch_y[:, -args.pred_len:, :4].std():.6f}")
                    print(f"📊 TRAIN - Scaled targets: mean={y_true_for_loss.mean():.6f}, std={y_true_for_loss.std():.6f}")
                    print(f"📊 TRAIN - Model outputs: mean={outputs_tensor.mean():.6f}, std={outputs_tensor.std():.6f}")
                    print(f"🎯 TRAIN - First batch loss will be computed on these scaled values")

                # Compute loss using the same approach as exp_long_term_forecasting.py
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

                # Add regularization loss from stochastic graph learner
                if getattr(model, 'use_stochastic_learner', False):
                    reg_loss = model.get_regularization_loss()
                    # CRITICAL FIX: Much smaller regularization weight to prevent loss explosion
                    reg_weight = getattr(args, 'reg_loss_weight', 0.001)  # Reduced from 0.1 to 0.001
                    reg_contribution = reg_loss * reg_weight
                    loss += reg_contribution

                    if i % 100 == 0:  # Log regularization loss occasionally
                        print(f"📊 Regularization loss: {reg_loss.item():.6f} (weighted: {reg_contribution.item():.6f})")

                # Backward pass with optional AMP scaling
                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
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

                if amp_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
                
                # Log progress
                if i % 50 == 0:
                    print(f"      Epoch {epoch+1:3d} | Batch {i:4d}/{len(train_loader)} | Loss: {loss.item():.6f}")
                    
                    # Debug: Check if outputs are changing
                    if isinstance(outputs_tensor, torch.Tensor):
                        output_mean = outputs_tensor.mean().item()
                        output_std = outputs_tensor.std().item()
                        print(f"         📊 Output stats: mean={output_mean:.6f}, std={output_std:.6f}")
                    
                    # Print celestial metadata
                    if metadata and 'celestial_metadata' in metadata:
                        celestial_meta = metadata['celestial_metadata']
                        if 'most_active_body' in celestial_meta:
                            body_names = celestial_meta.get('body_names', [])
                            if celestial_meta['most_active_body'] < len(body_names):
                                active_body = body_names[celestial_meta['most_active_body']]
                                print(f"         🌌 Most active celestial body: {active_body}")

                    if i == 0:
                        # Log criterion versus total loss breakdown for the first batch
                        print(f"         🟦 Criterion loss (train mode): {base_loss.item():.6f}")
                        if reg_contribution is not None:
                            print(f"         🟥 Regularizer contribution: {reg_contribution.item():.6f}")
                            print(f"         🟨 Total loss (train mode): {loss.item():.6f}")
                        else:
                            print(f"         🟨 Total loss (train mode): {loss.item():.6f}")

                        # Cross-check the same batch in evaluation mode for debugging
                        previous_mode_training = model.training
                        try:
                            model.eval()
                            with torch.no_grad():
                                with (autocast(device_type="cuda", dtype=torch.float16) if amp_enabled else nullcontext()):
                                    eval_outputs, eval_metadata = model(batch_x_input, batch_x_mark, dec_inp, batch_y_mark)

                                if (model.aggregate_waves_to_celestial and 
                                    'original_targets' in eval_metadata and 
                                    eval_metadata['original_targets'] is not None):
                                    eval_targets_for_loss = eval_metadata['original_targets'][:, -args.pred_len:, :]
                                else:
                                    eval_targets_unscaled = batch_y[:, -args.pred_len:, :]
                                    eval_targets_for_loss = scale_targets_for_loss(
                                        eval_targets_unscaled, target_scaler, target_indices, device
                                    )

                                if getattr(model, 'use_mixture_decoder', False) and isinstance(eval_outputs, dict):
                                    eval_means = eval_outputs['means']
                                    eval_log_stds = eval_outputs['log_stds']
                                    eval_log_weights = eval_outputs['log_weights']
                                    eval_loss = criterion((eval_means, eval_log_stds, eval_log_weights), eval_targets_for_loss)
                                elif getattr(model, 'use_mixture_decoder', False) and isinstance(eval_outputs, tuple):
                                    eval_means, eval_log_stds, eval_log_weights = eval_outputs
                                    eval_loss = criterion((eval_means, eval_log_stds, eval_log_weights), eval_targets_for_loss)
                                else:
                                    eval_out_time = eval_outputs[:, -args.pred_len:, :] if eval_outputs.shape[1] >= args.pred_len else eval_outputs
                                    if eval_out_time.shape[-1] >= len(target_indices):
                                        eval_out_slice = eval_out_time[:, :, :len(target_indices)]
                                    else:
                                        eval_out_slice = eval_out_time
                                    eval_loss = criterion(eval_out_slice, eval_targets_for_loss)

                                print(f"         🧪 Eval-mode criterion on same batch: {eval_loss.item():.6f}")
                        finally:
                            if previous_mode_training:
                                model.train()
                
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
                
                # Use data as-is from Dataset_Custom (already properly scaled/unscaled)
                batch_x_input = batch_x  # Already scaled
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
                
                try:
                    with (autocast(device_type="cuda", dtype=torch.float16) if amp_enabled else nullcontext()):
                        outputs_raw = model(batch_x_input, batch_x_mark, dec_inp, batch_y_mark)
                    
                    # Normalize model output using the same approach as exp_long_term_forecasting.py
                    outputs_tensor, aux_loss, mdn_outputs, metadata = _normalize_model_output(outputs_raw)

                    # Prepare y_true for loss: scale the target part of batch_y
                    c_out_evaluation = len(target_indices)
                    y_true_for_loss = scale_targets_for_loss(
                        batch_y[:, -args.pred_len:, :], target_scaler, target_indices, device
                    )

                    # Debug validation loss computation (first batch only)
                    if val_batches == 0:
                        print(f"🔍 VAL - Model I/O shapes: input={batch_x_input.shape}, output={outputs_tensor.shape}")
                        print(f"🔍 VAL - Loss computation: pred_shape={outputs_tensor[:, -args.pred_len:, :c_out_evaluation].shape}, target_shape={y_true_for_loss.shape}")
                        print(f"📊 VAL - Raw batch_y stats (all features): mean={batch_y.mean():.6f}, std={batch_y.std():.6f}")
                        print(f"📊 VAL - OHLC unscaled: mean={batch_y[:, -args.pred_len:, :4].mean():.6f}, std={batch_y[:, -args.pred_len:, :4].std():.6f}")
                        print(f"📊 VAL - Scaled targets: mean={y_true_for_loss.mean():.6f}, std={y_true_for_loss.std():.6f}")
                        print(f"📊 VAL - Model outputs: mean={outputs_tensor.mean():.6f}, std={outputs_tensor.std():.6f}")
                        print(f"🎯 VAL - Using same scaler as training (fitted on training data)")
                        
                        # Show the difference in data distributions
                        val_targets_unscaled = batch_y[:, -args.pred_len:, target_indices]
                        print(f"📈 VAL vs TRAIN data distribution:")
                        print(f"   - VAL unscaled std: {val_targets_unscaled.std():.6f}")
                        print(f"   - Scaler fitted on TRAIN std: ~{target_scaler.scale_[0]:.6f}")
                        print(f"   - This difference explains why VAL loss may differ from TRAIN loss")

                    # Compute loss using the same approach as exp_long_term_forecasting.py
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

                    if val_batches == 0:
                        print(f"� VAL  - Loss value: {loss.item():.6f}")
                    
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
                with (autocast(device_type="cuda", dtype=torch.float16) if amp_enabled else nullcontext()):
                    outputs_raw = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # Normalize model output using the same approach as exp_long_term_forecasting.py
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
                
                # Align time and target channels to evaluation dimensions
                # Ensure predictions and ground truth match (args.pred_len, eval_c)
                pred_aligned = pred_tensor[:, -args.pred_len:, :]
                true_aligned = true_tensor[:, -args.pred_len:, :]
                if target_indices is not None:
                    pred_aligned = pred_aligned[:, :, target_indices]
                    true_aligned = true_aligned[:, :, target_indices]

                # Convert to numpy for metrics computation
                # Dataset_Custom provides the right scaling - predictions and ground truth are on compatible scales
                pred = pred_aligned.detach().cpu().numpy()
                true = true_aligned.detach().cpu().numpy()
                
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

def main() -> int:
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