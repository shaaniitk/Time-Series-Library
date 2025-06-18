#!/usr/bin/env python3
"""
Training Script for Enhanced Autoformer with Advanced Features

This script demonstrates how to train the Enhanced Autoformer with:
- Adaptive loss functions
- Curriculum learning
- Enhanced monitoring and logging
- Memory optimization
- Support for MultiScaleTrendAwareLoss
"""

import os
import sys
import argparse
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from utils.tools import visual

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Suppress warnings
warnings.filterwarnings('ignore')

# Imports
from models.EnhancedAutoformer import Model as EnhancedAutoformer # Ensure this is the correct path
from utils.enhanced_losses import AdaptiveAutoformerLoss, CurriculumLossScheduler, create_enhanced_loss
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from utils.logger import logger, set_log_level
import logging

# Import get_loss_function from utils.losses
from utils.losses import get_loss_function # Import for general loss selection


class EnhancedAutoformerTrainer:
    """
    Enhanced training framework for Autoformer with advanced features.
    """
    
    def __init__(self, args):
        logger.info("Initializing EnhancedAutoformerTrainer")
        self.args = args
        self.device = self._get_device()
        
        # Setup model
        self.model = self._build_model()
        
        # Setup data
        self.train_data, self.train_loader = self._get_data('train')
        self.val_data, self.val_loader = self._get_data('val')
        self.test_data, self.test_loader = self._get_data('test')
        
        # Setup training components
        self.optimizer = self._select_optimizer()
        self.criterion = self._select_criterion()
        self.early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        
        # Setup curriculum learning if enabled
        if args.use_curriculum:
            self.curriculum = CurriculumLossScheduler(
                start_seq_len=args.curriculum_start_len,
                target_seq_len=args.seq_len,
                curriculum_epochs=args.curriculum_epochs,
                loss_fn=self.criterion # Pass the criterion to the scheduler
            )
        else:
            self.curriculum = None
            
        # Training metrics tracking
        self.training_metrics = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'epoch_times': [],
            'memory_usage': []
        }
        
    def _get_device(self):
        """Setup device for training."""
        if self.args.use_gpu and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.args.gpu}')
            logger.info(f"Using GPU: {device}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU")
        return device
        
    def _build_model(self):
        """Build Enhanced Autoformer model."""
        logger.info("Building Enhanced Autoformer model")
        
        # Pass quantile_levels if available and model supports it
        q_levels = getattr(self.args, 'quantile_levels', None)
        
        # Check if the model class accepts quantile_levels in its __init__
        # This requires importing the specific model class here
        # For simplicity, assuming EnhancedAutoformer accepts it based on previous diffs
        # If using train_dynamic_autoformer.py, this logic would be in Exp_Long_Term_Forecast._build_model
        
        model = EnhancedAutoformer(self.args, quantile_levels=q_levels).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            
        model = model.to(self.device)
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        return model
        
    def _get_data(self, flag):
        """Get data loader for specified split."""
        # Pass validation_length and test_length to data_provider
        self.args.validation_length = getattr(self.args, 'val_len', 150)
        self.args.test_length = getattr(self.args, 'test_len', 50)
        
        data_set, data_loader = data_provider(self.args, flag)
        logger.info(f"{flag} data: {len(data_set)} samples, {len(data_loader)} batches")
        return data_set, data_loader
        
    def _select_optimizer(self):
        """Select optimizer with enhanced scheduling."""
        optimizer_name = getattr(self.args, 'optimizer', 'adam')
        
        if optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.args.learning_rate,
                weight_decay=getattr(self.args, 'weight_decay', 0) # Default weight_decay to 0 if not in args
            )
        elif optimizer_name.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=getattr(self.args, 'weight_decay', 0.01) # Default weight_decay for AdamW
            )
        else:
            logger.warning(f"Unsupported optimizer: {optimizer_name}, defaulting to Adam")
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
            
        logger.info(f"Using optimizer: {optimizer_name}")
        return optimizer
        
    def _select_criterion(self):
        """Select enhanced loss function."""
        loss_name = getattr(self.args, 'loss_type', 'mse') # Use loss_type as the loss_name argument
        logger.info(f"Selecting loss function: {loss_name}")

        if loss_name.lower() == 'adaptive': # This is a specific loss from enhanced_losses.py
            criterion = AdaptiveAutoformerLoss(
                base_loss=getattr(self.args, 'base_loss', 'mse'),
                adaptive_weights=getattr(self.args, 'adaptive_weights', True),
                moving_avg=getattr(self.args, 'moving_avg', 25)
            )
            logger.info("Using adaptive autoformer loss")
        elif loss_name.lower() == 'multiscale_trend_aware':
            trend_window_sizes = getattr(self.args, 'trend_window_sizes', [60, 20, 5])
            trend_component_weights = getattr(self.args, 'trend_component_weights', [1.0, 0.8, 0.5])
            noise_component_weight = getattr(self.args, 'noise_component_weight', 0.2)
            base_loss_fn_str = getattr(self.args, 'base_loss_fn_str', 'mse')
            
            # Ensure weights match window sizes
            if len(trend_window_sizes) != len(trend_component_weights):
                 logger.warning(f"Mismatch between trend_window_sizes ({len(trend_window_sizes)}) and trend_component_weights ({len(trend_component_weights)}). Using default weights.")
                 # Simple fallback: repeat the first weight or use equal weights
                 trend_component_weights = [1.0] * len(trend_window_sizes) # Or np.ones(len(trend_window_sizes)) / len(trend_window_sizes)
                 
            criterion = get_loss_function(
                loss_name, # Pass the actual name 'multiscale_trend_aware'
                trend_window_sizes=trend_window_sizes,
                trend_component_weights=trend_component_weights,
                noise_component_weight=noise_component_weight,
                base_loss_fn_str=base_loss_fn_str
            )
            logger.info(f"Using MultiScaleTrendAwareLoss with base: {base_loss_fn_str}")
        elif loss_name.lower() == 'pinball':
             # Prioritize 'quantile_levels' if available, then 'quantiles'
            quantiles_to_use = getattr(self.args, 'quantile_levels', None)
            if quantiles_to_use is None:
                quantiles_to_use = getattr(self.args, 'quantiles', [0.1, 0.5, 0.9]) # Default
            if not isinstance(quantiles_to_use, list) or not all(isinstance(q, float) for q in quantiles_to_use if q is not None): # Allow None in list for default
                logger.warning(f"Invalid quantile_levels/quantiles: {quantiles_to_use}. Defaulting to [0.1, 0.5, 0.9]")
                quantiles_to_use = [0.1, 0.5, 0.9]
            criterion = get_loss_function(loss_name, quantiles=quantiles_to_use)
            logger.info(f"Using PinballLoss with quantiles: {quantiles_to_use}")
        else:
            # Fallback to general loss function getter for mse, mae, etc.
            try:
                # Pass quantile arg if it exists and loss is 'quantile'
                quantile_arg = getattr(self.args, 'quantile', 0.5) if loss_name.lower() == 'quantile' else None
                criterion = get_loss_function(loss_name, quantile=quantile_arg)
                logger.info(f"Using standard loss: {loss_name}")
            except ValueError:
                logger.warning(f"Unknown loss {loss_name}, defaulting to MSE")
                criterion = nn.MSELoss()
            
        return criterion
        
    def train_epoch(self, epoch):
        """Train for one epoch with enhanced features."""
        self.model.train()
        
        train_loss = []
        epoch_start_time = time.time()
        
        # Track memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # memory_start = torch.cuda.memory_allocated() # This can be noisy, maybe track max allocated
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.train_loader):
            
            # Apply curriculum learning if enabled
            if self.curriculum:
                batch_x, batch_y, batch_x_mark, batch_y_mark = self.curriculum.apply_curriculum(
                    batch_x, batch_y, batch_x_mark, batch_y_mark, epoch
                )
            
            # Move to device
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device) # For training, Dataset_Custom provides scaled targets
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)
            
            # Prepare decoder input (Standard Autoformer/EnhancedAutoformer approach)
            # Historical period: real targets + real historical covariates  
            dec_inp_historical = batch_y[:, :self.args.label_len, :].float().to(self.device)
            
            # Future period: zero targets + REAL future covariates (if available in batch_y)
            # Assuming targets are the first self.args.c_out columns in batch_y
            num_targets_in_batch_y = self.args.c_out # Assuming batch_y has at least c_out features
            
            # Check if batch_y has more features than c_out (i.e., includes covariates)
            if batch_y.shape[-1] > num_targets_in_batch_y:
                 # Zero out target columns in the future prediction period
                 future_targets_zero = torch.zeros_like(batch_y[:, -self.args.pred_len:, :num_targets_in_batch_y]).float().to(self.device)
                 # Keep future covariates as they are (they are already scaled by Dataset_Custom)
                 future_covariates = batch_y[:, -self.args.pred_len:, num_targets_in_batch_y:].float().to(self.device)
                 dec_inp_future = torch.cat([future_targets_zero, future_covariates], dim=-1)
            else:
                 # If batch_y only contains targets (S mode or c_out == total features), zero out everything in future
                 dec_inp_future = torch.zeros_like(batch_y[:, -self.args.pred_len:, :num_targets_in_batch_y]).float().to(self.device)
                 
            # Combine historical + future for decoder input
            dec_inp = torch.cat([dec_inp_historical, dec_inp_future], dim=1).float().to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Model output shape: [B, L, C] or [B, L, C, Q] if quantile mode
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            # Target ground truth for loss calculation
            # For training, batch_y is already scaled by Dataset_Custom
            # Slice batch_y to match the model's output features (c_out)
            batch_y_targets = batch_y[:, -self.args.pred_len:, :self.args.c_out].to(self.device)
            
            # Compute loss
            # If model outputs quantiles, outputs will be [B, L, C, Q]
            # If model outputs single prediction, outputs will be [B, L, C]
            # The criterion (PinballLoss or other) must handle the shape
            
            # If using a Bayesian model with compute_loss method
            if hasattr(self.model, 'compute_loss'):
                 # Bayesian model handles its own loss computation (including KL)
                 # It expects predictions (potentially with uncertainty/quantiles) and targets
                 # The criterion passed here is the data loss criterion (e.g., PinballLoss)
                 total_loss = self.model.compute_loss(outputs, batch_y_targets, self.criterion)
            else:
                 # Standard models use only data loss
                 total_loss = self.criterion(outputs, batch_y_targets)
            
            train_loss.append(total_loss.item())
            
            # Backward pass with gradient clipping
            total_loss.backward()
            if self.args.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            
            # Logging
            if i % self.args.log_interval == 0:
                logger.info(f'Epoch {epoch} [{i}/{len(self.train_loader)}] Loss: {total_loss.item():.6f}')
        
        # Calculate epoch metrics
        epoch_time = time.time() - epoch_start_time
        avg_train_loss = np.average(train_loss)
        
        # Memory usage (optional, requires GPU)
        # if torch.cuda.is_available():
        #     memory_end = torch.cuda.memory_allocated()
        #     memory_used = (memory_end - memory_start) / (1024**2)  # MB
        #     self.training_metrics['memory_usage'].append(memory_used)
        
        # Store metrics
        self.training_metrics['train_losses'].append(avg_train_loss)
        self.training_metrics['epoch_times'].append(epoch_time)
        self.training_metrics['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
        
        logger.info(f'Epoch {epoch} completed in {epoch_time:.2f}s, avg loss: {avg_train_loss:.6f}')
        
        return avg_train_loss
        
    def validate_epoch(self, epoch):
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.val_loader):
                batch_x = batch_x.float().to(self.device)
                # batch_y is kept on CPU for now as its target part might be unscaled
                # and needs processing before moving to device for loss calculation.
                # batch_y = batch_y.float() 

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # Prepare decoder input (Standard Autoformer/EnhancedAutoformer approach)
                # Historical period: real targets + real historical covariates  
                dec_inp_historical = batch_y[:, :self.args.label_len, :].float().to(self.device)
                
                # Future period: zero targets + REAL future covariates (if available in batch_y)
                num_targets_in_batch_y = self.args.c_out # Assuming batch_y has at least c_out features
                if batch_y.shape[-1] > num_targets_in_batch_y:
                     future_targets_zero = torch.zeros_like(batch_y[:, -self.args.pred_len:, :num_targets_in_batch_y]).float().to(self.device)
                     future_covariates = batch_y[:, -self.args.pred_len:, num_targets_in_batch_y:].float().to(self.device)
                     dec_inp_future = torch.cat([future_targets_zero, future_covariates], dim=-1)
                else:
                     dec_inp_future = torch.zeros_like(batch_y[:, -self.args.pred_len:, :num_targets_in_batch_y]).float().to(self.device)
                     
                dec_inp = torch.cat([dec_inp_historical, dec_inp_future], dim=1).float().to(self.device)
                
                # Forward pass
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # Model outputs are scaled, shape [B, pred_len, c_out] or [B, pred_len, c_out, Q]
                model_outputs_scaled = outputs[:, -self.args.pred_len:, :self.args.c_out]
                
                # batch_y from loader for val/test has UNCALED targets (first c_out) and SCALED covariates (rest)
                # We need to scale the target part of batch_y to compare with model_outputs_scaled.
                ground_truth_segment_unscaled_targets = batch_y[:, -self.args.pred_len:, :self.args.c_out].detach().cpu()
                
                # Initialize ground_truth_final_scaled with a copy (on CPU for now)
                ground_truth_final_scaled_cpu = ground_truth_segment_unscaled_targets.clone()

                # If using Dataset_Custom, its target_scaler was fit on training targets.
                # Validation batch_y has unscaled targets. We need to scale them.
                if hasattr(self.val_data, 'target_scaler') and self.val_data.scale and self.val_data.target_scaler is not None:
                    num_features_target_scaler_knows = self.val_data.target_scaler.n_features_in_
                    
                    # Number of target features within c_out that need explicit scaling
                    # This is min(features model outputs, features scaler was fit on)
                    num_targets_to_explicitly_scale = min(self.args.c_out, num_features_target_scaler_knows)

                    if num_targets_to_explicitly_scale > 0:
                        # Extract the raw, unscaled target features from batch_y that the scaler was fit on
                        # These are assumed to be the first 'num_features_target_scaler_knows' columns
                        unscaled_targets_for_scaler_np = batch_y[:, -self.args.pred_len:, :num_features_target_scaler_knows].numpy()
                        
                        # Scale these features
                        scaled_targets_full_set_np = self.val_data.target_scaler.transform(
                            unscaled_targets_for_scaler_np.reshape(-1, num_features_target_scaler_knows)
                        ).reshape(unscaled_targets_for_scaler_np.shape)
                        
                        # Update the target portion in ground_truth_final_scaled_cpu
                        # Only update the part that corresponds to the model's output (up to num_targets_to_explicitly_scale)
                        ground_truth_final_scaled_cpu[:, :, :num_targets_to_explicitly_scale] = torch.from_numpy(
                            scaled_targets_full_set_np[:, :, :num_targets_to_explicitly_scale]
                        ).float()
                # Else: assume batch_y is already appropriately scaled (e.g., ETT datasets or scale=False)

                # For validation, we typically only use data loss (no KL regularization for Bayesian models here)
                # The criterion (PinballLoss or other) must handle the shape
                loss = self.criterion(model_outputs_scaled, ground_truth_final_scaled_cpu.to(self.device))
                total_loss.append(loss.item())
        
        avg_val_loss = np.average(total_loss)
        self.training_metrics['val_losses'].append(avg_val_loss)
        
        logger.info(f'Validation loss: {avg_val_loss:.6f}')
        return avg_val_loss
        
    def test(self):
        """Test the model and compute metrics."""
        logger.info("Testing model")
        self.model.eval()
        
        preds_scaled_np = []
        trues_scaled_np = []
        
        # For visualization, store original unscaled targets if possible
        trues_original_for_viz_np = [] 

        # Determine if in quantile mode and find median index for metrics/viz
        is_quantile_mode = hasattr(self.args, 'quantile_levels') and \
                           isinstance(self.args.quantile_levels, list) and \
                           len(self.args.quantile_levels) > 0
        median_quantile_index = -1
        if is_quantile_mode:
            try:
                median_quantile_index = self.args.quantile_levels.index(0.5)
            except ValueError: # 0.5 not in list
                logger.warning("0.5 quantile not found in quantile_levels. Using first quantile for metrics/visualization.")
                median_quantile_index = 0 # Fallback to the first quantile if 0.5 is not present
        
        # Create directory for visualization plots
        viz_folder_path = os.path.join('./test_results', getattr(self.args, 'model_id', 'enhanced_autoformer_test_results'))
        os.makedirs(viz_folder_path, exist_ok=True)

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.test_loader):
                batch_x = batch_x.float().to(self.device)
                # batch_y kept on CPU for now
                # batch_y = batch_y.float() 

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Prepare decoder input (Standard Autoformer/EnhancedAutoformer approach)
                # Historical period: real targets + real historical covariates  
                dec_inp_historical = batch_y[:, :self.args.label_len, :].float().to(self.device)
                
                # Future period: zero targets + REAL future covariates (if available in batch_y)
                num_targets_in_batch_y = self.args.c_out # Assuming batch_y has at least c_out features
                if batch_y.shape[-1] > num_targets_in_batch_y:
                     future_targets_zero = torch.zeros_like(batch_y[:, -self.args.pred_len:, :num_targets_in_batch_y]).float().to(self.device)
                     future_covariates = batch_y[:, -self.args.pred_len:, num_targets_in_batch_y:].float().to(self.device)
                     dec_inp_future = torch.cat([future_targets_zero, future_covariates], dim=-1)
                else:
                     dec_inp_future = torch.zeros_like(batch_y[:, -self.args.pred_len:, :num_targets_in_batch_y]).float().to(self.device)
                     
                dec_inp = torch.cat([dec_inp_historical, dec_inp_future], dim=1).float().to(self.device)
                
                # Forward pass
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # Model output is scaled, shape [B, pred_len, C] or [B, pred_len, C, Q]
                # Extract the relevant part for metrics/viz (first c_out features)
                model_outputs_raw = outputs[:, -self.args.pred_len:, :self.args.c_out]
                
                # Log dimensions for debugging
                if i == 0:  # Log only for first batch to avoid spam
                    logger.info(f"Test batch dimensions:")
                    logger.info(f"  batch_x: {batch_x.shape}")
                    logger.info(f"  batch_y: {batch_y.shape}")
                    logger.info(f"  batch_x_mark: {batch_x_mark.shape}")
                    logger.info(f"  batch_y_mark: {batch_y_mark.shape}")
                    logger.info(f"  dec_inp: {dec_inp.shape}")
                    logger.info(f"  model outputs (raw): {outputs.shape}") # Full model output
                    logger.info(f"  model_outputs_raw (pred_len segment): {model_outputs_raw.shape}")
                
                # Prepare ground_truth_final_scaled for metrics (similar to vali)
                # batch_y from loader for val/test has UNCALED targets
                ground_truth_segment_unscaled_targets = batch_y[:, -self.args.pred_len:, :self.args.c_out].clone().detach().cpu()
                true_scaled_batch_cpu = ground_truth_segment_unscaled_targets.clone() # Start with a copy on CPU

                if hasattr(self.test_data, 'target_scaler') and self.test_data.scale and self.test_data.target_scaler is not None:
                    num_features_target_scaler_knows = self.test_data.target_scaler.n_features_in_
                    num_targets_to_explicitly_scale = min(self.args.c_out, num_features_target_scaler_knows)
                    
                    if num_targets_to_explicitly_scale > 0:
                        # Extract the raw, unscaled target features from batch_y that the scaler was fit on
                        # These are assumed to be the first 'num_features_target_scaler_knows' columns
                        unscaled_targets_for_scaler_np = batch_y[:, -self.args.pred_len:, :num_features_target_scaler_knows].numpy()
                        trues_original_for_viz_np.append(unscaled_targets_for_scaler_np[:,:,:num_targets_to_explicitly_scale]) # Store relevant part for viz

                        # Scale these features
                        scaled_targets_full_set_np = self.test_data.target_scaler.transform(
                            unscaled_targets_for_scaler_np.reshape(-1, num_features_target_scaler_knows)
                        ).reshape(unscaled_targets_for_scaler_np.shape)
                        
                        # Update the target portion in true_scaled_batch_cpu
                        # Only update the part that corresponds to the model's output (up to num_targets_to_explicitly_scale)
                        true_scaled_batch_cpu[:, :, :num_targets_to_explicitly_scale] = torch.from_numpy(
                            scaled_targets_full_set_np[:, :, :num_targets_to_explicitly_scale]
                        ).float()
                else: # If not Dataset_Custom or no scaling, assume batch_y is already what we need for metrics (or store as is for viz)
                    trues_original_for_viz_np.append(ground_truth_segment_unscaled_targets.numpy())

                # pred_np_batch will be [B, pred_len, C] or [B, pred_len, C, Q]
                pred_np_batch_raw = model_outputs_raw.detach().cpu().numpy()

                if is_quantile_mode and pred_np_batch_raw.ndim == 4 and median_quantile_index != -1:
                    # Extract median predictions for metrics: [B, pred_len, C, Q] -> [B, pred_len, C]
                    pred_np_batch = pred_np_batch_raw[:, :, :, median_quantile_index]
                else:
                    pred_np_batch = pred_np_batch_raw # Assumes [B, pred_len, C]
                
                true_np_batch = true_scaled_batch_cpu.numpy() # Use the correctly constructed scaled ground truth
                
                preds_scaled_np.append(pred_np_batch)
                trues_scaled_np.append(true_np_batch)

                # Visualization part (only for the first batch and first target feature)
                if i == 0 and self.args.c_out > 0: 
                    input_np = batch_x.detach().cpu().numpy()
                    
                    # For visualization, we want to show data in its original scale if possible
                    # pred_to_plot: model's output, needs inverse_transform_targets
                    # true_to_plot: original unscaled targets from batch_y

                    if is_quantile_mode and pred_np_batch_raw.ndim == 4 and median_quantile_index != -1:
                        pred_for_viz = pred_np_batch_raw[0, :, :, median_quantile_index] # Median prediction for viz
                    else:
                        pred_for_viz = pred_np_batch_raw[0, :, :] # Standard prediction for viz

                    # Ensure trues_original_for_viz_np has data for the first batch
                    true_for_viz_original = trues_original_for_viz_np[0][0, :, :] if trues_original_for_viz_np else true_np_batch[0, :, :] # Fallback if no original stored

                    if hasattr(self.test_data, 'inverse_transform_targets') and self.test_data.scale:
                        # Inverse transform only the target portion of predictions
                        # Assuming target_scaler was fit on num_features_target_scaler_knows features
                        # and these are the first ones in c_out.
                        num_pred_targets_to_inv_transform = min(pred_for_viz.shape[-1], self.test_data.target_scaler.n_features_in_)
                        
                        if num_pred_targets_to_inv_transform > 0:
                            pred_target_part_scaled = pred_for_viz[:, :num_pred_targets_to_inv_transform]
                            pred_target_part_original = self.test_data.inverse_transform_targets(pred_target_part_scaled)
                            
                            # For input, inverse transform its target part
                            input_target_part_unscaled = self.test_data.inverse_transform_targets(input_np[0, :, :self.test_data.target_scaler.n_features_in_])
                            
                            # Visualize the first target feature (index 0)
                            if input_target_part_unscaled.shape[-1] > 0 and true_for_viz_original.shape[-1] > 0 and pred_target_part_original.shape[-1] > 0:
                                gt_plot = np.concatenate((input_target_part_unscaled[:, 0], true_for_viz_original[:, 0]), axis=0)
                                pd_plot = np.concatenate((input_target_part_unscaled[:, 0], pred_target_part_original[:, 0]), axis=0)
                                visual(gt_plot, pd_plot, os.path.join(viz_folder_path, f'batch_{i}_target_0.pdf'))
                            else:
                                logger.warning("Cannot visualize: insufficient target features after inverse transform.")

                        else:
                             logger.warning("Cannot visualize: No target features to inverse transform.")

                    else: # If no inverse_transform_targets or not scaled, plot as is (likely scaled)
                        if input_np.shape[-1] > 0 and true_np_batch.shape[-1] > 0 and pred_np_batch.shape[-1] > 0:
                             # Visualize the first feature (index 0)
                             gt_plot = np.concatenate((input_np[0, :, 0], true_np_batch[0, :, 0]), axis=0)
                             pd_plot = np.concatenate((input_np[0, :, 0], pred_np_batch[0, :, 0]), axis=0)
                             visual(gt_plot, pd_plot, os.path.join(viz_folder_path, f'batch_{i}_target_0_scaled.pdf'))
                        else:
                             logger.warning("Cannot visualize: insufficient features.")


        preds = np.concatenate(preds_scaled_np, axis=0)
        trues = np.concatenate(trues_scaled_np, axis=0)
        logger.info(f'Test shape (scaled for metrics): preds={preds.shape}, trues={trues.shape}')
        
        # Calculate metrics on scaled data (consistent with training/validation)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        
        logger.info(f"Test Results (Scaled Space) - MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}")
        logger.info(f"Test Results (Scaled Space) - MAPE: {mape:.6f}, MSPE: {mspe:.6f}")
        
        # Save results
        results_folder_path = os.path.join('./results', getattr(self.args, 'model_id', 'enhanced_autoformer_results'))
        os.makedirs(results_folder_path, exist_ok=True)

        metrics_path = os.path.join(results_folder_path, 'metrics.npy')
        np.save(metrics_path, np.array([mae, mse, rmse, mape, mspe]))
        logger.info(f"Metrics saved to {metrics_path}")

        preds_path = os.path.join(results_folder_path, 'pred_scaled.npy')
        np.save(preds_path, preds) # Save scaled predictions
        logger.info(f"Scaled predictions saved to {preds_path}")

        trues_path = os.path.join(results_folder_path, 'true_scaled.npy')
        np.save(trues_path, trues) # Save scaled trues
        logger.info(f"Scaled true values saved to {trues_path}")
        
        return {
            'mse': mse, 'mae': mae, 'rmse': rmse, 'mape': mape, 'mspe': mspe,
            'predictions_scaled': preds, 'true_values_scaled': trues
        }
    
        
    def train(self):
        """Main training loop."""
        logger.info("Starting Enhanced Autoformer training")
        
        # NOTE: Scaling consistency fix
        # - Model predictions are in scaled space (trained on scaled data)
        # - Validation/test ground truth is unscaled (to avoid data leakage)
        # - We scale ground truth during loss/metric computation to match predictions
        
        best_val_loss = float('inf')
        training_start_time = time.time()
        
        # Create checkpoint directory with proper model_id handling
        model_id = getattr(self.args, 'model_id', 'enhanced_autoformer')
        checkpoint_dir = os.path.join('checkpoints', model_id)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(self.args.train_epochs):
            epoch_start = time.time()
            
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_loss = self.validate_epoch(epoch)
            
            # Learning rate adjustment
            adjust_learning_rate(self.optimizer, epoch + 1, self.args)
            
            # Early stopping
            self.early_stopping(val_loss, self.model, checkpoint_dir) # Use checkpoint_dir
            if self.early_stopping.early_stop:
                logger.info("Early stopping triggered")
                break
                
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(os.path.join(checkpoint_dir, 'best_model.pth')) # Save to checkpoint_dir
                
            epoch_time = time.time() - epoch_start
            logger.info(f'Epoch {epoch+1}/{self.args.train_epochs} completed in {epoch_time:.2f}s')
            
        # Final testing
        total_training_time = time.time() - training_start_time
        logger.info(f'Training completed in {total_training_time:.2f}s')
        
        # Load best model and test
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
             self.load_model(best_model_path)
        else:
             logger.warning(f"Best model checkpoint not found at {best_model_path}. Testing with last epoch model.")

        test_results = self.test()
        
        # Save training metrics
        self.save_training_metrics(checkpoint_dir) # Save metrics to checkpoint dir
        
        return {
            'best_val_loss': best_val_loss,
            'test_results': test_results,
            'training_metrics': self.training_metrics,
            'total_training_time': total_training_time
        }
        
    def save_model(self, full_path):
        """Save model checkpoint."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'args': self.args
        }, full_path)
        logger.info(f"Model saved to {full_path}")
        
    def load_model(self, full_path):
        """Load model checkpoint."""
        if os.path.exists(full_path):
            checkpoint = torch.load(full_path, map_location=self.device) # Removed weights_only=False for broader compatibility
            self.model.load_state_dict(checkpoint['model_state_dict'])
            # Optionally load optimizer state if resuming training
            # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
            logger.info(f"Model loaded from {full_path}")
        else:
            logger.warning(f"Checkpoint not found: {full_path}")
            
    def save_training_metrics(self, checkpoint_dir):
        """Save training metrics to file."""
        metrics_path = os.path.join(checkpoint_dir, 'training_metrics.json')
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {}
        for key, value in self.training_metrics.items():
            if isinstance(value, list):
                serializable_metrics[key] = [item.tolist() if hasattr(item, 'tolist') else item for item in value] # Handle lists of numpy arrays
            else:
                serializable_metrics[key] = value.tolist() if hasattr(value, 'tolist') else value
        
        try:
            with open(metrics_path, 'w') as f:
                json.dump(serializable_metrics, f, indent=2)
            logger.info(f"Training metrics saved to {metrics_path}")
        except Exception as e:
            logger.error(f"Failed to save training metrics to {metrics_path}: {e}")


def create_args():
    """Create argument parser for enhanced training."""
    parser = argparse.ArgumentParser(description='Enhanced Autoformer Training')
    
    # Basic model parameters
    parser.add_argument('--model', type=str, default='EnhancedAutoformer', help='Model name')
    parser.add_argument('--model_id', type=str, default='enhanced_autoformer_exp', help='Model identifier for checkpoints')
    parser.add_argument('--task_name', type=str, default='long_term_forecast', help='Task name')
    
    # Data settings
    parser.add_argument('--data', type=str, default='custom', help='Dataset type')
    parser.add_argument('--root_path', type=str, default='./data/', help='Root path of the data file')
    parser.add_argument('--data_path', type=str, default='prepared_financial_data.csv', help='Data file name')
    parser.add_argument('--features', type=str, default='MS', help='Forecasting task [M, S, MS]')
    parser.add_argument('--target', type=str, default='log_Open,log_High,log_Low,log_Close', help='Target feature(s) in S or MS task (comma-separated)')
    parser.add_argument('--freq', type=str, default='b', help='Freq for time features encoding [b=business day]')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='Location of model checkpoints')
    parser.add_argument('--scale', action='store_true', default=True, help='Scale input data') # Added for Dataset_Custom
    
    # Forecasting task sequence lengths
    parser.add_argument('--seq_len', type=int, default=250, help='Input sequence length')
    parser.add_argument('--label_len', type=int, default=15, help='Start token length for decoder')
    parser.add_argument('--pred_len', type=int, default=10, help='Prediction sequence length')
    parser.add_argument('--val_len', type=int, default=150, help='Validation length (number of samples)') # Added for Dataset_Custom
    parser.add_argument('--test_len', type=int, default=50, help='Test length (number of samples)') # Added for Dataset_Custom
    
    # Model architecture parameters
    # Note: enc_in, dec_in, c_out should ideally be set dynamically based on data and features mode
    # For this script, we'll set defaults suitable for financial data (118 total, 4 targets)
    parser.add_argument('--enc_in', type=int, default=118, help='Encoder input size')
    parser.add_argument('--dec_in', type=int, default=4, help='Decoder input size (should match c_out for MS/S)')
    parser.add_argument('--c_out', type=int, default=4, help='Output size (number of target features)')
    parser.add_argument('--d_model', type=int, default=128, help='Dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='Number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=2, help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=256, help='Dimension of FCN')
    parser.add_argument('--moving_avg', type=int, default=25, help='Window size of moving average for decomposition')
    parser.add_argument('--factor', type=int, default=1, help='Attention factor')
    parser.add_argument('--dropout', type=float, default=0.15, help='Dropout rate')
    parser.add_argument('--embed', type=str, default='timeF', help='Time features encoding [timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='Activation function')
    
    # Training parameters
    parser.add_argument('--train_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay for optimizer')
    parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer [adam, adamw]')
    parser.add_argument('--lradj', type=str, default='type1', help='Learning rate adjustment strategy')
    parser.add_argument('--log_interval', type=int, default=100, help='Log training loss every N batches')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level [DEBUG, INFO, WARNING, ERROR]')
    
    # Enhanced features (Loss, Curriculum, Grad Clip)
    parser.add_argument('--loss_type', type=str, default='mse', 
                        help='Type of loss function (e.g., mse, mae, adaptive, multiscale_trend_aware, pinball)')
    # Args for AdaptiveAutoformerLoss (if loss_type == 'adaptive')
    parser.add_argument('--base_loss', type=str, default='mse', help='Base loss for adaptive loss')
    parser.add_argument('--adaptive_weights', action='store_true', default=True, help='Use learnable weights in adaptive loss')
    # Args for MultiScaleTrendAwareLoss (if loss_type == 'multiscale_trend_aware')
    parser.add_argument('--trend_window_sizes', type=int, nargs='+', default=[60, 20, 5], help='Window sizes for MSTL')
    parser.add_argument('--trend_component_weights', type=float, nargs='+', default=[1.0, 0.8, 0.5], help='Weights for trend components in MSTL')
    parser.add_argument('--noise_component_weight', type=float, default=0.2, help='Weight for noise component in MSTL')
    parser.add_argument('--base_loss_fn_str', type=str, default='mse', help='Base loss for MSTL components (mse or mae)')
    # Args for PinballLoss (if loss_type == 'pinball')
    parser.add_argument('--quantile_levels', type=float, nargs='+', default=None, help='List of quantiles for PinballLoss') # Use this for multi-quantile
    parser.add_argument('--quantile', type=float, default=0.5, help='Single quantile for QuantileLoss (if loss_type is quantile)') # Use this for single quantile
    
    parser.add_argument('--use_curriculum', action='store_true', help='Enable curriculum learning')
    parser.add_argument('--curriculum_start_len', type=int, default=50, help='Starting sequence length for curriculum')
    parser.add_argument('--curriculum_epochs', type=int, default=20, help='Number of epochs for curriculum schedule')
    parser.add_argument('--use_grad_clip', action='store_true', default=True, help='Enable gradient clipping')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm for clipping')
    
    # Hardware settings
    parser.add_argument('--use_gpu', action='store_true', default=True, help='Use GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    parser.add_argument('--use_multi_gpu', action='store_true', help='Use multiple GPUs')
    parser.add_argument('--device_ids', type=str, default='0', help='Device ids of multiple GPUs (comma-separated)')
    parser.add_argument('--num_workers', type=int, default=4, help='Data loader num workers') # Reduced default for stability
    
    # Other settings (optional)
    parser.add_argument('--des', type=str, default='financial_forecast', help='Experiment description')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    
    args = parser.parse_args()
    
    # Process device ids for multi-gpu
    if args.use_multi_gpu and args.use_gpu:
        args.device_ids = [int(id_) for id_ in args.device_ids.split(',')]
    else:
        args.device_ids = [args.gpu] # Use single GPU id if not multi-gpu
        
    # --- Dynamic Dimension Setting based on Features Mode ---
    # This assumes the structure of prepared_financial_data.csv (4 targets + covariates)
    # For a truly dynamic script, this would involve analyzing the dataset file.
    # For this specific script tailored to financial data, we hardcode based on mode.
    
    TOTAL_FEATURES_FINANCIAL = 118 # 4 targets + 114 covariates
    TARGET_FEATURES_FINANCIAL = len(args.target.split(',')) # Number of targets specified
    
    if args.features == 'M':
        args.enc_in = TOTAL_FEATURES_FINANCIAL
        args.dec_in = TOTAL_FEATURES_FINANCIAL # Decoder input includes all features for label_len part
        args.c_out = TOTAL_FEATURES_FINANCIAL # Predict all features
        logger.info(f"🎯 Mode 'M': enc_in={args.enc_in}, dec_in={args.dec_in}, c_out={args.c_out}")
    elif args.features == 'MS':
        args.enc_in = TOTAL_FEATURES_FINANCIAL # Encoder sees all features
        args.dec_in = TARGET_FEATURES_FINANCIAL # Decoder input label_len part uses target features
        args.c_out = TARGET_FEATURES_FINANCIAL # Predict only target features
        logger.info(f"🎯 Mode 'MS': enc_in={args.enc_in}, dec_in={args.dec_in}, c_out={args.c_out}")
    elif args.features == 'S':
        args.enc_in = TARGET_FEATURES_FINANCIAL # Encoder sees only target features
        args.dec_in = TARGET_FEATURES_FINANCIAL # Decoder input label_len part uses target features
        args.c_out = TARGET_FEATURES_FINANCIAL # Predict only target features
        logger.info(f"🎯 Mode 'S': enc_in={args.enc_in}, dec_in={args.dec_in}, c_out={args.c_out}")
    else:
        logger.warning(f"⚠️ Unknown features mode: {args.features}. Dimension setting might be incorrect.")
        # Fallback to defaults from parser if mode is unknown
        pass # Keep parser defaults for enc_in, dec_in, c_out

    # Ensure dec_in passed to the model matches the actual number of features
    # provided to the decoder's embedding layer.
    # In Autoformer/EnhancedAutoformer, dec_inp is constructed from batch_y[:, :label_len, :]
    # which contains either all features (M) or only targets (MS/S based on Dataset_Custom).
    # Dataset_Custom in M/MS/S mode loads all features into batch_y.
    # The dec_inp construction logic in train_epoch/validate_epoch/test determines what goes into dec_embedding.
    # The current dec_inp construction uses batch_y[:, :label_len, :] for historical part.
    # If features='M', batch_y[:, :label_len, :] has TOTAL_FEATURES_FINANCIAL.
    # If features='MS' or 'S', batch_y[:, :label_len, :] has TOTAL_FEATURES_FINANCIAL.
    # This means dec_embedding always receives TOTAL_FEATURES_FINANCIAL features in its value part.
    # So, dec_in passed to the model should always be TOTAL_FEATURES_FINANCIAL for this script's data loading.
    # This contradicts the standard TSLib definition of dec_in for MS/S.
    # Let's adjust dec_in passed to the model to reflect what the embedding actually receives.
    # The model's decoder layers might still expect d_model input, not dec_in.
    # The dec_embedding maps dec_in to d_model.
    # So, dec_in should be the number of features in the decoder input tensor.
    
    # Re-evaluating dec_in based on dec_inp construction:
    # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp_future], dim=1)
    # batch_y[:, :self.args.label_len, :] has shape [B, label_len, batch_y.shape[-1]]
    # batch_y.shape[-1] is TOTAL_FEATURES_FINANCIAL (118) for Dataset_Custom with scale=True
    # dec_inp_future has shape [B, pred_len, batch_y.shape[-1]]
    # So dec_inp has shape [B, label_len + pred_len, batch_y.shape[-1]]
    # The dec_embedding receives dec_inp, so its input dimension should be batch_y.shape[-1].
    # This means dec_in should always be TOTAL_FEATURES_FINANCIAL (118) for this script.
    
    args.dec_in = TOTAL_FEATURES_FINANCIAL
    logger.info(f"Adjusted dec_in to match Dataset_Custom output: {args.dec_in}")


    # Display configuration summary
    print("\n🔧 CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"📁 Data: {args.data_path} (features: {args.features})")
    print(f"📏 Sequences: {args.seq_len} → {args.pred_len} (label: {args.label_len})")
    print(f"🧠 Model: d_model={args.d_model}, layers={args.e_layers}+{args.d_layers}, heads={args.n_heads}")
    print(f"⚡ Training: epochs={args.train_epochs}, batch={args.batch_size}, lr={args.learning_rate}")
    print(f"🎯 Loss: {args.loss_type}")
    if args.loss_type.lower() == 'multiscale_trend_aware':
        print(f"   MSTL Windows: {args.trend_window_sizes}, Weights: {args.trend_component_weights}, Noise W: {args.noise_component_weight}, Base: {args.base_loss_fn_str}")
    elif args.loss_type.lower() == 'pinball':
         print(f"   Quantiles: {args.quantile_levels}")
    elif args.loss_type.lower() == 'quantile':
         print(f"   Quantile: {args.quantile}")
    print(f"⚙️ Enhanced: Curriculum={args.use_curriculum}, Grad Clip={args.use_grad_clip}")
    print(f"💻 Hardware: GPU={args.use_gpu}, Devices={args.device_ids}, Workers={args.num_workers}")
    print("=" * 60)
    
    return args


def main():
    """Main training function."""
    # Parse arguments
    args = create_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level.upper())
    set_log_level(log_level)
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.use_gpu and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Create trainer and start training
    trainer = EnhancedAutoformerTrainer(args)
    
    logger.info("="*60)
    logger.info("ENHANCED AUTOFORMER TRAINING")
    logger.info("="*60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.data_path}")
    logger.info(f"Features: {args.features} (enc_in={args.enc_in}, dec_in={args.dec_in}, c_out={args.c_out})")
    logger.info(f"Sequence length: {args.seq_len}, Label length: {args.label_len}, Prediction length: {args.pred_len}")
    logger.info(f"Loss type: {args.loss_type}")
    logger.info(f"Curriculum learning: {args.use_curriculum}")
    logger.info("="*60)
    
    try:
        # Start training
        results = trainer.train()
        
        logger.info("Training completed successfully!")
        logger.info("="*60)
        logger.info("FINAL RESULTS")
        logger.info("="*60)
        logger.info(f"Best validation loss: {results['best_val_loss']:.6f}")
        logger.info(f"Test MSE (Scaled): {results['test_results']['mse']:.6f}")
        logger.info(f"Test MAE (Scaled): {results['test_results']['mae']:.6f}")
        logger.info(f"Total training time: {results['total_training_time']:.2f}s")
        logger.info("="*60)
        
        # You can optionally add production prediction here if implemented
        # trainer.predict_production()
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) # Exit with a non-zero code to indicate failure


if __name__ == '__main__':
    main()
