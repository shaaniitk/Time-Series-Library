from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.logger import logger
from argparse import Namespace # Import Namespace for model_init_args
from layers.modular.dimensions.dimension_manager import DimensionManager # Import DimensionManager
from utils.losses import PinballLoss # Import PinballLoss for type checking
from layers.modular.decoder.mixture_density_decoder import MixtureNLLLoss  # Import MixtureNLLLoss for MDN loss checking
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from layers.modular.curriculum.factory import CurriculumFactory
import inspect # For checking model constructor arguments
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
from typing import List, Optional, Sequence, Tuple, Union, cast

from utils.memory_diagnostics import MemoryDiagnostics

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        # The Exp_Long_Term_Forecast class is now initialized with pre-configured
        # DimensionManager, ScalerManager, and data loaders from train_dynamic_autoformer.py.
        # This simplifies its role to focus purely on the training/validation/testing loop.
        # The args object is still passed for general configuration.
        # The actual DM and ScalerManager are passed as separate arguments.
        logger.info(f"Initializing Exp_Long_Term_Forecast with provided managers and loaders.")
        logger.debug(f"Exp_Long_Term_Forecast __init__: args.scaler_manager is not None = {args.scaler_manager is not None}")

        self.memory_diagnostics: Optional[MemoryDiagnostics] = getattr(args, "memory_diagnostics", None)
        if self.memory_diagnostics is not None:
            self.memory_diagnostics.snapshot(
                "experiment_init_start",
                {
                    "model": getattr(args, "model", "unknown"),
                    "features": getattr(args, "features", None),
                },
            )
        
        # Store the managers and loaders
        self.dm = args.dim_manager # DimensionManager instance
        self.scaler_manager = args.scaler_manager # ScalerManager instance
        self.train_loader = args.train_loader
        self.vali_loader = args.vali_loader
        self.test_loader = args.test_loader

        # Construct model initialization arguments and evaluation info from DimensionManager and args
        # DimensionManager provides the core input/output feature dimensions
        # args provides other model hyperparameters (d_model, layers, etc.)

        # Start with a copy of all relevant args from the config file
        model_specific_args = {k: v for k, v in vars(args).items() if k not in [
            'dim_manager', 'scaler_manager', 'train_loader', 'vali_loader', 'test_loader',
            # Exclude data-related args that are handled by DM/ScalerManager or are not model-specific
            'root_path', 'data_path', 'target', 'features', 'scale', 'timeenc',
            'validation_length', 'test_length', 'val_len', 'test_len'
        ]}

        # Override/add core model dimensions from DimensionManager
        model_specific_args['enc_in'] = self.dm.enc_in
        model_specific_args['dec_in'] = self.dm.dec_in
        model_specific_args['c_out'] = self.dm.c_out_model # Use c_out_model for the final projection layer
        model_specific_args['c_out_evaluation'] = self.dm.c_out_evaluation # Pass the base evaluation c_out

        self.model_init_args = Namespace(**model_specific_args)
        
        self.eval_info = {
            'c_out_evaluation': self.dm.c_out_evaluation, # Number of actual target features for evaluation
            'num_quantiles': len(self.dm.quantiles) if self.dm.quantiles else 1,
            'quantile_levels': self.dm.quantiles,
            'loss_name': getattr(args, 'loss', self.dm.loss_function),
            'target_columns': self.dm.target_features # List of target column names
        }

        # Initialize the parent class *after* model_init_args is set.
        # Pass the resolved args to the parent, which will store it as self.args.
        print("DEBUG: Calling super().__init__", flush=True)
        super(Exp_Long_Term_Forecast, self).__init__(args)
        print("DEBUG: Returned from super().__init__", flush=True)
        
        # Ensure the args object used by the parent also has the managers
        # This is important if any other part of the code expects args.scaler_manager or args.dim_manager
        self.args.dim_manager = self.dm
        self.args.scaler_manager = self.scaler_manager

        print(f"DEBUG: Checking memory_diagnostics: {self.memory_diagnostics}", flush=True)
        if self.memory_diagnostics is not None:
            parameter_count = sum(param.numel() for param in self.model.parameters())
            self.memory_diagnostics.snapshot(
                "experiment_model_ready",
                {
                    "parameters": parameter_count,
                    "device": str(self.device),
                },
            )

        # Initialize Curriculum Learning Strategy
        print("DEBUG: initializing curriculum", flush=True)
        self.curriculum = CurriculumFactory.get_curriculum(self.args, self.device)
        print(f"DEBUG: curriculum initialized: {self.curriculum}", flush=True)
        if self.curriculum:
            logger.info(f"Initialized Curriculum Learning: {self.curriculum.__class__.__name__}")
        
        print("DEBUG: Exp_Long_Term_Forecast.__init__ complete", flush=True)
        
        # Test for GPU Deadlock
        import sys
        print("DEBUG: Synchronizing CUDA...", flush=True)
        torch.cuda.synchronize()
        print("DEBUG: Synchronized! GPU is responsive.", flush=True)
        
        # Checking if we accept return
        print("DEBUG: Returning from __init__", flush=True)

    def _build_model(self):
        ModelClass = self.model_dict[self.args.model]
        # Get model initialization parameters from the DimensionManager
        # self.model_init_args was set in __init__
        model = ModelClass(self.model_init_args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        from utils.losses import get_loss_function
        
        loss_name = getattr(self.args, 'loss', 'mse').lower()
        q_levels = getattr(self.args, 'quantile_levels', None) # For PinballLoss
        single_q_val = getattr(self.args, 'quantile', 0.5) # For single QuantileLoss

        criterion = None
        
        is_multi_quantile_scenario = isinstance(q_levels, list) and len(q_levels) > 0

        if loss_name == 'pinball' or (loss_name == 'quantile' and is_multi_quantile_scenario):
            if is_multi_quantile_scenario and q_levels is not None:
                levels_for_pinball_list: List[float] = [float(q) for q in q_levels]
            else:
                levels_for_pinball_list = [0.1, 0.5, 0.9]
                logger.warning(f"Loss is 'pinball' but no quantile_levels provided in args. Defaulting to {levels_for_pinball_list}")

            if not all(isinstance(q, float) and 0 < q < 1 for q in levels_for_pinball_list):
                logger.error(f"Invalid quantile_levels for PinballLoss: {levels_for_pinball_list}. Must be list of floats between 0 and 1.")
                logger.warning("Falling back to MSELoss due to invalid quantile_levels for PinballLoss.")
                criterion = nn.MSELoss()
            else:
                criterion = get_loss_function('pinball', quantile_levels=levels_for_pinball_list)
                logger.info(f"Using PinballLoss with quantiles: {levels_for_pinball_list}")
                logger.debug(
                    "PinballLoss activated | quantiles=%s | c_out_model=%s | c_out_eval=%s",
                    levels_for_pinball_list,
                    self.dm.c_out_model,
                    self.dm.c_out_evaluation,
                )

        elif loss_name == 'quantile' and not is_multi_quantile_scenario:
            if not (isinstance(single_q_val, float) and 0 < single_q_val < 1):
                logger.warning(f"Invalid single quantile value: {single_q_val}. Defaulting to 0.5.")
                single_q_val = 0.5
            criterion = get_loss_function('quantile', quantile=single_q_val)
            logger.info(f"Using QuantileLoss (single) with quantile: {single_q_val}")
        
        elif loss_name == 'ps_loss':
            criterion = get_loss_function(loss_name, 
                                        pred_len=self.args.pred_len,
                                        mse_weight=getattr(self.args, 'ps_mse_weight', 0.5),
                                        w_corr=getattr(self.args, 'ps_w_corr', 1.0),
                                        w_var=getattr(self.args, 'ps_w_var', 1.0),
                                        w_mean=getattr(self.args, 'ps_w_mean', 1.0))
            logger.info("Using PS_Loss")
        elif loss_name == 'huber':
            delta = getattr(self.args, 'huber_delta', 1.0)
            criterion = get_loss_function(loss_name, delta=delta)
            logger.info(f"Using HuberLoss with delta: {delta}")
        elif loss_name == 'multiscale_trend_aware':
            criterion = get_loss_function(
                loss_name,
                trend_window_sizes=getattr(self.args, 'trend_window_sizes', [60, 20, 5]),
                trend_component_weights=getattr(self.args, 'trend_component_weights', [1.0, 0.8, 0.5]),
                noise_component_weight=getattr(self.args, 'noise_component_weight', 0.2),
                base_loss_fn_str=getattr(self.args, 'base_loss_fn_str', 'mse')
            )
            logger.info(f"Using MultiScaleTrendAwareLoss with base: {getattr(self.args, 'base_loss_fn_str', 'mse')}")

        if criterion is None:
            try:
                criterion = get_loss_function(loss_name)
                logger.info(f"Using standard loss: {loss_name}")
            except ValueError:
                logger.warning(f"Unknown or misconfigured loss function: {loss_name}. Falling back to MSE.")
                criterion = nn.MSELoss()
        
        if hasattr(self.model, 'configure_optimizer_loss') and callable(getattr(self.model, 'configure_optimizer_loss')):
            logger.info(f"Model {self.args.model} has configure_optimizer_loss. Wrapping base criterion.")
            return self.model.configure_optimizer_loss(criterion, verbose=getattr(self.args, 'verbose_loss', False))
        
        logger.debug("Selected criterion: %s", type(criterion).__name__)
        return criterion

    @staticmethod
    def _extract_auxiliary_loss(candidate: object) -> Tuple[Optional[float], Optional[torch.Tensor]]:
        """Distinguish scalar auxiliary loss terms from structured tensor outputs."""
        if isinstance(candidate, torch.Tensor):
            if candidate.numel() == 1:
                return float(candidate.item()), None
            return None, candidate
        if isinstance(candidate, (int, float)):
            return float(candidate), None
        return None, None

    def vali(self, vali_loader, criterion):  # type: ignore[override]
        logger.info("Running validation phase")
        total_loss = []
        self.model.eval()
        if self.memory_diagnostics is not None:
            self.memory_diagnostics.snapshot(
                "validation_start",
                {
                    "len_vali_loader": len(vali_loader) if hasattr(vali_loader, "__len__") else None,
                    "criterion": type(criterion).__name__,
                },
            )
        
        # Check for empty loader
        if hasattr(vali_loader, "__len__") and len(vali_loader) == 0:
            logger.warning("Validation loader is empty. Skipping validation and returning infinity.")
            return float('inf')

        with torch.no_grad(): # Use self.vali_loader directly
            for i, batch_data in enumerate(vali_loader):
                if self.args.use_future_celestial_conditioning:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, future_celestial_x, _future_celestial_mark = batch_data
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data
                    future_celestial_x = None
                batch_x = batch_x.float().to(self.device)
                # batch_y has unscaled targets, scaled covariates (if M/MS)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Decoder input construction
                # batch_y contains ALL features (targets + covariates) in original scale
                total_features_in_batch_y = batch_y.shape[-1]
                c_out_evaluation = self.eval_info['c_out_evaluation'] # Number of actual targets (e.g., 4 for OHLC)

                # Historical part of decoder input (label_len)
                # Targets: scale using target_scaler
                hist_targets_unscaled = batch_y[:, :self.args.label_len, :c_out_evaluation].cpu().numpy()
                hist_targets_scaled = self.scaler_manager.target_scaler.transform(hist_targets_unscaled.reshape(-1, c_out_evaluation)).reshape(hist_targets_unscaled.shape)
                
                # Covariates: scale using main scaler (if present)
                hist_covariates_scaled = None
                if total_features_in_batch_y > c_out_evaluation and self.scaler_manager.scaler: # Check if covariates exist
                    hist_covariates_unscaled = batch_y[:, :self.args.label_len, c_out_evaluation:].cpu().numpy()
                    hist_covariates_scaled = self.scaler_manager.scaler.transform(hist_covariates_unscaled.reshape(-1, total_features_in_batch_y - c_out_evaluation)).reshape(hist_covariates_unscaled.shape)
                
                # Future part of decoder input (pred_len)
                # Targets: zero out
                future_targets_zeros = torch.zeros_like(batch_y[:, -self.args.pred_len:, :c_out_evaluation]).float().to(self.device)
                
                # Covariates: scale using main scaler (if present)
                future_covariates_scaled = None
                if total_features_in_batch_y > c_out_evaluation and self.scaler_manager.scaler:
                    future_covariates_unscaled = batch_y[:, -self.args.pred_len:, c_out_evaluation:].cpu().numpy()
                    future_covariates_scaled = self.scaler_manager.scaler.transform(future_covariates_unscaled.reshape(-1, total_features_in_batch_y - c_out_evaluation)).reshape(future_covariates_unscaled.shape)

                # Concatenate to form dec_inp
                dec_inp_val = self._construct_dec_inp(
                    hist_targets_scaled,
                    hist_covariates_scaled,
                    future_targets_zeros,
                    future_covariates_scaled,
                    future_celestial_data=future_celestial_x,
                )
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model in ['SOTA_Temporal_PGAT', 'Enhanced_SOTA_PGAT']:
                            wave_window = batch_x
                            outputs_raw = self.model(wave_window, dec_inp_val, batch_x_mark, batch_y_mark)
                        else:
                            outputs_raw = self.model(batch_x, batch_x_mark, dec_inp_val, batch_y_mark)
                else:
                    if self.args.model in ['SOTA_Temporal_PGAT', 'Enhanced_SOTA_PGAT']:
                        wave_window = batch_x
                        outputs_raw = self.model(wave_window, dec_inp_val, batch_x_mark, batch_y_mark)
                    else:
                        outputs_raw = self.model(batch_x, batch_x_mark, dec_inp_val, batch_y_mark)
                
                outputs_tensor_val, aux_loss_val, mdn_outputs_val = self._normalize_model_output(outputs_raw)

                if logger.isEnabledFor(10):
                    logger.debug(
                        "VAL forward | outputs_raw=%s | pred_len=%s | c_out_model=%s",
                        tuple(outputs_tensor_val.shape),
                        self.args.pred_len,
                        self.dm.c_out_model,
                    )
                
                # Prepare y_true for loss: scale the target part of batch_y_val_unscaled_targets
                y_true_targets_unscaled_val_loss = batch_y[:, -self.args.pred_len:, :c_out_evaluation].cpu().numpy()
                y_true_targets_scaled_val_loss_np = self.scaler_manager.target_scaler.transform(y_true_targets_unscaled_val_loss.reshape(-1, c_out_evaluation)).reshape(y_true_targets_unscaled_val_loss.shape)
                y_true_for_loss_val = torch.from_numpy(y_true_targets_scaled_val_loss_np).float().to(self.device)

                # Prepare y_pred for loss
                is_criterion_pinball = isinstance(criterion, PinballLoss) or \
                                       (hasattr(criterion, '_is_pinball_loss_based') and criterion._is_pinball_loss_based)

                if isinstance(criterion, MixtureNLLLoss) or mdn_outputs_val is not None:
                    if mdn_outputs_val is None:
                        raise ValueError("MixtureNLLLoss requires model to return a (means, stds, weights) tuple.")
                    means_v, stds_v, weights_v = mdn_outputs_val
                    if means_v.size(1) > self.args.pred_len:
                        means_v = means_v[:, -self.args.pred_len:, ...]
                        stds_v = stds_v[:, -self.args.pred_len:, ...]
                        weights_v = weights_v[:, -self.args.pred_len:, ...]
                    targets_v = y_true_for_loss_val.squeeze(-1) if y_true_for_loss_val.dim() == 3 and y_true_for_loss_val.size(-1) == 1 else y_true_for_loss_val
                    loss = criterion((means_v, stds_v, weights_v), targets_v)
                else:
                    if is_criterion_pinball:
                        y_pred_for_loss_val = outputs_tensor_val[:, -self.args.pred_len:, :]
                    else:
                        y_pred_for_loss_val = outputs_tensor_val[:, -self.args.pred_len:, :c_out_evaluation]
                    loss = criterion(y_pred_for_loss_val, y_true_for_loss_val)
                if logger.isEnabledFor(10):
                    logger.debug(
                        "VAL loss | criterion=%s | y_true=%s | pinball=%s | mdn=%s",
                        type(criterion).__name__,
                        tuple(y_true_for_loss_val.shape),
                        is_criterion_pinball,
                        isinstance(criterion, MixtureNLLLoss) or (mdn_outputs_val is not None),
                    )
                total_loss.append(loss.item())
        
        total_loss_avg = np.average(total_loss)
        self.model.train()
        if self.memory_diagnostics is not None:
            self.memory_diagnostics.snapshot(
                "validation_complete",
                {
                    "len_vali_loader": len(vali_loader) if hasattr(vali_loader, "__len__") else None,
                    "loss": float(total_loss_avg),
                },
            )
        return total_loss_avg

    def train(self, setting):  # type: ignore[override]
        print(f"DEBUG: Entered Exp_Long_Term_Forecast.train with setting: {setting}", flush=True)
        logger.info(f"Starting training with setting: {setting}")
        # Data loaders are passed in during initialization
        train_loader = self.train_loader
        vali_loader = self.vali_loader # Used in self.vali()
        test_loader = self.test_loader # Used in self.vali()

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        warmup_epochs = getattr(self.args, 'warmup_epochs', 0)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, warmup_epochs=warmup_epochs)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scaler_amp: Optional[torch.cuda.amp.GradScaler]
        scaler_amp = None
        if self.args.use_amp:
            scaler_amp = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())  # type: ignore[call-arg]

        if self.memory_diagnostics is not None:
            self.memory_diagnostics.snapshot(
                "train_loop_start",
                {
                    "setting": setting,
                    "train_steps": train_steps,
                    "epochs": self.args.train_epochs,
                },
            )

        epochs_completed = 0

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss_epoch_list = []

            self.model.train()
            
            # Feature: Support model-specific epoch updates (e.g. for stochastic warmup)
            if hasattr(self.model, 'set_current_epoch'):
                self.model.set_current_epoch(epoch)
            elif hasattr(self.model, 'module') and hasattr(self.model.module, 'set_current_epoch'):
                self.model.module.set_current_epoch(epoch)
                
            # Update curriculum at the start of the epoch
            if self.curriculum:
                self.curriculum.update_epoch(epoch)
                # Optional: log stats
                if epoch == 0: # Log once per epoch, not per batch
                    stats = self.curriculum.get_stats()
                    logger.info(f"Curriculum Stats: {stats}")

            epoch_time = time.time()
            print(f"\n=== Starting Epoch {epoch + 1}/{self.args.train_epochs} ===")
            print(f"Expected training steps: {train_steps}")

            if self.memory_diagnostics is not None:
                self.memory_diagnostics.snapshot(
                    "epoch_begin",
                    {
                        "epoch": epoch + 1,
                        "train_steps": train_steps,
                    },
                )
            
            for i, batch_data in enumerate(train_loader):
                if self.args.use_future_celestial_conditioning:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, future_celestial_x, future_celestial_mark = batch_data
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data
                    future_celestial_x = None  # Placeholder
                    future_celestial_mark = None  # Placeholder
                iter_count += 1
                model_optim.zero_grad()
                # batch_x is already scaled by ForecastingDataset
                batch_x = batch_x.float().to(self.device)
                # batch_y_unscaled_all_features is UNCALED from ForecastingDataset
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Decoder input construction for training
                # batch_y contains ALL features (targets + covariates) in original scale
                total_features_in_batch_y = batch_y.shape[-1]
                c_out_evaluation_train = self.eval_info['c_out_evaluation']  # Number of actual targets

                hist_targets_unscaled_train = batch_y[:, :self.args.label_len, :c_out_evaluation_train]
                hist_targets_scaled_train = self.scaler_manager.scale_targets_tensor(
                    hist_targets_unscaled_train, device=self.device
                )

                hist_covariates_scaled_train: Optional[torch.Tensor] = None
                if total_features_in_batch_y > c_out_evaluation_train and self.scaler_manager.has_covariate_scaler():
                    hist_covariates_unscaled_train = batch_y[:, :self.args.label_len, c_out_evaluation_train:]
                    hist_covariates_scaled_train = self.scaler_manager.scale_covariates_tensor(
                        hist_covariates_unscaled_train, device=self.device
                    )

                # Future part of decoder input (pred_len) - zero targets, real covariates
                future_targets_zeros_train = hist_targets_unscaled_train.new_zeros(
                    (batch_y.size(0), self.args.pred_len, c_out_evaluation_train)
                )

                future_covariates_scaled_train: Optional[torch.Tensor] = None
                if total_features_in_batch_y > c_out_evaluation_train and self.scaler_manager.has_covariate_scaler():
                    future_covariates_unscaled_train = batch_y[:, -self.args.pred_len:, c_out_evaluation_train:]
                    future_covariates_scaled_train = self.scaler_manager.scale_covariates_tensor(
                        future_covariates_unscaled_train, device=self.device
                    )

                dec_inp = self._construct_dec_inp(
                    hist_targets_scaled_train,
                    hist_covariates_scaled_train,
                    future_targets_zeros_train,
                    future_covariates_scaled_train,
                    future_celestial_data=future_celestial_x, # Pass future_celestial_x here
                )

                outputs_raw_train = None # Initialize
                if self.args.use_amp: # This branch is for AMP (Automatic Mixed Precision)
                    with torch.cuda.amp.autocast():
                        if self.args.model == 'Celestial_Enhanced_PGAT':
                             outputs_raw_train = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, future_celestial_x=future_celestial_x)
                        elif self.args.model in ['SOTA_Temporal_PGAT', 'Enhanced_SOTA_PGAT']:
                            # For PGAT models, the decoder input (dec_inp) is passed as the third argument.
                            # batch_y_mark is for time features, passed as fourth argument.
                            # wave_window is batch_x
                            wave_window = batch_x
                            # target_window is now dec_inp, which contains historical targets + future covariates
                            target_window = dec_inp # dec_inp now correctly has future covariates
                            outputs_raw_train = self.model(wave_window, target_window, batch_x_mark, batch_y_mark, future_celestial_x=future_celestial_x)
                        else:
                            outputs_raw_train = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.model == 'Celestial_Enhanced_PGAT':
                         outputs_raw_train = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, future_celestial_x=future_celestial_x)
                    elif self.args.model in ['SOTA_Temporal_PGAT', 'Enhanced_SOTA_PGAT', 'ModularAutoformer']:
                        # For PGAT models, the decoder input (dec_inp) is passed as the third argument.
                        # batch_y_mark is for time features, passed as fourth argument.
                        # wave_window is batch_x
                        wave_window = batch_x
                        # target_window is now dec_inp, which contains historical targets + future covariates
                        target_window = dec_inp # dec_inp now correctly has future covariates
                        outputs_raw_train = self.model(wave_window, target_window, batch_x_mark, batch_y_mark, future_celestial_x=future_celestial_x)
                    else:
                        outputs_raw_train = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # Prepare y_true for loss: scale the target part of batch_y
                y_true_targets_unscaled_train_loss = batch_y[:, -self.args.pred_len:, :c_out_evaluation_train]
                y_true_for_loss_train = self.scaler_manager.scale_targets_tensor(
                    y_true_targets_unscaled_train_loss, device=self.device
                )

                # --- Warmup Logic ---
                # Check for warmup_epochs in args, default to 0 if not present
                warmup_epochs = getattr(self.args, 'warmup_epochs', 0)
                if warmup_epochs > 0 and epoch < warmup_epochs:
                    # Linear warmup from 0.1 * lr to 1.0 * lr
                    warmup_progress = (epoch + i / len(train_loader)) / warmup_epochs
                    warmup_factor = 0.1 + 0.9 * warmup_progress
                    warmup_factor = min(1.0, warmup_factor)
                    
                    # Apply to optimizer
                    current_lr = self.args.learning_rate * warmup_factor
                    for param_group in model_optim.param_groups:
                        param_group['lr'] = current_lr
                    
                    if i % 100 == 0: # minimal logging
                         logger.debug(f"Warmup: Factor {warmup_factor:.4f}, LR {current_lr:.6f}")

                # --- Curriculum Learning & Loss Calculation ---
                curriculum_mask = None
                if self.curriculum:
                    curriculum_mask = self.curriculum.get_mask(self.args.pred_len)
                    self.curriculum.on_batch_start(batch_x, batch_y) # Optional hook

                # Use model's internal loss logic if available (Modular Approach)
                if hasattr(self.model, 'compute_loss'):
                     loss_train = self.model.compute_loss(
                         outputs_raw_train, 
                         y_true_for_loss_train, 
                         criterion, 
                         curriculum_mask=curriculum_mask,
                         logger=logger
                     )
                else:
                    # Fallback for non-modular models
                    outputs_tensor_train, aux_loss_train, mdn_outputs_train = self._normalize_model_output(outputs_raw_train)
                    if logger.isEnabledFor(10):
                        logger.debug(
                            "TRAIN forward | outputs_raw=%s | pred_len=%s | c_out_model=%s",
                            tuple(outputs_tensor_train.shape),
                            self.args.pred_len,
                            self.dm.c_out_model,
                        )
                    
                    is_criterion_pinball_train = isinstance(criterion, PinballLoss) or \
                                                 (hasattr(criterion, '_is_pinball_loss_based') and criterion._is_pinball_loss_based)
                    
                    if isinstance(criterion, MixtureNLLLoss) or mdn_outputs_train is not None:
                        if mdn_outputs_train is None:
                            raise ValueError("MixtureNLLLoss requires model to return a (means, stds, weights) tuple during training.")
                        means_t, stds_t, weights_t = mdn_outputs_train
                        if means_t.size(1) > self.args.pred_len:
                            means_t = means_t[:, -self.args.pred_len:, ...]
                            stds_t = stds_t[:, -self.args.pred_len:, ...]
                            weights_t = weights_t[:, -self.args.pred_len:, ...]
                        targets_t = y_true_for_loss_train.squeeze(-1) if y_true_for_loss_train.dim() == 3 and y_true_for_loss_train.size(-1) == 1 else y_true_for_loss_train
                        
                        if curriculum_mask is not None:
                            eff_len = int(curriculum_mask.sum().item())
                            means_t = means_t[:, :eff_len, ...]
                            stds_t = stds_t[:, :eff_len, ...]
                            weights_t = weights_t[:, :eff_len, ...]
                            targets_t = targets_t[:, :eff_len, ...]
                            
                        loss_train = criterion((means_t, stds_t, weights_t), targets_t)
                    else:
                        if is_criterion_pinball_train:
                            y_pred_for_loss_train = outputs_tensor_train[:, -self.args.pred_len:, :]
                        else:
                            y_pred_for_loss_train = outputs_tensor_train[:, -self.args.pred_len:, :c_out_evaluation_train]
                        
                        if curriculum_mask is not None:
                            eff_len = int(curriculum_mask.sum().item())
                            y_pred_for_loss_train = y_pred_for_loss_train[:, :eff_len, :]
                            y_true_for_loss_train = y_true_for_loss_train[:, :eff_len, :]

                        loss_train = criterion(y_pred_for_loss_train, y_true_for_loss_train)
                    
                    # Add auxiliary loss if present
                    if aux_loss_train:
                        loss_train = loss_train + aux_loss_train

                if logger.isEnabledFor(10):
                    logger.debug(
                        "TRAIN loss | criterion=%s | y_true=%s | pinball=%s | mdn=%s",
                        type(criterion).__name__,
                        tuple(y_true_for_loss_train.shape),
                        is_criterion_pinball_train,
                        isinstance(criterion, MixtureNLLLoss) or (mdn_outputs_train is not None),
                    )
                

                
                train_loss_epoch_list.append(loss_train.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss_train.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                # Progress indicator every 10 iterations for more granular feedback
                if (i + 1) % 10 == 0:
                    progress_pct = ((i + 1) / train_steps) * 100
                    print(f"\t[Epoch {epoch + 1}] Progress: {i + 1}/{train_steps} ({progress_pct:.1f}%) | Current Loss: {loss_train.item():.7f}")
                
                # Debug every 50 iterations if debug is enabled
                if (i + 1) % 50 == 0 and logger.isEnabledFor(10):
                    logger.debug(f"Training Step {i + 1}: batch_x={batch_x.shape}, outputs={tuple(outputs_tensor_train.shape)}")
                    adjusted_core_loss = loss_train.item() - (aux_loss_train if aux_loss_train else 0.0)
                    logger.debug(f"  aux_loss: {aux_loss_train}, main_loss: {adjusted_core_loss:.7f}")

                if self.args.use_amp and scaler_amp is not None:
                    scaler_amp.scale(loss_train).backward()
                    
                    # Gradient Clipping with AMP
                    clip_val = getattr(self.args, 'gradient_clip_val', 0)
                    if clip_val > 0:
                        scaler_amp.unscale_(model_optim)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_val)
                        
                    scaler_amp.step(model_optim)
                    scaler_amp.update()
                else:
                    loss_train.backward()
                    
                    # Gradient Clipping
                    clip_val = getattr(self.args, 'gradient_clip_val', 0)
                    if clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_val)
                        
                    # Deep Scan: Log Gradients
                    if getattr(self.args, 'collect_diagnostics', False) and (i + 1) % 100 == 0:
                         grad_stats = {}
                         if hasattr(self.model, 'log_gradients'):
                             grad_stats = self.model.log_gradients()
                         elif hasattr(self.model, 'module') and hasattr(self.model.module, 'log_gradients'):
                             grad_stats = self.model.module.log_gradients()
                         
                         if grad_stats:
                             logger.info(f"Step {i+1} Gradients: Total={grad_stats.get('grad_norm/total', 0):.4f}, Max={grad_stats.get('grad_norm/max', 0):.4f}")

                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss_avg = np.average(train_loss_epoch_list)
            
            # Clear cache to prevent OOM during validation (critical for deep scan)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion) # Using vali for test as placeholder

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss_avg, vali_loss, test_loss))
            if self.memory_diagnostics is not None:
                self.memory_diagnostics.snapshot(
                    "epoch_end",
                    {
                        "epoch": epoch + 1,
                        "train_loss": float(train_loss_avg),
                        "vali_loss": float(vali_loss),
                        "test_loss": float(test_loss),
                    },
                )
            epochs_completed = epoch + 1
            early_stopping(vali_loss, self.model, path, epoch)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        if self.memory_diagnostics is not None:
            self.memory_diagnostics.snapshot(
                "train_complete",
                {
                    "setting": setting,
                    "epochs_trained": epochs_completed,
                    "train_steps": train_steps,
                },
            )

        return self.model

    def test(self, setting: str, test: int = 0) -> None:  # type: ignore[override]
        logger.info(f"Starting test with setting: {setting}, test={test}")
        test_loader = self.test_loader # Use pre-stored test_loader from init
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        if self.memory_diagnostics is not None:
            self.memory_diagnostics.snapshot(
                "test_start",
                {
                    "setting": setting,
                    "len_test_loader": len(test_loader) if hasattr(test_loader, "__len__") else None,
                },
            )

        preds_scaled_np = []
        trues_original_np_targets_only = [] # Store original scale targets for direct comparison if needed
        
        # For visualization, store original unscaled targets if possible
        trues_original_for_viz_np = [] 
        
        c_out_evaluation_test = self.eval_info['c_out_evaluation']
        num_quantiles_test = self.eval_info['num_quantiles']
        is_quantile_mode_test = num_quantiles_test > 1 and self.eval_info['loss_name'] == 'pinball'
        median_quantile_index = -1
        if is_quantile_mode_test:
            try:
                median_quantile_index = self.eval_info['quantile_levels'].index(0.5)
            except (ValueError, AttributeError):
                logger.warning("0.5 quantile not found or quantile_levels not set. Using middle index for median.")
                median_quantile_index = num_quantiles_test // 2


        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, batch_data in enumerate(test_loader):
                if self.args.use_future_celestial_conditioning:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, future_celestial_x, _future_celestial_mark = batch_data
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data
                    future_celestial_x = None
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Decoder input
                # batch_y_unscaled_all_features contains ALL features (targets + covariates) in original scale
                total_features_in_batch_y = batch_y.shape[-1]
                # Use evaluation c_out (number of target features) from eval_info defined above
                # c_out_evaluation_test is already set from self.eval_info['c_out_evaluation']

                # Historical (label_len) targets and covariates for decoder input
                hist_targets_unscaled = batch_y[:, :self.args.label_len, :c_out_evaluation_test].cpu().numpy()
                hist_targets_scaled = self.scaler_manager.target_scaler.transform(
                    hist_targets_unscaled.reshape(-1, c_out_evaluation_test)
                ).reshape(hist_targets_unscaled.shape)

                hist_covariates_scaled = None
                if total_features_in_batch_y > c_out_evaluation_test and self.scaler_manager.scaler:
                    hist_covariates_unscaled = batch_y[:, :self.args.label_len, c_out_evaluation_test:].cpu().numpy()
                    hist_covariates_scaled = self.scaler_manager.scaler.transform(
                        hist_covariates_unscaled.reshape(-1, total_features_in_batch_y - c_out_evaluation_test)
                    ).reshape(hist_covariates_unscaled.shape)

                # Create zeros for future targets to build decoder input for inference
                future_targets_zeros = torch.zeros_like(batch_y[:, -self.args.pred_len:, :c_out_evaluation_test]).float().to(self.device)

                # Extract future covariates (unscaled) from batch_y which includes all features
                future_covariates_scaled = None
                if total_features_in_batch_y > c_out_evaluation_test and self.scaler_manager.scaler:
                    future_covariates_unscaled = batch_y[:, -self.args.pred_len:, c_out_evaluation_test:].cpu().numpy()
                    future_covariates_scaled = self.scaler_manager.scaler.transform(
                        future_covariates_unscaled.reshape(-1, total_features_in_batch_y - c_out_evaluation_test)
                    ).reshape(future_covariates_unscaled.shape)

                # Construct decoder input with historical (scaled) and future parts
                dec_inp_test = self._construct_dec_inp(
                    hist_targets_scaled,
                    hist_covariates_scaled,
                    future_targets_zeros,
                    future_covariates_scaled,
                    future_celestial_data=future_celestial_x,
                )
                
                outputs_raw_test = None  # Initialize
                if self.args.use_amp:  # This branch is for AMP (Automatic Mixed Precision)
                    with torch.cuda.amp.autocast():
                        if self.args.model in ['SOTA_Temporal_PGAT', 'Enhanced_SOTA_PGAT']:
                            wave_window = batch_x
                            outputs_raw_test = self.model(wave_window, dec_inp_test, batch_x_mark, batch_y_mark)
                        else:
                            outputs_raw_test = self.model(batch_x, batch_x_mark, dec_inp_test, batch_y_mark)
                else:
                    if self.args.model in ['SOTA_Temporal_PGAT', 'Enhanced_SOTA_PGAT']:
                        wave_window = batch_x
                        outputs_raw_test = self.model(wave_window, dec_inp_test, batch_x_mark, batch_y_mark)
                    else:
                        outputs_raw_test = self.model(batch_x, batch_x_mark, dec_inp_test, batch_y_mark)

                if outputs_raw_test is None:
                    raise ValueError("Model forward pass returned None during testing.")

                outputs_tensor_test, aux_loss_test, mdn_outputs_test = self._normalize_model_output(outputs_raw_test)
                if aux_loss_test:
                    logger.debug("TEST auxiliary loss detected: %s", aux_loss_test)

                if mdn_outputs_test is not None:
                    means_te, stds_te, weights_te = mdn_outputs_test
                    if means_te.size(1) > self.args.pred_len:
                        means_te = means_te[:, -self.args.pred_len:, ...]
                        stds_te = stds_te[:, -self.args.pred_len:, ...]
                        weights_te = weights_te[:, -self.args.pred_len:, ...]
                    
                    # Handle both univariate and multivariate mixture outputs
                    if means_te.dim() == 4:  # Multivariate: [batch, pred_len, targets, components]
                        # Compute mixture mean for each target separately
                        # weights_te: [batch, pred_len, components] -> [batch, pred_len, 1, components]
                        weights_expanded = weights_te.unsqueeze(2)  # Add target dimension
                        # Compute weighted mean: [batch, pred_len, targets, components] -> [batch, pred_len, targets]
                        model_outputs_pred_len_segment = (weights_expanded * means_te).sum(dim=-1)
                    else:  # Univariate: [batch, pred_len, components]
                        # Use mixture mean as point prediction for evaluation
                        model_outputs_pred_len_segment = (weights_te * means_te).sum(dim=-1).unsqueeze(-1)
                else:
                    model_outputs_pred_len_segment = outputs_tensor_test[:, -self.args.pred_len:, :]

                if logger.isEnabledFor(10):
                    logger.debug(
                        "TEST forward | outputs_tensor=%s | pred_len_segment=%s | c_out_model=%s | quantile_mode=%s | mdn=%s",
                        tuple(outputs_tensor_test.shape),
                        tuple(model_outputs_pred_len_segment.shape),
                        self.dm.c_out_model,
                        is_quantile_mode_test,
                        mdn_outputs_test is not None,
                    )
                
                # Store original unscaled targets (only the target columns) for direct comparison/saving
                true_targets_original_batch_np = batch_y[:, -self.args.pred_len:, :c_out_evaluation_test].cpu().numpy()
                trues_original_np_targets_only.append(true_targets_original_batch_np)

                # For visualization, store the full c_out_evaluation features in original scale
                trues_original_for_viz_np.append(batch_y[:, -self.args.pred_len:, :c_out_evaluation_test].cpu().numpy())

                # Extract point predictions (median if quantile) for metrics, scaled
                pred_point_scaled_batch_np = None # Initialize 
                if is_quantile_mode_test and model_outputs_pred_len_segment.shape[-1] == c_out_evaluation_test * num_quantiles_test:
                    pred_point_scaled_batch_np = model_outputs_pred_len_segment.view(
                        model_outputs_pred_len_segment.shape[0], self.args.pred_len, c_out_evaluation_test, num_quantiles_test
                    )[:, :, :, median_quantile_index].detach().cpu().numpy()
                else: # Standard point prediction or model output already c_out_evaluation
                    pred_point_scaled_batch_np = model_outputs_pred_len_segment[:, :, :c_out_evaluation_test].detach().cpu().numpy()
                
                preds_scaled_np.append(pred_point_scaled_batch_np)

                if i % 20 == 0 and c_out_evaluation_test > 0:
                    input_np = batch_x.detach().cpu().numpy()
                    
                    pred_for_viz_scaled = pred_point_scaled_batch_np[0] # First sample in batch, scaled point preds
                    true_for_viz_original = trues_original_for_viz_np[-1][0] # Corresponding original true values (all c_out_eval features)

                    scale_flag = getattr(self.args, 'scale', True)
                    logger.debug(f"Viz check: scale_flag={scale_flag}")
                    logger.debug(f"Viz check: self.scaler_manager is not None={self.scaler_manager is not None}")
                    logger.debug(f"Viz check: self.scaler_manager.target_scaler is not None={self.scaler_manager.target_scaler is not None if self.scaler_manager else 'N/A'}")

                    # Check if scaling is enabled and scaler manager is properly initialized
                    if scale_flag and self.scaler_manager and self.scaler_manager.target_scaler:
                        pred_for_viz_original = self.scaler_manager.inverse_transform_targets(
                            pred_for_viz_scaled.reshape(-1, c_out_evaluation_test)
                        ).reshape(pred_for_viz_scaled.shape) # Use target_scaler
                        
                        total_features = self.dm.enc_in # enc_in is the total number of features (targets + covariates)
                        input_for_viz_original = self.scaler_manager.inverse_transform_all_features(
                            input_np[0].reshape(-1, total_features)
                        ).reshape(input_np[0].shape)
                        
                        # Visualize the first target feature (index 0 of c_out_evaluation features)
                        if input_for_viz_original.shape[-1] > 0 and true_for_viz_original.shape[-1] > 0 and pred_for_viz_original.shape[-1] > 0: # Ensure dimensions are valid for plotting
                            gt_plot = np.concatenate((input_for_viz_original[:, 0], true_for_viz_original[:, 0]), axis=0)
                            pd_plot = np.concatenate((input_for_viz_original[:, 0], pred_for_viz_original[:, 0]), axis=0)
                            visual(gt_plot, pd_plot, os.path.join(folder_path, f"{i}.pdf"))
                        else: # This branch is taken if the shapes are invalid for plotting (e.g., 0-dimension)
                            logger.warning("Cannot visualize: insufficient target features after inverse transform.") 
                    else:
                        logger.warning("Cannot visualize in original scale: inverse_transform not available or scale=False.")


        # If no test windows were yielded (e.g., tiny dataset with seq_len+pred_len > test split),
        # skip concatenation and metrics gracefully.
        if len(preds_scaled_np) == 0:
            logger.warning(
                "No test samples available for evaluation (test split too short for given seq_len+pred_len). "
                "Skipping metrics and result saving."
            )
            return

        preds_final_scaled = np.concatenate(preds_scaled_np, axis=0)
        trues_final_original_targets_only = np.concatenate(trues_original_np_targets_only, axis=0)
        
        # Inverse transform predictions to original scale for final metrics using target_scaler
        preds_final_original = self.scaler_manager.inverse_transform_targets(preds_final_scaled.reshape(-1, c_out_evaluation_test)).reshape(preds_final_scaled.shape)
        
        # Metrics are calculated on original scale data (targets only)
        # Ensure both preds and trues for metric calculation are [samples, pred_len, num_target_cols]
        # target_cols from eval_info defines which columns are the true targets
        num_true_target_cols = len(self.eval_info['target_columns'])
        
        preds_for_metric = preds_final_original[:, :, :num_true_target_cols]
        trues_for_metric = trues_final_original_targets_only[:, :, :num_true_target_cols]

        logger.info(f'Test shape (original scale, targets only): preds={preds_for_metric.shape}, trues={trues_for_metric.shape}')

        # Explicitly log whether quantile path was active during test
        logger.debug(
            "TEST summary | loss_name=%s | quantiles=%s | num_quantiles=%s | median_index=%s",
            self.eval_info['loss_name'],
            self.eval_info['quantile_levels'],
            num_quantiles_test,
            median_quantile_index,
        )

        mae, mse, rmse, mape, mspe = metric(preds_for_metric, trues_for_metric)
        logger.info('Metrics (original scale, targets only): mse:{}, mae:{}, rmse:{}'.format(mse, mae, rmse))

        dtw_val = 'Not calculated' # Placeholder for DTW

        with open(os.path.join(folder_path, "result_long_term_forecast.txt"), 'a') as f:
            f.write(setting + "  \n")
            f.write('Metrics (original scale, targets only): mse:{}, mae:{}, rmse:{}, dtw:{}'.format(mse, mae, rmse, dtw_val))
            f.write('\n')
            f.write('\n')

        np.save(os.path.join(folder_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(folder_path, 'pred_original.npy'), preds_final_original) # Save original scale predictions (all c_out_eval)
        np.save(os.path.join(folder_path, 'true_original_targets.npy'), trues_final_original_targets_only) # Save original scale true targets

        if self.memory_diagnostics is not None:
            self.memory_diagnostics.snapshot(
                "test_complete",
                {
                    "setting": setting,
                    "mse": float(mse),
                    "mae": float(mae),
                    "rmse": float(rmse),
                },
            )

        return

    @staticmethod
    def _normalize_model_output(
        raw_output: Union[torch.Tensor, Sequence[torch.Tensor]],
    ) -> Tuple[torch.Tensor, float, Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """Convert mixed forward outputs into tensor, scalar aux loss, and optional MDN tuple."""
        aux_loss = 0.0
        mdn_tuple: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        output_tensor: Union[torch.Tensor, Sequence[torch.Tensor]] = raw_output

        if isinstance(raw_output, (tuple, list)):
            if len(raw_output) == 3:
                output_tensor = raw_output[0]
                
                # Handle 2nd element as aux loss
                scalar_aux, _ = Exp_Long_Term_Forecast._extract_auxiliary_loss(raw_output[1])
                if scalar_aux is not None:
                    aux_loss = scalar_aux
                    
                # Handle 3rd element
                if isinstance(raw_output[2], tuple):
                    mdn_tuple = raw_output[2]
                elif isinstance(raw_output[2], torch.Tensor) and all(isinstance(part, torch.Tensor) for part in raw_output):
                    # Legacy case: 3 tensors? Unclear use case, but preserving structure if needed
                    # Actually, the original code assumed 3 tensors were (pi, mu, sigma) for MDN only if returned directly?
                    pass

            elif len(raw_output) == 2:
                primary, secondary = raw_output
                scalar_aux, _ = Exp_Long_Term_Forecast._extract_auxiliary_loss(secondary)
                output_tensor = primary
                if scalar_aux is not None:
                    aux_loss = scalar_aux
            elif len(raw_output) >= 1:
                output_tensor = raw_output[0]

        if not isinstance(output_tensor, torch.Tensor):
            raise TypeError("Model forward pass must yield a tensor as primary output.")

        return output_tensor, aux_loss, mdn_tuple

    def _construct_dec_inp(
        self,
        hist_targets_scaled: Union[torch.Tensor, np.ndarray],
        hist_covariates_scaled: Optional[Union[torch.Tensor, np.ndarray]],
        future_targets_zeros: torch.Tensor,
        future_covariates_scaled: Optional[Union[torch.Tensor, np.ndarray]],
        future_celestial_data: Optional[Union[torch.Tensor, np.ndarray]] = None, # NEW: Add future celestial data
    ) -> torch.Tensor:
        """Construct decoder input without leaving the torch tensor domain."""

        # Convert all inputs to tensors and move to device
        hist_targets_scaled_t = hist_targets_scaled
        if isinstance(hist_targets_scaled_t, np.ndarray):
            hist_targets_scaled_t = torch.from_numpy(hist_targets_scaled_t).float()
        hist_targets_scaled_t = hist_targets_scaled_t.to(self.device)
            
        future_targets_zeros_t = future_targets_zeros.to(self.device)

        dec_inp_hist: torch.Tensor = hist_targets_scaled_t
        dec_inp_future: torch.Tensor = future_targets_zeros_t

        if self.args.features in ['M', 'MS']:
            # Handle historical covariates
            if hist_covariates_scaled is not None:
                hist_covariates_scaled_t = hist_covariates_scaled
                if isinstance(hist_covariates_scaled_t, np.ndarray):
                    hist_covariates_scaled_t = torch.from_numpy(hist_covariates_scaled_t).float()
                hist_covariates_scaled_t = hist_covariates_scaled_t.to(self.device)
                dec_inp_hist = torch.cat([hist_targets_scaled_t, hist_covariates_scaled_t], dim=-1)
            else:
                logger.warning(
                    "Historical covariates expected for M/MS mode but not provided to _construct_dec_inp. Proceeding with targets only for historical input."
                )

            # Handle future covariates (combining regular future_covariates and future_celestial_data)
            combined_future_covariates: Optional[torch.Tensor] = None
            if future_covariates_scaled is not None:
                future_covariates_scaled_t = future_covariates_scaled
                if isinstance(future_covariates_scaled_t, np.ndarray):
                    future_covariates_scaled_t = torch.from_numpy(future_covariates_scaled_t).float()
                future_covariates_scaled_t = future_covariates_scaled_t.to(self.device)
                combined_future_covariates = future_covariates_scaled_t
            
            if future_celestial_data is not None:
                future_celestial_data_t = future_celestial_data
                if isinstance(future_celestial_data_t, np.ndarray):
                    future_celestial_data_t = torch.from_numpy(future_celestial_data_t).float()
                future_celestial_data_t = future_celestial_data_t.to(self.device)
                if combined_future_covariates is not None:
                    # Concatenate if both exist
                    combined_future_covariates = torch.cat([combined_future_covariates, future_celestial_data_t], dim=-1)
                else:
                    # If only celestial data, use it as combined future covariates
                    combined_future_covariates = future_celestial_data_t

            if combined_future_covariates is not None:
                dec_inp_future = torch.cat([future_targets_zeros_t, combined_future_covariates], dim=-1)
            else:
                logger.warning(
                    "Future covariates (including celestial) expected for M/MS mode but not provided to _construct_dec_inp. Proceeding with zero targets only for future input."
                )

        elif self.args.features == 'S':
            dec_inp_hist = hist_targets_scaled_t
            dec_inp_future = future_targets_zeros_t
            if future_celestial_data is not None:
                logger.warning(
                    "Future celestial data provided for 'S' mode, but 'S' mode does not typically use covariates in decoder input. Ignoring future_celestial_data."
                )
        else:
            logger.warning(
                "Feature mode %s not explicitly handled; defaulting to targets only in decoder input.",
                self.args.features,
            )

        # Ensure feature dimensions match between hist and future
        if dec_inp_future.shape[-1] > dec_inp_hist.shape[-1]:
            diff = dec_inp_future.shape[-1] - dec_inp_hist.shape[-1]
            zeros_padding = torch.zeros(
                (dec_inp_hist.shape[0], dec_inp_hist.shape[1], diff),
                device=dec_inp_hist.device,
                dtype=dec_inp_hist.dtype
            )
            dec_inp_hist = torch.cat([dec_inp_hist, zeros_padding], dim=-1)
        elif dec_inp_hist.shape[-1] > dec_inp_future.shape[-1]:
            diff = dec_inp_hist.shape[-1] - dec_inp_future.shape[-1]
            zeros_padding = torch.zeros(
                (dec_inp_future.shape[0], dec_inp_future.shape[1], diff),
                device=dec_inp_future.device,
                dtype=dec_inp_future.dtype
            )
            dec_inp_future = torch.cat([dec_inp_future, zeros_padding], dim=-1)

        return torch.cat([dec_inp_hist, dec_inp_future], dim=1)
