from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.logger import logger
from argparse import Namespace # Import Namespace for model_init_args
from utils.dimension_manager import DimensionManager # Import DimensionManager
from utils.losses import PinballLoss # Import PinballLoss for type checking
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import inspect # For checking model constructor arguments
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        # The Exp_Long_Term_Forecast class is now initialized with pre-configured
        # DimensionManager, ScalerManager, and data loaders from train_dynamic_autoformer.py.
        # This simplifies its role to focus purely on the training/validation/testing loop.
        # The args object is still passed for general configuration.
        # The actual DM and ScalerManager are passed as separate arguments.
        logger.info(f"Initializing Exp_Long_Term_Forecast with provided managers and loaders.")
        
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
            'loss_name': self.dm.loss_function,
            'target_columns': self.dm.target_features # List of target column names
        }

        # Initialize the parent class *after* model_init_args is set.
        # Pass the resolved args to the parent, which will store it as self.args.
        super(Exp_Long_Term_Forecast, self).__init__(args)
        
        # Ensure the args object used by the parent also has the managers
        # This is important if any other part of the code expects args.scaler_manager or args.dim_manager
        self.args.dim_manager = self.dm
        self.args.scaler_manager = self.scaler_manager

    def _build_model(self):
        ModelClass = self.model_dict[self.args.model].Model
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
            levels_for_pinball = q_levels
            if not is_multi_quantile_scenario:
                levels_for_pinball = [0.1, 0.5, 0.9]
                logger.warning(f"Loss is 'pinball' but no quantile_levels provided in args. Defaulting to {levels_for_pinball}")
            
            if not all(isinstance(q, float) and 0 < q < 1 for q in levels_for_pinball):
                logger.error(f"Invalid quantile_levels for PinballLoss: {levels_for_pinball}. Must be list of floats between 0 and 1.")
                logger.warning("Falling back to MSELoss due to invalid quantile_levels for PinballLoss.")
                criterion = nn.MSELoss()
            else:
                criterion = get_loss_function('pinball', quantile_levels=levels_for_pinball)
                logger.info(f"Using PinballLoss with quantiles: {levels_for_pinball}")

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
        
        return criterion

    def vali(self, vali_loader, criterion):
        logger.info("Running validation phase")
        total_loss = []
        self.model.eval()
        with torch.no_grad(): # Use self.vali_loader directly
            for i, (batch_x, batch_y_val_unscaled_all_features, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                # batch_y_val_unscaled_targets has unscaled targets, scaled covariates (if M/MS)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Decoder input construction
                # batch_y_val_unscaled_targets contains ALL features (targets + covariates) in original scale
                total_features_in_batch_y = batch_y_val_unscaled_all_features.shape[-1]
                c_out_evaluation = self.eval_info['c_out_evaluation'] # Number of actual targets (e.g., 4 for OHLC)

                # Historical part of decoder input (label_len)
                # Targets: scale using target_scaler
                hist_targets_unscaled = batch_y_val_unscaled_all_features[:, :self.args.label_len, :c_out_evaluation].cpu().numpy()
                hist_targets_scaled = self.scaler_manager.target_scaler.transform(hist_targets_unscaled.reshape(-1, c_out_evaluation)).reshape(hist_targets_unscaled.shape)
                
                # Covariates: scale using main scaler (if present)
                hist_covariates_scaled = None
                if total_features_in_batch_y > c_out_evaluation and self.scaler_manager.scaler: # Check if covariates exist
                    hist_covariates_unscaled = batch_y_val_unscaled_all_features[:, :self.args.label_len, c_out_evaluation:].cpu().numpy()
                    hist_covariates_scaled = self.scaler_manager.scaler.transform(hist_covariates_unscaled.reshape(-1, total_features_in_batch_y - c_out_evaluation)).reshape(hist_covariates_unscaled.shape)
                
                # Future part of decoder input (pred_len)
                # Targets: zero out
                future_targets_zeros = torch.zeros_like(batch_y_val_unscaled_all_features[:, -self.args.pred_len:, :c_out_evaluation]).float().to(self.device)
                
                # Covariates: scale using main scaler (if present)
                future_covariates_scaled = None
                if total_features_in_batch_y > c_out_evaluation and self.scaler_manager.scaler:
                    future_covariates_unscaled = batch_y_val_unscaled_all_features[:, -self.args.pred_len:, c_out_evaluation:].cpu().numpy()
                    future_covariates_scaled = self.scaler_manager.scaler.transform(future_covariates_unscaled.reshape(-1, total_features_in_batch_y - c_out_evaluation)).reshape(future_covariates_unscaled.shape)

                # Concatenate to form dec_inp
                dec_inp_val = self._construct_dec_inp(hist_targets_scaled, hist_covariates_scaled, future_targets_zeros, future_covariates_scaled)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs_raw = self.model(batch_x, batch_x_mark, dec_inp_val, batch_y_mark)
                else:
                    outputs_raw = self.model(batch_x, batch_x_mark, dec_inp_val, batch_y_mark)
                
                # Prepare y_true for loss: scale the target part of batch_y_val_unscaled_targets
                y_true_targets_unscaled_val_loss = batch_y_val_unscaled_all_features[:, -self.args.pred_len:, :c_out_evaluation].cpu().numpy()
                y_true_targets_scaled_val_loss_np = self.scaler_manager.target_scaler.transform(y_true_targets_unscaled_val_loss.reshape(-1, c_out_evaluation)).reshape(y_true_targets_unscaled_val_loss.shape)
                y_true_for_loss_val = torch.from_numpy(y_true_targets_scaled_val_loss_np).float().to(self.device)

                # Prepare y_pred for loss
                is_criterion_pinball = isinstance(criterion, PinballLoss) or \
                                       (hasattr(criterion, '_is_pinball_loss_based') and criterion._is_pinball_loss_based)

                if is_criterion_pinball:
                    y_pred_for_loss_val = outputs_raw[:, -self.args.pred_len:, :] # Use full model output (c_out_model)
                else:
                    y_pred_for_loss_val = outputs_raw[:, -self.args.pred_len:, :c_out_evaluation]
                
                loss = criterion(y_pred_for_loss_val, y_true_for_loss_val)
                total_loss.append(loss.item())
        
        total_loss_avg = np.average(total_loss)
        self.model.train()
        return total_loss_avg

    def train(self, setting):
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

        if self.args.use_amp:
            scaler_amp = torch.cuda.amp.GradScaler() # type: ignore

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss_epoch_list = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y_unscaled_all_features, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                # batch_x is already scaled by ForecastingDataset
                batch_x = batch_x.float().to(self.device)
                # batch_y_unscaled_all_features is UNCALED from ForecastingDataset
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Decoder input construction for training
                # batch_y_unscaled_all_features contains ALL features (targets + covariates) in original scale
                total_features_in_batch_y = batch_y_unscaled_all_features.shape[-1]
                c_out_evaluation_train = self.eval_info['c_out_evaluation'] # Number of actual targets

                # Historical part of decoder input (label_len)
                hist_targets_unscaled_train = batch_y_unscaled_all_features[:, :self.args.label_len, :c_out_evaluation_train].cpu().numpy()
                hist_targets_scaled_train = self.scaler_manager.target_scaler.transform(hist_targets_unscaled_train.reshape(-1, c_out_evaluation_train)).reshape(hist_targets_unscaled_train.shape)
                
                hist_covariates_scaled_train = None
                if total_features_in_batch_y > c_out_evaluation_train and self.scaler_manager.scaler: # Check if covariates exist
                    hist_covariates_unscaled_train = batch_y_unscaled_all_features[:, :self.args.label_len, c_out_evaluation_train:].cpu().numpy()
                    hist_covariates_scaled_train = self.scaler_manager.scaler.transform(hist_covariates_unscaled_train.reshape(-1, total_features_in_batch_y - c_out_evaluation_train)).reshape(hist_covariates_unscaled_train.shape)
                
                # Future part of decoder input (pred_len) - zero targets, real covariates
                future_targets_zeros_train = torch.zeros_like(batch_y_unscaled_all_features[:, -self.args.pred_len:, :c_out_evaluation_train]).float().to(self.device)
                
                future_covariates_scaled_train = None
                if total_features_in_batch_y > c_out_evaluation_train and self.scaler_manager.scaler:
                    future_covariates_unscaled_train = batch_y_unscaled_all_features[:, -self.args.pred_len:, c_out_evaluation_train:].cpu().numpy()
                    future_covariates_scaled_train = self.scaler_manager.scaler.transform(future_covariates_unscaled_train.reshape(-1, total_features_in_batch_y - c_out_evaluation_train)).reshape(future_covariates_unscaled_train.shape)

                dec_inp = self._construct_dec_inp(hist_targets_scaled_train, hist_covariates_scaled_train, future_targets_zeros_train, future_covariates_scaled_train)

                outputs_raw_train = None # Initialize
                if self.args.use_amp: # This branch is for AMP (Automatic Mixed Precision)
                    with torch.cuda.amp.autocast():
                        outputs_raw_train = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs_raw_train = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # Prepare y_true for loss: scale the target part of batch_y
                y_true_targets_unscaled_train_loss = batch_y_unscaled_all_features[:, -self.args.pred_len:, :c_out_evaluation_train].cpu().numpy()
                y_true_for_loss_train = torch.from_numpy(self.scaler_manager.target_scaler.transform(y_true_targets_unscaled_train_loss.reshape(-1, c_out_evaluation_train)).reshape(y_true_targets_unscaled_train_loss.shape)).float().to(self.device)

                is_criterion_pinball_train = isinstance(criterion, PinballLoss) or \
                                             (hasattr(criterion, '_is_pinball_loss_based') and criterion._is_pinball_loss_based)
                
                y_pred_for_loss_train = None # Initialize
                if is_criterion_pinball_train:
                    y_pred_for_loss_train = outputs_raw_train[:, -self.args.pred_len:, :]
                else:
                    y_pred_for_loss_train = outputs_raw_train[:, -self.args.pred_len:, :c_out_evaluation_train]
                
                loss_train = criterion(y_pred_for_loss_train, y_true_for_loss_train)
                train_loss_epoch_list.append(loss_train.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss_train.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler_amp.scale(loss_train).backward()
                    scaler_amp.step(model_optim)
                    scaler_amp.update()
                else:
                    loss_train.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss_avg = np.average(train_loss_epoch_list)
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion) # Using vali for test as placeholder

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss_avg, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path, epoch)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        logger.info(f"Starting test with setting: {setting}, test={test}")
        test_loader = self.test_loader # Use pre-stored test_loader from init
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

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
            for i, (batch_x, batch_y_unscaled_all_features, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Decoder input
                # batch_y_unscaled_all_features contains ALL features (targets + covariates) in original scale
                total_features_in_batch_y = batch_y_unscaled_all_features.shape[-1]
                
                # Historical part of decoder input (label_len)
                hist_targets_unscaled = batch_y_unscaled_all_features[:, :self.args.label_len, :c_out_evaluation_test].cpu().numpy()
                hist_targets_scaled = self.scaler_manager.target_scaler.transform(hist_targets_unscaled.reshape(-1, c_out_evaluation_test)).reshape(hist_targets_unscaled.shape)
                
                hist_covariates_scaled = None
                if total_features_in_batch_y > c_out_evaluation_test and self.scaler_manager.scaler:
                    hist_covariates_unscaled = batch_y_unscaled_all_features[:, :self.args.label_len, c_out_evaluation_test:].cpu().numpy()
                    hist_covariates_scaled = self.scaler_manager.scaler.transform(hist_covariates_unscaled.reshape(-1, total_features_in_batch_y - c_out_evaluation_test)).reshape(hist_covariates_unscaled.shape)
                
                # Future part of decoder input (pred_len)
                future_targets_zeros = torch.zeros_like(batch_y_unscaled_all_features[:, -self.args.pred_len:, :c_out_evaluation_test]).float().to(self.device)
                
                future_covariates_scaled = None
                if total_features_in_batch_y > c_out_evaluation_test and self.scaler_manager.scaler:
                    future_covariates_unscaled = batch_y_unscaled_all_features[:, -self.args.pred_len:, c_out_evaluation_test:].cpu().numpy()
                    future_covariates_scaled = self.scaler_manager.scaler.transform(future_covariates_unscaled.reshape(-1, total_features_in_batch_y - c_out_evaluation_test)).reshape(future_covariates_unscaled.shape)

                # Concatenate to form dec_inp
                dec_inp_test = self._construct_dec_inp(hist_targets_scaled, hist_covariates_scaled, future_targets_zeros, future_covariates_scaled)
                
                outputs_raw_test = None # Initialize
                if self.args.use_amp: # This branch is for AMP (Automatic Mixed Precision)
                    with torch.cuda.amp.autocast():
                        outputs_raw_test = self.model(batch_x, batch_x_mark, dec_inp_test, batch_y_mark)
                else:
                    outputs_raw_test = self.model(batch_x, batch_x_mark, dec_inp_test, batch_y_mark)

                model_outputs_pred_len_segment = outputs_raw_test[:, -self.args.pred_len:, :] # [B, pred_len, c_out_model]
                
                # Store original unscaled targets (only the target columns) for direct comparison/saving
                true_targets_original_batch_np = batch_y_unscaled_all_features[:, -self.args.pred_len:, :c_out_evaluation_test].cpu().numpy()
                trues_original_np_targets_only.append(true_targets_original_batch_np)

                # For visualization, store the full c_out_evaluation features in original scale
                trues_original_for_viz_np.append(batch_y_unscaled_all_features[:, -self.args.pred_len:, :c_out_evaluation_test].cpu().numpy())

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

                    if self.scaler_manager is not None: # Check if a scaler manager exists, which implies scaling was done
                        pred_for_viz_original = self.scaler_manager.inverse_transform_targets(pred_for_viz_scaled.reshape(-1, c_out_evaluation_test)).reshape(pred_for_viz_scaled.shape) # Use target_scaler
                        # The input_np is batch_x, which has shape [B, seq_len, enc_in]
                        # enc_in is the total number of features (targets + covariates)
                        total_features = self.dm.enc_in
                        input_for_viz_original = self.scaler_manager.inverse_transform_all_features(input_np[0].reshape(-1, total_features)).reshape(input_np[0].shape)
                        
                        # Visualize the first target feature (index 0 of c_out_evaluation features)
                        if input_for_viz_original.shape[-1] > 0 and true_for_viz_original.shape[-1] > 0 and pred_for_viz_original.shape[-1] > 0:
                            gt_plot = np.concatenate((input_for_viz_original[:, 0], true_for_viz_original[:, 0]), axis=0)
                            pd_plot = np.concatenate((input_for_viz_original[:, 0], pred_for_viz_original[:, 0]), axis=0)
                            visual(gt_plot, pd_plot, os.path.join(folder_path, str(i) + '.pdf'))
                        else:
                            logger.warning("Cannot visualize: insufficient target features after inverse transform.")
                    else:
                        logger.warning("Cannot visualize in original scale: inverse_transform not available or scale=False.")


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

        return

    def _construct_dec_inp(self, hist_targets_scaled, hist_covariates_scaled, future_targets_zeros, future_covariates_scaled):
        """Helper to construct decoder input based on feature mode."""
        # Convert numpy arrays to tensors and move to device
        hist_targets_scaled_t = torch.from_numpy(hist_targets_scaled).float().to(self.device)
        future_targets_zeros_t = future_targets_zeros.float().to(self.device)

        # Determine if covariates should be included in decoder input based on feature mode
        if self.args.features in ['M', 'MS']:
            if hist_covariates_scaled is not None and future_covariates_scaled is not None:
                hist_covariates_scaled_t = torch.from_numpy(hist_covariates_scaled).float().to(self.device)
                future_covariates_scaled_t = torch.from_numpy(future_covariates_scaled).float().to(self.device)

                # Concatenate targets and covariates for historical part
                dec_inp_hist = torch.cat([hist_targets_scaled_t, hist_covariates_scaled_t], dim=-1)
                # Concatenate zeros for targets and scaled covariates for future part
                dec_inp_future = torch.cat([future_targets_zeros_t, future_covariates_scaled_t], dim=-1)
            else:
                # This case should ideally not happen if covariates exist and mode is M/MS,
                # but as a fallback, just use targets.
                logger.warning("Covariates expected for M/MS mode but not provided to _construct_dec_inp. Proceeding with targets only.")
                dec_inp_hist = hist_targets_scaled_t
                dec_inp_future = future_targets_zeros_t
        elif self.args.features == 'S':
            # For S mode, only targets are used in the decoder input
            dec_inp_hist = hist_targets_scaled_t
            dec_inp_future = future_targets_zeros_t

        # Combine historical and future parts
        dec_inp = torch.cat([dec_inp_hist, dec_inp_future], dim=1)
        return dec_inp
