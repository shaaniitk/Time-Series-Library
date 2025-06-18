from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.logger import logger
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
        logger.info(f"Initializing Exp_Long_Term_Forecast with args: {args}")
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        ModelClass = self.model_dict[self.args.model].Model
        
        # Check if the model class accepts quantile_levels in its __init__
        sig = inspect.signature(ModelClass.__init__)
        model_accepts_quantiles = 'quantile_levels' in sig.parameters

        q_levels = getattr(self.args, 'quantile_levels', None)

        if model_accepts_quantiles:
            logger.info(f"Model {self.args.model} accepts quantile_levels. Passing: {q_levels}")
            # This covers EnhancedAutoformer, HierarchicalEnhancedAutoformer (with previous diffs)
            # and BayesianEnhancedAutoformer (which has quantile_levels in its signature)
            # For BayesianEnhancedAutoformer, other specific args like uncertainty_method, kl_weight
            # are assumed to be part of self.args or handled by its defaults.
            model = ModelClass(self.args, quantile_levels=q_levels).float()
        elif self.args.model == 'BayesianEnhancedAutoformer' and not model_accepts_quantiles:
            # Fallback for BayesianEnhancedAutoformer if signature check fails but we know it handles quantiles
            logger.info(f"Passing quantile_levels to BayesianEnhancedAutoformer (fallback): {q_levels}")
            model = ModelClass(self.args, use_quantiles=bool(q_levels), quantile_levels=q_levels).float()
        else:
            # Standard models that don't take quantile_levels directly in constructor
            logger.info(f"Model {self.args.model} does not explicitly accept quantile_levels in constructor.")
            model = ModelClass(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        from utils.losses import get_loss_function
        
        # Get loss function name from args, default to MSE
        loss_name = getattr(self.args, 'loss', 'mse')
        
        # Handle special cases for losses that need extra parameters
        if loss_name.lower() == 'ps_loss':
            criterion = get_loss_function(loss_name, 
                                        pred_len=self.args.pred_len,
                                        mse_weight=getattr(self.args, 'ps_mse_weight', 0.5),
                                        w_corr=getattr(self.args, 'ps_w_corr', 1.0),
                                        w_var=getattr(self.args, 'ps_w_var', 1.0),
                                        w_mean=getattr(self.args, 'ps_w_mean', 1.0))
        elif loss_name.lower() == 'pinball':
            # Prioritize 'quantile_levels' if available, then 'quantiles'
            quantiles_to_use = getattr(self.args, 'quantile_levels', None)
            if quantiles_to_use is None:
                quantiles_to_use = getattr(self.args, 'quantiles', [0.1, 0.5, 0.9]) # Default
            if not isinstance(quantiles_to_use, list) or not all(isinstance(q, float) for q in quantiles_to_use if q is not None): # Allow None in list for default
                logger.warning(f"Invalid quantile_levels/quantiles: {quantiles_to_use}. Defaulting to [0.1, 0.5, 0.9]")
                quantiles_to_use = [0.1, 0.5, 0.9]
            criterion = get_loss_function(loss_name, quantiles=quantiles_to_use)
        elif loss_name.lower() == 'huber':
            delta = getattr(self.args, 'huber_delta', 1.0)
            criterion = get_loss_function(loss_name, delta=delta)
        elif loss_name.lower() == 'focal':
            alpha = getattr(self.args, 'focal_alpha', 1.0)
            gamma = getattr(self.args, 'focal_gamma', 2.0)
            criterion = get_loss_function(loss_name, alpha=alpha, gamma=gamma)
        elif loss_name.lower() == 'seasonal':
            season_length = getattr(self.args, 'season_length', 24)
            seasonal_weight = getattr(self.args, 'seasonal_weight', 1.0)
            criterion = get_loss_function(loss_name, season_length=season_length, seasonal_weight=seasonal_weight)
        elif loss_name.lower() == 'trend_aware':
            trend_weight = getattr(self.args, 'trend_weight', 1.0)
            noise_weight = getattr(self.args, 'noise_weight', 0.5)
            criterion = get_loss_function(loss_name, trend_weight=trend_weight, noise_weight=noise_weight)
        elif loss_name.lower() == 'quantile':
            quantile = getattr(self.args, 'quantile', 0.5)
            criterion = get_loss_function(loss_name, quantile=quantile)
        elif loss_name.lower() == 'dtw':
            gamma = getattr(self.args, 'dtw_gamma', 1.0)
            normalize = getattr(self.args, 'dtw_normalize', True)
            criterion = get_loss_function(loss_name, gamma=gamma, normalize=normalize)
        else:
            # Standard losses (mse, mae, mape, smape, mase, gaussian_nll)
            try:
                criterion = get_loss_function(loss_name)
            except ValueError:
                logger.warning(f"Unknown loss function: {loss_name}. Falling back to MSE.")
                criterion = nn.MSELoss()
                
        logger.info(f"Using loss function: {loss_name}")
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion):
        logger.info("Running validation phase")
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                # batch_y is kept on CPU for now as its target part might be unscaled
                # and needs processing before moving to device for loss calculation.
                # batch_y = batch_y.float() 

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                model_outputs_scaled = outputs[:, -self.args.pred_len:, :self.args.c_out]
                
                # Log dimensions for debugging
                if i == 0:  # Log only for first batch to avoid spam
                    logger.info(f"Validation batch dimensions:")
                    logger.info(f"  batch_x: {batch_x.shape}")
                    logger.info(f"  batch_y: {batch_y.shape}")
                    logger.info(f"  batch_x_mark: {batch_x_mark.shape}")
                    logger.info(f"  batch_y_mark: {batch_y_mark.shape}")
                    logger.info(f"  dec_inp: {dec_inp.shape}")
                    logger.info(f"  model outputs: {outputs.shape}")
                    logger.info(f"  model_outputs_scaled: {model_outputs_scaled.shape}")
                
                # Prepare ground_truth_final_scaled
                # Start with the relevant segment of batch_y. Covariates here are already scaled by Dataset_Custom.
                # Targets in batch_y (for val/test from Dataset_Custom) are unscaled.
                ground_truth_segment = batch_y[:, -self.args.pred_len:, :self.args.c_out].clone().detach()
                ground_truth_final_scaled = ground_truth_segment.to(self.device) # Move to device after potential modification

                # If using Dataset_Custom, its target_scaler was fit on training targets.
                # Validation batch_y has unscaled targets. We need to scale them.
                if hasattr(vali_data, 'target_scaler') and vali_data.scale and vali_data.target_scaler is not None:
                    num_features_target_scaler_knows = vali_data.target_scaler.n_features_in_
                    
                    # Number of target features within c_out that need explicit scaling
                    num_targets_to_explicitly_scale = min(self.args.c_out, num_features_target_scaler_knows)

                    if num_targets_to_explicitly_scale > 0:
                        # Extract the raw, unscaled target features from batch_y that the scaler was fit on
                        # These are assumed to be the first 'num_features_target_scaler_knows' columns
                        unscaled_targets_for_scaler_np = batch_y[:, -self.args.pred_len:, :num_features_target_scaler_knows].numpy()
                        
                        # Scale these features
                        scaled_targets_full_set_np = vali_data.target_scaler.transform(
                            unscaled_targets_for_scaler_np.reshape(-1, num_features_target_scaler_knows)
                        ).reshape(unscaled_targets_for_scaler_np.shape)
                        
                        # Update the target portion in ground_truth_final_scaled
                        # Only update the part that corresponds to the model's output (up to num_targets_to_explicitly_scale)
                        ground_truth_final_scaled[:, :, :num_targets_to_explicitly_scale] = torch.from_numpy(
                            scaled_targets_full_set_np[:, :, :num_targets_to_explicitly_scale]
                        ).float().to(self.device)
                # Else: assume batch_y is already appropriately scaled (e.g., ETT datasets or scale=False)

                # For validation, we typically only use data loss (no KL regularization)
                loss = criterion(model_outputs_scaled, ground_truth_final_scaled)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        logger.info(f"Starting training with setting: {setting}")
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device) # For training, Dataset_Custom provides scaled targets
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        # For training, batch_y is already scaled by Dataset_Custom
                        outputs = outputs[:, -self.args.pred_len:, :self.args.c_out]
                        batch_y_targets = batch_y[:, -self.args.pred_len:, :self.args.c_out].to(self.device)
                        
                        # Log dimensions for debugging (first batch of first epoch)
                        if epoch == 0 and i == 0:
                            logger.info(f"Training batch dimensions:")
                            logger.info(f"  batch_x: {batch_x.shape}")
                            logger.info(f"  batch_y: {batch_y.shape}")
                            logger.info(f"  batch_x_mark: {batch_x_mark.shape}")
                            logger.info(f"  batch_y_mark: {batch_y_mark.shape}")
                            logger.info(f"  dec_inp: {dec_inp.shape}")
                            logger.info(f"  model outputs: {outputs.shape}")
                            logger.info(f"  batch_y_targets: {batch_y_targets.shape}")
                        
                        # Compute loss (automatically includes KL for Bayesian models)
                        if hasattr(self.model, 'compute_loss'):
                            # Bayesian model handles its own loss computation
                            total_loss = self.model.compute_loss(outputs, batch_y_targets, criterion)
                            # Log loss components for first batch of first epoch for debugging
                            if epoch == 0 and i == 0:
                                loss_components = self.model.compute_loss(outputs, batch_y_targets, criterion, return_components=True)
                                logger.info(f"  data_loss: {loss_components['data_loss'].item():.6f}")
                                logger.info(f"  kl_loss: {loss_components['kl_contribution']:.6f}")
                                logger.info(f"  total_loss: {loss_components['total_loss'].item():.6f}")
                        else:
                            # Standard models use only data loss
                            total_loss = criterion(outputs, batch_y_targets)
                        
                        train_loss.append(total_loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    # For training, batch_y is already scaled by Dataset_Custom
                    outputs = outputs[:, -self.args.pred_len:, :self.args.c_out]
                    batch_y_targets = batch_y[:, -self.args.pred_len:, :self.args.c_out].to(self.device)
                    
                    # Log dimensions for debugging (first batch of first epoch)
                    if epoch == 0 and i == 0:
                        logger.info(f"Training batch dimensions:")
                        logger.info(f"  batch_x: {batch_x.shape}")
                        logger.info(f"  batch_y: {batch_y.shape}")
                        logger.info(f"  batch_x_mark: {batch_x_mark.shape}")
                        logger.info(f"  batch_y_mark: {batch_y_mark.shape}")
                        logger.info(f"  dec_inp: {dec_inp.shape}")
                        logger.info(f"  model outputs: {outputs.shape}")
                        logger.info(f"  batch_y_targets: {batch_y_targets.shape}")
                    
                    # Compute loss (automatically includes KL for Bayesian models)
                    if hasattr(self.model, 'compute_loss'):
                        # Bayesian model handles its own loss computation
                        total_loss = self.model.compute_loss(outputs, batch_y_targets, criterion)
                        # Log loss components for first batch of first epoch for debugging
                        if epoch == 0 and i == 0:
                            loss_components = self.model.compute_loss(outputs, batch_y_targets, criterion, return_components=True)
                            logger.info(f"  data_loss: {loss_components['data_loss'].item():.6f}")
                            logger.info(f"  kl_loss: {loss_components['kl_contribution']:.6f}")
                            logger.info(f"  total_loss: {loss_components['total_loss'].item():.6f}")
                    else:
                        # Standard models use only data loss
                        total_loss = criterion(outputs, batch_y_targets)
                    
                    train_loss.append(total_loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, total_loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(total_loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    total_loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion) # Using vali for test as placeholder

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        logger.info(f"Starting test with setting: {setting}, test={test}")
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds_scaled_np = []
        trues_scaled_np = []
        
        # For visualization, store original unscaled targets if possible
        trues_original_for_viz_np = [] 

        # Determine if in quantile mode and find median index
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
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                # batch_y kept on CPU for now
                # batch_y = batch_y.float() 

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # Model output is [B, pred_len, C] or [B, pred_len, C, Q]
                model_outputs_raw = outputs[:, -self.args.pred_len:, :]
                
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
                ground_truth_segment = batch_y[:, -self.args.pred_len:, :self.args.c_out].clone().detach()
                ground_truth_final_scaled = ground_truth_segment.to(self.device)

                # Store original unscaled targets for visualization (first c_out features)
                if hasattr(test_data, 'target_scaler') and test_data.scale and test_data.target_scaler is not None:
                    num_features_target_scaler_knows = test_data.target_scaler.n_features_in_
                    num_targets_to_explicitly_scale = min(self.args.c_out, num_features_target_scaler_knows)
                    
                    if num_targets_to_explicitly_scale > 0:
                        unscaled_targets_for_scaler_np = batch_y[:, -self.args.pred_len:, :num_features_target_scaler_knows].numpy()
                        trues_original_for_viz_np.append(unscaled_targets_for_scaler_np[:,:,:num_targets_to_explicitly_scale]) # Store relevant part

                        scaled_targets_full_set_np = test_data.target_scaler.transform(
                            unscaled_targets_for_scaler_np.reshape(-1, num_features_target_scaler_knows)
                        ).reshape(unscaled_targets_for_scaler_np.shape)
                        
                        ground_truth_final_scaled[:, :, :num_targets_to_explicitly_scale] = torch.from_numpy(
                            scaled_targets_full_set_np[:, :, :num_targets_to_explicitly_scale]
                        ).float().to(self.device)
                else: # If not Dataset_Custom or no scaling, assume batch_y is already what we need
                    trues_original_for_viz_np.append(ground_truth_segment.numpy())

                # pred_np_batch will be [B, pred_len, C] or [B, pred_len, C, Q]
                pred_np_batch_raw = model_outputs_raw.detach().cpu().numpy()

                if is_quantile_mode and pred_np_batch_raw.ndim == 4 and median_quantile_index != -1:
                    # Extract median predictions for metrics: [B, pred_len, C, Q] -> [B, pred_len, C]
                    pred_np_batch = pred_np_batch_raw[:, :, :, median_quantile_index]
                else:
                    pred_np_batch = pred_np_batch_raw # Assumes [B, pred_len, C]
                
                true_np_batch = ground_truth_final_scaled.detach().cpu().numpy()
                
                preds_scaled_np.append(pred_np_batch)
                trues_scaled_np.append(true_np_batch)

                if i % 20 == 0 and self.args.c_out > 0: # Visualization part
                    input_np = batch_x.detach().cpu().numpy()
                    
                    # For visualization, we want to show data in its original scale if possible
                    # pred_to_plot: model's output, needs inverse_transform_targets
                    # true_to_plot: original unscaled targets from batch_y

                    if is_quantile_mode and pred_np_batch_raw.ndim == 4 and median_quantile_index != -1:
                        pred_for_viz = pred_np_batch_raw[0, :, :, median_quantile_index] # Median prediction for viz
                    else:
                        pred_for_viz = pred_np_batch_raw[0, :, :] # Standard prediction for viz

                    true_for_viz_original = trues_original_for_viz_np[-1][0, :, :] # Corresponding original true values

                    if hasattr(test_data, 'inverse_transform_targets') and test_data.scale:
                        # Inverse transform only the target portion of predictions
                        # Assuming target_scaler was fit on num_features_target_scaler_knows features
                        # and these are the first ones in c_out.
                        num_pred_targets_to_inv_transform = min(pred_for_viz.shape[-1], num_features_target_scaler_knows)
                        
                        pred_target_part_scaled = pred_for_viz[:, :num_pred_targets_to_inv_transform]
                        pred_target_part_original = test_data.inverse_transform_targets(pred_target_part_scaled)
                        
                        # For input, inverse transform its target part
                        input_target_part_unscaled = test_data.inverse_transform_targets(input_np[0, :, :num_features_target_scaler_knows])
                        
                        # Visualize the first target feature
                        gt_plot = np.concatenate((input_target_part_unscaled[:, 0], true_for_viz_original[:, 0]), axis=0)
                        pd_plot = np.concatenate((input_target_part_unscaled[:, 0], pred_target_part_original[:, 0]), axis=0)
                        visual(gt_plot, pd_plot, os.path.join(folder_path, str(i) + '.pdf'))
                    else: # If no inverse_transform_targets or not scaled, plot as is (likely scaled)
                        gt_plot = np.concatenate((input_np[0, :, 0], true_np_batch[0, :, 0]), axis=0)
                        pd_plot = np.concatenate((input_np[0, :, 0], pred_np_batch[0, :, 0]), axis=0)
                        visual(gt_plot, pd_plot, os.path.join(folder_path, str(i) + '_scaled.pdf'))


        preds = np.concatenate(preds_scaled_np, axis=0)
        trues = np.concatenate(trues_scaled_np, axis=0)
        print('test shape (scaled for metrics):', preds.shape, trues.shape)
        # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1]) # Already in correct shape
        # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # result save
        folder_path_results = './results/' + setting + '/' # Changed from folder_path to avoid conflict
        if not os.path.exists(folder_path_results):
            os.makedirs(folder_path_results)

        # dtw calculation (if enabled)
        dtw_val = 'Not calculated'
        if getattr(self.args, 'use_dtw', False) and preds.shape[-1] == 1: # DTW typically for univariate
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x_dtw = preds[i] # Already [pred_len, 1] or similar
                y_dtw = trues[i]
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x_dtw, y_dtw, dist=manhattan_distance)
                dtw_list.append(d)
            dtw_val = np.array(dtw_list).mean()
        
        mae, mse, rmse, mape, mspe = metric(preds, trues) # Metrics on scaled data
        print('Metrics (on scaled data): mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw_val))
        
        with open(os.path.join(folder_path_results, "result_long_term_forecast.txt"), 'a') as f:
            f.write(setting + "  \n")
            f.write('Metrics (on scaled data): mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw_val))
            f.write('\n')
            f.write('\n')

        np.save(folder_path_results + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path_results + 'pred_scaled.npy', preds) # Save scaled predictions
        np.save(folder_path_results + 'true_scaled.npy', trues) # Save scaled trues

        return
