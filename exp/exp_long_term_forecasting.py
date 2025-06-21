from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.logger import logger
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
        logger.info(f"Initializing Exp_Long_Term_Forecast with args: {args}")
        self.dm = DimensionManager()
        # Analyze data and configure dimensions based on args
        data_path_for_dm = os.path.join(args.root_path, args.data_path)
        if not os.path.exists(data_path_for_dm):
             logger.warning(f"Data path for DimensionManager analysis not found: {data_path_for_dm}. Using args as is.")
             # If data path is not found, DM might not be able to analyze.
             # Proceed with caution, or ensure data_path is always correct.
             self.model_init_args = args # Fallback to original args
             self.eval_info = { # Provide default eval_info
                 'c_out_evaluation': args.c_out,
                 'num_quantiles': len(getattr(args, 'quantile_levels', [])) if getattr(args, 'quantile_levels', []) else 1,
                 'quantile_levels': getattr(args, 'quantile_levels', None),
                 'loss_name': args.loss,
                 'target_columns': args.target.split(',') if isinstance(args.target, str) else args.target
             }
        else:
            self.dm.analyze_and_configure(
                data_path=data_path_for_dm, # Use full path for analysis
                args=args,
                task_name=args.task_name,
                features_mode=args.features,
                target_columns=args.target,
                loss_name=args.loss,
                quantile_levels=getattr(args, 'quantile_levels', None)
            )
            self.model_init_args = self.dm.get_model_init_params()
            self.eval_info = self.dm.get_evaluation_info()

        # Initialize the parent class *after* model_init_args is set.
        # Pass the resolved args to the parent, which will store it as self.args.
        super(Exp_Long_Term_Forecast, self).__init__(self.model_init_args)

    def _build_model(self):
        ModelClass = self.model_dict[self.args.model].Model
        # Get model initialization parameters from the DimensionManager
        # self.model_init_args was set in __init__
        model = ModelClass(self.model_init_args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        # Pass the scaler from train_data to val_data and test_data args
        # The self.args object is shared and modified here.
        if flag == 'train':
            data_set, data_loader = data_provider(self.args, flag='train')
            # Store the fitted scaler from the training dataset in self.args
            # so it can be passed to val/test data_provider calls.
            self.args.scaler = data_set.scaler if hasattr(data_set, 'scaler') else None
            self.args.target_scaler = data_set.target_scaler if hasattr(data_set, 'target_scaler') else None
            logger.debug(f"Stored train_scaler in self.args: {self.args.scaler is not None}")
            logger.debug(f"Stored train_target_scaler in self.args: {self.args.target_scaler is not None}")
        else: # 'val' or 'test'
            # Ensure self.args.scaler (from training) is used
            data_set, data_loader = data_provider(self.args, flag=flag)
        return data_set, data_loader

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

    def vali(self, vali_data, vali_loader, criterion):
        logger.info("Running validation phase")
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y_val_unscaled_targets, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                # batch_y_val_unscaled_targets has unscaled targets, scaled covariates (if M/MS)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Decoder input construction
                # batch_y_val_unscaled_targets contains ALL features (targets + covariates) in original scale
                total_features_in_batch_y = batch_y_val_unscaled_targets.shape[-1]
                c_out_evaluation = self.eval_info['c_out_evaluation'] # Number of actual targets

                # Historical part of decoder input (label_len)
                # Targets: scale using target_scaler
                hist_targets_unscaled = batch_y_val_unscaled_targets[:, :self.args.label_len, :c_out_evaluation].cpu().numpy()
                hist_targets_scaled = vali_data.target_scaler.transform(hist_targets_unscaled.reshape(-1, c_out_evaluation)).reshape(hist_targets_unscaled.shape)
                
                # Covariates: scale using main scaler (if present)
                hist_covariates_scaled = None
                if total_features_in_batch_y > c_out_evaluation: # Check if covariates exist
                    hist_covariates_unscaled = batch_y_val_unscaled_targets[:, :self.args.label_len, c_out_evaluation:].cpu().numpy()
                    hist_covariates_scaled = vali_data.scaler.transform(hist_covariates_unscaled.reshape(-1, total_features_in_batch_y - c_out_evaluation)).reshape(hist_covariates_unscaled.shape)
                
                # Future part of decoder input (pred_len)
                # Targets: zero out
                future_targets_zeros = torch.zeros_like(batch_y_val_unscaled_targets[:, -self.args.pred_len:, :c_out_evaluation]).float().to(self.device)
                
                # Covariates: scale using main scaler (if present)
                future_covariates_scaled = None
                if total_features_in_batch_y > c_out_evaluation:
                    future_covariates_unscaled = batch_y_val_unscaled_targets[:, -self.args.pred_len:, c_out_evaluation:].cpu().numpy()
                    future_covariates_scaled = vali_data.scaler.transform(future_covariates_unscaled.reshape(-1, total_features_in_batch_y - c_out_evaluation)).reshape(future_covariates_unscaled.shape)

                # Concatenate to form dec_inp
                dec_inp_val = self._construct_dec_inp(hist_targets_scaled, hist_covariates_scaled, future_targets_zeros, future_covariates_scaled)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs_raw = self.model(batch_x, batch_x_mark, dec_inp_val, batch_y_mark)
                else:
                    outputs_raw = self.model(batch_x, batch_x_mark, dec_inp_val, batch_y_mark)
                
                # Prepare y_true for loss: scale the target part of batch_y_val_unscaled_targets
                y_true_targets_unscaled_val_loss = batch_y_val_unscaled_targets[:, -self.args.pred_len:, :c_out_evaluation].cpu().numpy()
                y_true_targets_scaled_val_loss_np = vali_data.target_scaler.transform(y_true_targets_unscaled_val_loss.reshape(-1, c_out_evaluation)).reshape(y_true_targets_unscaled_val_loss.shape)
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
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test') # Test loader uses train_scaler via self.args

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler_amp = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss_epoch_list = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device) # For training, Dataset_Custom provides scaled targets
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # For training, batch_y is already scaled by Dataset_Custom. So this is fine.
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device) 

                outputs_raw_train = None # Initialize
                if self.args.use_amp: # This branch is for AMP (Automatic Mixed Precision)
                    with torch.cuda.amp.autocast():
                        outputs_raw_train = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs_raw_train = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                c_out_evaluation_train = self.eval_info['c_out_evaluation']
                y_true_for_loss_train = batch_y[:, -self.args.pred_len:, :c_out_evaluation_train].to(self.device)

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
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion) # Using vali for test as placeholder

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss_avg, vali_loss, test_loss))
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
            for i, (batch_x, batch_y_unscaled_targets, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Decoder input
                # batch_y_unscaled_targets contains ALL features (targets + covariates) in original scale
                total_features_in_batch_y = batch_y_unscaled_targets.shape[-1]

                # Historical part of decoder input (label_len)
                hist_targets_unscaled = batch_y_unscaled_targets[:, :self.args.label_len, :c_out_evaluation_test].cpu().numpy()
                hist_targets_scaled = test_data.target_scaler.transform(hist_targets_unscaled.reshape(-1, c_out_evaluation_test)).reshape(hist_targets_unscaled.shape)
                
                hist_covariates_scaled = None
                if total_features_in_batch_y > c_out_evaluation_test:
                    hist_covariates_unscaled = batch_y_unscaled_targets[:, :self.args.label_len, c_out_evaluation_test:].cpu().numpy()
                    hist_covariates_scaled = test_data.scaler.transform(hist_covariates_unscaled.reshape(-1, total_features_in_batch_y - c_out_evaluation_test)).reshape(hist_covariates_unscaled.shape)
                
                # Future part of decoder input (pred_len)
                future_targets_zeros = torch.zeros_like(batch_y_unscaled_targets[:, -self.args.pred_len:, :c_out_evaluation_test]).float().to(self.device)
                
                future_covariates_scaled = None
                if total_features_in_batch_y > c_out_evaluation_test:
                    future_covariates_unscaled = batch_y_unscaled_targets[:, -self.args.pred_len:, c_out_evaluation_test:].cpu().numpy()
                    future_covariates_scaled = test_data.scaler.transform(future_covariates_unscaled.reshape(-1, total_features_in_batch_y - c_out_evaluation_test)).reshape(future_covariates_unscaled.shape)

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
                true_targets_original_batch_np = batch_y_unscaled_targets[:, -self.args.pred_len:, :c_out_evaluation_test].cpu().numpy()
                trues_original_np_targets_only.append(true_targets_original_batch_np)

                # For visualization, store the full c_out_evaluation features in original scale
                trues_original_for_viz_np.append(batch_y_unscaled_targets[:, -self.args.pred_len:, :c_out_evaluation_test].cpu().numpy())

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

                    if hasattr(test_data, 'inverse_transform') and test_data.scale:
                        pred_for_viz_original = test_data.inverse_transform_targets(pred_for_viz_scaled.reshape(-1, c_out_evaluation_test)).reshape(pred_for_viz_scaled.shape) # Use target_scaler
                        input_for_viz_original = test_data.inverse_transform(input_np[0].reshape(-1, c_out_evaluation_test)).reshape(input_np[0].shape)
                        
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
        preds_final_original = test_data.inverse_transform_targets(preds_final_scaled.reshape(-1, c_out_evaluation_test)).reshape(preds_final_scaled.shape)
        
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

        if hist_covariates_scaled is not None and future_covariates_scaled is not None:
            hist_covariates_scaled_t = torch.from_numpy(hist_covariates_scaled).float().to(self.device)
            future_covariates_scaled_t = torch.from_numpy(future_covariates_scaled).float().to(self.device)

            # Concatenate targets and covariates for historical part
            dec_inp_hist = torch.cat([hist_targets_scaled_t, hist_covariates_scaled_t], dim=-1)
            # Concatenate zeros for targets and scaled covariates for future part
            dec_inp_future = torch.cat([future_targets_zeros_t, future_covariates_scaled_t], dim=-1)
        else:
            # No covariates, just targets (e.g., S mode)
            dec_inp_hist = hist_targets_scaled_t
            dec_inp_future = future_targets_zeros_t

        # Combine historical and future parts
        dec_inp = torch.cat([dec_inp_hist, dec_inp_future], dim=1)
        return dec_inp
        # Ensure both preds and trues for metric calculation are [samples, pred_len, num_target_cols]
        # target_cols from eval_info defines which columns are the true targets
        num_true_target_cols = len(self.eval_info['target_columns'])
        
        preds_for_metric = preds_final_original[:, :, :num_true_target_cols]
        trues_for_metric = trues_final_original_targets_only[:, :, :num_true_target_cols]

        logger.info(f'Test shape (original scale for metrics): preds={preds_for_metric.shape}, trues={trues_for_metric.shape}')
        
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
