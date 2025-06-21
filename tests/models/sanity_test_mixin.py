import unittest
import torch
import numpy as np
import pandas as pd
import os
from data_provider.data_factory import data_provider
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from utils.logger import logger
from utils.dimension_manager import DimensionManager # Import DimensionManager
import inspect # Import inspect
from utils.losses import get_loss_function, PinballLoss # Import the loss getter and PinballLoss

class SanityTestMixin:
    def run_sanity_test(self, ModelClass, device='cpu', epochs=10, model_config=None, loss_config=None):
        logger.info(f"--- Running Sanity Test (Sine Wave Convergence) for: {ModelClass.__name__} ---")
        # Parameters
        seq_len = 100
        pred_len = 50
        n_points = 2000
        label_len = 50 # Standard for Autoformer family
        batch_size = 16

        # Model configuration defaults (can be overridden by model_config)
        default_model_params = {
            'd_model': 16, 'e_layers': 2, 'd_layers': 1,
            'd_ff': 32, 'n_heads': 2, 'factor': 1,
            'activation': 'gelu', 'output_attention': False,
            'moving_avg': 25, 'top_k': 2, 'num_kernels': 6 # For TimesNet
        }
        if model_config:
            default_model_params.update(model_config)

        # Synthetic Data Generation
        X_time = np.arange(n_points) * 5 * np.pi / 180
        X1_time = np.arange(n_points) * 10 * np.pi / 180
        X2_time = np.arange(n_points) * 15 * np.pi / 180
        cov1 = np.sin(X_time)
        cov2 = np.sin(X1_time)
        cov3 = np.sin(X2_time)
        t1 = np.sin(X_time - X1_time)
        t2 = np.sin(X1_time - X2_time)
        t3 = np.sin(X2_time - X_time)
        date_full = pd.date_range(start='2020-01-01', periods=n_points, freq='h')
        df_full = pd.DataFrame({'date': date_full, 'cov1': cov1, 'cov2': cov2, 'cov3': cov3, 't1': t1, 't2': t2, 't3': t3})
        csv_path = 'synthetic_timeseries_full.csv'
        df_full.to_csv(csv_path, index=False)
        target_cols = ['t1', 't2', 't3']

        # Base Args for data_provider and DimensionManager
        class Args:
            pass
        args = Args()
        args.seq_len = seq_len
        args.label_len = label_len
        args.pred_len = pred_len
        args.embed = 'timeF'
        args.freq = 'h'
        args.dropout = 0.1
        args.task_name = 'long_term_forecast'
        args.features = 'M' # Multivariate mode for sanity test (6 features)
        args.data = 'custom'
        args.target = ','.join(target_cols)
        args.root_path = '.'
        args.data_path = os.path.basename(csv_path) # Use just filename for data_path
        args.num_workers = 0
        args.batch_size = batch_size
        args.augmentation_ratio = 0
        args.seasonal_patterns = None
        args.use_amp = False
        args.use_multi_gpu = False
        args.scale = True
        args.validation_length = seq_len + pred_len # 150
        args.test_length = pred_len # 50
        # Add model-specific architectural defaults to args
        for k, v in default_model_params.items():
            setattr(args, k, v)
        # Ensure enc_in, dec_in, c_out are set based on 'M' mode for 6 features
        # These will be overridden by DimensionManager later for the model,
        # but data_provider might use them initially.
        args.enc_in = 6
        args.dec_in = 6
        args.c_out = 6


        # --- Dimension Management ---
        dm = DimensionManager()
        dm.analyze_and_configure(
            data_path=csv_path, # Full path for analysis
            args=args,
            task_name=args.task_name,
            features_mode=args.features,
            target_columns=target_cols,
            loss_name=loss_config.get('loss_name', 'mse') if loss_config else 'mse',
            quantile_levels=loss_config.get('quantile_levels') if loss_config else None
        )
        model_init_args = dm.get_model_init_params()
        eval_info = dm.get_evaluation_info()
        c_out_evaluation = eval_info['c_out_evaluation'] # Base target features for metrics/scaling

        # Data Loaders
        dataset, loader = data_provider(args, flag='train')
        train_scaler = dataset.scaler if hasattr(dataset, 'scaler') else None
        logger.debug(f"Train dataset scaler obtained: {train_scaler}")
        logger.info(f"Train scaler n_features_in_ (after fit): {train_scaler.n_features_in_ if train_scaler else 'N/A'}")

        # Model Initialization
        model = ModelClass(model_init_args).to(device) # Pass Namespace from DM
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Loss Function
        if loss_config and 'loss_name' in loss_config:
            criterion = get_loss_function(loss_config['loss_name'], **{k: v for k, v in loss_config.items() if k != 'loss_name'})
        else:
            criterion = torch.nn.MSELoss()
        
        # If model has its own loss computation (e.g., Bayesian models)
        if hasattr(model, 'configure_optimizer_loss') and callable(getattr(model, 'configure_optimizer_loss')):
            logger.info(f"Model {ModelClass.__name__} has configure_optimizer_loss. Wrapping base criterion.")
            criterion = model.configure_optimizer_loss(criterion, verbose=False)


        # Training Loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 5 # Increased patience

        for epoch in range(epochs):
            model.train()
            train_losses = []
            for batch_idx, batch_data_train in enumerate(loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data_train
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device) # Dataset_Custom scales all features for train
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                
                optimizer.zero_grad()
                
                dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)
                
                y_pred = model(batch_x, batch_x_mark, dec_inp, batch_y_mark) # Raw model output
                
                # Prepare y_true for loss: always the first c_out_evaluation features, scaled
                y_true_for_loss = batch_y[:, -pred_len:, :c_out_evaluation].to(device)

                is_model_quantile_train = hasattr(model, 'is_quantile_mode') and model.is_quantile_mode
                is_criterion_pinball_train = isinstance(criterion, PinballLoss) or \
                                             (hasattr(criterion, '_is_pinball_loss_based') and criterion._is_pinball_loss_based)

                if is_model_quantile_train and is_criterion_pinball_train:
                    y_pred_for_loss = y_pred[:, -pred_len:, :] # Use full model output (c_out_model)
                else:
                    y_pred_for_loss = y_pred[:, -pred_len:, :c_out_evaluation]
                
                loss = criterion(y_pred_for_loss, y_true_for_loss)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            
            # Validation Loop
            model.eval()
            val_args = Args() # Create a fresh Args for validation
            for attr_name, attr_value in vars(args).items(): # Copy attributes from original args
                setattr(val_args, attr_name, attr_value)
            val_args.scaler = train_scaler # Pass the scaler
            val_args.data_path = os.path.basename(csv_path) # Ensure correct data_path for val_dataset
            
            val_dataset, val_loader = data_provider(val_args, flag='val')
            
            val_losses_scaled = []
            val_losses_orig = []
            
            with torch.no_grad():
                for batch_idx_val, batch_data_val in enumerate(val_loader):
                    batch_x_val, batch_y_val, batch_x_mark_val, batch_y_mark_val = batch_data_val
                    batch_x_val = batch_x_val.float().to(device)
                    # batch_y_val from val_loader has unscaled targets (first c_out_evaluation), scaled covariates
                    batch_x_mark_val = batch_x_mark_val.float().to(device)
                    batch_y_mark_val = batch_y_mark_val.float().to(device)

                    # Create dec_inp for validation
                    # Historical part: scale the target portion of batch_y_val
                    hist_targets_unscaled_val = batch_y_val[:, :label_len, :c_out_evaluation].cpu().numpy()
                    hist_targets_scaled_val_np = val_dataset.scaler.transform(hist_targets_unscaled_val.reshape(-1, c_out_evaluation)).reshape(hist_targets_unscaled_val.shape)
                    dec_inp_hist_scaled_val = torch.from_numpy(hist_targets_scaled_val_np).float().to(device)
                    
                    dec_inp_future_zeros_val = torch.zeros_like(batch_y_val[:, -pred_len:, :c_out_evaluation]).float().to(device)
                    dec_inp_val = torch.cat([dec_inp_hist_scaled_val, dec_inp_future_zeros_val], dim=1)

                    y_pred_val_raw = model(batch_x_val, batch_x_mark_val, dec_inp_val, batch_y_mark_val) # Raw model output

                    # Ground truth for scaled loss: scale the target part of batch_y_val
                    y_true_targets_unscaled_val_loss = batch_y_val[:, -pred_len:, :c_out_evaluation].cpu().numpy()
                    y_true_targets_scaled_val_loss_np = val_dataset.scaler.transform(y_true_targets_unscaled_val_loss.reshape(-1, c_out_evaluation)).reshape(y_true_targets_unscaled_val_loss.shape)
                    y_true_targets_scaled_val_loss = torch.from_numpy(y_true_targets_scaled_val_loss_np).float().to(device)
                    
                    is_model_quantile_val = hasattr(model, 'is_quantile_mode') and model.is_quantile_mode
                    is_criterion_pinball_val = isinstance(criterion, PinballLoss) or \
                                               (hasattr(criterion, '_is_pinball_loss_based') and criterion._is_pinball_loss_based)

                    if is_model_quantile_val and is_criterion_pinball_val:
                        y_pred_for_loss_val = y_pred_val_raw[:, -pred_len:, :] # Use full model output (c_out_model)
                    else:
                        y_pred_for_loss_val = y_pred_val_raw[:, -pred_len:, :c_out_evaluation]
                    
                    val_loss_s = criterion(y_pred_for_loss_val, y_true_targets_scaled_val_loss).item()
                    val_losses_scaled.append(val_loss_s)
                    
                    # Original scale loss calculation
                    try:
                        pred_for_inv_transform_val = y_pred_for_loss_val # This is [B, L, C_eval] or [B, L, C_model]
                        if is_model_quantile_val and is_criterion_pinball_val:
                            median_idx = eval_info['num_quantiles'] // 2
                            pred_for_inv_transform_val = y_pred_for_loss_val.view(
                                y_pred_for_loss_val.shape[0], pred_len, c_out_evaluation, eval_info['num_quantiles']
                            )[:, :, :, median_idx]
                        
                        pred_orig_np_val = val_dataset.scaler.inverse_transform(
                            pred_for_inv_transform_val.cpu().numpy().reshape(-1, c_out_evaluation)
                        ).reshape(pred_for_inv_transform_val.shape)
                        
                        true_orig_np_val = batch_y_val[:, -pred_len:, :c_out_evaluation].cpu().numpy() # Already unscaled targets

                        val_loss_o = np.mean((pred_orig_np_val[:, :, -len(target_cols):] - true_orig_np_val[:, :, -len(target_cols):]) ** 2)
                        val_losses_orig.append(val_loss_o)
                    except Exception as e:
                        logger.warning(f"Could not compute original scale validation loss: {e}")
                        val_losses_orig.append(np.nan)
            
            avg_train_loss = np.mean(train_losses)
            avg_val_loss_scaled = np.mean(val_losses_scaled) if val_losses_scaled else float('inf')
            valid_val_losses_orig_clean = [loss for loss in val_losses_orig if not np.isnan(loss)]
            avg_val_loss_orig = np.mean(valid_val_losses_orig_clean) if valid_val_losses_orig_clean else np.nan
            
            log_msg = f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss (scaled): {avg_val_loss_scaled:.6f}"
            log_msg += f", Val Loss (original): {avg_val_loss_orig:.6f}" if not np.isnan(avg_val_loss_orig) else ", Val Loss (original): N/A"
            logger.info(log_msg)
            
            if avg_val_loss_scaled < 0.01:
                logger.info(f"Early stopping at epoch {epoch+1}, Val Loss (scaled) {avg_val_loss_scaled:.6f} < 0.01 (converged)")
                break
            
            if avg_val_loss_scaled < best_val_loss:
                best_val_loss = avg_val_loss_scaled
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss")
                    break
        
        # Test Loop
        logger.info("Starting final test evaluation")
        model.eval()
        test_args = Args() # Create a fresh Args for test
        for attr_name, attr_value in vars(args).items(): # Copy attributes from original args
            setattr(test_args, attr_name, attr_value)
        test_args.scaler = train_scaler
        test_args.data_path = os.path.basename(csv_path)
        
        test_dataset, test_loader = data_provider(test_args, flag='test')
        
        first_batch_preds_plot_orig = None
        first_batch_actuals_plot_orig = None
        all_preds_for_final_mse_orig = []
        all_trues_for_final_mse_orig = []

        with torch.no_grad():
            for batch_idx_test, batch_data_test in enumerate(test_loader):
                batch_x_test, batch_y_test, batch_x_mark_test, batch_y_mark_test = batch_data_test
                batch_x_test = batch_x_test.float().to(device)
                batch_x_mark_test = batch_x_mark_test.float().to(device)
                batch_y_mark_test = batch_y_mark_test.float().to(device)

                hist_targets_unscaled_test = batch_y_test[:, :label_len, :c_out_evaluation].cpu().numpy()
                hist_targets_scaled_test_np = test_dataset.scaler.transform(hist_targets_unscaled_test.reshape(-1, c_out_evaluation)).reshape(hist_targets_unscaled_test.shape)
                dec_inp_hist_scaled_test = torch.from_numpy(hist_targets_scaled_test_np).float().to(device)
                dec_inp_future_zeros_test = torch.zeros_like(batch_y_test[:, -pred_len:, :c_out_evaluation]).float().to(device)
                dec_inp_test = torch.cat([dec_inp_hist_scaled_test, dec_inp_future_zeros_test], dim=1)

                y_pred_test_raw = model(batch_x_test, batch_x_mark_test, dec_inp_test, batch_y_mark_test)
                y_pred_test_pred_len_segment = y_pred_test_raw[:, -pred_len:, :] 

                current_batch_size_test = y_pred_test_pred_len_segment.shape[0]
                current_output_features_test = y_pred_test_pred_len_segment.shape[-1]
                
                point_preds_batch_scaled_torch = torch.tensor([]) # Initialize

                is_model_quantile_test = hasattr(model, 'is_quantile_mode') and model.is_quantile_mode
                expected_quantile_feat_dim_test = c_out_evaluation * eval_info['num_quantiles']

                if is_model_quantile_test and eval_info['num_quantiles'] > 1 and \
                   current_output_features_test == expected_quantile_feat_dim_test:
                    median_idx_test = eval_info['num_quantiles'] // 2
                    point_preds_batch_scaled_torch = y_pred_test_pred_len_segment.view(
                        current_batch_size_test, pred_len, c_out_evaluation, eval_info['num_quantiles']
                    )[:, :, :, median_idx_test]
                elif current_output_features_test == c_out_evaluation:
                    point_preds_batch_scaled_torch = y_pred_test_pred_len_segment
                elif current_output_features_test > c_out_evaluation:
                    point_preds_batch_scaled_torch = y_pred_test_pred_len_segment[:, :, :c_out_evaluation]
                else:
                    logger.error(f"[TestLoop] Output features {current_output_features_test} < c_out_evaluation {c_out_evaluation}. This is an error.")
                    logger.error(f"[TestLoop] y_pred_test_pred_len_segment shape: {y_pred_test_pred_len_segment.shape}")
                    continue # Skip this problematic batch

                if point_preds_batch_scaled_torch.shape[-1] != c_out_evaluation:
                    logger.critical(f"[TestLoop] FATAL: point_preds_batch_scaled_torch last dim is {point_preds_batch_scaled_torch.shape[-1]}, expected {c_out_evaluation}.")
                    continue
                
                point_preds_batch_orig_np = test_dataset.scaler.inverse_transform(
                    point_preds_batch_scaled_torch.cpu().numpy().reshape(-1, c_out_evaluation)
                ).reshape(current_batch_size_test, pred_len, c_out_evaluation)
                
                true_batch_c_out_features_orig_np = batch_y_test[:, -pred_len:, :c_out_evaluation].cpu().numpy()
                true_targets_batch_orig_np = true_batch_c_out_features_orig_np[:, :, -len(target_cols):]
                
                all_preds_for_final_mse_orig.append(point_preds_batch_orig_np[:, :, -len(target_cols):])
                all_trues_for_final_mse_orig.append(true_targets_batch_orig_np)

                if batch_idx_test == 0:
                    first_batch_preds_plot_orig = point_preds_batch_orig_np[0]
                    first_batch_actuals_plot_orig = true_batch_c_out_features_orig_np[0]

        final_preds_orig = np.concatenate(all_preds_for_final_mse_orig, axis=0)
        final_trues_orig = np.concatenate(all_trues_for_final_mse_orig, axis=0)
        
        mse_forecast = np.mean((final_preds_orig - final_trues_orig) ** 2)
        logger.info(f"Final test MSE (targets only, original scale): {mse_forecast:.6f}")
        
        # Plotting
        if first_batch_preds_plot_orig is not None and first_batch_actuals_plot_orig is not None:
            plot_len_viz = min(pred_len, 50) # Visualize up to 50 steps or pred_len
            fig, axes = plt.subplots(len(target_cols), 1, figsize=(12, 4 * len(target_cols)))
            if len(target_cols) == 1: axes = [axes]

            for i, target_col_name in enumerate(target_cols):
                # Find the index of target_col_name within the c_out_evaluation features
                # This assumes target_cols are the last len(target_cols) of the c_out_evaluation features
                target_idx_in_c_out_eval = c_out_evaluation - len(target_cols) + i
                
                ax = axes[i]
                ax.plot(range(plot_len_viz), 
                        first_batch_actuals_plot_orig[:plot_len_viz, target_idx_in_c_out_eval], 
                        'g-', label=f'Actual {target_col_name}', linewidth=2)
                ax.plot(range(plot_len_viz), 
                        first_batch_preds_plot_orig[:plot_len_viz, target_idx_in_c_out_eval], 
                        'r--', label=f'Forecast {target_col_name}', linewidth=2)
                
                mse_target_plot = np.mean((first_batch_preds_plot_orig[:, target_idx_in_c_out_eval] - first_batch_actuals_plot_orig[:, target_idx_in_c_out_eval])**2)
                ax.set_title(f'{target_col_name} (Plot MSE: {mse_target_plot:.4f}, Original Scale)')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            os.makedirs('pic', exist_ok=True)
            plt.savefig(f'pic/sanity_test_{ModelClass.__name__}_forecast.png', dpi=150)
            logger.info(f"Forecast plot saved to pic/sanity_test_{ModelClass.__name__}_forecast.png")
            plt.close()
        
        if os.path.exists(csv_path):
            os.remove(csv_path)
        
        return mse_forecast, first_batch_preds_plot_orig, first_batch_actuals_plot_orig
