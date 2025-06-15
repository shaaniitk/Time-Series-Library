import unittest
import torch
import numpy as np
import pandas as pd
import os
from data_provider.data_factory import data_provider
from models.TimesNet import Model as TimesNet
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from utils.logger import logger

class SanityTestMixin:
    def run_sanity_test(self, ModelClass, device='cpu', epochs=10):
        logger.info(f"Running sanity test for {ModelClass.__name__}")
        
        # Parameters
        seq_len = 100
        pred_len = 50
        n_points = 2000  # Increased for better train/val split
        d_model = 16
        e_layers = 2
        d_ff = 32
        enc_in = 3
        c_out = 3
        num_kernels = 6
        label_len = 50
        top_k = 2
        batch_size = 16
        # Angles
        X = np.arange(n_points) * 5 * np.pi / 180
        X1 = np.arange(n_points) * 10 * np.pi / 180
        X2 = np.arange(n_points) * 15 * np.pi / 180
        # Covariates
        cov1 = np.sin(X)
        cov2 = np.sin(X1)
        cov3 = np.sin(X2)        # Targets
        t1 = np.sin(X - X1)
        t2 = np.sin(X1 - X2)
        t3 = np.sin(X2 - X)
        
        # Split data: training up to (n_points - pred_len), last pred_len for true forecasting
        train_end = n_points - pred_len
        logger.info(f"Training on first {train_end} points, forecasting last {pred_len} points")
        
        # Training data (up to train_end)
        date_train = pd.date_range(start='2020-01-01', periods=train_end, freq='h')
        df_train = pd.DataFrame({
            'date': date_train,
            'cov1': cov1[:train_end],
            'cov2': cov2[:train_end],
            'cov3': cov3[:train_end],
            't1': t1[:train_end],
            't2': t2[:train_end],
            't3': t3[:train_end]
        })        # Future covariates and actual targets for final test forecasting (test period only)
        future_covariates = np.column_stack([
            cov1[test_start:],
            cov2[test_start:], 
            cov3[test_start:]
        ])
        actual_targets_orig = np.column_stack([
            t1[test_start:],
            t2[test_start:],
            t3[test_start:]
        ])# Correct time series split based on validation requirements
        # If we want to validate on val_period points, we need seq_len + val_period points
        # in validation dataset to create sliding windows
        
        val_period = 50  # How many points we want to validate on
        test_period = pred_len  # How many points for final test (50)
        
        # Training ends at: total - val_period - test_period = 2000 - 50 - 50 = 1900
        train_end = n_points - val_period - test_period  # 1900
        
        # Validation needs: seq_len + val_period points = 100 + 50 = 150 points  
        # Validation starts at: train_end - seq_len + 1 = 1900 - 100 + 1 = 1801
        val_start = train_end - seq_len + 1  # 1801
        val_end = train_end + val_period  # 1950
        
        # Test starts where validation ends
        test_start = val_end  # 1950
        
        logger.info(f"Training data: points 0-{train_end} ({train_end} points)")
        logger.info(f"Validation data: points {val_start}-{val_end} ({val_end - val_start} points)")
        logger.info(f"Test data: points {test_start}-{n_points} ({n_points - test_start} points)")        # Save the FULL dataset for data loader  
        # The data loader will internally split based on borders
        csv_path = 'synthetic_timeseries_full.csv'
        df_train.to_csv(csv_path, index=False)  # Full dataset
          # Create training-only subset for reference
        df_train_only = df_train.iloc[:train_end]  # Points 0 to 1900
        
        # Only cov1, cov2, cov3 as features, t1, t2, t3 as targets
        target_cols = ['t1', 't2', 't3']# Args for data provider
        class Args:
            pass
        args = Args()
        args.seq_len = seq_len
        args.label_len = label_len
        args.pred_len = pred_len
        args.d_model = d_model
        args.e_layers = e_layers
        args.d_ff = d_ff
        args.enc_in = 6  # All 6 features: 3 covariates + 3 targets in 'M' mode
        args.c_out = 3  # Predict all three targets
        args.embed = 'timeF'
        args.freq = 'h'
        args.dropout = 0.1
        args.task_name = 'long_term_forecast'
        args.features = 'M'  # Multivariate mode - use both covariates and targets
        args.data = 'custom'
        args.target = ','.join(target_cols)  # use all three as targets
        args.root_path = '.'
        args.data_path = csv_path
        args.num_workers = 0
        args.batch_size = batch_size
        args.augmentation_ratio = 0
        args.seasonal_patterns = None
        args.top_k = top_k
        args.num_kernels = num_kernels
        args.use_amp = False  # Disable AMP for simplicity
        args.use_multi_gpu = False  # Disable multi-GPU for simplicity        args.scale = True  # Enable data scaling (default behavior)        
        
        # Use data provider for training (only the reduced training set)
        dataset, loader = data_provider(args, flag='train')
        
        model = ModelClass(args).to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        
        # Training loop with proper time series validation
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 3  # Early stopping patience
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_losses = []
            for batch in loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                optimizer.zero_grad()
                y_pred = model(batch_x, batch_x_mark, batch_y[:, :label_len, :], batch_y_mark)
                # Only compare prediction part
                y_true = batch_y[:, -pred_len:, :]
                y_pred = y_pred[:, -pred_len:, :]
                loss = criterion(y_pred, y_true)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())            # True forecasting validation phase using proper data loader
            model.eval()
            
            # Create validation dataset using the same full dataset
            # The data loader will use borders to get the right validation slice
            val_args = Args()
            for attr in dir(args):
                if not attr.startswith('_'):
                    setattr(val_args, attr, getattr(args, attr))
            val_args.data_path = csv_path  # Use same full dataset
            
            # Get validation dataset - data loader will handle the split
            val_dataset, val_loader = data_provider(val_args, flag='val')
            
            val_losses = []
            val_losses_orig = []
            
            with torch.no_grad():
                for batch in val_loader:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                    batch_x = batch_x.float().to(device)
                    batch_y = batch_y.float().to(device) 
                    batch_x_mark = batch_x_mark.float().to(device)
                    batch_y_mark = batch_y_mark.float().to(device)
                    
                    # Get prediction
                    y_pred = model(batch_x, batch_x_mark, batch_y[:, :label_len, :], batch_y_mark)
                    y_true = batch_y[:, -pred_len:, :]
                    y_pred = y_pred[:, -pred_len:, :]
                    
                    # Compute validation loss on scaled data (same as training)
                    val_loss_scaled = criterion(y_pred, y_true).item()
                    val_losses.append(val_loss_scaled)
                    
                    # Also compute validation loss on original scale
                    try:
                        # Inverse transform both prediction and true values
                        pred_orig = val_dataset.scaler.inverse_transform(y_pred.cpu().numpy().reshape(-1, 3))
                        true_orig = val_dataset.scaler.inverse_transform(y_true.cpu().numpy().reshape(-1, 3))
                        val_loss_orig = np.mean((pred_orig - true_orig) ** 2)
                        val_losses_orig.append(val_loss_orig)
                    except:
                        val_losses_orig.append(val_loss_scaled)
            
            # Calculate average losses
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
            avg_val_loss_orig = np.mean(val_losses_orig) if val_losses_orig else float('inf')
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Train MSE: {avg_train_loss:.6f}, Val MSE (scaled): {avg_val_loss:.6f}, Val MSE (original): {avg_val_loss_orig:.6f}")
            
            # Early stopping if either loss gets very low (converged) - use scaled loss for thresholds
            if avg_train_loss < 0.01:
                logger.info(f"Early stopping at epoch {epoch+1}, Train MSE {avg_train_loss:.6f} < 0.01 (converged)")
                break
                
            if avg_val_loss < 0.01:
                logger.info(f"Early stopping at epoch {epoch+1}, Val MSE (scaled) {avg_val_loss:.6f} < 0.01 (converged)")
                break
            
            # Early stopping based on validation loss improvement - use scaled loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                logger.info(f"New best validation loss (scaled): {best_val_loss:.6f}")
            else:
                patience_counter += 1
                logger.info(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")
                
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss")
                break        # TRUE FORECASTING EVALUATION ON TEST SET
        logger.info("Starting true forecasting evaluation on test set")
        model.eval()
        
        # For true forecasting: use sequence from train_end-seq_len to train_end to predict test period
        # This uses points 1800-1900 to predict 1950-2000 (with validation happening on 1900-1950)
        forecast_start = train_end - seq_len  # 1900 - 100 = 1800
        forecast_seq_data = df_train.iloc[forecast_start:train_end].copy()  # Points 1800-1900
          # Create input sequences: ALL 6 features (covariates + targets) for 'M' mode
        input_all_features = forecast_seq_data[['cov1', 'cov2', 'cov3', 't1', 't2', 't3']].values
        
        # Apply sequence-level scaling (same as data loader) 
        seq_scaler = StandardScaler()
        seq_scaler.fit(input_all_features)  # Fit on all 6 features of input sequence
        input_all_features_scaled = seq_scaler.transform(input_all_features)
          # Create time features for input sequence
        input_dates = pd.to_datetime(forecast_seq_data['date'])
        input_time_features = np.column_stack([
            input_dates.dt.month,
            input_dates.dt.day, 
            input_dates.dt.weekday,
            input_dates.dt.hour
        ])
          # Create future time features for prediction
        future_dates = pd.date_range(start=input_dates.iloc[-1] + pd.Timedelta(hours=1), 
                                   periods=pred_len, freq='h')  # Changed from 'H' to 'h'
        future_time_features = np.column_stack([
            future_dates.month,
            future_dates.day,
            future_dates.weekday, 
            future_dates.hour
        ])
          # Prepare model inputs
        # x_enc: historical multivariate features [1, seq_len, 6]  
        x_enc = torch.FloatTensor(input_all_features_scaled).unsqueeze(0).to(device)
        x_mark_enc = torch.FloatTensor(input_time_features).unsqueeze(0).to(device)
        
        # x_dec: decoder input with future covariates [1, label_len + pred_len, 6]
        # Need to construct future multivariate data: known covariates + unknown targets (zeros)
        
        # Future covariates are known
        future_covariates_data = future_covariates  # [pred_len, 3]
        
        # Future targets are unknown - initialize as zeros  
        future_targets_zeros = np.zeros((pred_len, 3))  # [pred_len, 3]
        
        # Combine future covariates + future targets (zeros)
        future_multivariate = np.hstack([future_covariates_data, future_targets_zeros])  # [pred_len, 6]
        
        # Use last label_len points from training + future multivariate data
        decoder_input = np.vstack([
            input_all_features_scaled[-label_len:],  # last label_len from training [label_len, 6]
            future_multivariate  # future data [pred_len, 6]
        ])
        decoder_time_input = np.vstack([
            input_time_features[-label_len:],  # last label_len time features
            future_time_features  # future time features
        ])
        
        x_dec = torch.FloatTensor(decoder_input).unsqueeze(0).to(device)
        x_mark_dec = torch.FloatTensor(decoder_time_input).unsqueeze(0).to(device)        # Make prediction
        with torch.no_grad():
            forecast = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            forecast = forecast[:, -pred_len:, :]  # Only take the prediction part [1, pred_len, 6]
        
        # Extract only the target predictions (last 3 features: t1, t2, t3)
        forecast_targets = forecast[:, :, -3:]  # [1, pred_len, 3] - only targets
        forecast_targets_np = forecast_targets.cpu().numpy()[0]  # [pred_len, 3]
        
        # For comparison, we need to scale actual targets using the same scaler
        # The scaler was fitted on all 6 features, so we need to handle this carefully
        
        # Create a dummy array with actual targets in the right position
        actual_multivariate = np.hstack([
            future_covariates,  # [pred_len, 3] - covariates
            actual_targets_orig  # [pred_len, 3] - targets  
        ])  # [pred_len, 6]
        
        # Scale the full multivariate data
        actual_multivariate_scaled = seq_scaler.transform(actual_multivariate)
        
        # Extract only the scaled target part  
        actual_targets_scaled = actual_multivariate_scaled[:, -3:]  # [pred_len, 3]
        
        logger.info("Comparing forecast and targets in scaled space")
        mse_forecast = np.mean((forecast_targets_np - actual_targets_scaled) ** 2)
        logger.info(f"True forecasting MSE (scaled space): {mse_forecast:.6f}")# For plotting, try to get original scale data if possible
        # Reconstruct the full dataset to get test data
        date_full = pd.date_range(start='2020-01-01', periods=n_points, freq='h')
        df_full = pd.DataFrame({
            'date': date_full,
            'cov1': cov1,
            'cov2': cov2,
            'cov3': cov3,
            't1': t1,
            't2': t2,
            't3': t3        })
          # For plotting, we want to show original scale data
        # Try to inverse transform forecasts to original scale
        try:
            # For inverse transform, we need the full 6-feature forecast
            forecast_full_np = forecast.cpu().numpy()[0]  # [pred_len, 6]
            full_forecast_orig = seq_scaler.inverse_transform(forecast_full_np)
            forecast_plot = full_forecast_orig[:, -3:]  # Extract original scale targets [pred_len, 3]
            actual_plot = actual_targets_orig  # Original scale actual targets
            
            logger.info("Successfully inverse transformed forecasts to original scale for plotting")
            
        except Exception as e:
            logger.warning(f"Could not inverse transform forecasts: {e}")
            logger.info("Using scaled data for plotting comparison")
            forecast_plot = forecast_targets_np  # Use scaled forecast targets
            actual_plot = actual_targets_scaled  # Use scaled actual targets
        
        # Plot comparison for all three targets (last 10 points + forecast)
        plot_len = min(10, pred_len)
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        target_names = ['t1', 't2', 't3']
        
        for i in range(3):
            # Plot last 10 training points for context
            context_start = max(0, train_end - plot_len)
            axes[i].plot(range(context_start, train_end), 
                        eval(f't{i+1}')[context_start:train_end], 
                        'b-', label='Historical', alpha=0.7)
              # Plot actual future values (use unscaled data)
            axes[i].plot(range(train_end, train_end + plot_len), 
                        actual_plot[:plot_len, i], 
                        'g-', label='Actual Future', linewidth=2)
            
            # Plot forecasted values (use unscaled data)
            axes[i].plot(range(train_end, train_end + plot_len), 
                        forecast_plot[:plot_len, i], 
                        'r--', label='Forecast', linewidth=2)
            
            axes[i].axvline(x=train_end, color='black', linestyle=':', alpha=0.7, label='Forecast Start')
            mse_target = np.mean((forecast_plot[:, i] - actual_plot[:, i])**2)
            axes[i].set_title(f'True Forecasting: {target_names[i]} (MSE: {mse_target:.4f})')
            axes[i].set_xlabel('Time Step')
            axes[i].set_ylabel('Value')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pic/true_forecasting_evaluation.png', dpi=150)
        logger.info("Forecast plot saved to pic/true_forecasting_evaluation.png")
        plt.close()  # Close the figure to free memory
        
        # Clean up temp file
        if os.path.exists(csv_path):
            os.remove(csv_path)
        
        return mse_forecast, forecast_plot, actual_plot
