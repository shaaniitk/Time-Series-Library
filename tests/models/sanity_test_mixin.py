import unittest
import torch
import numpy as np
import pandas as pd
import os
from data_provider.data_factory import data_provider
from models.TimesNet import Model as TimesNet
import matplotlib.pyplot as plt
from utils.logger import logger

class SanityTestMixin:
    def run_sanity_test(self, ModelClass, device='cpu', epochs=10):
        logger.info(f"Running sanity test for {ModelClass.__name__}")
        # Parameters
        seq_len = 100
        pred_len = 50
        n_points = 1000
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
        date_train = pd.date_range(start='2020-01-01', periods=train_end, freq='H')
        df_train = pd.DataFrame({
            'date': date_train,
            'cov1': cov1[:train_end],
            'cov2': cov2[:train_end],
            'cov3': cov3[:train_end],
            't1': t1[:train_end],
            't2': t2[:train_end],
            't3': t3[:train_end]
        })
        
        # Future covariates and actual targets for forecasting
        future_covariates = np.column_stack([
            cov1[train_end:],
            cov2[train_end:], 
            cov3[train_end:]
        ])
        actual_targets = np.column_stack([
            t1[train_end:],
            t2[train_end:],
            t3[train_end:]
        ])
        
        # Save training data to CSV
        csv_path = 'synthetic_timeseries_train.csv'
        df_train.to_csv(csv_path, index=False)
        # Only cov1, cov2, cov3 as features, t1, t2, t3 as targets
        target_cols = ['t1', 't2', 't3']        # Args for data provider
        class Args:
            pass
        args = Args()
        args.seq_len = seq_len
        args.label_len = label_len
        args.pred_len = pred_len
        args.d_model = d_model
        args.e_layers = e_layers
        args.d_ff = d_ff
        args.enc_in = 3  # Only cov1, cov2, cov3
        args.c_out = 3  # Predict all three targets
        args.embed = 'timeF'
        args.freq = 'h'
        args.dropout = 0.1
        args.task_name = 'long_term_forecast'
        args.features = 'M'
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
        args.use_multi_gpu = False  # Disable multi-GPU for simplicity
        args.scale = True  # Enable data scaling (default behavior)        # Use data provider
        dataset, loader = data_provider(args, flag='train')
        # Get validation data for proper evaluation
        val_dataset, val_loader = data_provider(args, flag='val')
        
        model = ModelClass(args).to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        
        # Training loop with early stopping
        for epoch in range(epochs):
            # Training phase
            model.train()
            epoch_losses = []
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
                epoch_losses.append(loss.item())
            
            # Validation phase
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                    batch_x = batch_x.float().to(device)
                    batch_y = batch_y.float().to(device)
                    batch_x_mark = batch_x_mark.float().to(device)
                    batch_y_mark = batch_y_mark.float().to(device)
                    y_pred = model(batch_x, batch_x_mark, batch_y[:, :label_len, :], batch_y_mark)
                    y_true = batch_y[:, -pred_len:, :]
                    y_pred = y_pred[:, -pred_len:, :]
                    loss = criterion(y_pred, y_true)
                    val_losses.append(loss.item())
            
            # Calculate average epoch losses
            avg_train_loss = np.mean(epoch_losses)
            avg_val_loss = np.mean(val_losses) if val_losses else avg_train_loss
            logger.info(f"Epoch {epoch+1}/{epochs}, Train MSE: {avg_train_loss:.6f}, Val MSE: {avg_val_loss:.6f}")
            
            # Early stopping based on validation loss
            if avg_val_loss < 0.01:
                logger.info(f"Early stopping at epoch {epoch+1}, Val MSE {avg_val_loss:.6f} < 0.01")
                break
        
        # TRUE FORECASTING EVALUATION
        logger.info("Starting true forecasting evaluation")
        model.eval()
        
        # Get the last seq_len points from training data for model input
        last_seq_data = df_train.iloc[-seq_len:].copy()
        
        # Create input sequences (last seq_len points from training)
        input_features = last_seq_data[['cov1', 'cov2', 'cov3']].values
        input_targets = last_seq_data[['t1', 't2', 't3']].values
        
        # Create time features for input sequence
        input_dates = pd.to_datetime(last_seq_data['date'])
        input_time_features = np.column_stack([
            input_dates.dt.month,
            input_dates.dt.day, 
            input_dates.dt.weekday,
            input_dates.dt.hour
        ])
        
        # Create future time features for prediction
        future_dates = pd.date_range(start=input_dates.iloc[-1] + pd.Timedelta(hours=1), 
                                   periods=pred_len, freq='H')
        future_time_features = np.column_stack([
            future_dates.month,
            future_dates.day,
            future_dates.weekday, 
            future_dates.hour
        ])
        
        # Prepare model inputs
        # x_enc: historical covariates [1, seq_len, 3]
        x_enc = torch.FloatTensor(input_features).unsqueeze(0).to(device)
        x_mark_enc = torch.FloatTensor(input_time_features).unsqueeze(0).to(device)
        
        # x_dec: decoder input with future covariates [1, label_len + pred_len, 3] 
        # Use last label_len points from training + future covariates
        decoder_input = np.vstack([
            input_features[-label_len:],  # last label_len from training
            future_covariates  # future covariates for prediction
        ])
        decoder_time_input = np.vstack([
            input_time_features[-label_len:],  # last label_len time features
            future_time_features  # future time features
        ])
        
        x_dec = torch.FloatTensor(decoder_input).unsqueeze(0).to(device)
        x_mark_dec = torch.FloatTensor(decoder_time_input).unsqueeze(0).to(device)
          # Make prediction
        with torch.no_grad():
            forecast = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            forecast = forecast[:, -pred_len:, :]  # Only take the prediction part
        
        # IMPORTANT: Apply inverse scaling for proper evaluation
        forecast_np = forecast.cpu().numpy()[0]  # [pred_len, 3]
        
        # Get the scaler from the dataset and apply inverse transform
        if hasattr(dataset, 'scaler') and dataset.scale:
            logger.info("Applying inverse scaling to forecasts and targets")
            # The dataset scaler was fitted on features only, but we need target scaling
            # Since we're using 'M' features, the scaler includes all columns (features + targets)
            # We need to create properly shaped arrays for inverse transform
            
            # Create dummy arrays with the same structure as the training data
            dummy_forecast = np.zeros((pred_len, 6))  # 3 features + 3 targets
            dummy_actual = np.zeros((pred_len, 6))
            
            # Fill in the target columns (assuming targets are last 3 columns)
            dummy_forecast[:, 3:6] = forecast_np  # targets: columns 3,4,5
            dummy_actual[:, 3:6] = actual_targets
            
            # Apply inverse scaling
            forecast_unscaled = dataset.scaler.inverse_transform(dummy_forecast)[:, 3:6]
            actual_unscaled = dataset.scaler.inverse_transform(dummy_actual)[:, 3:6]
            
            # Calculate MSE on unscaled data
            mse_forecast = np.mean((forecast_unscaled - actual_unscaled) ** 2)
            logger.info(f"True forecasting MSE (unscaled): {mse_forecast:.6f}")
            
            # Use unscaled data for plotting
            forecast_plot = forecast_unscaled
            actual_plot = actual_unscaled
        else:
            logger.info("No scaling applied - using raw forecasts")
            mse_forecast = np.mean((forecast_np - actual_targets) ** 2)
            logger.info(f"True forecasting MSE: {mse_forecast:.6f}")
            forecast_plot = forecast_np
            actual_plot = actual_targets
        
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
