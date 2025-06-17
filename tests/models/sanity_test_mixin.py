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
    def run_sanity_test(self, ModelClass, device='cpu', epochs=10, model_config=None):
        logger.info(f"Running sanity test for {ModelClass.__name__}")
          # Parameters
        seq_len = 100
        pred_len = 50
        n_points = 2000  # Increased for better train/val split
        
        # Model configuration - adjust based on model type for fair comparison
        if model_config:
            d_model = model_config.get('d_model', 16)
            e_layers = model_config.get('e_layers', 2)
            d_layers = model_config.get('d_layers', 1)
            d_ff = model_config.get('d_ff', 32)
            n_heads = model_config.get('n_heads', 2)
        else:
            # Default small configuration
            d_model = 16
            e_layers = 2
            d_layers = 1
            d_ff = 32
            n_heads = 2
            
        enc_in = 6  # All 6 features (3 covariates + 3 targets) for multivariate mode
        c_out = 6   # Output all 6 features to match input dimension
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
        t3 = np.sin(X2 - X)        # Split data for the full dataset that will be used by data loader
        # We need to create the FULL dataset (2000 points) but the data loader will split it
        
        # Create the complete dataset for data loader
        date_full = pd.date_range(start='2020-01-01', periods=n_points, freq='h')
        df_full = pd.DataFrame({
            'date': date_full,
            'cov1': cov1,
            'cov2': cov2,
            'cov3': cov3,
            't1': t1,
            't2': t2,
            't3': t3        })
        
        # Correct time series split based on validation requirements
        # Training: 0 to 1799 (1800 points)
        # Validation: 1800 to 1949 (150 points) - exactly seq_len + pred_len for 1 window
        # Test: 1950 to 1999 (50 points)
        
        train_end = 1800  # Training ends at point 1799 (1800 points: 0-1799)
        val_period = seq_len + pred_len  # 150 points needed for validation
        test_period = pred_len  # 50 points for test
        
        val_start = train_end  # 1800 (validation starts immediately after training)
        val_end = val_start + val_period  # 1950
          # Test starts where validation ends  
        test_start = val_end  # 1950
        
        # Future covariates and actual targets for final test forecasting (test period only)
        future_covariates = np.column_stack([
            cov1[test_start:],
            cov2[test_start:], 
            cov3[test_start:]
        ])
        actual_targets_orig = np.column_stack([
            t1[test_start:],
            t2[test_start:],
            t3[test_start:]
        ])
        
        logger.info(f"Training data: points 0-{train_end-1} ({train_end} points)")
        logger.info(f"Validation data: points {val_start}-{val_end-1} ({val_end - val_start} points)")
        logger.info(f"Test data: points {test_start}-{n_points-1} ({n_points - test_start} points)")# Save the FULL dataset for data loader  
        # The data loader will internally split based on borders
        csv_path = 'synthetic_timeseries_full.csv'
        df_full.to_csv(csv_path, index=False)  # Full dataset
        
        # Create training-only subset for reference
        df_train_only = df_full.iloc[:train_end]  # Points 0 to 1800
          # Only cov1, cov2, cov3 as features, t1, t2, t3 as targets
        target_cols = ['t1', 't2', 't3']
          # Args for data provider
        class Args:
            pass
        args = Args()
        args.seq_len = seq_len
        args.label_len = label_len
        args.pred_len = pred_len
        args.d_model = d_model
        args.e_layers = e_layers
        args.d_layers = d_layers
        args.d_ff = d_ff
        args.enc_in = enc_in  # All 6 features: 3 covariates + 3 targets in 'M' mode
        args.dec_in = enc_in  # Same as enc_in for consistency
        args.c_out = c_out   # Output all 6 features to match input dimension
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
        args.use_multi_gpu = False  # Disable multi-GPU for simplicity
        args.scale = True  # Enable data scaling (default behavior)
        
        # Model-specific parameters (for compatibility with different models)
        args.n_heads = n_heads  # Dynamic based on model requirements
        args.factor = 1  # Attention factor
        args.activation = 'gelu'  # Activation function
        args.output_attention = False  # Output attention weights
        args.moving_avg = 25  # For Autoformer decomposition
        
        # Configurable split parameters for data loader
        args.validation_length = 150  # v parameter
        args.test_length = 50         # t parameter
        
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
                        # FIX: Proper inverse transform handling
                        batch_size, pred_len, n_features = y_pred.shape
                        
                        # Reshape properly for inverse transform: [batch_size * pred_len, n_features]
                        pred_reshaped = y_pred.cpu().numpy().reshape(-1, n_features)
                        true_reshaped = y_true.cpu().numpy().reshape(-1, n_features)
                        
                        # Apply inverse transform
                        pred_orig = val_dataset.inverse_transform(pred_reshaped)
                        true_orig = val_dataset.inverse_transform(true_reshaped)
                        
                        # Reshape back: [batch_size, pred_len, n_features]
                        pred_orig = pred_orig.reshape(batch_size, pred_len, n_features)
                        true_orig = true_orig.reshape(batch_size, pred_len, n_features)
                        
                        # FIX: Compute MSE only on target features (last 3 if 6 features, all if 3)
                        if n_features == 6:  # Multivariate case: 3 covariates + 3 targets
                            target_pred = pred_orig[:, :, -3:]  # Last 3 features are targets
                            target_true = true_orig[:, :, -3:]
                        else:  # All features are targets
                            target_pred = pred_orig
                            target_true = true_orig
                            
                        val_loss_orig = np.mean((target_pred - target_true) ** 2)
                        val_losses_orig.append(val_loss_orig)
                        
                    except Exception as e:
                        logger.warning(f"Could not compute original scale validation loss: {e}")
                        # FIX: Don't silently use scaled loss - use NaN to indicate failure
                        val_losses_orig.append(np.nan)
            
            # Calculate average losses
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
            
            # FIX: Handle NaN values properly for original scale validation loss
            valid_val_losses_orig = [loss for loss in val_losses_orig if not np.isnan(loss)]
            avg_val_loss_orig = np.mean(valid_val_losses_orig) if valid_val_losses_orig else np.nan
            
            # Display losses with proper NaN handling
            if np.isnan(avg_val_loss_orig):
                logger.info(f"Epoch {epoch+1}/{epochs}, Train MSE: {avg_train_loss:.6f}, Val MSE (scaled): {avg_val_loss:.6f}, Val MSE (original): N/A")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs}, Train MSE: {avg_train_loss:.6f}, Val MSE (scaled): {avg_val_loss:.6f}, Val MSE (original): {avg_val_loss_orig:.6f}")
            
            # Early stopping if either loss gets very low (converged) - use scaled loss for thresholds
            if avg_train_loss < 0.01:
                logger.info(f"Early stopping at epoch {epoch+1}, Train MSE {avg_train_loss:.6f} < 0.01 (converged)")
                break
                
            if avg_val_loss < 0.01:
                logger.info(f"Early stopping at epoch {epoch+1}, Val MSE (scaled) {avg_val_loss:.6f} < 0.01 (converged)")
                break
            
            # FIX: Early stopping based on validation loss improvement - use scaled loss consistently            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                logger.info(f"New best validation loss (scaled): {best_val_loss:.6f}")
            else:
                patience_counter += 1
                logger.info(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss")
                    break
        
        # TEST EVALUATION USING DATA LOADER (same approach as validation)
        logger.info("Starting test evaluation using data loader")
        model.eval()
        
        # Create test dataset using the same approach as validation
        test_args = Args()
        for attr in dir(args):
            if not attr.startswith('_'):
                setattr(test_args, attr, getattr(args, attr))
        test_args.data_path = csv_path  # Use same full dataset
        
        # Get test dataset - data loader will handle the split using borders
        test_dataset, test_loader = data_provider(test_args, flag='test')
        
        test_losses = []
        test_losses_orig = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device) 
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                
                # Get prediction (same way as training/validation)
                y_pred = model(batch_x, batch_x_mark, batch_y[:, :label_len, :], batch_y_mark)
                y_true = batch_y[:, -pred_len:, :]
                y_pred = y_pred[:, -pred_len:, :]
                
                # Compute test loss on scaled data (same as training/validation)
                test_loss_scaled = criterion(y_pred, y_true).item()
                test_losses.append(test_loss_scaled)
                
                # Also compute test loss on original scale
                try:
                    # FIX: Proper inverse transform handling
                    batch_size, pred_len, n_features = y_pred.shape
                    
                    # Reshape properly for inverse transform: [batch_size * pred_len, n_features]
                    pred_reshaped = y_pred.cpu().numpy().reshape(-1, n_features)
                    true_reshaped = y_true.cpu().numpy().reshape(-1, n_features)
                    
                    # Apply inverse transform
                    pred_orig = test_dataset.inverse_transform(pred_reshaped)
                    true_orig = test_dataset.inverse_transform(true_reshaped)
                    
                    # Reshape back: [batch_size, pred_len, n_features]
                    pred_orig = pred_orig.reshape(batch_size, pred_len, n_features)
                    true_orig = true_orig.reshape(batch_size, pred_len, n_features)
                    
                    # FIX: Compute MSE only on target features (last 3 if 6 features, all if 3)
                    if n_features == 6:  # Multivariate case: 3 covariates + 3 targets
                        target_pred = pred_orig[:, :, -3:]  # Last 3 features are targets
                        target_true = true_orig[:, :, -3:]
                    else:  # All features are targets
                        target_pred = pred_orig
                        target_true = true_orig
                        
                    test_loss_orig = np.mean((target_pred - target_true) ** 2)
                    test_losses_orig.append(test_loss_orig)
                    
                except Exception as e:
                    logger.warning(f"Could not compute original scale test loss: {e}")
                    # FIX: Don't silently use scaled loss - use NaN to indicate failure
                    test_losses_orig.append(np.nan)
        
        # Calculate average test losses
        avg_test_loss = np.mean(test_losses) if test_losses else float('inf')
        
        # FIX: Handle NaN values properly for original scale test loss
        valid_test_losses_orig = [loss for loss in test_losses_orig if not np.isnan(loss)]
        avg_test_loss_orig = np.mean(valid_test_losses_orig) if valid_test_losses_orig else np.nan
        
        # Display test losses with proper NaN handling
        if np.isnan(avg_test_loss_orig):
            logger.info(f"Test MSE (scaled): {avg_test_loss:.6f}, Test MSE (original): N/A")
        else:
            logger.info(f"Test MSE (scaled): {avg_test_loss:.6f}, Test MSE (original): {avg_test_loss_orig:.6f}")
          # For plotting and return values, get the first batch predictions
        with torch.no_grad():
            for batch in test_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device) 
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                
                # Get prediction for plotting
                y_pred = model(batch_x, batch_x_mark, batch_y[:, :label_len, :], batch_y_mark)
                y_true = batch_y[:, -pred_len:, :]
                y_pred = y_pred[:, -pred_len:, :]
                
                # Use first batch for plotting
                forecast_plot = y_pred[0].cpu().numpy()  # [pred_len, 6]
                actual_plot = y_true[0].cpu().numpy()    # [pred_len, 6]
                break
        
        # Try to get original scale data for plotting
        try:
            # FIX: Proper inverse transform for plotting
            pred_len, n_features = forecast_plot.shape
            
            # Reshape for inverse transform
            forecast_reshaped = forecast_plot.reshape(-1, n_features)
            actual_reshaped = actual_plot.reshape(-1, n_features)
            
            # Apply inverse transform
            forecast_orig = test_dataset.inverse_transform(forecast_reshaped)
            actual_orig = test_dataset.inverse_transform(actual_reshaped)
            
            # Reshape back
            forecast_plot = forecast_orig.reshape(pred_len, n_features)
            actual_plot = actual_orig.reshape(pred_len, n_features)
            
            logger.info("Successfully inverse transformed data to original scale for plotting")
        except Exception as e:
            logger.warning(f"Could not inverse transform data: {e}. Using scaled data for plotting")
          # FIX: Calculate final MSE for the test - use only target features
        if forecast_plot.shape[-1] == 6:  # Multivariate case: 3 covariates + 3 targets
            target_forecast = forecast_plot[:, -3:]  # Last 3 features are targets
            target_actual = actual_plot[:, -3:]
        else:  # All features are targets
            target_forecast = forecast_plot
            target_actual = actual_plot
            
        mse_forecast = np.mean((target_forecast - target_actual) ** 2)
        logger.info(f"Final test MSE (targets only): {mse_forecast:.6f}")
        
        # Plot comparison for targets (simplified approach using the data we already have)
        plot_len = min(10, pred_len)
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        target_names = ['Target 1', 'Target 2', 'Target 3']
        
        # Use only the target columns for plotting (last 3 columns if 6 features, or all if 3 features)
        n_features = forecast_plot.shape[-1]
        if n_features == 6:
            # Multivariate case - use last 3 features as targets
            forecast_targets = forecast_plot[:, -3:]
            actual_targets = actual_plot[:, -3:]
        else:
            # Single variate case - use all features
            forecast_targets = forecast_plot
            actual_targets = actual_plot
        
        for i in range(min(3, forecast_targets.shape[-1])):
            # Plot actual future values 
            axes[i].plot(range(plot_len), 
                        actual_targets[:plot_len, i], 
                        'g-', label='Actual Future', linewidth=2)
              # Plot forecasted values
            axes[i].plot(range(plot_len), 
                        forecast_targets[:plot_len, i], 
                        'r--', label='Forecast', linewidth=2)
            
            mse_target = np.mean((forecast_targets[:, i] - actual_targets[:, i])**2)
            axes[i].set_title(f'{target_names[i]} (MSE: {mse_target:.4f})')
            axes[i].set_xlabel('Time Step')
            axes[i].set_ylabel('Value')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Ensure pic directory exists
        os.makedirs('pic', exist_ok=True)
        plt.savefig('pic/true_forecasting_evaluation.png', dpi=150)
        logger.info("Forecast plot saved to pic/true_forecasting_evaluation.png")
        plt.close()  # Close the figure to free memory
        
        # Clean up temp file
        if os.path.exists(csv_path):
            os.remove(csv_path)
        
        return mse_forecast, forecast_plot, actual_plot
