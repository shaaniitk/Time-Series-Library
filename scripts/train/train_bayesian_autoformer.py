"""
Training Script for Bayesian Enhanced Autoformer with Uncertainty Quantification

This script demonstrates how to train and evaluate the Bayesian Enhanced Autoformer
with various loss functions including quantile loss, providing uncertainty estimates
and certainty measures for predictions.
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
import json

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

# Suppress warnings
warnings.filterwarnings('ignore')

# Imports
from models.BayesianEnhancedAutoformer import BayesianEnhancedAutoformer
from utils.bayesian_losses import (
    BayesianAdaptiveLoss, BayesianQuantileLoss, BayesianFrequencyAwareLoss,
    UncertaintyCalibrationLoss, create_bayesian_loss
)
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from utils.logger import logger, set_log_level
import logging


class BayesianAutoformerTrainer:
    """
    Training framework for Bayesian Enhanced Autoformer with uncertainty quantification.
    """
    
    def __init__(self, args):
        logger.info("Initializing BayesianAutoformerTrainer")
        self.args = args
        self.device = self._get_device()
        
        # Setup model
        self.model = self._build_model()
        
        # Setup data
        self.train_data, self.train_loader = data_provider(args, flag='train')
        self.vali_data, self.vali_loader = data_provider(args, flag='val')
        self.test_data, self.test_loader = data_provider(args, flag='test')
        
        # Setup loss functions
        self.criterion = self._setup_loss_function()
        self.calibration_loss = UncertaintyCalibrationLoss()
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        
        # Tracking
        self.train_losses = []
        self.vali_losses = []
        self.uncertainty_metrics = []
        
    def _get_device(self):
        """Get device for training"""
        if self.args.use_gpu and torch.cuda.is_available():
            device = torch.device('cuda:{}'.format(self.args.gpu))
            logger.info(f"Using GPU: {device}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU")
        return device
    
    def _build_model(self):
        """Build Bayesian Enhanced Autoformer model"""
        logger.info("Building BayesianEnhancedAutoformer")
        
        model = BayesianEnhancedAutoformer(
            configs=self.args,
            uncertainty_method=self.args.uncertainty_method,
            n_samples=self.args.n_uncertainty_samples,
            bayesian_layers=self.args.bayesian_layers,
            kl_weight=self.args.kl_weight
        )
        
        # Enable quantile mode if using quantile loss
        if self.args.loss_type == 'quantile' and hasattr(self.args, 'quantile_levels'):
            model.enable_quantile_mode(self.args.quantile_levels)
        
        return model.to(self.device)
    
    def _setup_loss_function(self):
        """Setup Bayesian loss function"""
        logger.info(f"Setting up {self.args.loss_type} Bayesian loss")
        
        loss_kwargs = {
            'kl_weight': self.args.kl_weight,
            'uncertainty_weight': self.args.uncertainty_weight
        }
        
        if self.args.loss_type == 'quantile':
            loss_kwargs['quantiles'] = getattr(self.args, 'quantile_levels', [0.1, 0.5, 0.9])
        elif self.args.loss_type == 'frequency':
            loss_kwargs['freq_weight'] = getattr(self.args, 'freq_weight', 0.1)
        
        return create_bayesian_loss(self.args.loss_type, **loss_kwargs)
    
    def _setup_optimizer(self):
        """Setup optimizer"""
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        
        train_loss = []
        uncertainty_stats = []
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            
            # Move to device
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)
            
            # Decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            
            # Forward pass with uncertainty
            pred_result = self.model(
                batch_x, batch_x_mark, dec_inp, batch_y_mark,
                return_uncertainty=True,
                detailed_uncertainty=self.args.detailed_uncertainty
            )
            
            # Extract target
            f_dim = -1 if self.args.features == 'MS' else 0
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
            
            # Compute main loss
            loss_result = self.criterion(self.model, pred_result, batch_y)
            
            # Add calibration loss
            if self.args.use_calibration_loss:
                calib_loss = self.calibration_loss(pred_result, batch_y)
                loss_result['total_loss'] = loss_result['total_loss'] + self.args.calibration_weight * calib_loss
                loss_result['calibration_loss'] = calib_loss
            
            loss = loss_result['total_loss']
            train_loss.append(loss.item())
            
            # Collect uncertainty statistics
            if 'uncertainty' in pred_result:
                uncertainty_summary = self.model.get_uncertainty_summary(pred_result)
                uncertainty_stats.append(uncertainty_summary)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            
            self.optimizer.step()
            
            # Logging
            if (i + 1) % 100 == 0:
                logger.info(f'Epoch: {epoch}, Step: {i+1}, Loss: {loss.item():.7f}')
                
                # Log loss components
                for key, value in loss_result.items():
                    if isinstance(value, torch.Tensor):
                        logger.debug(f'  {key}: {value.item():.7f}')
        
        # Compute average metrics
        avg_train_loss = np.average(train_loss)
        
        if uncertainty_stats:
            avg_uncertainty_stats = self._average_uncertainty_stats(uncertainty_stats)
            logger.info(f"Epoch {epoch} uncertainty stats: {avg_uncertainty_stats}")
        
        return avg_train_loss
    
    def validate_epoch(self, epoch):
        """Validate one epoch"""
        self.model.eval()
        
        total_loss = []
        uncertainty_stats = []
        predictions = []
        truths = []
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.vali_loader):
                # Move to device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # Forward pass with uncertainty
                pred_result = self.model(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark,
                    return_uncertainty=True,
                    detailed_uncertainty=True
                )
                
                # Extract target
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                # Compute loss
                loss_result = self.criterion(self.model, pred_result, batch_y)
                
                if self.args.use_calibration_loss:
                    calib_loss = self.calibration_loss(pred_result, batch_y)
                    loss_result['total_loss'] = loss_result['total_loss'] + self.args.calibration_weight * calib_loss
                
                total_loss.append(loss_result['total_loss'].item())
                
                # Collect uncertainty statistics
                if 'uncertainty' in pred_result:
                    uncertainty_summary = self.model.get_uncertainty_summary(pred_result)
                    uncertainty_stats.append(uncertainty_summary)
                
                # Store predictions for metrics - scale ground truth to match predictions
                pred_np = pred_result['prediction'].detach().cpu().numpy()
                true_np = batch_y.detach().cpu().numpy()
                
                # Scale the ground truth to match model outputs (model outputs are scaled)
                if hasattr(self.vali_data, 'target_scaler') and self.vali_data.target_scaler is not None:
                    true_np = self.vali_data.target_scaler.transform(
                        true_np.reshape(-1, true_np.shape[-1])
                    ).reshape(true_np.shape)
                
                predictions.append(pred_np)
                truths.append(true_np)
        
        # Compute validation metrics
        vali_loss = np.average(total_loss)
        
        # Compute standard metrics
        predictions = np.concatenate(predictions, axis=0)
        truths = np.concatenate(truths, axis=0)
        mae, mse, rmse, mape, mspe = metric(predictions, truths)
        
        # Average uncertainty statistics
        if uncertainty_stats:
            avg_uncertainty_stats = self._average_uncertainty_stats(uncertainty_stats)
            
            # Store for tracking
            self.uncertainty_metrics.append({
                'epoch': epoch,
                'vali_loss': vali_loss,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                **avg_uncertainty_stats
            })
        
        logger.info(f'Validation - Loss: {vali_loss:.7f}, MAE: {mae:.7f}, RMSE: {rmse:.7f}')
        
        return vali_loss
    
    def _average_uncertainty_stats(self, stats_list):
        """Average uncertainty statistics across batches"""
        if not stats_list:
            return {}
        
        # Get all keys from first dict
        keys = stats_list[0].keys()
        averaged = {}
        
        for key in keys:
            values = [stats[key] for stats in stats_list if key in stats]
            if values:
                averaged[key] = np.mean(values)
        
        return averaged
    
    def train(self):
        """Main training loop"""
        logger.info(f"Starting training for {self.args.train_epochs} epochs")
        
        # NOTE: Scaling consistency fix
        # - Model predictions are in scaled space (trained on scaled data)
        # - Validation/test ground truth is unscaled (to avoid data leakage)  
        # - We scale ground truth during loss/metric computation to match predictions
        
        time_now = time.time()
        
        for epoch in range(self.args.train_epochs):
            epoch_time = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            vali_loss = self.validate_epoch(epoch)
            
            # Store losses
            self.train_losses.append(train_loss)
            self.vali_losses.append(vali_loss)
            
            logger.info(f"Epoch: {epoch+1} cost time: {time.time()-epoch_time:.2f}s")
            logger.info(f"Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f}")
            
            # Early stopping
            self.early_stopping(vali_loss, self.model, self.args.checkpoints)
            if self.early_stopping.early_stop:
                logger.info("Early stopping")
                break
            
            # Adjust learning rate
            adjust_learning_rate(self.optimizer, epoch + 1, self.args)
        
        total_time = time.time() - time_now
        logger.info(f"Training completed in {total_time:.2f}s")
        
        # Load best model for testing
        best_model_path = self.args.checkpoints + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model
    
    def test(self):
        """Test the trained model with uncertainty quantification"""
        logger.info("Testing model with uncertainty quantification")
        
        self.model.eval()
        
        predictions = []
        truths = []
        uncertainties = []
        confidence_intervals = []
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.test_loader):
                # Move to device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # Forward pass with uncertainty
                pred_result = self.model(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark,
                    return_uncertainty=True,
                    detailed_uncertainty=True
                )
                
                # Extract target
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                # Store results - scale ground truth to match predictions
                pred_np = pred_result['prediction'].detach().cpu().numpy()
                true_np = batch_y.detach().cpu().numpy()
                
                # Scale the ground truth to match model outputs (model outputs are scaled)
                if hasattr(self.test_data, 'target_scaler') and self.test_data.target_scaler is not None:
                    true_np = self.test_data.target_scaler.transform(
                        true_np.reshape(-1, true_np.shape[-1])
                    ).reshape(true_np.shape)
                
                predictions.append(pred_np)
                truths.append(true_np)
                
                if 'uncertainty' in pred_result:
                    uncertainties.append(pred_result['uncertainty'].detach().cpu().numpy())
                
                if 'confidence_intervals' in pred_result:
                    batch_intervals = {}
                    for conf_level, conf_data in pred_result['confidence_intervals'].items():
                        batch_intervals[conf_level] = {
                            'lower': conf_data['lower'].detach().cpu().numpy(),
                            'upper': conf_data['upper'].detach().cpu().numpy(),
                            'width': conf_data['width'].detach().cpu().numpy()
                        }
                    confidence_intervals.append(batch_intervals)
        
        # Concatenate results
        predictions = np.concatenate(predictions, axis=0)
        truths = np.concatenate(truths, axis=0)
        
        if uncertainties:
            uncertainties = np.concatenate(uncertainties, axis=0)
        
        # Compute metrics
        mae, mse, rmse, mape, mspe = metric(predictions, truths)
        
        logger.info(f'Test Results - MAE: {mae:.7f}, MSE: {mse:.7f}, RMSE: {rmse:.7f}')
        logger.info(f'               MAPE: {mape:.7f}, MSPE: {mspe:.7f}')
        
        # Uncertainty analysis
        if uncertainties is not None:
            mean_uncertainty = np.mean(uncertainties)
            uncertainty_std = np.std(uncertainties)
            
            logger.info(f'Uncertainty - Mean: {mean_uncertainty:.7f}, Std: {uncertainty_std:.7f}')
            
            # Compute uncertainty calibration metrics
            pred_errors = np.abs(predictions - truths)
            error_uncertainty_corr = np.corrcoef(pred_errors.flatten(), uncertainties.flatten())[0, 1]
            
            logger.info(f'Error-Uncertainty Correlation: {error_uncertainty_corr:.4f}')
        
        # Save results
        result_dict = {
            'predictions': predictions.tolist(),
            'truths': truths.tolist(),
            'uncertainties': uncertainties.tolist() if uncertainties is not None else None,
            'metrics': {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'mape': float(mape),
                'mspe': float(mspe)
            },
            'uncertainty_metrics': {
                'mean_uncertainty': float(mean_uncertainty) if uncertainties is not None else None,
                'uncertainty_std': float(uncertainty_std) if uncertainties is not None else None,
                'error_uncertainty_correlation': float(error_uncertainty_corr) if uncertainties is not None else None
            }
        }
        
        # Save to file
        result_file = os.path.join(self.args.checkpoints, 'bayesian_test_results.json')
        with open(result_file, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Results saved to {result_file}")
        
        return result_dict


def main():
    parser = argparse.ArgumentParser(description='Bayesian Enhanced Autoformer')
    
    # Basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='BayesianAutoformer', help='model name')
    
    # Data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M', choices=['M', 'S', 'MS'], help='forecasting task')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    
    # Forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
    
    # Model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    
    # Bayesian specific
    parser.add_argument('--uncertainty_method', type=str, default='bayesian', choices=['bayesian', 'dropout'])
    parser.add_argument('--n_uncertainty_samples', type=int, default=50)
    parser.add_argument('--bayesian_layers', type=str, nargs='+', default=['projection'], 
                       choices=['projection', 'encoder', 'decoder'])
    parser.add_argument('--kl_weight', type=float, default=1e-5)
    parser.add_argument('--uncertainty_weight', type=float, default=0.1)
    parser.add_argument('--detailed_uncertainty', action='store_true')
    
    # Loss function
    parser.add_argument('--loss_type', type=str, default='adaptive', 
                       choices=['adaptive', 'quantile', 'frequency', 'mse', 'mae'])
    parser.add_argument('--quantile_levels', type=float, nargs='+', default=[0.1, 0.5, 0.9])
    parser.add_argument('--freq_weight', type=float, default=0.1)
    parser.add_argument('--use_calibration_loss', action='store_true')
    parser.add_argument('--calibration_weight', type=float, default=0.1)
    
    # Optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='gradient clipping')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    
    args = parser.parse_args()
    
    # Setup logging
    set_log_level(logging.INFO)
    
    # Create checkpoint directory
    args.checkpoints = os.path.join(args.checkpoints, 
                                   f'bayesian_{args.model_id}_{args.data}_{args.uncertainty_method}_{args.loss_type}')
    os.makedirs(args.checkpoints, exist_ok=True)
    
    logger.info(f"Experiment: {args.model_id}")
    logger.info(f"Data: {args.data}")
    logger.info(f"Uncertainty method: {args.uncertainty_method}")
    logger.info(f"Loss type: {args.loss_type}")
    logger.info(f"Bayesian layers: {args.bayesian_layers}")
    
    if args.is_training:
        trainer = BayesianAutoformerTrainer(args)
        
        logger.info("Starting training...")
        model = trainer.train()
        
        logger.info("Starting testing...")
        results = trainer.test()
        
        logger.info("Training and testing completed!")
    else:
        logger.info("Testing mode not implemented yet")


if __name__ == '__main__':
    main()
