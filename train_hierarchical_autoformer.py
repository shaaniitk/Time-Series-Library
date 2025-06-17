"""
Training Script for Hierarchical Enhanced Autoformer

This script demonstrates how to train the Hierarchical Enhanced Autoformer
with multi-resolution processing and wavelet-based decomposition.
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
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Suppress warnings
warnings.filterwarnings('ignore')

# Imports
from models.HierarchicalEnhancedAutoformer import HierarchicalEnhancedAutoformer, create_hierarchical_autoformer
from utils.enhanced_losses import AdaptiveAutoformerLoss, FrequencyAwareLoss
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from utils.logger import logger, set_log_level
import logging


class HierarchicalAutoformerTrainer:
    """
    Training framework for Hierarchical Enhanced Autoformer.
    """
    
    def __init__(self, args):
        logger.info("Initializing HierarchicalAutoformerTrainer")
        self.args = args
        self.device = self._get_device()
        
        # Setup model
        self.model = self._build_model()
        
        # Setup data
        self.train_data, self.train_loader = data_provider(args, flag='train')
        self.vali_data, self.vali_loader = data_provider(args, flag='val')
        self.test_data, self.test_loader = data_provider(args, flag='test')
        
        # Setup loss function
        self.criterion = self._setup_loss_function()
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        
        # Tracking
        self.train_losses = []
        self.vali_losses = []
        self.hierarchy_metrics = []
        
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
        """Build Hierarchical Enhanced Autoformer model"""
        logger.info("Building HierarchicalEnhancedAutoformer")
        
        model = create_hierarchical_autoformer(
            configs=self.args,
            n_levels=self.args.n_hierarchy_levels,
            wavelet_type=self.args.wavelet_type,
            fusion_strategy=self.args.fusion_strategy,
            use_cross_attention=self.args.use_cross_attention
        )
        
        # Log model information
        hierarchy_info = model.get_hierarchy_info()
        logger.info(f"Hierarchical model info: {hierarchy_info}")
        
        return model.to(self.device)
    
    def _setup_loss_function(self):
        """Setup loss function"""
        logger.info(f"Setting up {self.args.loss_type} loss")
        
        if self.args.loss_type == 'adaptive':
            return AdaptiveAutoformerLoss(
                moving_avg=self.args.moving_avg,
                adaptive_weights=True
            )
        elif self.args.loss_type == 'frequency':
            return FrequencyAwareLoss(freq_weight=self.args.freq_weight)
        elif self.args.loss_type == 'mse':
            return nn.MSELoss()
        elif self.args.loss_type == 'mae':
            return nn.L1Loss()
        else:
            return nn.MSELoss()
    
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
        hierarchy_stats = []
        
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
            
            # Forward pass
            try:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            except Exception as e:
                logger.error(f"Forward pass failed: {e}")
                continue
            
            # Extract target
            f_dim = -1 if self.args.features == 'MS' else 0
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
            
            # Compute loss
            if hasattr(self.criterion, '__call__') and hasattr(self.criterion, 'forward'):
                loss_result = self.criterion(outputs, batch_y)
                if isinstance(loss_result, dict):
                    loss = loss_result.get('total_loss', loss_result.get('loss', 0))
                else:
                    loss = loss_result
            else:
                loss = self.criterion(outputs, batch_y)
            
            train_loss.append(loss.item())
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            
            self.optimizer.step()
            
            # Collect hierarchy statistics
            if self.args.log_hierarchy_stats and i % 100 == 0:
                hierarchy_stat = self._collect_hierarchy_statistics()
                hierarchy_stats.append(hierarchy_stat)
            
            # Logging
            if (i + 1) % 100 == 0:
                logger.info(f'Epoch: {epoch}, Step: {i+1}, Loss: {loss.item():.7f}')
        
        # Compute average metrics
        avg_train_loss = np.average(train_loss)
        
        if hierarchy_stats:
            avg_hierarchy_stats = self._average_hierarchy_stats(hierarchy_stats)
            logger.info(f"Epoch {epoch} hierarchy stats: {avg_hierarchy_stats}")
        
        return avg_train_loss
    
    def validate_epoch(self, epoch):
        """Validate one epoch"""
        self.model.eval()
        
        total_loss = []
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
                
                # Forward pass
                try:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                except Exception as e:
                    logger.warning(f"Validation forward pass failed: {e}")
                    continue
                
                # Extract target
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                # Compute loss
                if hasattr(self.criterion, '__call__') and hasattr(self.criterion, 'forward'):
                    loss_result = self.criterion(outputs, batch_y)
                    if isinstance(loss_result, dict):
                        loss = loss_result.get('total_loss', loss_result.get('loss', 0))
                    else:
                        loss = loss_result
                else:
                    loss = self.criterion(outputs, batch_y)
                
                total_loss.append(loss.item())
                
                # Store predictions for metrics
                predictions.append(outputs.detach().cpu().numpy())
                truths.append(batch_y.detach().cpu().numpy())
        
        # Compute validation metrics
        vali_loss = np.average(total_loss)
        
        # Compute standard metrics
        predictions = np.concatenate(predictions, axis=0)
        truths = np.concatenate(truths, axis=0)
        mae, mse, rmse, mape, mspe = metric(predictions, truths)
        
        logger.info(f'Validation - Loss: {vali_loss:.7f}, MAE: {mae:.7f}, RMSE: {rmse:.7f}')
        
        return vali_loss
    
    def _collect_hierarchy_statistics(self):
        """Collect statistics about hierarchical processing"""
        stats = {}
        
        # Get fusion weights if available
        if hasattr(self.model, 'fusion') and hasattr(self.model.fusion, 'fusion_weights'):
            fusion_weights = self.model.fusion.fusion_weights.detach().cpu().numpy()
            stats['fusion_weights'] = fusion_weights.tolist()
            stats['fusion_weights_entropy'] = -np.sum(fusion_weights * np.log(fusion_weights + 1e-8))
        
        # Get decomposer scale weights if available
        if hasattr(self.model, 'decomposer') and hasattr(self.model.decomposer, 'scale_weights'):
            scale_weights = self.model.decomposer.scale_weights.detach().cpu().numpy()
            stats['scale_weights'] = scale_weights.tolist()
        
        return stats
    
    def _average_hierarchy_stats(self, stats_list):
        """Average hierarchy statistics across batches"""
        if not stats_list:
            return {}
        
        # Get all keys from first dict
        keys = stats_list[0].keys()
        averaged = {}
        
        for key in keys:
            if key.endswith('_entropy'):
                # Average entropy values
                values = [stats[key] for stats in stats_list if key in stats]
                if values:
                    averaged[key] = np.mean(values)
            elif key.endswith('_weights'):
                # Average weight vectors
                weight_arrays = [np.array(stats[key]) for stats in stats_list if key in stats]
                if weight_arrays:
                    averaged[key] = np.mean(weight_arrays, axis=0).tolist()
        
        return averaged
    
    def train(self):
        """Main training loop"""
        logger.info(f"Starting hierarchical training for {self.args.train_epochs} epochs")
        
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
        logger.info(f"Hierarchical training completed in {total_time:.2f}s")
        
        # Load best model for testing
        best_model_path = self.args.checkpoints + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model
    
    def test(self):
        """Test the trained hierarchical model"""
        logger.info("Testing hierarchical model")
        
        self.model.eval()
        
        predictions = []
        truths = []
        
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
                
                # Forward pass
                try:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                except Exception as e:
                    logger.warning(f"Test forward pass failed: {e}")
                    continue
                
                # Extract target
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                # Store results
                predictions.append(outputs.detach().cpu().numpy())
                truths.append(batch_y.detach().cpu().numpy())
        
        # Concatenate results
        predictions = np.concatenate(predictions, axis=0)
        truths = np.concatenate(truths, axis=0)
        
        # Compute metrics
        mae, mse, rmse, mape, mspe = metric(predictions, truths)
        
        logger.info(f'Hierarchical Test Results:')
        logger.info(f'MAE: {mae:.7f}, MSE: {mse:.7f}, RMSE: {rmse:.7f}')
        logger.info(f'MAPE: {mape:.7f}, MSPE: {mspe:.7f}')
        
        # Save results
        result_dict = {
            'predictions': predictions.tolist(),
            'truths': truths.tolist(),
            'metrics': {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'mape': float(mape),
                'mspe': float(mspe)
            },
            'hierarchy_info': self.model.get_hierarchy_info(),
            'training_losses': self.train_losses,
            'validation_losses': self.vali_losses
        }
        
        # Save to file
        result_file = os.path.join(self.args.checkpoints, 'hierarchical_test_results.json')
        with open(result_file, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Results saved to {result_file}")
        
        return result_dict


def main():
    parser = argparse.ArgumentParser(description='Hierarchical Enhanced Autoformer')
    
    # Basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='HierarchicalAutoformer', help='model name')
    
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
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    
    # Hierarchical specific
    parser.add_argument('--n_hierarchy_levels', type=int, default=3, help='number of hierarchy levels')
    parser.add_argument('--wavelet_type', type=str, default='db4', help='wavelet type for decomposition')
    parser.add_argument('--fusion_strategy', type=str, default='weighted_concat', 
                       choices=['weighted_concat', 'weighted_sum', 'attention_fusion'])
    parser.add_argument('--use_cross_attention', action='store_true', help='use cross-resolution attention')
    parser.add_argument('--log_hierarchy_stats', action='store_true', help='log hierarchy statistics')
    
    # Loss function
    parser.add_argument('--loss_type', type=str, default='adaptive', 
                       choices=['adaptive', 'frequency', 'mse', 'mae'])
    parser.add_argument('--moving_avg', type=int, default=25, help='moving average window for decomposition')
    parser.add_argument('--freq_weight', type=float, default=0.1, help='frequency loss weight')
    
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
                                   f'hierarchical_{args.model_id}_{args.data}_{args.wavelet_type}_{args.fusion_strategy}')
    os.makedirs(args.checkpoints, exist_ok=True)
    
    logger.info(f"Hierarchical Experiment: {args.model_id}")
    logger.info(f"Data: {args.data}")
    logger.info(f"Hierarchy levels: {args.n_hierarchy_levels}")
    logger.info(f"Wavelet type: {args.wavelet_type}")
    logger.info(f"Fusion strategy: {args.fusion_strategy}")
    logger.info(f"Cross attention: {args.use_cross_attention}")
    
    if args.is_training:
        trainer = HierarchicalAutoformerTrainer(args)
        
        logger.info("Starting hierarchical training...")
        model = trainer.train()
        
        logger.info("Starting hierarchical testing...")
        results = trainer.test()
        
        logger.info("Hierarchical training and testing completed!")
    else:
        logger.info("Testing mode not implemented yet")


if __name__ == '__main__':
    main()
