#!/usr/bin/env python3
"""
Enhanced Training Script with KL Loss Tuning for BayesianEnhance    # Model args - Heavy model configuration
    parser.add_argument('--enc_in', type=int, default=7, help='Encoder input size (auto-detected)')
    parser.add_argument('--dec_in', type=int, default=7, help='Decoder input size (auto-detected)')
    parser.add_argument('--c_out', type=int, default=1, help='Output size (auto-detected)')
    parser.add_argument('--d_model', type=int, default=128)  # Increased model dimension
    parser.add_argument('--n_heads', type=int, default=8)    # More attention heads
    parser.add_argument('--e_layers', type=int, default=3)   # More encoder layers
    parser.add_argument('--d_layers', type=int, default=2)   # More decoder layers
    parser.add_argument('--d_ff', type=int, default=256)     # Larger feed-forward dimensioner
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import time
import warnings
import os
import sys

# Add the root directory to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from data_provider.data_factory import data_provider
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.kl_tuning import KLTuner, suggest_kl_weight
from utils.data_analysis import analyze_dataset

warnings.filterwarnings('ignore')

# CPU-specific optimizations
torch.set_num_threads(4)  # Limit CPU threads for stability
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

def detect_and_update_features(args):
    """Detect features from dataset and update args accordingly"""
    print("🔍 Detecting dataset features...")
    
    # Build full data path
    data_path = os.path.join(args.root_path, args.data_path)
    
    # Analyze dataset to get correct dimensions
    analysis = analyze_dataset(data_path)
    
    # Update args based on features mode
    if args.features == 'M':
        # Multivariate: all features input/output
        mode_config = analysis['mode_M']
    elif args.features == 'MS':
        # Multi-target: all features input, targets output
        mode_config = analysis['mode_MS']
    elif args.features == 'S':
        # Single/target-only: targets input/output
        mode_config = analysis['mode_S']
    else:
        raise ValueError(f"Unknown features mode: {args.features}")
    
    # Update args with detected dimensions
    args.enc_in = mode_config['enc_in']
    args.dec_in = mode_config['dec_in']
    args.c_out = mode_config['c_out']
    
    print(f"✅ Dynamic feature detection complete!")
    print(f"   Mode: {args.features} - {mode_config['description']}")
    print(f"   enc_in: {args.enc_in} (encoder input features)")
    print(f"   dec_in: {args.dec_in} (decoder input features)")
    print(f"   c_out: {args.c_out} (model output features)")
    print(f"   Total features: {analysis['n_total_features']}")
    print(f"   Target features: {analysis['n_targets']} {analysis['target_columns']}")
    print(f"   Covariate features: {analysis['n_covariates']}")
    
    return args

def create_enhanced_args():
    """Create enhanced arguments with KL tuning settings"""
    parser = argparse.ArgumentParser(description='BayesianEnhancedAutoformer with KL Tuning')
    
    # Basic forecasting args
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='bayesian_enhanced')
    parser.add_argument('--model', type=str, default='BayesianEnhancedAutoformer')
    
    # Data args
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--root_path', type=str, default='./data/')
    parser.add_argument('--data_path', type=str, default='prepared_financial_data.csv')
    parser.add_argument('--features', type=str, default='MS')  # Multi-target: 118 → 4
    parser.add_argument('--target', type=str, default='nifty_return')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    
    # Forecasting task args - Lighter config to prevent freezing
    parser.add_argument('--seq_len', type=int, default=200)   # Reduced from 250 to prevent freezing
    parser.add_argument('--label_len', type=int, default=20)  # Reduced from 48
    parser.add_argument('--pred_len', type=int, default=20)   # Keep 20 day prediction
    parser.add_argument('--val_len', type=int, default=100)   # Reduced validation length
    parser.add_argument('--test_len', type=int, default=0)    # No test set - use all data for train/val
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')
    parser.add_argument('--inverse', action='store_true', default=False)
    
    # Model args - Lighter model to prevent freezing
    parser.add_argument('--enc_in', type=int, default=7, help='Encoder input size (auto-detected)')
    parser.add_argument('--dec_in', type=int, default=7, help='Decoder input size (auto-detected)')
    parser.add_argument('--c_out', type=int, default=1, help='Output size (auto-detected)')
    parser.add_argument('--d_model', type=int, default=32)    # Smaller model dimension
    parser.add_argument('--n_heads', type=int, default=2)     # Fewer attention heads
    parser.add_argument('--e_layers', type=int, default=1)    # Single encoder layer
    parser.add_argument('--d_layers', type=int, default=1)    # Single decoder layer
    parser.add_argument('--d_ff', type=int, default=64)       # Smaller feed-forward dimension
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--output_attention', action='store_true', default=False)
    
    # Optimization args - Light config to prevent freezing
    parser.add_argument('--num_workers', type=int, default=0)  # Force CPU-friendly setting
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=3)  # Just 3 epochs for testing
    parser.add_argument('--batch_size', type=int, default=16)  # Smaller batch size
    parser.add_argument('--patience', type=int, default=3)     # Reduced patience
    parser.add_argument('--learning_rate', type=float, default=0.0007)
    parser.add_argument('--des', type=str, default='Exp')
    parser.add_argument('--loss', type=str, default='quantile')  # Use quantile loss with KL
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--scale', action='store_true', default=True, help='Scale the data')
    parser.add_argument('--timeenc', type=int, default=0, help='Time encoding')
    parser.add_argument('--embed', type=str, default='timeF', help='Time embedding type')
    
    # GPU args - Force CPU usage for stable long-sequence training
    parser.add_argument('--use_gpu', type=bool, default=False)  # Force CPU usage
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3')
    parser.add_argument('--use_amp', action='store_true', default=False, help='Use automatic mixed precision')
    
    # Bayesian-specific args
    parser.add_argument('--bayesian_layers', action='store_true', default=True)
    parser.add_argument('--prior_std', type=float, default=0.1)
    parser.add_argument('--kl_weight', type=float, default=1e-4)  # Much smaller initial weight
    
    # Quantile-specific args
    parser.add_argument('--quantile_mode', action='store_true', default=True)
    parser.add_argument('--quantiles', type=str, default='0.1,0.25,0.5,0.75,0.9')
    
    # KL Tuning args
    parser.add_argument('--kl_tuning_method', type=str, default='adaptive', 
                       choices=['fixed', 'adaptive', 'annealing'])
    parser.add_argument('--target_kl_percentage', type=float, default=0.05)  # More conservative target
    parser.add_argument('--min_kl_weight', type=float, default=1e-7)  # Much smaller minimum
    parser.add_argument('--max_kl_weight', type=float, default=1e-3)  # Much smaller maximum
    parser.add_argument('--annealing_schedule', type=str, default='linear',
                       choices=['linear', 'cosine', 'exponential', 'cyclical'])
    parser.add_argument('--estimate_initial_kl', action='store_true', default=True)
    parser.add_argument('--save_kl_plots', action='store_true', default=True)
    
    args = parser.parse_args()
    
    # Add missing attributes expected by exp_basic.py
    args.gpu_type = 'cuda'  # Set to 'cuda' for GPU usage
    args.embed = 'timeF'  # Default embedding type
    args.use_dtw = True  # Enable DTW by default
    args.validation_length = args.val_len  # Add compatibility for exp module
    
    # Parse quantiles
    if isinstance(args.quantiles, str):
        args.quantiles = [float(q.strip()) for q in args.quantiles.split(',')]
    
    return args

class BayesianTrainer:
    """Enhanced trainer with KL loss tuning"""
    
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.kl_tuner = None
        
    def _acquire_device(self):
        # Force CPU usage for long-sequence stability
        if self.args.use_gpu and torch.cuda.is_available():
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print(f'Use GPU: cuda:{self.args.gpu}')
        else:
            device = torch.device('cpu')
            # Force CPU usage regardless of args for stability
            self.args.use_gpu = False
            print('Use CPU (forced for long-sequence stability)')
        return device
    
    def estimate_initial_losses(self, exp):
        """Estimate initial data loss magnitude for KL weight suggestion"""
        print("🔍 Estimating initial loss magnitudes...")
        
        # Get a few batches to estimate loss scale
        train_data, train_loader = data_provider(self.args, flag='train')
        
        model = exp.model
        model.eval()
        
        total_data_loss = 0.0
        total_kl_loss = 0.0
        num_batches = min(5, len(train_loader))  # Use first 5 batches
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                if i >= num_batches:
                    break
                    
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # Forward pass
                outputs = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                
                # Get data loss (assuming MSE or similar)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                if hasattr(model, 'quantile_mode') and model.quantile_mode:
                    # For quantile models, use quantile loss from the model's compute_loss method
                    # Create a dummy quantile criterion that has quantiles attribute
                    class QuantileCriterion:
                        def __init__(self, quantiles):
                            self.quantiles = quantiles
                    
                    criterion = QuantileCriterion(model.quantiles)
                    data_loss = model._compute_quantile_loss(outputs, batch_y, criterion)
                else:
                    # Standard MSE loss
                    criterion = nn.MSELoss()
                    data_loss = criterion(outputs, batch_y)
                
                # Get KL loss
                if hasattr(model, 'kl_loss'):
                    kl_loss = model.kl_loss()
                else:
                    kl_loss = torch.tensor(0.0)
                
                total_data_loss += data_loss.item()
                total_kl_loss += kl_loss.item()
        
        avg_data_loss = total_data_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        
        print(f"📊 Initial loss estimates:")
        print(f"   Data loss: {avg_data_loss:.4f}")
        print(f"   KL loss: {avg_kl_loss:.4f}")
        
        return avg_data_loss, avg_kl_loss
    
    def setup_kl_tuning(self, exp):
        """Setup KL tuning for the experiment"""
        model = exp.model
        
        if self.args.estimate_initial_kl:
            # Estimate initial losses and suggest KL weight
            avg_data_loss, avg_kl_loss = self.estimate_initial_losses(exp)
            
            if avg_data_loss > 0:
                suggested_weight = suggest_kl_weight(
                    avg_data_loss, 
                    self.args.target_kl_percentage
                )
                
                # Update model's KL weight
                if hasattr(model, 'kl_weight'):
                    model.kl_weight = suggested_weight
                    print(f"🎯 Updated KL weight to: {suggested_weight:.2e}")
        
        # Create KL tuner
        if hasattr(model, 'kl_weight'):
            self.kl_tuner = KLTuner(
                model=model,
                target_kl_percentage=self.args.target_kl_percentage,
                min_weight=self.args.min_kl_weight,
                max_weight=self.args.max_kl_weight
            )
            print(f"✅ KL tuner ready (target: {self.args.target_kl_percentage*100:.0f}%)")
        else:
            print("⚠️  Model doesn't support KL loss, skipping KL tuning")
    
    def train_with_kl_tuning(self):
        """Main training loop with KL tuning"""
        print(f"🚀 Starting Bayesian training with KL tuning")
        print(f"Model: {self.args.model}")
        print(f"KL tuning method: {self.args.kl_tuning_method}")
        print(f"Loss type: Quantile Loss + KL Loss")
        print(f"Quantiles: {self.args.quantiles}")
        print("=" * 60)
        
        # Initialize experiment
        exp = Exp_Long_Term_Forecast(self.args)
        
        # Setup KL tuning
        self.setup_kl_tuning(exp)
        
        # Training
        print("Training...")
        print("📊 Loss components will be: Quantile Loss + KL Loss")
        best_model_path = exp.train(setting=f"{self.args.model_id}_kl_tuned")
        
        # If we have KL tuner, update during training would happen in exp.train
        # For now, we'll do a post-training analysis
        
        # Test
        print("Testing...")
        exp.test(setting=f"{self.args.model_id}_kl_tuned", test=1)
        
        # Save KL tuning plots if available
        if self.kl_tuner and self.args.save_kl_plots:
            plot_path = f"./checkpoints/{self.args.model_id}_kl_tuning_history.png"
            try:
                self.kl_tuner.plot_kl_tuning_history(plot_path)
                print(f"📈 KL tuning history saved to: {plot_path}")
            except Exception as e:
                print(f"⚠️  Could not save KL plot: {e}")
        
        return best_model_path
    
    def run_ablation_study(self):
        """Run ablation study with different KL settings"""
        print("🔬 Running KL Tuning Ablation Study")
        print("=" * 50)
        
        kl_settings = [
            ('no_kl', {'kl_weight': 0.0, 'kl_tuning_method': 'fixed'}),
            ('low_kl', {'target_kl_percentage': 0.05, 'kl_tuning_method': 'adaptive'}),
            ('medium_kl', {'target_kl_percentage': 0.10, 'kl_tuning_method': 'adaptive'}),
            ('high_kl', {'target_kl_percentage': 0.20, 'kl_tuning_method': 'adaptive'}),
            ('annealing', {'kl_tuning_method': 'annealing', 'annealing_schedule': 'cosine'}),
        ]
        
        results = {}
        
        for name, settings in kl_settings:
            print(f"\n🧪 Testing: {name}")
            print(f"Settings: {settings}")
            
            # Update args for this experiment
            for key, value in settings.items():
                setattr(self.args, key, value)
            
            self.args.model_id = f"ablation_{name}"
            
            try:
                best_path = self.train_with_kl_tuning()
                results[name] = {'status': 'success', 'model_path': best_path}
                print(f"✅ {name} completed successfully")
            except Exception as e:
                results[name] = {'status': 'failed', 'error': str(e)}
                print(f"❌ {name} failed: {e}")
        
        # Print results summary
        print("\n📊 Ablation Study Results:")
        print("-" * 40)
        for name, result in results.items():
            status = result['status']
            print(f"{name:12s}: {status}")
        
        return results

def validate_high_config_settings(args):
    """Validate and adjust settings for high-config experiment"""
    print("🔧 Validating high-config experiment settings...")
    
    # Ensure CPU-friendly settings
    args.use_gpu = False
    args.num_workers = 0
    
    # Validate sequence configuration
    if args.seq_len >= 200:
        print(f"✅ Long sequence length: {args.seq_len} days")
        # Reduce batch size for long sequences
        if args.batch_size > 16:
            args.batch_size = 16
            print(f"🔧 Reduced batch size to {args.batch_size} for long sequences")
    
    # Validate prediction configuration
    print(f"✅ Prediction setup: {args.pred_len} day prediction, {args.label_len} day label")
    
    # Memory-friendly settings
    if hasattr(args, 'd_model') and args.d_model > 128:
        args.d_model = 128
        print(f"🔧 Reduced d_model to {args.d_model} for memory efficiency")
    
    if hasattr(args, 'd_ff') and args.d_ff > 256:
        args.d_ff = 256
        print(f"🔧 Reduced d_ff to {args.d_ff} for memory efficiency")
    
    print("✅ High-config settings validated")
    return args

def main():
    # Parse arguments
    args = create_enhanced_args()
    
    # Detect and update feature dimensions dynamically
    args = detect_and_update_features(args)
    
    # Validate and adjust for high-config experiment
    args = validate_high_config_settings(args)
    
    # Print configuration
    print("🎯 Bayesian Enhanced Autoformer Training - High Config Experiment")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Data: {args.data_path}")
    print(f"Sequence length: {args.seq_len} days")
    print(f"Prediction length: {args.pred_len} days")
    print(f"Label length: {args.label_len} days")
    print(f"Batch size: {args.batch_size}")
    print(f"Num workers: {args.num_workers}")
    print(f"Use GPU: {args.use_gpu}")
    print(f"Bayesian layers: {args.bayesian_layers}")
    print(f"Quantile mode: {args.quantile_mode}")
    print(f"KL tuning method: {args.kl_tuning_method}")
    print(f"Target KL%: {args.target_kl_percentage*100:.0f}%")
    print("=" * 60)
    
    # Create trainer and run
    trainer = BayesianTrainer(args)
    
    if args.des == 'ablation':
        # Run ablation study
        results = trainer.run_ablation_study()
    else:
        # Run single training
        best_model_path = trainer.train_with_kl_tuning()
        print(f"\n✅ Training completed!")
        print(f"Best model saved at: {best_model_path}")

if __name__ == '__main__':
    main()
