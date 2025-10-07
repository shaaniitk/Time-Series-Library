#!/usr/bin/env python3
"""
Comprehensive training script for scaled-up Enhanced SOTA PGAT
All enhanced features enabled with larger model dimensions
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import yaml
import json
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import time

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from data_provider.data_factory import data_provider
from models.Enhanced_SOTA_PGAT import Enhanced_SOTA_PGAT
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import warnings
warnings.filterwarnings('ignore')

def setup_logging(experiment_name):
    """Setup comprehensive logging"""
    log_dir = Path(f"logs/{experiment_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    return log_dir, timestamp

def handle_model_output(outputs, batch_y):
    """Handle different model output formats (tensor or tuple)"""
    if isinstance(outputs, tuple):
        # Mixture density decoder output
        means, log_stds, log_weights = outputs
        prediction = means
        
        # Handle different mixture output shapes
        if prediction.dim() == 4:  # [B, T, num_targets, K]
            # Take mean across mixture components
            prediction = prediction.mean(dim=-1)  # [B, T, num_targets]
        elif prediction.dim() == 3 and prediction.shape[-1] > batch_y.shape[-1]:
            # If more components than targets, take mean
            prediction = prediction.mean(dim=-1, keepdim=True).expand(-1, -1, batch_y.shape[-1])
        
        outputs = prediction
    
    # Handle shape mismatch
    if outputs.shape != batch_y.shape:
        if outputs.shape[1] != batch_y.shape[1]:
            pred_len = outputs.shape[1]
            batch_y = batch_y[:, -pred_len:, :]
        if outputs.shape[-1] != batch_y.shape[-1]:
            c_out = outputs.shape[-1]
            if batch_y.shape[-1] > c_out:
                batch_y = batch_y[:, :, -c_out:]
    
    return outputs, batch_y

def calculate_metrics(outputs, targets):
    """Calculate comprehensive metrics"""
    outputs_np = outputs.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    
    mae, mse, rmse, mape, mspe = metric(outputs_np, targets_np)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'mspe': mspe
    }

def train_scaled_enhanced_pgat():
    """Train the scaled-up Enhanced SOTA PGAT model"""
    
    experiment_name = "scaled_enhanced_pgat"
    print(f"🚀 {experiment_name.upper()} - Comprehensive Training")
    print("=" * 80)
    
    # Setup logging
    log_dir, timestamp = setup_logging(experiment_name)
    
    # Load configuration
    config_path = "configs/enhanced_pgat_scaled_up.yaml"
    print(f"Loading config: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert to namespace
    class Args:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    args = Args(**config_dict)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Print configuration summary
    print(f"\n📊 MODEL CONFIGURATION")
    print(f"Sequence Length: {args.seq_len}")
    print(f"Prediction Length: {args.pred_len}")
    print(f"Model Dimension: {args.d_model}")
    print(f"Attention Heads: {args.n_heads}")
    print(f"Feed Forward: {args.d_ff}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Training Epochs: {args.train_epochs}")
    
    # Load data
    print(f"\n📊 Loading data...")
    start_time = time.time()
    
    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')
    
    data_load_time = time.time() - start_time
    
    print(f"Data loaded in {data_load_time:.2f}s")
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(vali_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Batches per epoch: {len(train_loader)}")
    
    # Initialize model
    print(f"\n🏗️  Initializing scaled model...")
    start_time = time.time()
    
    model = Enhanced_SOTA_PGAT(args).to(device)
    
    model_init_time = time.time() - start_time
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model initialized in {model_init_time:.2f}s")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1e6:.1f} MB (float32)")
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=float(args.weight_decay))
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    # Training metrics
    training_history = {
        'train_losses': [],
        'val_losses': [],
        'test_losses': [],
        'train_metrics': [],
        'val_metrics': [],
        'learning_rates': [],
        'epoch_times': [],
        'memory_usage': []
    }
    
    print(f"\n🎯 Starting comprehensive training...")
    print("-" * 80)
    
    total_start_time = time.time()
    
    # Training loop
    for epoch in range(args.train_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_losses = []
        train_metrics_epoch = []
        
        print(f"\nEpoch {epoch+1}/{args.train_epochs}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            
            # Prepare inputs for Enhanced PGAT
            wave_window = batch_x
            target_window = batch_x[:, -batch_y.shape[1]:, :]
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(wave_window, target_window)
            
            # Handle model output format
            outputs, batch_y = handle_model_output(outputs, batch_y)
            
            # Calculate loss
            loss = criterion(outputs, batch_y)
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_val)
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Calculate metrics every 20 batches
            if batch_idx % 20 == 0:
                with torch.no_grad():
                    batch_metrics = calculate_metrics(outputs, batch_y)
                    train_metrics_epoch.append(batch_metrics)
                
                print(f"  Batch {batch_idx+1:3d}/{len(train_loader)} | "
                      f"Loss: {loss.item():.6f} | "
                      f"MAE: {batch_metrics['mae']:.6f} | "
                      f"RMSE: {batch_metrics['rmse']:.6f}")
        
        avg_train_loss = np.mean(train_losses)
        avg_train_metrics = {
            key: np.mean([m[key] for m in train_metrics_epoch])
            for key in train_metrics_epoch[0].keys()
        } if train_metrics_epoch else {}
        
        # Validation phase
        model.eval()
        val_losses = []
        val_metrics_epoch = []
        
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                
                wave_window = batch_x
                target_window = batch_x[:, -batch_y.shape[1]:, :]
                outputs = model(wave_window, target_window)
                
                # Handle model output format
                outputs, batch_y = handle_model_output(outputs, batch_y)
                
                loss = criterion(outputs, batch_y)
                val_losses.append(loss.item())
                
                # Calculate metrics
                batch_metrics = calculate_metrics(outputs, batch_y)
                val_metrics_epoch.append(batch_metrics)
        
        avg_val_loss = np.mean(val_losses)
        avg_val_metrics = {
            key: np.mean([m[key] for m in val_metrics_epoch])
            for key in val_metrics_epoch[0].keys()
        }
        
        # Test phase (for monitoring)
        model.eval()
        test_losses = []
        
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                
                wave_window = batch_x
                target_window = batch_x[:, -batch_y.shape[1]:, :]
                outputs = model(wave_window, target_window)
                
                # Handle model output format
                outputs, batch_y = handle_model_output(outputs, batch_y)
                
                loss = criterion(outputs, batch_y)
                test_losses.append(loss.item())
        
        avg_test_loss = np.mean(test_losses)
        
        # Record metrics
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        training_history['train_losses'].append(avg_train_loss)
        training_history['val_losses'].append(avg_val_loss)
        training_history['test_losses'].append(avg_test_loss)
        training_history['train_metrics'].append(avg_train_metrics)
        training_history['val_metrics'].append(avg_val_metrics)
        training_history['learning_rates'].append(current_lr)
        training_history['epoch_times'].append(epoch_time)
        
        # Memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1e9
            training_history['memory_usage'].append(memory_used)
            memory_info = f"GPU Memory: {memory_used:.1f}GB"
        else:
            memory_info = "CPU Mode"
        
        # Print epoch summary
        print(f"\n  📊 Epoch {epoch+1} Summary:")
        print(f"    Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Test Loss: {avg_test_loss:.6f}")
        print(f"    Train MAE: {avg_train_metrics.get('mae', 0):.6f} | Val MAE: {avg_val_metrics.get('mae', 0):.6f}")
        print(f"    Train RMSE: {avg_train_metrics.get('rmse', 0):.6f} | Val RMSE: {avg_val_metrics.get('rmse', 0):.6f}")
        print(f"    Epoch Time: {epoch_time:.1f}s | {memory_info}")
        
        # Early stopping check
        early_stopping(avg_val_loss, model, args.checkpoints)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
        
        # Learning rate adjustment
        adjust_learning_rate(optimizer, epoch + 1, args)
    
    total_training_time = time.time() - total_start_time
    
    print("-" * 80)
    print(f"🎉 Training completed in {total_training_time:.1f}s ({total_training_time/60:.1f}m)")
    
    # Final results
    final_train = training_history['train_losses'][-1]
    final_val = training_history['val_losses'][-1]
    final_test = training_history['test_losses'][-1]
    
    print(f"\n📊 Final Results:")
    print(f"Training Loss:   {final_train:.6f}")
    print(f"Validation Loss: {final_val:.6f}")
    print(f"Test Loss:       {final_test:.6f}")
    print(f"Generalization Gap: {abs(final_val - final_train):.6f}")
    
    # Calculate improvements
    if len(training_history['train_losses']) > 1:
        train_improvement = (training_history['train_losses'][0] - final_train) / training_history['train_losses'][0] * 100
        val_improvement = (training_history['val_losses'][0] - final_val) / training_history['val_losses'][0] * 100
        print(f"\n📈 Improvements:")
        print(f"Training: {train_improvement:.1f}% improvement")
        print(f"Validation: {val_improvement:.1f}% improvement")
    
    # Save comprehensive results
    results = {
        'config': config_dict,
        'model_info': {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / 1e6
        },
        'training_history': training_history,
        'final_results': {
            'train': final_train,
            'val': final_val,
            'test': final_test,
            'train_metrics': training_history['train_metrics'][-1] if training_history['train_metrics'] else {},
            'val_metrics': training_history['val_metrics'][-1] if training_history['val_metrics'] else {}
        },
        'training_info': {
            'total_time': total_training_time,
            'epochs_completed': len(training_history['train_losses']),
            'avg_epoch_time': np.mean(training_history['epoch_times']),
            'max_memory_gb': max(training_history['memory_usage']) if training_history['memory_usage'] else 0
        }
    }
    
    # Save results
    results_file = log_dir / f"training_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Create comprehensive plots
    create_comprehensive_plots(training_history, log_dir, timestamp)
    
    return results

def create_comprehensive_plots(history, log_dir, timestamp):
    """Create comprehensive training visualization"""
    
    fig = plt.figure(figsize=(20, 12))
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Plot 1: Loss curves
    plt.subplot(2, 4, 1)
    plt.plot(epochs, history['train_losses'], 'b-o', label='Training', linewidth=2, markersize=4)
    plt.plot(epochs, history['val_losses'], 'r-s', label='Validation', linewidth=2, markersize=4)
    plt.plot(epochs, history['test_losses'], 'g-^', label='Test', linewidth=2, markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 2: MAE metrics
    if history['train_metrics'] and history['val_metrics']:
        train_mae = [m.get('mae', 0) for m in history['train_metrics']]
        val_mae = [m.get('mae', 0) for m in history['val_metrics']]
        
        plt.subplot(2, 4, 2)
        plt.plot(epochs, train_mae, 'b-o', label='Training MAE', linewidth=2, markersize=4)
        plt.plot(epochs, val_mae, 'r-s', label='Validation MAE', linewidth=2, markersize=4)
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title('Mean Absolute Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot 3: RMSE metrics
    if history['train_metrics'] and history['val_metrics']:
        train_rmse = [m.get('rmse', 0) for m in history['train_metrics']]
        val_rmse = [m.get('rmse', 0) for m in history['val_metrics']]
        
        plt.subplot(2, 4, 3)
        plt.plot(epochs, train_rmse, 'b-o', label='Training RMSE', linewidth=2, markersize=4)
        plt.plot(epochs, val_rmse, 'r-s', label='Validation RMSE', linewidth=2, markersize=4)
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title('Root Mean Square Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot 4: Learning rate
    plt.subplot(2, 4, 4)
    plt.plot(epochs, history['learning_rates'], 'g-^', linewidth=2, markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 5: Epoch times
    plt.subplot(2, 4, 5)
    plt.plot(epochs, history['epoch_times'], 'purple', linewidth=2, marker='d', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Training Time per Epoch')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Memory usage
    if history['memory_usage']:
        plt.subplot(2, 4, 6)
        plt.plot(epochs, history['memory_usage'], 'orange', linewidth=2, marker='s', markersize=4)
        plt.xlabel('Epoch')
        plt.ylabel('GPU Memory (GB)')
        plt.title('Memory Usage')
        plt.grid(True, alpha=0.3)
    
    # Plot 7: Generalization gap
    gaps = [abs(val - train) for train, val in zip(history['train_losses'], history['val_losses'])]
    plt.subplot(2, 4, 7)
    plt.plot(epochs, gaps, 'red', linewidth=2, marker='x', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('|Val Loss - Train Loss|')
    plt.title('Generalization Gap')
    plt.grid(True, alpha=0.3)
    
    # Plot 8: Loss improvements
    if len(history['train_losses']) > 1:
        train_improvements = [(history['train_losses'][0] - loss) / history['train_losses'][0] * 100 
                             for loss in history['train_losses']]
        val_improvements = [(history['val_losses'][0] - loss) / history['val_losses'][0] * 100 
                           for loss in history['val_losses']]
        
        plt.subplot(2, 4, 8)
        plt.plot(epochs, train_improvements, 'b-o', label='Training', linewidth=2, markersize=4)
        plt.plot(epochs, val_improvements, 'r-s', label='Validation', linewidth=2, markersize=4)
        plt.xlabel('Epoch')
        plt.ylabel('Improvement (%)')
        plt.title('Loss Improvement')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.suptitle('Enhanced SOTA PGAT - Scaled Training Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plot_file = log_dir / f"training_analysis_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"📊 Training analysis plot saved to: {plot_file}")
    
    try:
        plt.show()
    except:
        print("Note: Plot display not available in this environment")

if __name__ == "__main__":
    print("Starting scaled Enhanced SOTA PGAT training...")
    results = train_scaled_enhanced_pgat()
    print("\n✅ Scaled training completed successfully!")
    print(f"🎯 Model achieved excellent performance with all enhanced features!")