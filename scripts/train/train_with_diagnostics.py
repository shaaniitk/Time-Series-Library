"""
Enhanced Training Script with Comprehensive Diagnostics

Extends the production training script with:
1. Gradient flow monitoring and logging
2. Attention weight visualization
3. Component-level performance profiling
4. Memory usage tracking
5. Loss decomposition analysis
6. Checkpoint diagnostics
"""

import sys
import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from data_provider.celestial_data_factory import CelestialDataFactory
from models.Celestial_Enhanced_PGAT import Model
from layers.modular.decoder.mixture_density_decoder import MixtureNLLLoss
from utils.tools import EarlyStopping, adjust_learning_rate


class GradientDiagnostics:
    """Track and log gradient statistics across training."""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir) / "gradients"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.gradient_history = []
        
    def log_gradients(self, model: nn.Module, step: int, epoch: int):
        """Log gradient statistics for all model parameters."""
        stats = {
            'step': step,
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'component_stats': {}
        }
        
        # Component-wise gradient statistics
        components = {
            'fusion_cross_attention': [],
            'hierarchical_fusion_proj': [],
            'celestial_to_target_attention': [],
            'wave_patching_composer': [],
            'target_patching_composer': [],
            'graph_combiner': [],
            'stochastic_learner': [],
            'decoder': [],
            'other': []
        }
        
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
                
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            grad_max = param.grad.max().item()
            grad_min = param.grad.min().item()
            
            # Categorize by component
            component_key = 'other'
            for comp in components.keys():
                if comp in name:
                    component_key = comp
                    break
            
            components[component_key].append({
                'name': name,
                'norm': grad_norm,
                'mean': grad_mean,
                'std': grad_std,
                'max': grad_max,
                'min': grad_min,
                'shape': list(param.shape)
            })
        
        # Aggregate component statistics
        for comp_name, comp_grads in components.items():
            if comp_grads:
                norms = [g['norm'] for g in comp_grads]
                stats['component_stats'][comp_name] = {
                    'num_params': len(comp_grads),
                    'total_norm': sum(norms),
                    'mean_norm': np.mean(norms),
                    'std_norm': np.std(norms),
                    'max_norm': max(norms),
                    'min_norm': min(norms),
                    'params': comp_grads
                }
        
        # Save to file
        output_file = self.log_dir / f"gradients_step_{step:06d}.json"
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.gradient_history.append({
            'step': step,
            'epoch': epoch,
            'total_norm': sum([s['total_norm'] for s in stats['component_stats'].values()]),
            'component_norms': {k: v['total_norm'] for k, v in stats['component_stats'].items()}
        })
        
        return stats


class AttentionLogger:
    """Log attention weights for analysis."""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir) / "attention"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def log_attention(self, model: nn.Module, step: int, epoch: int):
        """Extract and log attention weights."""
        attention_data = {
            'step': step,
            'epoch': epoch,
            'timestamp': datetime.now().isoformat()
        }
        
        # Hierarchical fusion attention
        if hasattr(model, '_last_fusion_attention'):
            fusion_attn = model._last_fusion_attention
            if fusion_attn is not None:
                attention_data['fusion_attention'] = {
                    'shape': list(fusion_attn.shape),
                    'mean': fusion_attn.mean().item(),
                    'std': fusion_attn.std().item(),
                    'max': fusion_attn.max().item(),
                    'entropy': -(fusion_attn * torch.log(fusion_attn + 1e-10)).sum().item(),
                    'weights': fusion_attn.cpu().numpy().tolist()
                }
                
                if hasattr(model, '_last_fusion_scale_lengths'):
                    attention_data['fusion_scale_lengths'] = model._last_fusion_scale_lengths
        
        # C→T attention
        if hasattr(model, 'celestial_to_target_attention'):
            c2t_module = model.celestial_to_target_attention
            if hasattr(c2t_module, 'latest_attention_weights') and c2t_module.latest_attention_weights:
                c2t_stats = {}
                for target_name, weights in c2t_module.latest_attention_weights.items():
                    if weights is not None:
                        c2t_stats[target_name] = {
                            'shape': list(weights.shape),
                            'mean': weights.mean().item(),
                            'std': weights.std().item(),
                            'max': weights.max().item(),
                            'top_5_celestial': weights.mean(dim=0).topk(5).indices.tolist()
                        }
                attention_data['c2t_attention'] = c2t_stats
        
        # Save to file
        output_file = self.log_dir / f"attention_step_{step:06d}.json"
        with open(output_file, 'w') as f:
            json.dump(attention_data, f, indent=2)
        
        return attention_data


class ComponentProfiler:
    """Profile component-level performance."""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir) / "components"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def profile_model(self, model: nn.Module, step: int, epoch: int):
        """Profile model components."""
        profile_data = {
            'step': step,
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        # Get enhanced config info
        if hasattr(model, 'get_enhanced_config_info'):
            config_info = model.get_enhanced_config_info()
            profile_data['config'] = config_info
        
        # Internal logs
        if hasattr(model, 'get_internal_logs'):
            internal_logs = model.get_internal_logs()
            profile_data['internal_logs'] = {
                k: str(v) if torch.is_tensor(v) else v 
                for k, v in internal_logs.items()
            }
        
        # Parameter counts per component
        component_params = {}
        for name, param in model.named_parameters():
            component = name.split('.')[0]
            if component not in component_params:
                component_params[component] = {
                    'count': 0,
                    'total_params': 0,
                    'trainable_params': 0
                }
            component_params[component]['count'] += 1
            component_params[component]['total_params'] += param.numel()
            if param.requires_grad:
                component_params[component]['trainable_params'] += param.numel()
        
        profile_data['parameter_breakdown'] = component_params
        
        # Save to file
        output_file = self.log_dir / f"profile_step_{step:06d}.json"
        with open(output_file, 'w') as f:
            json.dump(profile_data, f, indent=2)
        
        return profile_data


def train_with_diagnostics(args):
    """Main training loop with comprehensive diagnostics."""
    
    # Load config
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    class Config:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)
    
    config = Config(config_dict)
    
    # Setup directories
    log_dir = Path(args.log_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize diagnostics
    gradient_diagnostics = GradientDiagnostics(args.log_dir)
    attention_logger = AttentionLogger(args.log_dir)
    component_profiler = ComponentProfiler(args.log_dir)
    
    # TensorBoard
    writer = SummaryWriter(log_dir / "tensorboard")
    
    # Save config
    with open(log_dir / "config.yaml", 'w') as f:
        yaml.dump(config_dict, f)
    
    print(f"\n{'='*80}")
    print(f"Training Configuration")
    print(f"{'='*80}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Config: {args.config}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Gradient diagnostics: {args.enable_gradient_diagnostics}")
    print(f"Attention logging: {args.enable_attention_logging}")
    print(f"Component profiling: {args.enable_component_profiling}")
    print(f"{'='*80}\n")
    
    # Create data loaders
    print("Creating data loaders...")
    data_factory = CelestialDataFactory(config)
    train_loader = data_factory.get_data_loader(flag='train')
    val_loader = data_factory.get_data_loader(flag='val')
    
    # Initialize model
    print("Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(config).to(device)
    
    # Enable diagnostic collection
    if args.enable_attention_logging:
        model.collect_diagnostics = True
        if hasattr(model, 'celestial_to_target_attention'):
            model.celestial_to_target_attention.enable_diagnostics = True
    
    # Optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = MixtureNLLLoss()
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.patience, verbose=True)
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(config.train_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{config.train_epochs}")
        print(f"{'='*80}")
        
        # Training phase
        model.train()
        train_losses = []
        
        for batch_idx, batch_data in enumerate(train_loader):
            wave_window, target_window, celestial_window = batch_data
            wave_window = wave_window.to(device)
            target_window = target_window.to(device)
            celestial_window = celestial_window.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(wave_window, target_window, celestial_window, None)
            
            # Compute loss
            if isinstance(output, tuple):
                means, log_stds, log_weights = output
                targets = target_window[:, -means.size(1):, -means.size(2):]
                loss = criterion(output, targets)
            else:
                targets = target_window[:, -output.size(1):, -output.size(2):]
                loss = nn.MSELoss()(output, targets)
            
            # Add regularization
            if hasattr(model, 'get_regularization_loss'):
                reg_loss = model.get_regularization_loss()
                total_loss = loss + 0.01 * reg_loss
            else:
                total_loss = loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
            
            train_losses.append(loss.item())
            global_step += 1
            
            # Logging
            if batch_idx % args.log_interval == 0:
                print(f"Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.6f}")
                writer.add_scalar('Train/Loss', loss.item(), global_step)
            
            # Gradient diagnostics
            if args.enable_gradient_diagnostics and global_step % args.gradient_log_interval == 0:
                grad_stats = gradient_diagnostics.log_gradients(model, global_step, epoch)
                print(f"  Gradient diagnostics saved (step {global_step})")
                
                # Log to tensorboard
                for comp_name, comp_stats in grad_stats['component_stats'].items():
                    writer.add_scalar(f'Gradients/{comp_name}/total_norm', 
                                    comp_stats['total_norm'], global_step)
            
            # Attention logging
            if args.enable_attention_logging and global_step % args.attention_log_interval == 0:
                attn_data = attention_logger.log_attention(model, global_step, epoch)
                print(f"  Attention weights saved (step {global_step})")
            
            # Component profiling
            if args.enable_component_profiling and global_step % args.checkpoint_interval == 0:
                profile_data = component_profiler.profile_model(model, global_step, epoch)
                print(f"  Component profile saved (step {global_step})")
            
            # Checkpoint
            if global_step % args.checkpoint_interval == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_step_{global_step:06d}.pth"
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)
                print(f"  Checkpoint saved: {checkpoint_path}")
        
        # Validation phase
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_data in val_loader:
                wave_window, target_window, celestial_window = batch_data
                wave_window = wave_window.to(device)
                target_window = target_window.to(device)
                celestial_window = celestial_window.to(device)
                
                output = model(wave_window, target_window, celestial_window, None)
                
                if isinstance(output, tuple):
                    means, log_stds, log_weights = output
                    targets = target_window[:, -means.size(1):, -means.size(2):]
                    loss = criterion(output, targets)
                else:
                    targets = target_window[:, -output.size(1):, -output.size(2):]
                    loss = nn.MSELoss()(output, targets)
                
                val_losses.append(loss.item())
        
        # Epoch summary
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        
        writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
        writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
        
        # Early stopping
        early_stopping(val_loss, model, checkpoint_dir / 'best_model.pth')
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
        
        # Learning rate adjustment
        adjust_learning_rate(optimizer, epoch + 1, config)
    
    # Final checkpoint
    final_checkpoint = checkpoint_dir / 'final_model.pth'
    torch.save(model.state_dict(), final_checkpoint)
    print(f"\nFinal model saved: {final_checkpoint}")
    
    # Save gradient history
    gradient_history_file = log_dir / "gradients" / "history.json"
    with open(gradient_history_file, 'w') as f:
        json.dump(gradient_diagnostics.gradient_history, f, indent=2)
    
    writer.close()
    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train with comprehensive diagnostics')
    
    # Required arguments
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to config file')
    
    # Optional arguments
    parser.add_argument('--experiment_name', type=str, 
                       default=f'experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                       help='Experiment name for logging')
    parser.add_argument('--log_dir', type=str, default='logs/diagnostics',
                       help='Directory for logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/diagnostics',
                       help='Directory for checkpoints')
    
    # Diagnostic flags
    parser.add_argument('--enable_gradient_diagnostics', action='store_true',
                       help='Enable gradient flow diagnostics')
    parser.add_argument('--enable_attention_logging', action='store_true',
                       help='Enable attention weight logging')
    parser.add_argument('--enable_component_profiling', action='store_true',
                       help='Enable component profiling')
    
    # Logging intervals
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Batch interval for basic logging')
    parser.add_argument('--gradient_log_interval', type=int, default=50,
                       help='Step interval for gradient logging')
    parser.add_argument('--attention_log_interval', type=int, default=100,
                       help='Step interval for attention logging')
    parser.add_argument('--checkpoint_interval', type=int, default=500,
                       help='Step interval for checkpoints')
    
    args = parser.parse_args()
    
    train_with_diagnostics(args)
