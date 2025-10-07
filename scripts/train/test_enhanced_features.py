#!/usr/bin/env python3
"""
Systematic testing of enhanced features to identify convergence issues
Enable features one by one to isolate problematic components
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import yaml
import json
from datetime import datetime
from copy import deepcopy

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from data_provider.data_factory import data_provider
from models.Enhanced_SOTA_PGAT import Enhanced_SOTA_PGAT
from utils.tools import EarlyStopping, adjust_learning_rate
import warnings
warnings.filterwarnings('ignore')

def create_test_config(base_config, feature_name, enable_feature=True):
    """Create a test configuration with specific feature enabled/disabled"""
    config = deepcopy(base_config)
    
    # Feature mapping
    feature_configs = {
        'stochastic_learner': {
            'use_stochastic_learner': enable_feature
        },
        'gated_graph_combiner': {
            'use_gated_graph_combiner': enable_feature
        },
        'mixture_density': {
            'use_mixture_density': enable_feature,
            'use_mixture_decoder': enable_feature
        },
        'complex_loss': {
            'loss': 'mixture' if enable_feature else 'mse',
            'loss_function_type': 'mixture' if enable_feature else 'mse'
        }
    }
    
    if feature_name in feature_configs:
        config.update(feature_configs[feature_name])
    
    return config

def run_quick_training_test(config_dict, test_name, epochs=3):
    """Run a quick training test to check convergence"""
    
    print(f"\n🧪 Testing: {test_name}")
    print("-" * 50)
    
    # Convert to namespace
    class Args:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    args = Args(**config_dict)
    args.train_epochs = epochs  # Quick test
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load data
        train_data, train_loader = data_provider(args, 'train')
        vali_data, vali_loader = data_provider(args, 'val')
        
        # Initialize model
        model = Enhanced_SOTA_PGAT(args).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {total_params:,}")
        
        # Setup training
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.MSELoss()
        
        # Training metrics
        train_losses = []
        val_losses = []
        
        # Quick training loop
        for epoch in range(args.train_epochs):
            # Training phase
            model.train()
            train_loss = []
            
            for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                if batch_idx >= 10:  # Only test first 10 batches for speed
                    break
                    
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                
                # Prepare inputs
                wave_window = batch_x
                target_window = batch_x[:, -batch_y.shape[1]:, :]
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(wave_window, target_window)
                
                # Handle mixture density decoder output (tuple)
                if isinstance(outputs, tuple):
                    # For mixture density decoder, use the means as the prediction
                    means, log_stds, log_weights = outputs
                    prediction = means
                    # If multivariate, take mean across mixture components
                    if prediction.dim() == 4:  # [B, T, num_targets, K]
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
                
                # Check for NaN/Inf in outputs
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"❌ NaN/Inf detected in outputs at epoch {epoch+1}, batch {batch_idx+1}")
                    return {
                        'success': False,
                        'error': 'NaN/Inf in outputs',
                        'epoch': epoch + 1,
                        'batch': batch_idx + 1
                    }
                
                loss = criterion(outputs, batch_y)
                
                # Check for NaN/Inf in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"❌ NaN/Inf detected in loss at epoch {epoch+1}, batch {batch_idx+1}")
                    return {
                        'success': False,
                        'error': 'NaN/Inf in loss',
                        'epoch': epoch + 1,
                        'batch': batch_idx + 1
                    }
                
                loss.backward()
                
                # Check gradients
                total_grad_norm = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        param_norm = param.grad.norm().item()
                        total_grad_norm += param_norm ** 2
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"❌ NaN/Inf detected in gradients at epoch {epoch+1}, batch {batch_idx+1}")
                            return {
                                'success': False,
                                'error': 'NaN/Inf in gradients',
                                'epoch': epoch + 1,
                                'batch': batch_idx + 1
                            }
                
                total_grad_norm = total_grad_norm ** 0.5
                
                # Check for gradient explosion
                if total_grad_norm > 100:
                    print(f"⚠️  Large gradient norm detected: {total_grad_norm:.2f}")
                
                optimizer.step()
                train_loss.append(loss.item())
            
            avg_train_loss = np.mean(train_loss)
            train_losses.append(avg_train_loss)
            
            # Quick validation
            model.eval()
            val_loss = []
            
            with torch.no_grad():
                for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                    if batch_idx >= 5:  # Only test first 5 validation batches
                        break
                        
                    batch_x = batch_x.float().to(device)
                    batch_y = batch_y.float().to(device)
                    
                    wave_window = batch_x
                    target_window = batch_x[:, -batch_y.shape[1]:, :]
                    outputs = model(wave_window, target_window)
                    
                    # Handle mixture density decoder output (tuple)
                    if isinstance(outputs, tuple):
                        # For mixture density decoder, use the means as the prediction
                        means, log_stds, log_weights = outputs
                        prediction = means
                        # If multivariate, take mean across mixture components
                        if prediction.dim() == 4:  # [B, T, num_targets, K]
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
                    
                    loss = criterion(outputs, batch_y)
                    val_loss.append(loss.item())
            
            avg_val_loss = np.mean(val_loss)
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}: Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f}")
        
        # Calculate improvement
        if len(train_losses) > 1:
            train_improvement = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
            val_improvement = (val_losses[0] - val_losses[-1]) / val_losses[0] * 100
        else:
            train_improvement = 0
            val_improvement = 0
        
        print(f"✅ Success! Train improvement: {train_improvement:.1f}%, Val improvement: {val_improvement:.1f}%")
        
        return {
            'success': True,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_improvement': train_improvement,
            'val_improvement': val_improvement,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'total_params': total_params
        }
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def test_enhanced_features():
    """Test each enhanced feature systematically"""
    
    print("🔬 Enhanced SOTA PGAT - Feature Testing")
    print("=" * 60)
    
    # Load base configuration (simplified, working version)
    config_path = "configs/enhanced_sota_pgat_simplified.yaml"
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'Baseline (All Enhanced Features Disabled)',
            'config': base_config,
            'description': 'Current working configuration'
        },
        {
            'name': 'Enable Stochastic Learner',
            'config': create_test_config(base_config, 'stochastic_learner', True),
            'description': 'Add stochastic graph learning'
        },
        {
            'name': 'Enable Gated Graph Combiner',
            'config': create_test_config(base_config, 'gated_graph_combiner', True),
            'description': 'Add gated graph combination'
        },
        {
            'name': 'Enable Mixture Density Decoder',
            'config': create_test_config(base_config, 'mixture_density', True),
            'description': 'Add probabilistic mixture decoder'
        },
        {
            'name': 'Enable Complex Loss Function',
            'config': create_test_config(base_config, 'complex_loss', True),
            'description': 'Use mixture loss instead of MSE'
        }
    ]
    
    # Add combined tests
    combined_config = deepcopy(base_config)
    combined_config.update({
        'use_stochastic_learner': True,
        'use_gated_graph_combiner': True
    })
    test_scenarios.append({
        'name': 'Stochastic + Gated Combiner',
        'config': combined_config,
        'description': 'Combine stochastic learner and gated combiner'
    })
    
    # Full enhanced config
    full_config = deepcopy(base_config)
    full_config.update({
        'use_stochastic_learner': True,
        'use_gated_graph_combiner': True,
        'use_mixture_density': True,
        'use_mixture_decoder': True,
        'loss': 'mixture',
        'loss_function_type': 'mixture'
    })
    test_scenarios.append({
        'name': 'All Enhanced Features',
        'config': full_config,
        'description': 'Enable all enhanced features'
    })
    
    # Run tests
    results = {}
    
    for scenario in test_scenarios:
        print(f"\n{'='*60}")
        print(f"🧪 {scenario['name']}")
        print(f"📝 {scenario['description']}")
        print(f"{'='*60}")
        
        result = run_quick_training_test(scenario['config'], scenario['name'])
        results[scenario['name']] = result
        
        if result['success']:
            print(f"✅ PASSED - Final losses: Train={result['final_train_loss']:.6f}, Val={result['final_val_loss']:.6f}")
        else:
            print(f"❌ FAILED - {result['error']}")
            if 'epoch' in result:
                print(f"   Failed at epoch {result['epoch']}, batch {result.get('batch', 'N/A')}")
    
    # Summary report
    print(f"\n{'='*60}")
    print("📊 FEATURE TESTING SUMMARY")
    print(f"{'='*60}")
    
    for name, result in results.items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        if result['success']:
            improvement = result['train_improvement']
            print(f"{status} | {name:<35} | Train Improvement: {improvement:>6.1f}%")
        else:
            error = result['error'][:30] + "..." if len(result['error']) > 30 else result['error']
            print(f"{status} | {name:<35} | Error: {error}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"logs/feature_testing_{timestamp}.json"
    os.makedirs("logs", exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    # Identify problematic features
    print(f"\n🔍 ANALYSIS:")
    failed_features = [name for name, result in results.items() if not result['success']]
    
    if not failed_features:
        print("🎉 All features work correctly! No convergence issues found.")
    else:
        print("⚠️  Problematic features identified:")
        for feature in failed_features:
            print(f"   - {feature}: {results[feature]['error']}")
    
    return results

if __name__ == "__main__":
    print("Starting systematic enhanced feature testing...")
    results = test_enhanced_features()
    print("\n🏁 Feature testing completed!")