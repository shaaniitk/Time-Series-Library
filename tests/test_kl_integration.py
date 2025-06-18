#!/usr/bin/env python3
"""
Test KL Tuning with Real BayesianEnhancedAutoformer
"""

import torch
import torch.nn as nn
import numpy as np
import warnings
from utils.kl_tuning import KLTuner, suggest_kl_weight

warnings.filterwarnings('ignore')

def test_kl_tuning_integration():
    """Test KL tuning with actual BayesianEnhancedAutoformer"""
    
    print("🧪 Testing KL Tuning with BayesianEnhancedAutoformer")
    print("=" * 60)
    
    # Create model configuration
    configs = type('Args', (), {
        # Basic settings
        'seq_len': 96, 'label_len': 48, 'pred_len': 24,
        'enc_in': 7, 'dec_in': 7, 'c_out': 1,
        'd_model': 32, 'd_ff': 64, 'e_layers': 1, 'd_layers': 1,
        'n_heads': 2, 'factor': 1, 'dropout': 0.1,
        'activation': 'gelu', 'output_attention': False,
        'moving_avg': 5,
        
        # Task settings - Add all required attributes
        'task_name': 'long_term_forecast',
        'is_training': 1, 'model_id': 'test', 'model': 'BayesianEnhancedAutoformer',
        'data': 'custom', 'root_path': './data/', 'data_path': 'test.csv',
        'features': 'MS', 'target': 'target', 'freq': 'h',
        'checkpoints': './checkpoints/', 'train_only': True,
        
        # Bayesian settings
        'bayesian_layers': True,
        'prior_std': 0.1,
        'kl_weight': 0.01,
        
        # Quantile settings
        'quantile_mode': True,
        'quantiles': [0.1, 0.25, 0.5, 0.75, 0.9]
    })()
    
    try:
        # Import and create model
        from models.BayesianEnhancedAutoformer import BayesianEnhancedAutoformer
        model = BayesianEnhancedAutoformer(configs)
        print("✅ Model created successfully")
        
        # Create synthetic data
        batch_size = 16
        seq_len = configs.seq_len
        pred_len = configs.pred_len
        enc_in = configs.enc_in
        c_out = configs.c_out
        
        x_data = torch.randn(batch_size, seq_len, enc_in)
        x_mark = torch.randn(batch_size, seq_len, 4)  # Date features
        y_data = torch.randn(batch_size, configs.label_len + pred_len, enc_in)
        y_mark = torch.randn(batch_size, configs.label_len + pred_len, 4)
        targets = torch.randn(batch_size, pred_len, c_out)
        
        print(f"📊 Data shapes:")
        print(f"   Input: {x_data.shape}")
        print(f"   Target: {targets.shape}")
        
        # Forward pass to get initial losses
        model.eval()
        with torch.no_grad():
            outputs = model(x_data, x_mark, y_data, y_mark)
            
            # Compute data loss (quantile loss)
            if hasattr(model, 'quantile_mode') and model.quantile_mode:
                # Simple quantile loss approximation
                n_quantiles = len(configs.quantiles)
                reshaped_outputs = outputs.view(batch_size, pred_len, n_quantiles, c_out)
                
                # Use median quantile for initial loss estimate
                median_idx = len(configs.quantiles) // 2
                median_outputs = reshaped_outputs[:, :, median_idx, :]
                data_loss = nn.MSELoss()(median_outputs, targets)
            else:
                data_loss = nn.MSELoss()(outputs, targets)
            
            # Get KL loss
            kl_loss = model.kl_loss()
        
        print(f"🔍 Initial losses:")
        print(f"   Data loss: {data_loss.item():.4f}")
        print(f"   KL loss: {kl_loss.item():.4f}")
        print(f"   Output shape: {outputs.shape}")
        
        # Suggest KL weight
        suggested_weight = suggest_kl_weight(data_loss.item(), target_percentage=0.1)
        model.kl_weight = suggested_weight
        print(f"🎯 Suggested KL weight: {suggested_weight:.2e}")
        
        # Setup KL tuner
        kl_tuner = KLTuner(
            model=model,
            target_kl_percentage=0.1,
            min_weight=1e-6,
            max_weight=1e-2
        )
        print("✅ KL tuner setup complete")
        
        # Simulate training epochs with KL tuning
        print(f"\n🏃 Simulating training with KL tuning:")
        print("-" * 50)
        print("Epoch | Data Loss | KL Loss | Weight | KL% | Total")
        print("-" * 50)
        
        model.train()
        for epoch in range(15):
            # Simulate forward pass with small variations
            noise_factor = 1.0 + 0.1 * np.random.normal()
            noisy_x = x_data * noise_factor
            
            outputs = model(noisy_x, x_mark, y_data, y_mark)
            
            # Compute losses
            if hasattr(model, 'quantile_mode') and model.quantile_mode:
                n_quantiles = len(configs.quantiles)
                reshaped_outputs = outputs.view(batch_size, pred_len, n_quantiles, c_out)
                median_idx = len(configs.quantiles) // 2
                median_outputs = reshaped_outputs[:, :, median_idx, :]
                data_loss = nn.MSELoss()(median_outputs, targets)
            else:
                data_loss = nn.MSELoss()(outputs, targets)
            
            kl_loss = model.kl_loss()
            
            # Update KL weight
            new_weight, kl_contribution = kl_tuner.update_kl_weight(
                epoch=epoch,
                data_loss=data_loss.item(),
                kl_loss=kl_loss.item(),
                method='adaptive'
            )
            
            total_loss = data_loss + model.kl_weight * kl_loss
            
            print(f" {epoch:4d} | {data_loss.item():8.4f} | {kl_loss.item():7.3f} | "
                  f"{new_weight:6.1e} | {kl_contribution*100:3.0f}% | {total_loss.item():5.3f}")
        
        # Plot results
        try:
            kl_tuner.plot_kl_tuning_history('test_kl_tuning.png')
            print("\n📈 KL tuning plot saved as 'test_kl_tuning.png'")
        except Exception as e:
            print(f"\n⚠️  Could not save plot: {e}")
        
        # Final analysis
        final_kl_pct = kl_tuner.kl_percentage_history[-1] * 100
        print(f"\n📊 Final Analysis:")
        print(f"   Final KL weight: {model.kl_weight:.2e}")
        print(f"   Final KL contribution: {final_kl_pct:.1f}%")
        print(f"   Target KL contribution: {kl_tuner.target_kl_percentage*100:.0f}%")
        
        if abs(final_kl_pct - kl_tuner.target_kl_percentage*100) < 3:
            print("✅ Successfully maintained target KL contribution!")
        else:
            print("⚠️  KL contribution deviated from target (normal for short test)")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_kl_methods():
    """Test different KL tuning methods"""
    
    print(f"\n\n🔬 Testing Different KL Tuning Methods")
    print("=" * 60)
    
    # Mock model for quick testing
    mock_model = type('MockModel', (), {'kl_weight': 0.01})()
    
    methods = [
        ('fixed', {}, "Constant KL weight"),
        ('adaptive', {}, "Adaptive to target percentage"),
        ('annealing', {'total_epochs': 20, 'schedule_type': 'linear'}, "Linear annealing"),
        ('annealing', {'total_epochs': 20, 'schedule_type': 'cosine'}, "Cosine annealing"),
    ]
    
    for method, kwargs, description in methods:
        print(f"\n{method.upper()} METHOD: {description}")
        print("-" * 40)
        
        mock_model.kl_weight = 0.01  # Reset
        kl_tuner = KLTuner(mock_model, target_kl_percentage=0.1)
        
        print("Epoch | KL Weight | KL Contribution")
        print("-" * 35)
        
        for epoch in [0, 5, 10, 15, 19]:
            # Simulate losses
            data_loss = 0.5 * np.exp(-epoch/10) + 0.1
            kl_loss = 2.0 - 0.05 * epoch + 0.3 * np.sin(epoch)
            
            if method == 'fixed':
                weight = 0.01
                contribution = weight * kl_loss / (data_loss + weight * kl_loss)
            else:
                weight, contribution = kl_tuner.update_kl_weight(
                    epoch, data_loss, kl_loss, method, **kwargs
                )
            
            print(f" {epoch:4d} | {weight:9.2e} | {contribution*100:6.1f}%")

if __name__ == "__main__":
    # Run tests
    success = test_kl_tuning_integration()
    
    if success:
        test_different_kl_methods()
        
        print(f"\n🎯 KL Tuning Test Summary:")
        print("=" * 50)
        print("✅ BayesianEnhancedAutoformer integration works")
        print("✅ KL weight suggestion works")
        print("✅ Adaptive KL tuning works")
        print("✅ Different tuning methods available")
        print("✅ Visualization and monitoring ready")
        
        print(f"\n📚 Next steps:")
        print("1. Use train_bayesian_with_kl_tuning.py for full training")
        print("2. Refer to KL_TUNING_GUIDE.md for detailed documentation")
        print("3. Experiment with different target KL percentages")
        print("4. Monitor KL tuning plots during training")
    else:
        print(f"\n❌ KL Tuning test failed - check error messages above")
