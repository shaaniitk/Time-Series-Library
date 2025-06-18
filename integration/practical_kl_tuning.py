#!/usr/bin/env python3
"""
Practical KL Tuning Integration for BayesianEnhancedAutoformer
"""

import torch
import torch.nn as nn
import numpy as np
from utils.kl_tuning import KLTuner, suggest_kl_weight

def create_bayesian_model_configs():
    """Create realistic model configurations"""
    configs = type('Args', (), {
        # Data settings
        'seq_len': 96, 'label_len': 48, 'pred_len': 24,
        'enc_in': 7, 'dec_in': 7, 'c_out': 1,
        
        # Model architecture
        'd_model': 64, 'd_ff': 128, 'e_layers': 2, 'd_layers': 1,
        'n_heads': 4, 'factor': 1, 'dropout': 0.1,
        'activation': 'gelu', 'output_attention': False,
        'moving_avg': 25,
        
        # Task settings
        'task_name': 'long_term_forecast',
        'is_training': 1,
        'model_id': 'bayesian_test',
        'model': 'BayesianEnhancedAutoformer',
        'data': 'custom', 'root_path': './data/',
        'data_path': 'comprehensive_dynamic_features.csv',
        'features': 'MS', 'target': 'nifty_return',
        'freq': 'h', 'checkpoints': './checkpoints/',
        'train_only': True,
        
        # Bayesian settings
        'bayesian_layers': True,
        'prior_std': 0.1,
        'kl_weight': 0.01,  # Will be updated by tuner
        
        # Quantile settings
        'quantile_mode': True,
        'quantiles': [0.1, 0.25, 0.5, 0.75, 0.9]
    })()
    return configs

def simulate_training_with_kl_tuning():
    """Simulate training loop with KL tuning"""
    
    print("🚀 Training Simulation with KL Tuning")
    print("=" * 50)
    
    # Create configurations
    configs = create_bayesian_model_configs()
    
    # Step 1: Estimate initial data loss magnitude
    print("Step 1: Estimating initial data loss magnitude...")
    
    # In practice, you'd run a few batches to estimate this
    estimated_data_loss = 0.4  # From initial training batches
    target_kl_percentage = 0.10  # 10% KL contribution
    
    suggested_weight = suggest_kl_weight(estimated_data_loss, target_kl_percentage)
    configs.kl_weight = suggested_weight
    
    print(f"✅ Set initial KL weight: {suggested_weight:.2e}")
    
    # Step 2: Create KL tuner
    print("\nStep 2: Setting up KL tuner...")
    
    # Mock model for demo (in practice, use your actual model)
    mock_model = type('MockModel', (), {
        'kl_weight': suggested_weight,
        'configs': configs
    })()
    
    kl_tuner = KLTuner(
        model=mock_model,
        target_kl_percentage=target_kl_percentage,
        min_weight=1e-5,
        max_weight=5e-2
    )
    
    print(f"✅ KL tuner ready (target: {target_kl_percentage*100:.0f}%)")
    
    # Step 3: Simulate training epochs
    print("\nStep 3: Training simulation...")
    print("-" * 70)
    print("Epoch | Data Loss | KL Loss | KL Weight | KL% | Total | Status")
    print("-" * 70)
    
    best_total_loss = float('inf')
    patience_counter = 0
    patience = 5
    
    for epoch in range(25):
        # Simulate realistic loss evolution
        # Data loss: decreases with training, some noise
        base_data_loss = 0.4 * np.exp(-epoch/15) + 0.1
        data_loss = base_data_loss + 0.05 * np.random.normal()
        
        # KL loss: fluctuates, generally decreases
        base_kl_loss = 2.0 + 0.5 * np.sin(epoch/3) - 0.02 * epoch
        kl_loss = max(0.1, base_kl_loss + 0.3 * np.random.normal())
        
        # Update KL weight using adaptive tuning
        new_weight, kl_contribution = kl_tuner.update_kl_weight(
            epoch=epoch,
            data_loss=data_loss,
            kl_loss=kl_loss,
            method='adaptive'
        )
        
        total_loss = data_loss + new_weight * kl_loss
        
        # Determine status
        status = ""
        if total_loss < best_total_loss:
            best_total_loss = total_loss
            patience_counter = 0
            status = "📈 BEST"
        else:
            patience_counter += 1
            if patience_counter >= patience:
                status = "⚠️  EARLY STOP"
        
        if kl_contribution > 0.25:
            status += " 🔥 HIGH KL"
        elif kl_contribution < 0.03:
            status += " ❄️  LOW KL"
        
        print(f" {epoch:4d} | {data_loss:8.3f} | {kl_loss:7.3f} | "
              f"{new_weight:9.2e} | {kl_contribution*100:3.0f}% | "
              f"{total_loss:5.3f} | {status}")
        
        # Early stopping simulation
        if patience_counter >= patience:
            print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
            break
    
    # Step 4: Analyze results
    print(f"\n📊 Training Analysis:")
    print("-" * 30)
    
    avg_kl_pct = np.mean(kl_tuner.kl_percentage_history[-5:]) * 100
    final_weight = mock_model.kl_weight
    
    print(f"Final KL weight: {final_weight:.2e}")
    print(f"Average KL% (last 5 epochs): {avg_kl_pct:.1f}%")
    print(f"Target KL%: {target_kl_percentage*100:.0f}%")
    print(f"Best total loss: {best_total_loss:.3f}")
    
    if abs(avg_kl_pct - target_kl_percentage*100) < 3:
        print("✅ Successfully maintained target KL contribution")
    else:
        print("⚠️  KL contribution deviated from target")
    
    # Generate plot
    try:
        kl_tuner.plot_kl_tuning_history('training_kl_tuning.png')
        print("📈 KL tuning plot saved as 'training_kl_tuning.png'")
    except Exception as e:
        print(f"⚠️  Could not save plot: {e}")

def show_kl_tuning_guidelines():
    """Show practical guidelines for KL tuning"""
    
    print("\n\n📖 KL Tuning Guidelines for Bayesian Models")
    print("=" * 50)
    
    guidelines = [
        ("🎯 Target Range", "5-20% of total loss for good regularization"),
        ("🚀 Start Point", "Use suggest_kl_weight() for initial estimate"),
        ("⚖️  Adaptive Tuning", "Automatically maintains target percentage"),
        ("📈 Monitor Trends", "Watch KL% over time, not just absolute values"),
        ("🔥 Too High KL", "Model underfits, reduce weight or check priors"),
        ("❄️  Too Low KL", "Model may overfit, increase weight"),
        ("📊 Early Stopping", "Consider both data loss and KL contribution"),
        ("🎨 Visualization", "Plot KL history to diagnose issues"),
    ]
    
    for category, description in guidelines:
        print(f"{category:15s}: {description}")
    
    print("\n🔧 Troubleshooting:")
    print("-" * 20)
    troubleshooting = [
        ("KL explodes", "Lower prior_std, check layer initialization"),
        ("KL too stable", "Increase learning rate or prior_std"),
        ("Erratic KL%", "Use longer smoothing window in adaptive tuning"),
        ("Poor convergence", "Try annealing schedule instead of adaptive"),
    ]
    
    for problem, solution in troubleshooting:
        print(f"• {problem:15s}: {solution}")

def demonstrate_different_tuning_methods():
    """Compare different KL tuning methods"""
    
    print("\n\n🔬 Comparing KL Tuning Methods")
    print("=" * 50)
    
    methods = [
        ('fixed', {}, "No adjustment, constant KL weight"),
        ('adaptive', {}, "Automatic adjustment to target percentage"),
        ('annealing', {'total_epochs': 50, 'schedule_type': 'linear'}, "Linear decrease over time"),
        ('annealing', {'total_epochs': 50, 'schedule_type': 'cosine'}, "Cosine annealing schedule"),
    ]
    
    mock_model = type('MockModel', (), {'kl_weight': 0.01})()
    
    for method, kwargs, description in methods:
        print(f"\n{method.capitalize()} Method: {description}")
        print("-" * 40)
        
        # Reset model
        mock_model.kl_weight = 0.01
        kl_tuner = KLTuner(mock_model, target_kl_percentage=0.1)
        
        # Show weights for sample epochs
        sample_epochs = [0, 10, 25, 40, 49]
        print("Epoch | KL Weight | Description")
        print("-" * 35)
        
        for epoch in sample_epochs:
            if method == 'fixed':
                weight = 0.01
                desc = "Constant"
            elif method == 'adaptive':
                # Simulate adaptive adjustment
                data_loss = 0.5 + 0.3 * np.exp(-epoch/15)
                kl_loss = 2.0 - 0.02 * epoch
                weight, _ = kl_tuner.update_kl_weight(epoch, data_loss, kl_loss, method, **kwargs)
                desc = f"Target: {kl_tuner.target_kl_percentage*100:.0f}%"
            else:  # annealing
                weight = kl_tuner.annealing_schedule(epoch, **kwargs)
                desc = kwargs['schedule_type']
            
            print(f" {epoch:4d} | {weight:9.2e} | {desc}")

if __name__ == "__main__":
    simulate_training_with_kl_tuning()
    show_kl_tuning_guidelines()
    demonstrate_different_tuning_methods()
    
    print("\n🎯 Summary:")
    print("=" * 50)
    print("✅ Use adaptive KL tuning for most cases")
    print("✅ Target 10% KL contribution as starting point")
    print("✅ Monitor KL% trends, not just absolute values")
    print("✅ Adjust based on overfitting/underfitting signals")
    print("✅ Save and visualize KL tuning history")
