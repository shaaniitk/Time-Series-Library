#!/usr/bin/env python3
"""
Example of how to use KL Loss Tuning in practice
"""

import torch
import torch.nn as nn
import numpy as np
from utils.kl_tuning import KLTuner, suggest_kl_weight
from models.BayesianEnhancedAutoformer import BayesianEnhancedAutoformer

def example_adaptive_kl_tuning():
    """Example of adaptive KL tuning during training"""
    
    # Create model
    configs = type('Args', (), {
        'seq_len': 96, 'label_len': 48, 'pred_len': 24,
        'enc_in': 7, 'dec_in': 7, 'c_out': 1,
        'd_model': 16, 'd_ff': 32, 'e_layers': 1, 'd_layers': 1,
        'n_heads': 2, 'factor': 1, 'dropout': 0.1,
        'activation': 'gelu', 'output_attention': False,
        'moving_avg': 5, 'train_only': True,
        
        # Bayesian settings
        'bayesian_layers': True,
        'prior_std': 0.1,
        'kl_weight': 0.01,  # Initial weight
        
        # Quantile settings
        'quantile_mode': True,
        'quantiles': [0.1, 0.25, 0.5, 0.75, 0.9]
    })()
    
    model = BayesianEnhancedAutoformer(configs)
    
    # Create KL tuner with target 10% KL contribution
    kl_tuner = KLTuner(
        model=model,
        target_kl_percentage=0.1,  # 10% of total loss
        min_weight=1e-6,
        max_weight=1e-1
    )
    
    # Simulate training loop
    print("🚀 Adaptive KL Tuning Example")
    print("=" * 50)
    
    for epoch in range(20):
        # Simulate losses (in practice, these come from your training loop)
        data_loss = 0.5 + 0.3 * np.exp(-epoch/10)  # Decreasing data loss
        kl_loss = 2.0 + 0.5 * np.sin(epoch/3)      # Fluctuating KL loss
        
        # Update KL weight adaptively
        new_weight, kl_contribution = kl_tuner.update_kl_weight(
            epoch=epoch,
            data_loss=data_loss,
            kl_loss=kl_loss,
            method='adaptive'
        )
        
        total_loss = data_loss + new_weight * kl_loss
        
        print(f"Epoch {epoch:2d}: "
              f"Data={data_loss:.3f}, KL={kl_loss:.3f}, "
              f"Weight={new_weight:.2e}, KL%={kl_contribution*100:.1f}%, "
              f"Total={total_loss:.3f}")
    
    # Plot the tuning history
    kl_tuner.plot_kl_tuning_history('adaptive_kl_tuning.png')

def example_annealing_schedules():
    """Example of different annealing schedules"""
    
    configs = type('Args', (), {
        'seq_len': 96, 'label_len': 48, 'pred_len': 24,
        'enc_in': 7, 'dec_in': 7, 'c_out': 1,
        'd_model': 16, 'd_ff': 32, 'e_layers': 1, 'd_layers': 1,
        'n_heads': 2, 'factor': 1, 'dropout': 0.1,
        'activation': 'gelu', 'output_attention': False,
        'moving_avg': 5, 'train_only': True,
        'bayesian_layers': True, 'prior_std': 0.1, 'kl_weight': 0.01
    })()
    
    model = BayesianEnhancedAutoformer(configs)
    kl_tuner = KLTuner(model, min_weight=1e-4, max_weight=1e-1)
    
    print("\n📊 Annealing Schedule Comparison")
    print("=" * 50)
    
    schedules = ['linear', 'cosine', 'exponential', 'cyclical']
    total_epochs = 50
    
    for schedule in schedules:
        print(f"\n{schedule.capitalize()} Schedule:")
        model.kl_weight = 0.01  # Reset
        
        for epoch in [0, 10, 25, 40, 49]:  # Sample epochs
            weight = kl_tuner.annealing_schedule(epoch, total_epochs, schedule)
            print(f"  Epoch {epoch:2d}: KL weight = {weight:.2e}")

def example_practical_usage():
    """How to integrate KL tuning into your training loop"""
    
    print("\n🛠️  Practical Integration Example")
    print("=" * 50)
    
    # 1. Suggest initial KL weight
    print("Step 1: Suggest initial KL weight")
    initial_data_loss = 0.8  # From your first few batches
    suggested_weight = suggest_kl_weight(initial_data_loss, target_percentage=0.1)
    
    # 2. Create model with suggested weight
    configs = type('Args', (), {
        'seq_len': 96, 'label_len': 48, 'pred_len': 24,
        'enc_in': 7, 'dec_in': 7, 'c_out': 1,
        'd_model': 16, 'd_ff': 32, 'e_layers': 1, 'd_layers': 1,
        'n_heads': 2, 'factor': 1, 'dropout': 0.1,
        'activation': 'gelu', 'output_attention': False,
        'moving_avg': 5, 'train_only': True,
        'bayesian_layers': True, 'prior_std': 0.1,
        'kl_weight': suggested_weight
    })()
    
    model = BayesianEnhancedAutoformer(configs)
    kl_tuner = KLTuner(model, target_kl_percentage=0.1)
    
    print(f"\nStep 2: Created model with KL weight = {suggested_weight:.2e}")
    
    # 3. Training loop with adaptive tuning
    print("\nStep 3: Training loop with adaptive KL tuning")
    
    """
    # In your actual training loop, you would do:
    
    for epoch in range(num_epochs):
        epoch_data_loss = 0.0
        epoch_kl_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Forward pass
            output = model(data)
            
            # Compute losses
            data_loss = criterion(output, target)
            kl_loss = model.kl_loss()
            
            total_loss = data_loss + model.kl_weight * kl_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_data_loss += data_loss.item()
            epoch_kl_loss += kl_loss.item()
        
        # Average losses for the epoch
        avg_data_loss = epoch_data_loss / len(train_loader)
        avg_kl_loss = epoch_kl_loss / len(train_loader)
        
        # Update KL weight adaptively
        new_weight, kl_contribution = kl_tuner.update_kl_weight(
            epoch=epoch,
            data_loss=avg_data_loss,
            kl_loss=avg_kl_loss,
            method='adaptive'
        )
        
        print(f"Epoch {epoch}: Data={avg_data_loss:.3f}, KL={avg_kl_loss:.3f}, "
              f"KL_weight={new_weight:.2e}, KL%={kl_contribution*100:.1f}%")
    
    # Plot tuning history
    kl_tuner.plot_kl_tuning_history('training_kl_history.png')
    """
    
    print("✅ See commented code above for actual training loop integration")

if __name__ == "__main__":
    # Run examples
    example_adaptive_kl_tuning()
    example_annealing_schedules() 
    example_practical_usage()
    
    print("\n🎯 KL Tuning Summary:")
    print("=" * 50)
    print("1. Use suggest_kl_weight() to get initial weight")
    print("2. Use adaptive tuning for automatic adjustment")
    print("3. Target 5-20% KL contribution for good regularization")
    print("4. Monitor plots to ensure stable training")
    print("5. Lower KL weight if model underfits")
    print("6. Higher KL weight if model overfits")
