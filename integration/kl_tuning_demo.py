#!/usr/bin/env python3
"""
Simple KL Loss Tuning Example
"""

import torch
import numpy as np
from utils.kl_tuning import KLTuner, suggest_kl_weight

class MockBayesianModel:
    """Simple mock model for demonstration"""
    def __init__(self, initial_kl_weight=0.01):
        self.kl_weight = initial_kl_weight

def demonstrate_kl_tuning():
    """Demonstrate KL tuning strategies"""
    
    print("🎯 KL Loss Tuning Demonstration")
    print("=" * 50)
    
    # 1. Suggest initial KL weight
    print("\n1. Initial KL Weight Suggestion:")
    print("-" * 30)
    data_loss_magnitude = 0.5  # Typical MSE loss
    suggested_weight = suggest_kl_weight(data_loss_magnitude, target_percentage=0.1)
    
    # 2. Create mock model and tuner
    model = MockBayesianModel(initial_kl_weight=suggested_weight)
    kl_tuner = KLTuner(
        model=model,
        target_kl_percentage=0.1,  # 10% of total loss
        min_weight=1e-6,
        max_weight=1e-1
    )
    
    # 3. Simulate adaptive tuning
    print("\n2. Adaptive KL Tuning Simulation:")
    print("-" * 35)
    print("Epoch | Data Loss | KL Loss | KL Weight | KL% | Total Loss")
    print("-" * 60)
    
    for epoch in range(15):
        # Simulate realistic loss patterns
        data_loss = 0.6 + 0.4 * np.exp(-epoch/8)  # Decreasing data loss
        kl_loss = 3.0 + 1.0 * np.sin(epoch/2) + 0.5 * np.random.normal()  # Fluctuating KL
        
        # Update KL weight adaptively
        new_weight, kl_contribution = kl_tuner.update_kl_weight(
            epoch=epoch,
            data_loss=data_loss,
            kl_loss=kl_loss,
            method='adaptive'
        )
        
        total_loss = data_loss + new_weight * kl_loss
        
        print(f" {epoch:4d} | {data_loss:8.3f} | {kl_loss:7.3f} | "
              f"{new_weight:9.2e} | {kl_contribution*100:3.0f}% | {total_loss:9.3f}")
    
    # 4. Show annealing schedules
    print("\n3. KL Weight Annealing Schedules:")
    print("-" * 35)
    
    schedules = ['linear', 'cosine', 'exponential', 'cyclical']
    total_epochs = 50
    
    print("Epoch |", end="")
    for schedule in schedules:
        print(f"  {schedule:11s} |", end="")
    print()
    print("-" * (8 + 15 * len(schedules)))
    
    for epoch in [0, 10, 20, 30, 40, 49]:
        print(f" {epoch:4d} |", end="")
        for schedule in schedules:
            weight = kl_tuner.annealing_schedule(epoch, total_epochs, schedule)
            print(f"  {weight:11.2e} |", end="")
        print()
    
    # 5. Create tuning history plot
    print(f"\n4. Generating KL tuning history plot...")
    try:
        fig = kl_tuner.plot_kl_tuning_history('kl_tuning_demo.png')
        print("✅ Plot saved as 'kl_tuning_demo.png'")
    except Exception as e:
        print(f"⚠️  Could not save plot: {e}")
    
    # 6. Recommendations
    print("\n5. KL Tuning Best Practices:")
    print("-" * 30)
    print("✅ Start with suggested initial weight")
    print("✅ Use adaptive tuning for stable training")
    print("✅ Target 5-20% KL contribution")
    print("✅ Monitor KL percentage over time")
    print("✅ Lower weight if underfitting")
    print("✅ Raise weight if overfitting")
    
    final_kl_pct = kl_tuner.kl_percentage_history[-1] * 100
    print(f"\n📊 Final KL contribution: {final_kl_pct:.1f}%")
    print(f"🎯 Target was: {kl_tuner.target_kl_percentage*100:.1f}%")
    
    if abs(final_kl_pct - kl_tuner.target_kl_percentage*100) < 2:
        print("✅ Successfully reached target KL contribution!")
    else:
        print("⚠️  May need more epochs to reach target")

def demonstrate_kl_scenarios():
    """Show KL tuning for different scenarios"""
    
    print("\n\n🔬 KL Tuning for Different Scenarios")
    print("=" * 50)
    
    scenarios = [
        ("Conservative (5% KL)", 0.5, 0.05, "Low regularization, focus on fit"),
        ("Balanced (10% KL)", 0.5, 0.10, "Standard choice for most cases"),
        ("Heavy Reg (20% KL)", 0.5, 0.20, "Strong regularization, prevent overfit"),
        ("Small Loss Scale", 0.1, 0.10, "When data loss is very small"),
        ("Large Loss Scale", 2.0, 0.10, "When data loss is large"),
    ]
    
    for name, data_loss, target_pct, description in scenarios:
        print(f"\n{name}:")
        print(f"  Description: {description}")
        suggested = suggest_kl_weight(data_loss, target_pct)
        print(f"  Suggested KL weight: {suggested:.2e}")

if __name__ == "__main__":
    demonstrate_kl_tuning()
    demonstrate_kl_scenarios()
