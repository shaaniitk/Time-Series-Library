#!/usr/bin/env python3
"""
KL Loss Tuning Utilities for Bayesian Models
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

class KLTuner:
    """Utility class for tuning KL loss in Bayesian models"""
    
    def __init__(self, model, target_kl_percentage=0.1, min_weight=1e-6, max_weight=1e-1):
        self.model = model
        self.target_kl_percentage = target_kl_percentage
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # History tracking
        self.kl_history = []
        self.data_loss_history = []
        self.kl_weight_history = []
        self.kl_percentage_history = []
    
    def compute_kl_contribution(self, data_loss, kl_loss, kl_weight):
        """Compute current KL contribution percentage"""
        total_loss = data_loss + kl_weight * kl_loss
        if total_loss > 0:
            return (kl_weight * kl_loss) / total_loss
        return 0.0
    
    def adaptive_kl_weight(self, data_loss, kl_loss, current_weight):
        """Adaptively adjust KL weight to target percentage"""
        if kl_loss <= 0:
            return current_weight
            
        # Calculate what weight would give us target percentage
        target_kl_contribution = self.target_kl_percentage * data_loss / (1 - self.target_kl_percentage)
        optimal_weight = target_kl_contribution / kl_loss
        
        # Clamp to reasonable range and smooth the adjustment
        optimal_weight = np.clip(optimal_weight, self.min_weight, self.max_weight)
        
        # Smooth adjustment (moving average)
        adjustment_rate = 0.1  # How fast to adjust
        new_weight = (1 - adjustment_rate) * current_weight + adjustment_rate * optimal_weight
        
        return new_weight
    
    def annealing_schedule(self, epoch, total_epochs, schedule_type='linear'):
        """Various annealing schedules for KL weight"""
        progress = epoch / total_epochs
        
        if schedule_type == 'linear':
            # Linear decrease from max to min
            return self.max_weight * (1 - progress) + self.min_weight * progress
            
        elif schedule_type == 'cosine':
            # Cosine annealing
            return self.min_weight + (self.max_weight - self.min_weight) * \
                   0.5 * (1 + np.cos(np.pi * progress))
                   
        elif schedule_type == 'exponential':
            # Exponential decay
            decay_rate = np.log(self.min_weight / self.max_weight) / total_epochs
            return self.max_weight * np.exp(decay_rate * epoch)
            
        elif schedule_type == 'cyclical':
            # Cyclical annealing (multiple cycles)
            cycles = 3
            cycle_progress = (progress * cycles) % 1.0
            return self.min_weight + (self.max_weight - self.min_weight) * \
                   0.5 * (1 + np.cos(np.pi * cycle_progress))
        
        return self.max_weight
    
    def update_kl_weight(self, epoch, data_loss, kl_loss, method='adaptive', **kwargs):
        """Update KL weight using specified method"""
        current_weight = self.model.kl_weight
        
        if method == 'adaptive':
            new_weight = self.adaptive_kl_weight(data_loss, kl_loss, current_weight)
            
        elif method == 'annealing':
            total_epochs = kwargs.get('total_epochs', 100)
            schedule_type = kwargs.get('schedule_type', 'linear')
            new_weight = self.annealing_schedule(epoch, total_epochs, schedule_type)
            
        elif method == 'fixed':
            new_weight = current_weight  # No change
            
        else:
            raise ValueError(f"Unknown KL tuning method: {method}")
        
        # Update model
        self.model.kl_weight = new_weight
        
        # Track history
        kl_contribution = self.compute_kl_contribution(data_loss, kl_loss, new_weight)
        self.kl_history.append(kl_loss)
        self.data_loss_history.append(data_loss)
        self.kl_weight_history.append(new_weight)
        self.kl_percentage_history.append(kl_contribution)
        
        return new_weight, kl_contribution
    
    def plot_kl_tuning_history(self, save_path='kl_tuning_history.png'):
        """Plot KL tuning history"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        epochs = range(len(self.kl_history))
        
        # KL Loss over time
        axes[0, 0].plot(epochs, self.kl_history, 'b-', label='KL Loss')
        axes[0, 0].set_title('KL Loss Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('KL Loss')
        axes[0, 0].grid(True)
        
        # Data Loss over time
        axes[0, 1].plot(epochs, self.data_loss_history, 'r-', label='Data Loss')
        axes[0, 1].set_title('Data Loss Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Data Loss')
        axes[0, 1].grid(True)
        
        # KL Weight over time
        axes[1, 0].plot(epochs, self.kl_weight_history, 'g-', label='KL Weight')
        axes[1, 0].set_title('KL Weight Over Time')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('KL Weight')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # KL Contribution Percentage
        target_line = [self.target_kl_percentage * 100] * len(epochs)
        axes[1, 1].plot(epochs, [p * 100 for p in self.kl_percentage_history], 'purple', label='KL Contribution %')
        axes[1, 1].plot(epochs, target_line, 'k--', alpha=0.7, label=f'Target: {self.target_kl_percentage*100:.1f}%')
        axes[1, 1].set_title('KL Contribution Percentage')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('KL Contribution (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"KL tuning history plot saved to: {save_path}")
        return fig

def suggest_kl_weight(data_loss_magnitude, target_percentage=0.1):
    """
    Suggest initial KL weight based on data loss magnitude
    
    Args:
        data_loss_magnitude: Typical magnitude of data loss (e.g., 0.5 for MSE)
        target_percentage: Desired KL contribution (0.1 = 10%)
    
    Returns:
        Suggested KL weight
    """
    # Rough heuristic: KL loss is often 1-10x larger than data loss initially
    estimated_kl_magnitude = data_loss_magnitude * 5  # Rough estimate
    
    target_kl_contribution = target_percentage * data_loss_magnitude / (1 - target_percentage)
    suggested_weight = target_kl_contribution / estimated_kl_magnitude
    
    # Clamp to reasonable range
    suggested_weight = np.clip(suggested_weight, 1e-6, 1e-1)
    
    print(f"Data loss magnitude: {data_loss_magnitude:.4f}")
    print(f"Target KL percentage: {target_percentage*100:.1f}%")
    print(f"Suggested KL weight: {suggested_weight:.2e}")
    
    return suggested_weight

if __name__ == "__main__":
    # Example usage
    print("🎯 KL Weight Suggestions:")
    print("=" * 40)
    
    # Different scenarios
    scenarios = [
        ("MSE Loss (small)", 0.1, 0.1),      # Small MSE, 10% KL
        ("MSE Loss (medium)", 0.5, 0.1),     # Medium MSE, 10% KL  
        ("MSE Loss (large)", 2.0, 0.1),      # Large MSE, 10% KL
        ("Conservative", 0.5, 0.05),         # 5% KL contribution
        ("Heavy Regularization", 0.5, 0.2),  # 20% KL contribution
    ]
    
    for name, data_loss, target_pct in scenarios:
        print(f"\n{name}:")
        suggest_kl_weight(data_loss, target_pct)
