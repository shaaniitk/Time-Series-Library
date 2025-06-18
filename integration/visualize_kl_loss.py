#!/usr/bin/env python3
"""
Visualization of KL Loss in Bayesian Neural Networks
Shows what KL divergence actually measures and represents
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from layers.BayesianLayers import BayesianLinear

def visualize_kl_divergence():
    """Visualize what KL divergence captures in Bayesian layers"""
    
    print("🔍 KL Divergence Visualization: What Does It Actually Show?")
    print("=" * 60)
    
    # Create a simple Bayesian layer
    bayesian_layer = BayesianLinear(10, 5)
    
    # Sample multiple weight realizations
    n_samples = 1000
    weight_samples = []
    bias_samples = []
    
    for _ in range(n_samples):
        # Sample weights from the learned variational distribution
        w_mean = bayesian_layer.weight_mu.data
        w_std = torch.exp(0.5 * bayesian_layer.weight_logvar.data)
        w_sample = torch.normal(w_mean, w_std)
        weight_samples.append(w_sample.numpy())
        
        # Sample biases
        b_mean = bayesian_layer.bias_mu.data
        b_std = torch.exp(0.5 * bayesian_layer.bias_logvar.data)
        b_sample = torch.normal(b_mean, b_std)
        bias_samples.append(b_sample.numpy())
    
    weight_samples = np.array(weight_samples)  # [n_samples, out_features, in_features]
    bias_samples = np.array(bias_samples)      # [n_samples, out_features]
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('KL Divergence: Parameter Uncertainty Visualization', fontsize=16, fontweight='bold')
    
    # 1. Weight Distribution Evolution
    # Take first weight parameter [0,0] as example
    weight_param = weight_samples[:, 0, 0]
    
    axes[0,0].hist(weight_param, bins=50, alpha=0.7, color='blue', density=True, label='Learned Distribution')
    
    # Overlay prior (standard normal)
    x_range = np.linspace(weight_param.min(), weight_param.max(), 100)
    prior_pdf = stats.norm.pdf(x_range, 0, 1)  # Standard normal prior
    axes[0,0].plot(x_range, prior_pdf, 'r-', linewidth=2, label='Prior (N(0,1))')
    
    axes[0,0].set_title('Weight Parameter Distribution\nvs Prior')
    axes[0,0].set_xlabel('Weight Value')
    axes[0,0].set_ylabel('Density')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Uncertainty Heatmap - Weight Matrix
    weight_means = bayesian_layer.weight_mu.data.numpy()
    weight_stds = torch.exp(0.5 * bayesian_layer.weight_logvar.data).numpy()
    
    im1 = axes[0,1].imshow(weight_means, cmap='RdBu', aspect='auto')
    axes[0,1].set_title('Weight Means\n(Learned Parameters)')
    axes[0,1].set_xlabel('Input Features')
    axes[0,1].set_ylabel('Output Features')
    plt.colorbar(im1, ax=axes[0,1])
    
    im2 = axes[0,2].imshow(weight_stds, cmap='Reds', aspect='auto')
    axes[0,2].set_title('Weight Uncertainties (Std)\n(What KL Loss Regularizes)')
    axes[0,2].set_xlabel('Input Features')
    axes[0,2].set_ylabel('Output Features')
    plt.colorbar(im2, ax=axes[0,2])
    
    # 3. KL Divergence Components
    # Compute KL divergence manually for visualization
    # KL(q||p) = 0.5 * [tr(Σ_p^-1 * Σ_q) + (μ_p - μ_q)^T * Σ_p^-1 * (μ_p - μ_q) - k + log(det(Σ_p)/det(Σ_q))]
    # For diagonal covariance and zero-mean prior: KL = 0.5 * [σ_q^2 + μ_q^2 - 1 - log(σ_q^2)]
    
    weight_vars = torch.exp(bayesian_layer.weight_logvar.data)
    weight_kl_per_param = 0.5 * (weight_vars + weight_means**2 - 1 - bayesian_layer.weight_logvar.data)
    
    bias_vars = torch.exp(bayesian_layer.bias_logvar.data)
    bias_means = bayesian_layer.bias_mu.data
    bias_kl_per_param = 0.5 * (bias_vars + bias_means**2 - 1 - bayesian_layer.bias_logvar.data)
    
    # Flatten for visualization
    all_kl_values = torch.cat([weight_kl_per_param.flatten(), bias_kl_per_param.flatten()])
    
    axes[1,0].hist(all_kl_values.numpy(), bins=30, alpha=0.7, color='green')
    axes[1,0].set_title('KL Divergence Per Parameter\n(Individual Contributions)')
    axes[1,0].set_xlabel('KL Value')
    axes[1,0].set_ylabel('Count')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Mean vs Uncertainty Trade-off
    weight_means_flat = weight_means.flatten()
    weight_stds_flat = weight_stds.flatten()
    
    axes[1,1].scatter(np.abs(weight_means_flat), weight_stds_flat, alpha=0.6, s=20)
    axes[1,1].set_title('Parameter Magnitude vs Uncertainty\n(KL Loss Balances Both)')
    axes[1,1].set_xlabel('|Weight Mean|')
    axes[1,1].set_ylabel('Weight Std (Uncertainty)')
    axes[1,1].grid(True, alpha=0.3)
    
    # 5. KL Loss Components Breakdown
    total_kl = bayesian_layer.kl_divergence()
    mean_penalty = 0.5 * (weight_means**2).sum() + 0.5 * (bias_means**2).sum()
    var_penalty = 0.5 * (weight_vars - 1 - bayesian_layer.weight_logvar.data).sum() + \
                  0.5 * (bias_vars - 1 - bayesian_layer.bias_logvar.data).sum()
    
    components = ['Mean Penalty\n(Prevents large weights)', 'Variance Penalty\n(Controls uncertainty)']
    values = [mean_penalty.item(), var_penalty.item()]
    colors = ['orange', 'purple']
    
    bars = axes[1,2].bar(components, values, color=colors, alpha=0.7)
    axes[1,2].set_title(f'KL Loss Components\nTotal KL = {total_kl.item():.4f}')
    axes[1,2].set_ylabel('Contribution')
    axes[1,2].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        axes[1,2].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                      f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save the plot
    import os
    os.makedirs('pic', exist_ok=True)
    plt.savefig('pic/kl_loss_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print interpretations
    print("\n📊 KL Loss Interpretation:")
    print(f"Total KL Divergence: {total_kl.item():.6f}")
    print(f"Mean Penalty: {mean_penalty.item():.6f} (prevents overly large weights)")
    print(f"Variance Penalty: {var_penalty.item():.6f} (regularizes uncertainty)")
    print("\n🎯 What KL Loss Does:")
    print("1. 📉 Keeps weights close to prior (usually zero)")
    print("2. 🎚️  Controls parameter uncertainty (not too certain, not too uncertain)")
    print("3. 🛡️  Prevents overfitting by regularizing weight distributions")
    print("4. ⚖️  Balances between fitting data and staying close to prior knowledge")
    
    return total_kl.item()

if __name__ == "__main__":
    kl_value = visualize_kl_divergence()
    print(f"\n✅ KL Loss visualization saved to: pic/kl_loss_visualization.png")
    print(f"🔢 Example KL value: {kl_value:.6f}")
