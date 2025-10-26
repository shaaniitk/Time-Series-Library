#!/usr/bin/env python3
"""
Analyze training_diagnostic.log to identify mode collapse and gradient flow issues.

Usage:
    python analyze_gradient_logs.py
"""

import re
from collections import defaultdict
from pathlib import Path


def parse_diagnostic_log(log_path: str = "training_diagnostic.log") -> dict:
    """Parse the training diagnostic log file."""
    if not Path(log_path).exists():
        print(f"❌ Log file not found: {log_path}")
        return {}
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Extract prediction/target statistics
    pred_stats = re.findall(r'y_pred_for_loss mean/std: ([-\d.]+) / ([\d.]+)', content)
    true_stats = re.findall(r'y_true_for_loss mean/std: ([-\d.]+) / ([\d.]+)', content)
    
    # Extract gradient norms
    grad_norms = re.findall(r'(\S+):\s*grad_norm: ([\d.]+)', content)
    
    # Extract loss values
    raw_losses = re.findall(r'raw_loss \(full batch loss\): ([\d.]+)', content)
    
    # Extract learning rates
    learning_rates = re.findall(r'current_lr: ([\d.]+)', content)
    
    return {
        'pred_stats': [(float(m), float(s)) for m, s in pred_stats],
        'true_stats': [(float(m), float(s)) for m, s in true_stats],
        'grad_norms': [(name, float(norm)) for name, norm in grad_norms],
        'raw_losses': [float(l) for l in raw_losses],
        'learning_rates': [float(lr) for lr in learning_rates]
    }


def analyze_mode_collapse(data: dict) -> None:
    """Analyze prediction variance for mode collapse."""
    print("\n" + "="*80)
    print("MODE COLLAPSE ANALYSIS")
    print("="*80)
    
    if not data.get('pred_stats') or not data.get('true_stats'):
        print("❌ No prediction/target statistics found in log")
        return
    
    pred_stats = data['pred_stats'][-20:]  # Last 20 batches
    true_stats = data['true_stats'][-20:]
    
    pred_stds = [s for _, s in pred_stats]
    true_stds = [s for _, s in true_stats]
    
    avg_pred_std = sum(pred_stds) / len(pred_stds)
    avg_true_std = sum(true_stds) / len(true_stds)
    variance_ratio = avg_true_std / avg_pred_std if avg_pred_std > 0 else float('inf')
    
    print(f"\n📊 Recent Statistics (last {len(pred_stats)} batches):")
    print(f"   Prediction Std:  {avg_pred_std:.6f}")
    print(f"   Target Std:      {avg_true_std:.6f}")
    print(f"   Variance Ratio:  1:{variance_ratio:.1f}")
    
    if variance_ratio > 10:
        print(f"\n❌ SEVERE MODE COLLAPSE DETECTED!")
        print(f"   Model predictions are {variance_ratio:.1f}x less variable than targets")
        print(f"   Model is outputting near-constant values")
        print(f"\n💡 Recommendations:")
        print(f"   1. Increase learning rate (try 3x current value)")
        print(f"   2. Reduce weight_decay (try 10x smaller)")
        print(f"   3. Increase clip_grad_norm (try 5.0 instead of 1.0)")
        print(f"   4. Check output layer initialization")
    elif variance_ratio > 3:
        print(f"\n⚠️  MODERATE MODE COLLAPSE")
        print(f"   Model predictions are {variance_ratio:.1f}x less variable than targets")
        print(f"   Consider increasing learning rate or reducing regularization")
    else:
        print(f"\n✅ Prediction variance looks healthy")


def analyze_gradients(data: dict) -> None:
    """Analyze gradient flow."""
    print("\n" + "="*80)
    print("GRADIENT FLOW ANALYSIS")
    print("="*80)
    
    if not data.get('grad_norms'):
        print("❌ No gradient norms found in log")
        return
    
    # Group by layer name
    layer_grads = defaultdict(list)
    for name, norm in data['grad_norms']:
        # Extract base layer name (before .weight/.bias)
        base_name = '.'.join(name.split('.')[:-1])
        layer_grads[base_name].append(norm)
    
    # Find dead layers (zero gradients)
    dead_layers = []
    active_layers = []
    
    for layer, norms in layer_grads.items():
        avg_norm = sum(norms) / len(norms)
        if avg_norm < 1e-8:
            dead_layers.append((layer, avg_norm))
        else:
            active_layers.append((layer, avg_norm))
    
    print(f"\n📊 Gradient Statistics:")
    print(f"   Total layers: {len(layer_grads)}")
    print(f"   Active layers: {len(active_layers)}")
    print(f"   Dead layers (grad ~0): {len(dead_layers)}")
    
    if dead_layers:
        print(f"\n❌ DEAD LAYERS DETECTED:")
        for layer, norm in sorted(dead_layers, key=lambda x: x[1]):
            print(f"   {layer}: {norm:.2e}")
        print(f"\n💡 These layers are not contributing to learning")
        print(f"   Check if they're frozen, poorly initialized, or bypassed")
    
    if active_layers:
        active_layers_sorted = sorted(active_layers, key=lambda x: x[1], reverse=True)
        print(f"\n✅ Top 10 Active Layers (by gradient norm):")
        for layer, norm in active_layers_sorted[:10]:
            print(f"   {layer}: {norm:.6f}")
        
        print(f"\n📉 Bottom 10 Active Layers (by gradient norm):")
        for layer, norm in active_layers_sorted[-10:]:
            print(f"   {layer}: {norm:.6f}")


def analyze_loss_trend(data: dict) -> None:
    """Analyze loss progression."""
    print("\n" + "="*80)
    print("LOSS TREND ANALYSIS")
    print("="*80)
    
    if not data.get('raw_losses'):
        print("❌ No loss values found in log")
        return
    
    losses = data['raw_losses']
    
    # Take first/last 50 batches
    early_losses = losses[:min(50, len(losses)//2)]
    recent_losses = losses[-50:]
    
    avg_early = sum(early_losses) / len(early_losses)
    avg_recent = sum(recent_losses) / len(recent_losses)
    improvement = ((avg_early - avg_recent) / avg_early) * 100
    
    print(f"\n📊 Loss Progression:")
    print(f"   Early batches avg:  {avg_early:.6f}")
    print(f"   Recent batches avg: {avg_recent:.6f}")
    print(f"   Improvement: {improvement:.1f}%")
    
    if improvement < 5:
        print(f"\n❌ MINIMAL IMPROVEMENT DETECTED")
        print(f"   Loss has not decreased significantly")
        print(f"   Model is not learning effectively")
    elif improvement < 20:
        print(f"\n⚠️  SLOW LEARNING")
        print(f"   Loss is decreasing but slowly")
    else:
        print(f"\n✅ Good learning progress")


def analyze_learning_rate(data: dict) -> None:
    """Analyze learning rate schedule."""
    print("\n" + "="*80)
    print("LEARNING RATE SCHEDULE")
    print("="*80)
    
    if not data.get('learning_rates'):
        print("❌ No learning rate values found in log")
        return
    
    lrs = data['learning_rates']
    
    print(f"\n📊 LR Progression (showing 10 samples):")
    step_size = max(1, len(lrs) // 10)
    for i in range(0, len(lrs), step_size):
        print(f"   Step {i:4d}: lr={lrs[i]:.8f}")
    
    if len(lrs) > 1:
        initial_lr = lrs[0]
        final_lr = lrs[-1]
        print(f"\n   Initial LR: {initial_lr:.8f}")
        print(f"   Current LR: {final_lr:.8f}")
        print(f"   Change: {((final_lr - initial_lr) / initial_lr * 100):+.1f}%")


def main():
    """Main analysis function."""
    print("🔍 Analyzing training diagnostics...")
    
    data = parse_diagnostic_log()
    
    if not data:
        print("\n❌ Failed to parse diagnostic log")
        print("   Ensure training_diagnostic.log exists and has been generated")
        return
    
    # Run analyses
    analyze_mode_collapse(data)
    analyze_gradients(data)
    analyze_loss_trend(data)
    analyze_learning_rate(data)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nSee TRAINING_FAILURE_ANALYSIS.md for detailed recommendations")


if __name__ == "__main__":
    main()
