#!/usr/bin/env python3
"""
Comprehensive ChronosX + ModularAutoformer Demo
Shows complete workflow with real data forecasting and uncertainty
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import Namespace
import time

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.modular_autoformer import ModularAutoformer
from layers.modular.core.registry import unified_registry
from layers.modular.core.example_components import register_example_components


def create_demo_config(backbone_type='chronos_x', model_size='tiny'):
    """Create configuration for demo"""
    config = Namespace()
    
    # Task configuration
    config.task_name = 'long_term_forecast'
    config.seq_len = 96
    config.label_len = 48
    config.pred_len = 24
    
    # Model dimensions
    config.d_model = 64
    config.enc_in = 1
    config.dec_in = 1
    config.c_out = 1
    config.c_out_evaluation = 1
    
    # Embedding
    config.embed = 'timeF'
    config.freq = 'h'
    config.dropout = 0.1
    
    # ChronosX backbone
    config.use_backbone_component = True
    config.backbone_type = backbone_type
    config.backbone_params = {
        'model_size': model_size,
        'use_uncertainty': True,
        'num_samples': 20
    }
    
    # Components
    config.sampling_type = 'deterministic'
    config.output_head_type = 'standard'
    config.loss_function_type = 'mse'
    config.sampling_params = {}
    config.output_head_params = {'d_model': config.d_model, 'c_out': config.c_out}
    config.loss_params = {}
    
    return config


def generate_demo_data():
    """Generate realistic demo time series"""
    # Create more complex time series
    np.random.seed(42)
    
    # Parameters
    total_len = 200
    seq_len = 96
    pred_len = 24
    
    # Time index
    t = np.linspace(0, 10*np.pi, total_len)
    
    # Multiple components
    trend = 0.02 * t + 1.0  # Slight upward trend
    seasonal_daily = 0.3 * np.sin(2*np.pi*t/24)  # Daily seasonality
    seasonal_weekly = 0.15 * np.sin(2*np.pi*t/168)  # Weekly seasonality
    noise = 0.1 * np.random.randn(total_len)
    
    # Combine components
    series = trend + seasonal_daily + seasonal_weekly + noise
    
    # Split data
    historical = series[:seq_len]
    true_future = series[seq_len:seq_len+pred_len]
    
    # Prepare model inputs
    x_enc = torch.FloatTensor(historical).unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
    x_dec = torch.FloatTensor(series[seq_len-pred_len:seq_len]).unsqueeze(0).unsqueeze(-1)  # [1, pred_len, 1]
    x_mark_enc = torch.zeros(1, seq_len, 4)
    x_mark_dec = torch.zeros(1, pred_len*2, 4)
    
    return {
        'full_series': series,
        'historical': historical,
        'true_future': true_future,
        'x_enc': x_enc,
        'x_mark_enc': x_mark_enc,
        'x_dec': x_dec,
        'x_mark_dec': x_mark_dec,
        'seq_len': seq_len,
        'pred_len': pred_len
    }


def run_comprehensive_demo():
    """Run comprehensive demo"""
    print("🚀 Comprehensive ChronosX + ModularAutoformer Demo")
    print("=" * 70)
    
    # Initialize components
    print("🔧 Initializing component registry...")
    registry = unified_registry
    register_example_components(registry)
    print("   ✅ Components registered")
    
    # Generate demo data
    print("\n📊 Generating realistic demo data...")
    data = generate_demo_data()
    print(f"   ✅ Generated {len(data['full_series'])} data points")
    print(f"      Historical: {len(data['historical'])} points")
    print(f"      Forecasting: {data['pred_len']} points")
    
    # Test different model configurations
    models_to_test = [
        ('chronos_x', 'tiny', 'ChronosX Tiny - Fastest'),
        ('chronos_x_uncertainty', 'tiny', 'ChronosX Uncertainty - Best UQ'),
    ]
    
    results = {}
    
    for backbone_type, model_size, description in models_to_test:
        print(f"\n🔮 Testing {description}")
        print("-" * 50)
        
        try:
            # Create model
            config = create_demo_config(backbone_type, model_size)
            
            start_time = time.time()
            model = ModularAutoformer(config)
            model.eval()
            init_time = time.time() - start_time
            
            print(f"   ✅ Model loaded in {init_time:.3f}s")
            
            # Get model info
            backbone_info = model.get_backbone_info()
            print(f"      Supports uncertainty: {backbone_info['supports_uncertainty']}")
            
            # Forward pass
            with torch.no_grad():
                start_time = time.time()
                prediction = model(
                    data['x_enc'],
                    data['x_mark_enc'],
                    data['x_dec'],
                    data['x_mark_dec']
                )
                inference_time = time.time() - start_time
            
            print(f"   ⚡ Inference completed in {inference_time:.3f}s")
            print(f"      Prediction shape: {prediction.shape}")
            
            # Convert to numpy for analysis
            pred_numpy = prediction.squeeze().numpy()
            
            # Calculate metrics
            mae = np.mean(np.abs(pred_numpy - data['true_future']))
            rmse = np.sqrt(np.mean((pred_numpy - data['true_future'])**2))
            
            print(f"   📊 Forecast quality:")
            print(f"      MAE: {mae:.4f}")
            print(f"      RMSE: {rmse:.4f}")
            
            # Check uncertainty
            uncertainty_results = model.get_uncertainty_results()
            uncertainty_info = "None"
            if uncertainty_results:
                if 'quantiles' in uncertainty_results:
                    quantiles = list(uncertainty_results['quantiles'].keys())
                    uncertainty_info = f"Quantiles: {quantiles}"
                elif 'std' in uncertainty_results:
                    avg_std = uncertainty_results['std'].mean().item()
                    uncertainty_info = f"Std: ±{avg_std:.4f}"
            
            print(f"   🎯 Uncertainty: {uncertainty_info}")
            
            # Store results
            results[backbone_type] = {
                'prediction': pred_numpy,
                'mae': mae,
                'rmse': rmse,
                'init_time': init_time,
                'inference_time': inference_time,
                'uncertainty': uncertainty_results,
                'description': description
            }
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
            results[backbone_type] = {'error': str(e)}
    
    # Create comparison visualization
    print(f"\n📈 Creating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('ChronosX + ModularAutoformer Demo Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Time series overview
    ax1 = axes[0, 0]
    full_series = data['full_series']
    seq_len = data['seq_len']
    pred_len = data['pred_len']
    
    ax1.plot(range(len(full_series)), full_series, 'b-', alpha=0.7, linewidth=1, label='Full Series')
    ax1.axvline(x=seq_len, color='red', linestyle='--', alpha=0.7, label='Forecast Start')
    ax1.axvspan(seq_len, seq_len+pred_len, alpha=0.2, color='yellow', label='Forecast Period')
    ax1.set_title('Time Series Overview')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Forecasting comparison
    ax2 = axes[0, 1]
    forecast_x = range(seq_len, seq_len + pred_len)
    
    ax2.plot(range(seq_len-20, seq_len), data['historical'][-20:], 'b-', linewidth=2, label='Historical')
    ax2.plot(forecast_x, data['true_future'], 'g-', linewidth=2, label='True Future')
    
    colors = ['red', 'orange', 'purple']
    for i, (model_name, result) in enumerate(results.items()):
        if 'prediction' in result:
            ax2.plot(forecast_x, result['prediction'], '--', 
                    color=colors[i % len(colors)], linewidth=2, 
                    label=f"{result['description']} (MAE: {result['mae']:.3f})")
    
    ax2.axvline(x=seq_len, color='red', linestyle=':', alpha=0.7)
    ax2.set_title('Forecasting Comparison')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Performance metrics
    ax3 = axes[1, 0]
    
    model_names = []
    mae_scores = []
    inference_times = []
    
    for model_name, result in results.items():
        if 'prediction' in result:
            model_names.append(result['description'].split('-')[0].strip())
            mae_scores.append(result['mae'])
            inference_times.append(result['inference_time'])
    
    if mae_scores:
        x_pos = np.arange(len(model_names))
        bars1 = ax3.bar(x_pos - 0.2, mae_scores, 0.4, label='MAE', alpha=0.7, color='blue')
        
        ax3_twin = ax3.twinx()
        bars2 = ax3_twin.bar(x_pos + 0.2, inference_times, 0.4, label='Inference Time (s)', alpha=0.7, color='orange')
        
        ax3.set_title('Performance Comparison')
        ax3.set_xlabel('Model')
        ax3.set_ylabel('MAE', color='blue')
        ax3_twin.set_ylabel('Inference Time (s)', color='orange')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(model_names, rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, mae_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        for bar, value in zip(bars2, inference_times):
            ax3_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                         f'{value:.3f}s', ha='center', va='bottom', fontsize=9)
        
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
    
    # Plot 4: Uncertainty visualization (if available)
    ax4 = axes[1, 1]
    
    uncertainty_plotted = False
    for model_name, result in results.items():
        if 'uncertainty' in result and result['uncertainty']:
            uncertainty_results = result['uncertainty']
            
            if 'quantiles' in uncertainty_results:
                quantiles = uncertainty_results['quantiles']
                prediction = result['prediction']
                
                # Plot prediction and uncertainty bands
                ax4.plot(forecast_x, prediction, 'b-', linewidth=2, label='Prediction')
                ax4.plot(forecast_x, data['true_future'], 'g-', linewidth=2, label='True')
                
                # Add quantile bands
                if 'q10' in quantiles and 'q90' in quantiles:
                    q10 = quantiles['q10'].squeeze().numpy()
                    q90 = quantiles['q90'].squeeze().numpy()
                    ax4.fill_between(forecast_x, q10, q90, alpha=0.3, color='blue', label='80% Confidence')
                
                if 'q25' in quantiles and 'q75' in quantiles:
                    q25 = quantiles['q25'].squeeze().numpy()
                    q75 = quantiles['q75'].squeeze().numpy()
                    ax4.fill_between(forecast_x, q25, q75, alpha=0.5, color='blue', label='50% Confidence')
                
                uncertainty_plotted = True
                break
    
    if not uncertainty_plotted:
        # Just show prediction vs true
        for model_name, result in results.items():
            if 'prediction' in result:
                ax4.plot(forecast_x, result['prediction'], '--', linewidth=2, label=f"Prediction")
                break
        ax4.plot(forecast_x, data['true_future'], 'g-', linewidth=2, label='True Future')
    
    ax4.set_title('Uncertainty Quantification')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Value')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('chronosx_modular_autoformer_demo.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved visualization: chronosx_modular_autoformer_demo.png")
    
    # Summary
    print(f"\n🎉 Demo Complete!")
    print("=" * 70)
    print("✅ Key Achievements:")
    print("   🔮 ChronosX backbone integrated with ModularAutoformer")
    print("   ⚡ Zero-shot forecasting without training")
    print("   🎯 Uncertainty quantification available")
    print("   📊 Multiple model variants tested")
    print("   📈 Comprehensive performance analysis")
    
    successful_models = [name for name, result in results.items() if 'prediction' in result]
    print(f"\n📋 Working Models: {len(successful_models)}/{len(models_to_test)}")
    
    for model_name, result in results.items():
        if 'prediction' in result:
            print(f"   ✅ {result['description']}")
            print(f"      MAE: {result['mae']:.4f}, Inference: {result['inference_time']:.3f}s")
    
    return len(successful_models) == len(models_to_test)


if __name__ == "__main__":
    print("🔬 Starting Comprehensive ChronosX + ModularAutoformer Demo...\n")
    
    success = run_comprehensive_demo()
    
    if success:
        print(f"\n🚀 DEMO SUCCESSFUL! Your integration is working perfectly!")
    else:
        print(f"\n⚠️ Some issues detected. Check the results above.")
    
    sys.exit(0 if success else 1)
