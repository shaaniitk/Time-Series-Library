#!/usr/bin/env python3
"""
Sanity Test for Enhanced Autoformer Models

Tests all three enhanced models with synthetic data to verify correct implementation.
Uses light configuration and 10 epochs to check validation error convergence.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import yaml
import argparse
from typing import Dict, List, Tuple

# Add project root to path
sys.path.append('/Users/shantanumisra/workspace/Time-Series-Library')

from models.EnhancedAutoformer import EnhancedAutoformer
from models.BayesianEnhancedAutoformer import BayesianEnhancedAutoformer
from models.HierarchicalEnhancedAutoformer import HierarchicalEnhancedAutoformer
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.logger import logger


def generate_synthetic_data(n_points=2000, freq='h'):
    """
    Generate synthetic time series data with known patterns for sanity testing.
    
    Returns:
        DataFrame with date, targets, and covariates
    """
    logger.info(f"Generating synthetic data with {n_points} points")
    
    # Create time index
    dates = pd.date_range('2020-01-01', periods=n_points, freq=freq)
    
    # Create angles for different frequencies
    X = np.arange(n_points) * 2 * np.pi / 100  # Main frequency
    X1 = np.arange(n_points) * 2 * np.pi / 50  # Higher frequency  
    X2 = np.arange(n_points) * 2 * np.pi / 200 # Lower frequency
    
    # Add noise
    noise = np.random.normal(0, 0.1, n_points)
    noise1 = np.random.normal(0, 0.05, n_points)
    noise2 = np.random.normal(0, 0.08, n_points)
    
    # Create covariates (these will be "future" covariates)
    cov1 = np.sin(X) + 0.3 * np.cos(X1) + noise1
    cov2 = np.cos(X1) + 0.2 * np.sin(X2) + noise2  
    cov3 = np.sin(X2) + 0.1 * np.cos(X) + noise
    
    # Create targets that depend on covariates (so models can learn the relationship)
    target1 = 0.7 * cov1 + 0.3 * np.sin(X) + noise
    target2 = 0.6 * cov2 + 0.4 * np.cos(X1) + noise1
    target3 = 0.8 * cov3 + 0.2 * np.sin(X2) + noise2
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'target1': target1,
        'target2': target2, 
        'target3': target3,
        'cov1': cov1,
        'cov2': cov2,
        'cov3': cov3
    })
    
    logger.info(f"Synthetic data shape: {df.shape}")
    logger.info(f"Targets: target1, target2, target3")
    logger.info(f"Covariates: cov1, cov2, cov3")
    
    return df


def create_light_config(model_type='enhanced'):
    """Create light configuration for fast testing."""
    
    config = {
        'model': 'EnhancedAutoformer',
        'data': 'custom',
        'data_path': 'synthetic_sanity_data.csv',
        'target': 'target1,target2,target3',
        'features': 'MS',  # Multi-variate forecasting
        
        # Sequence configuration (light)
        'seq_len': 48,
        'label_len': 24,  
        'pred_len': 12,
        
        # Model architecture (light)
        'd_model': 64,
        'd_ff': 128,
        'n_heads': 4,
        'e_layers': 2,
        'd_layers': 1,
        'dropout': 0.1,
        
        # Training configuration
        'train_epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.0001,
        'patience': 5,
        
        # Enhanced features
        'use_adaptive_correlation': True,
        'learnable_decomposition': True,
        'curriculum_learning': False,  # Disabled for sanity test
        
        # Other settings
        'embed': 'timeF',
        'freq': 'h',
        'activation': 'gelu',
        'factor': 1,
        'moving_avg': 25,
        'use_dtw': False,  # Disabled for speed
        'inverse': True,
        
        # Auto-detection placeholders
        'enc_in': 6,
        'dec_in': 3,
        'c_out': 3,
        
        # Infrastructure
        'use_gpu': False,
        'use_multi_gpu': False,
        'checkpoints': './sanity_checkpoints/',
        'des': 'sanity_test'
    }
    
    # Model-specific adjustments
    if model_type == 'bayesian':
        config.update({
            'n_mc_samples': 10,
            'kl_weight': 0.01,
            'uncertainty_method': 'bayesian'
        })
    elif model_type == 'hierarchical':
        config.update({
            'n_levels': 3,
            'wavelet_type': 'db4',
            'use_multiwavelet': True
        })
    
    return config


def run_single_model_test(model_type: str, config: dict) -> dict:
    """
    Run sanity test for a single model type.
    
    Returns:
        Dictionary with validation metrics over epochs
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {model_type.upper()} model")
    logger.info(f"{'='*60}")
    
    # Create config file
    config_file = f'sanity_config_{model_type}.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    # Convert config to args object
    class Args:
        pass
    
    args = Args()
    for key, value in config.items():
        setattr(args, key, value)
    
    # Map model type to class name
    model_map = {
        'enhanced': 'EnhancedAutoformer',
        'bayesian': 'BayesianEnhancedAutoformer', 
        'hierarchical': 'HierarchicalEnhancedAutoformer'
    }
    args.model = model_map[model_type]
    
    try:
        # Create experiment
        exp = Exp_Long_Term_Forecast(args)
        
        # Get model info
        total_params = sum(p.numel() for p in exp.model.parameters())
        trainable_params = sum(p.numel() for p in exp.model.parameters() if p.requires_grad)
        
        logger.info(f"Model: {args.model}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Training with validation tracking
        train_data, train_loader = exp._get_data(flag='train')
        vali_data, vali_loader = exp._get_data(flag='val')
        
        logger.info(f"Training data: {len(train_data)} samples")
        logger.info(f"Validation data: {len(vali_data)} samples")
        
        # Manual training loop to track validation errors
        model_optim = exp._select_optimizer()
        criterion = exp._select_criterion()
        
        val_losses = []
        train_losses = []
        
        for epoch in range(args.train_epochs):
            # Training
            exp.model.train()
            train_loss = 0.0
            train_steps = 0
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()
                
                # Forward pass
                batch_x = batch_x.float()
                batch_y = batch_y.float() 
                batch_x_mark = batch_x_mark.float()
                batch_y_mark = batch_y_mark.float()
                
                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float()
                
                # Model prediction
                if args.model == 'BayesianEnhancedAutoformer':
                    outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_uncertainty=False)
                else:
                    outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # Extract prediction part and compute loss
                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:]
                
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                model_optim.step()
                
                train_loss += loss.item()
                train_steps += 1
                
                if train_steps >= 50:  # Limit steps for sanity test
                    break
            
            # Validation
            val_loss = exp.vali(vali_data, vali_loader, criterion)
            
            avg_train_loss = train_loss / train_steps
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            
            logger.info(f"Epoch {epoch+1:2d}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Calculate metrics
        final_val_loss = val_losses[-1]
        min_val_loss = min(val_losses)
        val_improvement = val_losses[0] - min_val_loss
        convergence_epoch = val_losses.index(min_val_loss) + 1
        
        results = {
            'model_type': model_type,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'val_losses': val_losses,
            'train_losses': train_losses,
            'final_val_loss': final_val_loss,
            'min_val_loss': min_val_loss,
            'val_improvement': val_improvement,
            'convergence_epoch': convergence_epoch,
            'status': 'SUCCESS'
        }
        
        logger.info(f"✅ {model_type.upper()} test completed successfully")
        logger.info(f"   Final validation loss: {final_val_loss:.6f}")
        logger.info(f"   Best validation loss: {min_val_loss:.6f}")
        logger.info(f"   Improvement: {val_improvement:.6f}")
        logger.info(f"   Best epoch: {convergence_epoch}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ {model_type.upper()} test failed: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'model_type': model_type,
            'status': 'FAILED',
            'error': str(e)
        }
    
    finally:
        # Cleanup
        if os.path.exists(config_file):
            os.remove(config_file)


def plot_results(results: List[dict]):
    """Plot validation curves for comparison."""
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Validation Loss Curves
    plt.subplot(1, 3, 1)
    for result in results:
        if result['status'] == 'SUCCESS':
            epochs = range(1, len(result['val_losses']) + 1)
            plt.plot(epochs, result['val_losses'], 
                    label=f"{result['model_type'].title()}", 
                    marker='o', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Training vs Validation
    plt.subplot(1, 3, 2)
    for result in results:
        if result['status'] == 'SUCCESS':
            epochs = range(1, len(result['train_losses']) + 1)
            plt.plot(epochs, result['train_losses'], 
                    '--', alpha=0.7, label=f"{result['model_type'].title()} (Train)")
            plt.plot(epochs, result['val_losses'], 
                    '-', linewidth=2, label=f"{result['model_type'].title()} (Val)")
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Model Comparison
    plt.subplot(1, 3, 3)
    models = []
    final_losses = []
    min_losses = []
    
    for result in results:
        if result['status'] == 'SUCCESS':
            models.append(result['model_type'].title())
            final_losses.append(result['final_val_loss'])
            min_losses.append(result['min_val_loss'])
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, final_losses, width, label='Final Val Loss', alpha=0.8)
    plt.bar(x + width/2, min_losses, width, label='Best Val Loss', alpha=0.8)
    
    plt.xlabel('Model')
    plt.ylabel('Validation Loss')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sanity_test_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info("📊 Results plot saved as 'sanity_test_results.png'")


def main():
    """Run sanity tests for all three enhanced models."""
    
    logger.info("🧪 Enhanced Autoformer Models Sanity Test")
    logger.info("=" * 60)
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_data(n_points=1500, freq='h')
    data_file = 'synthetic_sanity_data.csv'
    synthetic_data.to_csv(data_file, index=False)
    logger.info(f"💾 Synthetic data saved to {data_file}")
    
    # Test all three models
    models_to_test = ['enhanced', 'bayesian', 'hierarchical']
    results = []
    
    for model_type in models_to_test:
        config = create_light_config(model_type)
        result = run_single_model_test(model_type, config)
        results.append(result)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SANITY TEST SUMMARY")
    logger.info("="*60)
    
    success_count = 0
    for result in results:
        if result['status'] == 'SUCCESS':
            success_count += 1
            logger.info(f"✅ {result['model_type'].upper()}:")
            logger.info(f"   Parameters: {result['trainable_params']:,}")
            logger.info(f"   Final Val Loss: {result['final_val_loss']:.6f}")
            logger.info(f"   Best Val Loss: {result['min_val_loss']:.6f}")
            logger.info(f"   Convergence: Epoch {result['convergence_epoch']}")
        else:
            logger.info(f"❌ {result['model_type'].upper()}: FAILED")
            logger.info(f"   Error: {result.get('error', 'Unknown error')}")
    
    logger.info(f"\nOverall: {success_count}/{len(models_to_test)} models passed sanity test")
    
    # Plot results if we have successful runs
    successful_results = [r for r in results if r['status'] == 'SUCCESS']
    if successful_results:
        plot_results(successful_results)
    
    # Cleanup
    if os.path.exists(data_file):
        os.remove(data_file)
    
    if success_count == len(models_to_test):
        logger.info("\n🎉 All models passed sanity tests! Ready for production use.")
    else:
        logger.info(f"\n⚠️  {len(models_to_test) - success_count} model(s) failed. Check implementation.")
    
    return results


if __name__ == '__main__':
    results = main()
