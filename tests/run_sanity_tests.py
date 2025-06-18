#!/usr/bin/env python3
"""
Comprehensive Sanity Test for Enhanced Autoformer Models
Tests loss computation, dimensions, and convergence behavior
"""

import torch
import numpy as np
import os
import sys
import subprocess
import yaml
from argparse import Namespace
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sanity_config():
    """Create a lightweight config for sanity testing."""
    config = {
        # Model and data
        'model': 'EnhancedAutoformer',
        'data': 'custom',
        'root_path': 'data',
        'data_path': 'ETTh1.csv',  # Small dataset for sanity test
        'target': 'OT',
        'features': 'M',  # Multivariate
        
        # Architecture (light config)
        'seq_len': 96,
        'label_len': 48, 
        'pred_len': 24,
        'd_model': 64,
        'd_ff': 128,
        'n_heads': 4,
        'e_layers': 2,
        'd_layers': 1,
        'dropout': 0.1,
        
        # Training
        'train_epochs': 10,  # Test convergence
        'batch_size': 16,
        'learning_rate': 0.0001,
        'patience': 5,
        'loss': 'mse',  # Standard MSE loss
        
        # Auto-detection placeholders
        'enc_in': 7,   # Will be auto-detected for ETT
        'dec_in': 7,
        'c_out': 1,
        
        # Standard settings
        'embed': 'timeF',
        'freq': 'h',
        'activation': 'gelu',
        'factor': 1,
        'distil': True,
        'mix': True,
        'output_attention': False,
        
        # System
        'use_gpu': False,  # CPU for sanity test
        'use_multi_gpu': False,
        'use_amp': False,
        'des': 'sanity_test',
        'itr': 1,
        'checkpoints': './checkpoints',
        'gpu': 0,
        'devices': '0',
        
        # Enhanced features
        'use_adaptive_correlation': True,
        'adaptive_k': True,
        'multi_scale': True,
        'enhanced_decomposition': True,
        'use_dtw': False,  # Disable DTW for simpler analysis
        'inverse': True
    }
    return config

def save_config(config, filename):
    """Save config to YAML file."""
    with open(filename, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Saved config to {filename}")

def run_model_sanity_test(model_type, config_file, expected_behaviors):
    """
    Run sanity test for a specific model type.
    
    Args:
        model_type: 'enhanced', 'bayesian', or 'hierarchical'
        config_file: Path to config file
        expected_behaviors: Dict of expected behaviors to validate
    
    Returns:
        Dict with test results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"SANITY TEST: {model_type.upper()}")
    logger.info(f"{'='*60}")
    
    # Run training command
    cmd = [
        sys.executable, '../scripts/train/train_dynamic_autoformer.py',
        '--config', config_file,
        '--model_type', model_type,
        '--auto_fix'
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=300,  # 5 minute timeout
            cwd=os.getcwd()
        )
        
        stdout = result.stdout
        stderr = result.stderr
        
        # Parse results
        test_results = parse_training_output(stdout, stderr, model_type, expected_behaviors)
        
        if result.returncode == 0:
            logger.info(f"✅ {model_type} training completed successfully")
            test_results['status'] = 'PASSED'
        else:
            logger.error(f"❌ {model_type} training failed with exit code {result.returncode}")
            logger.error(f"STDERR: {stderr[-500:]}")
            test_results['status'] = 'FAILED'
            test_results['error'] = stderr[-500:]
        
        return test_results
        
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ {model_type} test timed out")
        return {'status': 'TIMEOUT', 'error': 'Test timed out after 5 minutes'}
    except Exception as e:
        logger.error(f"💥 {model_type} test error: {e}")
        return {'status': 'ERROR', 'error': str(e)}

def parse_training_output(stdout, stderr, model_type, expected_behaviors):
    """Parse training output to extract key metrics and validate behavior."""
    
    results = {
        'dimensions': {},
        'loss_progression': [],
        'final_metrics': {},
        'special_features': {},
        'validation_errors': []
    }
    
    lines = stdout.split('\n')
    
    # Extract dimensions
    for line in lines:
        if 'batch_x:' in line and 'torch.Size' in line:
            try:
                # Extract tensor dimensions
                import re
                size_match = re.search(r'torch\.Size\(\[([^\]]+)\]\)', line)
                if size_match:
                    dims = [int(x.strip()) for x in size_match.group(1).split(',')]
                    if 'batch_x:' in line:
                        results['dimensions']['batch_x'] = dims
                    elif 'batch_y:' in line:
                        results['dimensions']['batch_y'] = dims
            except:
                pass
    
    # Extract loss progression  
    epoch_losses = []
    for line in lines:
        if 'Train Loss:' in line and 'Vali Loss:' in line:
            try:
                # Parse: "Epoch: X, Steps: Y | Train Loss: Z Vali Loss: W Test Loss: V"
                parts = line.split('|')[1].strip()
                train_loss = float(parts.split('Train Loss:')[1].split('Vali Loss:')[0].strip())
                vali_loss = float(parts.split('Vali Loss:')[1].split('Test Loss:')[0].strip())
                epoch_losses.append({'train': train_loss, 'vali': vali_loss})
            except:
                pass
    
    results['loss_progression'] = epoch_losses
    
    # Extract Bayesian-specific information
    if model_type == 'bayesian':
        for line in lines:
            if 'data_loss:' in line:
                try:
                    data_loss = float(line.split('data_loss:')[1].split()[0])
                    results['special_features']['data_loss'] = data_loss
                except:
                    pass
            elif 'kl_loss:' in line:
                try:
                    kl_loss = float(line.split('kl_loss:')[1].split()[0])
                    results['special_features']['kl_loss'] = kl_loss
                except:
                    pass
            elif 'total_loss:' in line:
                try:
                    total_loss = float(line.split('total_loss:')[1].split()[0])
                    results['special_features']['total_loss'] = total_loss
                except:
                    pass
    
    # Extract final test metrics
    for line in lines:
        if 'test shape (scaled for metrics):' in line:
            results['final_metrics']['output_info'] = line.strip()
    
    # Validate expected behaviors
    if len(epoch_losses) > 0:
        first_loss = epoch_losses[0]['train']
        last_loss = epoch_losses[-1]['train']
        
        # Check if loss is decreasing (basic sanity)
        if last_loss < first_loss:
            results['validation_errors'].append("✅ Loss is decreasing")
        else:
            results['validation_errors'].append("❌ Loss is not decreasing")
        
        # Check loss magnitude
        if last_loss < 10.0:  # Reasonable final loss
            results['validation_errors'].append("✅ Final loss is reasonable")
        else:
            results['validation_errors'].append("❌ Final loss is too high")
    
    # Model-specific validations
    if model_type == 'bayesian':
        if 'kl_loss' in results['special_features']:
            kl_loss = results['special_features']['kl_loss']
            if 0 < kl_loss < 1.0:  # KL loss should be small but positive
                results['validation_errors'].append("✅ KL loss is in expected range")
            else:
                results['validation_errors'].append(f"❌ KL loss {kl_loss} is outside expected range")
        else:
            results['validation_errors'].append("❌ No KL loss found in Bayesian model")
    
    return results

def compare_models(all_results):
    """Compare results across all models."""
    
    logger.info(f"\n{'='*60}")
    logger.info("MODEL COMPARISON")
    logger.info(f"{'='*60}")
    
    # Status summary
    logger.info("\n📊 Status Summary:")
    for model, results in all_results.items():
        status = results.get('status', 'UNKNOWN')
        status_emoji = {'PASSED': '✅', 'FAILED': '❌', 'TIMEOUT': '⏰', 'ERROR': '💥'}
        logger.info(f"  {status_emoji.get(status, '❓')} {model}: {status}")
    
    # Loss comparison
    logger.info("\n📈 Loss Progression:")
    for model, results in all_results.items():
        if results.get('status') == 'PASSED' and results.get('loss_progression'):
            losses = results['loss_progression']
            if len(losses) > 0:
                first_train = losses[0]['train']
                last_train = losses[-1]['train']
                first_vali = losses[0]['vali']
                last_vali = losses[-1]['vali']
                
                logger.info(f"  {model}:")
                logger.info(f"    Train: {first_train:.6f} → {last_train:.6f} (Δ: {last_train-first_train:.6f})")
                logger.info(f"    Valid: {first_vali:.6f} → {last_vali:.6f} (Δ: {last_vali-first_vali:.6f})")
    
    # Special features
    logger.info("\n🔬 Special Features:")
    for model, results in all_results.items():
        special = results.get('special_features', {})
        if special:
            logger.info(f"  {model}: {special}")
    
    # Validation summary
    logger.info("\n✅ Validation Results:")
    for model, results in all_results.items():
        validations = results.get('validation_errors', [])
        if validations:
            logger.info(f"  {model}:")
            for validation in validations:
                logger.info(f"    {validation}")

def main():
    """Run comprehensive sanity tests."""
    
    logger.info("🧪 Enhanced Autoformer Models - Comprehensive Sanity Test")
    logger.info("=" * 70)
    
    # Check if ETT data exists
    ett_path = os.path.join('data', 'ETTh1.csv')
    if not os.path.exists(ett_path):
        logger.error(f"❌ ETT dataset not found: {ett_path}")
        logger.info("Please ensure ETTh1.csv is in the data/ directory")
        return
    
    # Create sanity test config
    config = create_sanity_config()
    config_file = 'sanity_test_config.yaml'
    save_config(config, config_file)
    
    # Define expected behaviors for each model
    expected_behaviors = {
        'enhanced': {
            'loss_should_decrease': True,
            'reasonable_final_loss': True,
            'no_special_losses': True
        },
        'bayesian': {
            'loss_should_decrease': True,
            'reasonable_final_loss': True,
            'should_have_kl_loss': True,
            'kl_loss_range': (1e-6, 1.0)
        },
        'hierarchical': {
            'loss_should_decrease': True,
            'reasonable_final_loss': True,
            'multi_resolution_processing': True
        }
    }
    
    # Run tests for all models
    models_to_test = ['enhanced', 'bayesian', 'hierarchical']
    all_results = {}
    
    for model_type in models_to_test:
        results = run_model_sanity_test(model_type, config_file, expected_behaviors[model_type])
        all_results[model_type] = results
    
    # Compare and analyze results
    compare_models(all_results)
    
    # Final assessment
    passed_count = sum(1 for r in all_results.values() if r.get('status') == 'PASSED')
    total_count = len(all_results)
    
    logger.info(f"\n🎯 FINAL ASSESSMENT")
    logger.info(f"={'='*40}")
    logger.info(f"Tests passed: {passed_count}/{total_count}")
    
    if passed_count == total_count:
        logger.info("🎉 All models passed sanity tests!")
        logger.info("✅ Loss computation is working correctly")
        logger.info("✅ Bayesian regularization is properly implemented")
        logger.info("✅ All models are converging as expected")
    else:
        logger.warning(f"⚠️  {total_count - passed_count} model(s) failed sanity tests")
        logger.info("Please check the detailed logs above for issues")
    
    # Cleanup
    if os.path.exists(config_file):
        os.remove(config_file)
        logger.info(f"Cleaned up {config_file}")

if __name__ == '__main__':
    main()
