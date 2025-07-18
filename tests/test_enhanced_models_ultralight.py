"""
Test script for all three Enhanced Autoformer variants with future covariates
Runs ultra-light configurations for fast testing and dimension verification
"""

import os
import sys
import torch
import yaml
import argparse
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from utils.logger import logger
from train_dynamic_autoformer import main as train_dynamic

def load_yaml_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def yaml_to_args(config_dict):
    """Convert YAML config dictionary to argparse.Namespace"""
    args = argparse.Namespace()
    for key, value in config_dict.items():
        setattr(args, key, value)
    
    # Set any missing required attributes with defaults
    if not hasattr(args, 'c_out'):
        args.c_out = 1  # Will be dynamically detected
    if not hasattr(args, 'enc_in'):
        args.enc_in = 1  # Will be dynamically detected
    if not hasattr(args, 'dec_in'):
        args.dec_in = 1  # Will be dynamically detected
    
    return args

def test_model_with_config(config_path, model_name):
    """Test a single model with its configuration"""
    logger.info(f"="*60)
    logger.info(f"TESTING {model_name.upper()}")
    logger.info(f"Config: {config_path}")
    logger.info(f"="*60)
    
    try:
        # Load configuration
        config = load_yaml_config(config_path)
        args = yaml_to_args(config)
        
        # Set data path relative to script location
        args.root_path = './data/'
        
        logger.info(f"Loaded configuration for {model_name}")
        logger.info(f"  Model: {args.model}")
        logger.info(f"  Data path: {args.data_path}")
        logger.info(f"  Target: {args.target}")
        logger.info(f"  Features: {args.features}")
        logger.info(f"  Sequence length: {args.seq_len}")
        logger.info(f"  Prediction length: {args.pred_len}")
        logger.info(f"  Training epochs: {args.train_epochs}")
        
        # Run training
        logger.info(f"Starting training for {model_name}...")
        train_dynamic(args)
        
        logger.info(f"PASS {model_name} completed successfully!")
        
    except Exception as e:
        logger.error(f"FAIL {model_name} failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """Run tests for all three enhanced models"""
    logger.info("ROCKET Starting Enhanced Autoformer variants testing with future covariates")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    # Model configurations
    models_to_test = [
        {
            'name': 'EnhancedAutoformer',
            'config': '../config/config_enhanced_autoformer_ultralight.yaml'
        },
        {
            'name': 'BayesianEnhancedAutoformer', 
            'config': '../config/config_bayesian_enhanced_autoformer_ultralight.yaml'
        },
        {
            'name': 'HierarchicalEnhancedAutoformer',
            'config': '../config/config_hierarchical_enhanced_autoformer_ultralight.yaml'
        }
    ]
    
    # Test each model
    results = {}
    for model_info in models_to_test:
        model_name = model_info['name']
        config_path = model_info['config']
        
        if not os.path.exists(config_path):
            logger.error(f"FAIL Config file not found: {config_path}")
            results[model_name] = "CONFIG_NOT_FOUND"
            continue
        
        try:
            test_model_with_config(config_path, model_name)
            results[model_name] = "SUCCESS"
        except Exception as e:
            logger.error(f"FAIL {model_name} failed: {str(e)}")
            results[model_name] = f"FAILED: {str(e)}"
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    for model_name, result in results.items():
        status_emoji = "PASS" if result == "SUCCESS" else "FAIL"
        logger.info(f"{status_emoji} {model_name}: {result}")
    
    # Check data files
    logger.info("\n" + "="*60)
    logger.info("DATA FILES CHECK")
    logger.info("="*60)
    
    data_path = './data/prepared_financial_data.csv'
    if os.path.exists(data_path):
        import pandas as pd
        df = pd.read_csv(data_path)
        logger.info(f"PASS Data file found: {data_path}")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Columns: {len(df.columns)}")
        logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Count targets vs covariates
        target_cols = [col for col in df.columns if col.startswith('log_')]
        covariate_cols = [col for col in df.columns if col not in ['date'] + target_cols]
        
        logger.info(f"  Target columns: {len(target_cols)} - {target_cols}")
        logger.info(f"  Covariate columns: {len(covariate_cols)}")
        logger.info(f"  Sample covariates: {covariate_cols[:10]}...")
    else:
        logger.error(f"FAIL Data file not found: {data_path}")
    
    logger.info("\n All tests completed!")

if __name__ == "__main__":
    main()
