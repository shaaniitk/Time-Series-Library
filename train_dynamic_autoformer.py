#!/usr/bin/env python3
"""
Enhanced Autoformer Dynamic Training Script

This script can train any Enhanced Autoformer variant with any dataset
by automatically detecting and adapting to the data dimensions.
"""

import argparse
import os
import torch
import numpy as np
import random
import yaml
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.data_analysis import analyze_dataset, validate_config_with_data

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_and_validate_config(config_path, data_path=None):
    """Load config and optionally validate against data"""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"📋 Loaded config: {config_path}")
    print(f"   Model: {config.get('model', 'Unknown')}")
    print(f"   Mode: {config.get('features', 'Unknown')}")
    print(f"   Architecture: enc_in={config.get('enc_in')}, dec_in={config.get('dec_in')}, c_out={config.get('c_out')}")
    
    # Validate against data if data path is provided
    if data_path and os.path.exists(data_path):
        print(f"\n🔍 Validating config against data...")
        validation = validate_config_with_data(config_path, data_path)
        
        if validation['valid']:
            print(f"✅ Config validation passed")
        else:
            print(f"❌ Config validation failed:")
            for issue in validation['issues']:
                print(f"   - {issue['description']}")
            
            # Ask user if they want to continue anyway
            response = input("\nContinue training anyway? (y/n): ").strip().lower()
            if response not in ['y', 'yes']:
                print("Training cancelled.")
                return None
    
    return config

def create_args_from_config(config, model_type='enhanced'):
    """Convert YAML config to argparse Namespace"""
    
    class Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    # Basic args
    args = Args(
        # Model configuration
        model=config.get('model', 'EnhancedAutoformer'),
        model_id=config.get('model_id', 'enhanced_autoformer'),
        
        # Data configuration
        data=config.get('data', 'custom'),
        root_path=config.get('root_path', 'data'),
        data_path=config.get('data_path', 'prepared_financial_data.csv'),
        features=config.get('features', 'MS'),
        target=config.get('target', 'log_Close'),
        freq=config.get('freq', 'b'),
        
        # Sequence configuration
        seq_len=config.get('seq_len', 250),
        label_len=config.get('label_len', 15),
        pred_len=config.get('pred_len', 10),
        
        # Model architecture
        enc_in=config.get('enc_in', 118),
        dec_in=config.get('dec_in', 4),
        c_out=config.get('c_out', 4),
        d_model=config.get('d_model', 128),
        n_heads=config.get('n_heads', 8),
        e_layers=config.get('e_layers', 3),
        d_layers=config.get('d_layers', 2),
        d_ff=config.get('d_ff', 256),
        moving_avg=config.get('moving_avg', 25),
        factor=config.get('factor', 3),
        dropout=config.get('dropout', 0.15),
        
        # Enhanced features
        use_adaptive_correlation=config.get('use_adaptive_correlation', True),
        use_learnable_decomposition=config.get('use_learnable_decomposition', True),
        multi_scale_correlation=config.get('multi_scale_correlation', True),
        seasonal_learning_rate=config.get('seasonal_learning_rate', 0.001),
        
        # Training configuration
        train_epochs=config.get('train_epochs', 10),
        batch_size=config.get('batch_size', 32),
        learning_rate=config.get('learning_rate', 0.0001),
        loss=config.get('loss', 'MSE'),
        lradj=config.get('lradj', 'type1'),
        patience=config.get('patience', 3),
        
        # Other settings
        itr=config.get('itr', 1),
        des=config.get('des', 'exp'),
        use_amp=False,  # Disable AMP for CPU
        devices='cpu',  # Force CPU device
        test_flop=config.get('test_flop', False),
        num_workers=0,  # Disable multiprocessing to avoid pickle issues
        checkpoints=config.get('checkpoints', './checkpoints/'),
        do_predict=config.get('do_predict', False),
        inverse=config.get('inverse', False),
        cols=config.get('cols', None),
        
        # Model type (for determining which model to use)
        enhanced_model_type=model_type,
        
        # Additional args that might be needed
        embed='timeF',
        activation='gelu',
        output_attention=False,
        distil=True,
        mix=True,
        augmentation_ratio=0,
        validation_length=150,
        test_length=50,
        
        # GPU and device settings - Force CPU for compatibility
        use_gpu=False,  # Set to False for CPU-only training
        gpu=0,
        gpu_type='cpu',
        device_ids=[0],
        
        # Additional training args
        is_training=1,
        model_type=model_type,
        task_name=config.get('task_name', 'long_term_forecast'),
        seasonal_patterns=None,
        use_dtw=config.get('use_dtw', False)  # DTW metric calculation flag
    )
    
    return args

def get_model_class(model_type):
    """Get the appropriate model class based on model type"""
    
    if model_type == 'enhanced':
        return 'EnhancedAutoformer'
    elif model_type == 'bayesian':
        return 'BayesianEnhancedAutoformer'
    elif model_type == 'hierarchical':
        return 'HierarchicalEnhancedAutoformer'
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def main():
    parser = argparse.ArgumentParser(description='Dynamic Enhanced Autoformer Training')
    
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to YAML configuration file')
    parser.add_argument('--model_type', type=str, default='enhanced',
                        choices=['enhanced', 'bayesian', 'hierarchical'],
                        help='Type of Enhanced Autoformer to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--validate_data', action='store_true',
                        help='Validate config against actual data dimensions')
    parser.add_argument('--auto_fix', action='store_true',
                        help='Automatically fix dimension mismatches')
    
    args = parser.parse_args()
    
    print("🚀 Dynamic Enhanced Autoformer Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Model Type: {args.model_type}")
    print(f"Seed: {args.seed}")
    
    # Set random seed
    set_seed(args.seed)
    
    # Load and validate configuration
    config = load_and_validate_config(args.config)
    if config is None:
        return
    
    # Convert config to args
    exp_args = create_args_from_config(config, args.model_type)
    
    # Override GPU settings regardless of config
    exp_args.use_gpu = False
    exp_args.use_multi_gpu = False
    exp_args.gpu_type = 'cpu'
    exp_args.devices = 'cpu'
    
    # Override model based on model_type
    exp_args.model = get_model_class(args.model_type)
    
    # Data path for validation
    data_path = os.path.join(exp_args.root_path, exp_args.data_path)
    
    # Auto-fix dimensions if requested
    if args.auto_fix and os.path.exists(data_path):
        print(f"\n🔧 Auto-fixing configuration dimensions...")
        
        try:
            analysis = analyze_dataset(data_path)
            mode = exp_args.features
            mode_config = analysis[f'mode_{mode}']
            
            # Update dimensions
            exp_args.enc_in = mode_config['enc_in']
            exp_args.dec_in = mode_config['dec_in']
            exp_args.c_out = mode_config['c_out']
            
            print(f"✅ Updated dimensions: enc_in={exp_args.enc_in}, dec_in={exp_args.dec_in}, c_out={exp_args.c_out}")
            
        except Exception as e:
            print(f"❌ Auto-fix failed: {e}")
    
    # Display final configuration
    print(f"\n📊 Final Training Configuration:")
    print(f"   Model: {exp_args.model} ({args.model_type})")
    print(f"   Data: {exp_args.data_path}")
    print(f"   Mode: {exp_args.features}")
    print(f"   Architecture: {exp_args.enc_in} → {exp_args.c_out}")
    print(f"   Sequence: {exp_args.seq_len} → {exp_args.pred_len}")
    print(f"   Model dim: {exp_args.d_model}, Layers: {exp_args.e_layers}+{exp_args.d_layers}")
    print(f"   Batch size: {exp_args.batch_size}, Epochs: {exp_args.train_epochs}")
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return
    
    # Create experiment
    print(f"\n🏗️ Creating experiment...")
    
    try:
        exp = Exp_Long_Term_Forecast(exp_args)
        
        # Training
        print(f"\n🎯 Starting training...")
        print(f"   Using device: {exp.device}")
        print(f"   Model parameters: {sum(p.numel() for p in exp.model.parameters()):,}")
        
        # Train the model
        setting = f"{exp_args.model_id}_{exp_args.features}_{exp_args.model}_{exp_args.seq_len}_{exp_args.pred_len}"
        exp.train(setting)
        
        # Test the model
        print(f"\n📈 Testing model...")
        exp.test(setting)
        
        print(f"\n✅ Training completed successfully!")
        print(f"   Checkpoints saved to: {exp_args.checkpoints}")
        
        # Display final results
        checkpoint_path = os.path.join(exp_args.checkpoints, exp_args.model_id)
        if os.path.exists(checkpoint_path):
            print(f"   Model saved at: {checkpoint_path}")
    
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n🎉 Dynamic Enhanced Autoformer training completed!")

if __name__ == '__main__':
    main()
