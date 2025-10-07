#!/usr/bin/env python3
"""
Proper Enhanced SOTA PGAT training on financial data
Uses standard framework with proper target handling and scaling
"""

import os
import sys
import pandas as pd
import numpy as np
import yaml
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import warnings
warnings.filterwarnings('ignore')

def prepare_financial_data():
    """Prepare financial data with targets as first columns"""
    print("📊 Preparing financial data with proper column ordering...")
    
    # Load original data
    df = pd.read_csv("data/prepared_financial_data.csv")
    print(f"Original data shape: {df.shape}")
    
    # Define target columns (these should be FIRST in the dataset)
    target_cols = ['log_Open', 'log_High', 'log_Low', 'log_Close']
    
    # Get all other columns (excluding date and targets)
    other_cols = [col for col in df.columns if col not in ['date'] + target_cols]
    
    # Reorder columns: date, targets, then all other features
    new_column_order = ['date'] + target_cols + other_cols
    df_reordered = df[new_column_order]
    
    print(f"Reordered columns:")
    print(f"  Date: {new_column_order[0]}")
    print(f"  Targets (first 4): {new_column_order[1:5]}")
    print(f"  Features: {len(new_column_order[5:])} features")
    print(f"  Sample features: {new_column_order[5:10]}...")
    
    # Save reordered data
    output_path = "data/financial_data_reordered.csv"
    df_reordered.to_csv(output_path, index=False)
    print(f"Saved reordered data to: {output_path}")
    
    return output_path, len(target_cols), len(other_cols)

def main():
    """Main training function using standard experiment framework"""
    
    print("🚀 Enhanced SOTA PGAT - Financial Data Training (Proper Framework)")
    print("=" * 80)
    
    # Prepare data with correct column ordering
    data_path, num_targets, num_features = prepare_financial_data()
    
    # Load configuration
    config_path = "configs/enhanced_pgat_financial.yaml"
    print(f"\nLoading config: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Update config with reordered data path
    config_dict['data_path'] = 'financial_data_reordered.csv'
    config_dict['c_out'] = num_targets  # Ensure c_out matches target count
    config_dict['c_out_evaluation'] = num_targets
    config_dict['enc_in'] = num_targets + num_features  # Total features
    
    print(f"\n📊 UPDATED CONFIGURATION")
    print(f"Data: {config_dict['data_path']}")
    print(f"Total Features: {config_dict['enc_in']}")
    print(f"Target Features: {config_dict['c_out']} (first {num_targets} columns)")
    print(f"Sequence Length: {config_dict['seq_len']}")
    print(f"Prediction Length: {config_dict['pred_len']}")
    print(f"Model: {config_dict['model']}")
    print(f"Batch Size: {config_dict['batch_size']}")
    print(f"Learning Rate: {config_dict['learning_rate']}")
    print(f"Epochs: {config_dict['train_epochs']}")
    
    # Save updated config
    updated_config_path = "configs/enhanced_pgat_financial_proper.yaml"
    with open(updated_config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    print(f"Saved updated config to: {updated_config_path}")
    
    # Convert to argparse-like object
    class Args:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    args = Args(**config_dict)
    
    # Import and run using the standard framework
    try:
        # Try the new framework first
        from scripts.train.train_dynamic_autoformer import main as train_dynamic
        print(f"\n🏗️  Using dynamic training framework...")
        
        # Convert our args to the format expected by train_dynamic_autoformer
        sys.argv = [
            'train_dynamic_autoformer.py',
            '--model', args.model,
            '--data', args.data,
            '--data_path', args.data_path,
            '--seq_len', str(args.seq_len),
            '--pred_len', str(args.pred_len),
            '--enc_in', str(args.enc_in),
            '--c_out', str(args.c_out),
            '--d_model', str(args.d_model),
            '--n_heads', str(args.n_heads),
            '--batch_size', str(args.batch_size),
            '--learning_rate', str(args.learning_rate),
            '--train_epochs', str(args.train_epochs),
            '--patience', str(args.patience),
            '--target', args.target,
            '--features', args.features,
            '--freq', args.freq,
            '--checkpoints', args.checkpoints,
            '--is_training', str(args.is_training)
        ]
        
        result = train_dynamic()
        
    except ImportError:
        # Fallback to standard exp_long_term_forecasting
        print(f"\n🏗️  Using standard experiment framework...")
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        
        exp = Exp_Long_Term_Forecast(args)
        
        # Print model info
        total_params = sum(p.numel() for p in exp.model.parameters())
        trainable_params = sum(p.numel() for p in exp.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {total_params * 4 / 1e6:.1f} MB")
        
        # Training
        print(f"\n🎯 Starting training...")
        print("-" * 80)
        
        if args.is_training:
            # Train the model
            exp.train(args)
            
            # Test the model
            print(f"\n📊 Testing trained model...")
            exp.test(args)
        else:
            # Just test
            print(f"\n📊 Testing model...")
            exp.test(args)
        
        result = exp
    
    print(f"\n✅ Training completed successfully!")
    print(f"🎯 Model trained on financial data with proper target handling!")
    
    return result

if __name__ == "__main__":
    result = main()