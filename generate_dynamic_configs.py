#!/usr/bin/env python3
"""
Template-based Dynamic Configuration Generator

This script creates template configuration files that can automatically
adapt to any dataset dimensions using the data analysis utilities.
"""

import os
import yaml
from utils.data_analysis import analyze_dataset, generate_dynamic_config

def create_template_configs():
    """Create template configuration files for all complexity levels and modes"""
    
    # Base configuration template
    base_template = {
        # Model identification (will be updated dynamically)
        'model_id': "enhanced_autoformer_{mode}_{complexity}",
        'model': "EnhancedAutoformer",
        
        # Task configuration
        'task_name': "long_term_forecast",
        'features': "{mode}",  # Will be replaced with actual mode
        'target': "log_Close",  # Primary target (for S mode)
        
        # Data configuration
        'data': "custom",
        'root_path': "data",
        'data_path': "prepared_financial_data.csv",  # Default, can be overridden
        'freq': "b",  # Business day frequency
        
        # Model architecture (DYNAMIC - will be set from data analysis)
        'enc_in': None,  # Will be set dynamically
        'dec_in': None,  # Will be set dynamically  
        'c_out': None,   # Will be set dynamically
        
        # Enhanced features configuration
        'use_adaptive_correlation': True,
        'use_learnable_decomposition': True,
        'multi_scale_correlation': True,
        'seasonal_learning_rate': 0.001,
        
        # Training configuration
        'itr': 1,
        'train_epochs': 10,
        'batch_size': None,  # Will be set based on complexity
        'patience': 3,
        'learning_rate': 0.0001,
        'des': "exp",
        'loss': "MSE",
        'lradj': "type1",
        'use_amp': False,
        
        # Optimizer
        'use_multi_gpu': False,
        'devices': "0,1,2,3",
        'test_flop': False,
        'num_workers': 10,
        
        # Data processing
        'checkpoints': "./checkpoints/",
        'do_predict': False,
        'target': "log_Close",
        'inverse': False,
        'cols': None
    }
    
    # Complexity-specific parameters
    complexity_configs = {
        'ultralight': {
            'seq_len': 50,
            'label_len': 8,
            'pred_len': 5,
            'val_len': 5,
            'test_len': 5,
            'prod_len': 5,
            'd_model': 32,
            'n_heads': 4,
            'e_layers': 1,
            'd_layers': 1,
            'd_ff': 64,
            'moving_avg': 5,
            'dropout': 0.05,
            'factor': 1,
            'batch_size': 64,
            'description': "Minimal model for quick prototyping"
        },
        'light': {
            'seq_len': 100,
            'label_len': 10,
            'pred_len': 10,
            'val_len': 10,
            'test_len': 10,
            'prod_len': 10,
            'd_model': 64,
            'n_heads': 8,
            'e_layers': 2,
            'd_layers': 1,
            'd_ff': 128,
            'moving_avg': 15,
            'dropout': 0.1,
            'factor': 2,
            'batch_size': 48,
            'description': "Light model for development"
        },
        'medium': {
            'seq_len': 250,
            'label_len': 15,
            'pred_len': 10,
            'val_len': 10,
            'test_len': 10,
            'prod_len': 10,
            'd_model': 128,
            'n_heads': 8,
            'e_layers': 3,
            'd_layers': 2,
            'd_ff': 256,
            'moving_avg': 25,
            'dropout': 0.15,
            'factor': 3,
            'batch_size': 32,
            'description': "Balanced model for general use"
        },
        'heavy': {
            'seq_len': 400,
            'label_len': 20,
            'pred_len': 15,
            'val_len': 15,
            'test_len': 15,
            'prod_len': 15,
            'd_model': 256,
            'n_heads': 16,
            'e_layers': 4,
            'd_layers': 3,
            'd_ff': 512,
            'moving_avg': 35,
            'dropout': 0.2,
            'factor': 4,
            'batch_size': 16,
            'description': "Heavy model for high accuracy"
        },
        'veryheavy': {
            'seq_len': 500,
            'label_len': 25,
            'pred_len': 20,
            'val_len': 20,
            'test_len': 20,
            'prod_len': 20,
            'd_model': 512,
            'n_heads': 16,
            'e_layers': 6,
            'd_layers': 4,
            'd_ff': 1024,
            'moving_avg': 50,
            'dropout': 0.25,
            'factor': 5,
            'batch_size': 8,
            'description': "Maximum capacity model"
        }
    }
    
    modes = ['M', 'MS', 'S']
    complexities = ['ultralight', 'light', 'medium', 'heavy', 'veryheavy']
    
    created_files = []
    
    for mode in modes:
        for complexity in complexities:
            # Create config for this combination
            config = base_template.copy()
            config.update(complexity_configs[complexity])
            
            # Set mode-specific values
            config['model_id'] = f"enhanced_autoformer_{mode}_{complexity}"
            config['features'] = mode
            
            # Add placeholders for dynamic values (will be filled by data analysis)
            config['_dynamic_placeholders'] = {
                'enc_in': f"AUTO_DETECT_{mode}",
                'dec_in': f"AUTO_DETECT_{mode}",
                'c_out': f"AUTO_DETECT_{mode}",
                'mode_description': f"Will be set to actual dataset dimensions"
            }
            
            # Add mode documentation
            mode_docs = {
                'M': {
                    'description': "Multivariate forecasting - predicts all features",
                    'input': "All features from dataset",
                    'output': "All features",
                    'use_case': "Full ecosystem modeling"
                },
                'MS': {
                    'description': "Multi-target forecasting - uses all features to predict targets",
                    'input': "All features from dataset", 
                    'output': "Target features only",
                    'use_case': "Rich context prediction"
                },
                'S': {
                    'description': "Target-only forecasting - uses and predicts only targets",
                    'input': "Target features only",
                    'output': "Target features only", 
                    'use_case': "Pure target dynamics"
                }
            }
            
            config['_mode_info'] = mode_docs[mode]
            config['_complexity_info'] = {
                'level': complexity,
                'description': complexity_configs[complexity]['description']
            }
            
            # Save template config
            filename = f"template_enhanced_autoformer_{mode}_{complexity}.yaml"
            with open(filename, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            created_files.append(filename)
            print(f"✅ Created template: {filename}")
    
    print(f"\n🎯 Created {len(created_files)} template configuration files")
    return created_files

def generate_configs_for_dataset(data_path: str, target_columns: str = None):
    """Generate actual configs for a specific dataset"""
    
    print(f"🔍 Analyzing dataset: {data_path}")
    
    # Analyze the dataset
    from utils.data_analysis import analyze_dataset
    analysis = analyze_dataset(data_path, target_columns)
    
    print(f"📊 Dataset has {analysis['n_total_features']} features ({analysis['n_targets']} targets)")
    
    # Find all template files
    template_files = []
    for file in os.listdir('.'):
        if file.startswith('template_enhanced_autoformer_') and file.endswith('.yaml'):
            template_files.append(file)
    
    if not template_files:
        print("❌ No template files found. Creating templates first...")
        create_template_configs()
        # Reload template files
        template_files = []
        for file in os.listdir('.'):
            if file.startswith('template_enhanced_autoformer_') and file.endswith('.yaml'):
                template_files.append(file)
    
    print(f"📁 Found {len(template_files)} template files")
    
    generated_files = []
    
    for template_file in template_files:
        try:
            # Extract mode from filename
            parts = template_file.replace('.yaml', '').split('_')
            if len(parts) >= 5:
                mode = parts[3]  # M, MS, or S
                complexity = parts[4]  # ultralight, light, etc.
                
                if mode in ['M', 'MS', 'S']:
                    # Load template
                    with open(template_file, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    # Apply dynamic values
                    mode_config = analysis[f'mode_{mode}']
                    config['enc_in'] = mode_config['enc_in']
                    config['dec_in'] = mode_config['dec_in']
                    config['c_out'] = mode_config['c_out']
                    
                    # Update data path
                    config['data_path'] = os.path.basename(data_path)
                    
                    # Add analysis metadata
                    config['_data_analysis'] = {
                        'dataset_path': data_path,
                        'n_total_features': analysis['n_total_features'],
                        'n_targets': analysis['n_targets'],
                        'n_covariates': analysis['n_covariates'],
                        'target_columns': analysis['target_columns'],
                        'mode_dimensions': mode_config,
                        'generated_from_template': template_file
                    }
                    
                    # Remove placeholders
                    if '_dynamic_placeholders' in config:
                        del config['_dynamic_placeholders']
                    
                    # Generate output filename
                    output_file = f"config_enhanced_autoformer_{mode}_{complexity}_auto.yaml"
                    
                    # Save generated config
                    with open(output_file, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                    
                    generated_files.append(output_file)
                    print(f"✅ Generated: {output_file} ({mode_config['description']})")
        
        except Exception as e:
            print(f"❌ Failed to process {template_file}: {e}")
    
    print(f"\n🎯 Generated {len(generated_files)} dataset-specific configuration files")
    return generated_files

if __name__ == '__main__':
    import sys
    
    print("🎯 Dynamic Configuration Generator")
    print("=" * 50)
    
    if len(sys.argv) == 1:
        # No arguments - create templates
        print("Creating template configuration files...")
        create_template_configs()
        
    elif len(sys.argv) >= 2:
        # Dataset path provided - generate specific configs
        data_path = sys.argv[1]
        target_columns = sys.argv[2] if len(sys.argv) > 2 else None
        
        print(f"Generating configs for dataset: {data_path}")
        generate_configs_for_dataset(data_path, target_columns)
    
    print("\n✅ Configuration generation complete!")
