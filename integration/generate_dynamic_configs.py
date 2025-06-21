import os
import sys
import yaml

# Add the project root to the Python path to find utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_analysis import analyze_dataset
from utils.logger import logger

# --- Comprehensive List of Forecasting Models ---
# This list includes models generally suitable for long-term forecasting tasks
# and compatible with the DimensionManager's M/MS/S modes.
# Models for classification/anomaly detection or highly specialized inputs are excluded for clarity.
ALL_FORECASTING_MODELS = [
    'Autoformer',
    # 'Transformer',
    # 'DLinear',
    # 'FEDformer',
    # 'Informer',
    # 'TimesNet',
    # 'PatchTST',
    # 'Crossformer',
    # 'ETSformer',
    # 'Reformer',
    # 'Pyraformer',
    # 'MICN',
    # 'iTransformer',
    # 'Koopa',
    # 'TiDE',
    # 'FreTS',
    # 'TimeMixer',
    # 'TSMixer',
    # 'SegRNN',
    # 'TemporalFusionTransformer',
    # 'SCINet',
    # 'PAttn',
    # 'TimeXer',
    # 'WPMixer',
    # 'MultiPatchFormer',
    'EnhancedAutoformer',
    'BayesianEnhancedAutoformer',
    'HierarchicalEnhancedAutoformer',
]

def get_user_input():
    """Gets necessary input from the user to generate configs."""
    print("--- Dynamic Configuration Generator ---")
    
    # Get data path
    while True:
        data_path = input("Enter the path to your dataset CSV file: ").strip()
        if os.path.exists(data_path):
            break
        else:
            logger.error(f"File not found: {data_path}")
    
    # Get target columns
    while True:
        target_input = input("Enter comma-separated target column names (e.g., TARGET_1,TARGET_2): ").strip()
        if target_input:
            target_columns = [col.strip() for col in target_input.split(',')]
            break
        else:
            logger.warning("You must specify at least one target column.")
            
    return data_path, target_columns

def generate_configurations():
    """
    Main function to generate configuration files based on templates and user input.
    This version does NOT write low-level dimensions to the config files.
    """
    data_path, target_columns = get_user_input()

    # --- Base Template ---
    # This template contains only high-level settings.
    # `enc_in`, `dec_in`, `c_out` are intentionally omitted.
    base_template = {
        'task_name': 'long_term_forecast',
        'is_training': 1,
        'checkpoints': './checkpoints/',
        
        # --- Data Settings ---
        'root_path': os.path.dirname(data_path),
        'data_path': os.path.basename(data_path),
        'target': ','.join(target_columns),
        'features': '{mode}',  # Placeholder for M, MS, S
        'freq': 'h',           # Default, adjust in YAML if needed
        
        # --- Training Settings ---
        'train_epochs': 10,
        'patience': 3,
        'learning_rate': 0.0001,
        'loss': 'mse',
        'lradj': 'type1',
        'use_amp': False,
        
        # --- GPU Settings ---
        'use_gpu': True,
        'gpu': 0,
        'num_workers': 0,
    }

    # --- Complexity Variations ---
    # Each complexity level now defines the *base* parameters.
    # The script will iterate through ALL_FORECASTING_MODELS for each complexity.
    complexity_configs = {
        'ultralight': {
            'seq_len': 48, 'label_len': 24, 'pred_len': 12,
            'd_model': 64, 'n_heads': 4, 'e_layers': 1, 'd_layers': 1, 'd_ff': 128,
            'batch_size': 64, 'dropout': 0.05
        },
        'light': {
            # 'models': ['DLinear', 'EnhancedAutoformer'], # Removed: now iterates all models
            'seq_len': 96, 'label_len': 48, 'pred_len': 24,
            'd_model': 128, 'n_heads': 8, 'e_layers': 2, 'd_layers': 1, 'd_ff': 256,
            'batch_size': 32, 'dropout': 0.1
        },
        'medium': {
            # 'models': ['Autoformer', 'BayesianEnhancedAutoformer'], # Removed
            'seq_len': 192, 'label_len': 96, 'pred_len': 48,
            'd_model': 256, 'n_heads': 8, 'e_layers': 3, 'd_layers': 2, 'd_ff': 512,
            'batch_size': 16, 'dropout': 0.15
        },
        'heavy': {
            # 'models': ['Autoformer', 'HierarchicalEnhancedAutoformer'], # Removed
            'seq_len': 336, 'label_len': 168, 'pred_len': 96,
            'd_model': 512, 'n_heads': 16, 'e_layers': 4, 'd_layers': 3, 'd_ff': 1024,
            'batch_size': 8, 'dropout': 0.2
        },
        'veryheavy': {
            'seq_len': 512, 'label_len': 256, 'pred_len': 128,
            'd_model': 768, 'n_heads': 16, 'e_layers': 6, 'd_layers': 4, 'd_ff': 2048,
            'batch_size': 4, 'dropout': 0.25
        }
    }

    modes = ['M', 'MS', 'S']
    created_files = []
    output_dir = "configs"
    os.makedirs(output_dir, exist_ok=True) # Ensure the configs directory exists

    logger.info("--- Generating Configuration Files ---")

    for mode in modes:
        for complexity, settings in complexity_configs.items():
            # Iterate through ALL forecasting models for each complexity level
            for model_name in ALL_FORECASTING_MODELS:
                config = base_template.copy()
                config.update(settings) # Apply complexity settings
                
                # Set the model and feature mode
                config['model'] = model_name
                config['features'] = mode
                
                # Create a unique model_id
                config['model_id'] = f"{model_name}_{mode}_{complexity}"
                
                # Add a metadata note for clarity
                config['_metadata'] = {
                    'note': 'enc_in, dec_in, and c_out are determined automatically at runtime by DimensionManager.',
                    'generated_by': 'integration/generate_dynamic_configs.py'
                }

                filename = f"config_{config['model_id']}.yaml"
                output_path = os.path.join(output_dir, filename)
                
                try:
                    with open(output_path, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)
                    created_files.append(output_path)
                    logger.info(f"✅ Created: {output_path}")
                except Exception as e:
                    logger.error(f"❌ Failed to create {output_path}: {e}")

    logger.info(f"\n🎉 Generation complete! {len(created_files)} files created in '{output_dir}/'.")
    if created_files:
        example_config = os.path.relpath(created_files[0]).replace('\\', '/')
        logger.info(f"\n🚀 To run an experiment, use:\n   python scripts/train/train_dynamic_autoformer.py --config {example_config}")

if __name__ == "__main__":
    generate_configurations()
