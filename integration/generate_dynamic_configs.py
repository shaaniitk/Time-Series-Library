import os
import sys
import yaml

# Add the project root to the Python path to find utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_analysis import analyze_dataset
from utils.logger import logger

# --- Comprehensive List of Forecasting Models ---
ALL_FORECASTING_MODELS = [
    'Autoformer',
    'EnhancedAutoformer',
    'BayesianEnhancedAutoformer',
    'HierarchicalEnhancedAutoformer',
]

# --- Complexity Variations ---
complexity_configs = {
    'ultralight': {
        'seq_len': 48, 'label_len': 24, 'pred_len': 12,
        'd_model': 64, 'n_heads': 4, 'e_layers': 1, 'd_layers': 1, 'd_ff': 128*2,
        'batch_size': 64, 'dropout': 0.05
    },
    'light': {
        'seq_len': 96, 'label_len': 48, 'pred_len': 24,
        'd_model': 128, 'n_heads': 8, 'e_layers': 2, 'd_layers': 1, 'd_ff': 256*2,
        'batch_size': 32, 'dropout': 0.1
    },
    'medium': {
        'seq_len': 192, 'label_len': 96, 'pred_len': 48,
        'd_model': 256, 'n_heads': 8, 'e_layers': 3, 'd_layers': 2, 'd_ff': 512*2,
        'batch_size': 16, 'dropout': 0.15
    },
    'heavy': {
        'seq_len': 336, 'label_len': 168, 'pred_len': 96,
        'd_model': 512, 'n_heads': 16, 'e_layers': 6, 'd_layers': 4, 'd_ff': 512*4,
        'batch_size': 32, 'dropout': 0.15
    },
    'veryheavy': {
        'seq_len': 512, 'label_len': 256, 'pred_len': 128,
        'd_model': 768, 'n_heads': 16, 'e_layers': 6, 'd_layers': 4, 'd_ff': 768*4,
        'batch_size': 4, 'dropout': 0.25
    }
}

# --- Helper function to get user input with validation ---
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

    # --- Model and Complexity Selection ---
    print("\n--- Model and Complexity Selection ---")
    print("Available complexities:")
    for comp_level in complexity_configs.keys():
        print(f"  - {comp_level}")
    while True:
        complexity = input("Enter desired complexity (e.g., ultralight, medium, heavy): ").strip().lower()
        if complexity in complexity_configs:
            break
        else:
            logger.warning("Invalid complexity. Please choose from the list.")

    print("\nAvailable models:")
    for i, model_name in enumerate(ALL_FORECASTING_MODELS):
        print(f"  {i+1}. {model_name}")
    while True:
        model_choice = input("Enter model number (e.g., 1 for Autoformer): ").strip()
        try:
            model_index = int(model_choice) - 1
            if 0 <= model_index < len(ALL_FORECASTING_MODELS):
                model_name = ALL_FORECASTING_MODELS[model_index]
                break
            else:
                logger.warning("Invalid model number.")
        except ValueError:
            logger.warning("Invalid input. Please enter a number.")

    print("\nAvailable feature modes (how data is handled):")
    print("  - M: Multivariate (All features -> All features)")
    print("  - MS: Multi-target (All features -> Target features only)")
    print("  - S: Univariate (Target features only -> Target features only)")
    while True:
        feature_mode = input("Enter feature mode (M, MS, S): ").strip().upper()
        if feature_mode in ['M', 'MS', 'S']:
            break
        else:
            logger.warning("Invalid feature mode. Please choose M, MS, or S.")

    # --- Sequence Lengths ---
    default_seq_len = complexity_configs[complexity]['seq_len']
    default_pred_len = complexity_configs[complexity]['pred_len']
    seq_len = int(input(f"Enter sequence length (default: {default_seq_len}): ") or default_seq_len)
    pred_len = int(input(f"Enter prediction length (default: {default_pred_len}): ") or default_pred_len)
    label_len = int(input(f"Enter label length (default: {seq_len // 2}): ") or (seq_len // 2))

    # --- Loss Function Settings ---
    print("\n--- Loss Function Settings ---")
    print("Available loss types (common examples: mse, mae, pinball, huber, ps_loss, multiscale_trend_aware):")
    loss_type = input("Enter loss type: ").strip().lower() or 'mse'
    
    quantile_levels = []
    if loss_type in ['pinball', 'quantile']:
        while True:
            q_input = input("Enter comma-separated quantile levels (e.g., 0.1,0.5,0.9) or leave empty for default: ").strip()
            if q_input:
                try:
                    parsed_quantiles = sorted([float(q.strip()) for q in q_input.split(',')])
                    if all(0 < q < 1 for q in parsed_quantiles):
                        quantile_levels = parsed_quantiles
                        break
                    else:
                        logger.warning("Quantile levels must be floats between 0 and 1.")
                except ValueError:
                    logger.warning("Invalid input. Please enter comma-separated numbers.")
            else:
                quantile_levels = [0.1, 0.5, 0.9]
                logger.info(f"Using default quantile levels: {quantile_levels}")
                break
    
    # --- KL Loss settings ---
    enable_kl = False
    kl_weight = 0.0
    if model_name == 'BayesianEnhancedAutoformer':
        print("\n--- KL Loss Settings (for Bayesian models) ---")
        kl_input = input("Enable KL Loss for Bayesian models? (y/n): ").strip().lower()
        if kl_input == 'y':
            enable_kl = True
            while True:
                try:
                    kl_weight = float(input("Enter KL Loss weight (e.g., 1e-5, 1e-4): ").strip())
                    break
                except ValueError:
                    logger.warning("Invalid input. Please enter a number.")

    # --- Hierarchical Model Specific Prompts ---
    n_levels = 3
    fusion_strategy = 'weighted_concat'
    use_moe_ffn = True
    num_experts = 4
    if model_name == 'HierarchicalEnhancedAutoformer':
        print("\n--- Hierarchical Model Settings ---")
        try:
            n_levels = int(input("Enter number of hierarchy levels (e.g., 1, 2, 3): ").strip() or 3)
        except ValueError:
            logger.warning("Invalid input. Defaulting to 3 levels.")
            n_levels = 3
        fusion_strategy = input("Enter fusion strategy (weighted_sum, weighted_concat, attention_fusion): ").strip().lower() or 'weighted_concat'
        if fusion_strategy not in ['weighted_sum', 'weighted_concat', 'attention_fusion']:
            logger.warning("Invalid strategy. Defaulting to 'weighted_concat'.")
            fusion_strategy = 'weighted_concat'
        # Prompt for MoE FFN usage
        use_moe_ffn_input = input("Enable MoE FFN in decoder? (y/n, default: y): ").strip().lower()
        if use_moe_ffn_input == 'n':
            use_moe_ffn = False
        else:
            use_moe_ffn = True
        if use_moe_ffn:
            num_experts_input = input("Number of MoE experts (default: 4): ").strip()
            if num_experts_input:
                try:
                    num_experts = int(num_experts_input)
                except ValueError:
                    logger.warning("Invalid input for num_experts. Using default (4).")
            # else keep default

    # --- GPU prompt ---
    use_gpu_input = input("\nUse GPU for training? (y/n, default: y): ").strip().lower()
    use_gpu = use_gpu_input != 'n'
            
    return data_path, target_columns, complexity, model_name, feature_mode, \
           seq_len, label_len, pred_len, loss_type, quantile_levels, enable_kl, kl_weight, \
           n_levels, fusion_strategy, use_gpu, use_moe_ffn, num_experts

def generate_configurations():
    """Generates a single configuration file based on user input."""
    data_path, target_columns, complexity, model_name, feature_mode, \
    seq_len, label_len, pred_len, loss_type, quantile_levels, enable_kl, kl_weight, \
    n_levels, fusion_strategy, use_gpu, use_moe_ffn, num_experts = get_user_input()

    # --- Base Template ---
    base_template = {
        'task_name': 'long_term_forecast',
        'is_training': 1,
        'checkpoints': './checkpoints/',
        'root_path': os.path.dirname(data_path),
        'data_path': os.path.basename(data_path),
        'seq_len': seq_len,
        'label_len': label_len,
        'pred_len': pred_len,
        'target': ','.join(target_columns),
        'features': '{mode}',
        'freq': 'h',
        'train_epochs': 10,
        'patience': 3,
        'learning_rate': 0.0001,
        'loss': loss_type,
        'lradj': 'type1',
        'use_amp': False,
        'use_gpu': use_gpu,
        'gpu': 0,
        'num_workers': 0,
    }


    # Apply complexity settings
    config = base_template.copy()
    config.update(complexity_configs[complexity])

    # Ensure d_model and n_heads are always visible and set in config
    config['d_model'] = complexity_configs[complexity]['d_model']
    config['n_heads'] = complexity_configs[complexity]['n_heads']

    # --- General d_model/n_heads check for all models ---
    while config['d_model'] % config['n_heads'] != 0:
        print(f"\n[ERROR] d_model ({config['d_model']}) must be divisible by n_heads ({config['n_heads']})!")
        try:
            new_n_heads = int(input(f"Enter a valid n_heads (divisor of {config['d_model']}): ").strip())
            if new_n_heads > 0 and config['d_model'] % new_n_heads == 0:
                config['n_heads'] = new_n_heads
            else:
                print(f"[ERROR] {new_n_heads} is not a valid divisor of {config['d_model']}.")
        except ValueError:
            print("[ERROR] Please enter a valid integer.")

    # Override sequence lengths with user input
    config['seq_len'] = seq_len
    config['label_len'] = label_len
    config['pred_len'] = pred_len

    # Set model and feature mode
    config['model'] = model_name
    config['features'] = feature_mode

    # Add optional settings
    if quantile_levels:
        config['quantile_levels'] = quantile_levels
    if enable_kl:
        config['kl_weight'] = kl_weight
    if model_name == 'HierarchicalEnhancedAutoformer':
        config['n_levels'] = n_levels
        config['fusion_strategy'] = fusion_strategy
        config['use_moe_ffn'] = use_moe_ffn
        config['num_experts'] = num_experts

    # --- Dynamic Dimension Calculation ---
    data_analysis = analyze_dataset(
        data_path=data_path,
        target_columns=target_columns,
    )
    
    mode_config_dims = data_analysis[f'mode_{feature_mode}']
    config['enc_in'] = mode_config_dims['enc_in']
    config['dec_in'] = mode_config_dims['dec_in']
    config['c_out'] = mode_config_dims['c_out']

    # Create a unique model_id
    config['model_id'] = f"{model_name}_{feature_mode}_{complexity}_sl{seq_len}_pl{pred_len}"

    # Add a metadata note for clarity
    config['_metadata'] = {
        'note': 'enc_in, dec_in, and c_out are determined automatically at runtime by DimensionManager.',
        'generated_by': 'integration/generate_dynamic_configs.py'
    }

    output_dir = "configs"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"config_{config['model_id']}.yaml"
    output_path = os.path.join(output_dir, filename)

    try:
        relative_output_path = os.path.relpath(output_path).replace('\\', '/')
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)
        logger.info(f"\n🎉 Configuration generated successfully: {output_path}")
        logger.info(f"\n🚀 To run this experiment, use:\n   python scripts/train/train_dynamic_autoformer.py --config {relative_output_path}")
    except Exception as e:
        logger.error(f"❌ Failed to create {output_path}: {e}")

if __name__ == "__main__":
    generate_configurations()
