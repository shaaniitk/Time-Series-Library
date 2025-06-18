#!/usr/bin/env python3
"""
Enhanced Autoformer Dynamic Training Script with Quantile + KL Loss Support

This script can train any Enhanced Autoformer variant with any dataset
by automatically detecting and adapting to the data dimensions.

Features:
- Dynamic dimension detection and configuration
- Support for all model types (Enhanced, Bayesian, Hierarchical)
- Quantile regression with automatic quantile level generation
- KL loss tuning for Bayesian models
- Uncertainty quantification and prediction intervals
"""

import argparse
import os
import sys
import torch
import numpy as np
import random
import yaml
import warnings

# Add the root directory to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.data_analysis import analyze_dataset, validate_config_with_data
from utils.quantile_utils import get_standard_quantile_levels, quantile_levels_to_string, describe_quantiles
from utils.kl_tuning import KLTuner, suggest_kl_weight
from utils.dimension_manager import smart_dimension_setup

warnings.filterwarnings('ignore')

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

def create_args_from_config(config, model_type='enhanced', quantile_levels=None, kl_weight=None, kl_anneal=False):
    """Convert YAML config to argparse Namespace with quantile and KL support"""
    
    class Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    # Determine if quantile mode is enabled
    quantile_mode = quantile_levels is not None
    
    # Set loss function based on quantile mode
    loss_function = config.get('loss', 'MSE')
    if quantile_mode:
        loss_function = 'quantile'
    
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
        loss=loss_function,
        lradj=config.get('lradj', 'type1'),
        patience=config.get('patience', 3),
        
        # Quantile regression settings
        quantile_mode=quantile_mode,
        quantile_levels=quantile_levels if quantile_levels else [],
        
        # KL divergence settings (for Bayesian models)
        kl_weight=kl_weight if kl_weight is not None else 0.0,
        kl_anneal=kl_anneal,
        bayesian_layers=(model_type == 'bayesian'),
        
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
    parser = argparse.ArgumentParser(description='Dynamic Enhanced Autoformer Training with Quantile + KL Support')
    
    # Basic arguments
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
    
    # Quantile regression arguments
    parser.add_argument('--quantile_mode', action='store_true',
                        help='Enable quantile regression for uncertainty quantification')
    parser.add_argument('--num_quantiles', type=int, default=3,
                        choices=[1, 3, 5, 7, 9],
                        help='Number of quantiles (must be odd, includes median)')
    parser.add_argument('--quantile_levels', type=str, default=None,
                        help='Custom quantile levels as comma-separated values (e.g., "0.1,0.5,0.9")')
    parser.add_argument('--quantile_coverage', type=str, default='standard',
                        choices=['conservative', 'standard', 'wide'],
                        help='Coverage range for quantile generation (conservative: 0.1-0.9, standard: auto, wide: 0.05-0.95)')
    
    # Convergence test arguments
    parser.add_argument('--synthetic_data', action='store_true',
                        help='Use synthetic data for model convergence testing')
    parser.add_argument('--convergence_test', action='store_true',
                        help='Alias for --synthetic_data (convergence test mode)')
    parser.add_argument('--synthetic_type', type=str, default='sincos',
                        choices=['sincos', 'complex'],
                        help='Type of synthetic data to generate')
    parser.add_argument('--synthetic_n_points', type=int, default=2000,
                        help='Number of data points for synthetic data')
    parser.add_argument('--synthetic_noise', type=float, default=0.1,
                        help='Noise level for synthetic data (0.0 to 1.0)')
    parser.add_argument('--synthetic_complexity', type=str, default='medium',
                        choices=['simple', 'medium', 'complex'],
                        help='Complexity level for complex synthetic data')
    
    # KL loss arguments (for Bayesian models)
    parser.add_argument('--enable_kl', action='store_true',
                        help='Enable KL divergence loss (automatically enabled for Bayesian models in quantile mode)')
    parser.add_argument('--kl_weight', type=float, default=None,
                        help='KL loss weight (auto-suggested if not provided)')
    parser.add_argument('--kl_anneal', action='store_true',
                        help='Enable KL weight annealing during training')
    parser.add_argument('--kl_target_percentage', type=float, default=10.0,
                        help='Target KL loss percentage of total loss for auto-tuning')
    
    args = parser.parse_args()
    
    print("🚀 Dynamic Enhanced Autoformer Training with Quantile + KL Support")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Model Type: {args.model_type}")
    print(f"Seed: {args.seed}")
    
    # Handle convergence test mode
    convergence_test_mode = args.synthetic_data or args.convergence_test
    if convergence_test_mode:
        print(f"🔬 Convergence test mode enabled:")
        print(f"   Synthetic data type: {args.synthetic_type}")
        print(f"   Data points: {args.synthetic_n_points}")
        print(f"   Noise level: {args.synthetic_noise}")
        if args.synthetic_type == 'complex':
            print(f"   Complexity: {args.synthetic_complexity}")
    
    # Process quantile arguments
    quantile_levels = None
    if args.quantile_mode or args.quantile_levels:
        if args.quantile_levels:
            # Parse custom quantile levels
            quantile_levels = [float(q.strip()) for q in args.quantile_levels.split(',')]
            print(f"📊 Custom quantile levels: {quantile_levels}")
        else:
            # Generate standard quantile levels
            coverage_ranges = {
                'conservative': (0.1, 0.9),
                'standard': None,  # Will use auto-detection
                'wide': (0.05, 0.95)
            }
            if args.quantile_coverage == 'standard':
                quantile_levels = get_standard_quantile_levels(args.num_quantiles)
            else:
                from utils.quantile_utils import generate_quantile_levels
                coverage_range = coverage_ranges[args.quantile_coverage]
                quantile_levels = generate_quantile_levels(args.num_quantiles, coverage_range)
            
            print(f"📊 Generated {args.num_quantiles} quantile levels ({args.quantile_coverage} coverage):")
            print(f"    {quantile_levels}")
        
        print(f"📈 Quantile regression enabled:")
        print(describe_quantiles(quantile_levels))
        
        # Auto-enable KL loss for Bayesian models in quantile mode
        if args.model_type == 'bayesian' and not args.enable_kl:
            args.enable_kl = True
            print(f"🧠 Auto-enabled KL loss for Bayesian model in quantile mode")
    
    # KL loss configuration
    kl_weight = None
    if args.enable_kl or (args.model_type == 'bayesian' and args.quantile_mode):
        if args.kl_weight is not None:
            kl_weight = args.kl_weight
        else:
            # Auto-suggest KL weight with reasonable data loss magnitude estimate
            # Estimate based on typical MSE values for normalized financial data
            estimated_data_loss = 0.5  # Reasonable default for normalized data
            kl_weight = suggest_kl_weight(
                data_loss_magnitude=estimated_data_loss,
                target_percentage=args.kl_target_percentage  # Already in percentage format
            )
        
        print(f"🔥 KL divergence loss enabled:")
        print(f"    Weight: {kl_weight}")
        print(f"    Annealing: {args.kl_anneal}")
        print(f"    Target percentage: {args.kl_target_percentage}%")
    
    # Set random seed
    set_seed(args.seed)
    
    # Load and validate configuration
    config = load_and_validate_config(args.config)
    if config is None:
        return
    
    # Smart dimension management - automatically fix all dimension issues
    print(f"\n🔧 Smart dimension management...")
    data_path = os.path.join(config.get('root_path', 'data'), config.get('data_path', ''))
    
    if os.path.exists(data_path):
        # Create dimension manager
        dm = smart_dimension_setup(data_path)
        
        # Auto-fix the config with correct dimensions
        mode = config.get('features', 'MS')
        temp_config_path = f"/tmp/temp_config_{mode}_{args.model_type}.yaml"
        dm.update_config_file(args.config, mode, temp_config_path)
        
        # Reload the corrected config
        with open(temp_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        dims = dm.get_dimensions_for_mode(mode)
        print(f"✅ Auto-corrected dimensions: enc_in={dims['enc_in']}, dec_in={dims['dec_in']}, c_out={dims['c_out']}")
    else:
        print(f"⚠️ Data file not found: {data_path}, using config as-is")
    
    # Convert config to args with quantile and KL parameters
    exp_args = create_args_from_config(config, args.model_type, quantile_levels, kl_weight, args.kl_anneal)
    
    # Handle synthetic data for convergence testing
    if convergence_test_mode:
        print(f"\n🔬 Setting up synthetic data for convergence testing...")
        
        # Create synthetic config
        synthetic_config = {
            'type': args.synthetic_type,
            'n_points': args.synthetic_n_points,
            'noise_level': args.synthetic_noise,
            'seq_len': exp_args.seq_len,
            'pred_len': exp_args.pred_len
        }
        
        if args.synthetic_type == 'complex':
            synthetic_config['complexity'] = args.synthetic_complexity
            synthetic_config['n_features'] = exp_args.enc_in
            synthetic_config['n_targets'] = exp_args.c_out
        
        # Generate synthetic data analysis
        from utils.data_analysis import analyze_dataset, save_synthetic_data
        synthetic_analysis = analyze_dataset(
            data_path="",  # Will be ignored in synthetic mode
            test_model_convergence_simple_fn=True,
            synthetic_config=synthetic_config
        )
        
        # Save synthetic data to temporary file
        synthetic_data_path = save_synthetic_data(synthetic_analysis, f"data/temp_synthetic_{args.synthetic_type}.csv")
        
        # Update config to use synthetic data
        exp_args.data_path = os.path.basename(synthetic_data_path)
        exp_args.root_path = os.path.dirname(synthetic_data_path)
        
        # Update dimensions based on synthetic data
        mode = exp_args.features
        mode_config = synthetic_analysis[f'mode_{mode}']
        exp_args.enc_in = mode_config['enc_in']
        exp_args.dec_in = mode_config['dec_in']
        exp_args.c_out = mode_config['c_out']
        
        print(f"✅ Synthetic data prepared:")
        print(f"   Data file: {synthetic_data_path}")
        print(f"   Mathematical relationships: {len(synthetic_analysis.get('mathematical_relationships', {}))}")
        print(f"   Updated dimensions: enc_in={exp_args.enc_in}, dec_in={exp_args.dec_in}, c_out={exp_args.c_out}")
    
    # Override GPU settings regardless of config
    exp_args.use_gpu = False
    exp_args.use_multi_gpu = False
    exp_args.gpu_type = 'cpu'
    exp_args.devices = 'cpu'
    
    # Override model based on model_type
    exp_args.model = get_model_class(args.model_type)
    
    # Data path for validation
    data_path = os.path.join(exp_args.root_path, exp_args.data_path)
    
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
        
        # KL tuning for Bayesian models if enabled
        if args.enable_kl and args.model_type == 'bayesian':
            print(f"\n🧠 Initializing KL loss tuning...")
            
            # Create KL tuner
            kl_tuner = KLTuner(
                model=exp.model,
                target_kl_percentage=args.kl_target_percentage / 100.0  # Convert percentage to fraction
            )
            
            # Add KL tuner to experiment
            exp.kl_tuner = kl_tuner
            
            print(f"   KL tuner initialized with target {args.kl_target_percentage}% of total loss")
            if args.kl_anneal:
                print(f"   KL weight annealing enabled")
        
        # Training
        print(f"\n🎯 Starting training...")
        print(f"   Using device: {exp.device}")
        print(f"   Model parameters: {sum(p.numel() for p in exp.model.parameters()):,}")
        
        if quantile_levels:
            print(f"   Quantile regression: {len(quantile_levels)} quantiles")
            print(f"   Quantile levels: {quantile_levels}")
        
        if args.enable_kl:
            print(f"   KL divergence loss: weight={kl_weight}, annealing={args.kl_anneal}")
        
        # Train the model
        setting = f"{exp_args.model_id}_{exp_args.features}_{exp_args.model}_{exp_args.seq_len}_{exp_args.pred_len}"
        
        if quantile_levels:
            # Add quantile info to setting name for better tracking
            quantile_suffix = f"_q{len(quantile_levels)}"
            setting += quantile_suffix
            
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
