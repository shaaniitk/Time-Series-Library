#!/usr/bin/env python3
"""
Data Analysis Utilities for Dynamic Configuration

This module provides utilities to automatically analyze datasets and determine
the appropriate model configuration parameters (number of features, targets, etc.)

Features:
- Real dataset analysis
- Synthetic data generation for model convergence testing
- Dynamic configuration generation
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import yaml
import logging

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_dataset(data_path: str, 
                   target_columns: Union[str, List[str]] = None, 
                   date_column: str = 'date',
                   test_model_convergence_simple_fn: bool = False,
                   synthetic_config: Optional[Dict] = None) -> Dict:
    """
    Analyze a dataset to determine its characteristics for dynamic configuration.
    
    Args:
        data_path: Path to the CSV data file
        target_columns: Target column name(s). If None, assumes OHLC (first 4 non-date columns)
        date_column: Name of the date column
        test_model_convergence_simple_fn: If True, generate synthetic data for convergence testing
        synthetic_config: Configuration for synthetic data generation
        
    Returns:
        Dictionary with dataset characteristics and data arrays
    """
    if test_model_convergence_simple_fn:
        logger.info("🔬 Convergence test mode: generating synthetic data")
        return _generate_synthetic_analysis(synthetic_config or {})
    
    logger.info(f"Analyzing dataset: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded dataset with shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Identify date column
    if date_column not in df.columns:
        # Try common date column names
        date_candidates = ['date', 'Date', 'timestamp', 'Timestamp', 'time', 'Time']
        date_column = None
        for candidate in date_candidates:
            if candidate in df.columns:
                date_column = candidate
                break
        
        if date_column is None:
            logger.warning("No date column found, assuming first column is date")
            date_column = df.columns[0]
    
    logger.info(f"Using date column: {date_column}")
    
    # Get non-date columns
    feature_columns = [col for col in df.columns if col != date_column]
    
    # Determine target columns
    if target_columns is None:
        # Default: assume first 4 non-date columns are OHLC targets
        target_columns = feature_columns[:4]
        logger.info(f"Auto-detected target columns (OHLC): {target_columns}")
    elif isinstance(target_columns, str):
        target_columns = [col.strip() for col in target_columns.split(',')]
    
    # Validate target columns exist
    missing_targets = [col for col in target_columns if col not in df.columns]
    if missing_targets:
        raise ValueError(f"Target columns not found in dataset: {missing_targets}")
    
    # Separate target and covariate columns
    covariate_columns = [col for col in feature_columns if col not in target_columns]
    
    # Calculate dimensions
    n_total_features = len(feature_columns)
    n_targets = len(target_columns)
    n_covariates = len(covariate_columns)
    n_samples = len(df)
    
    # Extract covariate and target data arrays
    covariate_data = df[covariate_columns].values if covariate_columns else np.array([]).reshape(n_samples, 0)
    target_data = df[target_columns].values
    
    # Basic statistics
    analysis = {
        'data_path': data_path,
        'n_samples': n_samples,
        'n_total_features': n_total_features,
        'n_targets': n_targets,
        'n_covariates': n_covariates,
        'date_column': date_column,
        'target_columns': target_columns,
        'covariate_columns': covariate_columns,
        'all_feature_columns': feature_columns,
        'data_source': 'real',
        
        # Data arrays for training
        'covariate_data': covariate_data,
        'target_data': target_data,
        'dataframe': df,
        
        # Mode-specific dimensions
        'mode_M': {
            'enc_in': n_total_features,
            'dec_in': n_total_features,
            'c_out': n_total_features,
            'description': f"Multivariate: {n_total_features} → {n_total_features}"
        },
        'mode_MS': {
            'enc_in': n_total_features,
            'dec_in': n_total_features,
            'c_out': n_targets,
            'description': f"Multi-target: {n_total_features} → {n_targets}"
        },
        'mode_S': {
            'enc_in': n_targets,
            'dec_in': n_targets,
            'c_out': n_targets,
            'description': f"Target-only: {n_targets} → {n_targets}"
        }
    }
    
    logger.info(f"Dataset analysis complete:")
    logger.info(f"  Total features: {n_total_features}")
    logger.info(f"  Target features: {n_targets} {target_columns}")
    logger.info(f"  Covariate features: {n_covariates}")
    logger.info(f"  Samples: {n_samples}")
    
    return analysis

def _generate_synthetic_analysis(synthetic_config: Dict) -> Dict:
    """
    Generate synthetic data analysis for convergence testing.
    
    Args:
        synthetic_config: Configuration for synthetic data generation
        
    Returns:
        Analysis dictionary with synthetic data
    """
    from utils.synthetic_data_generator import generate_sincos_basic, generate_complex_synthetic
    
    # Default synthetic configuration
    default_config = {
        'type': 'sincos',  # 'sincos' or 'complex'
        'n_points': 2000,
        'seq_len': 100,
        'pred_len': 50,
        'noise_level': 0.1,
        'frequencies': [5, 10, 15]
    }
    
    # Merge with user config
    config = {**default_config, **synthetic_config}
    
    logger.info(f"Generating synthetic data with config: {config}")
    
    if config['type'] == 'sincos':
        synthetic_result = generate_sincos_basic(
            n_points=config['n_points'],
            seq_len=config['seq_len'],
            pred_len=config['pred_len'],
            noise_level=config['noise_level'],
            frequencies=config['frequencies']
        )
    else:
        # Complex synthetic data
        synthetic_result = generate_complex_synthetic(
            n_points=config['n_points'],
            n_features=config.get('n_features', 10),
            n_targets=config.get('n_targets', 3),
            complexity=config.get('complexity', 'medium'),
            noise_level=config['noise_level']
        )
    
    data = synthetic_result['data']
    metadata = synthetic_result['metadata']
    arrays = synthetic_result['arrays']
    
    # Extract analysis information
    feature_columns = [col for col in data.columns if col != 'date']
    target_columns = metadata['target_columns']
    covariate_columns = metadata['feature_columns']
    
    n_total_features = len(feature_columns)
    n_targets = len(target_columns)
    n_covariates = len(covariate_columns)
    n_samples = len(data)
    
    # Create analysis structure matching real data format
    analysis = {
        'data_path': 'synthetic_data',
        'n_samples': n_samples,
        'n_total_features': n_total_features,
        'n_targets': n_targets,
        'n_covariates': n_covariates,
        'date_column': 'date',
        'target_columns': target_columns,
        'covariate_columns': covariate_columns,
        'all_feature_columns': feature_columns,
        'data_source': 'synthetic',
        
        # Synthetic-specific metadata
        'synthetic_config': config,
        'synthetic_metadata': metadata,
        'mathematical_relationships': metadata.get('mathematical_relationships', {}),
        
        # Data arrays for training
        'covariate_data': arrays['covariates'],
        'target_data': arrays['targets'],
        'dataframe': data,
        
        # Mode-specific dimensions
        'mode_M': {
            'enc_in': n_total_features,
            'dec_in': n_total_features,
            'c_out': n_total_features,
            'description': f"Multivariate (Synthetic): {n_total_features} → {n_total_features}"
        },
        'mode_MS': {
            'enc_in': n_total_features,
            'dec_in': n_targets,
            'c_out': n_targets,
            'description': f"Multi-target (Synthetic): {n_total_features} → {n_targets}"
        },
        'mode_S': {
            'enc_in': n_targets,
            'dec_in': n_targets,
            'c_out': n_targets,
            'description': f"Target-only (Synthetic): {n_targets} → {n_targets}"
        }
    }
    
    logger.info(f"Synthetic data generated:")
    logger.info(f"  Type: {config['type']}")
    logger.info(f"  Total features: {n_total_features}")
    logger.info(f"  Target features: {n_targets} {target_columns}")
    logger.info(f"  Covariate features: {n_covariates}")
    logger.info(f"  Samples: {n_samples}")
    logger.info(f"  Mathematical relationships: {len(analysis['mathematical_relationships'])}")
    
    return analysis

def save_synthetic_data(analysis: Dict, output_path: str = None) -> str:
    """
    Save synthetic data to CSV file for use with training scripts.
    
    Args:
        analysis: Analysis result containing synthetic data
        output_path: Output path for CSV file
        
    Returns:
        Path to saved CSV file
    """
    if analysis['data_source'] != 'synthetic':
        raise ValueError("Analysis must contain synthetic data")
    
    if output_path is None:
        # Auto-generate path
        config = analysis['synthetic_config']
        output_path = f"data/synthetic_{config['type']}_{config['n_points']}pts.csv"
    
    # Save dataframe
    analysis['dataframe'].to_csv(output_path, index=False)
    
    logger.info(f"Synthetic data saved to: {output_path}")
    return output_path

def generate_dynamic_config(base_config_path: str, data_analysis: Dict, 
                          output_path: str = None, mode: str = 'MS') -> str:
    """
    Generate a dynamic configuration file based on data analysis.
    
    Args:
        base_config_path: Path to base configuration template
        data_analysis: Result from analyze_dataset()
        output_path: Output path for new config (if None, auto-generate)
        mode: Forecasting mode ('M', 'MS', 'S')
        
    Returns:
        Path to the generated configuration file
    """
    # Load base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update with dynamic values
    mode_config = data_analysis[f'mode_{mode}']
    
    config['enc_in'] = mode_config['enc_in']
    config['dec_in'] = mode_config['dec_in']  
    config['c_out'] = mode_config['c_out']
    config['features'] = mode
    
    # Update data path
    config['data_path'] = os.path.basename(data_analysis['data_path'])
    
    # Add analysis metadata
    config['_data_analysis'] = {
        'n_total_features': data_analysis['n_total_features'],
        'n_targets': data_analysis['n_targets'],
        'n_covariates': data_analysis['n_covariates'],
        'target_columns': data_analysis['target_columns'],
        'mode_description': mode_config['description'],
        'generated_from': base_config_path
    }
    
    # Generate output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(base_config_path))[0]
        complexity = base_name.split('_')[-1]  # Extract complexity level
        output_path = f"config/config_enhanced_autoformer_{mode}_{complexity}_dynamic.yaml"
    
    # Save dynamic config
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Generated dynamic config: {output_path}")
    logger.info(f"Mode {mode}: {mode_config['description']}")
    
    return output_path

def update_all_configs_for_dataset(data_path: str, target_columns: Union[str, List[str]] = None):
    """
    Update all existing config files to match a new dataset.
    
    Args:
        data_path: Path to the new dataset
        target_columns: Target column specification
    """
    # Analyze the dataset
    analysis = analyze_dataset(data_path, target_columns)
    
    # Find all existing config files
    config_files = []
    for file in os.listdir('config'):
        if file.startswith('config_enhanced_autoformer_') and file.endswith('.yaml'):
            config_files.append(file)
    
    logger.info(f"Found {len(config_files)} config files to update")
    
    updated_files = []
    for config_file in config_files:
        try:
            # Extract mode and complexity from filename
            parts = config_file.replace('.yaml', '').split('_')
            if len(parts) >= 5:
                mode = parts[3]  # M, MS, or S
                complexity = parts[4]  # ultralight, light, etc.
                
                if mode in ['M', 'MS', 'S']:
                    # Generate updated config
                    output_path = f"config/config_enhanced_autoformer_{mode}_{complexity}_updated.yaml"
                    new_config = generate_dynamic_config(f"config/{config_file}", analysis, output_path, mode)
                    updated_files.append(new_config)
                    logger.info(f"Updated: config/{config_file} → {new_config}")
        
        except Exception as e:
            logger.warning(f"Failed to update {config_file}: {e}")
    
    logger.info(f"Successfully updated {len(updated_files)} configurations")
    return updated_files

def validate_config_with_data(config_path: str, data_path: str) -> Dict:
    """
    Validate that a configuration file matches the actual data dimensions.
    
    Args:
        config_path: Path to configuration file
        data_path: Path to data file
        
    Returns:
        Validation results dictionary
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Analyze data
    target_columns = config.get('target', 'log_Close')
    analysis = analyze_dataset(data_path, target_columns)
    
    mode = config.get('features', 'MS')
    expected = analysis[f'mode_{mode}']
    
    # Check dimensions
    validation = {
        'config_path': config_path,
        'data_path': data_path,
        'mode': mode,
        'valid': True,
        'issues': []
    }
    
    # Validate each dimension
    checks = [
        ('enc_in', expected['enc_in']),
        ('dec_in', expected['dec_in']),
        ('c_out', expected['c_out'])
    ]
    
    for param, expected_value in checks:
        config_value = config.get(param)
        if config_value != expected_value:
            validation['valid'] = False
            validation['issues'].append({
                'parameter': param,
                'config_value': config_value,
                'expected_value': expected_value,
                'description': f"Config has {param}={config_value}, but data requires {param}={expected_value}"
            })
    
    return validation

if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_analysis.py <data_path> [target_columns]")
        sys.exit(1)
    
    data_path = sys.argv[1]
    target_columns = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Analyze dataset
    analysis = analyze_dataset(data_path, target_columns)
    
    print("\n🔍 Dataset Analysis Results:")
    print("=" * 50)
    print(f"Total features: {analysis['n_total_features']}")
    print(f"Target features: {analysis['n_targets']} {analysis['target_columns']}")
    print(f"Covariate features: {analysis['n_covariates']}")
    print(f"Samples: {analysis['n_samples']}")
    
    print("\n📊 Mode Configurations:")
    for mode in ['M', 'MS', 'S']:
        mode_config = analysis[f'mode_{mode}']
        print(f"  {mode}: {mode_config['description']}")
        print(f"      enc_in={mode_config['enc_in']}, dec_in={mode_config['dec_in']}, c_out={mode_config['c_out']}")
