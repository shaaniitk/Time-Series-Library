"""
Practical Guide: Handling Multiple Targets with Different Dates

This guide provides step-by-step instructions for handling time series data
where different targets have different date ranges or missing data patterns.
"""

import pandas as pd
import numpy as np

# Example 1: Multiple CSV files with different date ranges
def example_multiple_files():
    """
    Scenario: You have separate CSV files for different targets:
    - target_A_data.csv: 2020-2023
    - target_B_data.csv: 2021-2024  
    - target_C_data.csv: 2019-2022
    """
    
    from utils.temporal_alignment import TemporalDataAligner, IrregularDatasetLoader
    import pandas as pd
    
    # Step 1: Initialize the aligner
    aligner = TemporalDataAligner(freq='D', fill_method='interpolate')
    loader = IrregularDatasetLoader(aligner)
    
    # Step 2: Load and align multiple files
    file_paths = [
        'data/target_A_data.csv',
        'data/target_B_data.csv', 
        'data/target_C_data.csv'
    ]
    target_names = ['target_A', 'target_B', 'target_C']
    
    aligned_data = loader.load_from_multiple_files(
        file_paths=file_paths,
        target_names=target_names,
        date_col='date'
    )
    
    # Step 3: Save aligned data for model training
    aligned_data.to_csv('data/aligned_multi_targets.csv', index=False)
    print(f"Aligned data shape: {aligned_data.shape}")
    
    return aligned_data


# Example 2: Single file with irregular patterns
def example_single_file_irregular():
    """
    Scenario: Single CSV with multiple targets but missing data:
    
    date,target_A,target_B,target_C,covariate_1,covariate_2
    2020-01-01,100,NaN,50,1.2,0.8
    2020-01-02,NaN,200,51,1.3,0.9
    2020-01-03,102,201,NaN,1.1,0.7
    ...
    """
    
    from utils.temporal_alignment import TemporalDataAligner
    import pandas as pd
    
    # Step 1: Load the irregular data
    df = pd.read_csv('data/irregular_targets.csv')
    
    # Step 2: Initialize aligner
    aligner = TemporalDataAligner(freq='D', fill_method='interpolate')
    
    # Step 3: Create separate series for each target
    target_cols = ['target_A', 'target_B', 'target_C']
    data_dict = {}
    
    for target in target_cols:
        # Extract only non-null observations for this target
        target_data = df[['date', target] + ['covariate_1', 'covariate_2']].dropna(subset=[target])
        data_dict[target] = target_data
        print(f"{target}: {len(target_data)} valid observations")
    
    # Step 4: Align all targets
    aligned_data = aligner.align_multiple_series(data_dict, date_col='date')
    
    # Step 5: Save for model training
    aligned_data.to_csv('data/aligned_single_file.csv', index=False)
    
    return aligned_data


# Example 3: Using with existing training scripts
def example_training_integration():
    """
    How to integrate irregular data handling with existing training scripts.
    """
    
    # Option A: Preprocess data first, then use normal training
    def preprocess_and_train():
        # Step 1: Align your irregular data
        aligned_data = example_multiple_files()  # or example_single_file_irregular()
        
        # Step 2: Use normal training script with aligned data
        import subprocess
        cmd = [
            'python', '../scripts/train/train_dynamic_autoformer.py',
            '--data', 'custom',
            '--data_path', 'aligned_multi_targets.csv',
            '--target', 'target_A,target_B,target_C',
            '--features', 'MS',  # Use MS mode for multiple targets
            '--seq_len', '96',
            '--pred_len', '24'
        ]
        subprocess.run(cmd)
    
    # Option B: Use irregular dataset loader directly
    def train_with_irregular_loader():
        import argparse
        from utils.irregular_dataset import create_irregular_data_loader
        
        # Create args (normally from command line)
        args = argparse.Namespace(
            data='custom',
            data_path='target_*.csv',  # Glob pattern for multiple files
            target='target_A,target_B,target_C',
            features='MS',
            seq_len=96,
            label_len=48,
            pred_len=24,
            scale=True,
            timeenc=1,
            freq='D'
        )
        
        # Create irregular dataset
        dataset = create_irregular_data_loader(args, 'data/', flag='train')
        
        # Use with PyTorch DataLoader
        from torch.utils.data import DataLoader
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        print(f"Dataset info: {dataset.get_data_info()}")
        
        return data_loader


# Example 4: Different strategies for different scenarios
def strategy_selector(scenario_type):
    """
    Select the best strategy based on your specific scenario.
    """
    
    strategies = {
        'different_files': {
            'description': 'Multiple CSV files with different date ranges',
            'best_approach': 'Use IrregularDatasetLoader.load_from_multiple_files()',
            'code_example': 'example_multiple_files()'
        },
        
        'single_file_missing': {
            'description': 'Single file with missing values in different targets',
            'best_approach': 'Use TemporalDataAligner with interpolation',
            'code_example': 'example_single_file_irregular()'
        },
        
        'different_frequencies': {
            'description': 'Targets sampled at different frequencies',
            'best_approach': 'Resample to common frequency before alignment',
            'preprocessing': '''
            # Resample each target to daily frequency
            target_A_daily = target_A.resample('D').mean()
            target_B_daily = target_B.resample('D').mean()
            # Then align
            '''
        },
        
        'future_availability': {
            'description': 'Some targets have longer historical data',
            'best_approach': 'Use common_start/common_end parameters',
            'code_example': '''
            aligned = aligner.align_multiple_series(
                data_dict, 
                common_start='2021-01-01',  # Use intersection of dates
                common_end='2023-12-31'
            )
            '''
        },
        
        'mixed_granularity': {
            'description': 'Some daily, some monthly data',
            'best_approach': 'Convert all to finest granularity with forward-fill',
            'preprocessing': '''
            # Convert monthly to daily
            monthly_data_daily = monthly_data.resample('D').ffill()
            # Then align with daily data
            '''
        }
    }
    
    return strategies.get(scenario_type, 'Unknown scenario type')


# Example 5: Quality checks and validation
def validate_irregular_alignment(aligned_data, original_data_dict):
    """
    Validate that the alignment process worked correctly.
    """
    
    print("=== Alignment Validation Report ===")
    
    # Check date range
    print(f"Aligned date range: {aligned_data['date'].min()} to {aligned_data['date'].max()}")
    print(f"Total periods: {len(aligned_data)}")
    
    # Check data preservation
    for target_name, original_df in original_data_dict.items():
        if target_name.startswith('_'):
            continue
            
        # Find corresponding column in aligned data
        target_cols = [col for col in aligned_data.columns if target_name in col]
        
        if target_cols:
            aligned_col = target_cols[0]
            
            # Check overlap periods
            original_dates = set(pd.to_datetime(original_df['date']).dt.date)
            aligned_dates = set(pd.to_datetime(aligned_data['date']).dt.date)
            overlap_dates = original_dates.intersection(aligned_dates)
            
            print(f"\n{target_name}:")
            print(f"  Original observations: {len(original_df)}")
            print(f"  Overlap periods: {len(overlap_dates)}")
            print(f"  Missing values in aligned: {aligned_data[aligned_col].isnull().sum()}")
            print(f"  Coverage ratio: {len(overlap_dates) / len(original_dates):.2%}")
    
    # Check for data quality issues
    numeric_cols = aligned_data.select_dtypes(include=[np.number]).columns
    
    print(f"\n=== Data Quality ===")
    for col in numeric_cols:
        if col != 'date':
            print(f"{col}: {aligned_data[col].isnull().sum()} missing, "
                  f"range [{aligned_data[col].min():.2f}, {aligned_data[col].max():.2f}]")


if __name__ == "__main__":
    print("Irregular Data Handling Examples")
    print("================================")
    
    # Show available strategies
    scenarios = ['different_files', 'single_file_missing', 'different_frequencies', 
                'future_availability', 'mixed_granularity']
    
    for scenario in scenarios:
        strategy = strategy_selector(scenario)
        print(f"\nScenario: {strategy['description']}")
        print(f"Best approach: {strategy['best_approach']}")
        
    print("\nRun specific examples:")
    print("- example_multiple_files()")
    print("- example_single_file_irregular()")  
    print("- example_training_integration()")
