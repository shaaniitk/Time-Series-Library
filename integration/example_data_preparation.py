#!/usr/bin/env python3
"""
Example script demonstrating the Financial Data Manager usage.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_provider.data_prepare import FinancialDataManager
import pandas as pd


def example_data_preparation():
    """Demonstrate how to use the Financial Data Manager."""
    
    print("=== Financial Data Manager Demo ===\n")
    
    # Initialize the data manager
    data_manager = FinancialDataManager(data_root='data')
    
    print("1. Loading individual data files...")
      # Load target data (business days)
    try:
        target_df = data_manager.load_target_data('nifty50_returns.csv')  # Use CSV instead of parquet
        print(f"   ✓ Target data loaded: {target_df.shape}")
        print(f"     Date range: {target_df['date'].min()} to {target_df['date'].max()}")
        print(f"     Target columns: {data_manager.target_columns[:5]}...")  # Show first 5
    except Exception as e:
        print(f"   ✗ Error loading target data: {e}")
        return
    
    # Load dynamic covariates (daily)
    try:
        dynamic_df = data_manager.load_dynamic_covariates('comprehensive_dynamic_features_nifty.csv')
        if not dynamic_df.empty:
            print(f"   ✓ Dynamic covariates loaded: {dynamic_df.shape}")
            print(f"     Date range: {dynamic_df['date'].min()} to {dynamic_df['date'].max()}")
            print(f"     Covariate columns: {len(data_manager.dynamic_cov_columns)} features")
        else:
            print("   - No dynamic covariates loaded")
    except Exception as e:
        print(f"   ✗ Error loading dynamic covariates: {e}")
    
    # Load static covariates
    try:
        static_df = data_manager.load_static_covariates('india_static_features.csv')
        if not static_df.empty:
            print(f"   ✓ Static covariates loaded: {static_df.shape}")
            print(f"     Static features: {len(data_manager.static_cov_columns)} features")
        else:
            print("   - No static covariates loaded")
    except Exception as e:
        print(f"   ✗ Error loading static covariates: {e}")
    
    print("\n2. Aligning data by dates...")
    
    # Align all data using target dates as reference
    try:
        aligned_data = data_manager.align_data_by_dates(
            method='forward_fill',  # Use most recent covariate data for each target date
            max_gap_days=7         # Allow up to 7 days gap for alignment
        )
        print(f"   ✓ Data alignment completed: {aligned_data.shape}")
        
        # Show data info
        data_info = data_manager.get_data_info()
        print(f"   Total columns: {data_info['total_shape'][1]}")
        print(f"   Target columns: {len(data_info['columns']['target'])}")
        print(f"   Dynamic covariate columns: {len(data_info['columns']['dynamic_covariates'])}")
        print(f"   Static covariate columns: {len(data_info['columns']['static_covariates'])}")
          # Check for missing values
        missing_total = sum(v for v in data_info['missing_values'].values() if pd.notna(v))
        print(f"   Missing values: {missing_total}")
        
        # INVESTIGATE COLUMN OVERLAP ISSUE
        print(f"\n   DEBUG: Column overlap investigation:")
        target_cols = data_info['columns']['target']
        dynamic_cols = data_info['columns']['dynamic_covariates'] 
        static_cols = data_info['columns']['static_covariates']
        
        print(f"   Target columns ({len(target_cols)}): {target_cols}")
        print(f"   Dynamic columns ({len(dynamic_cols)}): {dynamic_cols[:5]}...")  # Show first 5
        print(f"   Static columns ({len(static_cols)}): {static_cols[:5]}...")   # Show first 5
        
        # Check overlaps
        target_dynamic_overlap = set(target_cols) & set(dynamic_cols)
        target_static_overlap = set(target_cols) & set(static_cols)
        dynamic_static_overlap = set(dynamic_cols) & set(static_cols)
        
        print(f"   Target-Dynamic overlap: {len(target_dynamic_overlap)} cols: {target_dynamic_overlap}")
        print(f"   Target-Static overlap: {len(target_static_overlap)} cols: {target_static_overlap}")
        print(f"   Dynamic-Static overlap: {len(dynamic_static_overlap)} cols: {dynamic_static_overlap}")
        
        total_unique_cols = len(set(target_cols) | set(dynamic_cols) | set(static_cols))
        print(f"   Total unique columns expected: {total_unique_cols}")
        print(f"   Actual columns in final data: {aligned_data.shape[1]}")
        print(f"   Missing columns: {total_unique_cols - aligned_data.shape[1]}")
        
    except Exception as e:
        print(f"   ✗ Error during alignment: {e}")
        return
    
    print("\n3. Data summary:")
    print(f"   Date range: {data_info['date_range']['start']} to {data_info['date_range']['end']}")
    print(f"   Total business days: {data_info['total_shape'][0]}")
    print(f"   Features per day: {data_info['total_shape'][1] - 1}")  # -1 for date column
    
    # Show sample data
    print("\n4. Sample data (first 5 rows, first 10 columns):")
    sample_cols = aligned_data.columns[:10]
    print(aligned_data[sample_cols].head())
    
    print("\n5. Saving prepared data...")
    try:
        output_path = 'data/prepared_financial_data.csv'
        data_manager.save_prepared_data(output_path, format='csv')
        print(f"   ✓ Data saved to: {output_path}")
    except Exception as e:
        print(f"   ✗ Error saving data: {e}")
    
    return data_manager, aligned_data


def complete_pipeline_example():
    """Example of using the complete pipeline in one call."""
    
    print("\n=== Complete Pipeline Example ===\n")
    
    # Use complete pipeline
    data_manager = FinancialDataManager(data_root='data')
    
    try:
        prepared_data = data_manager.prepare_data(
            target_file='nifty50_returns.csv',  # Use CSV instead of parquet
            dynamic_cov_file='comprehensive_dynamic_features_nifty.csv',
            static_cov_file='india_static_features.csv',
            alignment_method='forward_fill',
            max_gap_days=7
        )
        
        print(f"✓ Complete pipeline successful!")
        print(f"  Final data shape: {prepared_data.shape}")
        
        # Get detailed info
        info = data_manager.get_data_info()
        print(f"  Target features: {len(info['columns']['target'])}")
        print(f"  Dynamic covariates: {len(info['columns']['dynamic_covariates'])}")
        print(f"  Static covariates: {len(info['columns']['static_covariates'])}")
        
        return prepared_data
        
    except Exception as e:
        print(f"✗ Pipeline failed: {e}")
        return None


if __name__ == "__main__":
    # Run the examples
    data_manager, aligned_data = example_data_preparation()
    
    if aligned_data is not None:
        prepared_data = complete_pipeline_example()
        
        if prepared_data is not None:
            print(f"\n🎉 All examples completed successfully!")
            print(f"Ready for time series modeling with {prepared_data.shape[1]-1} features")
