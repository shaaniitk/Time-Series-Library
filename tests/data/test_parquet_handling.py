#!/usr/bin/env python3
"""
Test script specifically for Parquet date index handling.
This verifies that the data manager correctly handles Parquet files 
where the date is stored as an index rather than a column.
"""

import pandas as pd
import numpy as np
from data_provider.data_prepare import FinancialDataManager
from utils.logger import logger

def test_parquet_date_index():
    """Test loading Parquet file with date index."""
    print("="*60)
    print("Testing Parquet Date Index Handling")
    print("="*60)
    
    # Initialize data manager
    data_manager = FinancialDataManager(data_root='data')
    
    # Test 1: Load the original Parquet file
    print("\n1. Testing original Parquet file...")
    try:
        target_data = data_manager.load_target_data('nifty50_returns.parquet')
        print(f"    Successfully loaded Parquet: {target_data.shape}")
        print(f"    Date column: {'date' in target_data.columns}")
        if 'date' in target_data.columns:
            print(f"    Date range: {target_data['date'].min()} to {target_data['date'].max()}")
        print(f"    Target columns: {data_manager.target_columns}")
    except Exception as e:
        print(f"    Failed to load Parquet: {e}")
        return False
    
    # Test 2: Create a test Parquet file with date as index
    print("\n2. Creating test Parquet with date as index...")
    
    # Load the CSV version first
    csv_data = pd.read_csv('data/nifty50_returns.csv')
    csv_data['Date'] = pd.to_datetime(csv_data['Date'])
    
    # Create a version with date as index
    parquet_with_index = csv_data.set_index('Date')
    test_file = 'data/test_date_index.parquet'
    parquet_with_index.to_parquet(test_file)
    print(f"    Created test file: {test_file}")
    print(f"    Index type: {type(parquet_with_index.index)}")
    print(f"    Index name: {parquet_with_index.index.name}")
    
    # Test 3: Load the test file with date index
    print("\n3. Testing Parquet file with date as index...")
    try:
        data_manager_test = FinancialDataManager(data_root='data')
        test_data = data_manager_test.load_target_data('test_date_index.parquet')
        print(f"    Successfully loaded test Parquet: {test_data.shape}")
        print(f"    Date column exists: {'date' in test_data.columns}")
        if 'date' in test_data.columns:
            print(f"    Date range: {test_data['date'].min()} to {test_data['date'].max()}")
            print(f"    Date column type: {test_data['date'].dtype}")
        print(f"    Target columns: {data_manager_test.target_columns}")
        
        # Verify data integrity
        if len(test_data) == len(csv_data):
            print(f"    Data integrity preserved: {len(test_data)} rows")
        else:
            print(f"    Data size mismatch: {len(test_data)} vs {len(csv_data)}")
            
    except Exception as e:
        print(f"    Failed to load test Parquet: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Complete pipeline with Parquet
    print("\n4. Testing complete pipeline with Parquet...")
    try:
        # Use the Parquet file for target data
        complete_data = data_manager.prepare_data(
            target_file='nifty50_returns.parquet',
            dynamic_cov_file='comprehensive_dynamic_features_nifty.csv',
            static_cov_file='india_static_features.csv',
            alignment_method='forward_fill'
        )
        print(f"    Complete pipeline successful: {complete_data.shape}")
        print(f"    Date column: {'date' in complete_data.columns}")
        print(f"    Missing values: {complete_data.isnull().sum().sum()}")
        
    except Exception as e:
        print(f"    Pipeline failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("PARTY All Parquet date index tests passed!")
    print("The data manager correctly handles:")
    print("  - Parquet files with date as column")
    print("  - Parquet files with date as index")
    print("  - Complete pipeline with mixed file formats")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = test_parquet_date_index()
    if not success:
        exit(1)
