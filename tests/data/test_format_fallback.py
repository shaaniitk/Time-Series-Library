#!/usr/bin/env python3
"""
Test script to demonstrate CSV/Parquet format detection and fallback.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_provider.data_prepare import FinancialDataManager
import pandas as pd


def test_format_fallback():
    """Test the format detection and fallback functionality."""
    
    print("=== Testing CSV/Parquet Format Support ===\n")
    
    data_manager = FinancialDataManager(data_root='data')
    
    print("1. Testing Parquet loading...")
    try:
        # Load the Parquet version
        target_df = data_manager.load_target_data('nifty50_returns.parquet')
        print(f"    Parquet file loaded successfully: {target_df.shape}")
        print(f"     Date range: {target_df['date'].min()} to {target_df['date'].max()}")
        print(f"     Target columns: {len(data_manager.target_columns)} features")
    except Exception as e:
        print(f"    Error loading Parquet: {e}")
    
    print("\n2. Testing CSV loading...")
    try:
        # Reinitialize to clear previous data
        data_manager = FinancialDataManager(data_root='data')
        # Load the CSV version
        target_df = data_manager.load_target_data('nifty50_returns.csv')
        print(f"    CSV file loaded successfully: {target_df.shape}")
        print(f"     Date range: {target_df['date'].min()} to {target_df['date'].max()}")
        print(f"     Target columns: {len(data_manager.target_columns)} features")
    except Exception as e:
        print(f"    Error loading CSV: {e}")
    
    print("\n3. Testing auto-detection (file without clear extension)...")
    try:
        # Copy a file to test auto-detection
        import shutil
        test_file = 'data/test_auto_detect'
        shutil.copy('data/nifty50_returns.parquet', test_file)
        
        # Reinitialize
        data_manager = FinancialDataManager(data_root='data')
        target_df = data_manager.load_target_data('test_auto_detect')
        print(f"    Auto-detection successful: {target_df.shape}")
        
        # Clean up
        os.remove(test_file)
    except Exception as e:
        print(f"    Auto-detection failed: {e}")
        # Clean up on error
        try:
            os.remove('data/test_auto_detect')
        except:
            pass
    
    print("\n4. Testing fallback mechanism...")
    try:
        # Create a test file with wrong extension but correct content
        import shutil
        test_file = 'data/test_fallback.csv'  # CSV extension but Parquet content
        shutil.copy('data/nifty50_returns.parquet', test_file)
        
        # Reinitialize
        data_manager = FinancialDataManager(data_root='data')
        target_df = data_manager.load_target_data('test_fallback.csv')
        print(f"    Fallback mechanism worked: {target_df.shape}")
        print("     (CSV extension with Parquet content was handled)")
        
        # Clean up
        os.remove(test_file)
    except Exception as e:
        print(f"    Fallback mechanism failed: {e}")
        # Clean up on error
        try:
            os.remove('data/test_fallback.csv')
        except:
            pass
    
    print("\n5. Comparing CSV vs Parquet content...")
    try:
        # Load both versions and compare
        data_manager_csv = FinancialDataManager(data_root='data')
        csv_df = data_manager_csv.load_target_data('nifty50_returns.csv')
        
        data_manager_parquet = FinancialDataManager(data_root='data')
        parquet_df = data_manager_parquet.load_target_data('nifty50_returns.parquet')
        
        print(f"   CSV shape: {csv_df.shape}")
        print(f"   Parquet shape: {parquet_df.shape}")
        
        # Check if data is equivalent
        if csv_df.shape == parquet_df.shape:
            print("    Both files have same dimensions")
            
            # Compare columns
            csv_cols = set(csv_df.columns)
            parquet_cols = set(parquet_df.columns)
            if csv_cols == parquet_cols:
                print("    Both files have same columns")
            else:
                print(f"   ! Column differences: CSV only: {csv_cols - parquet_cols}, Parquet only: {parquet_cols - csv_cols}")
        else:
            print("   ! Files have different dimensions")
            
    except Exception as e:
        print(f"    Comparison failed: {e}")

    print("\nPARTY Format support testing completed!")


if __name__ == "__main__":
    test_format_fallback()
