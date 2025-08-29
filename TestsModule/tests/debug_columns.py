#!/usr/bin/env python3

import pandas as pd
import os

def debug_column_structure():
    """Debug the column structure in the data loader"""
    
    # Read the data
    data_path = os.path.join('data', 'prepared_financial_data.csv')
    df_raw = pd.read_csv(data_path)
    
    print("Original DataFrame columns:")
    print(df_raw.columns.tolist())
    print(f"Total columns: {len(df_raw.columns)}")
    
    # Simulate what Dataset_Custom does for features='M'
    cols_data = df_raw.columns[1:]  # All columns except date
    df_data = df_raw[cols_data]
    
    print("\nAfter removing date column (cols_data):")
    print(df_data.columns.tolist())
    print(f"Total columns: {len(df_data.columns)}")
    
    # Check first few columns
    print(f"\nFirst 4 columns (should be OHLC targets): {df_data.columns[:4].tolist()}")
    print(f"Remaining columns (should be covariates): {df_data.columns[4:].tolist()[:10]}...")  # Show first 10 covariates
    
    # Check data values
    print(f"\nFirst few rows of data:")
    print(df_data.head())
    
    print(f"\nData shape: {df_data.shape}")

if __name__ == "__main__":
    debug_column_structure()
