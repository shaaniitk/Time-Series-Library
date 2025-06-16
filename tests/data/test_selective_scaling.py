#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import sys
import argparse
from sklearn.preprocessing import StandardScaler

# Add the current directory to Python path
sys.path.append('.')

def test_data_structure():
    """Test that the data structure matches our assumptions"""
    
    # Read raw data 
    df_raw = pd.read_csv('data/prepared_financial_data.csv')
    print("Raw data columns:", df_raw.columns[:6].tolist())
    
    # Simulate what Dataset_Custom does
    cols_data = df_raw.columns[1:]  # Remove date column
    df_data = df_raw[cols_data]
    
    print("After removing date column:", df_data.columns[:6].tolist())
    print(f"First 4 columns (targets): {df_data.columns[:4].tolist()}")
    print(f"Next few columns (covariates): {df_data.columns[4:8].tolist()}")
    
    # Convert to numpy array (what the scaler sees)
    data = df_data.values
    print(f"Data shape: {data.shape}")
    
    # Test the selective scaling logic
    border1s = [0, 12 * 30 * 24 * 4 - 96, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - 96]  # example
    border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
    
    # Use actual data size for realistic borders
    n = len(data)
    border1s = [0, n - 200 - 96, n - 96 - 50]
    border2s = [n - 200, n - 50, n]
    
    print(f"Training data range: {border1s[0]} to {border2s[0]} (rows)")
    
    # Target columns (first 4)
    target_cols = list(range(4))
    covariate_cols = list(range(4, data.shape[1]))
    
    print(f"Target columns: {target_cols}")
    print(f"Covariate columns: first 10 = {covariate_cols[:10]}")
    
    # Test scaling
    print("\n--- Testing Selective Scaling ---")
    
    # Targets: scale only training data
    target_train_data = data[border1s[0]:border2s[0], target_cols]
    target_scaler = StandardScaler()
    target_scaler.fit(target_train_data)
    
    print(f"Target training data shape: {target_train_data.shape}")
    print(f"Target training mean: {target_train_data.mean(axis=0)}")
    print(f"Target training std: {target_train_data.std(axis=0)}")
    
    # Initialize scaled data
    scaled_data = data.copy()
    
    # Scale only training portion of targets
    scaled_data[border1s[0]:border2s[0], target_cols] = target_scaler.transform(
        data[border1s[0]:border2s[0], target_cols]
    )
    
    print(f"Targets after scaling (training portion): mean = {scaled_data[border1s[0]:border2s[0], target_cols].mean(axis=0)}")
    print(f"Targets after scaling (validation portion): mean = {scaled_data[border2s[0]:border2s[0]+10, target_cols].mean(axis=0)}")
      # Covariates: scale entire dataset using full dataset statistics
    if len(covariate_cols) > 0:
        covariate_scaler = StandardScaler()
        covariate_scaler.fit(data[:, covariate_cols])
        
        scaled_data[:, covariate_cols] = covariate_scaler.transform(data[:, covariate_cols])
        
        print(f"Covariate full data shape: {data[:, covariate_cols].shape}")
        print(f"Covariates after scaling (training): mean = {scaled_data[border1s[0]:border2s[0], covariate_cols].mean(axis=0)[:5]}")
        print(f"Covariates after scaling (validation): mean = {scaled_data[border2s[0]:border2s[0]+10, covariate_cols].mean(axis=0)[:5]}")
        print(f"Covariates after scaling (full): mean = {scaled_data[:, covariate_cols].mean(axis=0)[:5]}")  # Should be ~0
    
    print("\n✓ Selective scaling test completed successfully!")

if __name__ == "__main__":
    test_data_structure()
