import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def create_synthetic_dataset(n_samples=1000, n_features=7, filename='test_data.csv'):
    """
    Create a synthetic time series dataset for testing.
    
    Args:
        n_samples: Number of time steps
        n_features: Number of features (default 7 to match ETTh1)
        filename: Output filename
    """
    # Create date range
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(n_samples)]
    
    # Generate synthetic time series data
    np.random.seed(42)  # For reproducibility
    
    data = {}
    data['date'] = dates
    
    # Generate features with different patterns
    for i in range(n_features):
        # Base trend
        trend = np.linspace(0, 10, n_samples) + np.random.normal(0, 0.5, n_samples)
        
        # Seasonal component (daily and weekly patterns)
        daily_pattern = 2 * np.sin(2 * np.pi * np.arange(n_samples) / 24)
        weekly_pattern = 1.5 * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 7))
        
        # Random noise
        noise = np.random.normal(0, 1, n_samples)
        
        # Combine components
        feature_data = trend + daily_pattern + weekly_pattern + noise
        
        # Scale to reasonable range
        feature_data = (feature_data - feature_data.min()) / (feature_data.max() - feature_data.min()) * 100
        
        data[f'feature_{i}'] = feature_data
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create output directory if it doesn't exist
    os.makedirs('dataset/test', exist_ok=True)
    
    # Save to CSV
    output_path = f'dataset/test/{filename}'
    df.to_csv(output_path, index=False)
    
    print(f"Created synthetic dataset: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    return output_path

if __name__ == '__main__':
    # Create test dataset
    dataset_path = create_synthetic_dataset()
    print(f"\nDataset created successfully at: {dataset_path}")