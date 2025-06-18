#!/usr/bin/env python3
"""
Future Covariates Integration Utility

This script shows how to integrate 100-year future covariate data
into the Time Series Library framework.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def integrate_future_covariates(main_data_path, covariates_path, output_path):
    """
    Integrate future covariates with main time series data.
    
    Args:
        main_data_path: Path to your main financial data (prepared_financial_data.csv)
        covariates_path: Path to your 100-year covariate data
        output_path: Path for output file with integrated data
    """
    
    print("🔄 Loading main financial data...")
    df_main = pd.read_csv(main_data_path)
    df_main['date'] = pd.to_datetime(df_main['date'])
    print(f"   Main data shape: {df_main.shape}")
    print(f"   Date range: {df_main['date'].min()} to {df_main['date'].max()}")
    
    print("\n🔄 Loading future covariates...")
    df_covariates = pd.read_csv(covariates_path)
    df_covariates['date'] = pd.to_datetime(df_covariates['date'])
    print(f"   Covariate data shape: {df_covariates.shape}")
    print(f"   Date range: {df_covariates['date'].min()} to {df_covariates['date'].max()}")
    print(f"   Covariate columns: {df_covariates.columns.tolist()}")
    
    # Ensure we have sufficient future data for forecasting
    main_end_date = df_main['date'].max()
    covariate_end_date = df_covariates['date'].max()
    
    print(f"\n📅 Date Analysis:")
    print(f"   Main data ends: {main_end_date}")
    print(f"   Covariates end: {covariate_end_date}")
    print(f"   Future coverage: {(covariate_end_date - main_end_date).days} days")
    
    if covariate_end_date <= main_end_date:
        print("⚠️  WARNING: Covariate data doesn't extend beyond main data!")
        print("   For true forecasting, you need future covariate values.")
    
    print("\n🔗 Merging data...")
    # Merge on date - this will include future covariate values
    df_merged = pd.merge(df_main, df_covariates, on='date', how='left')
    
    print(f"   Merged data shape: {df_merged.shape}")
    print(f"   Missing covariate values: {df_merged.isnull().sum().sum()}")
    
    # Fill any missing covariate values (forward fill for future dates)
    covariate_cols = [col for col in df_covariates.columns if col != 'date']
    df_merged[covariate_cols] = df_merged[covariate_cols].fillna(method='ffill')
    
    print("\n💾 Saving integrated data...")
    df_merged.to_csv(output_path, index=False)
    print(f"   Saved to: {output_path}")
    
    print("\n✅ Integration Summary:")
    print(f"   Total features: {len(df_merged.columns) - 1}")  # -1 for date
    print(f"   Financial features: {len(df_main.columns) - 1}")
    print(f"   Covariate features: {len(covariate_cols)}")
    print(f"   Data points: {len(df_merged)}")
    
    return df_merged


def create_extended_dataset_with_future_predictions(main_data_path, covariates_path, 
                                                   prediction_horizon_days=365):
    """
    Create a dataset that extends beyond historical data for true future forecasting.
    
    This creates a dataset where:
    - Historical period: Your financial data
    - Future period: Only covariates (for forecasting)
    """
    
    print("🚀 Creating Extended Dataset for Future Forecasting...")
    
    # Load data
    df_main = pd.read_csv(main_data_path)
    df_main['date'] = pd.to_datetime(df_main['date'])
    
    df_covariates = pd.read_csv(covariates_path)
    df_covariates['date'] = pd.to_datetime(df_covariates['date'])
    
    # Get last date of financial data
    last_date = df_main['date'].max()
    
    # Create future dates
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=prediction_horizon_days,
        freq='D'
    )
    
    print(f"   Adding {prediction_horizon_days} future days")
    print(f"   Future period: {future_dates[0]} to {future_dates[-1]}")
    
    # Get future covariates
    df_future_covariates = df_covariates[
        df_covariates['date'].isin(future_dates)
    ].copy()
    
    if len(df_future_covariates) == 0:
        print("⚠️  No future covariate data found for specified dates!")
        return None
    
    # Create future rows with NaN for financial features
    financial_cols = [col for col in df_main.columns if col != 'date']
    covariate_cols = [col for col in df_covariates.columns if col != 'date']
    
    # Future data: NaN for financial features, actual values for covariates
    future_data = []
    for date in future_dates:
        row = {'date': date}
        
        # Financial features = NaN (unknown future)
        for col in financial_cols:
            row[col] = np.nan
        
        # Covariate features = actual values (known future)
        cov_row = df_covariates[df_covariates['date'] == date]
        if len(cov_row) > 0:
            for col in covariate_cols:
                row[col] = cov_row[col].iloc[0]
        else:
            # Fill with last known values if no exact match
            for col in covariate_cols:
                row[col] = df_covariates[df_covariates['date'] <= date][col].iloc[-1]
        
        future_data.append(row)
    
    df_future = pd.DataFrame(future_data)
    
    # Merge main data with covariates
    df_historical = pd.merge(df_main, df_covariates, on='date', how='left')
    
    # Combine historical and future
    df_extended = pd.concat([df_historical, df_future], ignore_index=True)
    
    print(f"\n✅ Extended Dataset Created:")
    print(f"   Historical data points: {len(df_historical)}")
    print(f"   Future data points: {len(df_future)}")
    print(f"   Total data points: {len(df_extended)}")
    print(f"   Total features: {len(df_extended.columns) - 1}")
    
    return df_extended


def analyze_covariate_data(covariates_path):
    """
    Analyze your 100-year covariate data to understand its structure.
    """
    print("🔍 Analyzing Covariate Data...")
    
    df = pd.read_csv(covariates_path)
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"   Total days: {(df['date'].max() - df['date'].min()).days}")
    
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Analyze each covariate
    covariate_cols = [col for col in df.columns if col != 'date']
    print(f"\n📊 Covariate Analysis:")
    
    for col in covariate_cols[:10]:  # Show first 10
        print(f"   {col}:")
        print(f"     Type: {df[col].dtype}")
        print(f"     Range: {df[col].min()} to {df[col].max()}")
        print(f"     Missing: {df[col].isnull().sum()}")
        print(f"     Unique values: {df[col].nunique()}")
    
    if len(covariate_cols) > 10:
        print(f"   ... and {len(covariate_cols) - 10} more columns")
    
    return df


# Example usage
if __name__ == "__main__":
    # Paths (adjust these to your actual file paths)
    main_data = "data/prepared_financial_data.csv"
    covariates_data = "data/your_100_year_covariates.csv"  # ← Your covariate file
    output_file = "data/integrated_data_with_covariates.csv"
    
    # Step 1: Analyze your covariate data
    print("=" * 60)
    if os.path.exists(covariates_data):
        analyze_covariate_data(covariates_data)
    else:
        print(f"Covariate file not found: {covariates_data}")
        print("Please update the path to your 100-year covariate data")
    
    # Step 2: Integrate covariates with financial data
    print("\n" + "=" * 60)
    if os.path.exists(main_data) and os.path.exists(covariates_data):
        integrated_data = integrate_future_covariates(
            main_data, covariates_data, output_file
        )
        
        # Step 3: Create extended dataset for true forecasting
        extended_data = create_extended_dataset_with_future_predictions(
            main_data, covariates_data, prediction_horizon_days=365
        )
        
        if extended_data is not None:
            extended_output = "data/extended_data_with_future_covariates.csv"
            extended_data.to_csv(extended_output, index=False)
            print(f"\n💾 Extended dataset saved to: {extended_output}")
    
    print("\n🎯 Next Steps:")
    print("1. Update your config file to use the integrated data:")
    print(f"   data_path: '{output_file}'")
    print("2. The model will automatically use future covariates via batch_y_mark")
    print("3. For true forecasting, use the extended dataset")
