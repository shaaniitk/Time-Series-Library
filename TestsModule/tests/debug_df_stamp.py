#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append('.')

def debug_df_stamp():
    """Debug the df_stamp issue"""
    
    # Read the data just like the data loader does
    df_raw = pd.read_csv('data/prepared_financial_data.csv')
    print(f"df_raw shape: {df_raw.shape}")
    print(f"df_raw columns: {df_raw.columns[:6].tolist()}")
    
    # Use borders like the test
    data_len = len(df_raw)
    n = data_len
    s = 96  # seq_len 
    v = 150  # validation_length
    t = 50   # test_length
    
    border1s = [0, n - t - s - v, n - s - t]
    border2s = [n - t - v, n - t, n]
    
    print(f"border1s: {border1s}")
    print(f"border2s: {border2s}")
    
    # Test for training set (set_type = 0)
    border1 = border1s[0]  # 0
    border2 = border2s[0]  # 6909
    
    print(f"Training: border1={border1}, border2={border2}")
    
    # Try to create df_stamp
    print(f"df_raw[['date']] shape: {df_raw[['date']].shape}")
    df_stamp = df_raw[['date']][border1:border2]
    print(f"df_stamp shape after slicing: {df_stamp.shape}")
    print(f"df_stamp columns: {df_stamp.columns.tolist()}")
    print(f"df_stamp head:\n{df_stamp.head()}")
    
    # Try pd.to_datetime
    df_stamp['date'] = pd.to_datetime(df_stamp.date)
    print(f"After pd.to_datetime, df_stamp dtypes:\n{df_stamp.dtypes}")
    
    # Add time features
    df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
    df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
    print(f"After adding time features, df_stamp columns: {df_stamp.columns.tolist()}")
    print(f"df_stamp shape: {df_stamp.shape}")
    
    # Try the drop operation
    try:
        data_stamp = df_stamp.drop(['date'], 1).values
        print(f" Drop operation successful, data_stamp shape: {data_stamp.shape}")
    except Exception as e:
        print(f" Drop operation failed: {e}")
        print(f"df_stamp info:\n{df_stamp.info()}")

if __name__ == "__main__":
    debug_df_stamp()
