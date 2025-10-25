#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

from data_provider.data_factory import data_provider
import argparse

def check_dataset_sizes():
    """Check dataset sizes for both Enhanced and base PGAT configurations"""
    
    # Enhanced PGAT config
    enhanced_args = argparse.Namespace(
        root_path='data',
        data_path='synthetic_multi_wave.csv',
        data='custom',
        features='M',
        target='target_0,target_1,target_2',
        freq='H',
        seq_len=96,
        label_len=48,
        pred_len=24,
        batch_size=8,
        num_workers=1,
        model='Enhanced_SOTA_PGAT'
    )
    
    print("=== Enhanced PGAT Dataset Analysis ===")
    try:
        train_data, train_loader = data_provider(enhanced_args, 'train')
        vali_data, vali_loader = data_provider(enhanced_args, 'val')
        test_data, test_loader = data_provider(enhanced_args, 'test')
        
        print(f"Enhanced Train dataset size: {len(train_data)}")
        print(f"Enhanced Train loader steps: {len(train_loader)}")
        print(f"Enhanced Validation dataset size: {len(vali_data)}")
        print(f"Enhanced Validation loader steps: {len(vali_loader)}")
        print(f"Enhanced Test dataset size: {len(test_data)}")
        print(f"Enhanced Test loader steps: {len(test_loader)}")
        
        # Check first batch
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            print(f"Enhanced First batch shapes: x={batch_x.shape}, y={batch_y.shape}")
            break
            
    except Exception as e:
        print(f"Enhanced dataset error: {e}")
    
    print("\n=== Base PGAT Dataset Analysis ===")
    # Base PGAT config
    base_args = argparse.Namespace(
        root_path='data',
        data_path='synthetic_multi_wave.csv',
        data='custom',
        features='M',
        target='target_0,target_1,target_2',
        freq='H',
        seq_len=96,
        label_len=48,
        pred_len=24,
        batch_size=8,
        num_workers=1,
        model='SOTA_Temporal_PGAT'
    )
    
    try:
        train_data, train_loader = data_provider(base_args, 'train')
        vali_data, vali_loader = data_provider(base_args, 'val')
        test_data, test_loader = data_provider(base_args, 'test')
        
        print(f"Base Train dataset size: {len(train_data)}")
        print(f"Base Train loader steps: {len(train_loader)}")
        print(f"Base Validation dataset size: {len(vali_data)}")
        print(f"Base Validation loader steps: {len(vali_loader)}")
        print(f"Base Test dataset size: {len(test_data)}")
        print(f"Base Test loader steps: {len(test_loader)}")
        
        # Check first batch
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            print(f"Base First batch shapes: x={batch_x.shape}, y={batch_y.shape}")
            break
            
    except Exception as e:
        print(f"Base dataset error: {e}")

if __name__ == "__main__":
    check_dataset_sizes()