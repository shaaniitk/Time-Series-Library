#!/usr/bin/env python3

import numpy as np
import sys
sys.path.append('.')

# Quick syntax check
try:
    from data_provider.data_loader import Dataset_Custom
    print("✅ Import successful - no syntax errors")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# Quick inverse transform test
try:
    import argparse
    
    args = argparse.Namespace()
    args.data = 'custom'
    args.validation_length = 150
    args.test_length = 50
    args.augmentation_ratio = 0
    
    dataset = Dataset_Custom(
        args=args,
        root_path='.',
        flag='train',
        size=[96, 48, 24],
        features='M',
        data_path='data/prepared_financial_data.csv',
        target='log_Open,log_High,log_Low,log_Close',
        scale=True,
        timeenc=0,
        freq='h'
    )
    
    print("✅ Dataset initialization successful")
    print(f"Has target_scaler: {hasattr(dataset, 'target_scaler')}")
    print(f"Has covariate_scaler: {hasattr(dataset, 'covariate_scaler')}")
    
    # Test inverse transform on sample data
    sample_targets = np.array([[0.5, -0.3, 0.8, -0.1]])
    result = dataset.inverse_transform(sample_targets)
    print(f"✅ Inverse transform successful: {result.shape}")
    
    # Test convenience method
    result2 = dataset.inverse_transform_targets(sample_targets)
    print(f"✅ Target-specific inverse transform successful: {result2.shape}")
    
    print("🎉 All tests passed!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
