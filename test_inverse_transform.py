import sys
import os
import numpy as np
from data_provider.data_loader import Dataset_Custom

def test_inverse_transform():
    """Test the inverse transform functionality with correct shapes"""
    
    # Create args for Dataset_Custom
    class TestArgs:
        def __init__(self):
            self.root_path = 'data'
            self.data_path = 'prepared_financial_data.csv'
            self.features = 'M'
            self.target = 'log_Close'
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 24
            self.freq = 'b'
            
    args = TestArgs()
    
    # Initialize dataset
    print("Initializing dataset...")
    dataset = Dataset_Custom(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag='train',
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        scale=True,
        timeenc=1,
        freq=args.freq
    )
    
    print("Dataset initialized successfully")
    print(f"Data shape: {dataset.data_x.shape}")
    print(f"Has target_scaler: {hasattr(dataset, 'target_scaler')}")
    print(f"Has covariate_scaler: {hasattr(dataset, 'covariate_scaler')}")
    
    if hasattr(dataset, 'covariate_scaler'):
        print(f"Covariate scaler expects: {dataset.covariate_scaler.n_features_in_} features")
    
    # --- Test 1: Target-only inverse transform ---
    print("\n--- Test 1: Target-only inverse transform ---")
    scaled_targets = np.array([
        [0.5, -0.3, 0.8, -0.1],
        [0.2, 0.4, -0.6, 0.9], 
        [-0.3, 0.7, 0.1, -0.8]
    ])
    print(f"Scaled targets shape: {scaled_targets.shape}")
    print("Scaled targets:")
    print(scaled_targets)
    
    unscaled_targets = dataset.inverse_transform(scaled_targets)
    print("Unscaled targets:")
    print(unscaled_targets)
    
    # --- Test 2: Target-specific inverse transform ---
    print("\n--- Test 2: Target-specific inverse transform ---")
    unscaled_targets_method2 = dataset.inverse_transform_targets(scaled_targets)
    print("Targets via convenience method:")
    print(unscaled_targets_method2)
    print(f"Methods give same result: {np.allclose(unscaled_targets, unscaled_targets_method2)}")
    
    # --- Test 3: Full data inverse transform ---
    print("\n--- Test 3: Full data inverse transform ---")
    
    # Get the correct number of features from the scaler
    if hasattr(dataset, 'covariate_scaler'):
        num_covariates = dataset.covariate_scaler.n_features_in_
        print(f"Number of covariates from scaler: {num_covariates}")
        
        # Create test data with correct shape: [samples, 4_targets + num_covariates]
        scaled_targets_test = np.array([
            [0.5, -0.3, 0.8, -0.1],
            [0.2, 0.4, -0.6, 0.9], 
            [-0.3, 0.7, 0.1, -0.8]
        ])
        scaled_covariates_test = np.random.randn(3, num_covariates)
        scaled_full_data = np.concatenate([scaled_targets_test, scaled_covariates_test], axis=1)
        
        print(f"Scaled targets shape: {scaled_targets_test.shape}")
        print(f"Scaled covariates shape: {scaled_covariates_test.shape}")
        print(f"Scaled full data shape: {scaled_full_data.shape}")
        
        unscaled_full_data = dataset.inverse_transform(scaled_full_data)
        print(f"Unscaled full data shape: {unscaled_full_data.shape}")
        print("First 4 columns (targets) from full inverse transform:")
        print(unscaled_full_data[:, :4])
        
        # Verify targets are consistent across methods
        targets_from_full = unscaled_full_data[:, :4]
        print(f"Target consistency across methods: {np.allclose(unscaled_targets, targets_from_full)}")
    else:
        print("No covariate scaler found - skipping full data test")
    
    print("\n✅ All inverse transform tests completed successfully!")

if __name__ == "__main__":
    test_inverse_transform()