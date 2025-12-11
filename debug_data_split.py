
import sys
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Add root to path
sys.path.append(os.getcwd())

from data_provider.data_factory import setup_financial_forecasting_data
from data_provider.data_loader import ForecastingDataset

class MockArgs:
    def __init__(self):
        self.root_path = './data'
        self.data_path = 'prepared_financial_data.csv'
        self.task_name = 'long_term_forecast'
        self.features = 'MS'
        self.target = 'log_Open,log_High,log_Low,log_Close'
        self.freq = 'd'
        self.seq_len = 750
        self.label_len = 48
        self.pred_len = 20
        self.validation_length = 1000
        self.test_length = 1000
        self.batch_size = 4
        self.num_workers = 0
        self.scale = True
        self.loss = 'mixture' 
        self.quantile_levels = []
        self.use_future_celestial_conditioning = True
        self.scaler_type = 'robust'

def check_splits():
    print("🔍 AUDITING DATA SPLITS & SCALING")
    print("=" * 50)
    
    args = MockArgs()
    
    # 1. Load Data using Factory
    print("Loading data via factory...")
    try:
        train_loader, vali_loader, test_loader, scaler_manager, dim_manager = setup_financial_forecasting_data(args)
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    train_set = train_loader.dataset
    vali_set = vali_loader.dataset
    test_set = test_loader.dataset

    print(f"\n📊 Dataset Sizes (Windows):")
    print(f"  Train: {len(train_set)}")
    print(f"  Val:   {len(vali_set)}")
    print(f"  Test:  {len(test_set)}")

    # 2. Check Overlap Integrity
    # Use internal raw data to check boundaries
    # data_x is scaled data
    train_raw = train_set.data_x
    vali_raw = vali_set.data_x
    
    print(f"\n📏 Raw Data Dimensions (Time Steps):")
    print(f"  Train Raw: {train_raw.shape}")
    print(f"  Val Raw:   {vali_raw.shape}")
    
    # Check if Val starts with Train's tail
    overlap_len = args.seq_len
    
    # The last 'overlap_len' of Train should match the first 'overlap_len' of Val
    # IF the split logic works as intended (Val = Train[-overlap:] + NewVal)
    
    # Extract
    train_tail = train_raw[-overlap_len:]
    vali_head = vali_raw[:overlap_len]
    
    print(f"\n🧬 Overlap Verification (Last {overlap_len} of Train vs First {overlap_len} of Val):")
    
    # Compare with tolerance
    diff = np.abs(train_tail - vali_head)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    if max_diff < 1e-6:
        print(f"  ✅ PERFECT MATCH. Overlap logic is working.")
        print(f"  (Max diff: {max_diff})")
    else:
        print(f"  ❌ MISMATCH DETECTED!")
        print(f"  Max Diff: {max_diff}")
        print(f"  Mean Diff: {mean_diff}")
        print("  This suggests Val dataset is NOT simply starting with Train history.")
        print("  It might be starting AFTER train, which implies GAP if overlap was intended.")

    # 3. Scaler Statistics Check
    # Check bounds of val data relative to train data
    print(f"\n⚖️  Scaling Statistics:")
    
    train_min = np.min(train_raw, axis=0)
    train_max = np.max(train_raw, axis=0)
    val_min = np.min(vali_raw, axis=0)
    val_max = np.max(vali_raw, axis=0)
    
    print(f"  Feature 0 (e.g. log_Open) Stats:")
    print(f"    Train Range: [{train_min[0]:.4f}, {train_max[0]:.4f}]")
    print(f"    Val Range:   [{val_min[0]:.4f}, {val_max[0]:.4f}]")
    
    # Out of bounds check
    oob_min_mask = val_min < (train_min - 0.5)
    oob_max_mask = val_max > (train_max + 0.5)
    
    if np.any(oob_min_mask) or np.any(oob_max_mask):
        print("  ⚠️  WARNING: Validation data significantly exceeds Training bounds!")
        
        # Identify indices
        oob_indices = np.where(oob_min_mask | oob_max_mask)[0]
        print(f"      Out of bounds features ({len(oob_indices)}): {oob_indices}")
        
        # Check if targets are affected (first 4 features)
        target_indices = [0, 1, 2, 3]
        affected_targets = [i for i in target_indices if i in oob_indices]
        if affected_targets:
             print(f"      🚨 CRITICAL: Target features {affected_targets} are out of bounds!")
        else:
             print(f"      (Target features are within bounds)")
             
        # Print sample detail for first OOB
        idx = oob_indices[0]
        print(f"      Example Feature {idx}: Train [{train_min[idx]:.4f}, {train_max[idx]:.4f}] vs Val [{val_min[idx]:.4f}, {val_max[idx]:.4f}]")
        
    else:
        print("  ✅ Validation data fits reasonably within training bounds.")
        
    print("\n🏁 Audit Complete.")

if __name__ == "__main__":
    check_splits()
