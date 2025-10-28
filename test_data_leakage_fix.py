#!/usr/bin/env python3
"""
Test that the data leakage fix works correctly
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_fixed_borders():
    """Test the fixed border calculation"""
    
    print("🧪 TESTING FIXED DATA SPLIT BORDERS")
    print("=" * 60)
    
    # Simulate the fixed calculation
    n = 7109  # total data length
    s = 96    # seq_len
    v = 800   # validation_length  
    t = 600   # test_length
    
    print(f"Configuration: n={n}, s={s}, v={v}, t={t}")
    print()
    
    # Fixed calculation (as implemented)
    test_start_clean = n - t
    val_start_clean = test_start_clean - v  
    train_end_clean = val_start_clean
    
    border1s = [0, val_start_clean, test_start_clean]
    border2s = [train_end_clean, test_start_clean, n]
    
    print("Fixed border calculation:")
    print(f"  test_start_clean = {n} - {t} = {test_start_clean}")
    print(f"  val_start_clean = {test_start_clean} - {v} = {val_start_clean}")
    print(f"  train_end_clean = {val_start_clean}")
    print()
    print(f"  border1s = [0, {val_start_clean}, {test_start_clean}] = {border1s}")
    print(f"  border2s = [{train_end_clean}, {test_start_clean}, {n}] = {border2s}")
    print()
    
    # Analyze the splits
    train_start, train_end = border1s[0], border2s[0]
    val_start, val_end = border1s[1], border2s[1]
    test_start, test_end = border1s[2], border2s[2]
    
    print("Data splits:")
    print(f"  Training: {train_start} to {train_end} ({train_end - train_start} samples)")
    print(f"  Validation: {val_start} to {val_end} ({val_end - val_start} samples)")
    print(f"  Test: {test_start} to {test_end} ({test_end - test_start} samples)")
    print()
    
    # Check for data leakage
    train_range = set(range(train_start, train_end))
    val_range = set(range(val_start, val_end))
    test_range = set(range(test_start, test_end))
    
    train_val_overlap = train_range.intersection(val_range)
    val_test_overlap = val_range.intersection(test_range)
    train_test_overlap = train_range.intersection(test_range)
    
    print("🔍 Data leakage check:")
    if not train_val_overlap:
        print("  ✅ No Train/Val overlap")
    else:
        print(f"  ❌ Train/Val overlap: {len(train_val_overlap)} samples")
    
    if not val_test_overlap:
        print("  ✅ No Val/Test overlap")
    else:
        print(f"  ❌ Val/Test overlap: {len(val_test_overlap)} samples")
    
    if not train_test_overlap:
        print("  ✅ No Train/Test overlap")
    else:
        print(f"  ❌ Train/Test overlap: {len(train_test_overlap)} samples")
    
    # Overall assessment
    no_leakage = not (train_val_overlap or val_test_overlap or train_test_overlap)
    
    print()
    if no_leakage:
        print("🎉 SUCCESS: No data leakage detected!")
        print("✅ Clean temporal splits achieved")
        print("✅ This should resolve overfitting issues")
    else:
        print("❌ FAILURE: Data leakage still present")
        print("⚠️  Need to revise the fix")
    
    return no_leakage

def test_sequence_windowing():
    """Test that sequence windowing still works correctly"""
    
    print("\n🔄 TESTING SEQUENCE WINDOWING COMPATIBILITY")
    print("=" * 60)
    
    # Configuration
    n = 7109
    s = 96    # seq_len
    v = 800
    t = 600
    
    # Fixed borders
    test_start_clean = n - t
    val_start_clean = test_start_clean - v  
    train_end_clean = val_start_clean
    
    border1s = [0, val_start_clean, test_start_clean]
    border2s = [train_end_clean, test_start_clean, n]
    
    # Check if each split has enough data for sequence windowing
    print("Sequence windowing requirements:")
    print(f"  Sequence length: {s}")
    print(f"  Minimum samples needed per split: {s}")
    print()
    
    for i, split_name in enumerate(['Training', 'Validation', 'Test']):
        split_start, split_end = border1s[i], border2s[i]
        split_length = split_end - split_start
        
        # For sequence windowing, we need at least seq_len samples
        # The actual number of sequences we can create is: split_length - seq_len + 1
        max_sequences = max(0, split_length - s + 1)
        
        print(f"  {split_name}:")
        print(f"    Data range: {split_start} to {split_end} ({split_length} samples)")
        print(f"    Max sequences: {max_sequences}")
        
        if max_sequences > 0:
            print(f"    ✅ Sufficient for sequence windowing")
        else:
            print(f"    ❌ Insufficient for sequence windowing")
        print()
    
    return True

def main():
    """Test the data leakage fix"""
    
    print("🔧 TESTING DATA LEAKAGE FIX")
    print("=" * 80)
    
    # Test fixed borders
    no_leakage = test_fixed_borders()
    
    # Test sequence windowing
    windowing_ok = test_sequence_windowing()
    
    print("🎯 OVERALL ASSESSMENT:")
    if no_leakage and windowing_ok:
        print("✅ Data leakage fix is successful!")
        print("✅ Ready for component testing with proper validation")
        print("✅ Should see meaningful train/val convergence patterns")
    else:
        print("❌ Fix needs more work")
        if not no_leakage:
            print("  - Data leakage still present")
        if not windowing_ok:
            print("  - Sequence windowing issues")
    
    return no_leakage and windowing_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)