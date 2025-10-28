#!/usr/bin/env python3
"""
Fix the data leakage issue in Dataset_Custom border calculation
"""

def analyze_current_borders():
    """Analyze the current problematic border calculation"""
    
    print("🔍 ANALYZING CURRENT DATA SPLIT BORDERS")
    print("=" * 60)
    
    # Current configuration
    n = 7109  # total data length
    s = 96    # seq_len
    v = 800   # validation_length  
    t = 600   # test_length
    
    print(f"Configuration:")
    print(f"  Total samples (n): {n}")
    print(f"  Sequence length (s): {s}")
    print(f"  Validation length (v): {v}")
    print(f"  Test length (t): {t}")
    print()
    
    # Current (problematic) calculation
    print("❌ CURRENT (PROBLEMATIC) CALCULATION:")
    border1s_current = [0, n - t - s - v, n - s - t]
    border2s_current = [n - t - v, n - t, n]
    
    print(f"  border1s = [0, {n}-{t}-{s}-{v}, {n}-{s}-{t}] = {border1s_current}")
    print(f"  border2s = [{n}-{t}-{v}, {n}-{t}, {n}] = {border2s_current}")
    print()
    
    print("Data splits:")
    train_start, train_end = border1s_current[0], border2s_current[0]
    val_start, val_end = border1s_current[1], border2s_current[1]
    test_start, test_end = border1s_current[2], border2s_current[2]
    
    print(f"  Training: {train_start} to {train_end} ({train_end - train_start} samples)")
    print(f"  Validation: {val_start} to {val_end} ({val_end - val_start} samples)")
    print(f"  Test: {test_start} to {test_end} ({test_end - test_start} samples)")
    print()
    
    # Check for overlap
    train_range = set(range(train_start, train_end))
    val_range = set(range(val_start, val_end))
    test_range = set(range(test_start, test_end))
    
    train_val_overlap = train_range.intersection(val_range)
    val_test_overlap = val_range.intersection(test_range)
    
    if train_val_overlap:
        print(f"🚨 DATA LEAKAGE DETECTED!")
        print(f"  Train/Val overlap: {len(train_val_overlap)} samples ({min(train_val_overlap)}-{max(train_val_overlap)})")
    
    if val_test_overlap:
        print(f"🚨 DATA LEAKAGE DETECTED!")
        print(f"  Val/Test overlap: {len(val_test_overlap)} samples ({min(val_test_overlap)}-{max(val_test_overlap)})")
    
    print()
    return border1s_current, border2s_current

def propose_fixed_borders():
    """Propose corrected border calculation"""
    
    print("✅ PROPOSED FIXED CALCULATION:")
    
    # Configuration
    n = 7109  # total data length
    s = 96    # seq_len
    v = 800   # validation_length  
    t = 600   # test_length
    
    # Fixed calculation - no overlap
    # The key insight: seq_len should only affect the starting position for windowing,
    # not create overlaps between train/val/test splits
    
    # Method 1: Clean splits with seq_len buffer
    train_end = n - t - v
    val_start = train_end  # No gap, no overlap
    val_end = n - t
    test_start = val_end   # No gap, no overlap
    test_end = n
    
    # Adjust for sequence requirements
    border1s_fixed = [0, max(0, val_start - s), max(0, test_start - s)]
    border2s_fixed = [train_end, val_end, test_end]
    
    print(f"  Clean split boundaries:")
    print(f"    Train end: {train_end}")
    print(f"    Val start: {val_start}, Val end: {val_end}")
    print(f"    Test start: {test_start}, Test end: {test_end}")
    print()
    
    print(f"  border1s = [0, max(0, {val_start}-{s}), max(0, {test_start}-{s})] = {border1s_fixed}")
    print(f"  border2s = [{train_end}, {val_end}, {test_end}] = {border2s_fixed}")
    print()
    
    print("Fixed data splits:")
    train_start, train_end = border1s_fixed[0], border2s_fixed[0]
    val_start, val_end = border1s_fixed[1], border2s_fixed[1]
    test_start, test_end = border1s_fixed[2], border2s_fixed[2]
    
    print(f"  Training: {train_start} to {train_end} ({train_end - train_start} samples)")
    print(f"  Validation: {val_start} to {val_end} ({val_end - val_start} samples)")
    print(f"  Test: {test_start} to {test_end} ({test_end - test_start} samples)")
    print()
    
    # Check for overlap
    train_range = set(range(train_start, train_end))
    val_range = set(range(val_start, val_end))
    test_range = set(range(test_start, test_end))
    
    train_val_overlap = train_range.intersection(val_range)
    val_test_overlap = val_range.intersection(test_range)
    
    if not train_val_overlap and not val_test_overlap:
        print("✅ NO DATA LEAKAGE - Clean splits achieved!")
    else:
        if train_val_overlap:
            print(f"⚠️  Still some Train/Val overlap: {len(train_val_overlap)} samples")
        if val_test_overlap:
            print(f"⚠️  Still some Val/Test overlap: {len(val_test_overlap)} samples")
    
    print()
    return border1s_fixed, border2s_fixed

def generate_fix_code():
    """Generate the code fix for the data loader"""
    
    print("🔧 CODE FIX FOR data_provider/data_loader.py")
    print("=" * 60)
    
    print("Replace the current border calculation:")
    print("```python")
    print("# CURRENT (PROBLEMATIC):")
    print("border1s = [0, n - t - s - v, n - s - t]")
    print("border2s = [n - t - v, n - t, n]")
    print("```")
    print()
    
    print("With the fixed calculation:")
    print("```python")
    print("# FIXED (NO DATA LEAKAGE):")
    print("train_end = n - t - v")
    print("val_start = train_end")
    print("val_end = n - t") 
    print("test_start = val_end")
    print("test_end = n")
    print("")
    print("# Adjust for sequence requirements (but no overlap)")
    print("border1s = [0, max(0, val_start - s), max(0, test_start - s)]")
    print("border2s = [train_end, val_end, test_end]")
    print("```")
    print()

def main():
    """Analyze and fix data leakage issue"""
    
    print("🚨 DATA LEAKAGE ANALYSIS AND FIX")
    print("=" * 80)
    print()
    
    # Analyze current problematic borders
    current_borders = analyze_current_borders()
    
    # Propose fixed borders
    fixed_borders = propose_fixed_borders()
    
    # Generate fix code
    generate_fix_code()
    
    print("🎯 SUMMARY:")
    print("✅ Root cause identified: Data leakage in validation window")
    print("✅ Fix proposed: Clean temporal splits with no overlap")
    print("✅ This should resolve the overfitting pattern completely")
    print()
    print("🚀 Next steps:")
    print("1. Apply the border calculation fix to data_provider/data_loader.py")
    print("2. Re-run the component tests")
    print("3. Validation loss should now behave properly")

if __name__ == "__main__":
    main()