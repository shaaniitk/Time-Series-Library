#!/usr/bin/env python3
"""
Correctly analyze data leakage - considering what's actually used for loss computation
"""

def analyze_original_borders():
    """Analyze the original border calculation to see if there's ACTUAL data leakage"""
    
    print("🔍 CORRECT DATA LEAKAGE ANALYSIS")
    print("=" * 80)
    
    # Configuration
    n = 7109  # total data length
    s = 96    # seq_len
    v = 800   # validation_length  
    t = 600   # test_length
    
    print(f"Configuration: n={n}, s={s}, v={v}, t={t}")
    print()
    
    # Original (supposedly problematic) calculation
    print("🔍 ORIGINAL BORDER CALCULATION:")
    border1s_orig = [0, n - t - s - v, n - s - t]
    border2s_orig = [n - t - v, n - t, n]
    
    print(f"  border1s = [0, {n}-{t}-{s}-{v}, {n}-{s}-{t}] = {border1s_orig}")
    print(f"  border2s = [{n}-{t}-{v}, {n}-{t}, {n}] = {border2s_orig}")
    print()
    
    # Data ranges
    train_start, train_end = border1s_orig[0], border2s_orig[0]
    val_start, val_end = border1s_orig[1], border2s_orig[1]
    test_start, test_end = border1s_orig[2], border2s_orig[2]
    
    print("📊 DATA RANGES:")
    print(f"  Training: {train_start} to {train_end-1} ({train_end - train_start} samples)")
    print(f"  Validation: {val_start} to {val_end-1} ({val_end - val_start} samples)")
    print(f"  Test: {test_start} to {test_end-1} ({test_end - test_start} samples)")
    print()
    
    # Check overlaps in data ranges
    train_range = set(range(train_start, train_end))
    val_range = set(range(val_start, val_end))
    test_range = set(range(test_start, test_end))
    
    train_val_overlap = train_range.intersection(val_range)
    val_test_overlap = val_range.intersection(test_range)
    
    print("🔍 DATA RANGE OVERLAPS:")
    if train_val_overlap:
        print(f"  Train/Val data overlap: {len(train_val_overlap)} samples ({min(train_val_overlap)}-{max(train_val_overlap)})")
    else:
        print("  ✅ No Train/Val data overlap")
    
    if val_test_overlap:
        print(f"  Val/Test data overlap: {len(val_test_overlap)} samples ({min(val_test_overlap)}-{max(val_test_overlap)})")
    else:
        print("  ✅ No Val/Test data overlap")
    
    print()
    
    # Now analyze what's ACTUALLY used for loss computation
    print("🎯 WHAT'S ACTUALLY USED FOR LOSS COMPUTATION:")
    print()
    
    # Training loss computation
    print("📈 TRAINING LOSS:")
    print("  Training data range: 0 to 5708")
    print("  First training sequence:")
    print(f"    Input: 0 to {s-1} (samples 0-95)")
    print(f"    Loss computed on: {s} to {s+24-1} (samples 96-119)")
    print("  Last training sequence:")
    last_train_seq_start = train_end - s - 24
    print(f"    Input: {last_train_seq_start} to {last_train_seq_start + s - 1}")
    print(f"    Loss computed on: {last_train_seq_start + s} to {last_train_seq_start + s + 24 - 1}")
    print(f"    = Loss on samples {last_train_seq_start + s} to {last_train_seq_start + s + 23}")
    print()
    
    # Validation loss computation  
    print("📊 VALIDATION LOSS:")
    print(f"  Validation data range: {val_start} to {val_end-1}")
    print("  First validation sequence:")
    print(f"    Input: {val_start} to {val_start + s - 1} (samples {val_start}-{val_start + s - 1})")
    print(f"    Loss computed on: {val_start + s} to {val_start + s + 24 - 1} (samples {val_start + s}-{val_start + s + 23})")
    print()
    
    # Check if there's overlap in LOSS COMPUTATION regions
    train_loss_start = s  # 96
    train_loss_end = last_train_seq_start + s + 24  # Last training loss sample
    
    val_loss_start = val_start + s  # 5709 + 96 = 5805
    val_loss_end = val_end  # 6509
    
    print("🔍 LOSS COMPUTATION OVERLAP CHECK:")
    print(f"  Training loss computed on samples: {train_loss_start} to {train_loss_end-1}")
    print(f"  Validation loss computed on samples: {val_loss_start} to {val_loss_end-1}")
    
    loss_overlap = set(range(train_loss_start, train_loss_end)).intersection(
        set(range(val_loss_start, val_loss_end))
    )
    
    if loss_overlap:
        print(f"  ❌ ACTUAL DATA LEAKAGE: {len(loss_overlap)} samples used for both train and val loss!")
        print(f"      Overlap range: {min(loss_overlap)} to {max(loss_overlap)}")
        return True
    else:
        print("  ✅ NO ACTUAL DATA LEAKAGE: Loss computed on different samples")
        return False

def analyze_my_fix():
    """Analyze whether my fix was necessary"""
    
    print("\n🔧 ANALYZING MY 'FIX'")
    print("=" * 60)
    
    # My "fixed" calculation
    n, s, v, t = 7109, 96, 800, 600
    
    test_start_clean = n - t  # 6509
    val_start_clean = test_start_clean - v  # 5709
    train_end_clean = val_start_clean  # 5709
    
    border1s_fixed = [0, val_start_clean, test_start_clean]
    border2s_fixed = [train_end_clean, test_start_clean, n]
    
    print("My 'fixed' borders:")
    print(f"  border1s = {border1s_fixed}")
    print(f"  border2s = {border2s_fixed}")
    print()
    
    print("Comparison with original:")
    border1s_orig = [0, n - t - s - v, n - s - t]
    border2s_orig = [n - t - v, n - t, n]
    
    print(f"  Original: border1s = {border1s_orig}, border2s = {border2s_orig}")
    print(f"  My fix:   border1s = {border1s_fixed}, border2s = {border2s_fixed}")
    print()
    
    # Check if they're actually the same for what matters
    orig_val_loss_start = border1s_orig[1] + s  # 5613 + 96 = 5709
    orig_val_loss_end = border2s_orig[1]        # 6509
    
    fixed_val_loss_start = border1s_fixed[1] + s  # 5709 + 96 = 5805
    fixed_val_loss_end = border2s_fixed[1]        # 6509
    
    print("Validation loss computation regions:")
    print(f"  Original: samples {orig_val_loss_start} to {orig_val_loss_end-1}")
    print(f"  My fix:   samples {fixed_val_loss_start} to {fixed_val_loss_end-1}")
    
    if orig_val_loss_start == fixed_val_loss_start and orig_val_loss_end == fixed_val_loss_end:
        print("  ✅ SAME VALIDATION LOSS REGION - My fix changed nothing meaningful!")
    else:
        print("  ⚠️  Different validation loss regions")

def main():
    """Main analysis"""
    
    print("🤔 WAS MY DATA LEAKAGE FIX WRONG?")
    print("=" * 80)
    
    # Analyze original borders correctly
    has_actual_leakage = analyze_original_borders()
    
    # Analyze my fix
    analyze_my_fix()
    
    print("\n🎯 CONCLUSION:")
    if not has_actual_leakage:
        print("✅ YOU ARE CORRECT!")
        print("  - The original borders were NOT causing data leakage")
        print("  - Overlapping data ranges are OK if they're only used as INPUT")
        print("  - Only LOSS COMPUTATION overlap matters")
        print("  - My 'fix' was unnecessary and potentially wrong")
        print()
        print("🔍 The real issue might be:")
        print("  - Model complexity (which we did address)")
        print("  - Training hyperparameters")
        print("  - Other overfitting causes")
        print("  - NOT data leakage")
    else:
        print("❌ There IS actual data leakage in loss computation")
        print("  - My fix was correct")

if __name__ == "__main__":
    main()