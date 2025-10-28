#!/usr/bin/env python3
"""
Analyze how validation loss is computed with sequence windowing
"""

def analyze_validation_computation():
    """Analyze the validation loss computation methodology"""
    
    print("🔍 VALIDATION LOSS COMPUTATION ANALYSIS")
    print("=" * 80)
    
    # Current configuration
    total_samples = 7109
    seq_len = 96      # 96 days input sequence
    label_len = 48    # 48 days overlap
    pred_len = 24     # 24 days prediction
    validation_length = 800
    batch_size = 16
    
    print("📊 CONFIGURATION:")
    print(f"  Total dataset: {total_samples} samples")
    print(f"  Sequence length (input): {seq_len} days")
    print(f"  Label length (overlap): {label_len} days") 
    print(f"  Prediction length: {pred_len} days")
    print(f"  Validation length: {validation_length} samples")
    print(f"  Batch size: {batch_size}")
    print()
    
    # Data splits (after our fix)
    test_start = total_samples - 600  # 6509
    val_start = test_start - validation_length  # 5709
    train_end = val_start  # 5709
    
    print("🗂️  DATA SPLITS (FIXED - NO LEAKAGE):")
    print(f"  Training: 0 to {train_end} ({train_end} samples)")
    print(f"  Validation: {val_start} to {test_start} ({validation_length} samples)")
    print(f"  Test: {test_start} to {total_samples} ({total_samples - test_start} samples)")
    print()
    
    # Validation sequence windowing
    print("🔄 VALIDATION SEQUENCE WINDOWING:")
    print("Each validation sequence works as follows:")
    print(f"  Input (seq_x): {seq_len} days of history")
    print(f"  Target (seq_y): {label_len + pred_len} days total")
    print(f"    - First {label_len} days: overlap with input (for decoder)")
    print(f"    - Last {pred_len} days: actual prediction target")
    print()
    
    # Calculate number of validation sequences
    # From Dataset_Custom.__len__(): len(data_x) - seq_len - pred_len + 1
    val_data_length = validation_length  # 800 samples in validation set
    num_val_sequences = val_data_length - seq_len - pred_len + 1
    
    print("📈 VALIDATION SEQUENCES:")
    print(f"  Validation data length: {val_data_length} samples")
    print(f"  Number of sequences: {val_data_length} - {seq_len} - {pred_len} + 1 = {num_val_sequences}")
    print()
    
    if num_val_sequences <= 0:
        print("❌ ERROR: Not enough validation data for sequence windowing!")
        return
    
    # Show example sequences
    print("📝 EXAMPLE VALIDATION SEQUENCES:")
    print("(Relative to validation data start)")
    
    for i in [0, 1, num_val_sequences//2, num_val_sequences-1]:
        if i >= num_val_sequences:
            continue
            
        s_begin = i
        s_end = s_begin + seq_len
        r_begin = s_end - label_len
        r_end = r_begin + label_len + pred_len
        
        print(f"  Sequence {i}:")
        print(f"    Input (seq_x): samples {s_begin} to {s_end-1} ({seq_len} days)")
        print(f"    Target (seq_y): samples {r_begin} to {r_end-1} ({label_len + pred_len} days)")
        print(f"    Prediction target: samples {s_end} to {s_end + pred_len - 1} ({pred_len} days)")
        print()
    
    # Batch computation
    num_batches = (num_val_sequences + batch_size - 1) // batch_size  # Ceiling division
    
    print("🔢 BATCH COMPUTATION:")
    print(f"  Total sequences: {num_val_sequences}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of batches: {num_batches}")
    print(f"  Last batch size: {num_val_sequences % batch_size if num_val_sequences % batch_size != 0 else batch_size}")
    print()
    
    # Loss computation process
    print("⚖️  LOSS COMPUTATION PROCESS:")
    print("For each validation batch:")
    print(f"  1. Model receives {seq_len} days of input features")
    print(f"  2. Model predicts next {pred_len} days")
    print(f"  3. Loss computed between predicted vs actual {pred_len} days")
    print(f"  4. Loss accumulated across all {num_val_sequences} sequences")
    print(f"  5. Final validation loss = average loss across all sequences")
    print()
    
    print("🎯 KEY INSIGHTS:")
    print(f"  ✅ Validation is NOT predicting {validation_length} consecutive days")
    print(f"  ✅ Instead: {num_val_sequences} separate {pred_len}-day predictions")
    print(f"  ✅ Each prediction uses {seq_len} days of history")
    print(f"  ✅ Validation loss = average of {num_val_sequences} prediction losses")
    print(f"  ✅ This measures model's ability to predict {pred_len} days ahead")
    print()
    
    # Temporal coverage
    first_input_end = seq_len - 1
    last_prediction_start = num_val_sequences - 1 + seq_len
    last_prediction_end = last_prediction_start + pred_len - 1
    
    print("📅 TEMPORAL COVERAGE:")
    print(f"  First input covers: day 0 to {first_input_end}")
    print(f"  Last prediction covers: day {last_prediction_start} to {last_prediction_end}")
    print(f"  Total temporal span: {last_prediction_end + 1} days")
    print(f"  Validation data span: {validation_length} days")
    
    if last_prediction_end + 1 <= validation_length:
        print("  ✅ All predictions fit within validation data")
    else:
        print("  ❌ Predictions extend beyond validation data!")
    
    return num_val_sequences, num_batches

def compare_with_training_log():
    """Compare analysis with actual training log"""
    
    print("\n🔍 COMPARISON WITH TRAINING LOG")
    print("=" * 60)
    
    # From training_diagnostic.log
    observed_val_batches = 30
    observed_avg_val_loss = 1.17251066
    
    print("📋 OBSERVED FROM TRAINING LOG:")
    print(f"  val_batches: {observed_val_batches}")
    print(f"  avg_val_loss: {observed_avg_val_loss}")
    print()
    
    # Our calculation
    calculated_sequences, calculated_batches = analyze_validation_computation()
    
    print("🧮 CALCULATED:")
    print(f"  Expected sequences: {calculated_sequences}")
    print(f"  Expected batches: {calculated_batches}")
    print()
    
    print("🔍 ANALYSIS:")
    if calculated_batches == observed_val_batches:
        print("  ✅ Batch count matches - calculation correct!")
    else:
        print(f"  ⚠️  Batch count mismatch:")
        print(f"    Expected: {calculated_batches}")
        print(f"    Observed: {observed_val_batches}")
        print(f"    Possible reasons:")
        print(f"      - Different batch size during validation")
        print(f"      - Different validation_length in actual run")
        print(f"      - Data loading configuration differences")

def main():
    """Main analysis function"""
    
    print("🔬 VALIDATION LOSS COMPUTATION ANALYSIS")
    print("=" * 80)
    print()
    
    # Run main analysis
    analyze_validation_computation()
    
    # Compare with training log
    compare_with_training_log()
    
    print("\n🎯 CONCLUSION:")
    print("The validation loss is computed by:")
    print("1. Creating ~681 overlapping 24-day prediction tasks")
    print("2. Each task uses 96 days of history to predict 24 days ahead")
    print("3. Computing MSE loss for each 24-day prediction")
    print("4. Averaging the loss across all prediction tasks")
    print()
    print("This is a proper evaluation methodology that measures")
    print("the model's ability to make 24-day forecasts, not")
    print("consecutive long-term prediction of 800 days.")

if __name__ == "__main__":
    main()