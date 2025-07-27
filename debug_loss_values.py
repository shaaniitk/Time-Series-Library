#!/usr/bin/env python3
"""
Debug script to examine actual loss values and tensor statistics
"""
import torch
import numpy as np

def analyze_tensor_stats(tensor, name):
    """Analyze tensor statistics"""
    print(f"\n{name} Statistics:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Mean: {tensor.mean().item():.6f}")
    print(f"  Std: {tensor.std().item():.6f}")
    print(f"  Min: {tensor.min().item():.6f}")
    print(f"  Max: {tensor.max().item():.6f}")
    print(f"  Sample values: {tensor.flatten()[:5].tolist()}")

# Add this debug code to the validation function in exp_long_term_forecasting.py
debug_code = '''
# ADD THIS DEBUG CODE IN THE VALIDATION LOOP (around line 220):

# Debug: Print actual tensor values
if iter_count == 0:  # Only debug first batch
    print("\\n=== VALIDATION DEBUG ===")
    analyze_tensor_stats(y_pred_for_loss_val, "y_pred_for_loss_val")
    analyze_tensor_stats(y_true_for_loss_val, "y_true_for_loss_val")
    
    # Compute loss components
    diff = y_pred_for_loss_val - y_true_for_loss_val
    analyze_tensor_stats(diff, "prediction_difference")
    
    print(f"Batch Loss: {loss.item():.7f}")
    print("========================\\n")
'''

print("Add this debug code to the validation function:")
print(debug_code)

print("\nAlso add this to the training loop for comparison:")
training_debug = '''
# ADD THIS DEBUG CODE IN THE TRAINING LOOP (around line 320):

if (i + 1) == 10:  # Debug 10th iteration only
    print("\\n=== TRAINING DEBUG ===")
    analyze_tensor_stats(y_pred_for_loss_train, "y_pred_for_loss_train") 
    analyze_tensor_stats(y_true_for_loss_train, "y_true_for_loss_train")
    
    diff = y_pred_for_loss_train - y_true_for_loss_train
    analyze_tensor_stats(diff, "prediction_difference")
    
    print(f"Batch Loss: {loss_train.item():.7f}")
    print("=========================\\n")
'''

print(training_debug)
