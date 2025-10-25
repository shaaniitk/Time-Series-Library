#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

import math

def calculate_steps(dataset_size, batch_size, drop_last=False):
    """Calculate number of steps based on PyTorch DataLoader logic"""
    if drop_last:
        steps = dataset_size // batch_size
    else:
        steps = math.ceil(dataset_size / batch_size)
    return steps

def analyze_batch_distribution():
    """Analyze how batches are distributed"""
    
    dataset_size = 611
    batch_size = 8
    
    print(f"=== Batch Analysis ===")
    print(f"Dataset size: {dataset_size}")
    print(f"Batch size: {batch_size}")
    
    # Calculate steps with different drop_last settings
    steps_no_drop = calculate_steps(dataset_size, batch_size, drop_last=False)
    steps_drop = calculate_steps(dataset_size, batch_size, drop_last=True)
    
    print(f"\nWith drop_last=False: {steps_no_drop} steps")
    print(f"With drop_last=True: {steps_drop} steps")
    
    # Analyze batch distribution for drop_last=False
    print(f"\n=== Batch Distribution (drop_last=False) ===")
    total_processed = 0
    for step in range(steps_no_drop):
        start_idx = step * batch_size
        end_idx = min(start_idx + batch_size, dataset_size)
        actual_batch_size = end_idx - start_idx
        total_processed += actual_batch_size
        
        print(f"Step {step + 1:2d}: samples {start_idx:3d}-{end_idx-1:3d} (batch size: {actual_batch_size})")
    
    print(f"\nTotal samples processed: {total_processed}")
    print(f"Expected samples: {dataset_size}")
    print(f"Match: {total_processed == dataset_size}")
    
    # Analyze batch distribution for drop_last=True
    print(f"\n=== Batch Distribution (drop_last=True) ===")
    total_processed_drop = 0
    for step in range(steps_drop):
        start_idx = step * batch_size
        end_idx = start_idx + batch_size
        actual_batch_size = batch_size
        total_processed_drop += actual_batch_size
        
        print(f"Step {step + 1:2d}: samples {start_idx:3d}-{end_idx-1:3d} (batch size: {actual_batch_size})")
    
    remaining = dataset_size - total_processed_drop
    print(f"\nTotal samples processed: {total_processed_drop}")
    print(f"Remaining samples (dropped): {remaining}")
    print(f"Expected samples: {dataset_size}")

if __name__ == "__main__":
    analyze_batch_distribution()