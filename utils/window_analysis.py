"""
Window Slicing and Shuffle Analysis for Time Series Library

This analysis examines how window slicing works and the impact of shuffling on time series forecasting.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from utils.logger import logger

def analyze_window_slicing():
    """
    Analyze how window slicing works in the Time Series Library
    """
    logger.info("Analyzing window slicing behavior")
    
    # Example parameters
    seq_len = 24     # Input sequence length
    label_len = 12   # Label length (overlap with prediction)
    pred_len = 12    # Prediction length
    data_len = 1000  # Total data length
    
    print("=== WINDOW SLICING ANALYSIS ===")
    print(f"seq_len (input): {seq_len}")
    print(f"label_len (overlap): {label_len}")
    print(f"pred_len (forecast): {pred_len}")
    print(f"data_len: {data_len}")
    print()
    
    # Calculate number of windows
    num_windows = data_len - seq_len - pred_len + 1
    print(f"Number of possible windows: {num_windows}")
    print()
    
    # Show first few windows
    print("WINDOW STRUCTURE:")
    print("Each window has:")
    print("- seq_x: input sequence")
    print("- seq_y: target sequence (includes label_len overlap + pred_len forecast)")
    print()
    
    for i in range(min(5, num_windows)):
        s_begin = i
        s_end = s_begin + seq_len
        r_begin = s_end - label_len
        r_end = r_begin + label_len + pred_len
        
        print(f"Window {i}:")
        print(f"  seq_x (input):  [{s_begin}:{s_end}] = {list(range(s_begin, s_end))}")
        print(f"  seq_y (target): [{r_begin}:{r_end}] = {list(range(r_begin, r_end))}")
        print(f"    - label part:  [{r_begin}:{s_end}] = {list(range(r_begin, s_end))}")
        print(f"    - pred part:   [{s_end}:{r_end}] = {list(range(s_end, r_end))}")
        print()

def analyze_shuffle_impact():
    """
    Analyze the impact of shuffling on time series training
    """
    logger.info("Analyzing shuffle impact on time series")
    
    print("=== SHUFFLE IMPACT ANALYSIS ===")
    print()
    
    # Simulate window indices
    seq_len, pred_len = 24, 12
    data_len = 200
    num_windows = data_len - seq_len - pred_len + 1
    
    print(f"Total windows available: {num_windows}")
    print()
    
    # Non-shuffled order
    non_shuffled = list(range(num_windows))
    print("NON-SHUFFLED ORDER (first 10 windows):")
    for i in range(min(10, len(non_shuffled))):
        idx = non_shuffled[i]
        start_time = idx
        end_time = idx + seq_len + pred_len
        print(f"  Batch {i}: Window {idx} -> Time range [{start_time}:{end_time}]")
    
    print()
    
    # Shuffled order (simulated)
    np.random.seed(42)
    shuffled = np.random.permutation(num_windows)
    print("SHUFFLED ORDER (first 10 windows):")
    for i in range(min(10, len(shuffled))):
        idx = shuffled[i]
        start_time = idx
        end_time = idx + seq_len + pred_len
        print(f"  Batch {i}: Window {idx} -> Time range [{start_time}:{end_time}]")
    
    print()

def shuffle_recommendations():
    """
    Provide recommendations for shuffle usage in time series
    """
    logger.info("Providing shuffle recommendations")
    
    print("=== SHUFFLE RECOMMENDATIONS ===")
    print()
    
    print("CURRENT TSLib BEHAVIOR:")
    print("- Training: shuffle=True")
    print("- Validation: shuffle=True") 
    print("- Test: shuffle=False")
    print()
    
    print("PROS OF SHUFFLING:")
    print("✓ Breaks temporal correlations between consecutive batches")
    print("✓ Reduces overfitting to temporal order")
    print("✓ Better gradient diversity")
    print("✓ More robust model learning")
    print("✓ Standard practice in many ML scenarios")
    print()
    
    print("CONS OF SHUFFLING:")
    print("✗ Loses natural temporal progression")
    print("✗ May disrupt learning of long-term dependencies")
    print("✗ Less realistic for online/streaming scenarios")
    print("✗ May hurt models that rely on temporal structure")
    print()
    
    print("RECOMMENDATIONS:")
    print("1. KEEP SHUFFLING FOR:")
    print("   - Training: Yes (current behavior is good)")
    print("   - Reduces overfitting to specific temporal patterns")
    print("   - Each window is independent for forecasting tasks")
    print()
    
    print("2. NO SHUFFLING FOR:")
    print("   - Test: No (current behavior is good)")
    print("   - Maintains realistic evaluation order")
    print("   - Better for deployment simulation")
    print()
    
    print("3. VALIDATION: CONSIDER BOTH")
    print("   - Current: shuffled (like training)")
    print("   - Alternative: non-shuffled (like test)")
    print("   - Non-shuffled validation might be more realistic")
    print()
    
    print("4. SPECIAL CASES:")
    print("   - Online learning: disable shuffle")
    print("   - Streaming: disable shuffle")
    print("   - Sequential dependencies: consider disabling")

def visualize_window_overlap():
    """
    Visualize how windows overlap in time series
    """
    logger.info("Visualizing window overlap")
    
    seq_len, label_len, pred_len = 10, 5, 5
    data_len = 30
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Timeline
    timeline = np.arange(data_len)
    
    # Show first few windows
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, color in enumerate(colors):
        if i >= data_len - seq_len - pred_len + 1:
            break
            
        s_begin = i
        s_end = s_begin + seq_len
        r_begin = s_end - label_len
        r_end = r_begin + label_len + pred_len
        
        # Input sequence
        ax1.barh(i, seq_len, left=s_begin, alpha=0.6, color=color, label=f'Window {i} Input')
        
        # Target sequence
        ax2.barh(i, label_len + pred_len, left=r_begin, alpha=0.6, color=color, label=f'Window {i} Target')
    
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Window Index')
    ax1.set_title('Input Sequences (seq_x)')
    ax1.set_xlim(-1, data_len)
    
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Window Index')
    ax2.set_title('Target Sequences (seq_y)')
    ax2.set_xlim(-1, data_len)
    
    plt.tight_layout()
    plt.savefig('window_slicing_analysis.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    analyze_window_slicing()
    analyze_shuffle_impact()
    shuffle_recommendations()
    visualize_window_overlap()
