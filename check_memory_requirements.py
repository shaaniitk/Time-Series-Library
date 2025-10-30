#!/usr/bin/env python3
"""
Check memory requirements for the production configuration
"""

import sys
import os
import torch
import psutil

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def estimate_memory_usage():
    """Estimate memory usage for the production configuration"""
    print("Estimating Memory Requirements...")
    
    # Configuration parameters
    batch_size = 8
    seq_len = 500
    pred_len = 20
    d_model = 416
    n_heads = 8
    e_layers = 6
    d_layers = 3
    enc_in = 113
    c_out = 4
    
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dimension: {d_model}")
    print(f"  Encoder layers: {e_layers}")
    print(f"  Decoder layers: {d_layers}")
    print(f"  Input features: {enc_in}")
    
    # Estimate model parameters
    # Rough estimation based on transformer architecture
    embedding_params = enc_in * d_model  # Input embedding
    encoder_params = e_layers * (
        4 * d_model * d_model +  # Self-attention (Q, K, V, O)
        2 * d_model * d_model * 4  # Feed-forward (2 layers)
    )
    decoder_params = d_layers * (
        4 * d_model * d_model +  # Self-attention
        4 * d_model * d_model +  # Cross-attention
        2 * d_model * d_model * 4  # Feed-forward
    )
    output_params = d_model * c_out  # Output projection
    
    total_params = embedding_params + encoder_params + decoder_params + output_params
    
    print(f"\nEstimated Model Parameters:")
    print(f"  Embedding: {embedding_params:,}")
    print(f"  Encoder: {encoder_params:,}")
    print(f"  Decoder: {decoder_params:,}")
    print(f"  Output: {output_params:,}")
    print(f"  Total: {total_params:,}")
    
    # Estimate memory usage (in bytes)
    # Parameters (float32 = 4 bytes each)
    param_memory = total_params * 4
    
    # Activations (rough estimate)
    # Input: batch_size * seq_len * enc_in
    input_memory = batch_size * seq_len * enc_in * 4
    
    # Hidden states through layers
    hidden_memory = batch_size * seq_len * d_model * (e_layers + d_layers) * 4
    
    # Attention matrices
    attention_memory = batch_size * n_heads * seq_len * seq_len * (e_layers + d_layers) * 4
    
    # Gradients (same size as parameters)
    gradient_memory = param_memory
    
    # Optimizer states (Adam uses 2x parameter memory)
    optimizer_memory = param_memory * 2
    
    total_memory = param_memory + input_memory + hidden_memory + attention_memory + gradient_memory + optimizer_memory
    
    print(f"\nEstimated Memory Usage:")
    print(f"  Parameters: {param_memory / 1024**2:.1f} MB")
    print(f"  Input activations: {input_memory / 1024**2:.1f} MB")
    print(f"  Hidden states: {hidden_memory / 1024**2:.1f} MB")
    print(f"  Attention matrices: {attention_memory / 1024**2:.1f} MB")
    print(f"  Gradients: {gradient_memory / 1024**2:.1f} MB")
    print(f"  Optimizer states: {optimizer_memory / 1024**2:.1f} MB")
    print(f"  Total estimated: {total_memory / 1024**2:.1f} MB ({total_memory / 1024**3:.2f} GB)")
    
    # Check system memory
    memory = psutil.virtual_memory()
    print(f"\nSystem Memory:")
    print(f"  Total: {memory.total / 1024**3:.2f} GB")
    print(f"  Available: {memory.available / 1024**3:.2f} GB")
    print(f"  Used: {memory.used / 1024**3:.2f} GB")
    print(f"  Percentage used: {memory.percent:.1f}%")
    
    # Memory safety check
    required_gb = total_memory / 1024**3
    available_gb = memory.available / 1024**3
    
    if required_gb > available_gb:
        print(f"\n⚠️  WARNING: Estimated memory requirement ({required_gb:.2f} GB) exceeds available memory ({available_gb:.2f} GB)")
        print("This could cause out-of-memory errors!")
        
        # Suggest optimizations
        print("\n🔧 Suggested optimizations:")
        print(f"  - Reduce batch size from {batch_size} to {max(1, batch_size // 2)}")
        print(f"  - Reduce sequence length from {seq_len} to {seq_len // 2}")
        print(f"  - Reduce model dimension from {d_model} to {d_model // 2}")
        print(f"  - Use gradient checkpointing")
        print(f"  - Enable mixed precision training")
        
        return False
    else:
        print(f"\n✅ Memory requirement ({required_gb:.2f} GB) is within available memory ({available_gb:.2f} GB)")
        safety_margin = (available_gb - required_gb) / available_gb * 100
        print(f"Safety margin: {safety_margin:.1f}%")
        
        if safety_margin < 20:
            print("⚠️  Low safety margin - consider reducing model size for stability")
        
        return True

if __name__ == "__main__":
    print("=" * 60)
    print("MEMORY REQUIREMENTS CHECK")
    print("=" * 60)
    
    memory_ok = estimate_memory_usage()
    
    print("\n" + "=" * 60)
    if memory_ok:
        print("🎉 MEMORY CHECK PASSED!")
        print("The configuration should fit in available memory.")
    else:
        print("❌ MEMORY CHECK FAILED!")
        print("The configuration may cause out-of-memory errors.")
    print("=" * 60)