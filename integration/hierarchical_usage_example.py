"""
Example: Using the Hierarchical Enhanced Autoformer

This script shows how to use the already-implemented hierarchical architecture
for multi-resolution time series forecasting.
"""

import torch
import sys
import os
from argparse import Namespace

# Add to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.HierarchicalEnhancedAutoformer import create_hierarchical_autoformer


def demo_hierarchical_autoformer():
    """Demonstrate the hierarchical model capabilities"""
    
    # Example configuration
    configs = Namespace(
        seq_len=96,          # Input sequence length
        label_len=48,        # Label length for decoder
        pred_len=24,         # Prediction length
        enc_in=7,           # Encoder input size (number of features)
        dec_in=7,           # Decoder input size
        c_out=7,            # Output size
        d_model=512,        # Model dimension
        n_heads=8,          # Number of attention heads
        e_layers=3,         # Number of encoder layers
        d_layers=2,         # Number of decoder layers
        d_ff=2048,          # Feed-forward dimension
        factor=1,           # Attention factor
        dropout=0.1,        # Dropout rate
        embed='timeF',      # Time embedding type
        freq='h',           # Frequency for time features
        activation='gelu'   # Activation function
    )
    
    print("🏗️ Creating Hierarchical Enhanced Autoformer...")
    
    # Create hierarchical model with different configurations
    models = {
        'basic_hierarchical': create_hierarchical_autoformer(
            configs,
            n_levels=3,                          # 3 resolution levels
            wavelet_type='db4',                  # Daubechies 4 wavelet
            fusion_strategy='weighted_concat',    # Concatenation + projection
            use_cross_attention=True             # Enable cross-resolution attention
        ),
        
        'advanced_hierarchical': create_hierarchical_autoformer(
            configs,
            n_levels=4,                          # 4 resolution levels (more detail)
            wavelet_type='db8',                  # Higher-order wavelet
            fusion_strategy='attention_fusion',   # Attention-based fusion
            use_cross_attention=True
        ),
        
        'lightweight_hierarchical': create_hierarchical_autoformer(
            configs,
            n_levels=2,                          # Fewer levels for speed
            wavelet_type='haar',                 # Simple Haar wavelet
            fusion_strategy='weighted_sum',      # Simple weighted sum
            use_cross_attention=False            # Disable cross-attention for speed
        )
    }
    
    # Test with sample data
    batch_size = 4
    x_enc = torch.randn(batch_size, configs.seq_len, configs.enc_in)
    x_mark_enc = torch.randn(batch_size, configs.seq_len, 4)  # Time features
    x_dec = torch.randn(batch_size, configs.label_len + configs.pred_len, configs.dec_in)
    x_mark_dec = torch.randn(batch_size, configs.label_len + configs.pred_len, 4)
    
    for name, model in models.items():
        print(f"\n📊 Testing {name}:")
        print(f"   Hierarchy info: {model.get_hierarchy_info()}")
        
        # Forward pass
        with torch.no_grad():
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            print(f"   Input shape: {x_enc.shape}")
            print(f"   Output shape: {output.shape}")
            print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return models


def demo_hierarchical_components():
    """Demonstrate individual hierarchical components"""
    
    print("\n🔧 Testing Individual Hierarchical Components:")
    
    from models.HierarchicalEnhancedAutoformer import (
        WaveletHierarchicalDecomposer,
        CrossResolutionAttention,
        HierarchicalFusion
    )
    
    # Test decomposer
    decomposer = WaveletHierarchicalDecomposer(
        seq_len=96, 
        d_model=256, 
        levels=3,
        wavelet_type='db4'
    )
    
    x = torch.randn(4, 96, 256)
    decomposed = decomposer(x)
    print(f"   Decomposer input: {x.shape}")
    print(f"   Decomposed outputs: {[d.shape for d in decomposed]}")
    
    # Test cross-resolution attention
    cross_attention = CrossResolutionAttention(
        d_model=256, 
        n_levels=3, 
        use_multiwavelet=True
    )
    
    attended = cross_attention(decomposed)
    print(f"   Cross-attention outputs: {[a.shape for a in attended]}")
    
    # Test fusion
    fusion = HierarchicalFusion(
        d_model=256, 
        n_levels=3, 
        fusion_strategy='weighted_concat'
    )
    
    fused = fusion(attended, target_length=24)
    print(f"   Fused output: {fused.shape}")


if __name__ == "__main__":
    print("🌊 Hierarchical Enhanced Autoformer Demo")
    print("=" * 50)
    
    # Demo the complete models
    models = demo_hierarchical_autoformer()
    
    # Demo individual components
    demo_hierarchical_components()
    
    print("\n✅ Hierarchical architecture is fully functional!")
    print("\n📖 Key Features:")
    print("   • Multi-resolution wavelet decomposition using existing DWT infrastructure")
    print("   • Cross-resolution attention with MultiWavelet components")
    print("   • Multiple fusion strategies (weighted sum, concat, attention)")
    print("   • Integration with Enhanced AutoCorrelation layers")
    print("   • Configurable number of hierarchy levels and wavelet types")
    
    print("\n🚀 Next Steps:")
    print("   • Run ../scripts/train/train_hierarchical_autoformer.py for training")
    print("   • Experiment with different wavelet types and fusion strategies")
    print("   • Compare performance against baseline models")
