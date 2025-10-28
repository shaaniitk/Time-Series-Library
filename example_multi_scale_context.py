#!/usr/bin/env python3
"""
Example: Multi-Scale Context Fusion in Celestial Enhanced PGAT

This script demonstrates how to use the different context fusion modes
and shows the impact on model behavior and performance.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List
import matplotlib.pyplot as plt
import numpy as np

# Mock configuration for demonstration
@dataclass
class ExampleConfig:
    # Core parameters
    seq_len: int = 96
    label_len: int = 48
    pred_len: int = 24
    enc_in: int = 118
    dec_in: int = 4
    c_out: int = 4
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 3
    d_layers: int = 2
    dropout: float = 0.1
    embed: str = 'timeF'
    freq: str = 'h'
    
    # Celestial system
    use_celestial_graph: bool = True
    celestial_fusion_layers: int = 3
    num_celestial_bodies: int = 13
    
    # Enhanced features
    use_mixture_decoder: bool = False
    use_stochastic_learner: bool = False
    use_hierarchical_mapping: bool = False
    use_efficient_covariate_interaction: bool = False
    
    # Petri Net
    use_petri_net_combiner: bool = True
    num_message_passing_steps: int = 2
    edge_feature_dim: int = 6
    use_temporal_attention: bool = True
    use_spatial_attention: bool = True
    bypass_spatiotemporal_with_petri: bool = True
    
    # Other features
    enable_adaptive_topk: bool = False
    use_stochastic_control: bool = False
    enable_mdn_decoder: bool = False
    use_target_autocorrelation: bool = True
    use_calendar_effects: bool = True
    use_celestial_target_attention: bool = True
    celestial_target_use_gated_fusion: bool = True
    celestial_target_diagnostics: bool = True
    use_c2t_edge_bias: bool = False
    aggregate_waves_to_celestial: bool = True
    num_input_waves: int = 118
    target_wave_indices: List[int] = None
    use_dynamic_spatiotemporal_encoder: bool = True
    
    # Logging
    verbose_logging: bool = True
    enable_memory_debug: bool = False
    collect_diagnostics: bool = True
    enable_fusion_diagnostics: bool = True
    
    # Multi-Scale Context Fusion - THIS IS THE KEY PART!
    use_multi_scale_context: bool = True
    context_fusion_mode: str = 'multi_scale'
    short_term_kernel_size: int = 5
    medium_term_kernel_size: int = 15
    long_term_kernel_size: int = 0
    context_fusion_dropout: float = 0.1
    enable_context_diagnostics: bool = True
    
    def __post_init__(self):
        if self.target_wave_indices is None:
            self.target_wave_indices = [0, 1, 2, 3]

def demonstrate_context_fusion_modes():
    """Demonstrate different context fusion modes."""
    print("🌟 Multi-Scale Context Fusion Demonstration")
    print("=" * 60)
    
    # Test different fusion modes
    fusion_modes = ['simple', 'gated', 'attention', 'multi_scale']
    
    for mode in fusion_modes:
        print(f"\n🔧 Testing {mode.upper()} fusion mode...")
        
        try:
            # Create configuration for this mode
            config = ExampleConfig()
            config.context_fusion_mode = mode
            
            # Import and create model
            from models.Celestial_Enhanced_PGAT_Modular import Model
            model = Model(config)
            
            # Create dummy data
            batch_size = 2
            x_enc = torch.randn(batch_size, config.seq_len, config.enc_in)
            x_mark_enc = torch.randn(batch_size, config.seq_len, 4)
            x_dec = torch.randn(batch_size, config.label_len + config.pred_len, config.dec_in)
            x_mark_dec = torch.randn(batch_size, config.label_len + config.pred_len, 4)
            
            # Forward pass
            with torch.no_grad():
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                
            # Extract results
            if isinstance(output, tuple):
                predictions, metadata = output[0], output[-1]
            else:
                predictions, metadata = output, {}
            
            print(f"✅ {mode.upper()} mode successful!")
            print(f"   Predictions shape: {predictions.shape}")
            print(f"   Context fusion mode: {model.get_context_fusion_mode()}")
            
            # Print context diagnostics if available
            if metadata and 'context_fusion_diagnostics' in metadata:
                context_diag = metadata['context_fusion_diagnostics']
                print(f"   Context diagnostics: {len(context_diag)} metrics")
                
                # Show key metrics
                if 'norm_ratio' in context_diag:
                    print(f"   Norm ratio: {context_diag['norm_ratio']:.4f}")
                if 'gate_mean' in context_diag:
                    print(f"   Gate mean: {context_diag['gate_mean']:.4f}")
                if 'num_context_scales' in context_diag:
                    print(f"   Context scales: {context_diag['num_context_scales']}")
            
            # Print full diagnostics for multi_scale mode
            if mode == 'multi_scale':
                print("\n📊 Detailed Multi-Scale Diagnostics:")
                model.print_context_fusion_diagnostics()
                
        except Exception as e:
            print(f"❌ {mode.upper()} mode failed: {e}")
            import traceback
            traceback.print_exc()

def compare_fusion_performance():
    """Compare performance across different fusion modes."""
    print("\n🏆 Performance Comparison")
    print("=" * 60)
    
    fusion_modes = ['simple', 'gated', 'multi_scale']
    results = {}
    
    for mode in fusion_modes:
        print(f"\n⚡ Benchmarking {mode.upper()} mode...")
        
        try:
            config = ExampleConfig()
            config.context_fusion_mode = mode
            config.enable_context_diagnostics = True
            
            from models.Celestial_Enhanced_PGAT_Modular import Model
            model = Model(config)
            
            # Create test data
            batch_size = 4
            x_enc = torch.randn(batch_size, config.seq_len, config.enc_in)
            x_mark_enc = torch.randn(batch_size, config.seq_len, 4)
            x_dec = torch.randn(batch_size, config.label_len + config.pred_len, config.dec_in)
            x_mark_dec = torch.randn(batch_size, config.label_len + config.pred_len, 4)
            
            # Measure performance
            import time
            
            # Warmup
            with torch.no_grad():
                _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # Timing
            start_time = time.time()
            num_runs = 10
            
            for _ in range(num_runs):
                with torch.no_grad():
                    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            avg_time = (time.time() - start_time) / num_runs
            
            # Extract metadata
            if isinstance(output, tuple):
                predictions, metadata = output[0], output[-1]
            else:
                predictions, metadata = output, {}
            
            # Store results
            results[mode] = {
                'avg_time': avg_time,
                'predictions_shape': predictions.shape,
                'context_diagnostics': metadata.get('context_fusion_diagnostics', {})
            }
            
            print(f"   Average time: {avg_time:.4f}s")
            print(f"   Predictions: {predictions.shape}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results[mode] = {'error': str(e)}
    
    # Summary
    print("\n📈 Performance Summary:")
    print("-" * 40)
    for mode, result in results.items():
        if 'error' not in result:
            print(f"{mode.upper():>12}: {result['avg_time']:.4f}s")
        else:
            print(f"{mode.upper():>12}: ERROR")

def demonstrate_configuration_options():
    """Show different configuration options."""
    print("\n⚙️  Configuration Options")
    print("=" * 60)
    
    configurations = [
        {
            'name': 'Financial Time Series',
            'config': {
                'context_fusion_mode': 'multi_scale',
                'short_term_kernel_size': 3,    # Intraday patterns
                'medium_term_kernel_size': 21,  # Monthly patterns
                'long_term_kernel_size': 0,     # Global trends
            }
        },
        {
            'name': 'High-Frequency Data',
            'config': {
                'context_fusion_mode': 'gated',  # Adaptive balance
                'context_fusion_dropout': 0.15,
            }
        },
        {
            'name': 'Variable Importance Sequences',
            'config': {
                'context_fusion_mode': 'attention',  # Dynamic weighting
                'context_fusion_dropout': 0.1,
            }
        },
        {
            'name': 'Speed-Critical Application',
            'config': {
                'context_fusion_mode': 'simple',  # Minimal overhead
                'enable_context_diagnostics': False,
            }
        }
    ]
    
    for config_info in configurations:
        print(f"\n🎯 {config_info['name']}:")
        for key, value in config_info['config'].items():
            print(f"   {key}: {value}")

def main():
    """Run all demonstrations."""
    print("🚀 Celestial Enhanced PGAT - Multi-Scale Context Fusion Demo")
    print("=" * 80)
    
    try:
        # 1. Demonstrate different fusion modes
        demonstrate_context_fusion_modes()
        
        # 2. Compare performance
        compare_fusion_performance()
        
        # 3. Show configuration options
        demonstrate_configuration_options()
        
        print("\n" + "=" * 80)
        print("🎉 Demo completed successfully!")
        print("\n💡 Key Takeaways:")
        print("   • Multi-scale fusion provides richest context")
        print("   • Gated fusion offers good balance of performance/quality")
        print("   • Attention fusion is best for variable importance sequences")
        print("   • Simple fusion is fastest for speed-critical applications")
        print("\n📚 See MULTI_SCALE_CONTEXT_FUSION_GUIDE.md for detailed documentation")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure the Celestial Enhanced PGAT modular components are available")
        print("   Run this script from the project root directory")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()