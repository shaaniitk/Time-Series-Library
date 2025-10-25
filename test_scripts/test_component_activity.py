#!/usr/bin/env python3
"""
Test to verify that Enhanced_SOTA_PGAT components are actually being executed.
"""

import torch
import sys
import os
from types import SimpleNamespace

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_component_execution():
    """Test that all enhanced components are actually being executed."""
    
    print("🔍 Testing Enhanced PGAT Component Execution")
    print("=" * 60)
    
    try:
        from models.Enhanced_SOTA_PGAT import Enhanced_SOTA_PGAT
        
        # Create configuration with all enhanced features
        config = SimpleNamespace(
            d_model=256, n_heads=8, seq_len=96, pred_len=24,
            enc_in=7, c_out=3, dropout=0.1,
            
            # Enhanced features - ALL ENABLED
            use_multi_scale_patching=True,
            use_hierarchical_mapper=True,
            use_stochastic_learner=True,
            use_gated_graph_combiner=True,
            patch_len=16, stride=8,
            num_wave_patch_latents=32,
            num_target_patch_latents=16,
            
            # Graph settings
            enable_dynamic_graph=True,
            enable_graph_attention=True,
            use_autocorr_attention=False,
            
            # Mixture density
            use_mixture_density=True,
            use_mixture_decoder=True,
            mixture_multivariate_mode='independent',
            mdn_components=3,
            
            enable_memory_optimization=True
        )\n        
        # Create model
        model = Enhanced_SOTA_PGAT(config)
        print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Check configuration
        config_info = model.get_enhanced_config_info()
        print(f"\n📊 Enhanced Features Status:")
        print(f"  Multi-scale patching: {config_info['use_multi_scale_patching']}")
        print(f"  Hierarchical mapper: {config_info['use_hierarchical_mapper']}")
        print(f"  Stochastic learner: {config_info['use_stochastic_learner']}")
        print(f"  Gated combiner: {config_info['use_gated_graph_combiner']}")
        print(f"  Mixture decoder: {config_info['use_mixture_decoder']}")
        
        # Check component existence
        print(f"\n🔧 Component Existence Check:")
        components = [
            ('wave_patching_composer', 'Multi-scale wave patching'),
            ('target_patching_composer', 'Multi-scale target patching'),
            ('wave_temporal_to_spatial', 'Hierarchical wave mapper'),
            ('target_temporal_to_spatial', 'Hierarchical target mapper'),
            ('stochastic_learner', 'Stochastic graph learner'),
            ('graph_combiner', 'Gated graph combiner'),
            ('decoder', 'Mixture density decoder')
        ]
        
        for attr_name, description in components:
            if hasattr(model, attr_name) and getattr(model, attr_name) is not None:
                component = getattr(model, attr_name)
                param_count = sum(p.numel() for p in component.parameters())
                print(f"  ✅ {description}: {param_count:,} parameters")
            else:
                print(f"  ❌ {description}: NOT FOUND")
        
        # Test forward pass with execution tracing
        print(f"\n🚀 Testing Forward Pass with Execution Tracing...")
        
        # Monkey patch components to trace execution
        execution_trace = []
        
        def trace_execution(component_name):
            def decorator(original_forward):
                def traced_forward(*args, **kwargs):
                    start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                    end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                    
                    if start_time:
                        start_time.record()
                    else:
                        start_cpu = time.time()
                    
                    result = original_forward(*args, **kwargs)
                    
                    if end_time:
                        end_time.record()
                        torch.cuda.synchronize()
                        elapsed = start_time.elapsed_time(end_time)
                    else:
                        elapsed = (time.time() - start_cpu) * 1000  # Convert to ms
                    
                    execution_trace.append((component_name, elapsed))
                    return result
                return traced_forward
            return decorator
        
        # Apply tracing to key components
        import time
        if hasattr(model, 'wave_patching_composer') and model.wave_patching_composer:
            model.wave_patching_composer.forward = trace_execution('wave_patching_composer')(model.wave_patching_composer.forward)
        
        if hasattr(model, 'target_patching_composer') and model.target_patching_composer:
            model.target_patching_composer.forward = trace_execution('target_patching_composer')(model.target_patching_composer.forward)
        
        if hasattr(model, 'wave_temporal_to_spatial') and model.wave_temporal_to_spatial:
            model.wave_temporal_to_spatial.forward = trace_execution('wave_temporal_to_spatial')(model.wave_temporal_to_spatial.forward)
        
        if hasattr(model, 'target_temporal_to_spatial') and model.target_temporal_to_spatial:
            model.target_temporal_to_spatial.forward = trace_execution('target_temporal_to_spatial')(model.target_temporal_to_spatial.forward)
        
        if hasattr(model, 'stochastic_learner') and model.stochastic_learner:
            model.stochastic_learner.forward = trace_execution('stochastic_learner')(model.stochastic_learner.forward)
        
        if hasattr(model, 'graph_combiner') and model.graph_combiner:
            model.graph_combiner.forward = trace_execution('graph_combiner')(model.graph_combiner.forward)
        
        # Test forward pass
        batch_size = 2
        wave_window = torch.randn(batch_size, config.seq_len, config.enc_in)
        target_window = torch.randn(batch_size, config.pred_len, config.c_out)
        
        with torch.no_grad():
            output = model(wave_window, target_window)
        
        print(f"✅ Forward pass completed")
        
        # Print execution trace
        print(f"\n⏱️ Component Execution Times:")
        total_component_time = 0
        for component_name, elapsed_time in execution_trace:
            print(f"  {component_name}: {elapsed_time:.2f}ms")
            total_component_time += elapsed_time
        
        print(f"  Total component time: {total_component_time:.2f}ms")
        
        # Check internal logs
        if hasattr(model, 'internal_logs'):
            logs = model.internal_logs
            print(f"\n📋 Internal Logs:")
            for key, value in logs.items():
                print(f"  {key}: {value}")
        
        # Analysis
        if total_component_time < 5:  # Less than 5ms total
            print(f"\n⚠️  WARNING: Total component execution time is very low ({total_component_time:.2f}ms)")
            print(f"     This suggests enhanced components may not be doing significant work.")
        else:
            print(f"\n✅ Component execution times look reasonable ({total_component_time:.2f}ms total)")
        
        return execution_trace
        
    except Exception as e:
        print(f"❌ Component execution test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    execution_trace = test_component_execution()