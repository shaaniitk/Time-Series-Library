#!/usr/bin/env python3
"""
Performance comparison test between base SOTA_Temporal_PGAT and Enhanced_SOTA_PGAT
to verify that the enhanced model is actually more computationally expensive.
"""

import torch
import time
import sys
import os
from types import SimpleNamespace

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_config(model_name, d_model=256):
    """Create test configuration for performance comparison."""
    return SimpleNamespace(
        # Core parameters
        d_model=d_model,
        n_heads=8,
        seq_len=96,
        pred_len=24,
        enc_in=7,
        c_out=3,
        dropout=0.1,
        
        # Enhanced features (only for Enhanced model)
        use_multi_scale_patching=(model_name == 'Enhanced_SOTA_PGAT'),
        use_hierarchical_mapper=(model_name == 'Enhanced_SOTA_PGAT'),
        use_stochastic_learner=(model_name == 'Enhanced_SOTA_PGAT'),
        use_gated_graph_combiner=(model_name == 'Enhanced_SOTA_PGAT'),
        patch_len=16,
        stride=8,
        num_wave_patch_latents=32,
        num_target_patch_latents=16,
        
        # Graph settings
        enable_dynamic_graph=True,
        enable_graph_attention=True,
        use_autocorr_attention=False,
        
        # Mixture density
        use_mixture_density=True,
        use_mixture_decoder=(model_name == 'Enhanced_SOTA_PGAT'),
        mixture_multivariate_mode='independent',
        mdn_components=3,
        
        # Memory optimization
        enable_memory_optimization=True
    )

def benchmark_model(model_class, config, num_iterations=10):
    """Benchmark model performance."""
    print(f"\n🔍 Benchmarking {model_class.__name__}")
    print(f"  Configuration: d_model={config.d_model}, seq_len={config.seq_len}")
    
    try:
        # Create model
        model = model_class(config)
        model.eval()  # Set to eval mode for consistent timing
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Create test data
        batch_size = 4
        wave_window = torch.randn(batch_size, config.seq_len, config.enc_in)
        # For base SOTA_Temporal_PGAT, target_window needs same feature dimension as wave_window
        if model_class.__name__ == 'SOTA_Temporal_PGAT':
            target_window = torch.randn(batch_size, config.pred_len, config.enc_in)  # Same as enc_in
        else:
            target_window = torch.randn(batch_size, config.pred_len, config.c_out)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(wave_window, target_window)
        
        # Benchmark forward pass
        forward_times = []
        with torch.no_grad():
            for i in range(num_iterations):
                start_time = time.time()
                output = model(wave_window, target_window)
                end_time = time.time()
                forward_times.append(end_time - start_time)
                
                if i == 0:  # Print output info for first iteration
                    if isinstance(output, tuple):
                        print(f"  Output: tuple with {len(output)} elements")
                        for j, elem in enumerate(output):
                            if hasattr(elem, 'shape'):
                                print(f"    Element {j}: {elem.shape}")
                    else:
                        print(f"  Output shape: {output.shape}")
        
        avg_forward_time = sum(forward_times) / len(forward_times)
        min_forward_time = min(forward_times)
        max_forward_time = max(forward_times)
        
        print(f"  Forward pass timing:")
        print(f"    Average: {avg_forward_time*1000:.2f}ms")
        print(f"    Min: {min_forward_time*1000:.2f}ms")
        print(f"    Max: {max_forward_time*1000:.2f}ms")
        
        # Benchmark backward pass (training simulation)
        model.train()
        backward_times = []
        
        for i in range(num_iterations):
            # Create fresh data with gradients
            wave_window = torch.randn(batch_size, config.seq_len, config.enc_in, requires_grad=True)
            if model.__class__.__name__ == 'SOTA_Temporal_PGAT':
                target_window = torch.randn(batch_size, config.pred_len, config.enc_in)  # Same as enc_in
            else:
                target_window = torch.randn(batch_size, config.pred_len, config.c_out)
            
            start_time = time.time()
            
            # Forward pass
            output = model(wave_window, target_window)
            
            # Compute loss
            if isinstance(output, tuple):
                # Mixture density output - use first element (means)
                if len(output) >= 3:
                    means, log_stds, log_weights = output
                    if means.dim() == 4:  # Multivariate
                        # Use mixture mean: [batch, pred_len, targets, components] -> [batch, pred_len, targets]
                        weights = torch.softmax(log_weights, dim=-1).unsqueeze(2)
                        pred = (weights * means).sum(dim=-1)
                    else:  # Univariate
                        weights = torch.softmax(log_weights, dim=-1)
                        pred = (weights * means).sum(dim=-1).unsqueeze(-1)
                else:
                    pred = output[0]
            else:
                pred = output
            
            # Simple MSE loss
            loss = torch.nn.functional.mse_loss(pred, target_window)
            
            # Backward pass
            loss.backward()
            
            end_time = time.time()
            backward_times.append(end_time - start_time)
            
            # Clear gradients
            model.zero_grad()
        
        avg_backward_time = sum(backward_times) / len(backward_times)
        min_backward_time = min(backward_times)
        max_backward_time = max(backward_times)
        
        print(f"  Forward+Backward timing:")
        print(f"    Average: {avg_backward_time*1000:.2f}ms")
        print(f"    Min: {min_backward_time*1000:.2f}ms")
        print(f"    Max: {max_backward_time*1000:.2f}ms")
        
        return {
            'model_name': model_class.__name__,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'avg_forward_time': avg_forward_time,
            'avg_backward_time': avg_backward_time,
            'forward_times': forward_times,
            'backward_times': backward_times
        }
        
    except Exception as e:
        print(f"  ❌ Error benchmarking {model_class.__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("🚀 PGAT Performance Comparison Test")
    print("This test compares the computational cost of base vs enhanced PGAT models.")
    print()
    
    try:
        # Import models
        from models.SOTA_Temporal_PGAT import SOTA_Temporal_PGAT
        from models.Enhanced_SOTA_PGAT import Enhanced_SOTA_PGAT
        
        # Test configurations
        d_models = [256]  # Start with same size as training
        
        results = []
        
        for d_model in d_models:
            print(f"\n{'='*60}")
            print(f"Testing with d_model={d_model}")
            print(f"{'='*60}")
            
            # Test base model
            base_config = create_test_config('SOTA_Temporal_PGAT', d_model)
            base_result = benchmark_model(SOTA_Temporal_PGAT, base_config)
            if base_result:
                results.append(base_result)
            
            # Test enhanced model
            enhanced_config = create_test_config('Enhanced_SOTA_PGAT', d_model)
            enhanced_result = benchmark_model(Enhanced_SOTA_PGAT, enhanced_config)
            if enhanced_result:
                results.append(enhanced_result)
            
            # Compare results
            if base_result and enhanced_result:
                print(f"\n📊 Comparison (d_model={d_model}):")
                print(f"  Parameter Ratio: {enhanced_result['total_params'] / base_result['total_params']:.2f}x")
                print(f"  Forward Time Ratio: {enhanced_result['avg_forward_time'] / base_result['avg_forward_time']:.2f}x")
                print(f"  Training Time Ratio: {enhanced_result['avg_backward_time'] / base_result['avg_backward_time']:.2f}x")
                
                if enhanced_result['avg_backward_time'] <= base_result['avg_backward_time']:
                    print(f"  ⚠️  WARNING: Enhanced model is not slower than base model!")
                    print(f"     This suggests some enhanced features may not be active.")
                else:
                    print(f"  ✅ Enhanced model is appropriately slower than base model.")
        
        # Summary
        print(f"\n🎯 Performance Analysis Summary:")
        for result in results:
            print(f"  {result['model_name']}:")
            print(f"    Parameters: {result['total_params']:,}")
            print(f"    Forward: {result['avg_forward_time']*1000:.2f}ms")
            print(f"    Training: {result['avg_backward_time']*1000:.2f}ms")
        
        return results
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()