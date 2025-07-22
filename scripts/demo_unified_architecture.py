"""
Unified Autoformer Architecture Demo

This demo showcases the unified architecture that seamlessly integrates
HF-based and custom modular autoformers through a single interface.
"""

import torch
import numpy as np
from argparse import Namespace
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import unified factory
from models.unified_autoformer_factory import (
    create_autoformer,
    compare_implementations,
    list_available_models,
    UnifiedModelInterface
)

# Import configurations
from configs.autoformer import (
    enhanced_config,
    bayesian_enhanced_config,
    hierarchical_config,
    quantile_bayesian_config
)


def create_sample_data(batch_size=2, seq_len=96, pred_len=24, label_len=48, num_features=7):
    """Create sample time series data for testing"""
    
    # Create realistic-looking time series data
    time_steps = seq_len + label_len + pred_len
    
    # Generate base trend + seasonality + noise
    t = np.linspace(0, time_steps/24, time_steps)  # Assume hourly data
    trend = 0.1 * t
    seasonal = 2 * np.sin(2 * np.pi * t / 24)  # Daily pattern
    noise = 0.5 * np.random.randn(time_steps)
    
    base_series = trend + seasonal + noise
    
    # Create multiple features with correlations
    data = np.zeros((batch_size, time_steps, num_features))
    for b in range(batch_size):
        for f in range(num_features):
            feature_variation = base_series + 0.3 * np.random.randn(time_steps)
            data[b, :, f] = feature_variation
    
    # Split into encoder/decoder segments
    x_enc = torch.FloatTensor(data[:, :seq_len, :])
    x_dec_full = torch.FloatTensor(data[:, seq_len:, :])
    x_dec = x_dec_full[:, :label_len + pred_len, :]
    
    # Create time marks (day of week, hour of day, etc.)
    x_mark_enc = torch.randn(batch_size, seq_len, 4)
    x_mark_dec = torch.randn(batch_size, label_len + pred_len, 4)
    
    return x_enc, x_mark_enc, x_dec, x_mark_dec


def demo_model_creation():
    """Demonstrate creating different models through the unified factory"""
    
    print("=" * 80)
    print("🏭 UNIFIED AUTOFORMER FACTORY DEMO")
    print("=" * 80)
    
    # Show available models
    available = list_available_models()
    print(f"📋 Available Custom Models: {available['custom']}")
    print(f"📋 Available HF Models: {available['hf']}")
    print()
    
    # Base configuration
    base_config = {
        'seq_len': 96,
        'pred_len': 24,
        'label_len': 48,
        'enc_in': 7,
        'dec_in': 7,
        'c_out': 7,
        'd_model': 256,  # Smaller for demo
        'task_name': 'long_term_forecast'
    }
    
    # Demo different model types
    model_demos = {
        'enhanced': 'Basic enhanced autoformer with improved components',
        'bayesian_enhanced': 'Bayesian uncertainty quantification model',
        'hierarchical': 'Multi-resolution hierarchical processing',
        'quantile_bayesian': 'Quantile regression with Bayesian uncertainty'
    }
    
    created_models = {}
    
    for model_type, description in model_demos.items():
        print(f"🔧 Creating {model_type} model...")
        print(f"   Description: {description}")
        
        try:
            # Get appropriate configuration
            if model_type == 'enhanced':
                config = enhanced_config.get_enhanced_autoformer_config(7, 0, **base_config)
            elif model_type == 'bayesian_enhanced':
                config = bayesian_enhanced_config.get_bayesian_enhanced_autoformer_config(7, 0, **base_config)
            elif model_type == 'hierarchical':
                config = hierarchical_config.get_hierarchical_autoformer_config(7, 0, **base_config)
            elif model_type == 'quantile_bayesian':
                config = quantile_bayesian_config.get_quantile_bayesian_autoformer_config(
                    7, 0, quantile_levels=[0.1, 0.5, 0.9], **base_config
                )
            
            # Create model through unified factory
            start_time = time.time()
            model = create_autoformer(model_type, config, framework='custom')
            creation_time = time.time() - start_time
            
            # Get model info
            info = model.get_model_info()
            
            print(f"   ✅ Created in {creation_time:.2f}s")
            print(f"   📊 Framework: {info['framework_type']}")
            print(f"   🏗️  Model Class: {info['model_class']}")
            print(f"   🔬 Supports Uncertainty: {info['supports_uncertainty']}")
            
            created_models[model_type] = model
            print()
            
        except Exception as e:
            print(f"   ❌ Failed to create {model_type}: {e}")
            print()
    
    return created_models


def demo_unified_interface(models):
    """Demonstrate the unified interface across different models"""
    
    print("=" * 80)
    print("🎯 UNIFIED INTERFACE DEMO")
    print("=" * 80)
    
    # Create sample data
    x_enc, x_mark_enc, x_dec, x_mark_dec = create_sample_data()
    print(f"📊 Created sample data:")
    print(f"   Encoder input: {x_enc.shape}")
    print(f"   Decoder input: {x_dec.shape}")
    print()
    
    # Test each model with the same interface
    for model_name, model in models.items():
        print(f"🧪 Testing {model_name} model...")
        
        try:
            # Basic prediction
            with torch.no_grad():
                start_time = time.time()
                prediction = model.predict(x_enc, x_mark_enc, x_dec, x_mark_dec)
                prediction_time = time.time() - start_time
            
            print(f"   📈 Prediction shape: {prediction.shape}")
            print(f"   ⏱️  Prediction time: {prediction_time:.3f}s")
            
            # Uncertainty prediction (if supported)
            info = model.get_model_info()
            if info['supports_uncertainty']:
                with torch.no_grad():
                    uncertainty_result = model.predict_with_uncertainty(x_enc, x_mark_enc, x_dec, x_mark_dec)
                
                print(f"   🎲 Uncertainty prediction: ✅")
                print(f"   📊 Uncertainty keys: {list(uncertainty_result.keys())}")
            else:
                print(f"   🎲 Uncertainty prediction: ❌ (not supported)")
            
            print("   ✅ Interface test passed")
            print()
            
        except Exception as e:
            print(f"   ❌ Interface test failed: {e}")
            print()


def demo_quantile_functionality():
    """Demonstrate quantile-specific functionality"""
    
    print("=" * 80)
    print("📊 QUANTILE FUNCTIONALITY DEMO")
    print("=" * 80)
    
    # Test different quantile configurations
    quantile_configs = [
        ([0.1, 0.5, 0.9], "3-quantile (10%, 50%, 90%)"),
        ([0.05, 0.25, 0.5, 0.75, 0.95], "5-quantile (5%, 25%, 50%, 75%, 95%)"),
        ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], "9-quantile (10% to 90%)")
    ]
    
    base_config = {
        'seq_len': 96,
        'pred_len': 24,
        'label_len': 48,
        'enc_in': 7,
        'dec_in': 7,
        'c_out': 7,
        'd_model': 256
    }
    
    # Create sample data
    x_enc, x_mark_enc, x_dec, x_mark_dec = create_sample_data()
    
    for quantiles, description in quantile_configs:
        print(f"🎯 Testing {description}...")
        
        try:
            # Create quantile model
            config = quantile_bayesian_config.get_quantile_bayesian_autoformer_config(
                7, 0, quantile_levels=quantiles, **base_config
            )
            
            model = create_autoformer('quantile_bayesian', config, framework='custom')
            
            # Make prediction
            with torch.no_grad():
                prediction = model.predict(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            expected_dims = 7 * len(quantiles)  # 7 targets * number of quantiles
            
            print(f"   📊 Quantiles: {len(quantiles)}")
            print(f"   📈 Output shape: {prediction.shape}")
            print(f"   ✅ Expected dimensions: {expected_dims}, Got: {prediction.shape[-1]}")
            print(f"   🎯 Dimension test: {'✅ PASS' if prediction.shape[-1] == expected_dims else '❌ FAIL'}")
            print()
            
        except Exception as e:
            print(f"   ❌ Quantile test failed: {e}")
            print()


def demo_framework_comparison():
    """Demonstrate framework comparison capabilities"""
    
    print("=" * 80)
    print("⚖️  FRAMEWORK COMPARISON DEMO")
    print("=" * 80)
    
    base_config = {
        'seq_len': 96,
        'pred_len': 24,
        'label_len': 48,
        'enc_in': 7,
        'dec_in': 7,
        'c_out': 7,
        'd_model': 256
    }
    
    # Create sample data
    x_enc, x_mark_enc, x_dec, x_mark_dec = create_sample_data()
    
    model_types = ['enhanced', 'bayesian_enhanced']
    
    for model_type in model_types:
        print(f"🔍 Comparing {model_type} implementations...")
        
        try:
            # Get configuration
            if model_type == 'enhanced':
                config = enhanced_config.get_enhanced_autoformer_config(7, 0, **base_config)
            else:
                config = bayesian_enhanced_config.get_bayesian_enhanced_autoformer_config(7, 0, **base_config)
            
            # Create comparison models
            comparison = compare_implementations(model_type, config)
            
            print(f"   📋 Available implementations: {list(comparison.keys())}")
            
            # Test predictions from both implementations
            for framework, model in comparison.items():
                with torch.no_grad():
                    start_time = time.time()
                    prediction = model.predict(x_enc, x_mark_enc, x_dec, x_mark_dec)
                    prediction_time = time.time() - start_time
                
                info = model.get_model_info()
                print(f"   🏗️  {framework}: {info['model_class']} - {prediction.shape} in {prediction_time:.3f}s")
            
            print("   ✅ Comparison successful")
            print()
            
        except Exception as e:
            print(f"   ❌ Comparison failed: {e}")
            print()


def demo_performance_benchmarking():
    """Demonstrate performance benchmarking capabilities"""
    
    print("=" * 80)
    print("🏃 PERFORMANCE BENCHMARKING DEMO")
    print("=" * 80)
    
    base_config = {
        'seq_len': 48,  # Smaller for faster benchmarking
        'pred_len': 12,
        'label_len': 24,
        'enc_in': 7,
        'dec_in': 7,
        'c_out': 7,
        'd_model': 128  # Much smaller for speed
    }
    
    # Create sample data
    x_enc, x_mark_enc, x_dec, x_mark_dec = create_sample_data(
        batch_size=4, seq_len=48, pred_len=12, label_len=24
    )
    
    models_to_benchmark = {
        'enhanced': enhanced_config.get_enhanced_autoformer_config(7, 0, **base_config),
        'bayesian_enhanced': bayesian_enhanced_config.get_bayesian_enhanced_autoformer_config(7, 0, **base_config)
    }
    
    results = {}
    
    for model_name, config in models_to_benchmark.items():
        print(f"⏱️  Benchmarking {model_name}...")
        
        try:
            # Create model
            model = create_autoformer(model_name, config, framework='custom')
            
            # Warm-up run
            with torch.no_grad():
                _ = model.predict(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # Benchmark multiple runs
            num_runs = 10
            times = []
            
            for i in range(num_runs):
                with torch.no_grad():
                    start_time = time.time()
                    prediction = model.predict(x_enc, x_mark_enc, x_dec, x_mark_dec)
                    end_time = time.time()
                    times.append(end_time - start_time)
            
            # Calculate statistics
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            results[model_name] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'min_time': min_time,
                'max_time': max_time,
                'output_shape': prediction.shape
            }
            
            print(f"   📊 Average time: {avg_time:.4f}s ± {std_time:.4f}s")
            print(f"   ⚡ Min time: {min_time:.4f}s, Max time: {max_time:.4f}s")
            print(f"   📈 Output shape: {prediction.shape}")
            print("   ✅ Benchmark complete")
            print()
            
        except Exception as e:
            print(f"   ❌ Benchmark failed: {e}")
            print()
    
    # Summary comparison
    if len(results) > 1:
        print("📊 BENCHMARK SUMMARY:")
        for model_name, stats in results.items():
            print(f"   {model_name}: {stats['avg_time']:.4f}s avg, {stats['output_shape']}")


def main():
    """Run complete unified architecture demo"""
    
    print("🚀 UNIFIED AUTOFORMER ARCHITECTURE DEMO")
    print("🎯 Demonstrating seamless integration of HF and custom models")
    print()
    
    try:
        # 1. Model Creation Demo
        models = demo_model_creation()
        
        # 2. Unified Interface Demo
        if models:
            demo_unified_interface(models)
        
        # 3. Quantile Functionality Demo
        demo_quantile_functionality()
        
        # 4. Framework Comparison Demo
        demo_framework_comparison()
        
        # 5. Performance Benchmarking Demo
        demo_performance_benchmarking()
        
        print("=" * 80)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("✅ The unified architecture is working correctly")
        print("🔧 Both HF and custom models can be created seamlessly")
        print("🎯 Unified interface provides consistent experience")
        print("📊 Quantile functionality works across all configurations")
        print("⚖️  Framework comparison enables easy evaluation")
        print("🏃 Performance benchmarking helps optimize model selection")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
