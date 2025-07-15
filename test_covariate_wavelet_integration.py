"""
Test CovariateAdapter with Chronos and WaveletProcessor Integration

This demonstrates how the CovariateAdapter enables covariate support for
any backbone while leveraging existing wavelet infrastructure.
"""

import torch
import torch.nn as nn
import sys
import os
import logging

# Add paths for our modular components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.modular_components.implementations.adapters import CovariateAdapter
from utils.modular_components.implementations.processors import WaveletProcessor
from utils.modular_components.implementations.backbones import ChronosBackbone
from utils.modular_components.config_schemas import BackboneConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_covariate_adapter_concept():
    """
    Test the CovariateAdapter concept with a mock Chronos backbone
    """
    print("=" * 60)
    print("🔧 Testing CovariateAdapter with Chronos")
    print("=" * 60)
    
    # Step 1: Create a mock Chronos backbone (since we may not have actual Chronos)
    class MockChronosBackbone:
        """Mock Chronos backbone for testing"""
        
        def __init__(self):
            self.config = BackboneConfig(d_model=512, dropout=0.1)
            self.tokenizer = MockTokenizer()
            
        def forward(self, input_ids, attention_mask=None):
            # Mock forward: just return some embeddings
            batch_size, seq_len = input_ids.shape
            return torch.randn(batch_size, seq_len, 512)
        
        def get_d_model(self):
            return 512
            
        def supports_seq2seq(self):
            return True
            
        def get_backbone_type(self):
            return "mock_chronos"
            
        def get_capabilities(self):
            return {
                'type': 'mock_chronos',
                'supports_time_series': True,
                'supports_forecasting': True,
                'd_model': 512
            }
    
    class MockTokenizer:
        """Mock tokenizer"""
        def encode(self, series):
            # Mock tokenization: just return integers
            return [int(x * 100) % 1000 for x in series]
    
    # Step 2: Create CovariateAdapter
    backbone = MockChronosBackbone()
    
    covariate_config = {
        'covariate_dim': 5,  # 5 covariate features
        'fusion_method': 'project',  # Project and combine approach
        'embedding_dim': 512,
        'temporal_features': True
    }
    
    adapter = CovariateAdapter(backbone, covariate_config)
    
    print(f"✅ CovariateAdapter created successfully")
    print(f"   - Base backbone: {backbone.get_backbone_type()}")
    print(f"   - Covariate support: {adapter.get_capabilities()['supports_covariates']}")
    print(f"   - Fusion method: {adapter.fusion_method}")
    
    # Step 3: Test with sample data
    batch_size, seq_len = 4, 96
    
    # Time series data (univariate)
    x_ts = torch.randn(batch_size, seq_len, 1)
    
    # Covariate data (5 features: temp, humidity, day_of_week, hour, is_weekend)
    x_covariates = torch.randn(batch_size, seq_len, 5)
    
    print(f"\n📊 Input Data:")
    print(f"   - Time series: {x_ts.shape}")
    print(f"   - Covariates: {x_covariates.shape}")
    
    # Step 4: Forward pass through adapter
    try:
        output = adapter.forward(x_ts, x_covariates)
        print(f"\n✅ Forward pass successful!")
        print(f"   - Output shape: {output.shape}")
        print(f"   - Expected: [{batch_size}, {seq_len}, 512]")
        
        # Verify output properties
        assert output.shape == (batch_size, seq_len, 512)
        assert not torch.isnan(output).any()
        print(f"   ✓ Output validation passed")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    return True


def test_wavelet_processor():
    """
    Test the WaveletProcessor using existing DWT infrastructure
    """
    print("\n" + "=" * 60)
    print("🌊 Testing WaveletProcessor with Existing DWT")
    print("=" * 60)
    
    # Configuration for wavelet processing
    wavelet_config = {
        'wavelet_type': 'db4',
        'decomposition_levels': 3,
        'residual_learning': True,
        'baseline_method': 'simple',
        'device': 'cpu'
    }
    
    try:
        # Create WaveletProcessor
        processor = WaveletProcessor(wavelet_config)
        print(f"✅ WaveletProcessor created successfully")
        print(f"   - Wavelet type: {processor.wavelet_type}")
        print(f"   - Decomposition levels: {processor.decomposition_levels}")
        print(f"   - Residual learning: {processor.residual_learning}")
        
        # Test with sample time series data
        batch_size, seq_len, n_features = 4, 96, 1
        x = torch.randn(batch_size, seq_len, n_features) * 10 + 100  # Add some scale and offset
        
        print(f"\n📊 Input Data:")
        print(f"   - Input shape: {x.shape}")
        print(f"   - Data range: [{x.min().item():.2f}, {x.max().item():.2f}]")
        
        # Forward pass through processor
        result = processor.forward(x)
        
        print(f"\n✅ Wavelet processing successful!")
        print(f"   - Processed shape: {result['processed'].shape}")
        
        if result['baseline'] is not None:
            print(f"   - Baseline shape: {result['baseline'].shape}")
        if result['residuals'] is not None:
            print(f"   - Residuals shape: {result['residuals'].shape}")
        
        # Check components
        components = result['components']
        if 'low_freq' in components and components['low_freq'] is not None:
            print(f"   - Low-freq coefficients: {components['low_freq'].shape}")
        if 'high_freq' in components:
            print(f"   - High-freq levels: {len(components['high_freq'])}")
        
        # Test inverse transform if baseline is available
        if result['baseline'] is not None:
            reconstructed = processor.inverse_transform(
                result['processed'], 
                components, 
                result['baseline']
            )
            print(f"   - Reconstructed shape: {reconstructed.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ WaveletProcessor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integrated_pipeline():
    """
    Test the complete pipeline: WaveletProcessor → CovariateAdapter → MockBackbone
    """
    print("\n" + "=" * 60)
    print("🔄 Testing Integrated Pipeline")
    print("=" * 60)
    
    # This demonstrates how the components work together
    print("Pipeline: Input → WaveletProcessor → CovariateAdapter → Backbone → Output")
    
    # Create components
    try:
        # 1. Wavelet processor
        wavelet_config = {
            'wavelet_type': 'db4',
            'decomposition_levels': 2,
            'residual_learning': True,
            'baseline_method': 'simple',
            'device': 'cpu'
        }
        wavelet_processor = WaveletProcessor(wavelet_config)
        
        # 2. Mock backbone
        class SimpleMockBackbone:
            def __init__(self):
                self.config = BackboneConfig(d_model=256, dropout=0.1)
                self.linear = nn.Linear(1, 256)
            
            def forward(self, x, **kwargs):
                # Simple processing: linear projection
                return self.linear(x)
            
            def get_d_model(self):
                return 256
            
            def supports_seq2seq(self):
                return False
            
            def get_backbone_type(self):
                return "simple_mock"
                
            def get_capabilities(self):
                return {
                    'type': 'simple_mock',
                    'd_model': 256
                }
        
        backbone = SimpleMockBackbone()
        
        # 3. Covariate adapter
        covariate_config = {
            'covariate_dim': 3,
            'fusion_method': 'add',
            'embedding_dim': 256
        }
        adapter = CovariateAdapter(backbone, covariate_config)
        
        print("✅ All components created successfully")
        
        # Test data
        batch_size, seq_len = 2, 48
        x_ts = torch.sin(torch.linspace(0, 4*3.14159, seq_len)).unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1)
        x_covariates = torch.randn(batch_size, seq_len, 3)
        
        print(f"\n📊 Pipeline Input:")
        print(f"   - Time series: {x_ts.shape}")
        print(f"   - Covariates: {x_covariates.shape}")
        
        # Step 1: Wavelet processing
        wavelet_result = wavelet_processor.forward(x_ts)
        processed_ts = wavelet_result['processed']
        
        print(f"\n🌊 After Wavelet Processing:")
        print(f"   - Processed shape: {processed_ts.shape}")
        if wavelet_result['residuals'] is not None:
            print(f"   - Using residuals for learning")
            processed_ts = wavelet_result['residuals']
        
        # Step 2: Covariate adaptation
        adapter_output = adapter.forward(processed_ts, x_covariates)
        
        print(f"\n🔧 After Covariate Adaptation:")
        print(f"   - Output shape: {adapter_output.shape}")
        print(f"   - Expected: [{batch_size}, {seq_len}, 256]")
        
        # Verify the complete pipeline
        assert adapter_output.shape == (batch_size, seq_len, 256)
        assert not torch.isnan(adapter_output).any()
        
        print(f"\n✅ Integrated pipeline test successful!")
        print(f"   ✓ Data flows correctly through all components")
        print(f"   ✓ Shapes are preserved and consistent")
        print(f"   ✓ No NaN values in output")
        
        return True
        
    except Exception as e:
        print(f"❌ Integrated pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("🧪 Testing Modular Components: CovariateAdapter + WaveletProcessor")
    print("=" * 80)
    
    results = []
    
    # Test 1: CovariateAdapter concept
    results.append(test_covariate_adapter_concept())
    
    # Test 2: WaveletProcessor with existing DWT
    results.append(test_wavelet_processor())
    
    # Test 3: Integrated pipeline
    results.append(test_integrated_pipeline())
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 Test Summary:")
    print("=" * 80)
    
    test_names = [
        "CovariateAdapter with Chronos",
        "WaveletProcessor with DWT",
        "Integrated Pipeline"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{i+1}. {name}: {status}")
    
    all_passed = all(results)
    overall_status = "✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED"
    print(f"\nOverall Status: {overall_status}")
    
    if all_passed:
        print("\n🎉 Ready to implement in the modular framework!")
        print("   - CovariateAdapter enables covariate support for any backbone")
        print("   - WaveletProcessor leverages existing DWT infrastructure")
        print("   - Components work together in modular pipeline")
    
    return all_passed


if __name__ == "__main__":
    main()
