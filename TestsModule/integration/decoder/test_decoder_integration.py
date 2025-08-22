"""
Integration tests for decoder implementations.
"""
import pytest
import torch
import torch.nn as nn
from layers.modular.decoder import (
    DecoderRegistry, UnifiedDecoderInterface, DecoderFactory,
    ComponentValidator, DecoderTestRunner, DecoderOutput
)
# Legacy registries not required in these integration tests


class TestDecoderIntegration:
    """Integration tests for decoder components."""
    
    @pytest.fixture
    def sample_components(self):
        """Create sample components for testing."""
        # Mock attention component
        class MockAttention(nn.Module):
            def __init__(self, d_model):
                super().__init__()
                self.d_model = d_model
            
            def forward(self, q, k, v, attn_mask=None):
                # Simple identity transformation
                return q, torch.ones(q.size(0), q.size(1), q.size(1))
        
        # Mock decomposition component
        class MockDecomposition(nn.Module):
            def __init__(self, kernel_size=25):
                super().__init__()
                self.kernel_size = kernel_size
            
            def forward(self, x):
                # Simple split: seasonal = x, trend = zeros
                trend = torch.zeros_like(x)
                return x, trend
        
        return {
            'attention': MockAttention(8),
            'decomposition': MockDecomposition()
        }
    
    def test_standard_decoder_integration(self, sample_components):
        """Test standard decoder with mock components."""
        from layers.modular.decoder import StandardDecoder
        
        decoder = StandardDecoder(
            d_layers=2,
            d_model=8,
            c_out=1,
            n_heads=2,
            d_ff=32,
            dropout=0.1,
            activation="relu",
            self_attention_comp=sample_components['attention'],
            cross_attention_comp=sample_components['attention'],
            decomp_comp=sample_components['decomposition']
        )
        
        # Test forward pass
        x = torch.randn(2, 10, 8)
        cross = torch.randn(2, 12, 8)
        
        seasonal, trend = decoder(x, cross)
        
        assert isinstance(seasonal, torch.Tensor)
        assert isinstance(trend, torch.Tensor)
        assert seasonal.shape == x.shape
    
    def test_enhanced_decoder_integration(self, sample_components):
        """Test enhanced decoder with mock components."""
        from layers.modular.decoder import EnhancedDecoder
        
        decoder = EnhancedDecoder(
            d_layers=2,
            d_model=8,
            c_out=1,
            n_heads=2,
            d_ff=32,
            dropout=0.1,
            activation="relu",
            self_attention_comp=sample_components['attention'],
            cross_attention_comp=sample_components['attention'],
            decomp_comp=sample_components['decomposition']
        )
        
        # Test forward pass
        x = torch.randn(2, 10, 8)
        cross = torch.randn(2, 12, 8)
        
        seasonal, trend = decoder(x, cross)
        
        assert isinstance(seasonal, torch.Tensor)
        assert isinstance(trend, torch.Tensor)
        assert seasonal.shape == x.shape
    
    def test_unified_interface_integration(self, sample_components):
        """Test unified interface with different decoder types."""
        from layers.modular.decoder import StandardDecoder
        
        # Create standard decoder
        standard_decoder = StandardDecoder(
            d_layers=1,
            d_model=8,
            c_out=1,
            n_heads=2,
            d_ff=32,
            dropout=0.1,
            activation="relu",
            self_attention_comp=sample_components['attention'],
            cross_attention_comp=sample_components['attention'],
            decomp_comp=sample_components['decomposition']
        )
        
        # Wrap with unified interface
        unified_decoder = UnifiedDecoderInterface(standard_decoder)
        
        # Test forward pass
        x = torch.randn(2, 10, 8)
        cross = torch.randn(2, 12, 8)
        
        seasonal, trend = unified_decoder(x, cross)
        
        assert isinstance(seasonal, torch.Tensor)
        assert isinstance(trend, torch.Tensor)
        
        # Test full output
        full_output = unified_decoder.get_full_output(x, cross)
        assert isinstance(full_output, DecoderOutput)
        assert torch.equal(full_output.seasonal, seasonal)
        assert torch.equal(full_output.trend, trend)
    
    def test_registry_integration(self, sample_components):
        """Test decoder registry functionality."""
        # Test getting existing components
        standard_class = DecoderRegistry.get("standard")
        enhanced_class = DecoderRegistry.get("enhanced")
        
        assert standard_class is not None
        assert enhanced_class is not None
        
        # Test listing components
        components = DecoderRegistry.list_components()
        assert "standard" in components
        assert "enhanced" in components
        assert "stable" in components
        
        # Test component info
        info = DecoderRegistry.get_component_info("standard")
        assert info['name'] == "standard"
        assert 'forward_signature' in info
    
    def test_validation_integration(self, sample_components):
        """Test component validation."""
        from layers.modular.decoder import StandardDecoder
        
        # Create decoder
        decoder = StandardDecoder(
            d_layers=1,
            d_model=8,
            c_out=1,
            n_heads=2,
            d_ff=32,
            dropout=0.1,
            activation="relu",
            self_attention_comp=sample_components['attention'],
            cross_attention_comp=sample_components['attention'],
            decomp_comp=sample_components['decomposition']
        )
        
        # Test component validation
        validator = ComponentValidator()
        validation_result = validator.validate_decoder_component(decoder)
        
        assert validation_result['valid'] is True
        assert len(validation_result['errors']) == 0
        
        # Test tensor validation
        x = torch.randn(2, 10, 8)
        cross = torch.randn(2, 12, 8)
        
        tensor_validation = validator.validate_tensor_compatibility(x, cross)
        assert tensor_validation['valid'] is True
    
    def test_decoder_test_runner(self, sample_components):
        """Test decoder test runner functionality."""
        from layers.modular.decoder import StandardDecoder
        
        # Create decoder
        decoder = StandardDecoder(
            d_layers=1,
            d_model=8,
            c_out=1,
            n_heads=2,
            d_ff=32,
            dropout=0.1,
            activation="relu",
            self_attention_comp=sample_components['attention'],
            cross_attention_comp=sample_components['attention'],
            decomp_comp=sample_components['decomposition']
        )
        
        # Create test runner
        test_runner = DecoderTestRunner(decoder)
        
        # Run smoke test
        smoke_results = test_runner.run_smoke_test()
        assert smoke_results['passed'] is True
        assert 'forward_time' in smoke_results['performance']
        
        # Run gradient test
        gradient_results = test_runner.run_gradient_test()
        assert gradient_results['passed'] is True
        assert 'gradient_info' in gradient_results
    
    def test_factory_integration(self, sample_components):
        """Test decoder factory functionality."""
        from layers.modular.decoder import StandardDecoder
        
        # Create standard decoder
        standard_decoder = StandardDecoder(
            d_layers=1,
            d_model=8,
            c_out=1,
            n_heads=2,
            d_ff=32,
            dropout=0.1,
            activation="relu",
            self_attention_comp=sample_components['attention'],
            cross_attention_comp=sample_components['attention'],
            decomp_comp=sample_components['decomposition']
        )
        
        # Test factory methods
        unified_decoder = DecoderFactory.create_unified_decoder(standard_decoder)
        assert isinstance(unified_decoder, UnifiedDecoderInterface)
        
        wrapped_decoder = DecoderFactory.wrap_existing_decoder(standard_decoder)
        assert isinstance(wrapped_decoder, UnifiedDecoderInterface)
        assert wrapped_decoder.validate_inputs is True
    
    def test_error_handling_integration(self, sample_components):
        """Test error handling across components."""
        from layers.modular.decoder import StandardDecoder
        
        # Create decoder
        decoder = StandardDecoder(
            d_layers=1,
            d_model=8,
            c_out=1,
            n_heads=2,
            d_ff=32,
            dropout=0.1,
            activation="relu",
            self_attention_comp=sample_components['attention'],
            cross_attention_comp=sample_components['attention'],
            decomp_comp=sample_components['decomposition']
        )
        
        # Wrap with unified interface
        unified_decoder = UnifiedDecoderInterface(decoder)
        
        # Test invalid inputs
        with pytest.raises(TypeError):
            unified_decoder("invalid", torch.randn(2, 10, 8))
        
        with pytest.raises(ValueError):
            unified_decoder(torch.randn(10, 8), torch.randn(2, 10, 8))  # Wrong dimensions
        
        with pytest.raises(ValueError):
            unified_decoder(torch.randn(2, 10, 8), torch.randn(2, 10, 6))  # Feature mismatch
    
    def test_backward_compatibility(self, sample_components):
        """Test backward compatibility with existing interfaces."""
        from layers.modular.decoder import StandardDecoder
        
        # Create decoder
        decoder = StandardDecoder(
            d_layers=1,
            d_model=8,
            c_out=1,
            n_heads=2,
            d_ff=32,
            dropout=0.1,
            activation="relu",
            self_attention_comp=sample_components['attention'],
            cross_attention_comp=sample_components['attention'],
            decomp_comp=sample_components['decomposition']
        )
        
        # Test that old-style tuple unpacking still works
        x = torch.randn(2, 10, 8)
        cross = torch.randn(2, 12, 8)
        
        seasonal, trend = decoder(x, cross)
        
        # Should be able to unpack as tuple
        assert isinstance(seasonal, torch.Tensor)
        assert isinstance(trend, torch.Tensor)
        
        # Test with unified interface
        unified_decoder = UnifiedDecoderInterface(decoder)
        seasonal2, trend2 = unified_decoder(x, cross)
        
        # Results should be consistent
        assert seasonal2.shape == seasonal.shape
        assert trend2.shape == trend.shape