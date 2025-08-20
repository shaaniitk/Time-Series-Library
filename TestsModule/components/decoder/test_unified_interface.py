"""
Tests for unified decoder interface.
"""
import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock
from layers.modular.decoder.unified_interface import UnifiedDecoderInterface, DecoderFactory
from layers.modular.decoder.decoder_output import DecoderOutput


class MockDecoder(nn.Module):
    """Mock decoder for testing."""
    
    def __init__(self, return_format="2_tuple"):
        super().__init__()
        self.return_format = return_format
        self.call_count = 0
    
    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        self.call_count += 1
        
        # Create mock outputs
        seasonal = torch.randn_like(x)
        trend = torch.randn_like(x) if trend is None else trend + torch.randn_like(trend)
        
        if self.return_format == "2_tuple":
            return seasonal, trend
        elif self.return_format == "3_tuple":
            return seasonal, trend, 0.1
        elif self.return_format == "decoder_output":
            return DecoderOutput(seasonal=seasonal, trend=trend, aux_loss=0.2)
        else:
            raise ValueError(f"Unknown return format: {self.return_format}")


class TestUnifiedDecoderInterface:
    """Test unified decoder interface functionality."""
    
    def test_interface_creation(self):
        """Test creating unified interface."""
        mock_decoder = MockDecoder()
        interface = UnifiedDecoderInterface(mock_decoder)
        
        assert interface.decoder_impl is mock_decoder
        assert interface.validate_inputs is True
        assert interface._forward_count == 0
    
    def test_input_validation_enabled(self):
        """Test input validation when enabled."""
        mock_decoder = MockDecoder()
        interface = UnifiedDecoderInterface(mock_decoder, validate_inputs=True)
        
        # Valid inputs should work
        x = torch.randn(2, 10, 8)
        cross = torch.randn(2, 12, 8)
        
        seasonal, trend = interface(x, cross)
        assert seasonal.shape == x.shape
        assert trend.shape == x.shape
    
    def test_input_validation_type_errors(self):
        """Test input validation type errors."""
        mock_decoder = MockDecoder()
        interface = UnifiedDecoderInterface(mock_decoder, validate_inputs=True)
        
        with pytest.raises(TypeError, match="x and cross must be torch.Tensor"):
            interface("not_tensor", torch.randn(2, 10, 8))
    
    def test_input_validation_dimension_errors(self):
        """Test input validation dimension errors."""
        mock_decoder = MockDecoder()
        interface = UnifiedDecoderInterface(mock_decoder, validate_inputs=True)
        
        # Wrong dimensions
        with pytest.raises(ValueError, match="Expected 3D tensors"):
            interface(torch.randn(10, 8), torch.randn(2, 10, 8))
    
    def test_input_validation_feature_mismatch(self):
        """Test input validation feature dimension mismatch."""
        mock_decoder = MockDecoder()
        interface = UnifiedDecoderInterface(mock_decoder, validate_inputs=True)
        
        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            interface(torch.randn(2, 10, 8), torch.randn(2, 10, 6))
    
    def test_input_validation_disabled(self):
        """Test that validation can be disabled."""
        mock_decoder = MockDecoder()
        interface = UnifiedDecoderInterface(mock_decoder, validate_inputs=False)
        
        # This should not raise an error even with invalid inputs
        # (though the mock decoder might fail)
        try:
            interface("invalid", "invalid")
        except (TypeError, AttributeError):
            # Expected from mock decoder, not from validation
            pass
    
    def test_2_tuple_output_handling(self):
        """Test handling of 2-tuple decoder output."""
        mock_decoder = MockDecoder(return_format="2_tuple")
        interface = UnifiedDecoderInterface(mock_decoder)
        
        x = torch.randn(2, 10, 8)
        cross = torch.randn(2, 12, 8)
        
        seasonal, trend = interface(x, cross)
        
        assert isinstance(seasonal, torch.Tensor)
        assert isinstance(trend, torch.Tensor)
        assert seasonal.shape == x.shape
        assert mock_decoder.call_count == 1
    
    def test_3_tuple_output_handling(self):
        """Test handling of 3-tuple decoder output."""
        mock_decoder = MockDecoder(return_format="3_tuple")
        interface = UnifiedDecoderInterface(mock_decoder)
        
        x = torch.randn(2, 10, 8)
        cross = torch.randn(2, 12, 8)
        
        seasonal, trend = interface(x, cross)
        
        assert isinstance(seasonal, torch.Tensor)
        assert isinstance(trend, torch.Tensor)
        # aux_loss should be handled internally, not returned in basic forward
    
    def test_decoder_output_handling(self):
        """Test handling of DecoderOutput format."""
        mock_decoder = MockDecoder(return_format="decoder_output")
        interface = UnifiedDecoderInterface(mock_decoder)
        
        x = torch.randn(2, 10, 8)
        cross = torch.randn(2, 12, 8)
        
        seasonal, trend = interface(x, cross)
        
        assert isinstance(seasonal, torch.Tensor)
        assert isinstance(trend, torch.Tensor)
    
    def test_get_full_output(self):
        """Test getting full DecoderOutput."""
        mock_decoder = MockDecoder(return_format="3_tuple")
        interface = UnifiedDecoderInterface(mock_decoder)
        
        x = torch.randn(2, 10, 8)
        cross = torch.randn(2, 12, 8)
        
        output = interface.get_full_output(x, cross)
        
        assert isinstance(output, DecoderOutput)
        assert output.aux_loss == 0.1  # From mock decoder
    
    def test_forward_count_tracking(self):
        """Test forward pass counting."""
        mock_decoder = MockDecoder()
        interface = UnifiedDecoderInterface(mock_decoder)
        
        x = torch.randn(2, 10, 8)
        cross = torch.randn(2, 12, 8)
        
        assert interface._forward_count == 0
        
        interface(x, cross)
        assert interface._forward_count == 1
        
        interface(x, cross)
        assert interface._forward_count == 2
    
    def test_stats_and_reset(self):
        """Test statistics tracking and reset."""
        mock_decoder = MockDecoder()
        interface = UnifiedDecoderInterface(mock_decoder)
        
        x = torch.randn(2, 10, 8)
        cross = torch.randn(2, 12, 8)
        
        interface(x, cross)
        interface(x, cross)
        
        stats = interface.get_stats()
        assert stats['forward_count'] == 2
        assert stats['decoder_type'] == 'MockDecoder'
        
        interface.reset_stats()
        assert interface._forward_count == 0
    
    def test_error_handling(self):
        """Test error handling and wrapping."""
        mock_decoder = Mock()
        mock_decoder.side_effect = RuntimeError("Mock decoder error")
        
        interface = UnifiedDecoderInterface(mock_decoder)
        
        x = torch.randn(2, 10, 8)
        cross = torch.randn(2, 12, 8)
        
        with pytest.raises(RuntimeError, match="Decoder forward pass failed at step 1"):
            interface(x, cross)


class TestDecoderFactory:
    """Test decoder factory functionality."""
    
    def test_create_unified_decoder(self):
        """Test factory method for creating unified decoder."""
        mock_decoder = MockDecoder()
        
        interface = DecoderFactory.create_unified_decoder(mock_decoder, validate_inputs=False)
        
        assert isinstance(interface, UnifiedDecoderInterface)
        assert interface.decoder_impl is mock_decoder
        assert interface.validate_inputs is False
    
    def test_wrap_existing_decoder(self):
        """Test wrapping existing decoder."""
        mock_decoder = MockDecoder()
        
        interface = DecoderFactory.wrap_existing_decoder(mock_decoder)
        
        assert isinstance(interface, UnifiedDecoderInterface)
        assert interface.decoder_impl is mock_decoder
        assert interface.validate_inputs is True  # Default for wrapping