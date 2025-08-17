"""
Tests for DecoderOutput standardization.
"""
import pytest
import torch
from layers.modular.decoder.decoder_output import DecoderOutput, standardize_decoder_output


class TestDecoderOutput:
    """Test DecoderOutput dataclass functionality."""
    
    def test_decoder_output_creation(self):
        """Test basic DecoderOutput creation."""
        seasonal = torch.randn(2, 10, 8)
        trend = torch.randn(2, 10, 8)
        
        output = DecoderOutput(seasonal=seasonal, trend=trend)
        
        assert torch.equal(output.seasonal, seasonal)
        assert torch.equal(output.trend, trend)
        assert output.aux_loss == 0.0
    
    def test_decoder_output_with_aux_loss(self):
        """Test DecoderOutput with auxiliary loss."""
        seasonal = torch.randn(2, 10, 8)
        trend = torch.randn(2, 10, 8)
        aux_loss = 0.5
        
        output = DecoderOutput(seasonal=seasonal, trend=trend, aux_loss=aux_loss)
        
        assert output.aux_loss == aux_loss
    
    def test_decoder_output_iteration(self):
        """Test tuple unpacking compatibility."""
        seasonal = torch.randn(2, 10, 8)
        trend = torch.randn(2, 10, 8)
        
        # Without aux_loss
        output = DecoderOutput(seasonal=seasonal, trend=trend)
        s, t = output
        assert torch.equal(s, seasonal)
        assert torch.equal(t, trend)
        
        # With aux_loss
        output_with_aux = DecoderOutput(seasonal=seasonal, trend=trend, aux_loss=0.5)
        s, t, aux = output_with_aux
        assert torch.equal(s, seasonal)
        assert torch.equal(t, trend)
        assert aux == 0.5
    
    def test_decoder_output_length(self):
        """Test length property."""
        seasonal = torch.randn(2, 10, 8)
        trend = torch.randn(2, 10, 8)
        
        output_no_aux = DecoderOutput(seasonal=seasonal, trend=trend)
        assert len(output_no_aux) == 2
        
        output_with_aux = DecoderOutput(seasonal=seasonal, trend=trend, aux_loss=0.5)
        assert len(output_with_aux) == 3
    
    def test_to_tuple_conversion(self):
        """Test conversion to tuple format."""
        seasonal = torch.randn(2, 10, 8)
        trend = torch.randn(2, 10, 8)
        aux_loss = 0.3
        
        output = DecoderOutput(seasonal=seasonal, trend=trend, aux_loss=aux_loss)
        
        # With aux_loss
        tuple_with_aux = output.to_tuple(include_aux_loss=True)
        assert len(tuple_with_aux) == 3
        assert torch.equal(tuple_with_aux[0], seasonal)
        assert torch.equal(tuple_with_aux[1], trend)
        assert tuple_with_aux[2] == aux_loss
        
        # Without aux_loss
        tuple_no_aux = output.to_tuple(include_aux_loss=False)
        assert len(tuple_no_aux) == 2
        assert torch.equal(tuple_no_aux[0], seasonal)
        assert torch.equal(tuple_no_aux[1], trend)


class TestStandardizeDecoderOutput:
    """Test output standardization functionality."""
    
    def test_standardize_existing_decoder_output(self):
        """Test standardizing already standardized output."""
        seasonal = torch.randn(2, 10, 8)
        trend = torch.randn(2, 10, 8)
        
        original = DecoderOutput(seasonal=seasonal, trend=trend)
        standardized = standardize_decoder_output(original)
        
        assert standardized is original  # Should return same instance
    
    def test_standardize_2_tuple(self):
        """Test standardizing 2-tuple output."""
        seasonal = torch.randn(2, 10, 8)
        trend = torch.randn(2, 10, 8)
        
        tuple_output = (seasonal, trend)
        standardized = standardize_decoder_output(tuple_output)
        
        assert isinstance(standardized, DecoderOutput)
        assert torch.equal(standardized.seasonal, seasonal)
        assert torch.equal(standardized.trend, trend)
        assert standardized.aux_loss == 0.0
    
    def test_standardize_3_tuple(self):
        """Test standardizing 3-tuple output."""
        seasonal = torch.randn(2, 10, 8)
        trend = torch.randn(2, 10, 8)
        aux_loss = 0.7
        
        tuple_output = (seasonal, trend, aux_loss)
        standardized = standardize_decoder_output(tuple_output)
        
        assert isinstance(standardized, DecoderOutput)
        assert torch.equal(standardized.seasonal, seasonal)
        assert torch.equal(standardized.trend, trend)
        assert standardized.aux_loss == aux_loss
    
    def test_standardize_list_input(self):
        """Test standardizing list input."""
        seasonal = torch.randn(2, 10, 8)
        trend = torch.randn(2, 10, 8)
        
        list_output = [seasonal, trend]
        standardized = standardize_decoder_output(list_output)
        
        assert isinstance(standardized, DecoderOutput)
        assert torch.equal(standardized.seasonal, seasonal)
        assert torch.equal(standardized.trend, trend)
    
    def test_standardize_invalid_tuple_length(self):
        """Test error handling for invalid tuple length."""
        seasonal = torch.randn(2, 10, 8)
        
        with pytest.raises(ValueError, match="expected 2 or 3 elements"):
            standardize_decoder_output((seasonal,))
        
        with pytest.raises(ValueError, match="expected 2 or 3 elements"):
            standardize_decoder_output((seasonal, seasonal, seasonal, seasonal))
    
    def test_standardize_invalid_type(self):
        """Test error handling for invalid input type."""
        with pytest.raises(TypeError, match="Invalid decoder output type"):
            standardize_decoder_output("invalid")
        
        with pytest.raises(TypeError, match="Invalid decoder output type"):
            standardize_decoder_output(42)