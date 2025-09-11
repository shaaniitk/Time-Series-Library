import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

# Import the registries
from layers.modular.encoder.registry import EncoderRegistry
from layers.modular.output_heads.registry import OutputHeadRegistry
from layers.modular.backbone.registry import BackboneRegistry

class TestModularComponentsSmoke:
    """
    Smoke tests for newly added modular components.
    
    These tests verify that components can be instantiated through
    the registry system without errors. They don't test functionality,
    just basic instantiation and interface compliance.
    """
    
    def test_level_output_head_instantiation(self):
        """Test that LevelOutputHead can be instantiated via registry."""
        # Create a mock configuration
        config = type('Config', (), {
            'd_model': 512,
            'c_out': 1,
            'use_time_smoothing': True
        })
        
        # Get the component from registry
        level_output_head_class = OutputHeadRegistry.get('level')
        
        # Instantiate the component
        output_head = level_output_head_class(config)
        
        # Basic checks
        assert output_head is not None
        assert isinstance(output_head, nn.Module)
        assert hasattr(output_head, 'forward')
        
        # Test forward pass with dummy data
        batch_size, seq_len, d_model = 2, 24, 512
        dummy_input = torch.randn(batch_size, seq_len, d_model)
        
        with torch.no_grad():
            output = output_head(dummy_input)
            assert output is not None
            assert output.shape[0] == batch_size  # Batch dimension preserved
    
    @patch('layers.modular.encoder.crossformer_encoder.PatchEmbedding')
    @patch('layers.modular.encoder.crossformer_encoder.Encoder')
    def test_crossformer_encoder_instantiation(self, mock_encoder, mock_patch_embedding):
        """Test that CrossformerEncoder can be instantiated via registry."""
        # Mock the Crossformer components to avoid import issues
        mock_patch_embedding.return_value = MagicMock()
        mock_encoder.return_value = MagicMock()
        
        # Create a mock configuration
        config = type('Config', (), {
            'enc_in': 7,
            'seq_len': 96,
            'd_model': 512,
            'n_heads': 8,
            'd_ff': 2048,
            'e_layers': 2,
            'dropout': 0.1,
            'factor': 3
        })
        
        # Get the component from registry
        crossformer_encoder_class = EncoderRegistry.get('crossformer')
        
        # Instantiate the component
        encoder = crossformer_encoder_class(config)
        
        # Basic checks
        assert encoder is not None
        assert isinstance(encoder, nn.Module)
        assert hasattr(encoder, 'forward')
    
    @patch('layers.modular.backbone.crossformer_backbone.CrossformerEncoder')
    def test_crossformer_backbone_instantiation(self, mock_crossformer_encoder):
        """Test that CrossformerBackbone can be instantiated via registry."""
        # Mock the CrossformerEncoder to avoid circular dependencies
        mock_encoder_instance = MagicMock()
        mock_encoder_instance.return_value = (torch.randn(2, 7, 12, 512), None)
        mock_crossformer_encoder.return_value = mock_encoder_instance
        
        # Create a mock configuration
        from layers.modular.config_schemas import BackboneConfig
        config = BackboneConfig(
            d_model=512,
            dropout=0.1,
            enc_in=7,
            seq_len=96,
            n_heads=8,
            d_ff=2048,
            e_layers=2,
            factor=3
        )
        
        # Get the component from registry
        crossformer_backbone_class = BackboneRegistry.get('crossformer')
        
        # Instantiate the component
        backbone = crossformer_backbone_class(config)
        
        # Basic checks
        assert backbone is not None
        assert isinstance(backbone, nn.Module)
        assert hasattr(backbone, 'forward')
        assert hasattr(backbone, 'get_d_model')
        assert hasattr(backbone, 'supports_seq2seq')
        
        # Test interface methods
        assert backbone.get_d_model() == 512
        assert backbone.supports_seq2seq() is True
        assert backbone.get_backbone_type() == 'crossformer'
    
    def test_registry_component_listing(self):
        """Test that all new components are properly listed in registries."""
        # Check encoder registry
        encoder_components = EncoderRegistry.list_components()
        assert 'crossformer' in encoder_components
        
        # Check output head registry
        output_head_components = OutputHeadRegistry.list_components()
        assert 'level' in output_head_components
        
        # Check backbone registry
        backbone_components = BackboneRegistry.list_components()
        assert 'crossformer' in backbone_components
    
    def test_component_interface_compliance(self):
        """Test that components follow expected interface patterns."""
        # Test LevelOutputHead interface
        config = type('Config', (), {
            'd_model': 256,
            'c_out': 1,
            'use_time_smoothing': False
        })
        
        level_output_head_class = OutputHeadRegistry.get('level')
        output_head = level_output_head_class(config)
        
        # Check that it's a proper PyTorch module
        assert isinstance(output_head, nn.Module)
        
        # Check that it has the expected methods
        assert callable(getattr(output_head, 'forward', None))
        
        # Test with different input sizes
        test_inputs = [
            torch.randn(1, 10, 256),  # Single sample
            torch.randn(4, 24, 256),  # Small batch
            torch.randn(8, 48, 256),  # Larger batch
        ]
        
        for test_input in test_inputs:
            with torch.no_grad():
                output = output_head(test_input)
                assert output is not None
                assert len(output.shape) >= 2  # At least batch and feature dimensions
    
    def test_error_handling(self):
        """Test that registries handle invalid component names gracefully."""
        # Test invalid encoder name
        with pytest.raises(ValueError, match="not found"):
            EncoderRegistry.get('nonexistent_encoder')
        
        # Test invalid output head name
        with pytest.raises(ValueError, match="not found"):
            OutputHeadRegistry.get('nonexistent_output_head')
        
        # Test invalid backbone name
        with pytest.raises(ValueError, match="not found"):
            BackboneRegistry.get('nonexistent_backbone')
    
    @patch('layers.modular.encoder.crossformer_encoder.logger')
    def test_logging_integration(self, mock_logger):
        """Test that components integrate properly with the logging system."""
        # This test ensures that components use the logger properly
        # and don't cause logging-related errors
        
        config = type('Config', (), {
            'd_model': 128,
            'c_out': 1,
            'use_time_smoothing': True
        })
        
        # Instantiate a component that uses logging
        level_output_head_class = OutputHeadRegistry.get('level')
        output_head = level_output_head_class(config)
        
        # Verify no logging errors occurred during instantiation
        assert output_head is not None
        
        # Test forward pass (which may also log)
        dummy_input = torch.randn(2, 12, 128)
        with torch.no_grad():
            output = output_head(dummy_input)
            assert output is not None

if __name__ == '__main__':
    # Allow running tests directly
    pytest.main([__file__, '-v'])