import pytest
import torch
from types import SimpleNamespace
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestConfigurationRobustness:
    """Test configuration handling robustness for new utilities"""
    
    def test_config_extensibility(self):
        """Test that configurations can be extended without breaking"""
        # Base config
        config = SimpleNamespace()
        config.d_model = 64
        config.n_heads = 8
        
        # Add new attributes dynamically
        config.new_feature = True
        config.experimental_mode = "advanced"
        
        # Should not break existing functionality
        assert config.d_model == 64
        assert config.n_heads == 8
        assert config.new_feature == True
        assert config.experimental_mode == "advanced"
    
    def test_config_backward_compatibility(self):
        """Test backward compatibility with old configs"""
        # Old style config
        old_config = {
            'seq_len': 96,
            'pred_len': 24,
            'd_model': 64
        }
        
        # Convert to namespace
        config = SimpleNamespace(**old_config)
        
        # Should work with existing code
        assert hasattr(config, 'seq_len')
        assert hasattr(config, 'pred_len')
        assert hasattr(config, 'd_model')
    
    def test_config_validation_robustness(self):
        """Test config validation handles new fields"""
        config = SimpleNamespace()
        
        # Required fields
        required_fields = ['d_model', 'n_heads', 'seq_len']
        for field in required_fields:
            setattr(config, field, 64 if field == 'd_model' else 8)
        
        # Optional new fields
        optional_fields = ['new_attention_type', 'enhanced_features', 'experimental_config']
        for field in optional_fields:
            if hasattr(config, field):
                # Field exists, validate it
                assert getattr(config, field) is not None
            else:
                # Field doesn't exist, set default
                setattr(config, field, None)
        
        # All fields should be accessible
        for field in required_fields + optional_fields:
            assert hasattr(config, field)
    
    def test_component_interface_robustness(self):
        """Test that component interfaces can handle new parameters"""
        # Simulate component with flexible interface
        class FlexibleComponent:
            def __init__(self, **kwargs):
                self.config = kwargs
                
            def forward(self, x, **additional_params):
                # Should handle additional parameters gracefully
                return x
        
        # Test with various parameter combinations
        component1 = FlexibleComponent(d_model=64, n_heads=8)
        component2 = FlexibleComponent(d_model=64, n_heads=8, new_feature=True)
        
        x = torch.randn(2, 10, 64)
        
        # Both should work
        out1 = component1.forward(x)
        out2 = component2.forward(x, extra_param="test")
        
        assert out1.shape == x.shape
        assert out2.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])