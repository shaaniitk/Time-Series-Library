"""
Tests for type annotation correctness.
"""
import pytest
import torch
import inspect
from typing import get_type_hints
from layers.modular.decoder.base import BaseDecoder
from layers.modular.layers.base import BaseEncoderLayer, BaseDecoderLayer


class TestTypeAnnotations:
    """Test that type annotations are correct."""
    
    def test_base_decoder_type_hints(self):
        """Test BaseDecoder has correct type hints."""
        # Get type hints for the forward method
        forward_method = BaseDecoder.forward
        type_hints = get_type_hints(forward_method)
        
        # Check parameter types
        assert type_hints.get('x') == torch.Tensor
        assert type_hints.get('cross') == torch.Tensor
        
        # Check optional parameters
        from typing import Optional
        assert type_hints.get('x_mask') == Optional[torch.Tensor]
        assert type_hints.get('cross_mask') == Optional[torch.Tensor]
        assert type_hints.get('trend') == Optional[torch.Tensor]
        
        # Check return type
        from typing import Tuple
        expected_return = Tuple[torch.Tensor, torch.Tensor]
        assert type_hints.get('return') == expected_return
    
    def test_base_encoder_layer_type_hints(self):
        """Test BaseEncoderLayer has correct type hints."""
        forward_method = BaseEncoderLayer.forward
        type_hints = get_type_hints(forward_method)
        
        assert type_hints.get('x') == torch.Tensor
        
        from typing import Optional
        assert type_hints.get('attn_mask') == Optional[torch.Tensor]
        
        # Check return type
        from typing import Tuple
        expected_return = Tuple[torch.Tensor, Optional[torch.Tensor]]
        assert type_hints.get('return') == expected_return
    
    def test_base_decoder_layer_type_hints(self):
        """Test BaseDecoderLayer has correct type hints."""
        forward_method = BaseDecoderLayer.forward
        type_hints = get_type_hints(forward_method)
        
        assert type_hints.get('x') == torch.Tensor
        assert type_hints.get('cross') == torch.Tensor
        
        from typing import Optional
        assert type_hints.get('x_mask') == Optional[torch.Tensor]
        assert type_hints.get('cross_mask') == Optional[torch.Tensor]
        
        # Check return type
        from typing import Tuple
        expected_return = Tuple[torch.Tensor, torch.Tensor]
        assert type_hints.get('return') == expected_return
    
    def test_no_nn_module_in_type_hints(self):
        """Test that nn.Module is not used in type hints."""
        import torch.nn as nn
        
        # Check BaseDecoder
        forward_method = BaseDecoder.forward
        type_hints = get_type_hints(forward_method)
        
        for param_name, param_type in type_hints.items():
            # Recursively check that nn.Module is not in the type annotation
            self._assert_no_nn_module_in_type(param_type, param_name)
    
    def _assert_no_nn_module_in_type(self, type_annotation, param_name):
        """Recursively check that nn.Module is not in type annotation."""
        import torch.nn as nn
        from typing import get_origin, get_args
        
        # Direct check
        if type_annotation is nn.Module:
            pytest.fail(f"Found nn.Module in type annotation for {param_name}")
        
        # Check generic types (Optional, Tuple, etc.)
        origin = get_origin(type_annotation)
        if origin is not None:
            args = get_args(type_annotation)
            for arg in args:
                if arg is not type(None):  # Skip NoneType from Optional
                    self._assert_no_nn_module_in_type(arg, param_name)
    
    def test_method_signatures_consistency(self):
        """Test that method signatures are consistent across base classes."""
        # Get BaseDecoder forward signature
        base_decoder_sig = inspect.signature(BaseDecoder.forward)
        base_decoder_params = list(base_decoder_sig.parameters.keys())
        
        # Expected parameters for decoder
        expected_decoder_params = ['self', 'x', 'cross', 'x_mask', 'cross_mask', 'trend']
        assert base_decoder_params == expected_decoder_params
        
        # Get BaseDecoderLayer forward signature
        base_layer_sig = inspect.signature(BaseDecoderLayer.forward)
        base_layer_params = list(base_layer_sig.parameters.keys())
        
        # Expected parameters for decoder layer (no trend parameter)
        expected_layer_params = ['self', 'x', 'cross', 'x_mask', 'cross_mask']
        assert base_layer_params == expected_layer_params
    
    def test_parameter_defaults(self):
        """Test that optional parameters have correct defaults."""
        # Check BaseDecoder
        base_decoder_sig = inspect.signature(BaseDecoder.forward)
        
        # These should have None as default
        optional_params = ['x_mask', 'cross_mask', 'trend']
        for param_name in optional_params:
            param = base_decoder_sig.parameters[param_name]
            assert param.default is None, f"Parameter {param_name} should default to None"
        
        # Check BaseDecoderLayer
        base_layer_sig = inspect.signature(BaseDecoderLayer.forward)
        
        optional_layer_params = ['x_mask', 'cross_mask']
        for param_name in optional_layer_params:
            param = base_layer_sig.parameters[param_name]
            assert param.default is None, f"Parameter {param_name} should default to None"