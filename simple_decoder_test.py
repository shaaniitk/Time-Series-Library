#!/usr/bin/env python3
"""
Simple test to validate core decoder improvements.
"""

def test_basic_imports():
    """Test that our core modules can be imported."""
    try:
        # Test basic imports
        from layers.modular.decoder.decoder_output import DecoderOutput, standardize_decoder_output
        print("✓ DecoderOutput imports successful")
        
        from layers.modular.decoder.base import BaseDecoder
        print("✓ BaseDecoder imports successful")
        
        from layers.modular.decoder.unified_interface import UnifiedDecoderInterface
        print("✓ UnifiedDecoderInterface imports successful")
        
        from layers.modular.decoder.validation import ComponentValidator
        print("✓ ComponentValidator imports successful")
        
        return True
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False

def test_decoder_output_basic():
    """Test basic DecoderOutput functionality without torch."""
    try:
        from layers.modular.decoder.decoder_output import DecoderOutput
        
        # Create mock tensors (just objects for structure test)
        class MockTensor:
            def __init__(self, shape):
                self.shape = shape
        
        seasonal = MockTensor((2, 10, 8))
        trend = MockTensor((2, 10, 8))
        
        # Test DecoderOutput creation
        output = DecoderOutput(seasonal=seasonal, trend=trend)
        assert output.seasonal == seasonal
        assert output.trend == trend
        assert output.aux_loss == 0.0
        print("✓ DecoderOutput creation successful")
        
        # Test with aux_loss
        output_with_aux = DecoderOutput(seasonal=seasonal, trend=trend, aux_loss=0.5)
        assert output_with_aux.aux_loss == 0.5
        print("✓ DecoderOutput with aux_loss successful")
        
        return True
    except Exception as e:
        print(f"✗ DecoderOutput basic test failed: {e}")
        return False

def test_type_annotations_basic():
    """Test type annotations without torch dependency."""
    try:
        from layers.modular.decoder.base import BaseDecoder
        import inspect
        
        # Check that forward method exists
        assert hasattr(BaseDecoder, 'forward')
        print("✓ BaseDecoder has forward method")
        
        # Check method signature
        sig = inspect.signature(BaseDecoder.forward)
        params = list(sig.parameters.keys())
        expected = ['self', 'x', 'cross', 'x_mask', 'cross_mask', 'trend']
        assert params == expected
        print("✓ BaseDecoder signature correct")
        
        return True
    except Exception as e:
        print(f"✗ Type annotation basic test failed: {e}")
        return False

def test_registry_basic():
    """Test registry functionality without torch."""
    try:
        from layers.modular.decoder.registry import DecoderRegistry
        
        # Test listing components
        components = DecoderRegistry.list_components()
        assert isinstance(components, list)
        assert len(components) > 0
        print("✓ Registry listing successful")
        
        # Test getting component info
        if 'standard' in components:
            info = DecoderRegistry.get_component_info('standard')
            assert 'name' in info
            assert info['name'] == 'standard'
            print("✓ Registry component info successful")
        
        return True
    except Exception as e:
        print(f"✗ Registry basic test failed: {e}")
        return False

def main():
    """Run basic validation tests."""
    print("=" * 40)
    print("BASIC DECODER VALIDATION")
    print("=" * 40)
    
    tests = [
        test_basic_imports,
        test_decoder_output_basic,
        test_type_annotations_basic,
        test_registry_basic,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print(f"\n--- {test.__name__} ---")
        if test():
            passed += 1
            print(f"✓ {test.__name__} PASSED")
        else:
            print(f"✗ {test.__name__} FAILED")
    
    print("\n" + "=" * 40)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 40)
    
    if passed == total:
        print("🎉 Basic decoder improvements validated!")
        return 0
    else:
        print("❌ Some basic tests failed.")
        return 1

if __name__ == "__main__":
    exit(main())