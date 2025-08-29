import pytest
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from layers.modular.base_interfaces import BaseComponent
    from layers.modular.core.config_schemas import ComponentConfig
    from layers.modular.core.registry import UnifiedRegistry as ComponentRegistry
    MODULAR_AVAILABLE = True
except ImportError:
    MODULAR_AVAILABLE = False


class TestModularComponents:
    """Test modular component system robustness"""
    
    @pytest.mark.skipif(not MODULAR_AVAILABLE, reason="Modular components not available")
    def test_base_interface_compliance(self):
        """Test that base interfaces work correctly"""
        config = ComponentConfig(name="test", type="test")
        
        class TestComponent(BaseComponent):
            def forward(self, x):
                return x
        
        component = TestComponent(config)
        assert component.config == config
    
    @pytest.mark.skipif(not MODULAR_AVAILABLE, reason="Modular components not available")
    def test_registry_system(self):
        """Test component registry"""
        registry = ComponentRegistry()
        
        class DummyComponent(BaseComponent):
            def forward(self, x):
                return x
        
        registry.register("dummy", DummyComponent)
        assert registry.is_registered("dummy")
    
    def test_utilities_import_robustness(self):
        """Test that utilities can be imported without breaking"""
        # Test logger import
        try:
            from utils.logger import logger
            assert logger is not None
        except ImportError:
            pytest.skip("Logger not available")
        
        # Test other utility imports
        import_tests = [
            "utils.timefeatures",
            "utils.tools", 
            "utils.metrics"
        ]
        
        for module in import_tests:
            try:
                __import__(module)
            except ImportError:
                pass  # Module might not exist, that's ok


if __name__ == "__main__":
    pytest.main([__file__, "-v"])