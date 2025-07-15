import pytest
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestSimpleInterfaces:
    """Test basic interface and factory concepts"""
    
    def test_base_interface_concept(self):
        """Test that components follow the same interface"""
        
        # Create a simple base interface
        class SimpleComponent:
            def process(self, x):
                raise NotImplementedError
            
            def get_size(self):
                raise NotImplementedError
        
        # Two different implementations
        class ComponentA(SimpleComponent):
            def process(self, x):
                return x * 2
            
            def get_size(self):
                return 64
        
        class ComponentB(SimpleComponent):
            def process(self, x):
                return x + 1
            
            def get_size(self):
                return 128
        
        # Both follow same interface
        comp_a = ComponentA()
        comp_b = ComponentB()
        
        x = torch.tensor([1.0, 2.0, 3.0])
        
        # Same method names work on both
        result_a = comp_a.process(x)
        result_b = comp_b.process(x)
        size_a = comp_a.get_size()
        size_b = comp_b.get_size()
        
        # Different results but same interface
        assert not torch.equal(result_a, result_b)
        assert size_a != size_b
        assert hasattr(comp_a, 'process')
        assert hasattr(comp_b, 'process')
    
    def test_factory_pattern_concept(self):
        """Test factory pattern for creating components"""
        
        # Simple factory
        class ComponentFactory:
            @staticmethod
            def create(component_type, config):
                if component_type == "doubler":
                    return DoublerComponent(config)
                elif component_type == "adder":
                    return AdderComponent(config)
                else:
                    raise ValueError(f"Unknown type: {component_type}")
        
        class DoublerComponent:
            def __init__(self, config):
                self.multiplier = config.get('multiplier', 2)
            
            def process(self, x):
                return x * self.multiplier
        
        class AdderComponent:
            def __init__(self, config):
                self.addend = config.get('addend', 1)
            
            def process(self, x):
                return x + self.addend
        
        # Use factory to create components
        factory = ComponentFactory()
        
        comp1 = factory.create("doubler", {'multiplier': 3})
        comp2 = factory.create("adder", {'addend': 5})
        
        x = torch.tensor([1.0, 2.0])
        
        result1 = comp1.process(x)
        result2 = comp2.process(x)
        
        assert torch.equal(result1, torch.tensor([3.0, 6.0]))
        assert torch.equal(result2, torch.tensor([6.0, 7.0]))
    
    def test_interface_compliance(self):
        """Test that new components follow required interface"""
        
        # Required interface
        required_methods = ['forward', 'get_output_dim']
        
        # Test component
        class TestComponent:
            def forward(self, x):
                return x
            
            def get_output_dim(self):
                return 64
        
        component = TestComponent()
        
        # Check interface compliance
        for method in required_methods:
            assert hasattr(component, method), f"Missing required method: {method}"
            assert callable(getattr(component, method)), f"Method {method} is not callable"
    
    def test_factory_extensibility(self):
        """Test that factory can handle new component types"""
        
        class ExtensibleFactory:
            def __init__(self):
                self.creators = {}
            
            def register(self, component_type, creator_func):
                self.creators[component_type] = creator_func
            
            def create(self, component_type, config):
                if component_type not in self.creators:
                    raise ValueError(f"Unknown component type: {component_type}")
                return self.creators[component_type](config)
        
        # Create factory
        factory = ExtensibleFactory()
        
        # Register component types
        factory.register("simple", lambda config: SimpleTestComponent())
        
        # Should work
        component = factory.create("simple", {})
        assert component is not None
        
        # Should fail for unregistered type
        with pytest.raises(ValueError):
            factory.create("unknown", {})


class SimpleTestComponent:
    def forward(self, x):
        return x


if __name__ == "__main__":
    pytest.main([__file__, "-v"])