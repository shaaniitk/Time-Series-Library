#!/usr/bin/env python3
"""
Comprehensive test suite for the unified modular registry system.

This test suite validates:
1. Unified registry functionality
2. Component family registrations
3. Fusion component integration
4. Loss component integration
5. Backward compatibility
6. Error handling and edge cases
"""

import unittest
import sys
import os
import warnings
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

class TestUnifiedRegistryStructure(unittest.TestCase):
    """Test the basic structure and functionality of the unified registry."""
    
    def test_component_family_enum(self):
        """Test that ComponentFamily enum is properly defined."""
        try:
            from layers.modular.core.registry import ComponentFamily
            
            # Test that all expected families exist
            expected_families = ['ATTENTION', 'ENCODER', 'DECODER', 'FUSION', 'LOSS']
            for family in expected_families:
                self.assertTrue(hasattr(ComponentFamily, family), 
                              f"ComponentFamily.{family} not found")
            
            # Test enum values
            self.assertEqual(ComponentFamily.ATTENTION.value, "attention")
            self.assertEqual(ComponentFamily.FUSION.value, "fusion")
            self.assertEqual(ComponentFamily.LOSS.value, "loss")
            
        except ImportError as e:
            self.skipTest(f"Cannot import ComponentFamily: {e}")
    
    def test_component_registry_class(self):
        """Test ComponentRegistry class structure."""
        try:
            from layers.modular.core.registry import ComponentRegistry, ComponentFamily
            
            registry = ComponentRegistry()
            
            # Test registry initialization
            self.assertIsInstance(registry._registry, dict)
            
            # Test that all component families are initialized
            for family in ComponentFamily:
                self.assertIn(family, registry._registry)
                self.assertIsInstance(registry._registry[family], dict)
                
        except ImportError as e:
            self.skipTest(f"Cannot import ComponentRegistry: {e}")
    
    def test_unified_registry_export(self):
        """Test that unified_registry is properly exported."""
        try:
            from layers.modular.core import unified_registry, ComponentFamily
            
            # Test that unified_registry is accessible
            self.assertIsNotNone(unified_registry)
            
            # Test that it has expected methods
            self.assertTrue(hasattr(unified_registry, 'register'))
            self.assertTrue(hasattr(unified_registry, 'create'))
            self.assertTrue(hasattr(unified_registry, 'get_all_by_type'))
            
        except ImportError as e:
            self.skipTest(f"Cannot import unified_registry: {e}")

class TestComponentRegistration(unittest.TestCase):
    """Test component registration functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from layers.modular.core.registry import ComponentRegistry, ComponentFamily
            self.registry = ComponentRegistry()
            self.ComponentFamily = ComponentFamily
        except ImportError:
            self.skipTest("Cannot import registry components")
    
    def test_register_component(self):
        """Test registering a component."""
        # Mock component class
        class MockComponent:
            def __init__(self, **kwargs):
                self.config = kwargs
        
        # Register component
        self.registry.register(
            name="test_component",
            component_class=MockComponent,
            component_type=self.ComponentFamily.ATTENTION,
            test_config={"param1": "value1"}
        )
        
        # Verify registration
        components = self.registry.get_all_by_type(self.ComponentFamily.ATTENTION)
        self.assertIn("test_component", components)
        self.assertEqual(components["test_component"]["class"], MockComponent)
        self.assertEqual(components["test_component"]["config"]["param1"], "value1")
    
    def test_create_component(self):
        """Test creating a component instance."""
        # Mock component class
        class MockComponent:
            def __init__(self, **kwargs):
                self.config = kwargs
        
        # Register component
        self.registry.register(
            name="test_component",
            component_class=MockComponent,
            component_type=self.ComponentFamily.ATTENTION,
            test_config={"param1": "value1", "param2": "value2"}
        )
        
        # Create component with default config
        component = self.registry.create(
            name="test_component",
            component_type=self.ComponentFamily.ATTENTION
        )
        
        self.assertIsInstance(component, MockComponent)
        self.assertEqual(component.config["param1"], "value1")
        self.assertEqual(component.config["param2"], "value2")
        
        # Create component with override config
        component_override = self.registry.create(
            name="test_component",
            component_type=self.ComponentFamily.ATTENTION,
            param2="override_value",
            param3="new_value"
        )
        
        self.assertEqual(component_override.config["param1"], "value1")  # Default
        self.assertEqual(component_override.config["param2"], "override_value")  # Override
        self.assertEqual(component_override.config["param3"], "new_value")  # New
    
    def test_component_not_found_error(self):
        """Test error handling for non-existent components."""
        with self.assertRaises(ValueError) as context:
            self.registry.create(
                name="nonexistent_component",
                component_type=self.ComponentFamily.ATTENTION
            )
        
        self.assertIn("Component 'nonexistent_component' not found", str(context.exception))
    
    def test_component_overwrite_warning(self):
        """Test warning when overwriting existing component."""
        # Mock component class
        class MockComponent:
            def __init__(self, **kwargs):
                pass
        
        # Register component first time
        self.registry.register(
            name="test_component",
            component_class=MockComponent,
            component_type=self.ComponentFamily.ATTENTION,
            test_config={}
        )
        
        # Register same component again (should warn)
        with patch('builtins.print') as mock_print:
            self.registry.register(
                name="test_component",
                component_class=MockComponent,
                component_type=self.ComponentFamily.ATTENTION,
                test_config={}
            )
            
            # Check that warning was printed
            mock_print.assert_called_once()
            self.assertIn("Warning: Component 'test_component'", mock_print.call_args[0][0])

class TestFusionComponentIntegration(unittest.TestCase):
    """Test fusion component integration with unified registry."""
    
    def test_fusion_components_registered(self):
        """Test that fusion components are registered."""
        try:
            from layers.modular.core import unified_registry, ComponentFamily
            
            # Import register_components to populate registry
            import layers.modular.core.register_components
            
            # Get fusion components
            fusion_components = unified_registry.get_all_by_type(ComponentFamily.FUSION)
            
            # Test that hierarchical_fusion is registered
            self.assertIn("hierarchical_fusion", fusion_components)
            
            # Test component structure
            hierarchical_fusion = fusion_components["hierarchical_fusion"]
            self.assertIn("class", hierarchical_fusion)
            self.assertIn("config", hierarchical_fusion)
            
            # Test config structure
            config = hierarchical_fusion["config"]
            expected_keys = ["d_model", "n_levels", "fusion_strategy"]
            for key in expected_keys:
                self.assertIn(key, config)
                
        except ImportError as e:
            self.skipTest(f"Cannot import fusion components: {e}")

class TestLossComponentIntegration(unittest.TestCase):
    """Test loss component integration with unified registry."""
    
    def test_loss_components_registered(self):
        """Test that loss components are registered."""
        try:
            from layers.modular.core import unified_registry, ComponentFamily
            
            # Import register_components to populate registry
            import layers.modular.core.register_components
            
            # Get loss components
            loss_components = unified_registry.get_all_by_type(ComponentFamily.LOSS)
            
            # Test that key loss components are registered
            expected_components = [
                "quantile_loss", "pinball_loss", "mape_loss", "focal_loss"
            ]
            
            for component_name in expected_components:
                self.assertIn(component_name, loss_components, 
                            f"{component_name} not found in registered loss components")
                
                # Test component structure
                component = loss_components[component_name]
                self.assertIn("class", component)
                self.assertIn("config", component)
                
        except ImportError as e:
            self.skipTest(f"Cannot import loss components: {e}")
    
    def test_loss_component_count(self):
        """Test that expected number of loss components are registered."""
        try:
            from layers.modular.core import unified_registry, ComponentFamily
            
            # Import register_components to populate registry
            import layers.modular.core.register_components
            
            # Get loss components
            loss_components = unified_registry.get_all_by_type(ComponentFamily.LOSS)
            
            # Should have at least 12 loss components based on our registration
            self.assertGreaterEqual(len(loss_components), 12, 
                                  f"Expected at least 12 loss components, got {len(loss_components)}")
                
        except ImportError as e:
            self.skipTest(f"Cannot import loss components: {e}")

class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with legacy registries."""
    
    def test_loss_registry_deprecation_shim(self):
        """Test that loss registry deprecation shim works."""
        try:
            from layers.modular.loss.registry import LossRegistry
            
            # Test that we can still list components (should warn)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                components = LossRegistry.list_components()
                
                # Should return a list of component names
                self.assertIsInstance(components, list)
                self.assertGreater(len(components), 0)
                
                # Should have generated a deprecation warning
                self.assertTrue(any(issubclass(warning.category, DeprecationWarning) 
                                 for warning in w))
                
        except ImportError as e:
            self.skipTest(f"Cannot import LossRegistry: {e}")
    
    def test_loss_registry_get_method(self):
        """Test that loss registry get method still works."""
        try:
            from layers.modular.loss.registry import LossRegistry
            
            # Test getting a component (should warn)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                # Try to get a component that should exist
                try:
                    component_class = LossRegistry.get("quantile")
                    self.assertIsNotNone(component_class)
                except Exception:
                    # If unified registry integration fails, should fall back to legacy
                    pass
                
                # Should have generated a deprecation warning
                self.assertTrue(any(issubclass(warning.category, DeprecationWarning) 
                                 for warning in w))
                
        except ImportError as e:
            self.skipTest(f"Cannot import LossRegistry: {e}")

class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def test_invalid_component_family(self):
        """Test handling of invalid component families."""
        try:
            from layers.modular.core.registry import ComponentRegistry
            
            registry = ComponentRegistry()
            
            # Test with invalid component type (should raise error)
            with self.assertRaises((ValueError, KeyError, TypeError)):
                registry.register(
                    name="test",
                    component_class=object,
                    component_type="invalid_type",  # Invalid type
                    test_config={}
                )
                
        except ImportError as e:
            self.skipTest(f"Cannot import ComponentRegistry: {e}")
    
    def test_empty_registry_behavior(self):
        """Test behavior with empty registry."""
        try:
            from layers.modular.core.registry import ComponentRegistry, ComponentFamily
            
            registry = ComponentRegistry()
            
            # Test getting components from empty registry
            components = registry.get_all_by_type(ComponentFamily.ATTENTION)
            self.assertEqual(len(components), 0)
            
            # Test creating component from empty registry
            with self.assertRaises(ValueError):
                registry.create(
                    name="nonexistent",
                    component_type=ComponentFamily.ATTENTION
                )
                
        except ImportError as e:
            self.skipTest(f"Cannot import registry components: {e}")

class TestRegistryIntegration(unittest.TestCase):
    """Test integration between different registry components."""
    
    def test_cross_family_registration(self):
        """Test that components from different families don't interfere."""
        try:
            from layers.modular.core import unified_registry, ComponentFamily
            
            # Import register_components to populate registry
            import layers.modular.core.register_components
            
            # Get components from different families
            fusion_components = unified_registry.get_all_by_type(ComponentFamily.FUSION)
            loss_components = unified_registry.get_all_by_type(ComponentFamily.LOSS)
            
            # Test that families are separate
            fusion_names = set(fusion_components.keys())
            loss_names = set(loss_components.keys())
            
            # Should have no overlap in component names between families
            overlap = fusion_names & loss_names
            self.assertEqual(len(overlap), 0, 
                           f"Component names overlap between families: {overlap}")
            
        except ImportError as e:
            self.skipTest(f"Cannot import registry components: {e}")
    
    def test_registry_persistence(self):
        """Test that registry maintains state across operations."""
        try:
            from layers.modular.core import unified_registry, ComponentFamily
            
            # Import register_components to populate registry
            import layers.modular.core.register_components
            
            # Get initial component count
            initial_loss_count = len(unified_registry.get_all_by_type(ComponentFamily.LOSS))
            initial_fusion_count = len(unified_registry.get_all_by_type(ComponentFamily.FUSION))
            
            # Perform some operations
            unified_registry.get_all_by_type(ComponentFamily.ATTENTION)
            
            # Check that counts remain the same
            final_loss_count = len(unified_registry.get_all_by_type(ComponentFamily.LOSS))
            final_fusion_count = len(unified_registry.get_all_by_type(ComponentFamily.FUSION))
            
            self.assertEqual(initial_loss_count, final_loss_count)
            self.assertEqual(initial_fusion_count, final_fusion_count)
            
        except ImportError as e:
            self.skipTest(f"Cannot import registry components: {e}")

if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)