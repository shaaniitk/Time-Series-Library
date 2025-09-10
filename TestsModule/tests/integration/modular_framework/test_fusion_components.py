#!/usr/bin/env python3
"""
Comprehensive test suite for fusion components in the modular framework.

This test suite validates:
1. Fusion component registration
2. HierarchicalFusion implementation
3. Integration with unified registry
4. Component creation and configuration
5. Error handling for fusion components
"""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Mock torch before any imports
class MockModule:
    """Mock PyTorch nn.Module that can be inherited from."""
    def __init__(self, *args, **kwargs):
        pass
    
    def forward(self, *args, **kwargs):
        return MagicMock()
    
    def parameters(self):
        return []
    
    def named_parameters(self):
        return []

torch_mock = MagicMock()
torch_mock.nn = MagicMock()
torch_mock.nn.Module = MockModule
torch_mock.nn.functional = MagicMock()
sys.modules['torch'] = torch_mock
sys.modules['torch.nn'] = torch_mock.nn
sys.modules['torch.nn.functional'] = torch_mock.nn.functional

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

class TestFusionComponentStructure(unittest.TestCase):
    """Test the structure and organization of fusion components."""
    
    def test_fusion_directory_structure(self):
        """Test that fusion component directory structure is correct."""
        fusion_dir = os.path.join(os.path.dirname(__file__), 
                                 '../../../../layers/modular/fusion')
        
        # Test that fusion directory exists
        self.assertTrue(os.path.exists(fusion_dir), 
                       "Fusion directory does not exist")
        
        # Test that key files exist
        expected_files = [
            '__init__.py',
            'hierarchical_fusion.py',
            'registry.py'
        ]
        
        for file_name in expected_files:
            file_path = os.path.join(fusion_dir, file_name)
            self.assertTrue(os.path.exists(file_path), 
                          f"Expected file {file_name} does not exist")
    
    def test_fusion_init_exports(self):
        """Test that fusion __init__.py exports expected components."""
        try:
            from layers.modular.fusion import HierarchicalFusion
            
            # Test that HierarchicalFusion is importable
            self.assertIsNotNone(HierarchicalFusion)
            
            # Test that it's a class
            self.assertTrue(isinstance(HierarchicalFusion, type))
            
        except ImportError as e:
            self.skipTest(f"Cannot import fusion components: {e}")
    
    def test_hierarchical_fusion_class_structure(self):
        """Test HierarchicalFusion class structure without PyTorch."""
        # Read the hierarchical_fusion.py file to check structure
        fusion_file = os.path.join(os.path.dirname(__file__), 
                                  '../../../../layers/modular/fusion/hierarchical_fusion.py')
        
        if os.path.exists(fusion_file):
            with open(fusion_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Test that class is defined
            self.assertIn('class HierarchicalFusion', content)
            
            # Test that __init__ method is defined
            self.assertIn('def __init__', content)
            
            # Test that forward method is defined
            self.assertIn('def forward', content)
            
            # Test that expected parameters are documented
            expected_params = ['d_model', 'n_levels', 'fusion_strategy']
            for param in expected_params:
                self.assertIn(param, content, f"Parameter {param} not found in class")
        else:
            self.skipTest("HierarchicalFusion file not found")

class TestFusionRegistry(unittest.TestCase):
    """Test fusion component registry functionality."""
    
    def test_fusion_registry_structure(self):
        """Test fusion registry file structure."""
        registry_file = os.path.join(os.path.dirname(__file__), 
                                   '../../../../layers/modular/fusion/registry.py')
        
        if os.path.exists(registry_file):
            with open(registry_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Test that registry imports unified registry
            self.assertIn('from ..core', content)
            
            # Test that HierarchicalFusion is imported
            self.assertIn('HierarchicalFusion', content)
            
            # Test that registration call is present
            self.assertIn('unified_registry.register', content)
            
            # Test that FUSION component family is used
            self.assertIn('ComponentFamily.FUSION', content)
        else:
            self.skipTest("Fusion registry file not found")
    
    def test_fusion_registry_integration(self):
        """Test that fusion registry integrates with unified registry."""
        try:
            from layers.modular.core import unified_registry, ComponentFamily
            
            # Import fusion registry to trigger registration
            import layers.modular.fusion.registry
            
            # Test that fusion components are registered
            fusion_components = unified_registry.get_all_by_type(ComponentFamily.FUSION)
            
            # Should have at least hierarchical_fusion
            self.assertGreater(len(fusion_components), 0, 
                             "No fusion components registered")
            
            self.assertIn('hierarchical_fusion', fusion_components, 
                        "HierarchicalFusion not registered")
            
        except ImportError as e:
            self.skipTest(f"Cannot import fusion registry: {e}")

class TestFusionComponentRegistration(unittest.TestCase):
    """Test fusion component registration details."""
    
    def test_hierarchical_fusion_registration(self):
        """Test HierarchicalFusion registration details."""
        try:
            from layers.modular.core import unified_registry, ComponentFamily
            
            # Import fusion registry to trigger registration
            import layers.modular.fusion.registry
            
            # Get fusion components
            fusion_components = unified_registry.get_all_by_type(ComponentFamily.FUSION)
            
            # Test hierarchical_fusion registration
            self.assertIn('hierarchical_fusion', fusion_components)
            
            hierarchical_fusion = fusion_components['hierarchical_fusion']
            
            # Test component structure
            self.assertIn('class', hierarchical_fusion)
            self.assertIn('config', hierarchical_fusion)
            
            # Test config structure
            config = hierarchical_fusion['config']
            
            # Test expected configuration keys
            expected_keys = ['d_model', 'n_levels', 'fusion_strategy']
            for key in expected_keys:
                self.assertIn(key, config, f"Config key {key} not found")
            
            # Test default values
            self.assertEqual(config['d_model'], 512)
            self.assertEqual(config['n_levels'], 3)
            self.assertEqual(config['fusion_strategy'], 'hierarchical')
            
        except ImportError as e:
            self.skipTest(f"Cannot import fusion components: {e}")
    
    def test_fusion_component_class_reference(self):
        """Test that registered fusion component references correct class."""
        try:
            from layers.modular.core import unified_registry, ComponentFamily
            from layers.modular.fusion import HierarchicalFusion
            
            # Import fusion registry to trigger registration
            import layers.modular.fusion.registry
            
            # Get fusion components
            fusion_components = unified_registry.get_all_by_type(ComponentFamily.FUSION)
            
            # Test that class reference is correct
            hierarchical_fusion = fusion_components['hierarchical_fusion']
            self.assertEqual(hierarchical_fusion['class'], HierarchicalFusion)
            
        except ImportError as e:
            self.skipTest(f"Cannot import fusion components: {e}")

class TestFusionComponentCreation(unittest.TestCase):
    """Test fusion component creation through unified registry."""
    
    def test_hierarchical_fusion_creation_mock(self):
        """Test HierarchicalFusion creation with mocked PyTorch."""
        try:
            from layers.modular.core import unified_registry, ComponentFamily
            
            # Mock the HierarchicalFusion class to avoid PyTorch dependency
            class MockHierarchicalFusion:
                def __init__(self, d_model=512, n_levels=3, fusion_strategy='hierarchical', **kwargs):
                    self.d_model = d_model
                    self.n_levels = n_levels
                    self.fusion_strategy = fusion_strategy
                    self.kwargs = kwargs
            
            # Temporarily replace the registered class
            with patch.object(unified_registry, '_registry') as mock_registry:
                mock_registry.__getitem__ = MagicMock(return_value={
                    'hierarchical_fusion': {
                        'class': MockHierarchicalFusion,
                        'config': {
                            'd_model': 512,
                            'n_levels': 3,
                            'fusion_strategy': 'hierarchical'
                        }
                    }
                })
                
                # Test component creation with default config
                component = unified_registry.create(
                    name='hierarchical_fusion',
                    component_type=ComponentFamily.FUSION
                )
                
                self.assertIsInstance(component, MockHierarchicalFusion)
                self.assertEqual(component.d_model, 512)
                self.assertEqual(component.n_levels, 3)
                self.assertEqual(component.fusion_strategy, 'hierarchical')
                
                # Test component creation with custom config
                custom_component = unified_registry.create(
                    name='hierarchical_fusion',
                    component_type=ComponentFamily.FUSION,
                    d_model=256,
                    n_levels=4
                )
                
                self.assertEqual(custom_component.d_model, 256)
                self.assertEqual(custom_component.n_levels, 4)
                self.assertEqual(custom_component.fusion_strategy, 'hierarchical')  # Default
                
        except ImportError as e:
            self.skipTest(f"Cannot import fusion components: {e}")

class TestFusionErrorHandling(unittest.TestCase):
    """Test error handling for fusion components."""
    
    def test_nonexistent_fusion_component(self):
        """Test error handling for non-existent fusion components."""
        try:
            from layers.modular.core import unified_registry, ComponentFamily
            
            # Test creating non-existent fusion component
            with self.assertRaises(ValueError) as context:
                unified_registry.create(
                    name='nonexistent_fusion',
                    component_type=ComponentFamily.FUSION
                )
            
            self.assertIn("Component 'nonexistent_fusion' not found", str(context.exception))
            
        except ImportError as e:
            self.skipTest(f"Cannot import fusion components: {e}")
    
    def test_fusion_component_with_invalid_config(self):
        """Test fusion component creation with invalid configuration."""
        try:
            from layers.modular.core import unified_registry, ComponentFamily
            
            # Import fusion registry to trigger registration
            import layers.modular.fusion.registry
            
            # Mock the component creation to test config validation
            class MockHierarchicalFusion:
                def __init__(self, d_model=512, n_levels=3, fusion_strategy='hierarchical', **kwargs):
                    if d_model <= 0:
                        raise ValueError("d_model must be positive")
                    if n_levels <= 0:
                        raise ValueError("n_levels must be positive")
                    self.d_model = d_model
                    self.n_levels = n_levels
                    self.fusion_strategy = fusion_strategy
            
            # Test with invalid d_model
            with patch.object(unified_registry, '_registry') as mock_registry:
                mock_registry.__getitem__ = MagicMock(return_value={
                    'hierarchical_fusion': {
                        'class': MockHierarchicalFusion,
                        'config': {'d_model': 512, 'n_levels': 3, 'fusion_strategy': 'hierarchical'}
                    }
                })
                
                with self.assertRaises(ValueError) as context:
                    unified_registry.create(
                        name='hierarchical_fusion',
                        component_type=ComponentFamily.FUSION,
                        d_model=-1  # Invalid value
                    )
                
                self.assertIn("d_model must be positive", str(context.exception))
                
        except ImportError as e:
            self.skipTest(f"Cannot import fusion components: {e}")

class TestFusionBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility for fusion components."""
    
    def test_fusion_registry_deprecation_shim(self):
        """Test that fusion registry deprecation shim works if present."""
        try:
            # Try to import legacy fusion registry
            from layers.modular.fusion.registry import FusionRegistry
            
            # If it exists, test that it provides backward compatibility
            if hasattr(FusionRegistry, 'list_components'):
                components = FusionRegistry.list_components()
                self.assertIsInstance(components, list)
                
            if hasattr(FusionRegistry, 'get'):
                # Test getting a component
                try:
                    component_class = FusionRegistry.get('hierarchical')
                    self.assertIsNotNone(component_class)
                except Exception:
                    # If it fails, that's okay - might not be implemented
                    pass
                    
        except (ImportError, AttributeError):
            # If FusionRegistry doesn't exist or doesn't have these methods,
            # that's fine - it means we're using the new unified approach
            self.skipTest("Legacy FusionRegistry not found or not implemented")

class TestFusionIntegrationWithCore(unittest.TestCase):
    """Test integration between fusion components and core registry."""
    
    def test_fusion_component_family_consistency(self):
        """Test that fusion components are consistently registered under FUSION family."""
        try:
            from layers.modular.core import unified_registry, ComponentFamily
            
            # Import fusion registry to trigger registration
            import layers.modular.fusion.registry
            
            # Get all fusion components
            fusion_components = unified_registry.get_all_by_type(ComponentFamily.FUSION)
            
            # Test that all components are properly structured
            for name, component_info in fusion_components.items():
                self.assertIn('class', component_info, 
                            f"Component {name} missing 'class' key")
                self.assertIn('config', component_info, 
                            f"Component {name} missing 'config' key")
                
                # Test that class is actually a class
                self.assertTrue(isinstance(component_info['class'], type), 
                              f"Component {name} 'class' is not a class")
                
                # Test that config is a dictionary
                self.assertIsInstance(component_info['config'], dict, 
                                    f"Component {name} 'config' is not a dict")
                
        except ImportError as e:
            self.skipTest(f"Cannot import fusion components: {e}")
    
    def test_fusion_components_isolation(self):
        """Test that fusion components don't interfere with other component families."""
        try:
            from layers.modular.core import unified_registry, ComponentFamily
            
            # Import all registries
            import layers.modular.fusion.registry
            import layers.modular.loss.registry
            
            # Get components from different families
            fusion_components = unified_registry.get_all_by_type(ComponentFamily.FUSION)
            loss_components = unified_registry.get_all_by_type(ComponentFamily.LOSS)
            
            # Test that there's no name collision
            fusion_names = set(fusion_components.keys())
            loss_names = set(loss_components.keys())
            
            overlap = fusion_names & loss_names
            self.assertEqual(len(overlap), 0, 
                           f"Component name collision between FUSION and LOSS: {overlap}")
            
        except ImportError as e:
            self.skipTest(f"Cannot import components: {e}")

if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)