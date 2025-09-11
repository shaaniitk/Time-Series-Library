#!/usr/bin/env python3
"""
Comprehensive test suite for loss components in the modular framework.

This test suite validates:
1. Loss component registration
2. Various loss implementations
3. Integration with unified registry
4. Component creation and configuration
5. Backward compatibility with legacy registry
6. Error handling for loss components
"""

import unittest
import sys
import os
import warnings
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

class TestLossComponentStructure(unittest.TestCase):
    """Test the structure and organization of loss components."""
    
    def test_loss_directory_structure(self):
        """Test that loss component directory structure is correct."""
        loss_dir = os.path.join(os.path.dirname(__file__), 
                               '../../../../layers/modular/loss')
        
        # Test that loss directory exists
        self.assertTrue(os.path.exists(loss_dir), 
                       "Loss directory does not exist")
        
        # Test that key files exist
        expected_files = [
            '__init__.py',
            'registry.py'
        ]
        
        for file_name in expected_files:
            file_path = os.path.join(loss_dir, file_name)
            self.assertTrue(os.path.exists(file_path), 
                          f"Expected file {file_name} does not exist")
    
    def test_loss_component_files(self):
        """Test that individual loss component files exist."""
        loss_dir = os.path.join(os.path.dirname(__file__), 
                               '../../../../layers/modular/loss')
        
        # Expected loss component files
        expected_loss_files = [
            'quantile_loss.py',
            'pinball_loss.py', 
            'mape_loss.py',
            'focal_loss.py',
            'huber_loss.py',
            'adaptive_loss.py',
            'bayesian_loss.py'
        ]
        
        # Shim files that only contain imports
        shim_files = {'bayesian_loss.py'}
        
        for file_name in expected_loss_files:
            file_path = os.path.join(loss_dir, file_name)
            if os.path.exists(file_path):
                # If file exists, check it has basic structure
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if file_name in shim_files:
                    # For shim files, check for imports and __all__
                    self.assertIn('from .', content, f"{file_name} should contain relative imports")
                    self.assertIn('__all__', content, f"{file_name} should have __all__ export list")
                else:
                    # For regular files, check for class definitions
                    self.assertIn('class', content, f"{file_name} should contain a class definition")
    
    def test_loss_init_structure(self):
        """Test loss __init__.py structure."""
        init_file = os.path.join(os.path.dirname(__file__), 
                                '../../../../layers/modular/loss/__init__.py')
        
        if os.path.exists(init_file):
            with open(init_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Test that it imports from registry
            self.assertIn('registry', content.lower())

# TestLossRegistry class removed - redundant with unified registry tests in test_unified_registry_comprehensive.py

class TestLossComponentRegistration(unittest.TestCase):
    """Test loss component registration with unified registry."""
    
    def test_loss_components_registered(self):
        """Test that loss components are registered in unified registry."""
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
    
    def test_specific_loss_registrations(self):
        """Test specific loss component registrations."""
        try:
            from layers.modular.core import unified_registry, ComponentFamily
            
            # Import register_components to populate registry
            import layers.modular.core.register_components
            
            # Get loss components
            loss_components = unified_registry.get_all_by_type(ComponentFamily.LOSS)
            
            # Test quantile_loss registration
            if "quantile_loss" in loss_components:
                quantile_loss = loss_components["quantile_loss"]
                config = quantile_loss["config"]
                
                # Test expected config keys
                expected_keys = ["quantiles", "reduction"]
                for key in expected_keys:
                    self.assertIn(key, config, f"Config key {key} not found in quantile_loss")
            
            # Test focal_loss registration
            if "focal_loss" in loss_components:
                focal_loss = loss_components["focal_loss"]
                config = focal_loss["config"]
                
                # Test expected config keys
                expected_keys = ["alpha", "gamma", "reduction"]
                for key in expected_keys:
                    self.assertIn(key, config, f"Config key {key} not found in focal_loss")
                
        except ImportError as e:
            self.skipTest(f"Cannot import loss components: {e}")

class TestLossComponentCreation(unittest.TestCase):
    """Test loss component creation through unified registry."""
    
    def test_loss_component_creation_mock(self):
        """Test loss component creation with mocked PyTorch."""
        try:
            from layers.modular.core import unified_registry, ComponentFamily
            
            # Mock loss component classes to avoid PyTorch dependency
            class MockQuantileLoss:
                def __init__(self, quantiles=[0.1, 0.5, 0.9], reduction='mean', **kwargs):
                    self.quantiles = quantiles
                    self.reduction = reduction
                    self.kwargs = kwargs
            
            class MockFocalLoss:
                def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', **kwargs):
                    self.alpha = alpha
                    self.gamma = gamma
                    self.reduction = reduction
                    self.kwargs = kwargs
            
            # Temporarily replace the registered classes
            with patch.object(unified_registry, '_registry') as mock_registry:
                mock_registry.__getitem__ = MagicMock(return_value={
                    'quantile_loss': {
                        'class': MockQuantileLoss,
                        'config': {
                            'quantiles': [0.1, 0.5, 0.9],
                            'reduction': 'mean'
                        }
                    },
                    'focal_loss': {
                        'class': MockFocalLoss,
                        'config': {
                            'alpha': 1.0,
                            'gamma': 2.0,
                            'reduction': 'mean'
                        }
                    }
                })
                
                # Test quantile loss creation with default config
                quantile_component = unified_registry.create(
                    name='quantile_loss',
                    component_type=ComponentFamily.LOSS
                )
                
                self.assertIsInstance(quantile_component, MockQuantileLoss)
                self.assertEqual(quantile_component.quantiles, [0.1, 0.5, 0.9])
                self.assertEqual(quantile_component.reduction, 'mean')
                
                # Test focal loss creation with custom config
                focal_component = unified_registry.create(
                    name='focal_loss',
                    component_type=ComponentFamily.LOSS,
                    alpha=0.5,
                    gamma=3.0
                )
                
                self.assertEqual(focal_component.alpha, 0.5)
                self.assertEqual(focal_component.gamma, 3.0)
                self.assertEqual(focal_component.reduction, 'mean')  # Default
                
        except ImportError as e:
            self.skipTest(f"Cannot import loss components: {e}")

class TestLossErrorHandling(unittest.TestCase):
    """Test error handling for loss components."""
    
    def test_nonexistent_loss_component(self):
        """Test error handling for non-existent loss components."""
        try:
            from layers.modular.core import unified_registry, ComponentFamily
            
            # Test creating non-existent loss component
            with self.assertRaises(ValueError) as context:
                unified_registry.create(
                    name='nonexistent_loss',
                    component_type=ComponentFamily.LOSS
                )
            
            self.assertIn("Component 'nonexistent_loss' not found", str(context.exception))
            
        except ImportError as e:
            self.skipTest(f"Cannot import loss components: {e}")
    
    def test_loss_component_with_invalid_config(self):
        """Test loss component creation with invalid configuration."""
        try:
            from layers.modular.core import unified_registry, ComponentFamily
            
            # Mock loss component that validates config
            class MockValidatingLoss:
                def __init__(self, alpha=1.0, gamma=2.0, **kwargs):
                    if alpha <= 0:
                        raise ValueError("alpha must be positive")
                    if gamma < 0:
                        raise ValueError("gamma must be non-negative")
                    self.alpha = alpha
                    self.gamma = gamma
            
            # Test with invalid config
            with patch.object(unified_registry, '_registry') as mock_registry:
                mock_registry.__getitem__ = MagicMock(return_value={
                    'validating_loss': {
                        'class': MockValidatingLoss,
                        'config': {'alpha': 1.0, 'gamma': 2.0}
                    }
                })
                
                with self.assertRaises(ValueError) as context:
                    unified_registry.create(
                        name='validating_loss',
                        component_type=ComponentFamily.LOSS,
                        alpha=-1.0  # Invalid value
                    )
                
                self.assertIn("alpha must be positive", str(context.exception))
                
        except ImportError as e:
            self.skipTest(f"Cannot import loss components: {e}")

class TestLossComponentCategories(unittest.TestCase):
    """Test different categories of loss components."""
    
    def test_standard_loss_components(self):
        """Test standard loss components registration."""
        try:
            from layers.modular.core import unified_registry, ComponentFamily
            
            # Import register_components to populate registry
            import layers.modular.core.register_components
            
            # Get loss components
            loss_components = unified_registry.get_all_by_type(ComponentFamily.LOSS)
            
            # Test standard losses
            standard_losses = ["mse_loss", "mae_loss", "huber_loss"]
            
            for loss_name in standard_losses:
                if loss_name in loss_components:
                    component = loss_components[loss_name]
                    self.assertIn("class", component)
                    self.assertIn("config", component)
                    
        except ImportError as e:
            self.skipTest(f"Cannot import loss components: {e}")
    
    def test_advanced_loss_components(self):
        """Test advanced loss components registration."""
        try:
            from layers.modular.core import unified_registry, ComponentFamily
            
            # Import register_components to populate registry
            import layers.modular.core.register_components
            
            # Get loss components
            loss_components = unified_registry.get_all_by_type(ComponentFamily.LOSS)
            
            # Test advanced losses
            advanced_losses = ["quantile_loss", "pinball_loss", "focal_loss"]
            
            for loss_name in advanced_losses:
                if loss_name in loss_components:
                    component = loss_components[loss_name]
                    self.assertIn("class", component)
                    self.assertIn("config", component)
                    
                    # Advanced losses should have more complex configs
                    config = component["config"]
                    self.assertGreater(len(config), 0, 
                                     f"{loss_name} should have configuration parameters")
                    
        except ImportError as e:
            self.skipTest(f"Cannot import loss components: {e}")
    
    def test_bayesian_loss_components(self):
        """Test Bayesian loss components registration."""
        try:
            from layers.modular.core import unified_registry, ComponentFamily
            
            # Import register_components to populate registry
            import layers.modular.core.register_components
            
            # Get loss components
            loss_components = unified_registry.get_all_by_type(ComponentFamily.LOSS)
            
            # Test Bayesian losses
            bayesian_losses = ["variational_loss", "kl_divergence_loss"]
            
            for loss_name in bayesian_losses:
                if loss_name in loss_components:
                    component = loss_components[loss_name]
                    self.assertIn("class", component)
                    self.assertIn("config", component)
                    
        except ImportError as e:
            self.skipTest(f"Cannot import loss components: {e}")

class TestLossIntegrationWithCore(unittest.TestCase):
    """Test integration between loss components and core registry."""
    
    def test_loss_component_family_consistency(self):
        """Test that loss components are consistently registered under LOSS family."""
        try:
            from layers.modular.core import unified_registry, ComponentFamily
            
            # Import register_components to populate registry
            import layers.modular.core.register_components
            
            # Get all loss components
            loss_components = unified_registry.get_all_by_type(ComponentFamily.LOSS)
            
            # Test that all components are properly structured
            for name, component_info in loss_components.items():
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
            self.skipTest(f"Cannot import loss components: {e}")
    
    def test_loss_components_isolation(self):
        """Test that loss components don't interfere with other component families."""
        try:
            from layers.modular.core import unified_registry, ComponentFamily
            
            # Import all registries
            import layers.modular.core.register_components
            
            # Get components from different families
            loss_components = unified_registry.get_all_by_type(ComponentFamily.LOSS)
            fusion_components = unified_registry.get_all_by_type(ComponentFamily.FUSION)
            
            # Test that there's no name collision
            loss_names = set(loss_components.keys())
            fusion_names = set(fusion_components.keys())
            
            overlap = loss_names & fusion_names
            self.assertEqual(len(overlap), 0, 
                           f"Component name collision between LOSS and FUSION: {overlap}")
            
        except ImportError as e:
            self.skipTest(f"Cannot import components: {e}")
    
    def test_loss_registry_consistency_with_unified(self):
        """Test that loss registry is consistent with unified registry."""
        try:
            from layers.modular.core import unified_registry, ComponentFamily
            from layers.modular.loss.registry import LossRegistry
            
            # Import register_components to populate registry
            import layers.modular.core.register_components
            
            # Get components from both registries
            unified_components = unified_registry.get_all_by_type(ComponentFamily.LOSS)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                legacy_components = LossRegistry.list_components()
            
            # Test that legacy registry returns reasonable results
            self.assertIsInstance(legacy_components, list)
            
            # If legacy registry has components, they should be consistent
            if len(legacy_components) > 0:
                # At least some components should be in both
                unified_names = set(unified_components.keys())
                legacy_names = set(legacy_components)
                
                # There should be some overlap (but not necessarily complete)
                overlap = unified_names & legacy_names
                self.assertGreater(len(overlap), 0, 
                                 "No overlap between unified and legacy loss registries")
                
        except ImportError as e:
            self.skipTest(f"Cannot import loss registries: {e}")

if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)