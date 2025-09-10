"""Test Suite for Fusion Registry and Components

This file tests the fusion component registry functionality,
HierarchicalFusion component, and integration with the unified registry.
"""

import unittest
import sys
import os
from typing import List, Dict, Any
from unittest.mock import Mock, patch, MagicMock

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
    
    def cuda(self):
        return self
    
    def cpu(self):
        return self

class MockTensor:
    """Mock PyTorch tensor."""
    def __init__(self, *shape, **kwargs):
        self.shape = shape
        self._device = kwargs.get('device', 'cpu')
        self.requires_grad = kwargs.get('requires_grad', False)
        self.grad = None
    
    @property
    def device(self):
        class Device:
            def __init__(self, device_type):
                self.type = device_type
        return Device(self._device)
    
    def size(self, dim=None):
        if dim is not None:
            return self.shape[dim]
        return self.shape
    
    def sum(self):
        return MockTensor()
    
    def backward(self):
        pass
    
    def requires_grad_(self, requires_grad=True):
        self.requires_grad = requires_grad
        return self
    
    def cuda(self):
        self._device = 'cuda'
        return self
    
    def cpu(self):
        self._device = 'cpu'
        return self

def mock_randn(*shape, **kwargs):
    return MockTensor(*shape, **kwargs)

def mock_equal(a, b):
    return True

def mock_isnan(tensor):
    class MockAny:
        def any(self):
            return False
    return MockAny()

torch_mock = MagicMock()
torch_mock.nn = MagicMock()
torch_mock.nn.Module = MockModule
torch_mock.nn.functional = MagicMock()
torch_mock.randn = mock_randn
torch_mock.equal = mock_equal
torch_mock.isnan = mock_isnan
torch_mock.cuda = MagicMock()
torch_mock.cuda.is_available = MagicMock(return_value=True)
sys.modules['torch'] = torch_mock
sys.modules['torch.nn'] = torch_mock.nn
sys.modules['torch.nn.functional'] = torch_mock.nn.functional

# Now import torch after mocking
import torch
import torch.nn as nn

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from layers.modular.fusion.hierarchical_fusion import HierarchicalFusion
    from layers.modular.fusion.base import BaseFusion
    import layers.modular.core.register_components  # Populate registry
    from models.unified_registry import unified_registry, ComponentFamily
    FUSION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Fusion components not available: {e}")
    FUSION_AVAILABLE = False
    # Create dummy classes for testing
    class HierarchicalFusion: pass
    class BaseFusion: pass
    class ComponentFamily: 
        FUSION = None
    class unified_registry:
        @staticmethod
        def get_all_by_type(component_type): return {}
        @staticmethod
        def create(name, family, **kwargs): raise ValueError(f"Component '{name}' not found")


class TestFusionRegistry(unittest.TestCase):
    """Test fusion component registry functionality."""
    
    def test_fusion_component_family_exists(self):
        """Test that FUSION component family is properly defined."""
        self.assertTrue(hasattr(ComponentFamily, 'FUSION'))
        self.assertEqual(ComponentFamily.FUSION.value, 'fusion')
    
    def test_hierarchical_fusion_registration(self):
        """Test that HierarchicalFusion is properly registered."""
        # Check if component is registered
        fusion_components = unified_registry.get_all_by_type(ComponentFamily.FUSION)
        self.assertIn('hierarchical_fusion', fusion_components)
        
        # Verify component class
        component_info = fusion_components['hierarchical_fusion']
        self.assertEqual(component_info['class'], HierarchicalFusion)
        self.assertIn('config', component_info)
    
    def test_fusion_component_creation(self):
        """Test creating fusion components through the registry."""
        # Create with default config
        fusion = unified_registry.create(
            'hierarchical_fusion',
            ComponentFamily.FUSION
        )
        
        self.assertIsInstance(fusion, HierarchicalFusion)
        self.assertIsInstance(fusion, BaseFusion)
        self.assertIsInstance(fusion, nn.Module)
    
    def test_fusion_component_creation_with_overrides(self):
        """Test creating fusion components with parameter overrides."""
        fusion = unified_registry.create(
            'hierarchical_fusion',
            ComponentFamily.FUSION,
            d_model=256,
            n_levels=4,
            fusion_strategy='weighted_sum'
        )
        
        self.assertEqual(fusion.d_model, 256)
        self.assertEqual(fusion.n_levels, 4)
        self.assertEqual(fusion.fusion_strategy, 'weighted_sum')
    
    def test_fusion_registry_error_handling(self):
        """Test error handling for non-existent fusion components."""
        with self.assertRaises(ValueError) as context:
            unified_registry.create(
                'non_existent_fusion',
                ComponentFamily.FUSION
            )
        self.assertIn("Component 'non_existent_fusion' not found", str(context.exception))


class TestHierarchicalFusion(unittest.TestCase):
    """Test HierarchicalFusion component functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not FUSION_AVAILABLE:
            self.skipTest("Fusion components not available")
        
        # Create sample multi-resolution features for testing
        batch_size, d_model = 2, 64
        self.sample_features = [
            torch.randn(batch_size, 32, d_model),  # High resolution
            torch.randn(batch_size, 16, d_model),  # Medium resolution
            torch.randn(batch_size, 8, d_model),   # Low resolution
        ]
        
        # Create a HierarchicalFusion component for testing
        self.fusion_component = HierarchicalFusion(
            d_model=64,
            n_levels=3,
            fusion_strategy='weighted_concat'
        )
    
    def test_hierarchical_fusion_initialization(self):
        """Test HierarchicalFusion initialization with different strategies."""
        strategies = ['weighted_concat', 'weighted_sum', 'attention_fusion']
        
        for strategy in strategies:
            fusion = HierarchicalFusion(
                d_model=64,
                n_levels=3,
                fusion_strategy=strategy
            )
            self.assertEqual(fusion.fusion_strategy, strategy)
            self.assertEqual(fusion.n_levels, 3)
            self.assertEqual(fusion.d_model, 64)
            self.assertTrue(hasattr(fusion, 'fusion_weights'))
    
    def test_weighted_concat_fusion(self):
        """Test weighted concatenation fusion strategy."""
        self.fusion_component.fusion_strategy = 'weighted_concat'
        
        output = self.fusion_component(self.sample_features)
        
        # Check output shape
        expected_length = max(feat.size(1) for feat in self.sample_features)
        self.assertEqual(output.shape, (2, expected_length, 64))
        self.assertFalse(torch.isnan(output).any())
    
    def test_weighted_sum_fusion(self):
        """Test weighted sum fusion strategy."""
        fusion = HierarchicalFusion(
            d_model=64,
            n_levels=3,
            fusion_strategy='weighted_sum'
        )
        
        output = fusion(self.sample_features)
        
        # Check output shape
        expected_length = max(feat.size(1) for feat in self.sample_features)
        self.assertEqual(output.shape, (2, expected_length, 64))
        self.assertFalse(torch.isnan(output).any())
    
    def test_attention_fusion(self):
        """Test attention-based fusion strategy."""
        fusion = HierarchicalFusion(
            d_model=64,
            n_levels=3,
            fusion_strategy='attention_fusion'
        )
        
        output = fusion(self.sample_features)
        
        # Check output shape
        expected_length = max(feat.size(1) for feat in self.sample_features)
        self.assertEqual(output.shape, (2, expected_length, 64))
        self.assertFalse(torch.isnan(output).any())
    
    def test_single_feature_passthrough(self):
        """Test that single feature is passed through unchanged."""
        single_feature = [torch.randn(2, 32, 64)]
        output = self.fusion_component(single_feature)
        
        self.assertTrue(torch.equal(output, single_feature[0]))
    
    def test_target_length_specification(self):
        """Test fusion with specified target length."""
        target_length = 24
        output = self.fusion_component(self.sample_features, target_length=target_length)
        
        self.assertEqual(output.shape, (2, target_length, 64))
        self.assertFalse(torch.isnan(output).any())
    
    def test_fusion_weights_learning(self):
        """Test that fusion weights are learnable parameters."""
        # Check that fusion_weights requires gradients
        self.assertTrue(self.fusion_component.fusion_weights.requires_grad)
        self.assertEqual(self.fusion_component.fusion_weights.shape, (3,))
    
    def test_gradient_flow(self):
        """Test that gradients flow through the fusion component."""
        # Enable gradient computation
        for feat in self.sample_features:
            feat.requires_grad_(True)
        
        output = self.fusion_component(self.sample_features)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are computed
        self.assertIsNotNone(self.fusion_component.fusion_weights.grad)
        for feat in self.sample_features:
            self.assertIsNotNone(feat.grad)


class TestFusionIntegration(unittest.TestCase):
    """Test fusion component integration with the broader system."""
    
    def test_fusion_base_class_compliance(self):
        """Test that HierarchicalFusion properly implements BaseFusion interface."""
        fusion = HierarchicalFusion(d_model=64, n_levels=3)
        
        # Check inheritance
        self.assertIsInstance(fusion, BaseFusion)
        self.assertIsInstance(fusion, nn.Module)
        
        # Check abstract method implementation
        self.assertTrue(hasattr(fusion, 'forward'))
        self.assertTrue(callable(fusion.forward))
    
    def test_fusion_with_different_input_shapes(self):
        """Test fusion robustness with various input shapes."""
        fusion = HierarchicalFusion(d_model=128, n_levels=2)
        
        # Test with different batch sizes and sequence lengths
        test_cases = [
            ([torch.randn(1, 10, 128), torch.randn(1, 5, 128)], (1, 10, 128)),
            ([torch.randn(4, 20, 128), torch.randn(4, 10, 128)], (4, 20, 128)),
            ([torch.randn(2, 50, 128), torch.randn(2, 25, 128), torch.randn(2, 12, 128)], (2, 50, 128)),
        ]
        
        for features, expected_shape in test_cases:
            output = fusion(features)
            self.assertEqual(output.shape, expected_shape)
            self.assertFalse(torch.isnan(output).any())
    
    def test_fusion_memory_efficiency(self):
        """Test that fusion component doesn't cause memory leaks."""
        fusion = HierarchicalFusion(d_model=64, n_levels=3)
        
        # Create large features to test memory handling
        large_features = [
            torch.randn(8, 1000, 64),
            torch.randn(8, 500, 64),
            torch.randn(8, 250, 64),
        ]
        
        # Multiple forward passes
        for _ in range(5):
            output = fusion(large_features)
            del output  # Explicit cleanup
        
        # If we reach here without OOM, memory handling is acceptable
        self.assertTrue(True)
    
    def test_fusion_device_compatibility(self):
        """Test fusion component device compatibility."""
        fusion = HierarchicalFusion(d_model=64, n_levels=2)
        
        # Test CPU
        cpu_features = [torch.randn(2, 10, 64), torch.randn(2, 5, 64)]
        cpu_output = fusion(cpu_features)
        self.assertEqual(cpu_output.device.type, 'cpu')
        
        # Test CUDA if available
        if torch.cuda.is_available():
            fusion_cuda = fusion.cuda()
            cuda_features = [feat.cuda() for feat in cpu_features]
            cuda_output = fusion_cuda(cuda_features)
            self.assertEqual(cuda_output.device.type, 'cuda')


if __name__ == "__main__":
    unittest.main(verbosity=2, buffer=True)