#!/usr/bin/env python3
"""
Test configuration and utilities for modular framework tests.

This module provides:
1. Test configuration constants
2. Mock implementations for testing without dependencies
3. Test utilities and helpers
4. Expected test outcomes and validation data
"""

import os
import sys
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, MagicMock

# Test Configuration Constants
TEST_CONFIG = {
    'project_root': os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')),
    'layers_path': 'layers/modular',
    'core_path': 'layers/modular/core',
    'attention_path': 'layers/modular/attention',
    'fusion_path': 'layers/modular/fusion',
    'loss_path': 'layers/modular/loss',
    'tests_path': 'TestsModule/tests/integration/modular_framework'
}

# Expected Component Families
EXPECTED_FAMILIES = {
    'ATTENTION': {
        'description': 'Attention mechanism components',
        'components': [
            'MultiHeadAttention', 'SelfAttention', 'CrossAttention',
            'ScaledDotProductAttention', 'AdditiveAttention'
        ]
    },
    'FUSION': {
        'description': 'Data fusion components',
        'components': [
            'HierarchicalFusion', 'AdaptiveFusion', 'CrossModalFusion',
            'TemporalFusion', 'SpatialFusion', 'FeatureFusion'
        ]
    },
    'LOSS': {
        'description': 'Loss function components',
        'components': [
            'MSELoss', 'MAELoss', 'HuberLoss', 'QuantileLoss',
            'FocalLoss', 'AdaptiveLoss', 'BayesianLoss'
        ]
    }
}

# Expected File Structure
EXPECTED_STRUCTURE = {
    'core_files': [
        'registry.py',
        'register_components.py',
        '__init__.py'
    ],
    'attention_files': [
        'registry.py',
        '__init__.py',
        'multi_head.py',
        'self_attention.py',
        'cross_attention.py'
    ],
    'fusion_files': [
        'registry.py',
        '__init__.py',
        'hierarchical_fusion.py',
        'adaptive_fusion.py',
        'cross_modal_fusion.py'
    ],
    'loss_files': [
        'registry.py',
        '__init__.py',
        'standard_losses.py',
        'advanced_losses.py',
        'adaptive_losses.py'
    ]
}

# Mock Classes for Testing Without Dependencies
class MockTorchModule:
    """Mock PyTorch module for testing without torch dependency."""
    
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.training = True
    
    def forward(self, *args, **kwargs):
        return Mock()
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def train(self, mode=True):
        self.training = mode
        return self
    
    def eval(self):
        return self.train(False)

class MockTorchTensor:
    """Mock PyTorch tensor for testing without torch dependency."""
    
    def __init__(self, data=None, shape=None):
        self.data = data
        self.shape = shape or (1,)
    
    def size(self):
        return self.shape
    
    def dim(self):
        return len(self.shape)

# Mock Registry Components
MOCK_COMPONENTS = {
    'attention': {
        'MultiHeadAttention': MockTorchModule,
        'SelfAttention': MockTorchModule,
        'CrossAttention': MockTorchModule
    },
    'fusion': {
        'HierarchicalFusion': MockTorchModule,
        'AdaptiveFusion': MockTorchModule,
        'CrossModalFusion': MockTorchModule
    },
    'loss': {
        'MSELoss': MockTorchModule,
        'MAELoss': MockTorchModule,
        'HuberLoss': MockTorchModule
    }
}

# Test Utilities
class TestUtils:
    """Utility functions for modular framework tests."""
    
    @staticmethod
    def get_project_path(*paths):
        """Get absolute path relative to project root."""
        return os.path.join(TEST_CONFIG['project_root'], *paths)
    
    @staticmethod
    def file_exists(relative_path):
        """Check if file exists relative to project root."""
        full_path = TestUtils.get_project_path(relative_path)
        return os.path.isfile(full_path)
    
    @staticmethod
    def dir_exists(relative_path):
        """Check if directory exists relative to project root."""
        full_path = TestUtils.get_project_path(relative_path)
        return os.path.isdir(full_path)
    
    @staticmethod
    def read_file_content(relative_path):
        """Read file content relative to project root."""
        full_path = TestUtils.get_project_path(relative_path)
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except (FileNotFoundError, IOError):
            return None
    
    @staticmethod
    def check_import_in_file(relative_path, import_statement):
        """Check if import statement exists in file."""
        content = TestUtils.read_file_content(relative_path)
        if content is None:
            return False
        return import_statement in content
    
    @staticmethod
    def check_class_in_file(relative_path, class_name):
        """Check if class definition exists in file."""
        content = TestUtils.read_file_content(relative_path)
        if content is None:
            return False
        return f'class {class_name}' in content
    
    @staticmethod
    def check_function_in_file(relative_path, function_name):
        """Check if function definition exists in file."""
        content = TestUtils.read_file_content(relative_path)
        if content is None:
            return False
        return f'def {function_name}' in content
    
    @staticmethod
    def mock_torch_environment():
        """Create mock torch environment for testing."""
        mock_torch = Mock()
        mock_torch.nn = Mock()
        mock_torch.nn.Module = MockTorchModule
        mock_torch.Tensor = MockTorchTensor
        
        # Mock common torch functions
        mock_torch.zeros = lambda *args, **kwargs: MockTorchTensor(shape=args)
        mock_torch.ones = lambda *args, **kwargs: MockTorchTensor(shape=args)
        mock_torch.randn = lambda *args, **kwargs: MockTorchTensor(shape=args)
        
        return mock_torch

# Test Validation Data
VALIDATION_DATA = {
    'registry_patterns': {
        'component_family_enum': r'class ComponentFamily\(Enum\):',
        'registry_class': r'class UnifiedRegistry:',
        'register_method': r'def register\(',
        'create_method': r'def create\(',
        'get_available_method': r'def get_available\('
    },
    'component_patterns': {
        'torch_import': r'import torch',
        'nn_module_inheritance': r'class.*\(.*nn\.Module.*\):',
        'forward_method': r'def forward\(',
        'init_method': r'def __init__\('
    },
    'registration_patterns': {
        'register_call': r'registry\.register\(',
        'family_specification': r'ComponentFamily\.',
        'component_import': r'from.*import'
    }
}

# Expected Test Outcomes
EXPECTED_OUTCOMES = {
    'unified_registry_tests': {
        'test_registry_structure': 'PASS',
        'test_component_families': 'PASS',
        'test_component_registration': 'PASS',
        'test_component_creation': 'SKIP_IF_NO_TORCH',
        'test_backward_compatibility': 'PASS',
        'test_error_handling': 'PASS'
    },
    'fusion_component_tests': {
        'test_fusion_directory_structure': 'PASS',
        'test_fusion_registry_integration': 'PASS',
        'test_fusion_component_creation': 'SKIP_IF_NO_TORCH',
        'test_fusion_error_handling': 'PASS',
        'test_fusion_backward_compatibility': 'PASS',
        'test_fusion_core_integration': 'PASS'
    },
    'loss_component_tests': {
        'test_loss_directory_structure': 'PASS',
        'test_loss_registry_integration': 'PASS',
        'test_loss_component_creation': 'SKIP_IF_NO_TORCH',
        'test_loss_error_handling': 'PASS',
        'test_loss_backward_compatibility': 'PASS',
        'test_loss_core_integration': 'PASS'
    }
}

# Test Environment Setup
def setup_test_environment():
    """Setup test environment with necessary paths and mocks."""
    # Add project root to Python path
    project_root = TEST_CONFIG['project_root']
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Create mock torch if not available
    try:
        import torch
        return {'torch_available': True, 'mock_torch': None}
    except ImportError:
        mock_torch = TestUtils.mock_torch_environment()
        sys.modules['torch'] = mock_torch
        sys.modules['torch.nn'] = mock_torch.nn
        return {'torch_available': False, 'mock_torch': mock_torch}

def teardown_test_environment(env_info):
    """Cleanup test environment."""
    if not env_info['torch_available'] and env_info['mock_torch']:
        # Remove mock torch modules
        if 'torch' in sys.modules:
            del sys.modules['torch']
        if 'torch.nn' in sys.modules:
            del sys.modules['torch.nn']

# Export all configuration
__all__ = [
    'TEST_CONFIG',
    'EXPECTED_FAMILIES',
    'EXPECTED_STRUCTURE',
    'MOCK_COMPONENTS',
    'MockTorchModule',
    'MockTorchTensor',
    'TestUtils',
    'VALIDATION_DATA',
    'EXPECTED_OUTCOMES',
    'setup_test_environment',
    'teardown_test_environment'
]