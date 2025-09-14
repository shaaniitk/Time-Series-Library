"""Warning suppression utilities for test modules.

This module provides functions to suppress various warnings that appear
during testing, particularly from optional dependencies and library imports.
"""

import warnings

def suppress_optional_dependency_warnings():
    """Suppress warnings from optional dependencies in torch_geometric and other libraries."""
    import logging
    
    # Suppress torch_geometric optional dependency warnings
    warnings.filterwarnings('ignore', message='.*pyg-lib.*')
    warnings.filterwarnings('ignore', message='.*torch-scatter.*')
    warnings.filterwarnings('ignore', message='.*torch-sparse.*')
    warnings.filterwarnings('ignore', message='.*torch-cluster.*')
    warnings.filterwarnings('ignore', message='.*torch-spline-conv.*')
    
    # Suppress TSNormalizer import warnings
    warnings.filterwarnings('ignore', message='.*Could not import TSNormalizer.*')
    
    # Suppress wavelet component warnings
    warnings.filterwarnings('ignore', message='.*Wavelet components not available.*')
    warnings.filterwarnings('ignore', message='.*Processor components not available.*')
    
    # Suppress registry warnings
    warnings.filterwarnings('ignore', message='.*Could not register with unified registry.*')
    
    # Suppress RMSNorm warnings
    warnings.filterwarnings('ignore', message='.*Could not import RMSNorm.*')
    
    # Suppress other common warnings in testing
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    # Suppress UserWarnings from torch_geometric
    warnings.filterwarnings('ignore', category=UserWarning, module='torch_geometric.*')
    
    # Suppress logging warnings from registry modules
    logging.getLogger('layers.modular.normalization.registry').setLevel(logging.ERROR)
    logging.getLogger('layers.modular.processor.registry').setLevel(logging.ERROR)
    logging.getLogger('root').setLevel(logging.ERROR)

def suppress_all_warnings():
    """Suppress all warnings - use with caution in testing only."""
    warnings.filterwarnings('ignore')

def suppress_test_warnings():
    """Suppress common test-related warnings while keeping important ones."""
    # Suppress pytest warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='pytest.*')
    warnings.filterwarnings('ignore', category=FutureWarning, module='pytest.*')
    
    # Suppress numpy warnings
    warnings.filterwarnings('ignore', category=FutureWarning, module='numpy.*')
    
    # Suppress torch warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='torch.*')