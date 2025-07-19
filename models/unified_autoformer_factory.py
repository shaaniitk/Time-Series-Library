"""
Unified Autoformer Factory

This factory provides a single interface to create both HF-based and custom modular autoformers,
enabling seamless switching between implementations while maintaining consistent APIs.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List
from argparse import Namespace
import logging

from models.base_forecaster import BaseTimeSeriesForecaster
from models.modular_autoformer import ModularAutoformer
from models.HFAutoformerSuite import (
    HFEnhancedAutoformer, 
    HFBayesianAutoformer, 
    HFHierarchicalAutoformer, 
    HFQuantileAutoformer
)
from models.HFEnhancedAutoformerAdvanced import HFEnhancedAutoformerAdvanced
from models.HFBayesianAutoformerProduction import HFBayesianAutoformerProduction

logger = logging.getLogger(__name__)


class UnifiedAutoformerFactory:
    """
    Factory for creating unified autoformer models that can be either HF-based or custom modular.
    
    This factory abstracts the choice between implementations, allowing users to specify
    preferred backends while maintaining consistent interfaces.
    """
    
    # Model type mappings
    CUSTOM_MODELS = {
        'standard': 'modular',
        'enhanced': 'modular', 
        'fixed': 'modular',
        'enhanced_fixed': 'modular',
        'bayesian_enhanced': 'modular',
        'hierarchical': 'modular',
        'quantile_bayesian': 'modular',
        'enhanced_advanced': 'modular',
        'bayesian_production': 'modular'
    }
    
    HF_MODELS = {
        'hf_enhanced': HFEnhancedAutoformer,
        'hf_bayesian': HFBayesianAutoformer,
        'hf_hierarchical': HFHierarchicalAutoformer,
        'hf_quantile': HFQuantileAutoformer,
        'hf_enhanced_advanced': HFEnhancedAutoformerAdvanced,
        'hf_bayesian_production': HFBayesianAutoformerProduction
    }
    
    @classmethod
    def create_model(cls, 
                    model_type: str, 
                    config: Union[Namespace, Dict],
                    framework_preference: str = 'auto') -> BaseTimeSeriesForecaster:
        """
        Create a unified autoformer model.
        
        Args:
            model_type: Type of model ('enhanced', 'bayesian_enhanced', 'hf_enhanced', etc.)
            config: Model configuration
            framework_preference: 'custom', 'hf', or 'auto'
            
        Returns:
            BaseTimeSeriesForecaster instance (either ModularAutoformer or HF variant)
        """
        if isinstance(config, dict):
            config = Namespace(**config)
            
        # Determine implementation based on preference and model type
        use_hf = cls._should_use_hf(model_type, framework_preference)
        
        if use_hf:
            return cls._create_hf_model(model_type, config)
        else:
            return cls._create_custom_model(model_type, config)
    
    @classmethod
    def _should_use_hf(cls, model_type: str, framework_preference: str) -> bool:
        """Determine whether to use HF or custom implementation"""
        
        # Explicit preference
        if framework_preference == 'hf':
            return True
        elif framework_preference == 'custom':
            return False
            
        # Auto detection based on model type
        if model_type.startswith('hf_'):
            return True
        elif model_type in cls.CUSTOM_MODELS:
            return False
        else:
            # Default fallback to custom for unknown types
            logger.warning(f"Unknown model type '{model_type}', defaulting to custom implementation")
            return False
    
    @classmethod
    def _create_hf_model(cls, model_type: str, config: Namespace) -> BaseTimeSeriesForecaster:
        """Create HF-based model"""
        
        # Map custom model types to HF equivalents
        hf_type_mapping = {
            'enhanced': 'hf_enhanced',
            'bayesian_enhanced': 'hf_bayesian', 
            'hierarchical': 'hf_hierarchical',
            'quantile_bayesian': 'hf_quantile',
            'enhanced_advanced': 'hf_enhanced_advanced',
            'bayesian_production': 'hf_bayesian_production'
        }
        
        if model_type in hf_type_mapping:
            model_type = hf_type_mapping[model_type]
        
        # Remove 'hf_' prefix for lookup
        lookup_type = model_type.replace('hf_', '', 1) if model_type.startswith('hf_') else model_type
        
        if f'hf_{lookup_type}' in cls.HF_MODELS:
            model_class = cls.HF_MODELS[f'hf_{lookup_type}']
            logger.info(f"Creating HF model: {model_class.__name__}")
            return model_class(config)
        else:
            raise ValueError(f"HF model type '{model_type}' not supported. Available: {list(cls.HF_MODELS.keys())}")
    
    @classmethod
    def _create_custom_model(cls, model_type: str, config: Namespace) -> BaseTimeSeriesForecaster:
        """Create custom modular model"""
        
        # All custom models use ModularAutoformer with different configurations
        logger.info(f"Creating custom modular model: {model_type}")
        
        # Add framework identifier to config
        config.framework_type = 'custom'
        config.model_variant = model_type
        
        return ModularAutoformer(config)
    
    @classmethod
    def get_available_models(cls) -> Dict[str, List[str]]:
        """Get all available model types organized by framework"""
        return {
            'custom': list(cls.CUSTOM_MODELS.keys()),
            'hf': list(cls.HF_MODELS.keys())
        }
    
    @classmethod
    def create_compatible_pair(cls, 
                             model_type: str, 
                             config: Union[Namespace, Dict]) -> Dict[str, BaseTimeSeriesForecaster]:
        """
        Create both HF and custom versions of the same model for comparison.
        
        Args:
            model_type: Base model type ('enhanced', 'bayesian_enhanced', etc.)
            config: Model configuration
            
        Returns:
            Dict with 'custom' and 'hf' model instances
        """
        if isinstance(config, dict):
            config = Namespace(**config)
            
        models = {}
        
        try:
            # Create custom version
            models['custom'] = cls._create_custom_model(model_type, config)
            logger.info(f"Created custom {model_type} model")
        except Exception as e:
            logger.warning(f"Failed to create custom {model_type}: {e}")
            
        try:
            # Create HF version  
            models['hf'] = cls._create_hf_model(model_type, config)
            logger.info(f"Created HF {model_type} model")
        except Exception as e:
            logger.warning(f"Failed to create HF {model_type}: {e}")
            
        return models


class UnifiedModelInterface:
    """
    Unified interface wrapper that provides consistent methods regardless of underlying implementation.
    """
    
    def __init__(self, model: BaseTimeSeriesForecaster):
        self.model = model
        self.framework_type = getattr(model, 'framework_type', 'unknown')
        
    def predict(self, x_enc, x_mark_enc, x_dec, x_mark_dec, **kwargs):
        """Unified prediction interface"""
        return self.model(x_enc, x_mark_enc, x_dec, x_mark_dec, **kwargs)
    
    def predict_with_uncertainty(self, x_enc, x_mark_enc, x_dec, x_mark_dec, **kwargs):
        """Unified uncertainty prediction interface"""
        
        # Check if model supports uncertainty
        if hasattr(self.model, 'supports_uncertainty') and self.model.supports_uncertainty():
            if hasattr(self.model, 'predict_with_uncertainty'):
                return self.model.predict_with_uncertainty(x_enc, x_mark_enc, x_dec, x_mark_dec, **kwargs)
            else:
                # Fallback: use forward pass and extract uncertainty from results
                prediction = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec, **kwargs)
                uncertainty_results = getattr(self.model, 'last_uncertainty_results', None)
                return {
                    'prediction': prediction,
                    'uncertainty': uncertainty_results
                }
        else:
            # Model doesn't support uncertainty - return basic prediction
            prediction = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec, **kwargs)
            return {
                'prediction': prediction,
                'uncertainty': None
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        return {
            'framework_type': self.framework_type,
            'model_type': getattr(self.model, 'model_type', 'unknown'),
            'model_class': self.model.__class__.__name__,
            'supports_uncertainty': getattr(self.model, 'supports_uncertainty', lambda: False)(),
            'config': getattr(self.model, 'configs', None)
        }
    
    def benchmark_performance(self, test_data, metrics=['mse', 'mae']) -> Dict[str, float]:
        """Unified performance benchmarking"""
        # Implementation would depend on specific test data format
        # This is a placeholder for the interface
        pass


# Convenience functions for common use cases

def create_autoformer(model_type: str, 
                     config: Union[Namespace, Dict],
                     framework: str = 'auto') -> UnifiedModelInterface:
    """
    Convenience function to create a unified autoformer model.
    
    Args:
        model_type: Type of autoformer ('enhanced', 'bayesian_enhanced', etc.)
        config: Model configuration
        framework: Framework preference ('custom', 'hf', 'auto')
        
    Returns:
        UnifiedModelInterface wrapping the created model
    """
    model = UnifiedAutoformerFactory.create_model(model_type, config, framework)
    return UnifiedModelInterface(model)


def compare_implementations(model_type: str, 
                          config: Union[Namespace, Dict]) -> Dict[str, UnifiedModelInterface]:
    """
    Create both custom and HF implementations for direct comparison.
    
    Args:
        model_type: Type of autoformer to compare
        config: Model configuration
        
    Returns:
        Dict with wrapped model interfaces for comparison
    """
    models = UnifiedAutoformerFactory.create_compatible_pair(model_type, config)
    return {
        framework: UnifiedModelInterface(model) 
        for framework, model in models.items()
    }


def list_available_models() -> Dict[str, List[str]]:
    """List all available model types organized by framework"""
    return UnifiedAutoformerFactory.get_available_models()


# Example usage demonstration
def demo_unified_usage():
    """Demonstrate unified model usage"""
    
    # Example configuration
    config = {
        'seq_len': 96,
        'pred_len': 24,
        'label_len': 48,
        'enc_in': 7,
        'dec_in': 7,
        'c_out': 7,
        'd_model': 512
    }
    
    print("=== Unified Autoformer Factory Demo ===")
    
    # Create custom implementation
    custom_model = create_autoformer('enhanced', config, framework='custom')
    print(f"Custom model: {custom_model.get_model_info()}")
    
    # Create HF implementation
    try:
        hf_model = create_autoformer('enhanced', config, framework='hf')
        print(f"HF model: {hf_model.get_model_info()}")
    except Exception as e:
        print(f"HF model creation failed: {e}")
    
    # Test new HF models
    try:
        hf_advanced_model = create_autoformer('hf_enhanced_advanced', config, framework='hf')
        print(f"HF Advanced model: {hf_advanced_model.get_model_info()}")
    except Exception as e:
        print(f"HF Advanced model creation failed: {e}")
        
    try:
        hf_production_model = create_autoformer('hf_bayesian_production', config, framework='hf')
        print(f"HF Production model: {hf_production_model.get_model_info()}")
    except Exception as e:
        print(f"HF Production model creation failed: {e}")
    
    # Compare implementations
    comparison = compare_implementations('enhanced', config)
    print(f"Available for comparison: {list(comparison.keys())}")
    
    # List available models
    available = list_available_models()
    print(f"Available models: {available}")
    print(f"Total HF models available: {len(available.get('hf', []))}")
    print(f"Total Custom models available: {len(available.get('custom', []))}")


if __name__ == "__main__":
    demo_unified_usage()
