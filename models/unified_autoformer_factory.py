"""
Unified Autoformer Factory (Modularized)
"""

from argparse import Namespace
from typing import Dict, List, Union

from models.base_forecaster import BaseTimeSeriesForecaster
from models.modular_autoformer import ModularAutoformer
from models.HFAutoformerSuite import HFEnhancedAutoformer, HFBayesianAutoformer, HFHierarchicalAutoformer, HFQuantileAutoformer
from models.HFEnhancedAutoformerAdvanced import HFEnhancedAutoformerAdvanced
from models.HFBayesianAutoformerProduction import HFBayesianAutoformerProduction
from utils.logger import logger

class UnifiedAutoformerFactory:
    CUSTOM_MODELS = {
        'standard': 'modular', 'enhanced': 'modular', 'fixed': 'modular',
        'enhanced_fixed': 'modular', 'bayesian_enhanced': 'modular',
        'hierarchical': 'modular', 'quantile_bayesian': 'modular',
        'enhanced_advanced': 'modular', 'bayesian_production': 'modular'
    }
    
    HF_MODELS = {
        'hf_enhanced': HFEnhancedAutoformer, 'hf_bayesian': HFBayesianAutoformer,
        'hf_hierarchical': HFHierarchicalAutoformer, 'hf_quantile': HFQuantileAutoformer,
        'hf_enhanced_advanced': HFEnhancedAutoformerAdvanced,
        'hf_bayesian_production': HFBayesianAutoformerProduction
    }
    
    @classmethod
    def create_model(cls, model_type: str, config: Union[Namespace, Dict], framework_preference: str = 'auto') -> BaseTimeSeriesForecaster:
        config = Namespace(**config) if isinstance(config, dict) else config
        cls._ensure_hf_config_completeness(config)
            
        use_hf = cls._should_use_hf(model_type, framework_preference)
        
        return cls._create_hf_model(model_type, config) if use_hf else cls._create_custom_model(model_type, config)
    
    @classmethod
    def _should_use_hf(cls, model_type: str, framework_preference: str) -> bool:
        if framework_preference in ['hf', 'custom']:
            return framework_preference == 'hf'
        return model_type.startswith('hf_') or model_type not in cls.CUSTOM_MODELS
    
    @classmethod
    def _ensure_hf_config_completeness(cls, config: Namespace) -> None:
        hf_defaults = {
            'embed': 'timeF', 'freq': 'h', 'dropout': 0.1, 'activation': 'gelu',
            'factor': 1, 'output_attention': False, 'use_amp': False,
            'task_name': 'long_term_forecast', 'e_layers': 2, 'd_layers': 1,
            'n_heads': 8, 'd_ff': 2048, 'moving_avg': 25, 'norm_type': 'LayerNorm'
        }
        for param, default_value in hf_defaults.items():
            if not hasattr(config, param):
                setattr(config, param, default_value)
    
    @classmethod
    def _create_hf_model(cls, model_type: str, config: Namespace) -> BaseTimeSeriesForecaster:
        hf_type_mapping = {
            'enhanced': 'hf_enhanced', 'bayesian_enhanced': 'hf_bayesian', 
            'hierarchical': 'hf_hierarchical', 'quantile_bayesian': 'hf_quantile',
            'enhanced_advanced': 'hf_enhanced_advanced', 'bayesian_production': 'hf_bayesian_production'
        }
        model_type = hf_type_mapping.get(model_type, model_type)
        
        lookup_type = model_type.replace('hf_', '', 1) if model_type.startswith('hf_') else model_type
        model_key = f'hf_{lookup_type}'
        
        if model_key in cls.HF_MODELS:
            model_class = cls.HF_MODELS[model_key]
            logger.info(f"Creating HF model: {model_class.__name__}")
            return model_class(config)
        else:
            raise ValueError(f"HF model type '{model_type}' not supported.")
    
    @classmethod
    def _create_custom_model(cls, model_type: str, config: Namespace) -> BaseTimeSeriesForecaster:
        logger.info(f"Creating custom modular model: {model_type}")
        config.framework_type = 'custom'
        config.model_variant = model_type
        return ModularAutoformer(config)

def create_autoformer(model_type: str, config: Union[Namespace, Dict], framework: str = 'auto') -> BaseTimeSeriesForecaster:
    return UnifiedAutoformerFactory.create_model(model_type, config, framework)