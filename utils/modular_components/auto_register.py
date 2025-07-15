"""
Auto-registration utility for modular components

This module automatically registers all available components
with the global component registry.
"""

import logging
from .registry import get_global_registry
from .config_schemas import BackboneConfig, EmbeddingConfig, AttentionConfig

"""
Auto-registration utility for modular components

This module automatically registers all available components
with the global registry.
"""

import logging
from .registry import get_global_registry
from .config_schemas import BackboneConfig, EmbeddingConfig, AttentionConfig

logger = logging.getLogger(__name__)


def auto_register_components():
    """
    Automatically register all available components with the global registry
    """
    registry = get_global_registry()
    
    # Register simple backbones (always available)
    try:
        from .implementations.simple_backbones import (
            SimpleTransformerBackbone, RobustHFBackbone
        )
        
        registry.register('backbone', 'simple_transformer', SimpleTransformerBackbone, {
            'description': 'Simple transformer backbone with no external dependencies',
            'supports_time_series': True,
            'no_external_deps': True
        })
        
        registry.register('backbone', 'robust_hf', RobustHFBackbone, {
            'description': 'Robust HuggingFace backbone with fallbacks',
            'supports_time_series': True,
            'supports_seq2seq': True,
            'has_fallback': True
        })
        
        # Add aliases
        registry.register('backbone', 'simple', SimpleTransformerBackbone, {
            'description': 'Alias for simple_transformer',
            'alias_for': 'simple_transformer'
        })
        
        registry.register('backbone', 'auto', RobustHFBackbone, {
            'description': 'Alias for robust_hf (automatic selection)',
            'alias_for': 'robust_hf'
        })
        
        logger.info("Registered simple backbone components")
        
    except ImportError as e:
        logger.error(f"Failed to register simple backbones: {e}")
    
    # Try to register advanced backbones (may fail)
    try:
        from .implementations.backbones import (
            ChronosBackbone, T5Backbone, BERTBackbone
        )
        
        registry.register('backbone', 'chronos', ChronosBackbone, {
            'description': 'Chronos-based backbone for time series forecasting',
            'supports_time_series': True,
            'supports_forecasting': True,
            'pretrained_available': True
        })
        
        registry.register('backbone', 't5', T5Backbone, {
            'description': 'T5-based backbone for sequence-to-sequence modeling',
            'supports_seq2seq': True,
            'pretrained_available': True
        })
        
        registry.register('backbone', 'bert', BERTBackbone, {
            'description': 'BERT-based backbone for representation learning',
            'supports_representation_learning': True,
            'pretrained_available': True
        })
        
        logger.info("Registered advanced backbone components")
        
    except ImportError as e:
        logger.warning(f"Advanced backbones not available: {e}")
    
    # Register embedding components
    try:
        from .implementations.embeddings import (
            TemporalEmbedding, ValueEmbedding, CovariateEmbedding, HybridEmbedding
        )
        
        registry.register('embedding', 'temporal', TemporalEmbedding, {
            'description': 'Temporal embedding with multiple time feature support',
            'supports_positional': True,
            'supports_temporal_features': True
        })
        
        registry.register('embedding', 'value', ValueEmbedding, {
            'description': 'Value-based embedding for continuous time series values',
            'supports_continuous': True,
            'supports_binning': True
        })
        
        registry.register('embedding', 'covariate', CovariateEmbedding, {
            'description': 'Covariate embedding for external features',
            'supports_categorical': True,
            'supports_numerical': True
        })
        
        registry.register('embedding', 'hybrid', HybridEmbedding, {
            'description': 'Hybrid embedding combining multiple strategies',
            'supports_all': True,
            'configurable': True
        })
        
        logger.info("Registered embedding components")
        
    except ImportError as e:
        logger.warning(f"Embedding components not available: {e}")
    
    # Register attention components
    try:
        from .implementations.attentions import (
            MultiHeadAttention, AutoCorrelationAttention, SparseAttention,
            LogSparseAttention, ProbSparseAttention
        )
        
        registry.register('attention', 'multi_head', MultiHeadAttention, {
            'description': 'Standard multi-head attention mechanism',
            'supports_masking': True,
            'supports_self_attention': True,
            'supports_cross_attention': True
        })
        
        registry.register('attention', 'autocorrelation', AutoCorrelationAttention, {
            'description': 'AutoCorrelation attention for seasonal patterns',
            'supports_seasonal_patterns': True,
            'supports_fft': True,
            'time_series_optimized': True
        })
        
        registry.register('attention', 'sparse', SparseAttention, {
            'description': 'Sparse attention for efficient long sequences',
            'supports_long_sequences': True,
            'computational_complexity': 'O(n*sqrt(n))',
            'supports_masking': True
        })
        
        registry.register('attention', 'log_sparse', LogSparseAttention, {
            'description': 'LogSparse attention for very long sequences',
            'supports_very_long_sequences': True,
            'computational_complexity': 'O(n*log(n))',
            'logarithmic_pattern': True
        })
        
        registry.register('attention', 'prob_sparse', ProbSparseAttention, {
            'description': 'ProbSparse attention from Informer',
            'supports_long_sequences': True,
            'probability_based_sampling': True,
            'from_informer': True
        })
        
        logger.info("Registered attention components")
        
    except ImportError as e:
        logger.warning(f"Attention components not available: {e}")
    
    logger.info("Component auto-registration completed")
    
    # Log registration summary
    try:
        components = registry.list_components()
        for component_type, component_names in components.items():
            logger.info(f"Available {component_type} components: {component_names}")
    except Exception as e:
        logger.warning(f"Could not list components: {e}")


def get_available_components():
    """
    Get a dictionary of all available components by type
    
    Returns:
        Dict with component types as keys and lists of available components as values
    """
    registry = get_global_registry()
    return registry.list_components()


def get_component_info(component_type: str, component_name: str):
    """
    Get detailed information about a specific component
    
    Args:
        component_type: Type of component (backbone, embedding, attention, etc.)
        component_name: Name of the specific component
        
    Returns:
        Dictionary with component information
    """
    registry = get_global_registry()
    
    if registry.is_registered(component_type, component_name):
        # Get the component class and metadata
        class_info = registry.components[component_type][component_name]
        return {
            'type': component_type,
            'name': component_name,
            'class': class_info['class'].__name__,
            'metadata': class_info['metadata']
        }
    else:
        return None


def create_default_configs():
    """
    Create default configurations for all component types
    
    Returns:
        Dictionary with default configurations
    """
    return {
        'backbone': {
            'chronos': BackboneConfig(
                d_model=512,
                dropout=0.1,
                model_name='amazon/chronos-t5-small',
                pretrained=True,
                fallback_models=['amazon/chronos-t5-tiny']
            ),
            't5': BackboneConfig(
                d_model=512,
                dropout=0.1,
                model_name='t5-small',
                pretrained=True,
                fallback_models=['t5-base']
            ),
            'simple_transformer': BackboneConfig(
                d_model=512,
                dropout=0.1,
                num_layers=6,
                num_heads=8,
                dim_feedforward=2048
            )
        },
        'embedding': {
            'temporal': EmbeddingConfig(
                d_model=512,
                dropout=0.1,
                max_len=5000,
                temp_feature_dim=4
            ),
            'value': EmbeddingConfig(
                d_model=512,
                dropout=0.1,
                num_features=1,
                use_binning=False
            ),
            'hybrid': EmbeddingConfig(
                d_model=512,
                dropout=0.1,
                use_temporal=True,
                use_value=True,
                use_covariate=False,
                combination_strategy='add'
            )
        },
        'attention': {
            'multi_head': AttentionConfig(
                d_model=512,
                dropout=0.1,
                num_heads=8
            ),
            'autocorrelation': AttentionConfig(
                d_model=512,
                dropout=0.1,
                factor=1
            ),
            'sparse': AttentionConfig(
                d_model=512,
                dropout=0.1,
                num_heads=8,
                sparsity_factor=4
            )
        }
    }
