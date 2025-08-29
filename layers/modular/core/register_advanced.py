"""
Register Advanced Components with the Component Registry

This module registers all the sophisticated implementations we've integrated
from the existing codebase into the modular framework.
"""

import logging
from typing import Dict, Any

from ..registry import register_component
from ..base_interfaces import BaseComponent

logger = logging.getLogger(__name__)


def register_advanced_losses():
    """Register advanced loss function implementations"""
    try:
        # Apply patches to ensure compute_loss methods exist
        from . import patch_losses
        
        from .advanced_losses import (
            BayesianMSELoss, BayesianMAELoss, AdaptiveStructuralLoss,
            FrequencyAwareLoss, PatchStructuralLoss, DTWAlignmentLoss,
            MultiScaleTrendLoss, BayesianQuantileLoss
        )
        
        # Register Bayesian losses
        register_component(
            'loss', 'bayesian_mse', BayesianMSELoss,
            metadata={
                'type': 'bayesian',
                'supports_uncertainty': True,
                'has_kl_divergence': True,
                'primary_loss': 'mse'
            }
        )
        
        register_component(
            'loss', 'bayesian_mae', BayesianMAELoss,
            metadata={
                'type': 'bayesian',
                'supports_uncertainty': True,
                'has_kl_divergence': True,
                'primary_loss': 'mae'
            }
        )
        
        register_component(
            'loss', 'bayesian_quantile', BayesianQuantileLoss,
            metadata={
                'type': 'bayesian',
                'supports_uncertainty': True,
                'has_kl_divergence': True,
                'primary_loss': 'quantile'
            }
        )
        
        # Register advanced structural losses
        register_component(
            'loss', 'adaptive_structural', AdaptiveStructuralLoss,
            metadata={
                'type': 'structural',
                'adaptive': True,
                'multi_component': True
            }
        )
        
        register_component(
            'loss', 'frequency_aware', FrequencyAwareLoss,
            metadata={
                'type': 'frequency',
                'domain': 'frequency',
                'uses_fft': True
            }
        )
        
        register_component(
            'loss', 'patch_structural', PatchStructuralLoss,
            metadata={
                'type': 'structural',
                'patch_based': True,
                'local_patterns': True
            }
        )
        
        register_component(
            'loss', 'dtw_alignment', DTWAlignmentLoss,
            metadata={
                'type': 'alignment',
                'temporal_alignment': True,
                'dynamic_warping': True
            }
        )
        
        register_component(
            'loss', 'multiscale_trend', MultiScaleTrendLoss,
            metadata={
                'type': 'trend',
                'multi_scale': True,
                'trend_aware': True
            }
        )
        
        logger.info("Successfully registered 8 advanced loss functions")
        
    except Exception as e:
        logger.error(f"Failed to register advanced losses: {e}")


def register_advanced_attentions():
    """Register advanced attention mechanisms"""
    try:
        from .advanced_attentions import (
            OptimizedAutoCorrelationAttention, AdaptiveAutoCorrelationAttention,
            EnhancedAutoCorrelationLayer, MemoryEfficientAttention
        )
        
        register_component(
            'attention', 'optimized_autocorrelation', OptimizedAutoCorrelationAttention,
            metadata={
                'type': 'autocorrelation',
                'optimized': True,
                'memory_efficient': True,
                'chunked_processing': True,
                'mixed_precision': True
            }
        )
        
        register_component(
            'attention', 'adaptive_autocorrelation', AdaptiveAutoCorrelationAttention,
            metadata={
                'type': 'autocorrelation',
                'adaptive': True,
                'multi_scale': True,
                'adaptive_k': True
            }
        )
        
        register_component(
            'attention', 'enhanced_autocorrelation', EnhancedAutoCorrelationLayer,
            metadata={
                'type': 'autocorrelation',
                'enhanced': True,
                'has_projections': True,
                'multi_scale': True,
                'adaptive_k': True
            }
        )
        
        register_component(
            'attention', 'memory_efficient', MemoryEfficientAttention,
            metadata={
                'type': 'general',
                'memory_efficient': True,
                'gradient_checkpointing': True,
                'fallback_capable': True
            }
        )
        
        logger.info("Successfully registered 4 advanced attention mechanisms")
        
    except Exception as e:
        logger.error(f"Failed to register advanced attentions: {e}")


def register_specialized_processors():
    """Register specialized signal processors"""
    try:
        from .specialized_processors import (
            FrequencyDomainProcessor, StructuralPatchProcessor,
            DTWAlignmentProcessor, TrendProcessor, QuantileProcessor,
            IntegratedSignalProcessor
        )
        
        # Note: These processors don't directly inherit from BaseComponent
        # They are utility classes used by other components
        # We'll register them as processors with custom metadata
        
        register_component(
            'processor', 'frequency_domain', FrequencyDomainProcessor,
            metadata={
                'type': 'frequency',
                'domain': 'frequency',
                'fft_based': True,
                'spectral_features': True,
                'utility_class': True
            }
        )
        
        register_component(
            'processor', 'structural_patch', StructuralPatchProcessor,
            metadata={
                'type': 'structural',
                'patch_based': True,
                'statistical_features': True,
                'utility_class': True
            }
        )
        
        register_component(
            'processor', 'dtw_alignment', DTWAlignmentProcessor,
            metadata={
                'type': 'alignment',
                'temporal_alignment': True,
                'dynamic_warping': True,
                'utility_class': True
            }
        )
        
        register_component(
            'processor', 'trend_analysis', TrendProcessor,
            metadata={
                'type': 'trend',
                'multi_scale': True,
                'trend_extraction': True,
                'utility_class': True
            }
        )
        
        register_component(
            'processor', 'quantile_analysis', QuantileProcessor,
            metadata={
                'type': 'quantile',
                'quantile_regression': True,
                'uncertainty_estimation': True,
                'utility_class': True
            }
        )
        
        register_component(
            'processor', 'integrated_signal', IntegratedSignalProcessor,
            metadata={
                'type': 'integrated',
                'multi_modal': True,
                'comprehensive': True,
                'all_processors': True,
                'utility_class': True
            }
        )
        
        logger.info("Successfully registered 6 specialized processors")
        
    except Exception as e:
        logger.error(f"Failed to register specialized processors: {e}")


def register_all_advanced_components():
    """Register all advanced components at once"""
    logger.info("Registering all advanced components...")
    
    register_advanced_losses()
    register_advanced_attentions()
    register_specialized_processors()
    
    logger.info("Advanced component registration complete")


# Auto-register when module is imported
try:
    register_all_advanced_components()
except Exception as e:
    logger.error(f"Failed to auto-register advanced components: {e}")


def get_advanced_component_summary() -> Dict[str, Any]:
    """Get summary of all registered advanced components"""
    from ..registry import get_global_registry
    
    registry = get_global_registry()
    components = registry.list_components()
    
    summary = {
        'total_advanced_components': 0,
        'by_category': {},
        'bayesian_components': [],
        'optimized_components': [],
        'utility_processors': []
    }
    
    for category, component_list in components.items():
        summary['by_category'][category] = len(component_list)
        summary['total_advanced_components'] += len(component_list)
        
        # Categorize special components
        for comp_name in component_list:
            try:
                metadata = registry.get_metadata(category, comp_name)
                
                if metadata.get('type') == 'bayesian' or metadata.get('has_kl_divergence'):
                    summary['bayesian_components'].append(f"{category}.{comp_name}")
                
                if metadata.get('optimized') or metadata.get('memory_efficient'):
                    summary['optimized_components'].append(f"{category}.{comp_name}")
                
                if metadata.get('utility_class'):
                    summary['utility_processors'].append(f"{category}.{comp_name}")
                    
            except Exception:
                pass  # Skip if metadata not available
    
    return summary


def validate_bayesian_integration() -> Dict[str, bool]:
    """Validate that Bayesian components are properly integrated"""
    from ..registry import get_global_registry
    
    registry = get_global_registry()
    validation = {
        'bayesian_mse_registered': False,
        'bayesian_mae_registered': False,
        'bayesian_quantile_registered': False,
        'kl_divergence_supported': False,
        'uncertainty_supported': False
    }
    
    try:
        # Check Bayesian MSE
        if registry.is_registered('loss', 'bayesian_mse'):
            validation['bayesian_mse_registered'] = True
            metadata = registry.get_metadata('loss', 'bayesian_mse')
            if metadata.get('has_kl_divergence'):
                validation['kl_divergence_supported'] = True
            if metadata.get('supports_uncertainty'):
                validation['uncertainty_supported'] = True
        
        # Check Bayesian MAE
        validation['bayesian_mae_registered'] = registry.is_registered('loss', 'bayesian_mae')
        
        # Check Bayesian Quantile
        validation['bayesian_quantile_registered'] = registry.is_registered('loss', 'bayesian_quantile')
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
    
    return validation
