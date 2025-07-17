"""
Concrete Implementations of Modular Components

This package contains concrete implementations of all the abstract
component interfaces for building modular time series models.

CRITICAL INTEGRATION UPDATE:
This now includes integration of ALL existing sophisticated implementations
from the Time-Series-Library codebase, particularly:
- Bayesian losses with KL divergence (bayesian_losses.py)
- Advanced structural losses (enhanced_losses.py, losses.py)
- Optimized attention mechanisms (AutoCorrelation_Optimized.py)
- Specialized signal processors (frequency, DTW, trend analysis)
"""

import logging

logger = logging.getLogger(__name__)

# Import core implementation modules with fallbacks
try:
    from . import backbones
    BACKBONES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import advanced backbones: {e}")
    BACKBONES_AVAILABLE = False

# Always import simple backbones (no external dependencies)
from . import simple_backbones
from . import embeddings  
from . import attentions

# Import advanced integrations
try:
    from . import advanced_losses
    ADVANCED_LOSSES_AVAILABLE = True
    logger.info("Advanced losses integration available")
except ImportError as e:
    logger.warning(f"Could not import advanced losses: {e}")
    ADVANCED_LOSSES_AVAILABLE = False

try:
    from . import advanced_attentions
    ADVANCED_ATTENTIONS_AVAILABLE = True
    logger.info("Advanced attention mechanisms available")
except ImportError as e:
    logger.warning(f"Could not import advanced attentions: {e}")
    ADVANCED_ATTENTIONS_AVAILABLE = False

try:
    from . import specialized_processors
    SPECIALIZED_PROCESSORS_AVAILABLE = True
    logger.info("Specialized processors available")
except ImportError as e:
    logger.warning(f"Could not import specialized processors: {e}")
    SPECIALIZED_PROCESSORS_AVAILABLE = False

# Auto-register all advanced components
try:
    from . import register_advanced
    logger.info("Advanced components auto-registration complete")
    ADVANCED_REGISTRATION_COMPLETE = True
except ImportError as e:
    logger.error(f"CRITICAL: Could not register advanced components: {e}")
    ADVANCED_REGISTRATION_COMPLETE = False

__all__ = [
    'simple_backbones',
    'embeddings', 
    'attentions'
]

if BACKBONES_AVAILABLE:
    __all__.append('backbones')

if ADVANCED_LOSSES_AVAILABLE:
    __all__.append('advanced_losses')

if ADVANCED_ATTENTIONS_AVAILABLE:
    __all__.append('advanced_attentions')

if SPECIALIZED_PROCESSORS_AVAILABLE:
    __all__.append('specialized_processors')

if ADVANCED_REGISTRATION_COMPLETE:
    __all__.append('register_advanced')


def get_integration_status():
    """Get status of all advanced integrations"""
    return {
        'backbones_available': BACKBONES_AVAILABLE,
        'advanced_losses_available': ADVANCED_LOSSES_AVAILABLE,
        'advanced_attentions_available': ADVANCED_ATTENTIONS_AVAILABLE,
        'specialized_processors_available': SPECIALIZED_PROCESSORS_AVAILABLE,
        'advanced_registration_complete': ADVANCED_REGISTRATION_COMPLETE,
        'total_modules': len(__all__)
    }


def validate_critical_integrations():
    """Validate that critical Bayesian integrations are working"""
    if not ADVANCED_REGISTRATION_COMPLETE:
        logger.error("CRITICAL: Advanced component registration failed!")
        return False
    
    try:
        from .register_advanced import validate_bayesian_integration
        validation = validate_bayesian_integration()
        
        critical_checks = [
            validation.get('bayesian_mse_registered', False),
            validation.get('kl_divergence_supported', False),
            validation.get('uncertainty_supported', False)
        ]
        
        if all(critical_checks):
            logger.info("✅ Critical Bayesian integrations validated successfully")
            return True
        else:
            logger.error(f"❌ Critical validation failed: {validation}")
            return False
            
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return False


# Perform validation on import
try:
    if validate_critical_integrations():
        logger.info("🎉 Modular framework with advanced integrations ready!")
    else:
        logger.warning("⚠️ Some advanced integrations may not be fully functional")
except Exception as e:
    logger.error(f"Validation check failed: {e}")
