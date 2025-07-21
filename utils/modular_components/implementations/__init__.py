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


# Import unified attention implementations
try:
    from . import Attention  # Our unified attention module
    ATTENTION_AVAILABLE = True
    logger.info("Unified Attention module integration available")
except ImportError as e:
    logger.warning(f"Could not import unified Attention: {e}")
    ATTENTION_AVAILABLE = False

# Import unified loss implementations
try:
    from . import Losses  # Our unified loss module
    LOSSES_AVAILABLE = True
    logger.info("Unified Losses module integration available")
except ImportError as e:
    logger.warning(f"Could not import unified Losses: {e}")
    LOSSES_AVAILABLE = False

# Import unified decomposition implementations
try:
    from . import Decomposition  # Our unified decomposition module
    DECOMPOSITION_AVAILABLE = True
    logger.info("Unified Decomposition module integration available")
except ImportError as e:
    logger.warning(f"Could not import unified Decomposition: {e}")
    DECOMPOSITION_AVAILABLE = False

# Skip advanced_attentions - we use unified Attention.py instead
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
    'embeddings'
]


if ATTENTION_AVAILABLE:
    __all__.append('Attention')  # Our unified attention module

if LOSSES_AVAILABLE:
    __all__.append('Losses')     # Our unified loss module

if DECOMPOSITION_AVAILABLE:
    __all__.append('Decomposition')  # Our unified decomposition module

if BACKBONES_AVAILABLE:
    __all__.append('backbones')

if LOSSES_AVAILABLE:
    __all__.append('Losses')

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
        'attention_available': ATTENTION_AVAILABLE,
        'losses_available': LOSSES_AVAILABLE,
        'advanced_attentions_available': ADVANCED_ATTENTIONS_AVAILABLE,
        'specialized_processors_available': SPECIALIZED_PROCESSORS_AVAILABLE,
        'advanced_registration_complete': ADVANCED_REGISTRATION_COMPLETE,
        'total_modules': len(__all__)
    }


def validate_critical_integrations():
    """Validate that critical unified modules are working"""
    success = True
    

    # Validate Losses module
    if not LOSSES_AVAILABLE:
        logger.error("CRITICAL: Unified Losses module failed to load!")
        success = False
    else:
        try:
            from .Losses import LOSS_REGISTRY, get_loss_function
            # Test critical loss functions
            critical_losses = ['bayesian_mse', 'bayesian_mae', 'mse', 'mae']
            available_losses = list(LOSS_REGISTRY.keys())
            missing_losses = [loss for loss in critical_losses if loss not in available_losses]
            if missing_losses:
                logger.error(f"❌ Critical losses missing: {missing_losses}")
                success = False
            else:
                # Test creating a Bayesian loss function
                try:
                    from .Losses import LossConfig
                    config = LossConfig()
                    bayesian_mse = get_loss_function('bayesian_mse', config)
                    logger.info("✅ Bayesian MSE loss successfully created")
                    logger.info("✅ Critical loss functions validated successfully")
                except Exception as e:
                    logger.error(f"❌ Failed to create Bayesian MSE loss: {e}")
                    success = False
        except Exception as e:
            logger.error(f"❌ Failed to validate loss functions: {e}")
            success = False

    # Validate Decomposition module
    if not DECOMPOSITION_AVAILABLE:
        logger.error("CRITICAL: Unified Decomposition module failed to load!")
        success = False
    else:
        try:
            from .Decomposition import DECOMPOSITION_REGISTRY, get_decomposition_method
            # Test critical decomposition methods
            critical_decomps = ['series_decomp', 'stable_decomp', 'learnable_decomp', 'wavelet_decomp']
            available_decomps = list(DECOMPOSITION_REGISTRY.keys())
            missing_decomps = [d for d in critical_decomps if d not in available_decomps]
            if missing_decomps:
                logger.error(f"❌ Critical decompositions missing: {missing_decomps}")
                success = False
            else:
                # Test creating a SeriesDecomposition
                try:
                    series_decomp = get_decomposition_method('series_decomp', kernel_size=7)
                    logger.info("✅ SeriesDecomposition successfully created")
                    logger.info("✅ Critical decomposition methods validated successfully")
                except Exception as e:
                    logger.error(f"❌ Failed to create SeriesDecomposition: {e}")
                    success = False
        except Exception as e:
            logger.error(f"❌ Failed to validate decomposition methods: {e}")
            success = False
                    
        except Exception as e:
            logger.error(f"❌ Failed to validate loss functions: {e}")
            success = False
    
    if success:
        logger.info("✅ Critical unified integrations validated successfully")
    
    return success


# Perform validation on import
try:
    if validate_critical_integrations():
        logger.info("🎉 Modular framework with advanced integrations ready!")
    else:
        logger.warning("⚠️ Some advanced integrations may not be fully functional")
except Exception as e:
    logger.error(f"Validation check failed: {e}")


# Component registration is now handled by unified modules
# All components are available through their respective unified files:
# - Attention.py contains all 29 attention components with comprehensive registry
# - Losses.py contains all 25+ loss functions with Bayesian support and KL divergence
# - componentHelpers.py contains utility classes (BayesianLinear, WaveletDecomposition, etc.)
logger.info("🎉 Modular framework with unified components ready!")