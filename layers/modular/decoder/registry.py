
from .standard_decoder import StandardDecoder
from .enhanced_decoder import EnhancedDecoder
from .stable_decoder import StableDecoder
from .validation import ComponentValidator
from utils.logger import logger
import inspect
import torch.nn

class DecoderRegistry:
    """
    A registry for all available decoder components with validation.
    """
    _registry = {
        "standard": StandardDecoder,
        "enhanced": EnhancedDecoder,
        "stable": StableDecoder,
    }
    _validator = ComponentValidator()

    @classmethod
    def register(cls, name, component_class, validate=True):
        """Register a decoder component with optional validation."""
        if name in cls._registry:
            logger.warning(f"Component '{name}' is already registered and will be overwritten.")
        
        if validate:
            # Validate the component class
            try:
                # Check if it's a class and has the right interface
                if not inspect.isclass(component_class):
                    raise ValueError(f"Component '{name}' must be a class")
                
                # Check if it has forward method
                if not hasattr(component_class, 'forward'):
                    raise ValueError(f"Component '{name}' must have a forward method")
                
                logger.info(f"Validation passed for decoder component: {name}")
            except Exception as e:
                logger.error(f"Validation failed for component '{name}': {str(e)}")
                raise ValueError(f"Component validation failed: {str(e)}")
        
        cls._registry[name] = component_class
        logger.info(f"Registered decoder component: {name}")

    @classmethod
    def get(cls, name):
        component = cls._registry.get(name)
        if component is None:
            logger.error(f"Decoder component '{name}' not found.")
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Decoder component '{name}' not found. Available: {available}")
        return component

    @classmethod
    def list_components(cls):
        return list(cls._registry.keys())
    
    @classmethod
    def validate_component(cls, name):
        """Validate a registered component."""
        component_class = cls.get(name)
        
        # Create a dummy instance for validation (if possible)
        try:
            # This is a basic validation - in practice, you'd need proper parameters
            validation_results = {
                'name': name,
                'class': component_class.__name__,
                'has_forward': hasattr(component_class, 'forward'),
                'is_subclass_of_module': issubclass(component_class, torch.nn.Module) if hasattr(component_class, '__bases__') else False
            }
            return validation_results
        except Exception as e:
            logger.error(f"Error validating component '{name}': {str(e)}")
            return {'name': name, 'error': str(e)}
    
    @classmethod
    def get_component_info(cls, name):
        """Get detailed information about a component."""
        component_class = cls.get(name)
        
        info = {
            'name': name,
            'class_name': component_class.__name__,
            'module': component_class.__module__,
            'doc': component_class.__doc__,
        }
        
        # Get forward method signature
        if hasattr(component_class, 'forward'):
            try:
                sig = inspect.signature(component_class.forward)
                info['forward_signature'] = str(sig)
                info['parameters'] = list(sig.parameters.keys())
            except Exception as e:
                info['signature_error'] = str(e)
        
        return info

def get_decoder_component(name, **kwargs):
    """Get a decoder component instance with validation."""
    component_class = DecoderRegistry.get(name)
    
    try:
        instance = component_class(**kwargs)
        logger.info(f"Created decoder component '{name}' successfully")
        return instance
    except Exception as e:
        logger.error(f"Failed to create decoder component '{name}': {str(e)}")
        raise RuntimeError(f"Failed to create decoder component '{name}': {str(e)}") from e

# ---------------- Deprecation Shim: Forward to unified registry -----------------
try:
    from layers.modular.core import unified_registry, ComponentFamily  # type: ignore
    import warnings
    _dec_dep_warned = False

    def _warn_dec():
        global _dec_dep_warned
        if not _dec_dep_warned:
            warnings.warn(
                "DecoderRegistry is deprecated – use unified_registry.create(ComponentFamily.DECODER, name, **kwargs)",
                DeprecationWarning,
                stacklevel=2,
            )
            _dec_dep_warned = True

    _LEGACY_TO_UNIFIED = {}

    @classmethod  # type: ignore
    def _shim_register(cls, name, component_class, validate=True):
        _warn_dec()
        unified_registry.register(ComponentFamily.DECODER, name, component_class)

    @classmethod  # type: ignore
    def _shim_get(cls, name):
        _warn_dec()
        lookup = _LEGACY_TO_UNIFIED.get(name, name)
        try:
            return unified_registry.resolve(ComponentFamily.DECODER, lookup).cls
        except Exception:
            return cls._registry.get(name)

    @classmethod  # type: ignore
    def _shim_list(cls):
        _warn_dec()
        names = list(unified_registry.list(ComponentFamily.DECODER)[ComponentFamily.DECODER.value])
        for n in cls._registry.keys():
            if n not in names:
                names.append(n)
        return sorted(names)

    DecoderRegistry.register = _shim_register  # type: ignore
    DecoderRegistry.get = _shim_get  # type: ignore
    DecoderRegistry.list_components = _shim_list  # type: ignore
except Exception:  # pragma: no cover
    pass
