
from .standard_encoder import StandardEncoder
from .enhanced_encoder import EnhancedEncoder
from .stable_encoder import StableEncoder
from .hierarchical_encoder import HierarchicalEncoder
from .graph_encoder import GraphTimeSeriesEncoder, HybridGraphEncoder, AdaptiveGraphEncoder
from utils.logger import logger

class EncoderRegistry:
    """
    A registry for all available encoder components.
    """
    _registry = {
        "standard": StandardEncoder,
        "enhanced": EnhancedEncoder,
        "stable": StableEncoder,
        "hierarchical": HierarchicalEncoder,
        "graph": GraphTimeSeriesEncoder,
        "hybrid_graph": HybridGraphEncoder,
        "adaptive_graph": AdaptiveGraphEncoder,
    }

    @classmethod
    def register(cls, name, component_class):
        if name in cls._registry:
            logger.warning(f"Component '{name}' is already registered and will be overwritten.")
        cls._registry[name] = component_class
        logger.info(f"Registered encoder component: {name}")

    @classmethod
    def get(cls, name):
        component = cls._registry.get(name)
        if component is None:
            logger.error(f"Encoder component '{name}' not found.")
            raise ValueError(f"Encoder component '{name}' not found.")
        return component

    @classmethod
    def list_components(cls):
        return list(cls._registry.keys())

def get_encoder_component(name, **kwargs):
    component_class = EncoderRegistry.get(name)
    # Support two hierarchical pathways:
    # 1) Modular hierarchical encoder defined in layers/modular/encoder/hierarchical_encoder.py expecting
    #    (e_layers, d_model, n_heads, d_ff, dropout, activation, attention_type, decomp_type, decomp_params,...)
    # 2) Enhancedcomponents HierarchicalEncoder variant expecting (configs, n_levels, share_weights)
    if name == 'hierarchical':
        # Map legacy kwarg 'e_layers' if present before any constructor call
        if 'e_layers' in kwargs:
            kwargs = {**kwargs}
            if 'num_encoder_layers' not in kwargs:
                kwargs['num_encoder_layers'] = kwargs.get('e_layers')
            kwargs.pop('e_layers', None)
        try:
            return component_class(**kwargs)
        except TypeError:
            # Attempt configs-based signature fallback if provided
            configs = kwargs.get('configs')
            n_levels = kwargs.get('n_levels', 2)
            share_weights = kwargs.get('share_weights', False)
            if configs is None:
                raise
            return component_class(configs, n_levels, share_weights)
    return component_class(**kwargs)

# ---------------- Deprecation Shim: Forward to unified registry -----------------
try:
    from layers.modular.core import unified_registry, ComponentFamily  # type: ignore
    import warnings
    _enc_dep_warned = False

    def _warn_enc():
        global _enc_dep_warned
        if not _enc_dep_warned:
            warnings.warn(
                "EncoderRegistry is deprecated – use unified_registry.create(ComponentFamily.ENCODER, name, **kwargs)",
                DeprecationWarning,
                stacklevel=2,
            )
            _enc_dep_warned = True

    _LEGACY_TO_UNIFIED = {}

    @classmethod  # type: ignore
    def _shim_register(cls, name, component_class):
        _warn_enc()
        unified_registry.register(ComponentFamily.ENCODER, name, component_class)

    @classmethod  # type: ignore
    def _shim_get(cls, name):
        _warn_enc()
        lookup = _LEGACY_TO_UNIFIED.get(name, name)
        try:
            return unified_registry.resolve(ComponentFamily.ENCODER, lookup).cls
        except Exception:
            return cls._registry.get(name)

    @classmethod  # type: ignore
    def _shim_list(cls):
        _warn_enc()
        names = list(unified_registry.list(ComponentFamily.ENCODER)[ComponentFamily.ENCODER.value])
        for n in cls._registry.keys():
            if n not in names:
                names.append(n)
        return sorted(names)

    EncoderRegistry.register = _shim_register  # type: ignore
    EncoderRegistry.get = _shim_get  # type: ignore
    EncoderRegistry.list_components = _shim_list  # type: ignore
except Exception:  # pragma: no cover
    pass
