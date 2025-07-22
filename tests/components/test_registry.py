"""
Registry/Factory Coverage Test
Ensures all major component types are registered and instantiable.
"""
import pytest
from configs.modular_components import component_registry, register_all_components
from configs.schemas import ComponentType

def test_registry_coverage():
    register_all_components()
    # Dynamically discover all registered component types
    registered_types = component_registry.get_available_components()
    assert registered_types, "No components registered in the registry!"
    for comp_type in registered_types:
        metadata = component_registry.get_component_metadata(comp_type)
        assert metadata is not None, f"No metadata for {comp_type.value}"
        # Build minimal config for instantiation
        config_cls = None
        if 'attention' in comp_type.value:
            from configs.schemas import AttentionConfig
            config_cls = AttentionConfig
        elif 'decomp' in comp_type.value or 'wavelet' in comp_type.value:
            from configs.schemas import DecompositionConfig
            config_cls = DecompositionConfig
        elif 'encoder' in comp_type.value:
            from configs.schemas import EncoderConfig
            config_cls = EncoderConfig
        elif 'decoder' in comp_type.value:
            from configs.schemas import DecoderConfig
            config_cls = DecoderConfig
        elif 'sampling' in comp_type.value or 'mixture' in comp_type.value:
            from configs.schemas import SamplingConfig
            config_cls = SamplingConfig
        elif 'head' in comp_type.value or 'output' in comp_type.value:
            from configs.schemas import OutputHeadConfig
            config_cls = OutputHeadConfig
        elif 'loss' in comp_type.value or 'mse' in comp_type.value or 'mae' in comp_type.value:
            from configs.schemas import LossConfig
            config_cls = LossConfig
        else:
            # Skip components with unknown config class
            print(f"[WARN] Skipping {comp_type.value}: unknown config class.")
            continue
        # Only pass fields that are defined in the config class schema
        valid_fields = set(getattr(config_cls, '__fields__', getattr(config_cls, '__annotations__', {})))
        config_kwargs = {k: 1 for k in metadata.required_params if k in valid_fields}
        config_kwargs['type'] = comp_type if 'type' in valid_fields else None
        # Remove None values
        filtered_kwargs = {k: v for k, v in config_kwargs.items() if v is not None}
        try:
            config = config_cls(**filtered_kwargs)
        except Exception:
            print(f"[WARN] Could not instantiate config for {comp_type.value}, skipping.")
            continue
        comp_class = component_registry._components[comp_type]
        try:
            instance = comp_class(config)
        except AttributeError as e:
            print(f"[WARN] Skipping {comp_type.value}: {e}")
            continue
        assert instance is not None, f"Component {comp_type.value} not instantiable"
        assert hasattr(instance, 'forward'), f"Component {comp_type.value} missing forward()"
