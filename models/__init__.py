"""Lightweight models package initializer with lazy loading.

Previously this module eagerly imported every model variant, which cascaded into
optional heavy dependencies (e.g., patoolib via M4 dataset utilities) even when
only a single lightweight component (like ChronosXAutoformer) was required.

To improve startup time and reduce extraneous dependency errors for scripts that
only need a subset of models, we now expose a lazy import mechanism. Attribute
access triggers an on-demand import of the underlying module/class.

Adding a new model:
    1. Append an entry to _MODEL_SPECS with key == public attribute name.
    2. Value is a (module_path, attribute_name) tuple. If the class exported is
       named ``Model`` in the module, keep attribute_name as "Model" and choose a
       descriptive key (e.g. "Autoformer").
    3. The class will become importable via ``from models import Autoformer``.

This preserves backward compatibility while avoiding import side effects unless
the model is actually requested.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Dict, Tuple, Any

_MODEL_SPECS: Dict[str, Tuple[str, str]] = {
    # Standard models: public name -> (module path, attribute/class name)
    "Autoformer": (".Autoformer", "Model"),
    "Transformer": (".Transformer", "Model"),
    "TimesNet": (".TimesNet", "Model"),
    "Nonstationary_Transformer": (".Nonstationary_Transformer", "Model"),
    "DLinear": (".DLinear", "Model"),
    "FEDformer": (".FEDformer", "Model"),
    "Informer": (".Informer", "Model"),
    "LightTS": (".LightTS", "Model"),
    "Reformer": (".Reformer", "Model"),
    "ETSformer": (".ETSformer", "Model"),
    "Pyraformer": (".Pyraformer", "Model"),
    "PatchTST": (".PatchTST", "Model"),
    "MICN": (".MICN", "Model"),
    "Crossformer": (".Crossformer", "Model"),
    "FiLM": (".FiLM", "Model"),
    "iTransformer": (".iTransformer", "Model"),
    "Koopa": (".Koopa", "Model"),
    "TiDE": (".TiDE", "Model"),
    "FreTS": (".FreTS", "Model"),
    "TimeMixer": (".TimeMixer", "Model"),
    "TSMixer": (".TSMixer", "Model"),
    "SegRNN": (".SegRNN", "Model"),
    "MambaSimple": (".MambaSimple", "Model"),
    "TemporalFusionTransformer": (".TemporalFusionTransformer", "Model"),
    "SCINet": (".SCINet", "Model"),
    "PAttn": (".PAttn", "Model"),
    "TimeXer": (".TimeXer", "Model"),
    "WPMixer": (".WPMixer", "Model"),
    "MultiPatchFormer": (".MultiPatchFormer", "Model"),
    "SOTA_Temporal_PGAT": (".SOTA_Temporal_PGAT", "SOTA_Temporal_PGAT"),
    # Enhanced models
    "EnhancedAutoformer": (".Autoformer", "EnhancedAutoformer"),
    "BayesianEnhancedAutoformer": (".BayesianEnhancedAutoformer", "BayesianEnhancedAutoformer"),
    "HierarchicalEnhancedAutoformer": (".HierarchicalEnhancedAutoformer", "HierarchicalEnhancedAutoformer"),
    # HF models / advanced factory outputs
    "HFEnhancedAutoformer": (".HFEnhancedAutoformer", "HFEnhancedAutoformer"),
    "HFBayesianAutoformerProduction": (".HFBayesianAutoformerProduction", "HFBayesianAutoformerProduction"),
    "HFBayesianEnhancedAutoformer": (".HFAdvancedFactory", "HFBayesianEnhancedAutoformer"),
    "HFHierarchicalEnhancedAutoformer": (".HFAdvancedFactory", "HFHierarchicalEnhancedAutoformer"),
    "HFQuantileEnhancedAutoformer": (".HFAdvancedFactory", "HFQuantileEnhancedAutoformer"),
    "HFFullEnhancedAutoformer": (".HFAdvancedFactory", "HFFullEnhancedAutoformer"),
    "create_hf_model_from_config": (".HFAdvancedFactory", "create_hf_model_from_config"),
}

__all__ = list(_MODEL_SPECS.keys())


def __getattr__(name: str) -> Any:  # pragma: no cover - simple delegation
    spec = _MODEL_SPECS.get(name)
    if spec is None:
        raise AttributeError(f"module 'models' has no attribute '{name}'")
    module_path, attr_name = spec
    module: ModuleType = import_module(module_path, package=__name__)
    try:
        attr = getattr(module, attr_name)
    except AttributeError as exc:  # re-raise with clearer context
        raise AttributeError(f"Attribute '{attr_name}' not found in module '{module_path}' for '{name}'") from exc
    globals()[name] = attr  # cache for future lookups
    return attr


def __dir__():  # pragma: no cover
    return sorted(list(globals().keys()) + __all__)
