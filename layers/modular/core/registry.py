"""Unified component registry.

All component families (attention, encoder, decoder, decomposition, sampling,
output_head, loss, fusion, adapter, backbone) register here. This replaces the
multiple per-domain registries and utilities registries.
"""
from __future__ import annotations
from typing import Any, Dict, Type, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import inspect

from configs.schemas import ComponentType  # canonical enum
from .interfaces import (
    AttentionLike, EncoderLike, DecoderLike, DecompositionLike,
    SamplingLike, OutputHeadLike, LossLike, CAPABILITY_FLAGS
)
from utils.logger import logger

Family = str

class ComponentFamily(str, Enum):
    ATTENTION = 'attention'
    ENCODER = 'encoder'
    DECODER = 'decoder'
    DECOMPOSITION = 'decomposition'
    SAMPLING = 'sampling'
    OUTPUT_HEAD = 'output_head'
    LOSS = 'loss'
    FUSION = 'fusion'
    ADAPTER = 'adapter'
    BACKBONE = 'backbone'
    EMBEDDING = 'embedding'
    PROCESSOR = 'processor'
    FEEDFORWARD = 'feedforward'
    OUTPUT = 'output'

PROTO_MAP = {
    ComponentFamily.ATTENTION: AttentionLike,
    ComponentFamily.ENCODER: EncoderLike,
    ComponentFamily.DECODER: DecoderLike,
    ComponentFamily.DECOMPOSITION: DecompositionLike,
    ComponentFamily.SAMPLING: SamplingLike,
    ComponentFamily.OUTPUT_HEAD: OutputHeadLike,
    ComponentFamily.LOSS: LossLike,
}

@dataclass
class Registration:
    family: ComponentFamily
    name: str
    cls: Type
    component_type: Optional[ComponentType] = None
    aliases: tuple[str, ...] = tuple()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches_protocol(self) -> bool:
        proto = PROTO_MAP.get(self.family)
        if not proto:
            return True  # families without protocol enforcement
        try:
            return isinstance(self.cls, type) and all(hasattr(self.cls, '__dict__') for _ in [0]) and hasattr(self.cls, 'forward')
        except Exception:
            return False

class UnifiedRegistry:
    def __init__(self):
        self._by_family: Dict[ComponentFamily, Dict[str, Registration]] = {f: {} for f in ComponentFamily}
        self._alias_index: Dict[str, Registration] = {}

    # ---------------- Registration -----------------
    def register(self, family: ComponentFamily, name: str, cls: Type, *, component_type: Optional[ComponentType]=None,
                 aliases: Optional[list[str]]=None, metadata: Optional[Dict[str, Any]]=None):
        reg = Registration(family=family, name=name, cls=cls, component_type=component_type,
                           aliases=tuple(aliases or ()), metadata=metadata or {})
        if not reg.matches_protocol():
            logger.warning(f"Class {cls.__name__} for {family}:{name} does not clearly match expected protocol signature")
        if name in self._by_family[family]:
            logger.warning(f"Overwriting existing registration {family}:{name}")
        self._by_family[family][name] = reg
        for alias in reg.aliases:
            self._alias_index[f"{family.value}:{alias}"] = reg
        logger.info(f"Registered {family.value} component: {name} (aliases={reg.aliases})")

    # ---------------- Lookup -----------------
    def resolve(self, family: ComponentFamily, name: str) -> Registration:
        reg = self._by_family[family].get(name)
        if reg:
            return reg
        alias_key = f"{family.value}:{name}"
        if alias_key in self._alias_index:
            return self._alias_index[alias_key]
        raise KeyError(f"Component not found: {family.value}:{name}")

    def create(self, family: ComponentFamily, name: str, **kwargs):
        reg = self.resolve(family, name)
        try:
            instance = reg.cls(**kwargs)
        except TypeError as e:
            # Attempt constructor argument normalization: common alias mapping
            norm_kwargs = self._normalize_constructor_args(reg.cls, kwargs)
            instance = reg.cls(**norm_kwargs)
        self._attach_capabilities(instance, reg)
        return instance

    # ---------------- Utilities -----------------
    def list(self, family: Optional[ComponentFamily]=None) -> Dict[str, list[str]]:
        if family:
            return {family.value: sorted(self._by_family[family].keys())}
        return {f.value: sorted(regs.keys()) for f, regs in self._by_family.items() if regs}

    def describe(self, family: ComponentFamily, name: str) -> Dict[str, Any]:
        reg = self.resolve(family, name)
        info = {
            'family': reg.family.value,
            'name': reg.name,
            'component_type': reg.component_type.value if reg.component_type else None,
            'aliases': list(reg.aliases),
            'metadata': reg.metadata,
        }
        try:
            sig = inspect.signature(reg.cls.__init__)
            info['init_signature'] = str(sig)
        except Exception:
            pass
        return info

    # ---------------- Internal helpers -----------------
    @staticmethod
    def _normalize_constructor_args(cls: Type, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        mapping = {
            'e_layers': 'num_encoder_layers',
            'd_layers': 'num_decoder_layers',
        }
        norm = dict(kwargs)
        for old, new in mapping.items():
            if old in norm and new not in norm:
                norm[new] = norm.pop(old)
        # Reverse mapping (new -> legacy) if target class still uses legacy parameter names
        try:
            sig = inspect.signature(cls.__init__)
            params = sig.parameters
            reverse_pairs = [
                ('num_encoder_layers', 'e_layers'),
                ('num_decoder_layers', 'd_layers'),
            ]
            for new, legacy in reverse_pairs:
                if new in norm and legacy in params and legacy not in norm:
                    # Move value to legacy name for constructor compatibility
                    norm[legacy] = norm.pop(new)
        except Exception:  # pragma: no cover - defensive
            pass
        # Attention-specific normalization
        try:
            sig = inspect.signature(cls.__init__)
            params = sig.parameters
            if 'attention_dropout' in params and 'dropout' in norm and 'attention_dropout' not in norm:
                # Rename generic dropout to attention-specific name
                norm['attention_dropout'] = norm.pop('dropout')
            # If target class does not accept a 'dropout' or 'attention_dropout', remove stray value to avoid TypeError
            if 'dropout' in norm and 'dropout' not in params and 'attention_dropout' not in params:
                norm.pop('dropout')
            # Remove seq_len if not accepted
            if 'seq_len' in norm and 'seq_len' not in params:
                norm.pop('seq_len')
        except Exception:
            pass
        return norm

    @staticmethod
    def _attach_capabilities(instance, reg: Registration):
        # Provide consistent capability flags defaulting to False if absent
        for flag in CAPABILITY_FLAGS:
            if not hasattr(instance, flag):
                setattr(instance, flag, reg.metadata.get(flag, False))

# Global singleton
unified_registry = UnifiedRegistry()
