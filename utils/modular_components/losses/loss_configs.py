"""
Centralized configuration dataclasses for all modular loss functions.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

@dataclass
class LossConfig:
    pass

@dataclass
class BayesianLossConfig(LossConfig):
    pass

@dataclass  
class AdaptiveLossConfig(LossConfig):
    pass

@dataclass
class FrequencyLossConfig(LossConfig):
    pass

@dataclass
class StructuralLossConfig(LossConfig):
    pass

@dataclass
class QuantileConfig(LossConfig):
    pass

@dataclass
class FocalLossConfig(LossConfig):
    pass

@dataclass
class DTWConfig(LossConfig):
    pass

@dataclass
class CustomLossConfig(LossConfig):
    pass
