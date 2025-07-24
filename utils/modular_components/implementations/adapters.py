"""
Adapter Components for Modular Framework

This module provides adapter implementations that extend backbone capabilities
while preserving their core functionality.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List, Tuple

from ..base_interfaces import BaseBackbone
from ..config_schemas import BackboneConfig

logger = logging.getLogger(__name__)



# Modular Adapter family imports
from .Attention.Adapter.covariate_adapter import CovariateAdapter
from .Attention.Adapter.multiscale_adapter import MultiScaleAdapter
