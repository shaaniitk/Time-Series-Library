"""
Wavelet & Hierarchical Processing Module for HF Models (Modularized)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from typing import List, Dict, Optional
from argparse import Namespace

from utils.modular_components.factories import create_processor
from utils.modular_components.config_schemas import ProcessorConfig
from utils.logger import logger

class WaveletProcessor(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        self.wavelet_type = getattr(configs, 'wavelet_type', 'db4')
        self.decomposition_levels = getattr(configs, 'decomposition_levels', 3)
        
        processor_config = ProcessorConfig(
            processor_type='wavelet',
            custom_params={
                'wavelet_type': self.wavelet_type,
                'levels': self.decomposition_levels
            }
        )
        self.processor = create_processor(processor_config)
        
    def forward(self, x, process_method='decompose_reconstruct'):
        return self.processor(x)

class HierarchicalProcessor(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        self.hierarchy_levels = getattr(configs, 'hierarchy_levels', [1, 2, 4])
        
        processor_config = ProcessorConfig(
            processor_type='hierarchical',
            custom_params={'scales': self.hierarchy_levels}
        )
        self.processor = create_processor(processor_config)
        
    def forward(self, x, target_level=None):
        return self.processor(x)

def create_wavelet_processor(configs):
    return WaveletProcessor(configs)

def create_hierarchical_processor(configs):
    return HierarchicalProcessor(configs)