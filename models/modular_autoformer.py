import torch
import torch.nn as nn
from argparse import Namespace
from typing import Union, Dict, Any

# GCLI Structured Configuration System
from configs.schemas import ModularAutoformerConfig
from utils.modular_components.implementations.processorss import get_processor_component

# Import unified base framework
from models.base_forecaster import BaseTimeSeriesForecaster, CustomFrameworkMixin
from utils.logger import logger

class ModularAutoformer(BaseTimeSeriesForecaster, CustomFrameworkMixin):
    """
    A completely modular Autoformer that uses a processor-centric architecture.
    The processor component defines the model's forward pass logic.
    """
    def __init__(self, configs: Union[Namespace, ModularAutoformerConfig]):
        super(ModularAutoformer, self).__init__(configs)
        logger.info("Initializing ModularAutoformer with processor-centric architecture")

        self.framework_type = 'custom'
        self.model_type = 'modular_autoformer'

        if isinstance(configs, dict):
            configs = Namespace(**configs)
        self.configs = configs

        # The processor is now the heart of the model
        self.processor = get_processor_component(configs.processor['type'], configs=configs)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        The forward pass is now delegated entirely to the processor.
        """
        return self.processor.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)