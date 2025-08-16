import torch
import torch.nn as nn
from models.modular_autoformer import ModularAutoformer
from configs.schemas import ModularAutoformerConfig
from layers.modular.decomposition.learnable_decomposition import LearnableSeriesDecomposition as LearnableSeriesDecomp
from layers.modular.encoder.enhanced_encoder import EnhancedEncoder
from layers.modular.decoder.enhanced_decoder import EnhancedDecoder
from layers.modular.attention.enhanced_autocorrelation import EnhancedAutoCorrelation
from layers.modular.output_heads.standard_output_head import StandardOutputHead
from configs.schemas import create_hierarchical_config

class Model(ModularAutoformer):
    def __init__(self, configs):
        config = create_hierarchical_config(num_targets=configs.c_out, num_covariates=0, **vars(configs))
        super().__init__(config)