import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from models.modular_autoformer import ModularAutoformer
from configs.schemas import ModularAutoformerConfig
from layers.modular.decomposition.wavelet_decomposition import WaveletHierarchicalDecomposition
from layers.modular.attention.fourier_attention import FourierAttention
from layers.modular.attention.temporal_conv_attention import TemporalConvNet
from layers.modular.attention.adaptive_components import AdaptiveMixture
from layers.modular.output_heads.standard_output_head import StandardOutputHead
from utils.modular_components.implementations.embeddings import HybridEmbedding as EnhancedEmbedding
from utils.kl_tuning import KLTuner as KLTuning
from utils.bayesian_losses import BayesianQuantileLoss
from configs.schemas import create_adaptive_mixture_config

class Model(ModularAutoformer):
    def __init__(self, configs):
        config = create_adaptive_mixture_config(num_targets=configs.c_out, num_covariates=0, **vars(configs))
        super().__init__(config)
        self.hybrid_attention = self._build_hybrid_attention(config)
        self.meta_adapter = nn.Linear(config.d_model, config.d_model)  # Simple meta-learning adapter
        self.kl_tuning = KLTuning(config.d_model)

    def _build_hybrid_attention(self, config):
        attentions = []
        # Hardcode components for hybrid attention
        attentions.append(FourierAttention(config.d_model))
        attentions.append(TemporalConvNet(config.d_model, kernel_size=3))
        return nn.ModuleList(attentions)

    def forward(self, x: torch.Tensor, x_mark: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        # Decomposition
        decomposed = self.decomposition(x)
        # Embedding
        embedded = self.embedding(decomposed, x_mark)
        # Hybrid Attention
        attn_out = embedded
        for attn in self.hybrid_attention:
            attn_out = attn(attn_out)
        # Fusion
        fused = self.fusion(attn_out)
        # Meta adaptation
        adapted = self.meta_adapter(fused)
        # KL Tuning
        tuned = self.kl_tuning(adapted)
        # Multi-scale processing (simplified)
        multi_scale = F.avg_pool1d(tuned.transpose(1,2), kernel_size=2).transpose(1,2) + tuned
        # Output heads
        outputs = self.output_head(multi_scale, **kwargs)
        return outputs

    def loss(self, outputs, targets, quantiles=None):
        base_loss = super().loss(outputs, targets)
        if self.config.task == 'probabilistic':
            bayes_loss = BayesianQuantileLoss()(outputs['mean'], outputs['std'], targets, quantiles)
            return base_loss + bayes_loss
        return base_loss