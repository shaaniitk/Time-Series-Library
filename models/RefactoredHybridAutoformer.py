import torch
import torch.nn as nn
from models.modular_autoformer import ModularAutoformer, ModularAutoformerConfig
from layers.modular.attention.temporal_conv_attention import TemporalConvNet
from layers.modular.attention.adaptive_components import MetaLearningAdapter
from layers.modular.decomposition.wavelet_decomposition import WaveletDecomposition
from layers.modular.attention.fourier_attention import FourierAttention
from layers.modular.sampling.adaptive_mixture import AdaptiveMixture

class RefactoredHybridAutoformer(ModularAutoformer):
    def __init__(self, config: ModularAutoformerConfig):
        super().__init__(config)
        # Hybrid-specific configurations
        self.wavelet_decomp = WaveletDecomposition(config.d_model)
        self.fourier_attn = FourierAttention(config.d_model, config.n_heads)
        self.temporal_conv = TemporalConvNet(config.d_model, config.n_heads)
        self.meta_adapter = MetaLearningAdapter(config.d_model)
        self.adaptive_mixture = AdaptiveMixture(config.d_model, num_experts=4)
        # Modular heads for different tasks
        self.forecast_head = nn.Linear(config.d_model, config.pred_len * config.c_out)
        self.classification_head = nn.Linear(config.d_model, config.num_classes) if config.task_name == 'classification' else None

    def forward(self, x, x_mark=None, dec_inp=None, batch_y_mark=None):
        # Apply wavelet decomposition
        low_freq, high_freq = self.wavelet_decomp(x)
        # Encoder with hybrid attention
        enc_out = self.encoder(low_freq)
        enc_out = self.fourier_attn(enc_out)
        enc_out = self.temporal_conv(enc_out)
        enc_out = self.meta_adapter(lambda: enc_out)  # Simplified adaptation
        # Decoder with adaptive mixture
        dec_out = self.decoder(enc_out, dec_inp)
        dec_out = self.adaptive_mixture(dec_out)
        # Task-specific output
        if self.config.task_name == 'long_term_forecast':
            output = self.forecast_head(dec_out).view(dec_out.size(0), self.config.pred_len, self.config.c_out)
        elif self.config.task_name == 'classification':
            output = self.classification_head(dec_out.mean(dim=1))
        return output