import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modular_autoformer import ModularAutoformer, ModularAutoformerConfig
from layers.modular.decomposition.learnable_decomposition import LearnableSeriesDecomposition as LearnableSeriesDecomp
from layers.modular.encoder.enhanced_encoder import EnhancedEncoder
from layers.modular.decoder.enhanced_decoder import EnhancedDecoder
from layers.modular.attention.enhanced_autocorrelation import EnhancedAutoCorrelation
from layers.modular.output_heads.standard_output_head import StandardOutputHead
from layers.modular.sampling.monte_carlo_sampling import MonteCarloSampling
from layers.modular.losses.bayesian_loss import BayesianLoss
from layers.Autoformer_EncDec import series_decomp

class Model(ModularAutoformer):
    def __init__(self, config):
        super().__init__(config)
        self.config = ModularAutoformerConfig.from_legacy(config)
        
        # Set Bayesian-specific configurations
        self.config.sampling_method = 'monte_carlo'
        self.config.num_samples = config.get('num_samples', 10)
        self.config.loss_type = 'bayesian'
        self.config.kl_weight = config.get('kl_weight', 1e-4)
        
        # Decomposition
        self.decomp = series_decomp(config.moving_avg)
        
        # Create attention components
        self_attention = EnhancedAutoCorrelation(
            factor=config.factor, 
            attention_dropout=config.dropout, 
            output_attention=config.output_attention
        )
        cross_attention = EnhancedAutoCorrelation(
            factor=config.factor, 
            attention_dropout=config.dropout, 
            output_attention=False
        )
        
        # Encoder
        self.encoder = EnhancedEncoder(
            e_layers=config.e_layers,
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            activation=config.activation,
            attention_comp=self_attention,
            decomp_comp=self.decomp,
            norm_layer=nn.LayerNorm(config.d_model)
        )
        
        # Decoder
        self.decoder = EnhancedDecoder(
            d_layers=config.d_layers,
            d_model=config.d_model,
            c_out=config.c_out,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            activation=config.activation,
            self_attention_comp=self_attention,
            cross_attention_comp=cross_attention,
            decomp_comp=self.decomp,
            norm_layer=nn.LayerNorm(config.d_model),
            projection=nn.Linear(config.d_model, config.c_out, bias=True)
        )
        
        # Sampling
        self.sampling = MonteCarloSampling(num_samples=self.config.num_samples)
        
        # Output Head
        self.output_head = StandardOutputHead(config)
        
        # Loss Function
        self.loss_fn = BayesianLoss(kl_weight=self.config.kl_weight)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        predictions = []
        kls = []
        for _ in range(self.config.num_samples):
            enc_out = self.encoder(x_enc, attn_mask=enc_self_mask)
            dec_out = self.decoder(x_dec, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
            pred = self.output_head(dec_out)
            predictions.append(pred)
            # Compute KL divergence (simplified)
            kl = torch.mean(torch.abs(enc_out - dec_out))
            kls.append(kl)
        
        mean_pred = torch.mean(torch.stack(predictions), dim=0)
        std_pred = torch.std(torch.stack(predictions), dim=0)
        total_kl = torch.mean(torch.stack(kls))
        
        return mean_pred, std_pred, total_kl

    def compute_kl_divergence(self):
        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_divergence'):
                kl += module.kl_divergence()
        return kl

    def loss(self, pred, target, kl_term):
        return self.loss_fn(pred, target) + self.config.bayesian.kl_weight * kl_term