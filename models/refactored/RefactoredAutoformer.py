import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modular_autoformer import ModularAutoformer, ModularAutoformerConfig
from models.Autoformer import LearnableSeriesDecomp
from layers.modular.encoder import EnhancedEncoder
from layers.modular.decoder import EnhancedDecoder
from layers.modular.attention import EnhancedAutoCorrelation
from layers.modular.output_heads import StandardOutputHead

class RefactoredAutoformer(ModularAutoformer):
    def __init__(self, config):
        super().__init__(config)
        # Use ModularAutoformerConfig to set up components
        self.config = ModularAutoformerConfig.from_legacy(config)
        
        # Decomposition
        self.decomp = LearnableSeriesDecomp(moving_avg=config.moving_avg)
        
        # Encoder
        self.encoder = EnhancedEncoder(
            [
                EnhancedEncoderLayer(
                    EnhancedAutoCorrelation(
                        factor=config.factor, attention_dropout=config.dropout, output_attention=config.output_attention
                    ),
                    config.d_model,
                    n_heads=config.n_heads,
                    dim_feedforward=config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation
                ) for _ in range(config.e_layers)
            ],
            norm_layer=nn.LayerNorm(config.d_model)
        )
        
        # Decoder
        self.decoder = EnhancedDecoder(
            [
                EnhancedDecoderLayer(
                    EnhancedAutoCorrelation(
                        factor=config.factor, attention_dropout=config.dropout, output_attention=False
                    ),
                    EnhancedAutoCorrelation(
                        factor=config.factor, attention_dropout=config.dropout, output_attention=False
                    ),
                    d_model=config.d_model,
                    c_out=config.c_out,
                    d_ff=config.d_ff,
                    moving_avg=config.moving_avg,
                    dropout=config.dropout,
                    activation=config.activation
                ) for _ in range(config.d_layers)
            ],
            norm_layer=nn.LayerNorm(config.d_model),
            projection=nn.Linear(config.d_model, config.c_out, bias=True)
        )
        
        # Output Head
        self.output_head = StandardOutputHead(config)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # Follow the forward logic from original, but using modular components
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.config.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.config.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_dec)
        trend_init = torch.cat([trend_init[:, -self.config.label_len:, :], zeros], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.config.label_len:, :], zeros], dim=1)
        
        enc_out = self.encoder(x_enc, attn_mask=enc_self_mask)
        dec_out = self.decoder(seasonal_init, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, trend=trend_init)
        
        if self.config.output_attention:
            return dec_out[:, -self.config.pred_len:, :], attns
        else:
            return dec_out[:, -self.config.pred_len:, :]

    # Add other methods as needed, like predict, etc., adapted similarly