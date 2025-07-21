import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_wo_pos
from utils.modular_components.implementations.attentions_unified import get_attention_component
from utils.modular_components.implementations.decomposition_unified import get_decomposition_component
from utils.modular_components.implementations.encoder_unified import get_encoder_component
from utils.modular_components.implementations.decoders import get_decoder_component
from utils.modular_components.config_schemas import ComponentConfig
from layers.Normalization import get_norm_layer
from utils.logger import logger
from typing import Optional, List

class EnhancedAutoformer(nn.Module):
    """
    Enhanced Autoformer with adaptive autocorrelation and learnable decomposition.
    """

    def __init__(self, configs, quantile_levels: Optional[List[float]] = None):
        super().__init__()
        logger.info(f"Initializing EnhancedAutoformer with configs: {configs}")

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.dec_in = configs.dec_in
        self.c_out = configs.c_out
        self.d_model = configs.d_model

        self.is_quantile_mode = False
        self.num_quantiles = 1
        self.num_target_variables = getattr(configs, 'c_out_evaluation', configs.c_out)

        if hasattr(configs, 'quantile_levels') and isinstance(configs.quantile_levels, list) and len(configs.quantile_levels) > 0:
            self.is_quantile_mode = True
            self.quantiles = sorted(configs.quantile_levels)
            self.num_quantiles = len(self.quantiles)
            logger.info(f"EnhancedAutoformer: Quantile mode ON. Quantiles: {self.quantiles}")

        # Decomposition
        decomp_config = ComponentConfig(component_name='learnable_decomp', custom_params={'input_dim': configs.dec_in})
        self.decomp = get_decomposition_component('learnable_decomp', decomp_config)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, 
                                                  configs.embed, configs.freq, configs.dropout)

        # Attention
        attention_config = ComponentConfig(
            component_name='adaptive_autocorrelation', 
            d_model=configs.d_model, 
            custom_params={'n_heads': configs.n_heads, 'factor': configs.factor, 'attention_dropout': configs.dropout}
        )
        
        # Encoder
        encoder_config = {
            'e_layers': configs.e_layers,
            'd_model': configs.d_model,
            'n_heads': configs.n_heads,
            'd_ff': configs.d_ff,
            'dropout': configs.dropout,
            'activation': configs.activation,
            'attention_comp': get_attention_component('adaptive_autocorrelation_layer', attention_config),
            'decomp_comp': self.decomp,
            'norm_layer': get_norm_layer(configs.norm_type, configs.d_model)
        }
        self.encoder = get_encoder_component('enhanced', **encoder_config)

        # Decoder
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, 
                                                      configs.embed, configs.freq, configs.dropout)
            
            decoder_config = {
                'd_layers': configs.d_layers,
                'd_model': self.d_model,
                'c_out': self.num_target_variables,
                'n_heads': configs.n_heads,
                'd_ff': configs.d_ff,
                'dropout': configs.dropout,
                'activation': configs.activation,
                'self_attention_comp': get_attention_component('adaptive_autocorrelation_layer', attention_config),
                'cross_attention_comp': get_attention_component('adaptive_autocorrelation_layer', attention_config),
                'decomp_comp': self.decomp,
                'norm_layer': get_norm_layer(configs.norm_type, configs.d_model),
                'projection': nn.Linear(self.d_model, self.c_out, bias=True)
            }
            self.decoder = get_decoder_component('enhanced', **decoder_config)

        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        seasonal_init_full, trend_init_full = self.decomp(x_dec)
        trend_init_for_decoder_accumulation = trend_init_full[:, :, :self.num_target_variables]
        mean_for_trend_extension = torch.mean(x_dec[:, :, :self.num_target_variables], dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        trend_arg_to_decoder = torch.cat([trend_init_for_decoder_accumulation[:, -self.label_len:, :], mean_for_trend_extension], dim=1)
        zeros_for_seasonal_extension = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_arg_to_dec_embedding = torch.cat([seasonal_init_full[:, -self.label_len:, :], zeros_for_seasonal_extension], dim=1)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(seasonal_arg_to_dec_embedding, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, 
                                                cross_mask=None, trend=trend_arg_to_decoder)

        if self.is_quantile_mode and self.num_quantiles > 1:
            trend_part_expanded = trend_part.unsqueeze(-1).repeat(1, 1, 1, self.num_quantiles)
            trend_part_expanded = trend_part_expanded.view(seasonal_part.shape)
            dec_out = trend_part_expanded + seasonal_part
        else:
            dec_out = trend_part + seasonal_part
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        seasonal_init, trend_init = self.decomp(x_enc)
        dec_out = self.dec_embedding(seasonal_init, x_mark_enc)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)
        dec_out = trend_part + seasonal_part
        return dec_out

    def anomaly_detection(self, x_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        seasonal_init, trend_init = self.decomp(x_enc)
        dec_out = self.dec_embedding(seasonal_init, None)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)
        dec_out = trend_part + seasonal_part
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        logger.debug("EnhancedAutoformer forward")
        
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            output = dec_out[:, -self.pred_len:, :]
            return output
        if self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        return None

Model = EnhancedAutoformer