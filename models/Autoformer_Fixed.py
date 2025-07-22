import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_wo_pos
from utils.modular_components.implementations.decomposition_unified import get_decomposition_component
from utils.modular_components.implementations.encoder_unified import get_encoder_component
from utils.modular_components.implementations.decoder_migrated import get_decoder_component
from utils.modular_components.implementations.attentions_unified import get_attention_component
from utils.modular_components.config_schemas import ComponentConfig
from layers.Normalization import get_norm_layer
from utils.logger import logger

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self._validate_configs(configs)
        
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        
        kernel_size = getattr(configs, 'moving_avg', 25)
        decomp_config = ComponentConfig(component_name='stable_decomp', custom_params={'kernel_size': kernel_size})
        self.decomp = get_decomposition_component('stable_decomp', decomp_config)

        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        
        self.gradient_scale = nn.Parameter(torch.ones(1) * 0.1)
        
        attention_config = ComponentConfig(
            component_name='autocorrelation', 
            d_model=configs.d_model, 
            custom_params={'n_heads': configs.n_heads, 'factor': configs.factor, 'attention_dropout': configs.dropout}
        )
        autocorrelation = get_attention_component('autocorrelation_layer', attention_config)

        encoder_config = {
            'e_layers': configs.e_layers,
            'd_model': configs.d_model,
            'n_heads': configs.n_heads,
            'd_ff': configs.d_ff,
            'dropout': configs.dropout,
            'activation': configs.activation,
            'attention_comp': autocorrelation,
            'decomp_comp': self.decomp,
            'norm_layer': get_norm_layer(configs.norm_type, configs.d_model)
        }
        self.encoder = get_encoder_component('standard', **encoder_config)

        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
            
            decoder_config = {
                'd_layers': configs.d_layers,
                'd_model': configs.d_model,
                'c_out': configs.c_out,
                'n_heads': configs.n_heads,
                'd_ff': configs.d_ff,
                'dropout': configs.dropout,
                'activation': configs.activation,
                'self_attention_comp': get_attention_component('autocorrelation_layer', attention_config),
                'cross_attention_comp': get_attention_component('autocorrelation_layer', attention_config),
                'decomp_comp': self.decomp,
                'norm_layer': get_norm_layer(configs.norm_type, configs.d_model),
                'projection': nn.Linear(configs.d_model, configs.c_out, bias=True)
            }
            self.decoder = get_decoder_component('standard', **decoder_config)
        
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def _validate_configs(self, configs):
        required = ['task_name', 'seq_len', 'pred_len', 'enc_in', 'd_model']
        for attr in required:
            if not hasattr(configs, attr):
                raise ValueError(f"Missing config: {attr}")

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        mean = torch.mean(x_enc, dim=1, keepdim=True).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        
        seasonal_init, trend_init = self.decomp(x_enc)
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        
        enc_out = enc_out * self.gradient_scale

        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)

        return trend_part + seasonal_part

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if hasattr(self, 'enc_in') and x_enc.size(-1) != self.enc_in:
            logger.warning(f"Input dimension mismatch detected")
        
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        
        if self.task_name == 'imputation':
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
            enc_out, _ = self.encoder(enc_out, attn_mask=None)
            return self.projection(enc_out)
        
        if self.task_name == 'anomaly_detection':
            enc_out = self.enc_embedding(x_enc, None)
            enc_out, _ = self.encoder(enc_out, attn_mask=None)
            return self.projection(enc_out)
        
        if self.task_name == 'classification':
            enc_out = self.enc_embedding(x_enc, None)
            enc_out, _ = self.encoder(enc_out, attn_mask=None)
            output = self.act(enc_out)
            output = self.dropout(output)
            output = output * x_mark_enc.unsqueeze(-1)
            output = output.reshape(output.shape[0], -1)
            return self.projection(output)
        
        return None
