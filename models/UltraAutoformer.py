import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_wo_pos
from utils.modular_components.implementations.attentions_unified import get_attention_component
from utils.modular_components.implementations.decomposition_unified import get_decomposition_component
from utils.modular_components.implementations.encoder_unified import get_encoder_component
from utils.modular_components.implementations.decoder_migrated import get_decoder_component
from utils.modular_components.config_schemas import ComponentConfig
from layers.Normalization import get_norm_layer
import math

class UltraAutoformer(nn.Module):
    """Ultra-enhanced Autoformer with advanced features"""
    
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        
        # Multi-scale decomposition
        decomp_config = ComponentConfig(component_name='adaptive', custom_params={'input_dim': configs.dec_in if hasattr(configs, 'dec_in') else configs.enc_in})
        self.decomp = get_decomposition_component('adaptive', decomp_config)
        
        # Enhanced embeddings with learnable scaling
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.embed_scale = nn.Parameter(torch.ones(1))
        
        # Ultra encoder layers
        attention_config = ComponentConfig(component_name='dynamic', d_model=configs.d_model, custom_params={'n_heads': configs.n_heads})
        encoder_config = {
            'd_model': configs.d_model,
            'n_heads': configs.n_heads,
            'd_ff': configs.d_ff,
            'dropout': configs.dropout,
            'attention_comp': get_attention_component('dynamic', attention_config),
            'decomp_comp': self.decomp
        }
        self.encoder_layers = nn.ModuleList([
            get_encoder_component('ultra', **encoder_config)
            for _ in range(configs.e_layers)
        ])
        
        self.encoder_norm = get_norm_layer(configs.norm_type, configs.d_model)
        
        # Task-specific heads
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
            
            # Ultra decoder with cross-attention
            self.decoder_layers = nn.ModuleList([
                get_encoder_component('ultra', **encoder_config)
                for _ in range(configs.d_layers)
            ])
            
            # Multi-head prediction
            self.prediction_heads = nn.ModuleList([
                nn.Linear(configs.d_model, configs.c_out)
                for _ in range(3)  # Ensemble of 3 heads
            ])
            
            self.head_weights = nn.Parameter(torch.ones(3) / 3)
            
        # Adaptive learning rate scaling
        self.lr_scale = nn.Parameter(torch.ones(1))
        
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        seasonal_init, trend_init = self.decomp(x_dec)
        
        enc_out = self.enc_embedding(x_enc, x_mark_enc) * self.embed_scale
        
        for layer in self.encoder_layers:
            enc_out, _ = layer(enc_out)
        
        enc_out = self.encoder_norm(enc_out)
        
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        
        for layer in self.decoder_layers:
            dec_out, _ = layer(dec_out)
        
        predictions = []
        weights = F.softmax(self.head_weights, dim=0)
        
        for i, head in enumerate(self.prediction_heads):
            pred = head(dec_out)
            predictions.append(pred * weights[i])
        
        seasonal_part = sum(predictions)
        
        trend_part = trend_init[:, -self.pred_len:, :] if trend_init.size(1) >= self.pred_len else trend_init
        
        return seasonal_part + trend_part
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        
        enc_out = self.enc_embedding(x_enc, x_mark_enc) * self.embed_scale
        
        for layer in self.encoder_layers:
            enc_out, _ = layer(enc_out)
        
        enc_out = self.encoder_norm(enc_out)
        
        if self.task_name == 'classification':
            pooled = enc_out.mean(dim=1)
            return F.linear(pooled, self.prediction_heads[0].weight[:self.configs.num_class])
        
        return enc_out

Model = UltraAutoformer
