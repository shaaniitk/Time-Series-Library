"""
Hierarchical Enhanced Autoformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from argparse import Namespace

from layers.Embed import DataEmbedding_wo_pos
from utils.modular_components.implementations.decomposition_unified import get_decomposition_component
from utils.modular_components.implementations.encoder_unified import get_encoder_component
from utils.modular_components.implementations.decoders import get_decoder_component
from utils.modular_components.implementations.attentions_unified import get_attention_component
from utils.modular_components.implementations.fusion_migrated import get_fusion_component
from utils.modular_components.config_schemas import ComponentConfig
from utils.logger import logger

class HierarchicalEnhancedAutoformer(nn.Module):
    """
    Complete Hierarchical Enhanced Autoformer using modular components.
    """
    
    def __init__(self, configs, quantile_levels: Optional[List[float]] = None):
        super(HierarchicalEnhancedAutoformer, self).__init__()
        logger.info("Initializing HierarchicalEnhancedAutoformer with modular components")
        
        n_levels = getattr(configs, 'n_levels', 3)
        wavelet_type = getattr(configs, 'wavelet_type', 'db4')
        fusion_strategy = getattr(configs, 'fusion_strategy', 'weighted_concat')
        share_encoder_weights = getattr(configs, 'share_encoder_weights', False)
        use_cross_attention = getattr(configs, 'use_cross_attention', True)

        self.configs = configs
        self.n_levels = n_levels
        self.enc_in = configs.enc_in
        self.dec_in = configs.dec_in
        self.c_out = configs.c_out
        self.num_target_variables = getattr(configs, 'c_out_evaluation', configs.c_out)

        self.is_quantile_mode = False
        if hasattr(configs, 'quantile_levels') and isinstance(configs.quantile_levels, list) and len(configs.quantile_levels) > 0:
            self.is_quantile_mode = True
            self.quantiles = sorted(configs.quantile_levels)
            self.num_quantiles = len(self.quantiles)
        
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        
        decomp_params = {'seq_len': configs.seq_len, 'd_model': configs.d_model, 'wavelet_type': wavelet_type, 'levels': n_levels}
        self.decomposer = get_decomposition_component('wavelet_decomp', ComponentConfig(custom_params=decomp_params))
        
        encoder_params = {'e_layers': configs.e_layers, 'd_model': configs.d_model, 'n_heads': configs.n_heads, 'd_ff': configs.d_ff, 'dropout': configs.dropout, 'activation': configs.activation, 'attention_type': 'adaptive_autocorrelation', 'decomp_type': 'wavelet_decomp', 'decomp_params': decomp_params, 'n_levels': n_levels, 'share_weights': share_encoder_weights}
        self.encoder = get_encoder_component('hierarchical', ComponentConfig(custom_params=encoder_params))
        
        attention_config = ComponentConfig(custom_params={'d_model': configs.d_model, 'n_heads': configs.n_heads, 'dropout': configs.dropout, 'factor': configs.factor})
        self_attention = get_attention_component('adaptive_autocorrelation_layer', attention_config)
        cross_attention = get_attention_component('adaptive_autocorrelation_layer', attention_config)
        
        decoder_params = {'d_layers': configs.d_layers, 'd_model': configs.d_model, 'c_out': self.num_target_variables, 'n_heads': configs.n_heads, 'd_ff': configs.d_ff, 'dropout': configs.dropout, 'activation': configs.activation, 'self_attention_comp': self_attention, 'cross_attention_comp': cross_attention, 'decomp_comp': self.decomposer, 'projection': nn.Linear(configs.d_model, self.c_out, bias=True)}
        self.decoder = get_decoder_component('enhanced', **decoder_params)
        
        self.decomp = get_decomposition_component('learnable_decomp', ComponentConfig(custom_params={'input_dim': configs.dec_in}))
        
        if use_cross_attention:
            cross_attention_params = {'d_model': configs.d_model, 'n_levels': n_levels, 'use_multiwavelet': True}
            self.cross_attention = get_attention_component('cross_resolution', ComponentConfig(custom_params=cross_attention_params))
        else:
            self.cross_attention = None
        
        fusion_params = {'d_model': configs.d_model, 'n_levels': n_levels, 'fusion_strategy': fusion_strategy}
        self.fusion = get_fusion_component('hierarchical_fusion', **fusion_params)
        
        self.projection = nn.Linear(configs.c_out, self.c_out, bias=True)
        
    def _prepare_decoder_inputs(self, x_dec):
        seasonal_init_full, trend_init_full = self.decomp(x_dec)
        trend_init_for_decoder_accumulation = trend_init_full[:, :, :self.num_target_variables]
        mean_for_trend_extension = torch.mean(x_dec[:, :, :self.num_target_variables], dim=1).unsqueeze(1).repeat(1, self.configs.pred_len, 1)
        trend_arg_to_decoder = torch.cat([trend_init_for_decoder_accumulation[:, -self.configs.label_len:, :], mean_for_trend_extension], dim=1)
        zeros_for_seasonal_extension = torch.zeros([x_dec.shape[0], self.configs.pred_len, x_dec.shape[2]], device=x_dec.device)
        seasonal_arg_to_dec_embedding = torch.cat([seasonal_init_full[:, -self.configs.label_len:, :], zeros_for_seasonal_extension], dim=1)
        return trend_arg_to_decoder, seasonal_arg_to_dec_embedding

    def _encode_and_fuse(self, x_enc, x_mark_enc, dec_out, enc_self_mask):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        multi_res_enc_features = self.decomposer(enc_out)
        encoded_features = self.encoder(multi_res_enc_features, attn_mask=enc_self_mask)
        cross_attended_features = self.cross_attention(encoded_features) if self.cross_attention is not None else encoded_features
        fused_encoder_context = self.fusion(cross_attended_features, target_length=dec_out.size(1))
        return fused_encoder_context

    def _decode_and_combine(self, dec_out, fused_encoder_context, trend_arg_to_decoder, dec_self_mask, dec_enc_mask):
        fused_seasonal, fused_trend = self.decoder(dec_out, fused_encoder_context, x_mask=dec_self_mask, cross_mask=dec_enc_mask, trend=trend_arg_to_decoder)
        if self.is_quantile_mode and self.num_quantiles > 1:
            s_reshaped = fused_seasonal.view(fused_seasonal.size(0), fused_seasonal.size(1), self.num_target_variables, self.num_quantiles)
            t_expanded = fused_trend.unsqueeze(-1).expand_as(s_reshaped)
            dec_out_reshaped = t_expanded + s_reshaped
            dec_out = dec_out_reshaped.view(fused_seasonal.size(0), fused_seasonal.size(1), self.num_target_variables * self.num_quantiles)
        else:
            dec_out = fused_trend + fused_seasonal
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        trend_arg_to_decoder, seasonal_arg_to_dec_embedding = self._prepare_decoder_inputs(x_dec)
        dec_out_embedding = self.dec_embedding(seasonal_arg_to_dec_embedding, x_mark_dec)
        fused_encoder_context = self._encode_and_fuse(x_enc, x_mark_enc, dec_out_embedding, enc_self_mask)
        dec_out = self._decode_and_combine(dec_out_embedding, fused_encoder_context, trend_arg_to_decoder, dec_self_mask, dec_enc_mask)
        projected_output = self.projection(dec_out)
        output = projected_output[:, -self.configs.pred_len:, :]
        return output

Model = HierarchicalEnhancedAutoformer