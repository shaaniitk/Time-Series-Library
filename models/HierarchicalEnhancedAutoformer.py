"""
Hierarchical Enhanced Autoformer

This module implements a hierarchical architecture that leverages existing
wavelet infrastructure (MultiWaveletCorrelation.py and DWT_Decomposition.py)
to process time series at multiple resolutions simultaneously.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple

from models.EnhancedAutoformer import EnhancedAutoformer, LearnableSeriesDecomp
from layers.MultiWaveletCorrelation import MultiWaveletTransform, MultiWaveletCross
from layers.DWT_Decomposition import DWT1DForward, DWT1DInverse
from layers.Embed import DataEmbedding_wo_pos
from utils.logger import logger


class WaveletHierarchicalDecomposer(nn.Module):
    """
    Hierarchical decomposition using existing DWT infrastructure.
    
    Integrates with existing DWT_Decomposition.py to create multi-resolution
    time series representations.
    """
    
    def __init__(self, seq_len, d_model, wavelet_type='db4', levels=3, 
                 use_learnable_weights=True):
        super(WaveletHierarchicalDecomposer, self).__init__()
        logger.info(f"Initializing WaveletHierarchicalDecomposer with {levels} levels")
        
        self.seq_len = seq_len
        self.d_model = d_model
        self.levels = levels
        self.wavelet_type = wavelet_type
        
        # Use existing DWT components
        self.dwt_forward = DWT1DForward(J=levels, wave=wavelet_type, mode='symmetric')
        self.dwt_inverse = DWT1DInverse(wave=wavelet_type, mode='symmetric')
        
        # Learnable scale weights for each decomposition level
        if use_learnable_weights:
            self.scale_weights = nn.Parameter(torch.ones(levels + 1) / (levels + 1))
        else:
            self.register_buffer('scale_weights', torch.ones(levels + 1) / (levels + 1))
        
        # Projection layers to maintain d_model dimension across scales
        self.scale_projections = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(levels + 1)
        ])
        
    def forward(self, x):
        """
        Decompose input into multiple resolution levels.
        
        Args:
            x: [batch_size, seq_len, d_model] input tensor
            
        Returns:
            List of tensors at different resolution levels
        """
        batch_size, seq_len, d_model = x.shape
        
        # Check for DWT compatibility before attempting the transform
        if seq_len < 2 ** self.levels:
            logger.warning(
                f"Sequence length {seq_len} is too short for {self.levels} DWT levels. "
                f"Required minimum length is {2 ** self.levels}. Using fallback decomposition."
            )
            return self._fallback_decomposition(x)

        # Prepare input for DWT (needs [B, C, L] format)
        x_dwt = x.transpose(1, 2)  # [B, d_model, seq_len]
        
        # Apply DWT decomposition
        try:
            low_freq, high_freqs = self.dwt_forward(x_dwt)
        except (ValueError, RuntimeError) as e:
            logger.error(f"DWT failed with a critical error: {e}", exc_info=True)
            return self._fallback_decomposition(x)
        
        # Process decomposed components
        decomposed_scales = []
        
        # Process low frequency component (coarsest scale)
        low_freq = low_freq.transpose(1, 2)  # Back to [B, L, C]
        low_freq = self._resize_to_target_length(low_freq, seq_len // (2 ** self.levels))
        low_freq = self.scale_projections[0](low_freq)
        decomposed_scales.append(low_freq)
        
        # Process high frequency components (finest to coarsest)
        for i, high_freq in enumerate(high_freqs):
            high_freq = high_freq.transpose(1, 2)  # Back to [B, L, C]
            target_length = seq_len // (2 ** (self.levels - i))
            high_freq = self._resize_to_target_length(high_freq, target_length)
            high_freq = self.scale_projections[i + 1](high_freq)
            decomposed_scales.append(high_freq)
        
        return decomposed_scales
    
    def _resize_to_target_length(self, x, target_length):
        """Resize tensor to target sequence length"""
        current_length = x.size(1)
        if current_length == target_length:
            return x
        elif current_length < target_length:
            # Upsample using interpolation
            x_resized = F.interpolate(
                x.transpose(1, 2), 
                size=target_length, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
        else:
            # Downsample using adaptive pooling
            x_resized = F.adaptive_avg_pool1d(
                x.transpose(1, 2), 
                target_length
            ).transpose(1, 2)
        
        return x_resized
    
    def _fallback_decomposition(self, x):
        """Fallback decomposition using simple pooling"""
        logger.info("Using fallback multi-scale decomposition")
        scales = []
        current = x
        
        for i in range(self.levels + 1):
            if i == 0:
                scales.append(current)
            else:
                # Simple downsampling
                pooled = F.avg_pool1d(
                    current.transpose(1, 2), 
                    kernel_size=2, 
                    stride=2
                ).transpose(1, 2)
                scales.append(pooled)
                current = pooled
        
        return scales


class CrossResolutionAttention(nn.Module):
    """
    Cross-resolution attention using existing MultiWaveletCross components.
    """
    
    def __init__(self, d_model, n_levels, use_multiwavelet=True, n_heads=8):
        super(CrossResolutionAttention, self).__init__()
        logger.info(f"Initializing CrossResolutionAttention for {n_levels} levels")
        
        self.n_levels = n_levels
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = d_model // n_heads
        self.use_multiwavelet = use_multiwavelet
        
        if self.use_multiwavelet:
            # Use existing MultiWaveletCross for cross-resolution attention with proper configuration
            self.cross_res_attention = nn.ModuleList([
                MultiWaveletCross(
                    in_channels=self.head_dim,  # Dimension per head (e.g., 64)
                    out_channels=self.head_dim, # Dimension per head (e.g., 64) 
                    modes=min(32, self.head_dim // 2), # seq_len_q and seq_len_kv will be set dynamically
                    c=64,
                    k=8,
                    ich=d_model,  # Total dimension (e.g., 512)
                    base='legendre',
                    activation='tanh'  # Use supported activation
                ) for _ in range(n_levels - 1)
            ])
        else:
            # Use standard multi-head attention
            self.cross_res_attention = nn.ModuleList([
                nn.MultiheadAttention(d_model, n_heads, batch_first=True)
                for _ in range(n_levels - 1)
            ])
        
        # Cross-resolution projection layers
        self.cross_projections = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_levels - 1)
        ])
    
    def forward(self, multi_res_features):
        """
        Apply cross-resolution attention between different scales.
        
        Args:
            multi_res_features: List of tensors at different resolutions
            
        Returns:
            List of cross-attended features
        """
        if len(multi_res_features) < 2:
            return multi_res_features
        
        attended_features = []
        
        for i in range(len(multi_res_features) - 1):
            coarse_features = multi_res_features[i]     # Lower resolution
            fine_features = multi_res_features[i + 1]   # Higher resolution
            
            # Align feature dimensions
            aligned_fine = self._align_features(fine_features, coarse_features)
            
            if self.use_multiwavelet:
                # Use existing MultiWaveletCross
                try:
                    cross_attended = self._apply_multiwavelet_cross(
                        coarse_features, aligned_fine, i
                    )
                except (AttributeError, RuntimeError) as e:
                    logger.error(f"MultiWavelet cross-attention failed with a critical error: {e}", exc_info=True)
                    cross_attended = self._apply_standard_attention(
                        coarse_features, aligned_fine, i
                    )
            else:
                cross_attended = self._apply_standard_attention(
                    coarse_features, aligned_fine, i
                )
            
            # Apply cross-resolution projection
            cross_attended = self.cross_projections[i](cross_attended)
            attended_features.append(cross_attended)
        
        # Add the finest scale without cross-attention
        attended_features.append(multi_res_features[-1])
        
        return attended_features
    
    def _align_features(self, source, target):
        """Align source features to target resolution"""
        if source.size(1) == target.size(1):
            return source
        
        # Resize to match target sequence length
        source_aligned = F.interpolate(
            source.transpose(1, 2),
            size=target.size(1),
            mode='linear',
            align_corners=False
        ).transpose(1, 2)
        
        return source_aligned
    
    def _apply_multiwavelet_cross(self, query_features, key_value_features, layer_idx):
        """Apply MultiWaveletCross attention with proper 4D format."""
        batch_size, seq_len, d_model = query_features.shape
        
        # Reshape to 4D format: (B, N, H, E)
        query_4d = query_features.view(batch_size, seq_len, self.n_heads, self.head_dim)
        key_4d = key_value_features.view(batch_size, key_value_features.size(1), self.n_heads, self.head_dim)
        value_4d = key_4d  # Use same as key for cross-attention style
        
        # Apply MultiWaveletCross
        # BUG FIX: Was calling an undefined function 'attention_layer'.
        # Corrected to call the appropriate module from the list.
        attention_layer = self.cross_res_attention[layer_idx]
        attended, _ = attention_layer(
            query_4d, key_4d, value_4d, query_features.size(1), key_value_features.size(1)
        )
        
        # Reshape back to 3D: (B, N, D)
        attended = attended.view(batch_size, seq_len, d_model)
        
        return attended
    
    def _apply_standard_attention(self, query_features, key_value_features, layer_idx):
        """Apply standard multi-head attention"""
        attended, _ = self.cross_res_attention[layer_idx](
            query_features,
            key_value_features,
            key_value_features
        )
        return attended


class HierarchicalEncoder(nn.Module):
    """
    Hierarchical encoder that processes multiple resolutions with Enhanced Autoformer layers.
    """
    
    def __init__(self, configs, n_levels=3, use_enhanced_layers=True, share_weights=False, _attention_factory=None, decomp_params=None):
        super(HierarchicalEncoder, self).__init__()
        logger.info(f"Initializing HierarchicalEncoder with {n_levels} levels")
        
        self.n_levels = n_levels
        self.d_model = configs.d_model
        self.share_weights = share_weights
        
        if decomp_params:
            self.decomposer = WaveletHierarchicalDecomposer(**decomp_params)
        else:
            self.decomposer = None
        
        def create_single_encoder():
            from models.EnhancedAutoformer import EnhancedEncoder, EnhancedEncoderLayer
            from layers.Normalization import get_norm_layer

            # Use the factory if provided, otherwise use the legacy method
            if _attention_factory:
                autocorr_layer = _attention_factory()
            else:
                from layers.Attention import get_attention_layer
                autocorr_layer = get_attention_layer(configs)

            enhanced_layer = EnhancedEncoderLayer(
                autocorr_layer, configs.d_model, configs.d_ff,
                dropout=configs.dropout, activation=configs.activation
            )
            return EnhancedEncoder([enhanced_layer], norm_layer=get_norm_layer('LayerNorm', configs.d_model))

        if self.share_weights:
            logger.info(f"Using one shared encoder for all {n_levels} levels.")
            shared_encoder = create_single_encoder()
            self.resolution_encoders = nn.ModuleList([shared_encoder] * n_levels)
        else:
            logger.info(f"Initialized {n_levels} separate encoders.")
            self.resolution_encoders = nn.ModuleList([create_single_encoder() for _ in range(n_levels)])
        
        self.use_multiwavelet = True
        if self.use_multiwavelet:
            try:
                self.multiwavelet_transforms = nn.ModuleList([
                    MultiWaveletTransform(
                        ich=configs.d_model,
                        k=8,
                        alpha=16,
                        c=min(128, configs.d_model),
                        nCZ=1,
                        L=0,
                        base='legendre'
                    ) for _ in range(n_levels)
                ])
            except ImportError as e:
                logger.warning(f"Failed to import MultiWaveletTransform: {e}")
                self.use_multiwavelet = False
    
    def forward(self, multi_res_features, attn_mask=None):
        """
        Process multiple resolutions through hierarchical encoding.
        
        Args:
            multi_res_features: List of features at different resolutions
            attn_mask: Attention mask (optional)
            
        Returns:
            List of encoded features at different resolutions
        """
        encoded_features = []
        
        for i, (features, encoder) in enumerate(zip(multi_res_features, self.resolution_encoders)):
            # Optional: Apply multi-wavelet transform
            if self.use_multiwavelet and hasattr(self, 'multiwavelet_transforms'):
                try:
                    # Adapt format for MultiWaveletTransform
                    B, L, D = features.shape
                    features_adapted = features.view(B, L, 1, D)
                    features_wavelet, _ = self.multiwavelet_transforms[i](
                        features_adapted, features_adapted, features_adapted, attn_mask
                    )
                    features = features_wavelet.squeeze(2)
                except (AttributeError, RuntimeError) as e:
                    logger.error(f"MultiWavelet transform failed for level {i} with a critical error: {e}", exc_info=True)
            
            # Apply resolution-specific encoder
            if hasattr(encoder, 'forward'):
                encoded = encoder(features, attn_mask=attn_mask)
                # EnhancedEncoder returns (output, attentions), we only need output
                if isinstance(encoded, tuple):
                    encoded = encoded[0]
            else:
                # Standard transformer encoder
                encoded = encoder(features, src_key_padding_mask=attn_mask)
            
            encoded_features.append(encoded)
        
        return encoded_features


class HierarchicalFusion(nn.Module):
    """
    Fuse multi-resolution features into final output.
    """
    
    def __init__(self, d_model, n_levels, fusion_strategy='weighted_concat'):
        super(HierarchicalFusion, self).__init__()
        logger.info(f"Initializing HierarchicalFusion with strategy: {fusion_strategy}")
        
        self.fusion_strategy = fusion_strategy
        self.n_levels = n_levels
        self.d_model = d_model
        
        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(n_levels) / n_levels)
        
        if fusion_strategy == 'weighted_concat':
            # Concatenate all levels and project down
            self.fusion_projection = nn.Sequential(
                nn.Linear(d_model * n_levels, d_model * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model * 2, d_model)
            )
        elif fusion_strategy == 'weighted_sum':
            # Simple weighted sum
            self.fusion_projection = nn.Identity()
        elif fusion_strategy == 'attention_fusion':
            # Attention-based fusion
            self.attention_fusion = nn.MultiheadAttention(
                d_model, num_heads=8, batch_first=True
            )
            self.fusion_projection = nn.Linear(d_model, d_model)
    
    def forward(self, multi_res_features, target_length=None):
        """
        Fuse multi-resolution features.
        
        Args:
            multi_res_features: List of features at different resolutions
            target_length: Target sequence length for output
            
        Returns:
            Fused feature tensor
        """
        if len(multi_res_features) == 1:
            return multi_res_features[0]
        
        # Align all features to the same sequence length
        if target_length is None:
            target_length = max(feat.size(1) for feat in multi_res_features)
        
        aligned_features = []
        for i, features in enumerate(multi_res_features):
            if features.size(1) != target_length:
                features = F.interpolate(
                    features.transpose(1, 2),
                    size=target_length,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            aligned_features.append(features)
        
        # Apply fusion strategy
        if self.fusion_strategy == 'weighted_concat':
            # Apply learnable weights to each aligned feature before concatenation
            weights = F.softmax(self.fusion_weights, dim=0)
            weighted_aligned_features = [
                features * weights[i] for i, features in enumerate(aligned_features)
            ]
            # Concatenate and project
            concatenated = torch.cat(weighted_aligned_features, dim=-1)
            fused = self.fusion_projection(concatenated)
            
        elif self.fusion_strategy == 'weighted_sum':
            # Weighted sum of features
            fused = torch.zeros_like(aligned_features[0])
            weights = F.softmax(self.fusion_weights, dim=0)
            for i, features in enumerate(aligned_features):
                fused += weights[i] * features

        elif self.fusion_strategy == 'attention_fusion':
            # Use attention to fuse features
            stacked_features = torch.stack(aligned_features, dim=2)  # [B, L, n_levels, D]
            B, L, n_levels, D = stacked_features.shape
            
            # Reshape for attention
            query = aligned_features[-1]  # Use finest scale as query
            key_value = stacked_features.view(B, L * n_levels, D)
            
            fused, _ = self.attention_fusion(query, key_value, key_value)
            fused = self.fusion_projection(fused)
        
        return fused


class HierarchicalEnhancedAutoformer(nn.Module):
    """
    Complete Hierarchical Enhanced Autoformer.

    This version refactors the decoder to be more efficient. Instead of a full
    hierarchical decoder, it fuses the multi-scale encoder outputs and uses a
    single, standard EnhancedDecoder.
    """
    
    def __init__(self, configs, quantile_levels: Optional[List[float]] = None):
        super(HierarchicalEnhancedAutoformer, self).__init__()
        logger.info("Initializing HierarchicalEnhancedAutoformer")
        
        # Read hierarchical parameters from configs object with defaults
        n_levels = getattr(configs, 'n_levels', 3)
        wavelet_type = getattr(configs, 'wavelet_type', 'db4')
        fusion_strategy = getattr(configs, 'fusion_strategy', 'weighted_concat')
        share_encoder_weights = getattr(configs, 'share_encoder_weights', False)
        use_cross_attention = getattr(configs, 'use_cross_attention', True)

        self.configs = configs
        self.n_levels = n_levels

        # Store model dimensions from configs (set by DimensionManager)
        self.enc_in = configs.enc_in
        self.dec_in = configs.dec_in
        self.c_out = configs.c_out # This is c_out_model from DimensionManager

        self.is_quantile_mode = False
        self.num_quantiles = 1
        # num_target_variables is the number of *base* targets (c_out_evaluation from DimensionManager)
        self.num_target_variables = getattr(configs, 'c_out_evaluation', configs.c_out) # Use c_out_evaluation if available

        if hasattr(configs, 'quantile_levels') and isinstance(configs.quantile_levels, list) and len(configs.quantile_levels) > 0:
            self.is_quantile_mode = True
            self.quantiles = sorted(configs.quantile_levels)
            self.num_quantiles = len(self.quantiles)
        
        # Input embedding
        self.enc_embedding = DataEmbedding_wo_pos(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        self.dec_embedding = DataEmbedding_wo_pos(
            configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        
        # Hierarchical decomposition
        self.decomposer = WaveletHierarchicalDecomposer(
            seq_len=configs.seq_len,
            d_model=configs.d_model,
            wavelet_type=wavelet_type,
            levels=n_levels
        )
        
        # Hierarchical encoder
        self.encoder = HierarchicalEncoder(configs, n_levels=n_levels, share_weights=share_encoder_weights)
        
        # --- DECODER REFACTOR ---
        # Replace HierarchicalDecoder with a single, more efficient EnhancedDecoder.
        from models.EnhancedAutoformer import EnhancedDecoder, EnhancedDecoderLayer
        from layers.EnhancedAutoCorrelation import AdaptiveAutoCorrelationLayer, AdaptiveAutoCorrelation

        decoder_layers = []
        for _ in range(configs.d_layers):
            # Self-attention in decoder
            self_adaptive_autocorr = AdaptiveAutoCorrelation(
                True, configs.factor,
                attention_dropout=configs.dropout,
                output_attention=False,
                adaptive_k=True,
                multi_scale=True,
                scales=[1, 2]
            )
            self_autocorr_layer = AdaptiveAutoCorrelationLayer(self_adaptive_autocorr, configs.d_model, configs.n_heads)
            
            # Cross-attention with encoder
            cross_adaptive_autocorr = AdaptiveAutoCorrelation(
                False, configs.factor,
                attention_dropout=configs.dropout,
                output_attention=False,
                adaptive_k=True,
                multi_scale=False
            )
            cross_autocorr_layer = AdaptiveAutoCorrelationLayer(cross_adaptive_autocorr, configs.d_model, configs.n_heads)
            
            decoder_layers.append(EnhancedDecoderLayer(
                self_autocorr_layer,
                cross_autocorr_layer,
                configs.d_model,
                self.num_target_variables, # Trend part dim
                configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation,
            ))

        self.decoder = EnhancedDecoder(decoder_layers, norm_layer=torch.nn.LayerNorm(configs.d_model),
                                       projection=nn.Linear(configs.d_model, self.c_out, bias=True)) # Seasonal part dim
        
        # Hierarchical decomposition for trend/seasonal initialization
        self.decomp = LearnableSeriesDecomp(input_dim=configs.dec_in, init_kernel_size=25)
        
        # Cross-resolution attention
        if use_cross_attention:
            self.cross_attention = CrossResolutionAttention(
                d_model=configs.d_model,
                n_levels=n_levels,
                use_multiwavelet=True
            )
        else:
            self.cross_attention = None
        
        # Hierarchical fusion
        self.fusion = HierarchicalFusion(
            d_model=configs.d_model,
            n_levels=n_levels,
            fusion_strategy=fusion_strategy
        )
        
        # Output projection
        # The input to this projection is dec_out, which is fused_trend + fused_seasonal.
        # Both fused_trend and fused_seasonal have configs.c_out features after HierarchicalFusion.
        # Therefore, the in_features for this projection should be configs.c_out.
        self.projection = nn.Linear(configs.c_out, 
                                     self.c_out, bias=True) # Project to c_out_model
        
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

        if self.cross_attention is not None:
            cross_attended_features = self.cross_attention(encoded_features)
        else:
            cross_attended_features = encoded_features

        fused_encoder_context = self.fusion(cross_attended_features, target_length=dec_out.size(1))
        return fused_encoder_context

    def _decode_and_combine(self, dec_out, fused_encoder_context, trend_arg_to_decoder, dec_self_mask, dec_enc_mask):
        fused_seasonal, fused_trend = self.decoder(
            dec_out,
            fused_encoder_context,
            x_mask=dec_self_mask,
            cross_mask=dec_enc_mask,
            trend=trend_arg_to_decoder
        )

        if self.is_quantile_mode and self.num_quantiles > 1:
            s_reshaped = fused_seasonal.view(
                fused_seasonal.size(0), fused_seasonal.size(1),
                self.num_target_variables, self.num_quantiles
            )
            t_expanded = fused_trend.unsqueeze(-1).expand_as(s_reshaped)
            dec_out_reshaped = t_expanded + s_reshaped
            dec_out = dec_out_reshaped.view(
                fused_seasonal.size(0), fused_seasonal.size(1),
                self.num_target_variables * self.num_quantiles
            )
        else:
            dec_out = fused_trend + fused_seasonal

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        Hierarchical Enhanced Autoformer forward pass with a simplified, efficient decoder.
        """
        trend_arg_to_decoder, seasonal_arg_to_dec_embedding = self._prepare_decoder_inputs(x_dec)

        dec_out_embedding = self.dec_embedding(seasonal_arg_to_dec_embedding, x_mark_dec)

        fused_encoder_context = self._encode_and_fuse(x_enc, x_mark_enc, dec_out_embedding, enc_self_mask)

        dec_out = self._decode_and_combine(dec_out_embedding, fused_encoder_context, trend_arg_to_decoder, dec_self_mask, dec_enc_mask)
        
        projected_output = self.projection(dec_out)
        output = projected_output[:, -self.configs.pred_len:, :]
        return output


    def get_hierarchy_info(self):
        """Get information about the hierarchical structure"""
        return {
            'n_levels': self.n_levels,
            'wavelet_type': self.decomposer.wavelet_type,
            'fusion_strategy': self.fusion.fusion_strategy,
            'uses_cross_attention': self.cross_attention is not None,
            'uses_multiwavelet': getattr(self.cross_attention, 'use_multiwavelet', False)
        }


# Factory function for easy creation
def create_hierarchical_autoformer(configs, **kwargs):
    """
    Factory function to create Hierarchical Enhanced Autoformer.
    
    Args:
        configs: Model configuration
        **kwargs: Additional arguments (n_levels, wavelet_type, etc.)
        
    Returns:
        Configured Hierarchical Enhanced Autoformer
    """
    logger.info("Creating Hierarchical Enhanced Autoformer")
    
    # Set defaults
    defaults = {
        'n_levels': 3,
        'wavelet_type': 'db4',
        'fusion_strategy': 'weighted_concat',
        'use_cross_attention': True
    }
    defaults.update(kwargs)
    
    return HierarchicalEnhancedAutoformer(configs, **defaults)


# Example usage and testing
if __name__ == "__main__":
    from argparse import Namespace
    
    # Mock configuration
    configs = Namespace(
        seq_len=96,
        label_len=48,
        pred_len=24,
        enc_in=7,
        dec_in=7,
        c_out=7,
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=2048,
        factor=1,
        dropout=0.1,
        embed='timeF',
        freq='h',
        activation='gelu'
    )
    
    logger.info("Testing HierarchicalEnhancedAutoformer...")
    
    # Create model
    model = HierarchicalEnhancedAutoformer(
        configs,
        n_levels=3,
        wavelet_type='db4',
        fusion_strategy='weighted_concat'
    )
    
    # Test data
    batch_size = 4
    x_enc = torch.randn(batch_size, configs.seq_len, configs.enc_in)
    x_mark_enc = torch.randn(batch_size, configs.seq_len, 4)
    x_dec = torch.randn(batch_size, configs.label_len + configs.pred_len, configs.dec_in)
    x_mark_dec = torch.randn(batch_size, configs.label_len + configs.pred_len, 4)
    
    # Forward pass
    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    
    print(f"Input shape: {x_enc.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model hierarchy info: {model.get_hierarchy_info()}")
    
    logger.info("HierarchicalEnhancedAutoformer test completed!")


# Alias for consistency with other model files and test scripts
Model = HierarchicalEnhancedAutoformer
