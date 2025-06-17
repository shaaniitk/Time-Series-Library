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
        
        # Prepare input for DWT (needs [B, C, L] format)
        x_dwt = x.transpose(1, 2)  # [B, d_model, seq_len]
        
        # Apply DWT decomposition
        try:
            low_freq, high_freqs = self.dwt_forward(x_dwt)
        except Exception as e:
            logger.warning(f"DWT failed, using fallback decomposition: {e}")
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
        self.use_multiwavelet = use_multiwavelet
        
        if use_multiwavelet:
            # Use existing MultiWaveletCross for cross-resolution attention
            self.cross_res_attention = nn.ModuleList([
                MultiWaveletCross(
                    in_channels=d_model,
                    out_channels=d_model,
                    seq_len_q=96,  # Will be adapted dynamically
                    seq_len_kv=96,
                    modes=min(32, d_model // 2),
                    c=64,
                    k=8,
                    base='legendre',
                    activation='relu'
                ) for _ in range(n_levels - 1)
            ])
        else:
            # Fallback to standard multi-head attention
            self.cross_res_attention = nn.ModuleList([
                nn.MultiheadAttention(d_model, n_heads, batch_first=True)
                for _ in range(n_levels - 1)
            ])
        
        # Learnable fusion weights for cross-resolution features
        self.fusion_weights = nn.Parameter(torch.ones(n_levels) / n_levels)
        
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
                except Exception as e:
                    logger.warning(f"MultiWavelet cross-attention failed, using fallback: {e}")
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
        """Apply MultiWaveletCross attention"""
        # Adapt input format for MultiWaveletCross
        batch_size, seq_len, d_model = query_features.shape
        
        # MultiWaveletCross expects specific input format
        attended = self.cross_res_attention[layer_idx](
            query_features,
            key_value_features
        )
        
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
    
    def __init__(self, configs, n_levels=3, use_enhanced_layers=True):
        super(HierarchicalEncoder, self).__init__()
        logger.info(f"Initializing HierarchicalEncoder with {n_levels} levels")
        
        self.n_levels = n_levels
        self.d_model = configs.d_model
        
        # Multi-resolution Enhanced Autoformer encoders
        if use_enhanced_layers:
            from models.EnhancedAutoformer import EnhancedEncoder, EnhancedEncoderLayer
            from layers.EnhancedAutoCorrelation import AdaptiveAutoCorrelationLayer
            
            self.resolution_encoders = nn.ModuleList()
            for i in range(n_levels):
                # Create Enhanced encoder for each resolution
                encoder_layers = [
                    EnhancedEncoderLayer(
                        AdaptiveAutoCorrelationLayer(
                            mask_flag=False,
                            factor=configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                            adaptive_k=True,
                            multi_scale=True
                        ),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for _ in range(configs.e_layers)
                ]
                
                encoder = EnhancedEncoder(encoder_layers, norm_layer=torch.nn.LayerNorm(configs.d_model))
                self.resolution_encoders.append(encoder)
        else:
            # Fallback to simpler encoders
            self.resolution_encoders = nn.ModuleList([
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=configs.d_model,
                        nhead=configs.n_heads,
                        dim_feedforward=configs.d_ff,
                        dropout=configs.dropout,
                        batch_first=True
                    ),
                    num_layers=configs.e_layers
                ) for _ in range(n_levels)
            ])
        
        # Multi-wavelet transforms for each resolution (optional)
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
            except Exception as e:
                logger.warning(f"Failed to initialize MultiWaveletTransform: {e}")
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
                except Exception as e:
                    logger.debug(f"MultiWavelet transform failed for level {i}: {e}")
            
            # Apply resolution-specific encoder
            if hasattr(encoder, 'forward'):
                encoded = encoder(features, attn_mask=attn_mask)
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
            # Concatenate and project
            concatenated = torch.cat(aligned_features, dim=-1)
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
    Complete Hierarchical Enhanced Autoformer integrating all components.
    """
    
    def __init__(self, configs, n_levels=3, wavelet_type='db4', 
                 fusion_strategy='weighted_concat', use_cross_attention=True):
        super(HierarchicalEnhancedAutoformer, self).__init__()
        logger.info("Initializing HierarchicalEnhancedAutoformer")
        
        self.configs = configs
        self.n_levels = n_levels
        
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
        self.encoder = HierarchicalEncoder(configs, n_levels=n_levels)
        
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
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        Hierarchical Enhanced Autoformer forward pass.
        """
        # Input embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        
        # Hierarchical decomposition
        multi_res_features = self.decomposer(enc_out)
        
        # Hierarchical encoding
        encoded_features = self.encoder(multi_res_features, attn_mask=enc_self_mask)
        
        # Cross-resolution attention
        if self.cross_attention is not None:
            cross_attended_features = self.cross_attention(encoded_features)
        else:
            cross_attended_features = encoded_features
        
        # Hierarchical fusion
        fused_features = self.fusion(
            cross_attended_features, 
            target_length=self.configs.pred_len
        )
        
        # Final projection
        output = self.projection(fused_features)
        
        return output[:, -self.configs.pred_len:, :]  # Return prediction length
    
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
