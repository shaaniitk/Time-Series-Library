
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseAttention
from layers.MultiWaveletCorrelation import MultiWaveletCross
from layers.modular.core.logger import logger

class CrossResolutionAttention(BaseAttention):
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
        
        if not self.use_multiwavelet:
            # Use standard multi-head attention
            self.cross_res_attention = nn.ModuleList([
                nn.MultiheadAttention(d_model, n_heads, batch_first=True)
                for _ in range(n_levels - 1)
            ])
        
        # Cross-resolution projection layers
        self.cross_projections = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_levels - 1)
        ])
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Standard attention interface for compatibility.
        
        Args:
            queries: Query tensor
            keys: Key tensor  
            values: Value tensor
            attn_mask: Attention mask (optional, ignored for now)
            tau: Temperature parameter (optional, ignored)
            delta: Delta parameter (optional, ignored)
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # For now, just apply standard attention since this is being called in regular flow
        # TODO: Implement proper cross-resolution logic when multi-resolution data is available
        
        batch_size, seq_len_q, d_model = queries.shape
        _, seq_len_k, _ = keys.shape
        
        # Simple attention mechanism for compatibility
        scale = (d_model // self.n_heads) ** -0.5
        
        # Reshape for multi-head attention
        q = queries.view(batch_size, seq_len_q, self.n_heads, d_model // self.n_heads).transpose(1, 2)
        k = keys.view(batch_size, seq_len_k, self.n_heads, d_model // self.n_heads).transpose(1, 2)
        v = values.view(batch_size, seq_len_k, self.n_heads, d_model // self.n_heads).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, -1e9)
            
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len_q, d_model)
        
        return out, attn_weights
    
    def forward_multi_resolution(self, multi_res_features):
        """
        Apply cross-resolution attention between different scales.
        
        Args:
            multi_res_features: List of tensors at different resolutions
            
        Returns:
            List of cross-attended features
        """
        if len(multi_res_features) < 2:
            return multi_res_features, None
        
        
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
        
        return attended_features, None
    
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
        batch_size, seq_len_q, d_model = query_features.shape
        _, seq_len_kv, _ = key_value_features.shape
        
        attention_layer = MultiWaveletCross(
            in_channels=self.head_dim,
            out_channels=self.head_dim,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            modes=min(32, self.head_dim // 2),
            c=64,
            k=8,
            ich=d_model,
            base='legendre',
            activation='tanh'
        ).to(query_features.device)

        # Reshape to 4D format: (B, N, H, E)
        query_4d = query_features.view(batch_size, seq_len_q, self.n_heads, self.head_dim)
        key_4d = key_value_features.view(batch_size, seq_len_kv, self.n_heads, self.head_dim)
        value_4d = key_4d  # Use same as key for cross-attention style
        
        attended, _ = attention_layer(
            query_4d, key_4d, value_4d, seq_len_q, seq_len_kv
        )
        
        # Reshape back to 3D: (B, N, D)
        attended = attended.view(batch_size, seq_len_q, d_model)
        
        return attended
