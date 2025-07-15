import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_wo_pos
from layers.AdvancedComponents import (
    FourierAttention, WaveletDecomposition, MetaLearningAdapter,
    TemporalConvNet, AdaptiveMixture
)
from layers.Normalization import get_norm_layer


class HybridAutoformer(nn.Module):
    """Hybrid model combining multiple advanced techniques"""
    
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        
        # Multi-resolution decomposition
        self.wavelet_decomp = WaveletDecomposition(configs.enc_in, levels=3)
        
        # Enhanced embeddings
        self.enc_embedding = DataEmbedding_wo_pos(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        
        # Hybrid attention mechanisms
        self.fourier_attention = FourierAttention(configs.d_model, configs.n_heads, configs.seq_len)
        self.temporal_conv = TemporalConvNet(configs.d_model, [configs.d_model] * 3)
        
        # Adaptive mixture of experts
        self.mixture_experts = AdaptiveMixture(configs.d_model, num_experts=4)
        
        # Meta-learning adapter
        self.meta_adapter = MetaLearningAdapter(configs.d_model)
        
        # Multi-scale processing
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv1d(configs.d_model, configs.d_model, kernel_size=k, padding=k//2)
            for k in [3, 5, 7, 9]
        ])
        
        self.scale_fusion = nn.Linear(configs.d_model * 4, configs.d_model)
        
        # Task-specific heads
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.dec_embedding = DataEmbedding_wo_pos(
                configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout
            )
            
            # Progressive prediction heads
            self.prediction_layers = nn.ModuleList([
                nn.Linear(configs.d_model, configs.c_out)
                for _ in range(3)  # Short, medium, long-term predictions
            ])
            
            # Attention-based fusion
            self.temporal_fusion = nn.MultiheadAttention(
                configs.d_model, configs.n_heads, dropout=configs.dropout, batch_first=True
            )
        
        # Regularization
        self.dropout = nn.Dropout(configs.dropout)
        self.layer_norm = nn.LayerNorm(configs.d_model)
        
    def multi_scale_processing(self, x):
        """Multi-scale feature extraction"""
        x_conv = x.transpose(1, 2)  # [B, D, L]
        
        scale_features = []
        for conv in self.multi_scale_conv:
            scale_feat = F.relu(conv(x_conv))
            scale_features.append(scale_feat)
        
        # Concatenate and fuse
        combined = torch.cat(scale_features, dim=1)  # [B, D*4, L]
        fused = self.scale_fusion(combined.transpose(1, 2))  # [B, L, D]
        
        return fused
    
    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None):
        # Multi-resolution decomposition
        x_decomp, wavelet_components = self.wavelet_decomp(x_enc)
        
        # Enhanced embedding
        enc_out = self.enc_embedding(x_decomp, x_mark_enc)
        
        # Fourier-based attention
        fourier_out, _ = self.fourier_attention(enc_out)
        
        # Temporal convolution
        tcn_out = self.temporal_conv(enc_out)
        
        # Multi-scale processing
        multi_scale_out = self.multi_scale_processing(enc_out)
        
        # Combine different representations
        combined = fourier_out + tcn_out + multi_scale_out
        combined = self.layer_norm(combined)
        
        # Adaptive mixture of experts
        expert_out = self.mixture_experts(combined)
        
        # Meta-learning adaptation
        adapted_out = self.meta_adapter(expert_out)
        
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            return self._forecast(adapted_out, x_dec, x_mark_dec)
        elif self.task_name == 'classification':
            return self._classify(adapted_out, x_mark_enc)
        else:
            return adapted_out
    
    def _forecast(self, enc_out, x_dec, x_mark_dec):
        """Enhanced forecasting with progressive prediction"""
        if x_dec is not None:
            dec_out = self.dec_embedding(x_dec, x_mark_dec)
        else:
            # Auto-regressive generation
            dec_out = enc_out[:, -self.pred_len:, :]
        
        # Temporal fusion with cross-attention
        fused_out, _ = self.temporal_fusion(dec_out, enc_out, enc_out)
        
        # Progressive predictions
        predictions = []
        for pred_layer in self.prediction_layers:
            pred = pred_layer(fused_out)
            predictions.append(pred)
        
        # Weighted ensemble
        weights = F.softmax(torch.ones(len(predictions)), dim=0)
        final_pred = sum(pred * w for pred, w in zip(predictions, weights))
        
        return final_pred[:, -self.pred_len:, :]
    
    def _classify(self, enc_out, x_mark_enc):
        """Enhanced classification"""
        # Global attention pooling
        attn_weights = F.softmax(
            torch.sum(enc_out * x_mark_enc.unsqueeze(-1), dim=-1), dim=1
        )
        pooled = torch.sum(enc_out * attn_weights.unsqueeze(-1), dim=1)
        
        return self.prediction_layers[0](pooled)


Model = HybridAutoformer