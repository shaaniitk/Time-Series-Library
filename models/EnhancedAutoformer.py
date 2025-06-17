import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_wo_pos
from layers.EnhancedAutoCorrelation import AdaptiveAutoCorrelation, AdaptiveAutoCorrelationLayer
from layers.Autoformer_EncDec import my_Layernorm
import math
import numpy as np
from utils.logger import logger


class LearnableSeriesDecomp(nn.Module):
    """
    Enhanced series decomposition with learnable parameters.
    
    Improvements over standard moving average:
    1. Learnable trend extraction weights
    2. Adaptive kernel size selection
    3. Feature-specific decomposition parameters
    """
    
    def __init__(self, input_dim, init_kernel_size=25, max_kernel_size=50):
        super(LearnableSeriesDecomp, self).__init__()
        logger.info("Initializing LearnableSeriesDecomp")
        
        self.input_dim = input_dim
        self.max_kernel_size = max_kernel_size
        
        # Learnable trend extraction weights
        self.trend_weights = nn.Parameter(torch.ones(max_kernel_size) / max_kernel_size)
        
        # Adaptive kernel size predictor
        # Note: This operates on input features (before embedding)
        self.kernel_predictor = nn.Sequential(
            nn.Linear(input_dim, max(input_dim // 2, 4)),  # Ensure minimum size
            nn.ReLU(),
            nn.Linear(max(input_dim // 2, 4), 1),
            nn.Sigmoid()
        )
        
        # Initialize with reasonable kernel size
        self.init_kernel_size = init_kernel_size
        
    def forward(self, x):
        """
        Args:
            x: [B, L, D] input time series
            
        Returns:
            seasonal: [B, L, D] seasonal component
            trend: [B, L, D] trend component
        """
        B, L, D = x.shape
        
        # Predict optimal kernel size for each sample
        # Use global average to determine kernel size
        x_global = x.mean(dim=1)  # [B, D]
        kernel_logits = self.kernel_predictor(x_global)  # [B, 1]
        
        # Convert to kernel size (between 5 and max_kernel_size)
        kernel_sizes = (kernel_logits * (self.max_kernel_size - 5) + 5).round().int()
        
        # Batch-wise processing with different kernel sizes
        seasonal_outputs = []
        trend_outputs = []
        
        for b in range(B):
            k = kernel_sizes[b].item()
            k = max(5, min(k, self.max_kernel_size, L//3))  # Ensure reasonable bounds
            
            # Get learnable weights for this kernel size
            weights = F.softmax(self.trend_weights[:k], dim=0)
            
            # Apply learnable convolution for trend extraction
            x_b = x[b:b+1].transpose(1, 2)  # [1, D, L]
            
            # Padding for convolution to maintain length
            padding = k // 2
            x_padded = F.pad(x_b, (padding, padding), mode='replicate')
            
            # Apply convolution with learnable weights
            trend_b = F.conv1d(
                x_padded,
                weights.view(1, 1, k).repeat(self.input_dim, 1, 1),
                groups=self.input_dim,
                padding=0
            )
            
            # Ensure output length matches input length
            if trend_b.size(2) != L:
                # Adjust by cropping or padding
                if trend_b.size(2) > L:
                    trend_b = trend_b[:, :, :L]
                else:
                    pad_len = L - trend_b.size(2)
                    trend_b = F.pad(trend_b, (0, pad_len), mode='replicate')
            
            trend_b = trend_b.transpose(1, 2)  # [1, L, D]
            seasonal_b = x[b:b+1] - trend_b
            
            seasonal_outputs.append(seasonal_b)
            trend_outputs.append(trend_b)
        
        seasonal = torch.cat(seasonal_outputs, dim=0)
        trend = torch.cat(trend_outputs, dim=0)
        
        return seasonal, trend


class EnhancedEncoderLayer(nn.Module):
    """
    Enhanced Autoformer encoder layer with improved decomposition and attention.
    """
    
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EnhancedEncoderLayer, self).__init__()
        logger.info("Initializing EnhancedEncoderLayer")
        
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        
        # Enhanced feed-forward network with gated mechanism
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        
        # Gated mechanism for better information flow
        self.gate = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Enhanced decomposition
        self.decomp1 = LearnableSeriesDecomp(d_model)
        self.decomp2 = LearnableSeriesDecomp(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        
        # Layer scaling for better gradient flow
        self.attention_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.ffn_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x, attn_mask=None):
        # Enhanced attention with scaling
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(self.attention_scale * new_x)
        
        # First enhanced decomposition
        x, _ = self.decomp1(x)
        
        # Enhanced feed-forward with gating
        residual = x
        y = x.transpose(-1, 1)
        
        # Apply gated mechanism
        gate_values = self.gate(y)
        y = y * gate_values
        
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y)).transpose(-1, 1)
        
        x = residual + self.ffn_scale * y
        
        # Second enhanced decomposition
        res, _ = self.decomp2(x)
        
        return res, attn


class EnhancedEncoder(nn.Module):
    """
    Enhanced Autoformer encoder with hierarchical processing.
    """
    
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(EnhancedEncoder, self).__init__()
        logger.info("Initializing EnhancedEncoder")
        
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        
        # Cross-layer connections for better information flow
        if len(attn_layers) > 1:
            self.cross_layer_connections = nn.ModuleList([
                nn.Linear(attn_layers[0].attention.inner_correlation.factor * 64, 64)  # Adjust based on d_model
                for _ in range(len(attn_layers) - 1)
            ])

    def forward(self, x, attn_mask=None):
        attns = []
        layer_outputs = []
        
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
                layer_outputs.append(x)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for i, attn_layer in enumerate(self.attn_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)
                layer_outputs.append(x)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class EnhancedDecoderLayer(nn.Module):
    """
    Enhanced Autoformer decoder layer with advanced decomposition.
    """
    
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(EnhancedDecoderLayer, self).__init__()
        logger.info("Initializing EnhancedDecoderLayer")
        
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        
        # Enhanced feed-forward network
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        
        # Enhanced decomposition layers
        self.decomp1 = LearnableSeriesDecomp(d_model)
        self.decomp2 = LearnableSeriesDecomp(d_model)
        self.decomp3 = LearnableSeriesDecomp(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Enhanced projection with residual connections
        self.projection = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, 
                     stride=1, padding=1, padding_mode='circular', bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=False)
        )
        
        self.activation = F.relu if activation == "relu" else F.gelu
        
        # Attention scaling
        self.self_attn_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.cross_attn_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # Enhanced self-attention
        residual = x
        new_x = self.self_attention(x, x, x, attn_mask=x_mask)[0]
        x = residual + self.dropout(self.self_attn_scale * new_x)
        x, trend1 = self.decomp1(x)
        
        # Enhanced cross-attention
        residual = x
        new_x = self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0]
        x = residual + self.dropout(self.cross_attn_scale * new_x)
        x, trend2 = self.decomp2(x)
        
        # Enhanced feed-forward
        residual = x
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x = residual + y
        x, trend3 = self.decomp3(x)

        # Enhanced trend aggregation
        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        
        return x, residual_trend


class EnhancedDecoder(nn.Module):
    """
    Enhanced Autoformer decoder with improved trend-seasonal integration.
    """
    
    def __init__(self, layers, norm_layer=None, projection=None):
        super(EnhancedDecoder, self).__init__()
        logger.info("Initializing EnhancedDecoder")
        
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection
        
        # Trend integration module
        if len(layers) > 0:
            d_model = layers[0].conv1.in_channels
            self.trend_integration = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model)
            )

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        accumulated_trends = []
        
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend
            accumulated_trends.append(residual_trend)

        # Enhanced trend integration
        if hasattr(self, 'trend_integration') and len(accumulated_trends) > 1:
            # Integrate trends from all layers
            stacked_trends = torch.stack(accumulated_trends, dim=-1)  # [B, L, D, num_layers]
            integrated_trend = torch.mean(stacked_trends, dim=-1)  # [B, L, D]
            trend_enhancement = self.trend_integration(integrated_trend)
            trend = trend + trend_enhancement

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
            
        return x, trend


class EnhancedAutoformer(nn.Module):
    """
    Enhanced Autoformer with adaptive autocorrelation and learnable decomposition.
    
    Key enhancements:
    1. Adaptive AutoCorrelation with multi-scale analysis
    2. Learnable series decomposition
    3. Enhanced encoder/decoder layers with better gradient flow
    4. Improved numerical stability and memory efficiency
    """

    def __init__(self, configs):
        super().__init__()
        logger.info(f"Initializing EnhancedAutoformer with configs: {configs}")
        
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        # Enhanced decomposition with learnable parameters
        self.decomp = LearnableSeriesDecomp(configs.c_out)  # Work on target features only

        # Embedding (same as original)
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, 
                                                  configs.embed, configs.freq, configs.dropout)

        # Enhanced encoder with adaptive autocorrelation
        enhanced_autocorr_layers = []
        for l in range(configs.e_layers):
            adaptive_autocorr = AdaptiveAutoCorrelation(
                False, configs.factor, 
                attention_dropout=configs.dropout,
                output_attention=False,
                adaptive_k=True,
                multi_scale=True,
                scales=[1, 2, 4]  # Multi-scale analysis
            )
            
            autocorr_layer = AdaptiveAutoCorrelationLayer(
                adaptive_autocorr,
                configs.d_model, 
                configs.n_heads
            )
            
            enhanced_layer = EnhancedEncoderLayer(
                autocorr_layer,
                configs.d_model,
                configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation
            )
            enhanced_autocorr_layers.append(enhanced_layer)

        self.encoder = EnhancedEncoder(
            enhanced_autocorr_layers,
            norm_layer=my_Layernorm(configs.d_model)
        )

        # Enhanced decoder for forecasting tasks
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, 
                                                      configs.embed, configs.freq, configs.dropout)
            
            enhanced_decoder_layers = []
            for l in range(configs.d_layers):
                # Self-attention
                self_adaptive_autocorr = AdaptiveAutoCorrelation(
                    True, configs.factor,
                    attention_dropout=configs.dropout,
                    output_attention=False,
                    adaptive_k=True,
                    multi_scale=True,
                    scales=[1, 2]  # Smaller scales for decoder
                )
                self_autocorr_layer = AdaptiveAutoCorrelationLayer(
                    self_adaptive_autocorr,
                    configs.d_model,
                    configs.n_heads
                )
                
                # Cross-attention
                cross_adaptive_autocorr = AdaptiveAutoCorrelation(
                    False, configs.factor,
                    attention_dropout=configs.dropout,
                    output_attention=False,
                    adaptive_k=True,
                    multi_scale=False  # Single scale for cross-attention
                )
                cross_autocorr_layer = AdaptiveAutoCorrelationLayer(
                    cross_adaptive_autocorr,
                    configs.d_model,
                    configs.n_heads
                )
                
                enhanced_decoder_layer = EnhancedDecoderLayer(
                    self_autocorr_layer,
                    cross_autocorr_layer,
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                enhanced_decoder_layers.append(enhanced_decoder_layer)

            self.decoder = EnhancedDecoder(
                enhanced_decoder_layers,
                norm_layer=my_Layernorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            )

        # Other task projections (same as original)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """Enhanced forecasting with improved decomposition and correlation."""
        
        # Enhanced decomposition initialization
        # Extract target features (first c_out columns) for decomposition
        target_features = x_enc[:, :, :x_dec.shape[2]]
        
        # Use enhanced decomposition on target features only
        seasonal_init, trend_init = self.decomp(target_features)
        
        # Mean and zeros for target features
        mean = torch.mean(target_features, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        
        # Decoder input preparation
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)

        # Enhanced encoding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Enhanced decoding
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, 
                                                cross_mask=None, trend=trend_init)

        # Final combination
        dec_out = trend_part + seasonal_part
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """Enhanced imputation (same as original for now)."""
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        """Enhanced anomaly detection (same as original for now)."""
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        """Enhanced classification (same as original for now)."""
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """Enhanced forward pass."""
        logger.debug("EnhancedAutoformer forward")
        
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out
        return None


# Alias for compatibility
Model = EnhancedAutoformer
