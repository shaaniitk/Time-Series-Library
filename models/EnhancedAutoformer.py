import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_wo_pos
from layers.Attention import get_attention_layer
from layers.Normalization import get_norm_layer
import math
import numpy as np # type: ignore
from utils.logger import logger
from typing import Optional, List


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
        
        # Learnable trend extraction weights: one set of weights per input feature/channel
        # Shape: [input_dim, 1 (as out_channel for conv1d group), max_kernel_size]
        self.feature_specific_trend_weights = nn.Parameter(torch.randn(input_dim, 1, max_kernel_size))
        
        # Adaptive kernel size predictor
        # This still predicts a single kernel size for the entire sample based on global features.
        # This kernel size will then be used to select the appropriate portion of feature_specific_trend_weights.
        # Note: This operates on input features (before embedding)
        self.kernel_predictor = nn.Sequential(
            nn.Linear(input_dim, max(input_dim // 2, 4)),  # Ensure minimum size
            nn.ReLU(),
            nn.Linear(max(input_dim // 2, 4), 1),
            nn.Sigmoid()
        )
        
        # Initialize with reasonable kernel size
        self.init_kernel_size = init_kernel_size
        # Initialize weights to be like a simple moving average initially for each feature
        if self.init_kernel_size > 0 and self.init_kernel_size <= self.max_kernel_size:
            nn.init.constant_(self.feature_specific_trend_weights[:, :, :self.init_kernel_size], 1.0 / self.init_kernel_size)
        
    def forward(self, x):
        """
        Vectorized implementation of the forward pass.
        Groups samples by predicted kernel size to avoid looping over the batch.
        
        Args:
            x: [B, L, D] input time series
            
        Returns:
            seasonal: [B, L, D] seasonal component
            trend: [B, L, D] trend component
        """
        B, L, D = x.shape
        assert D == self.input_dim, f"Input feature dimension mismatch. Expected {self.input_dim}, got {D}"

        # Predict optimal kernel size for each sample (vectorized)
        x_global = x.mean(dim=1)
        kernel_logits = self.kernel_predictor(x_global)
        predicted_k_float = (kernel_logits * (self.max_kernel_size - 5) + 5)

        # Calculate kernel sizes for the whole batch
        k_float = predicted_k_float.squeeze(-1)
        kernel_sizes = torch.round(k_float).int()
        kernel_sizes = torch.clamp(kernel_sizes, 3, min(self.max_kernel_size, L // 2))
        # Make kernel sizes odd
        kernel_sizes = kernel_sizes - (kernel_sizes % 2 == 0).int()
        kernel_sizes = torch.clamp(kernel_sizes, min=3)

        trend = torch.zeros_like(x)
        
        # Group samples by kernel size to process them in batches
        unique_kernels = torch.unique(kernel_sizes)

        for k_val in unique_kernels:
            k = k_val.item()
            
            # Find indices of samples that use this kernel size
            indices = torch.where(kernel_sizes == k)[0]
            
            # Gather the samples for the current kernel size
            batch_subset = x[indices]
            
            # Prepare for convolution
            padding = k // 2
            subset_permuted = batch_subset.transpose(1, 2)
            subset_padded = F.pad(subset_permuted, (padding, padding), mode='replicate')
            
            # Get and normalize weights for this kernel size
            current_weights_unnormalized = self.feature_specific_trend_weights[:, :, :k]
            normalized_weights = F.softmax(current_weights_unnormalized, dim=2)
            
            # Perform convolution on the subset
            trend_conv = F.conv1d(
                subset_padded,
                normalized_weights,
                groups=self.input_dim,
                padding=0
            )
            
            # Scatter the results back to their original positions
            trend[indices] = trend_conv.transpose(1, 2)

        seasonal = x - trend
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
        self.activation = getattr(F, activation, F.relu)
        
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

    def forward(self, x, attn_mask=None):
        attns = []
        layer_outputs = []
        
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                layer_output = attn_layer(x, attn_mask=attn_mask)
                # Handle different return signatures (2 or 3 values)
                if isinstance(layer_output, tuple) and len(layer_output) == 3:
                    x, attn, _ = layer_output  # Ignore aux_loss for now
                else:
                    x, attn = layer_output
                x = conv_layer(x)
                attns.append(attn)
                layer_outputs.append(x)
            layer_output = self.attn_layers[-1](x)
            # Handle different return signatures for last layer too
            if isinstance(layer_output, tuple) and len(layer_output) == 3:
                x, attn, _ = layer_output
            else:
                x, attn = layer_output
            attns.append(attn)
        else:
            for i, attn_layer in enumerate(self.attn_layers):
                layer_output = attn_layer(x, attn_mask=attn_mask)
                # Handle different return signatures (2 or 3 values)
                if isinstance(layer_output, tuple) and len(layer_output) == 3:
                    x, attn, _ = layer_output  # Ignore aux_loss for now
                else:
                    x, attn = layer_output
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
        
        self.activation = getattr(F, activation, F.relu)
        
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

        # Enhanced trend aggregation - keep in d_model space for hierarchical fusion
        residual_trend = trend1 + trend2 + trend3
        # Don't project trend here - let hierarchical fusion work in d_model space
        # residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        
        return x, residual_trend


class EnhancedDecoder(nn.Module):
    """
    Enhanced Autoformer decoder with improved trend-seasonal integration.
    """
    
    def __init__(self, layers, c_out, norm_layer=None, projection=None, use_moe_ffn=False):
        super(EnhancedDecoder, self).__init__()
        logger.info(f"Initializing EnhancedDecoder (MoE enabled: {use_moe_ffn})")
        
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection
        self.use_moe_ffn = use_moe_ffn
        
        # Trend integration module
        if len(layers) > 0:
            self.trend_integration = nn.Sequential(
                nn.Linear(c_out, c_out),
                nn.ReLU(),
                nn.Linear(c_out, c_out)
            )

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        accumulated_trends = []
        total_aux_loss = 0.0
        
        for layer in self.layers:
            layer_output = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            
            # Handle different return signatures from MoE vs. standard layers
            if len(layer_output) == 3:
                x, residual_trend, aux_loss = layer_output
                if isinstance(aux_loss, torch.Tensor):
                    total_aux_loss += aux_loss
            else:
                x, residual_trend = layer_output

            if trend is not None:
                trend = trend + residual_trend
            else:
                trend = residual_trend
            accumulated_trends.append(residual_trend)

        # Enhanced trend integration
        if hasattr(self, 'trend_integration') and len(accumulated_trends) > 1 and trend is not None:
            # Integrate trends from all layers
            stacked_trends = torch.stack(accumulated_trends, dim=-1)  # [B, L, D, num_layers]
            integrated_trend = torch.mean(stacked_trends, dim=-1)  # [B, L, D]
            trend_enhancement = self.trend_integration(integrated_trend)
            trend = trend + trend_enhancement

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
            
        return x, trend, total_aux_loss


class EnhancedAutoformer(nn.Module):
    """
    Enhanced Autoformer with adaptive autocorrelation and learnable decomposition.
    
    Key enhancements:
    1. Adaptive AutoCorrelation with multi-scale analysis
    2. Learnable series decomposition
    3. Enhanced encoder/decoder layers with better gradient flow
    4. Improved numerical stability and memory efficiency
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
        # num_target_variables is the number of *base* targets (c_out_evaluation from DimensionManager)
        # This is needed for reshaping when combining trend/seasonal in quantile mode.
        # It should be derived from the original config or passed explicitly if needed.
        # For now, assuming configs.c_out in the original args was the base c_out_evaluation.
        self.num_target_variables = getattr(configs, 'c_out_evaluation', configs.c_out) # Use c_out_evaluation if available

        if hasattr(configs, 'quantile_levels') and isinstance(configs.quantile_levels, list) and len(configs.quantile_levels) > 0:
            self.is_quantile_mode = True
            self.quantiles = sorted(configs.quantile_levels) # type: ignore
            self.num_quantiles = len(self.quantiles)
            logger.info(f"EnhancedAutoformer: Quantile mode ON. Quantiles: {self.quantiles}")

        # Enhanced decomposition with learnable parameters
        # This should operate on the features that are actually fed into the decoder's embedding.
        # In Exp_Long_Term_Forecast, dec_inp is constructed from scaled historical targets + scaled historical covariates
        # and zeroed future targets + scaled future covariates.
        # The dimension of dec_inp is `dec_in` from DimensionManager, which is `num_targets + num_covariates` for MS/M.
        # So, self.decomp should be initialized with configs.dec_in.
        self.decomp = LearnableSeriesDecomp(configs.dec_in)

        # Embedding (same as original)
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, 
                                                  configs.embed, configs.freq, configs.dropout)

        # Enhanced encoder with adaptive autocorrelation
        enhanced_autocorr_layers = []
        for l in range(configs.e_layers):
            autocorr_layer = get_attention_layer(configs)
            
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
            norm_layer=get_norm_layer(configs.norm_type, configs.d_model)
        )

        # Enhanced decoder for forecasting tasks
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, 
                                                      configs.embed, configs.freq, configs.dropout)
            
            enhanced_decoder_layers = []
            for l in range(configs.d_layers):
                self_autocorr_layer = get_attention_layer(configs)
                
                cross_autocorr_layer = get_attention_layer(configs)
                
                enhanced_decoder_layer = EnhancedDecoderLayer(
                    self_autocorr_layer,
                    cross_autocorr_layer,
                    self.d_model,
                    self.num_target_variables, # Trend part should have c_out_evaluation features
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                enhanced_decoder_layers.append(enhanced_decoder_layer)

            self.decoder = EnhancedDecoder(
                enhanced_decoder_layers,
                self.num_target_variables, # Pass c_out directly
                norm_layer=get_norm_layer(configs.norm_type, configs.d_model),
                # Output layer dimension is now configs.c_out, which is set by DimensionManager
                projection=nn.Linear(self.d_model, self.c_out, bias=True)
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
        # x_dec is the pre-constructed decoder input from Exp_Long_Term_Forecast
        # It contains historical targets (scaled) + historical covariates (scaled)
        # and zeroed-out future targets + future covariates (scaled)

        # Enhanced decomposition initialization on the full decoder input (x_dec)
        # seasonal_init_full and trend_init_full will have shape [B, L, configs.dec_in] (e.g., 118 features)
        seasonal_init_full, trend_init_full = self.decomp(x_dec)

        # --- Prepare trend_init for decoder accumulation ---
        # This trend is accumulated in the decoder and must match the output dimension of EnhancedDecoderLayer's projection.
        # EnhancedDecoderLayer's projection outputs `self.num_target_variables` features (e.g., 20).
        # So, we take only the target part of the initial trend.
        trend_init_for_decoder_accumulation = trend_init_full[:, :, :self.num_target_variables]

        # Mean for extending the `trend_init_for_decoder_accumulation`.
        # This mean should also be over the target features only.
        mean_for_trend_extension = torch.mean(x_dec[:, :, :self.num_target_variables], dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)

        # Concatenate for the `trend` argument to `self.decoder`.
        trend_arg_to_decoder = torch.cat([trend_init_for_decoder_accumulation[:, -self.label_len:, :], mean_for_trend_extension], dim=1)

        # --- Prepare seasonal_init for dec_embedding ---
        # `dec_embedding` is initialized with `configs.dec_in` (e.g., 118 features).
        # So, the seasonal input to `dec_embedding` must have `configs.dec_in` features.
        # `seasonal_init_full` already has `configs.dec_in` features.
        # The `zeros` for seasonal extension should match the full `x_dec` dimension.
        zeros_for_seasonal_extension = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)

        # Concatenate for the `seasonal_init` argument to `self.dec_embedding`.
        seasonal_arg_to_dec_embedding = torch.cat([seasonal_init_full[:, -self.label_len:, :], zeros_for_seasonal_extension], dim=1)

        # Enhanced encoding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Enhanced decoding
        # `dec_embedding` receives `seasonal_arg_to_dec_embedding` (118 features).
        dec_out = self.dec_embedding(seasonal_arg_to_dec_embedding, x_mark_dec)

        # `self.decoder` receives `trend_arg_to_decoder` (20 features).
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, 
                                                cross_mask=None, trend=trend_arg_to_decoder)

        # Final combination
        # The output of the decoder is the seasonal part. The trend part is added to it.
        # The shape of seasonal_part is [B, L, c_out], where c_out can be num_target_variables * num_quantiles.
        # The shape of trend_part is [B, L, num_target_variables].
        if self.is_quantile_mode and self.num_quantiles > 1:
            # In quantile mode, the trend is added to each quantile's seasonal component.
            # The seasonal_part is already shaped [B, L, num_target_variables * num_quantiles].
            # We need to expand the trend to match this shape.
            trend_part_expanded = trend_part.unsqueeze(-1).repeat(1, 1, 1, self.num_quantiles)
            trend_part_expanded = trend_part_expanded.view(seasonal_part.shape)
            dec_out = trend_part_expanded + seasonal_part
        else:
            dec_out = trend_part + seasonal_part
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """Enhanced imputation with encoder-decoder architecture."""
        # Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Decoder
        # For imputation, the decoder input is the same as the encoder input
        seasonal_init, trend_init = self.decomp(x_enc)

        # The rest of the logic is similar to forecast, but without the future prediction part
        dec_out = self.dec_embedding(seasonal_init, x_mark_enc)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)

        # Final combination
        dec_out = trend_part + seasonal_part
        return dec_out

    def anomaly_detection(self, x_enc):
        """Enhanced anomaly detection with encoder-decoder architecture."""
        # Encoder
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Decoder
        # For anomaly detection, the decoder input is the same as the encoder input
        seasonal_init, trend_init = self.decomp(x_enc)

        # The rest of the logic is similar to forecast, but without the future prediction part
        # We use x_enc for both seasonal_init and x_mark_enc for dec_embedding
        dec_out = self.dec_embedding(seasonal_init, None)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)

        # Final combination
        dec_out = trend_part + seasonal_part
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
            # dec_out shape is [B, L, num_target_variables * num_quantiles]
            output = dec_out[:, -self.pred_len:, :] # Output is [B, pred_len, num_target_variables * num_quantiles]
            # PinballLoss expects a 3D tensor and handles reshaping internally.
            # DO NOT reshape to 4D here if PinballLoss is used.
            # If a different loss expects 4D, that loss or the training script should handle it.
            return output
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
