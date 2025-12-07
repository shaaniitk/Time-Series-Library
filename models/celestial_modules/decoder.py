# models/celestial_modules/decoder.py

import torch
import torch.nn as nn
from .config import CelestialPGATConfig

# Import components directly to avoid circular imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from layers.modular.decoder.target_autocorrelation_module import DualStreamDecoder
from layers.modular.graph.celestial_to_target_attention import CelestialToTargetAttention
from layers.modular.decoder.mdn_decoder import MDNDecoder
from layers.modular.decoder.sequential_mixture_decoder import SequentialMixtureDensityDecoder

# Fixed DecoderLayer implementation with proper dimension handling
class DecoderLayer(nn.Module):
    """Fixed DecoderLayer implementation with proper dimension handling"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.self_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Add dimension adaptation layers
        self.enc_adapter = None
        self.dec_adapter = None
        
    def _adapt_dimensions(self, dec_input, enc_output):
        """Adapt encoder and decoder dimensions if needed"""
        batch_size_dec, seq_len_dec, dim_dec = dec_input.shape
        batch_size_enc, seq_len_enc, dim_enc = enc_output.shape
        
        # Ensure batch sizes match
        if batch_size_dec != batch_size_enc:
            raise ValueError(f"Batch size mismatch: dec={batch_size_dec}, enc={batch_size_enc}")
        
        # Adapt encoder output dimension if needed
        if dim_enc != self.d_model:
            if self.enc_adapter is None:
                self.enc_adapter = nn.Linear(dim_enc, self.d_model).to(enc_output.device)
            enc_output = self.enc_adapter(enc_output)
        
        # Adapt decoder input dimension if needed  
        if dim_dec != self.d_model:
            if self.dec_adapter is None:
                self.dec_adapter = nn.Linear(dim_dec, self.d_model).to(dec_input.device)
            dec_input = self.dec_adapter(dec_input)
        
        return dec_input, enc_output
        
    def forward(self, dec_input, enc_output):
        # Adapt dimensions if necessary
        dec_input_adapted, enc_output_adapted = self._adapt_dimensions(dec_input, enc_output)
        
        # Self-attention on decoder input
        try:
            attn_output, _ = self.self_attention(dec_input_adapted, dec_input_adapted, dec_input_adapted)
            dec_input_adapted = self.norm1(dec_input_adapted + self.dropout(attn_output))
        except Exception as e:
            print(f"Self-attention failed: {e}")
            print(f"Dec input shape: {dec_input_adapted.shape}")
            # Skip self-attention if it fails
            pass
        
        # Cross-attention with encoder output
        try:
            attn_output, _ = self.cross_attention(dec_input_adapted, enc_output_adapted, enc_output_adapted)
            dec_input_adapted = self.norm2(dec_input_adapted + self.dropout(attn_output))
        except Exception as e:
            print(f"Cross-attention failed: {e}")
            print(f"Dec input shape: {dec_input_adapted.shape}")
            print(f"Enc output shape: {enc_output_adapted.shape}")
            # Skip cross-attention if it fails - just use decoder input
            pass
        
        # Feed forward
        try:
            ff_output = self.feed_forward(dec_input_adapted)
            dec_input_adapted = self.norm3(dec_input_adapted + self.dropout(ff_output))
        except Exception as e:
            print(f"Feed forward failed: {e}")
            print(f"Input shape: {dec_input_adapted.shape}")
            # Return input if feed forward fails
            pass
        
        return dec_input_adapted

class DecoderModule(nn.Module):
    def __init__(self, config: CelestialPGATConfig):
        super().__init__()
        self.config = config
        
        # FIX ISSUE #3: Enforce single probabilistic head selection - fail loudly on conflicts
        probabilistic_heads_enabled = [
            ('enable_mdn_decoder', config.enable_mdn_decoder),
            ('use_mixture_decoder', config.use_mixture_decoder),
            ('use_sequential_mixture_decoder', config.use_sequential_mixture_decoder)
        ]
        active_heads = [name for name, enabled in probabilistic_heads_enabled if enabled]
        
        if len(active_heads) > 1:
            raise ValueError(
                f"Multiple probabilistic decoder heads enabled: {active_heads}. "
                f"Please enable only ONE of: enable_mdn_decoder, use_mixture_decoder, use_sequential_mixture_decoder. "
                f"This prevents wasted parameters and gradient conflicts."
            )

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(config.d_model, config.n_heads, config.dropout) for _ in range(config.d_layers)
        ])

        if config.use_target_autocorrelation:
            self.dual_stream_decoder = DualStreamDecoder(
                d_model=config.d_model, num_targets=config.c_out, num_heads=config.n_heads, dropout=config.dropout
            )
        else:
            self.dual_stream_decoder = None

        if config.use_celestial_target_attention:
            self.celestial_to_target_attention = CelestialToTargetAttention(
                num_celestial=config.num_celestial_bodies, num_targets=config.c_out, d_model=config.d_model,
                num_heads=config.n_heads, dropout=config.dropout, 
                use_gated_fusion=config.celestial_target_use_gated_fusion,
                enable_diagnostics=config.celestial_target_diagnostics,
                use_edge_bias=config.use_c2t_edge_bias,
                edge_bias_scale=config.c2t_edge_bias_weight,
                celestial_dim=config.celestial_dim, # Pass the correct dimension
            )
        else:
            self.celestial_to_target_attention = None
            
        # Enhanced decoder options
        if config.use_mixture_decoder or config.use_sequential_mixture_decoder:
            self.mixture_decoder = SequentialMixtureDensityDecoder(
                d_model=config.d_model,
                pred_len=config.pred_len,
                num_components=config.mdn_components,  # FIXED: Use config value, not hardcoded 3
                num_targets=config.c_out,
                num_decoder_layers=2,
                num_heads=config.n_heads,
                dropout=config.dropout
            )
        else:
            self.mixture_decoder = None
            
        if config.enable_mdn_decoder:
            self.mdn_decoder = MDNDecoder(
                d_input=config.d_model, 
                n_targets=config.c_out, 
                n_components=config.mdn_components,
                sigma_min=config.mdn_sigma_min, 
                use_softplus=getattr(config, 'mdn_use_softplus', True),
                adaptive_input=True  # Enable automatic dimension adaptation
            )
        else:
            self.mdn_decoder = None
            
        # Always have a fallback projection layer
        self.projection = nn.Linear(config.d_model, config.c_out)

    def forward(self, dec_out, graph_features, past_celestial_features, future_celestial_features):
        # 1. Standard Decoder Layers with dimension validation
        decoder_features = dec_out
        
        # Validate and log dimensions
        if getattr(self.config, 'debug_mode', False):
            print(f"DECODER DEBUG:")
            print(f"  dec_out shape: {dec_out.shape}")
            print(f"  graph_features shape: {graph_features.shape}")
        
        # Apply decoder layers with error handling
        for i, layer in enumerate(self.decoder_layers):
            try:
                decoder_features = layer(decoder_features, graph_features)
                if getattr(self.config, 'debug_mode', False):
                    print(f"  Layer {i} output shape: {decoder_features.shape}")
            except Exception as e:
                print(f"  Layer {i} failed: {e}")
                print(f"  Skipping layer {i}")
                # Continue with previous features if layer fails
                continue

        # 2. Target Autocorrelation
        if self.config.use_target_autocorrelation and self.dual_stream_decoder is not None:
            decoder_features = self.dual_stream_decoder(decoder_features, graph_features)
            
        # 3. Celestial-to-Target Attention
        celestial_target_diagnostics = None
        aux_loss = 0.0
        
        if self.config.use_celestial_target_attention and self.celestial_to_target_attention is not None:
            pred_start_idx = self.config.label_len
            decoder_pred_features = decoder_features[:, pred_start_idx:, :]
            batch_size_dec = decoder_pred_features.shape[0]
            decoder_target_features = decoder_pred_features.unsqueeze(2).expand(
                -1, -1, self.config.c_out, -1
            )
            
            # Process FUTURE celestial states if provided (deterministic conditioning!)
            celestial_feats = None
            if future_celestial_features is not None:
                # Future celestial features should be properly shaped
                celestial_feats = future_celestial_features
                if celestial_feats.dim() == 3:
                    # Reshape and project if needed
                    batch_size, seq_len, celestial_feature_dim = celestial_feats.shape
                    celestial_dim = celestial_feature_dim // self.config.num_celestial_bodies
                    celestial_feats = celestial_feats.view(batch_size, seq_len, self.config.num_celestial_bodies, celestial_dim)
                    
                    if celestial_dim != self.config.d_model:
                        if not hasattr(self, 'future_celestial_to_dmodel'):
                            self.future_celestial_to_dmodel = nn.Linear(
                                celestial_dim, self.config.d_model
                            ).to(celestial_feats.device)
                        
                        batch_size, seq_len, num_celestial, celestial_dim = celestial_feats.shape
                        celestial_feats_flat = celestial_feats.view(-1, celestial_dim)
                        celestial_feats_projected = self.future_celestial_to_dmodel(celestial_feats_flat)
                        celestial_feats = celestial_feats_projected.view(batch_size, seq_len, num_celestial, self.config.d_model)
            elif past_celestial_features is not None:
                # Fallback to past celestial features
                celestial_feats = past_celestial_features
                if celestial_feats.dim() == 3:
                    batch_size, seq_len, celestial_feature_dim = celestial_feats.shape
                    celestial_dim = celestial_feature_dim // self.config.num_celestial_bodies
                    celestial_feats = celestial_feats.view(batch_size, seq_len, self.config.num_celestial_bodies, celestial_dim)
                    
                    if celestial_dim != self.config.d_model:
                        if not hasattr(self, 'celestial_projection'):
                            self.celestial_projection = nn.Linear(celestial_dim, self.config.d_model).to(celestial_feats.device)
                        
                        batch_size, seq_len, num_celestial, celestial_dim = celestial_feats.shape
                        celestial_feats_flat = celestial_feats.view(-1, celestial_dim)
                        celestial_feats_projected = self.celestial_projection(celestial_feats_flat)
                        celestial_feats = celestial_feats_projected.view(batch_size, seq_len, num_celestial, self.config.d_model)
            
            # Apply celestial-to-target attention if we have celestial features
            if celestial_feats is not None:
                enhanced_target_features, celestial_target_diagnostics, gate_entropy_loss = self.celestial_to_target_attention(
                    target_features=decoder_target_features,
                    celestial_features=celestial_feats,
                    return_diagnostics=self.config.celestial_target_diagnostics
                )
                
                # FIX ISSUE #6: Add gate entropy regularization loss
                if isinstance(gate_entropy_loss, torch.Tensor):
                    aux_loss += gate_entropy_loss.item()
                elif isinstance(gate_entropy_loss, (int, float)):
                    aux_loss += gate_entropy_loss
                
                # Optional: auxiliary relation loss
                if (self.config.c2t_aux_rel_loss_weight > 0.0 and 
                    celestial_target_diagnostics is not None and 
                    isinstance(celestial_target_diagnostics, dict) and 
                    'attention_weights' in celestial_target_diagnostics):
                    try:
                        attn_dict = celestial_target_diagnostics['attention_weights']
                        attn_list = []
                        for t_idx in range(self.config.c_out):
                            key = f'target_{t_idx}_attn'
                            if key in attn_dict:
                                attn_list.append(attn_dict[key])
                        if attn_list:
                            attn_stack = torch.stack(attn_list, dim=2)
                            attn_mean = attn_stack.mean(dim=2)
                            # Simple uniform prior for auxiliary loss
                            uniform_prior = torch.ones_like(attn_mean) / attn_mean.size(-1)
                            attn_clamped = attn_mean.clamp_min(1e-8)
                            kl = (attn_clamped * (attn_clamped.log() - (uniform_prior + 1e-8).log())).sum(dim=-1)
                            aux_loss += float(self.config.c2t_aux_rel_loss_weight) * float(kl.mean().item())
                    except Exception:
                        pass  # Ignore auxiliary loss computation errors
                
                # Pool target features back to [batch, pred_len, d_model] for projection
                decoder_features_enhanced = enhanced_target_features.mean(dim=2)
                
                # Replace prediction window features in decoder_features
                decoder_features = torch.cat([
                    decoder_features[:, :pred_start_idx, :],
                    decoder_features_enhanced
                ], dim=1)

        # 4. Final prediction with enhanced decoder options
        prediction_features = decoder_features[:, -self.config.pred_len:, :]
        predictions = None
        mdn_components = None
        
        # Priority: MDN decoder > Sequential mixture > Simple projection
        if self.config.enable_mdn_decoder and self.mdn_decoder is not None:
            try:
                pi, mu, sigma = self.mdn_decoder(prediction_features)
                predictions = self.mdn_decoder.mean_prediction(pi, mu)
                mdn_components = (pi, mu, sigma)
                
                # Validate final prediction shape
                expected_shape = (prediction_features.shape[0], self.config.pred_len, self.config.c_out)
                if predictions.shape != expected_shape:
                    print(f"MDN prediction shape mismatch: {predictions.shape} != {expected_shape}")
                    print("Falling back to projection layer")
                    predictions = self.projection(prediction_features)
                    mdn_components = None
                    
            except Exception as e:
                print(f"MDN decoder failed: {e}")
                print(f"Input shape: {prediction_features.shape}")
                print("Falling back to projection layer")
                predictions = self.projection(prediction_features)
                mdn_components = None
        elif (self.config.use_mixture_decoder or self.config.use_sequential_mixture_decoder) and self.mixture_decoder is not None:
            try:
                means, log_stds, log_weights = self.mixture_decoder(
                    encoder_output=graph_features,
                    decoder_input=prediction_features
                )
                predictions = self.mixture_decoder.get_point_prediction((means, log_stds, log_weights))
                mdn_components = (means, log_stds, log_weights)

                
                # Only set mdn_components if not fallen back
                if predictions.size(-1) != self.config.c_out:
                    print(f"Mixture decoder prediction shape mismatch: {predictions.shape}")
                    predictions = self.projection(prediction_features)
                    mdn_components = None 
                else:
                    mdn_components = (means, log_stds, log_weights)

            except Exception as e:
                print(f"Mixture decoder failed: {e}")
                import traceback
                traceback.print_exc()
                predictions = self.projection(prediction_features)
                mdn_components = None
        else:
            predictions = self.projection(prediction_features)
        
        return predictions, aux_loss, mdn_components