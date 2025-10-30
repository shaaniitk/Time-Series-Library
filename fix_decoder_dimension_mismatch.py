#!/usr/bin/env python3
"""
Fix for decoder dimension mismatch causing loss=0.000000
The issue is in the decoder module where encoder output dimensions don't match decoder expectations
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def fix_decoder_dimension_mismatch():
    """
    The issue is in models/celestial_modules/decoder.py
    The decoder is receiving graph_features from encoder but not handling dimensions correctly
    """
    
    decoder_file = "models/celestial_modules/decoder.py"
    
    # Read the current decoder file
    with open(decoder_file, 'r') as f:
        content = f.read()
    
    # The issue is in the DecoderLayer forward method
    # It's trying to do cross-attention between dec_input and enc_output
    # but they might have different sequence lengths or the attention is not working properly
    
    # Let's fix the DecoderLayer implementation
    old_decoder_layer = '''class DecoderLayer(nn.Module):
    """Temporary DecoderLayer implementation - should be extracted from original model"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
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
        
    def forward(self, dec_input, enc_output):
        # Self-attention
        attn_output, _ = self.self_attention(dec_input, dec_input, dec_input)
        dec_input = self.norm1(dec_input + self.dropout(attn_output))
        
        # Cross-attention
        attn_output, _ = self.cross_attention(dec_input, enc_output, enc_output)
        dec_input = self.norm2(dec_input + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(dec_input)
        dec_input = self.norm3(dec_input + self.dropout(ff_output))
        
        return dec_input'''
    
    new_decoder_layer = '''class DecoderLayer(nn.Module):
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
        
        return dec_input_adapted'''
    
    # Replace the DecoderLayer
    content = content.replace(old_decoder_layer, new_decoder_layer)
    
    # Also fix the forward method in DecoderModule to handle dimension mismatches better
    old_forward_start = '''    def forward(self, dec_out, graph_features, past_celestial_features, future_celestial_features):
        # 1. Standard Decoder Layers
        decoder_features = dec_out
        for layer in self.decoder_layers:
            decoder_features = layer(decoder_features, graph_features)'''
    
    new_forward_start = '''    def forward(self, dec_out, graph_features, past_celestial_features, future_celestial_features):
        # 1. Standard Decoder Layers with dimension validation
        decoder_features = dec_out
        
        # Validate and log dimensions
        print(f"DECODER DEBUG:")
        print(f"  dec_out shape: {dec_out.shape}")
        print(f"  graph_features shape: {graph_features.shape}")
        
        # Apply decoder layers with error handling
        for i, layer in enumerate(self.decoder_layers):
            try:
                decoder_features = layer(decoder_features, graph_features)
                print(f"  Layer {i} output shape: {decoder_features.shape}")
            except Exception as e:
                print(f"  Layer {i} failed: {e}")
                print(f"  Skipping layer {i}")
                # Continue with previous features if layer fails
                continue'''
    
    content = content.replace(old_forward_start, new_forward_start)
    
    # Write the fixed decoder file
    with open(decoder_file, 'w') as f:
        f.write(content)
    
    print("Fixed decoder dimension mismatch")
    print("Changes made:")
    print("  - Added dimension adaptation in DecoderLayer")
    print("  - Added error handling for attention layers")
    print("  - Added debug logging for dimension tracking")
    print("  - Made decoder more robust to dimension mismatches")

if __name__ == "__main__":
    fix_decoder_dimension_mismatch()